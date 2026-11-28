#!/usr/bin/env node
/**
 * NeatapticTS Gate MCP Server
 *
 * Exposes the Tier-1 gate scripts as a direct MCP server so agents can run
 * named gate checks through MCP tool calls rather than raw shell commands.
 *
 * Tools:
 *   - list_gates      — Return the available Tier-1 gate IDs and their owners.
 *   - run_gate_check  — Run a named gate and return its structured contract result.
 *   - query_tier_graph — Return the current tier inventory and validation summary.
 *   - query_customization_routing_table — Return the canonical routing table and freshness status.
 *   - get_slice_context — Return an assembled context window for a plan slice from the active step.
 *
 * Usage:
 *   node scripts/agent-customization/mcp/neataptic-gate-mcp.mjs [--self-check] [--json]
 *   node scripts/agent-customization/mcp/neataptic-gate-mcp.mjs --help
 */

import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  createMcpServer,
  createSelfCheckReport,
  createTool,
  emitSelfCheckReport,
  invokeServerRequest,
  MCP_PROTOCOL_VERSION,
  MCP_REPO_ROOT,
  parseMcpCliArgs,
  printMcpUsage,
  requireString,
  runStdioMcpServer,
  selfCheckError,
} from './mcp-utils.mjs';
import { createCustomizationRoutingTableTool } from './customization-routing-table-tool.mjs';
import {
  createSliceContextTool,
  createTierGraphTool,
} from './cortex-tier-tool.mjs';
import {
  describeTimeoutBudgets,
  findGatesWithoutBudget,
  GATE_MAX_BUFFER_BYTES,
  isTimeoutError,
  resolveGateTimeoutMs,
} from '../gate-timeout-budgets.mjs';

const SERVER_NAME = 'neataptic_gate_mcp';
const SERVER_VERSION = '0.1.0';

/**
 * Outer spawn budget (ms) for running a Tier-1 gate script.
 *
 * The budget is resolved PER GATE from `gate-timeout-budgets.mjs`, which is
 * the single source of truth shared with `slice-advancement.gate.mjs`. A
 * single flat ceiling is wrong in both directions:
 *
 * - Too small: the facade kills a legitimate run, the child's stdout is
 *   truncated or empty, and the caller sees the opaque
 *   "Gate '<id>' did not return valid JSON" instead of the real cause. This
 *   is the defect that made the consolidated `slice-advancement` gate
 *   (7 sequential sub-gates, one of which runs build + lint + targeted tests)
 *   unreportable through the relay.
 * - Too large: a genuinely hung gate blocks the relay for the full ceiling.
 *
 * The consolidated gate therefore gets a budget that covers the SUM of its
 * sub-gate budgets (its sub-gates run sequentially), while the cheap
 * file-reading gates keep a tight ceiling so a hang is caught quickly.
 */

const TIER_1_GATES = [
  {
    id: 'plan-sync',
    owner: 'validate-plan-sync.mjs',
    description:
      'Checks that all [WIP] plans are registered in README and Roadmap.',
  },
  {
    id: 'step-packet',
    owner: 'validate-plan-phase-packets.mjs',
    description:
      'Checks that active [WIP] step packets have required fields and sections.',
  },
  {
    id: 'plan-slice-quality',
    owner: 'plan-slice-quality.gate.mjs',
    description:
      'Checks that every WIP plan slice has estimate_hours <= 4. Oversized slices must be broken down before verification can pass.',
  },
  {
    id: 'plan-command-lint',
    owner: 'plan-command-lint.gate.mjs',
    description:
      'Validates shell commands referenced in plan Markdown files against actual CLI help output to catch flag drift.',
  },
  {
    id: 'agent-graph',
    owner: 'validate-agent-graph.mjs',
    description:
      'Checks that all agent delegation references resolve and no cycles exist.',
  },
  {
    id: 'agent-quality',
    owner: 'validate-agent-quality.mjs',
    description:
      'Checks that every agent body has the required sections and tier-specific structured-v1 contract.',
  },
  {
    id: 'tier-enforcement',
    owner: 'validate-agent-graph.mjs',
    description:
      'Checks that every agent has a valid tier assignment and only legal tier edges exist.',
  },
  {
    id: 'routing-table-freshness',
    owner: 'generate-agent-skill-routing-table.mjs',
    description:
      'Checks that the generated canonical routing table matches current agent and skill sources.',
  },
  {
    id: 'learning-event',
    owner: '.github/ai-learning/learning-log.jsonl',
    description:
      'Checks that the learning event log exists and contains at least one event.',
  },
  {
    id: 'stale-wip-plans',
    owner: 'stale-wip-plans.gate.mjs',
    description:
      'Detects plans whose top-level Status is [WIP] but all implementation phase/step markers are [DONE] — the missed-closure condition.',
  },
  {
    id: 'cortex-index',
    owner: 'cortex-index.gate.mjs',
    description:
      'Checks the semantic index, Repo Cortex MCP, workflow MCP binding, and semantic snapshot freshness.',
  },
  {
    id: 'cortex-first-search',
    owner: 'cortex-first-search.gate.mjs',
    description:
      'Checks the prerequisites for the Cortex-first search policy before corpus-bound work relies on Repo Cortex.',
  },
  {
    id: 'devtools-coverage',
    owner: 'devtools-coverage.gate.mjs',
    description:
      'Checks that the 03-red-testing and 05-green-testing agents include the devtools skill and reference the three Chrome DevTools MCP specialists.',
  },
  {
    id: 'delegate-skill-coverage',
    owner: 'delegate-skill-coverage.gate.mjs',
    description:
      'Checks that every Tier 1 and Tier 2 agent includes the execute skill in its skills array.',
  },
  {
    id: 'code-coverage',
    owner: 'code-coverage.gate.mjs',
    description:
      'Checks that changed src/, scripts/agent-customization/, and scripts/mcp-semantic/ files meet their coverage baseline (100% for new files, no regression for legacy files).',
  },
  {
    id: 'specialist-review',
    owner: 'specialist-review.gate.mjs',
    description:
      'Checks that [WIP] plans contain specialist review evidence in VALIDATION_EVIDENCE before 05-green-testing is dispatched. TRIVIAL slices are exempt; FULL slices require 1 Tier-3 specialist review.',
  },
  {
    id: 'slice-advancement',
    owner: 'slice-advancement.gate.mjs',
    description:
      'Master consolidated gate that runs all applicable per-slice gates in one call. Classifies slice as TRIVIAL or FULL and routes accordingly. TRIVIAL: plan-sync + step-packet + plan-slice-quality + plan-command-lint. FULL: also adds shared-validation + code-coverage + specialist-review. Pass --slice-id, --changed-files, and optionally --plan via run_gate_check args; the plan is forwarded to the plan-command-lint sub-gate instead of its stale default.',
  },
];

const GATES_DIR = path.join(
  MCP_REPO_ROOT,
  'scripts',
  'agent-customization',
  'gates',
);

/**
 * Detects whether a `spawnSync` call was killed by its timeout budget.
 *
 * Node reports a timeout through one or more of: an `ETIMEDOUT` error code
 * (sometimes nested under `cause`), a `SIGTERM` signal, and the `killed`
 * flag. When any of those is present the child never finished, so its
 * stdout is truncated or empty and the failure must be reported as a
 * timeout rather than as malformed output.
 *
 * @param {object} spawned - The `spawnSync` result object.
 * @returns {boolean} True when the child was killed by the timeout budget.
 */
function isSpawnTimeout(spawned) {
  return isTimeoutError(spawned);
}

// ---------------------------------------------------------------------------

export function createGateTools() {
  return [
    createTool({
      name: 'list_gates',
      description:
        'Return the available Tier-1 gate IDs, their owners, and descriptions.',
      annotations: { readOnlyHint: true },
      handler: async () => ({
        gates: TIER_1_GATES,
        gatesDir: GATES_DIR,
        schema: '.github/flows/flow.schema.yml',
      }),
    }),
    createTool({
      name: 'run_gate_check',
      description:
        'Run a named Tier-1 gate script and return its structured gate contract result ' +
        '(pass, evidence, fixHint, owner).',
      annotations: { readOnlyHint: true },
      inputSchema: {
        type: 'object',
        properties: {
          gate: {
            type: 'string',
            enum: TIER_1_GATES.map((gateDescriptor) => gateDescriptor.id),
            description:
              'Gate ID to run (plan-sync, step-packet, plan-slice-quality, plan-command-lint, agent-graph, agent-quality, tier-enforcement, routing-table-freshness, learning-event, stale-wip-plans, cortex-index, cortex-first-search, devtools-coverage, delegate-skill-coverage, code-coverage, specialist-review, or slice-advancement).',
          },
          args: {
            type: 'object',
            description:
              'Optional extra CLI arguments for the gate script (e.g., { "slice-id": "C3-impl", "changed-files": "src/foo.ts", "plan": "plans/<Plan>.plans.md" } for slice-advancement; plan is forwarded to plan-command-lint).',
            additionalProperties: true,
          },
        },
        required: ['gate'],
        additionalProperties: false,
      },
      handler: async (argumentsObject) => {
        const gateId = requireString(argumentsObject.gate, 'gate');
        const scriptPath = path.join(GATES_DIR, `${gateId}.gate.mjs`);

        const extraArgs = [];
        if (argumentsObject.args && typeof argumentsObject.args === 'object') {
          for (const [key, value] of Object.entries(argumentsObject.args)) {
            extraArgs.push(`--${key}=${value}`);
          }
        }

        const timeoutMs = resolveGateTimeoutMs(gateId);
        const spawned = spawnSync(
          'node',
          [scriptPath, '--json', ...extraArgs],
          {
            encoding: 'utf8',
            timeout: timeoutMs,
            maxBuffer: GATE_MAX_BUFFER_BYTES,
            cwd: MCP_REPO_ROOT,
          },
        );

        const stdout = spawned.stdout ?? '';
        try {
          return JSON.parse(stdout);
        } catch {
          // A timeout is NOT an "invalid JSON" condition: the child was
          // killed mid-run, so it never produced a verdict. Reporting it as
          // invalid JSON hides the only actionable fact (the run was
          // truncated). Detect it and say so.
          if (isSpawnTimeout(spawned)) {
            throw new Error(
              `Gate '${gateId}' timed out after ${timeoutMs} ms and was killed before it produced a gate contract. ` +
                `This is a budget-sizing failure, not a content failure. ` +
                `Run it directly to see where it stalls: node scripts/agent-customization/gates/${gateId}.gate.mjs --json ` +
                `(or narrow the slice / raise the budget in scripts/agent-customization/gate-timeout-budgets.mjs). ` +
                `stderr: ${spawned.stderr?.slice(0, 300) ?? '(empty)'}`,
            );
          }
          throw new Error(
            `Gate '${gateId}' did not return valid JSON. ` +
              `exit status: ${String(spawned.status)}, signal: ${String(spawned.signal)}. ` +
              `stderr: ${spawned.stderr?.slice(0, 300) ?? '(empty)'}`,
          );
        }
      },
    }),
    createTierGraphTool(),
    createCustomizationRoutingTableTool(),
    createSliceContextTool(),
  ];
}

export async function runGateSelfCheck({ server }) {
  const issues = [];

  // Step 0: Confirm every exposed gate has an explicit timeout budget.
  // A gate without one would silently inherit the default budget, which may
  // be too small — the exact inversion that produced the "did not return
  // valid JSON" failure mode.
  const gatesWithoutBudget = findGatesWithoutBudget(
    TIER_1_GATES.map((gateDescriptor) => gateDescriptor.id),
  );
  if (gatesWithoutBudget.length > 0) {
    issues.push(
      selfCheckError(
        'gate-mcp',
        `Gates without an explicit timeout budget: ${gatesWithoutBudget.join(', ')}. ` +
          `Add them to GATE_TIMEOUT_BUDGETS in scripts/agent-customization/gate-timeout-budgets.mjs.`,
      ),
    );
  }

  // Step 1: Confirm protocol handshake.
  const initializeResult = await invokeServerRequest(server, {
    method: 'initialize',
    params: {
      protocolVersion: MCP_PROTOCOL_VERSION,
      capabilities: {},
      clientInfo: { name: 'self-check', version: SERVER_VERSION },
    },
  });

  if (initializeResult.protocolVersion !== MCP_PROTOCOL_VERSION) {
    issues.push(
      selfCheckError(
        'gate-mcp',
        `Expected protocol version ${MCP_PROTOCOL_VERSION}, received ${String(initializeResult.protocolVersion)}.`,
      ),
    );
  }

  // Step 2: Confirm tool count.
  const toolListResult = await invokeServerRequest(server, {
    method: 'tools/list',
  });
  const expectedToolCount = server.tools.length;

  if (
    !Array.isArray(toolListResult.tools) ||
    toolListResult.tools.length !== expectedToolCount
  ) {
    issues.push(
      selfCheckError(
        'gate-mcp',
        `Expected ${expectedToolCount} gate tools, found ${Array.isArray(toolListResult.tools) ? toolListResult.tools.length : 'none'}.`,
      ),
    );
  }

  // Step 3: Run one gate check to confirm gate scripts are reachable.
  const gateCheckResult = await invokeServerRequest(server, {
    method: 'tools/call',
    params: {
      name: 'run_gate_check',
      arguments: { gate: 'learning-event' },
    },
  });

  if (gateCheckResult.isError) {
    issues.push(
      selfCheckError(
        'gate-mcp',
        'run_gate_check tool returned an error during self-check.',
      ),
    );
  }

  const gatePayload = gateCheckResult.structuredContent ?? {};

  if (typeof gatePayload.pass !== 'boolean') {
    issues.push(
      selfCheckError(
        'gate-mcp',
        'Gate result missing required "pass" boolean field.',
      ),
    );
  }

  // Step 4: Run the tier graph query to confirm the new tool surface is live.
  const tierGraphResult = await invokeServerRequest(server, {
    method: 'tools/call',
    params: {
      name: 'query_tier_graph',
      arguments: { includeAgents: false },
    },
  });

  if (tierGraphResult.isError) {
    issues.push(
      selfCheckError(
        'gate-mcp',
        'query_tier_graph returned an error during self-check.',
      ),
    );
  }

  const tierGraphPayload = tierGraphResult.structuredContent ?? {};
  if (
    typeof tierGraphPayload.summary?.total !== 'number' ||
    tierGraphPayload.summary.total < 1
  ) {
    issues.push(
      selfCheckError(
        'gate-mcp',
        'query_tier_graph did not report a valid agent total.',
      ),
    );
  }

  // Step 5: Run the routing-table query to confirm freshness reporting is live.
  const routingTableResult = await invokeServerRequest(server, {
    method: 'tools/call',
    params: {
      name: 'query_customization_routing_table',
      arguments: { includeRows: false },
    },
  });

  if (routingTableResult.isError) {
    issues.push(
      selfCheckError(
        'gate-mcp',
        'query_customization_routing_table returned an error during self-check.',
      ),
    );
  }

  const routingTablePayload = routingTableResult.structuredContent ?? {};
  if (
    typeof routingTablePayload.summary?.agents !== 'number' ||
    routingTablePayload.summary.agents < 1
  ) {
    issues.push(
      selfCheckError(
        'gate-mcp',
        'query_customization_routing_table did not report a valid agent total.',
      ),
    );
  }

  if (typeof routingTablePayload.freshness?.pass !== 'boolean') {
    issues.push(
      selfCheckError(
        'gate-mcp',
        'query_customization_routing_table did not report a freshness result.',
      ),
    );
  }

  return createSelfCheckReport('gate-mcp self-check', issues, {
    gatesTested: ['learning-event'],
    toolsTested: [
      'run_gate_check',
      'query_tier_graph',
      'query_customization_routing_table',
    ],
    toolCount: Array.isArray(toolListResult.tools)
      ? toolListResult.tools.length
      : 0,
  });
}

export async function main(
  argv = process.argv.slice(2),
  { runSelfCheck = runGateSelfCheck } = {},
) {
  const options = parseMcpCliArgs(argv);

  if (argv.includes('--print-timeout-budgets')) {
    process.stdout.write(`${JSON.stringify(describeTimeoutBudgets(), null, 2)}\n`);
    process.exit(0);
    return;
  }

  if (options.help) {
    printMcpUsage({
      title: 'Expose Tier-1 gate scripts as a direct MCP server.',
      entrypoint: 'scripts/agent-customization/mcp/neataptic-gate-mcp.mjs',
      summary:
        'Without flags this script starts a dependency-light stdio MCP server. ' +
        'Use --self-check to confirm that the gate scripts are accessible and return valid contracts.',
      tools: createGateTools(),
    });
    process.exit(0);
    return;
  }

  const server = createMcpServer({
    serverName: SERVER_NAME,
    serverVersion: SERVER_VERSION,
    tools: createGateTools(),
  });

  if (options.selfCheck) {
    const report = await runSelfCheck({ server });
    emitSelfCheckReport(report, options);
    process.exitCode = report.ok ? 0 : 1;
  } else {
    await runStdioMcpServer(server);
  }
}

export function bootstrapMain() {
  main().then(
    () => {},
    (error) => {
      console.error(error);
      process.exitCode = 1;
    },
  );
}

/* istanbul ignore next */
if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
) {
  bootstrapMain();
}
