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
 *
 * Usage:
 *   node scripts/agent-customization/mcp/neataptic-gate-mcp.mjs [--self-check] [--json]
 *   node scripts/agent-customization/mcp/neataptic-gate-mcp.mjs --help
 */

import { spawnSync } from 'node:child_process';
import path from 'node:path';
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
import { createTierGraphTool } from './cortex-tier-tool.mjs';

const SERVER_NAME = 'neataptic-gate-mcp';
const SERVER_VERSION = '0.1.0';

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
];

const GATES_DIR = path.join(
  MCP_REPO_ROOT,
  'scripts',
  'agent-customization',
  'gates',
);
const GATE_TOOLS = createGateTools();

const options = parseMcpCliArgs(process.argv.slice(2));

if (options.help) {
  printMcpUsage({
    title: 'Expose Tier-1 gate scripts as a direct MCP server.',
    entrypoint: 'scripts/agent-customization/mcp/neataptic-gate-mcp.mjs',
    summary:
      'Without flags this script starts a dependency-light stdio MCP server. ' +
      'Use --self-check to confirm that the gate scripts are accessible and return valid contracts.',
    tools: GATE_TOOLS,
  });
  process.exit(0);
}

const server = createMcpServer({
  serverName: SERVER_NAME,
  serverVersion: SERVER_VERSION,
  tools: GATE_TOOLS,
});

if (options.selfCheck) {
  const report = await runGateSelfCheck({ server });
  emitSelfCheckReport(report, options);
  process.exitCode = report.ok ? 0 : 1;
} else {
  await runStdioMcpServer(server);
}

// ---------------------------------------------------------------------------

function createGateTools() {
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
              'Gate ID to run (plan-sync, step-packet, agent-graph, agent-quality, tier-enforcement, routing-table-freshness, learning-event, stale-wip-plans, cortex-index, cortex-first-search, devtools-coverage, or delegate-skill-coverage).',
          },
        },
        required: ['gate'],
        additionalProperties: false,
      },
      handler: async (argumentsObject) => {
        const gateId = requireString(argumentsObject.gate, 'gate');
        const scriptPath = path.join(GATES_DIR, `${gateId}.gate.mjs`);

        const spawned = spawnSync('node', [scriptPath, '--json'], {
          encoding: 'utf8',
          timeout: 15_000,
          cwd: MCP_REPO_ROOT,
        });

        try {
          return JSON.parse(spawned.stdout);
        } catch {
          throw new Error(
            `Gate '${gateId}' did not return valid JSON. ` +
              `stderr: ${spawned.stderr?.slice(0, 300) ?? '(empty)'}`,
          );
        }
      },
    }),
    createTierGraphTool(),
    createCustomizationRoutingTableTool(),
  ];
}

async function runGateSelfCheck({ server }) {
  const issues = [];

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
