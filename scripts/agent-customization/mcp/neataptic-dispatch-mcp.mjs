#!/usr/bin/env node
/**
 * NeatapticTS Dispatch MCP Server
 *
 * Exposes the agent dispatch policy and inventory as a direct MCP server so
 * orchestrators can ask "is this delegation legal?" and receive a pre-validated
 * dispatch packet without spawning the target agent themselves.
 *
 * Tools:
 *   - list_dispatchable_agents — Sorted inventory from `.github/agents/*.agent.md`.
 *   - build_dispatch_packet    — Validate a delegation request and return a packet.
 *   - get_dispatch_policy      — Return allowed delegation edges and rule summary.
 *
 * Usage:
 *   node scripts/agent-customization/mcp/neataptic-dispatch-mcp.mjs [--self-check] [--json]
 *   node scripts/agent-customization/mcp/neataptic-dispatch-mcp.mjs --help
 */

import { runCustomizationInventory } from '../inventory-customizations.mjs';
import {
  createMcpServer,
  createSelfCheckReport,
  createTool,
  emitSelfCheckReport,
  invokeServerRequest,
  MCP_PROTOCOL_VERSION,
  parseMcpCliArgs,
  printMcpUsage,
  requireString,
  runStdioMcpServer,
  selfCheckError,
} from './mcp-utils.mjs';

const SERVER_NAME = 'neataptic-dispatch-mcp';
const SERVER_VERSION = '0.1.0';

const ALLOWED_CALLER_TIERS = [0, 1, 2, 3, 4];

const TIER_LABELS = {
  1: 'User-invocable phase orchestrators',
  2: 'Tier 2 coordinators',
  3: 'Hidden scouts and specialists',
  4: 'Cross-tier helpers and execution specialists',
};

const ALLOWED_EDGES = [
  { from: 0, to: 1 },
  { from: 1, to: 2 },
  { from: 1, to: 3 },
  { from: 1, to: 4 },
  { from: 2, to: 3 },
  { from: 2, to: 4 },
  { from: 3, to: 4 },
];

const DISPATCH_TOOLS = createDispatchTools();

const options = parseMcpCliArgs(process.argv.slice(2));

if (options.help) {
  printMcpUsage({
    title:
      'Expose the NeatapticTS agent dispatch policy as a direct MCP server.',
    entrypoint: 'scripts/agent-customization/mcp/neataptic-dispatch-mcp.mjs',
    summary:
      'Without flags this script starts a dependency-light stdio MCP server. ' +
      'Use --self-check to confirm that the dispatch tools return valid contracts.',
    tools: DISPATCH_TOOLS,
  });
  process.exit(0);
}

const server = createMcpServer({
  serverName: SERVER_NAME,
  serverVersion: SERVER_VERSION,
  tools: DISPATCH_TOOLS,
});

if (options.selfCheck) {
  const report = await runDispatchSelfCheck({ server });
  emitSelfCheckReport(report, options);
  process.exitCode = report.ok ? 0 : 1;
} else {
  await runStdioMcpServer(server);
}

// ---------------------------------------------------------------------------

function createDispatchTools() {
  return [
    createTool({
      name: 'list_dispatchable_agents',
      description:
        'Return a sorted inventory of every dispatchable agent from .github/agents/*.agent.md ' +
        'with name, tier, model, skills, agents, tools, userInvocable, and file path.',
      annotations: { readOnlyHint: true },
      handler: async () => {
        const report = await runCustomizationInventory();
        const agents = (report.agents ?? [])
          .map((agent) => {
            const tierNum = Number(agent.tier);
            return {
              name: agent.name,
              tier: tierNum,
              model: agent.model ?? null,
              skills: agent.skills,
              agents: agent.agents,
              tools: agent.tools,
              userInvocable: agent.userInvocable,
              file: agent.path,
            };
          })
          .toSorted((a, b) => a.name.localeCompare(b.name));

        const tiers = agents.reduce((counts, agent) => {
          counts[agent.tier] = (counts[agent.tier] ?? 0) + 1;
          return counts;
        }, {});

        return {
          agents,
          total: agents.length,
          tiers,
        };
      },
    }),
    createTool({
      name: 'build_dispatch_packet',
      description:
        'Validate a requested delegation and, if allowed, return a structured dispatch packet ' +
        'for the target agent. Returns ok=false with a reason when the target is unknown, ' +
        'the caller tier is invalid, the delegation direction is illegal, or the target ' +
        'violates the user-invocable rule.',
      annotations: { readOnlyHint: true },
      inputSchema: {
        type: 'object',
        properties: {
          target_agent: {
            type: 'string',
            description:
              'Agent name as declared in the frontmatter of a .github/agents/*.agent.md file.',
          },
          caller_tier: {
            type: 'integer',
            minimum: 0,
            maximum: 4,
            description:
              'Tier of the caller: 0 (Agent Zero), 1 (phase orchestrator), 2 (coordinator), ' +
              '3 (specialist), or 4 (auxiliary).',
          },
          prompt: {
            type: 'string',
            description:
              'Prompt text embedded verbatim into the dispatch packet.',
          },
          context_tier: {
            type: 'string',
            enum: ['default', 'long_context'],
            description:
              'Context tier for the dispatched agent; default is "default".',
          },
        },
        required: ['target_agent', 'caller_tier'],
        additionalProperties: false,
      },
      handler: async (argumentsObject) => {
        const targetName = requireString(
          argumentsObject.target_agent,
          'target_agent',
        );
        const callerTier = Number(argumentsObject.caller_tier);
        const prompt =
          typeof argumentsObject.prompt === 'string'
            ? argumentsObject.prompt
            : '';
        const contextTier =
          argumentsObject.context_tier === 'long_context'
            ? 'long_context'
            : 'default';

        const report = await runCustomizationInventory();
        const agents = report.agents ?? [];
        const target = agents.find((agent) => agent.name === targetName);

        if (!target) {
          return {
            ok: false,
            dispatch_allowed: false,
            reason: `Unknown agent '${targetName}'`,
          };
        }

        const targetTier = Number(target.tier);

        if (
          !Number.isInteger(callerTier) ||
          !ALLOWED_CALLER_TIERS.includes(callerTier)
        ) {
          return {
            ok: false,
            dispatch_allowed: false,
            reason: 'caller_tier must be 0, 1, 2, 3, or 4',
          };
        }

        if (targetTier <= callerTier) {
          return {
            ok: false,
            dispatch_allowed: false,
            reason: `Delegation from Tier ${callerTier} to Tier ${targetTier} is not allowed because delegation must be downward; the target tier must be greater than the caller tier.`,
          };
        }

        if (target.userInvocable === true && targetTier !== 1) {
          return {
            ok: false,
            dispatch_allowed: false,
            reason: `userInvocable is only valid for Tier 1 agents; '${targetName}' is Tier ${targetTier}`,
          };
        }

        return {
          ok: true,
          dispatch_allowed: true,
          reason: `Tier ${callerTier} may delegate to Tier ${targetTier}`,
          agent: {
            name: target.name,
            tier: targetTier,
            tier_label: TIER_LABELS[targetTier] ?? `Tier ${targetTier}`,
            model: target.model ?? null,
            skills: target.skills,
            agents: target.agents,
            tools: target.tools,
            userInvocable: target.userInvocable,
            file: target.path,
          },
          dispatch_packet: {
            agent_type: target.name,
            name: target.name,
            description: target.description,
            model: target.model ?? null,
            prompt,
            context_tier: contextTier,
            skills: target.skills,
          },
        };
      },
    }),
    createTool({
      name: 'get_dispatch_policy',
      description:
        'Return the allowed delegation edges and the user-invocable rule summary.',
      annotations: { readOnlyHint: true },
      handler: async () => ({
        allowed_edges: ALLOWED_EDGES,
        user_invocable_rule: 'Only Tier 1 agents may be userInvocable',
        notes: [
          'This server returns a dispatch packet only; it does not spawn subagents.',
          'Tier 0 (orchestrator) may only call Tier 1 agents.',
        ],
      }),
    }),
  ];
}

async function runDispatchSelfCheck({ server }) {
  const issues = [];

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
        'dispatch-mcp',
        `Expected protocol version ${MCP_PROTOCOL_VERSION}, received ${String(initializeResult.protocolVersion)}.`,
      ),
    );
  }

  const toolListResult = await invokeServerRequest(server, {
    method: 'tools/list',
  });

  if (
    !Array.isArray(toolListResult.tools) ||
    toolListResult.tools.length !== server.tools.length
  ) {
    issues.push(
      selfCheckError(
        'dispatch-mcp',
        `Expected ${server.tools.length} dispatch tools, found ${Array.isArray(toolListResult.tools) ? toolListResult.tools.length : 'none'}.`,
      ),
    );
  }

  const toolNames = (toolListResult.tools ?? [])
    .map((tool) => tool.name)
    .sort();
  const expectedNames = [
    'build_dispatch_packet',
    'get_dispatch_policy',
    'list_dispatchable_agents',
  ];
  if (JSON.stringify(toolNames) !== JSON.stringify(expectedNames)) {
    issues.push(
      selfCheckError(
        'dispatch-mcp',
        `Unexpected tool list: ${toolNames.join(', ')}.`,
      ),
    );
  }

  const listResult = await invokeServerRequest(server, {
    method: 'tools/call',
    params: { name: 'list_dispatchable_agents', arguments: {} },
  });

  if (listResult.isError) {
    issues.push(
      selfCheckError(
        'dispatch-mcp',
        'list_dispatchable_agents returned an error during self-check.',
      ),
    );
  }

  const listPayload = listResult.structuredContent ?? {};
  if (!Array.isArray(listPayload.agents) || listPayload.agents.length === 0) {
    issues.push(
      selfCheckError(
        'dispatch-mcp',
        'list_dispatchable_agents did not return a non-empty agent list.',
      ),
    );
  } else {
    const names = listPayload.agents.map((agent) => agent.name);
    const sorted = names.every(
      (name, index) => index === 0 || names[index - 1] <= name,
    );
    if (!sorted) {
      issues.push(
        selfCheckError(
          'dispatch-mcp',
          'list_dispatchable_agents did not return a sorted list.',
        ),
      );
    }
  }

  const validDispatchResult = await invokeServerRequest(server, {
    method: 'tools/call',
    params: {
      name: 'build_dispatch_packet',
      arguments: {
        target_agent: 'plan-scout',
        caller_tier: 1,
        prompt: 'Continue from the active plan.',
        context_tier: 'default',
      },
    },
  });

  if (validDispatchResult.isError) {
    issues.push(
      selfCheckError(
        'dispatch-mcp',
        'build_dispatch_packet returned an error for a valid delegation.',
      ),
    );
  }

  const validPayload = validDispatchResult.structuredContent ?? {};
  if (validPayload.ok !== true || validPayload.dispatch_allowed !== true) {
    issues.push(
      selfCheckError(
        'dispatch-mcp',
        'build_dispatch_packet did not allow a valid downward delegation.',
      ),
    );
  }

  const upwardResult = await invokeServerRequest(server, {
    method: 'tools/call',
    params: {
      name: 'build_dispatch_packet',
      arguments: { target_agent: '01-planning', caller_tier: 3 },
    },
  });

  if (upwardResult.isError) {
    issues.push(
      selfCheckError(
        'dispatch-mcp',
        'build_dispatch_packet returned an error for the upward delegation test.',
      ),
    );
  }

  const upwardPayload = upwardResult.structuredContent ?? {};
  if (
    upwardPayload.ok !== false ||
    upwardPayload.dispatch_allowed !== false ||
    typeof upwardPayload.reason !== 'string'
  ) {
    issues.push(
      selfCheckError(
        'dispatch-mcp',
        'build_dispatch_packet did not reject an upward delegation.',
      ),
    );
  }

  const unknownResult = await invokeServerRequest(server, {
    method: 'tools/call',
    params: {
      name: 'build_dispatch_packet',
      arguments: { target_agent: '__nonexistent-agent__', caller_tier: 1 },
    },
  });

  if (unknownResult.isError) {
    issues.push(
      selfCheckError(
        'dispatch-mcp',
        'build_dispatch_packet returned an error for the unknown-agent test.',
      ),
    );
  }

  const unknownPayload = unknownResult.structuredContent ?? {};
  if (
    unknownPayload.ok !== false ||
    unknownPayload.dispatch_allowed !== false ||
    typeof unknownPayload.reason !== 'string'
  ) {
    issues.push(
      selfCheckError(
        'dispatch-mcp',
        'build_dispatch_packet did not reject an unknown target agent.',
      ),
    );
  }

  const policyResult = await invokeServerRequest(server, {
    method: 'tools/call',
    params: { name: 'get_dispatch_policy', arguments: {} },
  });

  if (policyResult.isError) {
    issues.push(
      selfCheckError(
        'dispatch-mcp',
        'get_dispatch_policy returned an error during self-check.',
      ),
    );
  }

  const policyPayload = policyResult.structuredContent ?? {};
  if (
    !Array.isArray(policyPayload.allowed_edges) ||
    policyPayload.allowed_edges.length === 0
  ) {
    issues.push(
      selfCheckError(
        'dispatch-mcp',
        'get_dispatch_policy did not return allowed_edges.',
      ),
    );
  }

  if (
    policyPayload.user_invocable_rule !==
    'Only Tier 1 agents may be userInvocable'
  ) {
    issues.push(
      selfCheckError(
        'dispatch-mcp',
        'get_dispatch_policy did not return the expected user-invocable rule.',
      ),
    );
  }

  return createSelfCheckReport('dispatch-mcp self-check', issues, {
    toolsTested: [
      'list_dispatchable_agents',
      'build_dispatch_packet',
      'get_dispatch_policy',
    ],
    toolCount: Array.isArray(toolListResult.tools)
      ? toolListResult.tools.length
      : 0,
  });
}
