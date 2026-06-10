/**
 * @module cortex-tier-tool
 * @description MCP tool factory for the `query_tier_graph` tool.
 *
 * Exposes the agent-tier inventory and validation results as a single
 * read-only MCP tool, used by the Repo Cortex MCP server to let agents
 * inspect the delegation graph without running scripts directly.
 */
import {
  collectTierInventory,
  runValidateAgentGraph,
} from '../validate-agent-graph.mjs';
import { createTool, MCP_REPO_ROOT } from './mcp-utils.mjs';

/**
 * Create the `query_tier_graph` MCP tool descriptor.
 *
 * Returns the current agent tier inventory and validation issues. The
 * `includeAgents` and `includeViolations` options allow callers to trim
 * the response when only summary counts are needed.
 *
 * @param {{ workspaceRoot?: string }} [options={}] - Optional workspace root override.
 * @returns {{ name: string, description: string, inputSchema: Record<string, unknown>, handler: Function }} Tool descriptor.
 */
export function createTierGraphTool({ workspaceRoot = MCP_REPO_ROOT } = {}) {
  return createTool({
    name: 'query_tier_graph',
    description:
      'Return the current agent tier inventory, validation issues, and summary counts.',
    annotations: { readOnlyHint: true },
    inputSchema: {
      type: 'object',
      properties: {
        includeAgents: {
          type: 'boolean',
          description:
            'When false, omit the full per-agent inventory from the response.',
        },
        includeViolations: {
          type: 'boolean',
          description:
            'When false, omit the validation issue list from the response.',
        },
      },
      additionalProperties: false,
    },
    handler: async (argumentsObject) => {
      const includeAgents = argumentsObject.includeAgents !== false;
      const includeViolations = argumentsObject.includeViolations !== false;
      const inventory = await collectTierInventory({ workspaceRoot });
      const validation = await runValidateAgentGraph({ workspaceRoot });

      return {
        generated_at: inventory.generated_at,
        summary: inventory.summary,
        agents: includeAgents ? inventory.agents : [],
        violations: includeViolations ? (validation.issues ?? []) : [],
        validation: {
          ok: validation.ok,
          issueCount: (validation.issues ?? []).length,
        },
      };
    },
  });
}
