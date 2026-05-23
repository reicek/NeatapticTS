import { collectTierInventory, runValidateAgentGraph } from '../validate-agent-graph.mjs';
import { createTool, MCP_REPO_ROOT } from './mcp-utils.mjs';

export function createTierGraphTool({ workspaceRoot = MCP_REPO_ROOT } = {}) {
  return createTool({
    name: 'query_tier_graph',
    description: 'Return the current agent tier inventory, validation issues, and summary counts.',
    annotations: { readOnlyHint: true },
    inputSchema: {
      type: 'object',
      properties: {
        includeAgents: {
          type: 'boolean',
          description: 'When false, omit the full per-agent inventory from the response.',
        },
        includeViolations: {
          type: 'boolean',
          description: 'When false, omit the validation issue list from the response.',
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
        violations: includeViolations ? validation.issues ?? [] : [],
        validation: {
          ok: validation.ok,
          issueCount: (validation.issues ?? []).length,
        },
      };
    },
  });
}