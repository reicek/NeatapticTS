/**
 * @module cortex-tier-tool
 * @description MCP tool factories for Repo Cortex-aware tier tools.
 *
 * Exposes `query_tier_graph` (agent-tier inventory) and `get_slice_context`
 * (assembled plan slice context) as read-only tools that can be served by the
 * gate MCP server and lazy cortex facade without requiring callers to run raw
 * scripts.
 */
import {
  collectTierInventory,
  runValidateAgentGraph,
} from '../validate-agent-graph.mjs';
import { createTool, MCP_REPO_ROOT } from './mcp-utils.mjs';
import { createWorkflowTools } from './neataptic-workflow-mcp.mjs';

/** Default plan used when no `plan_path` is supplied to the slice-context tool. */
const DEFAULT_SLICE_CONTEXT_PLAN_PATH = 'plans/mcp-active-binding.plans.md';

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

/**
 * Create the `get_slice_context` MCP tool descriptor.
 *
 * Reuses the canonical slice-context handler from
 * {@link createWorkflowTools} so that gate and facade servers expose the same
 * compact context window as the standalone workflow MCP server. The returned
 * tool accepts `{ slice_id, plan_path? }` and returns a compact summary
 * (~700 bytes) of essential slice fields. To retrieve the full plan document,
 * use neataptic-cortex-mcp:load_document instead.
 *
 * @param {{ planPath?: string }} [options={}] - Tool options.
 * @param {string} [options.planPath='plans/mcp-active-binding.plans.md'] - Default repo-relative plan path used when a caller does not supply `plan_path`.
 * @returns {{ name: string, description: string, annotations: Record<string, unknown>, inputSchema: Record<string, unknown>, handler: Function }} Tool descriptor.
 */
export function createSliceContextTool({
  planPath = DEFAULT_SLICE_CONTEXT_PLAN_PATH,
} = {}) {
  const workflowTools = createWorkflowTools({ planPath });
  const sliceTool = workflowTools.find(
    (tool) => tool.name === 'get_slice_context',
  );
  if (!sliceTool) {
    throw new Error(
      'get_slice_context tool descriptor missing from neataptic-workflow-mcp tool set.',
    );
  }

  return createTool({
    name: sliceTool.name,
    description: sliceTool.description,
    annotations: sliceTool.annotations,
    inputSchema: sliceTool.inputSchema,
    handler: sliceTool.handler,
  });
}

/**
 * Rebuild the semantic index used by Repo Cortex search tools.
 *
 * This helper is the reusable implementation behind the `cortex-index` gate's
 * `--auto-rebuild` mode and the `04-implementing` preflight freshness check.
 * It runs the same incremental builder as `node rag-index/build-index.mjs` and
 * returns a structured result so callers can decide whether to re-validate.
 *
 * @param {{ databasePath?: string }} [options={}] - Rebuild options.
 * @param {string} [options.databasePath] - Absolute or repo-relative path to the SQLite database. Defaults to the canonical semantic-index database.
 * @returns {Promise<{ success: boolean; error?: string }>} Result object. `success` is true when the builder finished without throwing; `error` contains the message on failure.
 */
export async function rebuildIndex(options = {}, buildModule = null) {
  try {
    const { buildSemanticIndex } =
      buildModule ?? (await import('../../../rag-index/build-index.mjs'));
    await buildSemanticIndex({
      databasePath: options.databasePath,
    });
    return { success: true };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}
