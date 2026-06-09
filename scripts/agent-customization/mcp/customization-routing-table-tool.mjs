/**
 * @module customization-routing-table-tool
 * @description MCP tool factory for querying the generated canonical routing table and freshness state.
 */
import {
  collectCustomizationRoutingTable,
  ROUTING_TABLE_PATH,
} from '../generate-agent-skill-routing-table.mjs';
import { runRoutingTableFreshnessGate } from '../gates/routing-table-freshness.gate.mjs';
import { createTool } from './mcp-utils.mjs';

/**
 * Create the `query_customization_routing_table` MCP tool descriptor.
 *
 * @returns {{ name: string, description: string, inputSchema: Record<string, unknown>, handler: Function }} Tool descriptor.
 */
export function createCustomizationRoutingTableTool() {
  return createTool({
    name: 'query_customization_routing_table',
    description:
      'Return the generated canonical agent/skill routing table plus freshness status.',
    annotations: { readOnlyHint: true },
    inputSchema: {
      type: 'object',
      properties: {
        includeRows: {
          type: 'boolean',
          description: 'When false, omit the per-row agent and skill data.',
        },
        includeMarkdown: {
          type: 'boolean',
          description: 'When true, include the full generated markdown body.',
        },
      },
      additionalProperties: false,
    },
    handler: async (argumentsObject) => {
      const includeRows = argumentsObject.includeRows !== false;
      const includeMarkdown = argumentsObject.includeMarkdown === true;
      const table = await collectCustomizationRoutingTable();
      const freshness = await runRoutingTableFreshnessGate();

      return {
        tablePath: ROUTING_TABLE_PATH,
        sourceHash: table.sourceHash,
        summary: {
          agents: table.agentRows.length,
          skills: table.skillRows.length,
          sourceFiles: table.sourceFiles.length,
        },
        freshness,
        rows: includeRows
          ? { agents: table.agentRows, skills: table.skillRows }
          : {},
        markdown: includeMarkdown ? table.markdown : null,
      };
    },
  });
}
