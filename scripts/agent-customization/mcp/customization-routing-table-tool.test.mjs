/**
 * @module customization-routing-table-tool.test
 * @description Coverage tests for `customization-routing-table-tool.mjs`.
 */
import { jest } from '@jest/globals';

const fakeTable = {
  sourceHash: 'abc123',
  sourceFiles: ['src/agents/a.md', 'src/skills/b.md'],
  agentRows: [{ name: 'agent-a' }, { name: 'agent-b' }],
  skillRows: [{ name: 'skill-x' }],
  markdown: '# Routing Table\n',
};

const fakeFreshness = {
  pass: true,
  evidence: { hash: 'abc123' },
};

async function importTool() {
  jest.unstable_mockModule(
    '../generate-agent-skill-routing-table.mjs',
    () => ({
      collectCustomizationRoutingTable: jest
        .fn()
        .mockResolvedValue(fakeTable),
      ROUTING_TABLE_PATH: '.github/agent-skill-routing-table.md',
    }),
  );
  jest.unstable_mockModule(
    '../gates/routing-table-freshness.gate.mjs',
    () => ({
      runRoutingTableFreshnessGate: jest
        .fn()
        .mockResolvedValue(fakeFreshness),
    }),
  );
  jest.resetModules();

  const mod = await import('./customization-routing-table-tool.mjs');
  return mod;
}

describe('customization-routing-table-tool.mjs', () => {
  describe('createCustomizationRoutingTableTool', () => {
    it('returns a tool named query_customization_routing_table with the correct schema', async () => {
      const { createCustomizationRoutingTableTool } = await importTool();
      const tool = createCustomizationRoutingTableTool();

      expect(tool.name).toBe('query_customization_routing_table');
      expect(tool.inputSchema).toEqual({
        type: 'object',
        properties: {
          includeRows: {
            type: 'boolean',
            description:
              'When false, omit the per-row agent and skill data.',
          },
          includeMarkdown: {
            type: 'boolean',
            description:
              'When true, include the full generated markdown body.',
          },
        },
        additionalProperties: false,
      });
    });

    it('returns rows and null markdown with default parameters', async () => {
      const { createCustomizationRoutingTableTool } = await importTool();
      const tool = createCustomizationRoutingTableTool();
      const result = await tool.handler({});

      expect(result.tablePath).toBe('.github/agent-skill-routing-table.md');
      expect(result.sourceHash).toBe('abc123');
      expect(result.summary).toEqual({
        agents: 2,
        skills: 1,
        sourceFiles: 2,
      });
      expect(result.freshness).toEqual(fakeFreshness);
      expect(result.rows).toEqual({
        agents: fakeTable.agentRows,
        skills: fakeTable.skillRows,
      });
      expect(result.markdown).toBeNull();
    });

    it('omits rows when includeRows is false', async () => {
      const { createCustomizationRoutingTableTool } = await importTool();
      const tool = createCustomizationRoutingTableTool();
      const result = await tool.handler({ includeRows: false });

      expect(result.rows).toEqual({});
    });

    it('includes markdown when includeMarkdown is true', async () => {
      const { createCustomizationRoutingTableTool } = await importTool();
      const tool = createCustomizationRoutingTableTool();
      const result = await tool.handler({ includeMarkdown: true });

      expect(result.markdown).toBe('# Routing Table\n');
    });

    it('omits rows and includes markdown when both flags are set', async () => {
      const { createCustomizationRoutingTableTool } = await importTool();
      const tool = createCustomizationRoutingTableTool();
      const result = await tool.handler({
        includeRows: false,
        includeMarkdown: true,
      });

      expect(result.rows).toEqual({});
      expect(result.markdown).toBe('# Routing Table\n');
    });
  });
});