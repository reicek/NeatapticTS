/**
 * @module cortex-tier-tool.direct.test
 * @description Native-ESM Jest coverage tests for `cortex-tier-tool.mjs`.
 *
 * Runs in the `agent-customization-mjs` Jest project so V8 instruments the
 * source `.mjs` file directly, bypassing the coverage-attribution false negatives
 * caused by `isolateModulesAsync` + dynamic ESM imports under the
 * `agent-customization-scripts` ts-jest transform.
 */
import {
  createSliceContextTool,
  createTierGraphTool,
} from './cortex-tier-tool.mjs';

describe('cortex-tier-tool.mjs direct import coverage', () => {
  describe('createTierGraphTool', () => {
    it('returns a tool named query_tier_graph with the correct schema', () => {
      const tool = createTierGraphTool();

      expect(tool).toEqual(
        expect.objectContaining({
          name: 'query_tier_graph',
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
        }),
      );
    });

    it('returns agents and violations with default parameters', async () => {
      const tool = createTierGraphTool();
      const result = await tool.handler({});

      expect(result).toEqual(
        expect.objectContaining({
          summary: expect.any(Object),
          agents: expect.arrayContaining([]),
          violations: expect.any(Array),
          validation: expect.objectContaining({
            ok: true,
            issueCount: expect.any(Number),
          }),
        }),
      );
    });

    it('omits per-agent details when includeAgents is false', async () => {
      const tool = createTierGraphTool();
      const result = await tool.handler({ includeAgents: false });

      expect(result).toEqual(
        expect.objectContaining({
          agents: [],
          violations: expect.any(Array),
          validation: expect.objectContaining({
            ok: true,
            issueCount: expect.any(Number),
          }),
        }),
      );
    });

    it('omits violations when includeViolations is false', async () => {
      const tool = createTierGraphTool();
      const result = await tool.handler({ includeViolations: false });

      expect(result).toEqual(
        expect.objectContaining({
          agents: expect.arrayContaining([]),
          violations: [],
          validation: expect.objectContaining({
            ok: true,
            issueCount: expect.any(Number),
          }),
        }),
      );
    });
  });

  describe('createSliceContextTool', () => {
    it('returns a tool named get_slice_context with the correct schema', () => {
      const tool = createSliceContextTool();

      expect(tool).toEqual(
        expect.objectContaining({
          name: 'get_slice_context',
          inputSchema: {
            type: 'object',
            properties: {
              slice_id: {
                type: 'string',
                description:
                  'Unique slice identifier (exact slice_id) or the symbolic step label (e.g. B1) for the active step.',
              },
              plan_path: {
                type: 'string',
                description:
                  'Optional repo-relative plan path within plans/ to load for this call only.',
              },
            },
            required: ['slice_id'],
            additionalProperties: false,
          },
        }),
      );
    });
  });
});
