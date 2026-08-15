/**
 * @module cortex-tier-tool.direct.test
 * @description Native-ESM Jest coverage tests for `cortex-tier-tool.mjs`.
 *
 * Runs in the `agent-customization-mjs` Jest project so V8 instruments the
 * source `.mjs` file directly, bypassing the coverage-attribution false negatives
 * caused by `isolateModulesAsync` + dynamic ESM imports under the
 * `agent-customization-scripts` ts-jest transform.
 */
import { jest } from '@jest/globals';
import os from 'node:os';
import path from 'node:path';
import {
  createSliceContextTool,
  createTierGraphTool,
  rebuildIndex,
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

    it('includes violations when includeViolations is true', async () => {
      const tool = createTierGraphTool();
      const result = await tool.handler({ includeViolations: true });

      expect(result).toEqual(
        expect.objectContaining({
          agents: expect.arrayContaining([]),
          violations: expect.any(Array),
          validation: expect.objectContaining({
            ok: true,
            issueCount: expect.any(Number),
          }),
        }),
      );
    });

    it('omits both agents and violations when both flags are false', async () => {
      const tool = createTierGraphTool();
      const result = await tool.handler({
        includeAgents: false,
        includeViolations: false,
      });

      expect(result).toEqual(
        expect.objectContaining({
          agents: [],
          violations: [],
          validation: expect.objectContaining({
            ok: true,
            issueCount: expect.any(Number),
          }),
        }),
      );
    });

    it('uses an empty issue list when validation omits issues', async () => {
      jest.unstable_mockModule('../validate-agent-graph.mjs', () => ({
        collectTierInventory: jest.fn().mockResolvedValue({
          generated_at: '2024-01-01T00:00:00.000Z',
          summary: {},
          agents: [],
        }),
        runValidateAgentGraph: jest.fn().mockResolvedValue({ ok: false }),
      }));
      jest.resetModules();

      const { createTierGraphTool: factory } =
        await import('./cortex-tier-tool.mjs');

      const tool = factory();
      const result = await tool.handler({ includeAgents: false });

      expect(result.violations).toEqual([]);
      expect(result.validation).toEqual({
        ok: false,
        issueCount: 0,
      });
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

    it('throws when the workflow tool set omits get_slice_context', async () => {
      jest.unstable_mockModule('./neataptic-workflow-mcp.mjs', () => ({
        createWorkflowTools: jest.fn().mockReturnValue([]),
      }));
      jest.resetModules();

      const { createSliceContextTool: factory } =
        await import('./cortex-tier-tool.mjs');

      expect(factory).toThrow(
        'get_slice_context tool descriptor missing from neataptic-workflow-mcp tool set.',
      );
    });
  });

  describe('rebuildIndex', () => {
    it('returns success true when buildSemanticIndex resolves', async () => {
      const buildSemanticIndex = jest.fn().mockResolvedValue(undefined);
      jest.unstable_mockModule('../../../rag-index/build-index.mjs', () => ({
        buildSemanticIndex,
      }));
      jest.resetModules();

      const { rebuildIndex: rebuild } = await import('./cortex-tier-tool.mjs');
      const result = await rebuild({
        databasePath: path.join(os.tmpdir(), 'neatapicts-rebuild-test.db'),
      });

      expect(buildSemanticIndex).toHaveBeenCalledWith({
        databasePath: path.join(os.tmpdir(), 'neatapicts-rebuild-test.db'),
      });
      expect(result.success).toBe(true);
    });

    it('returns success false with the error message when buildSemanticIndex throws', async () => {
      jest.unstable_mockModule('../../../rag-index/build-index.mjs', () => ({
        buildSemanticIndex: jest
          .fn()
          .mockRejectedValue(new Error('index build failed')),
      }));
      jest.resetModules();

      const { rebuildIndex: rebuild } = await import('./cortex-tier-tool.mjs');
      const result = await rebuild();

      expect(result.success).toBe(false);
      expect(result.error).toBe('index build failed');
    });

    it('uses an injected buildModule for success', async () => {
      const buildSemanticIndex = jest.fn().mockResolvedValue(undefined);
      const buildModule = { buildSemanticIndex };
      const databasePath = path.join(os.tmpdir(), 'neatapicts-injected.db');

      const result = await rebuildIndex({ databasePath }, buildModule);

      expect(buildSemanticIndex).toHaveBeenCalledWith({ databasePath });
      expect(result.success).toBe(true);
    });

    it('uses an injected buildModule for Error failures', async () => {
      const buildSemanticIndex = jest
        .fn()
        .mockRejectedValue(new Error('injected error'));
      const buildModule = { buildSemanticIndex };
      const databasePath = path.join(os.tmpdir(), 'neatapicts-injected.db');

      const result = await rebuildIndex({ databasePath }, buildModule);

      expect(result.success).toBe(false);
      expect(result.error).toBe('injected error');
    });

    it('uses an injected buildModule for non-Error rejections', async () => {
      const buildSemanticIndex = jest.fn().mockImplementation(() => {
        throw 'string rejection';
      });
      const buildModule = { buildSemanticIndex };
      const databasePath = path.join(os.tmpdir(), 'neatapicts-injected.db');

      const result = await rebuildIndex({ databasePath }, buildModule);

      expect(result.success).toBe(false);
      expect(result.error).toBe('string rejection');
    });
  });
});
