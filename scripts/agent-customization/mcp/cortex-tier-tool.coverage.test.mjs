/**
 * @module cortex-tier-tool.coverage.test
 * @description Branch-coverage supplement for `cortex-tier-tool.mjs`.
 *
 * Covers the default-parameter branches that are missed when the tool factories
 * are only called without an explicit options argument.
 */
import {
  createSliceContextTool,
  createTierGraphTool,
} from './cortex-tier-tool.mjs';

describe('cortex-tier-tool.mjs default-parameter branch coverage', () => {
  it('accepts an explicit workspaceRoot without using the default', () => {
    const tool = createTierGraphTool({ workspaceRoot: '/explicit/path' });

    expect(tool.name).toBe('query_tier_graph');
  });

  it('accepts an explicit planPath without using the default', () => {
    const tool = createSliceContextTool({
      planPath: 'plans/test-coverage.plans.md',
    });

    expect(tool.name).toBe('get_slice_context');
  });
});