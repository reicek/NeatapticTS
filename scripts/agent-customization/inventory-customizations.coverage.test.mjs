/**
 * @module inventory-customizations.coverage.test
 * @description Targeted branch-coverage test for inventory-customizations.mjs.
 *
 * Covers the `typeof data.complexity === 'string'` true branch in readAgent
 * (line 97) by providing an agent with an explicit string complexity field.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

describe('inventory-customizations branch coverage', () => {
  it('readAgent preserves a string complexity from frontmatter', async () => {
    const utils = await import('./customization-utils.mjs');
    jest.unstable_mockModule('./customization-utils.mjs', () => ({
      ...utils,
      readWorkspaceFile: jest.fn(() =>
        Promise.resolve(
          '---\nname: Complex\ntier: 2\ncomplexity: trivial\n---\nbody\n',
        ),
      ),
    }));
    try {
      await jest.isolateModulesAsync(async () => {
        const { readAgent } = await import('./inventory-customizations.mjs');
        const result = await readAgent('agents/test.agent.md');
        assert.equal(result.name, 'Complex');
        assert.equal(result.complexity, 'trivial');
      });
    } finally {
      jest.unstable_mockModule('./customization-utils.mjs', () => utils);
    }
  });
});