import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

jest.unstable_mockModule('./lazy-facade-core.mjs', () => ({
  createLazyFacade: jest.fn((config) => ({ fake: true, config })),
  runFacadeMain: jest.fn(),
}));

jest.unstable_mockModule('./cortex-tier-tool.mjs', () => ({
  createSliceContextTool: jest.fn(() => ({ name: 'get_slice_context', description: 'slice context', handler: jest.fn() })),
}));

let mockLazyFacade;
let mockTierTool;

beforeEach(async () => {
  jest.resetModules();
  mockLazyFacade = await import('./lazy-facade-core.mjs');
  mockTierTool = await import('./cortex-tier-tool.mjs');
  jest.clearAllMocks();
});

describe('cortex-facade', () => {
  it('exports createCortexFacade that delegates to createLazyFacade', async () => {
    const { createCortexFacade } = await import('./cortex-facade.mjs');

    const result = createCortexFacade();

    assert.ok(mockLazyFacade.createLazyFacade.mock.calls.length >= 1);
    const config = mockLazyFacade.createLazyFacade.mock.calls[0][0];
    assert.strictEqual(config.name, 'cortex');
    assert.strictEqual(config.target, 'cortex');
    assert.strictEqual(config.title, 'NeatapticTS Repo Cortex');
    assert.strictEqual(config.version, '0.1.0');
    assert.ok(config.defaultSnapshotPath);
    assert.deepStrictEqual(config.defaultSpawnCommand, ['node', 'scripts/mcp-semantic/repo-cortex-mcp.mjs']);
    assert.ok(config.localTools.length > 0);
    assert.strictEqual(result.fake, true);
  });

  it('passes custom options through to createLazyFacade', async () => {
    const { createCortexFacade } = await import('./cortex-facade.mjs');

    createCortexFacade({ snapshotPath: '/custom/path', spawnCommand: ['custom'] });

    const config = mockLazyFacade.createLazyFacade.mock.calls[0][0];
    assert.strictEqual(config.snapshotPath, '/custom/path');
    assert.deepStrictEqual(config.spawnCommand, ['custom']);
  });

  it('calls createSliceContextTool for default local tools', async () => {
    await import('./cortex-facade.mjs');

    assert.ok(mockTierTool.createSliceContextTool.mock.calls.length >= 1);
  });

  it('does not call runFacadeMain when not main module', async () => {
    await import('./cortex-facade.mjs');

    assert.strictEqual(mockLazyFacade.runFacadeMain.mock.calls.length, 0);
  });

  // MUST BE LAST TEST — sets process.argv[1] so isMain is true, exercising the
  // runFacadeMain main-entry branch (line 58).
  it('calls runFacadeMain when invoked as main entry point', async () => {
    const facadePath = path.resolve(
      path.dirname(fileURLToPath(import.meta.url)),
      'cortex-facade.mjs',
    );
    const origArgv1 = process.argv[1];
    process.argv[1] = facadePath;
    try {
      await import('./cortex-facade.mjs');
      assert.ok(mockLazyFacade.runFacadeMain.mock.calls.length > 0);
      const config = mockLazyFacade.runFacadeMain.mock.calls[0][0];
      assert.strictEqual(config.name, 'cortex');
    } finally {
      process.argv[1] = origArgv1;
    }
  });
});