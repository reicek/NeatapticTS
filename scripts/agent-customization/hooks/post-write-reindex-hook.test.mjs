/**
 * @module post-write-reindex-hook.test
 * @description Jest unit tests for the post-write auto-reindex hook.
 *
 * Mocks `node:child_process` spawn and `node:fs` to verify the hook triggers a
 * background reindex for write tools and no-ops for non-write tools. All
 * imports of the module-under-test are dynamic after the mock is installed.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

const mockSpawn = jest.fn();
const mockChild = {
  pid: 12345,
  unref: jest.fn(),
};

jest.unstable_mockModule('node:child_process', () => ({
  spawn: (...args) => {
    mockSpawn(...args);
    return mockChild;
  },
}));

const mockReadFileSync = jest.fn();
const mockAppendFileSync = jest.fn();
const mockMkdirSync = jest.fn();

jest.unstable_mockModule('node:fs', () => ({
  readFileSync: (...args) => mockReadFileSync(...args),
  appendFileSync: (...args) => mockAppendFileSync(...args),
  mkdirSync: (...args) => mockMkdirSync(...args),
}));

const { extractFilePath, triggerBackgroundReindex, safeLog } =
  await import('./post-write-reindex-hook.mjs');

describe('post-write-reindex-hook', () => {
  beforeEach(() => {
    mockSpawn.mockClear();
    mockChild.unref.mockClear();
    mockReadFileSync.mockClear();
    mockAppendFileSync.mockClear();
    mockMkdirSync.mockClear();
    mockReadFileSync.mockReturnValue('{}');
  });

  describe('extractFilePath', () => {
    it('extracts path from tool_input', () => {
      const result = extractFilePath({ tool_input: { path: 'src/neat.ts' } });
      assert.strictEqual(result, 'src/neat.ts');
    });

    it('extracts file_path from arguments', () => {
      const result = extractFilePath({
        arguments: { file_path: 'src/neat.ts' },
      });
      assert.strictEqual(result, 'src/neat.ts');
    });

    it('extracts path from top-level hook input', () => {
      const result = extractFilePath({ path: 'src/neat.ts' });
      assert.strictEqual(result, 'src/neat.ts');
    });

    it('returns null when no path field exists', () => {
      assert.strictEqual(extractFilePath({ tool_input: {} }), null);
      assert.strictEqual(extractFilePath({}), null);
      assert.strictEqual(extractFilePath(null), null);
    });

    it('ignores empty string paths', () => {
      assert.strictEqual(extractFilePath({ path: '  ' }), null);
      assert.strictEqual(extractFilePath({ path: '' }), null);
    });
  });

  describe('triggerBackgroundReindex', () => {
    it('spawns a detached node process and unrefs it', () => {
      triggerBackgroundReindex('src/neat.ts');
      assert.strictEqual(mockSpawn.mock.calls.length, 1);
      assert.strictEqual(mockChild.unref.mock.calls.length, 1);

      const spawnArgs = mockSpawn.mock.calls[0];
      assert.strictEqual(spawnArgs[0], process.execPath);
      assert.strictEqual(spawnArgs[1][0], '--input-type=module');
      assert.strictEqual(spawnArgs[1][1], '-e');
      assert.ok(spawnArgs[1][2].includes('reindexFiles'));
      assert.ok(spawnArgs[1][2].includes('src/neat.ts'));
      assert.strictEqual(spawnArgs[2].detached, true);
      assert.strictEqual(spawnArgs[2].stdio, 'ignore');
    });

    it('logs to the reindex log when spawning', () => {
      triggerBackgroundReindex('src/neat.ts');
      assert.ok(mockAppendFileSync.mock.calls.length > 0);
      const logLine = mockAppendFileSync.mock.calls[0][1];
      assert.ok(logLine.includes('post-write-reindex'));
      assert.ok(logLine.includes('src/neat.ts'));
    });

    it('does not throw when spawn throws', () => {
      mockSpawn.mockImplementationOnce(() => {
        throw new Error('spawn EAGAIN');
      });
      assert.doesNotThrow(() => triggerBackgroundReindex('src/neat.ts'));
    });
  });

  describe('safeLog', () => {
    it('appends a timestamped message', () => {
      safeLog('test message');
      assert.strictEqual(mockAppendFileSync.mock.calls.length, 1);
      assert.ok(mockAppendFileSync.mock.calls[0][1].includes('test message'));
    });

    it('does not throw when appendFileSync fails', () => {
      mockAppendFileSync.mockImplementationOnce(() => {
        throw new Error('ENOENT');
      });
      assert.doesNotThrow(() => safeLog('test'));
    });
  });
});
