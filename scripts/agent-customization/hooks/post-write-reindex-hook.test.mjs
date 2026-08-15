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
    const result = mockSpawn(...args);
    return result ?? mockChild;
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

const {
  extractFilePath,
  triggerBackgroundReindex,
  safeLog,
  main,
  readHookInput,
} = await import('./post-write-reindex-hook.mjs');

describe('post-write-reindex-hook', () => {
  beforeEach(() => {
    mockSpawn.mockClear();
    mockSpawn.mockReturnValue(mockChild);
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

    it('logs "pid ?" when the spawned child has no pid', () => {
      mockSpawn.mockImplementationOnce(() => ({
        unref: jest.fn(),
      }));
      triggerBackgroundReindex('src/neat.ts');
      assert.ok(mockAppendFileSync.mock.calls.length > 0);
      const logLine = mockAppendFileSync.mock.calls[0][1];
      assert.ok(logLine.includes('pid ?'));
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

  describe('main', () => {
    let writeSpy;

    beforeEach(() => {
      writeSpy = jest
        .spyOn(process.stdout, 'write')
        .mockImplementation(() => true);
    });

    afterEach(() => {
      writeSpy.mockRestore();
    });

    it('writes continue:true for non-write tools', () => {
      main({ tool_name: 'read', tool_input: { path: 'src/foo.ts' } });
      assert.strictEqual(writeSpy.mock.calls.length, 1);
      assert.strictEqual(writeSpy.mock.calls[0][0], '{"continue":true}\n');
      assert.strictEqual(mockSpawn.mock.calls.length, 0);
    });

    it('writes continue:true when a write tool has no filePath', () => {
      main({ tool_name: 'edit', tool_input: {} });
      assert.strictEqual(writeSpy.mock.calls.length, 1);
      assert.strictEqual(writeSpy.mock.calls[0][0], '{"continue":true}\n');
      assert.strictEqual(mockSpawn.mock.calls.length, 0);
    });

    it('triggers reindex and writes context for a write tool with a path', () => {
      main({ tool_name: 'edit', tool_input: { path: 'src/foo.ts' } });
      assert.strictEqual(mockSpawn.mock.calls.length, 1);
      assert.strictEqual(writeSpy.mock.calls.length, 1);
      const output = JSON.parse(writeSpy.mock.calls[0][0]);
      assert.strictEqual(output.continue, true);
      assert.strictEqual(
        output.hookSpecificOutput.hookEventName,
        'PostToolUse',
      );
      assert.ok(
        output.hookSpecificOutput.additionalContext.includes('src/foo.ts'),
      );
    });
  });

  describe('readHookInput', () => {
    it('returns {} when stdin is empty', () => {
      mockReadFileSync.mockReturnValue('');
      assert.deepStrictEqual(readHookInput(), {});
    });

    it('parses valid JSON from stdin', () => {
      mockReadFileSync.mockReturnValue('{"tool_name":"edit"}');
      assert.deepStrictEqual(readHookInput(), { tool_name: 'edit' });
    });

    it('returns {} when JSON parsing fails', () => {
      mockReadFileSync.mockReturnValue('not json');
      assert.deepStrictEqual(readHookInput(), {});
    });
  });
});
