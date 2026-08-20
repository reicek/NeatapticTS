/**
 * @module pre-dispatch-freshness-hook.test
 * @description Jest unit tests for the pre-dispatch freshness hook.
 *
 * Mocks `node:child_process` (spawn/spawnSync) and `node:fs` to verify
 * staleness detection, background reindex triggering, and graceful error
 * handling. All imports of the module-under-test are dynamic after the mock
 * is installed.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

const mockSpawn = jest.fn();
const mockSpawnSync = jest.fn();
const mockChild = {
  pid: 4242,
  unref: jest.fn(),
  on: jest.fn(),
};

jest.unstable_mockModule('node:child_process', () => ({
  spawn: (...args) => {
    mockSpawn(...args);
    return mockChild;
  },
  spawnSync: (...args) => {
    const result = mockSpawnSync(...args);
    return result ?? { status: 0, stdout: '', stderr: '' };
  },
}));

const mockExistsSync = jest.fn();
const mockReadFileSync = jest.fn();
const mockWriteFileSync = jest.fn();
const mockAppendFileSync = jest.fn();
const mockMkdirSync = jest.fn();

jest.unstable_mockModule('node:fs', () => ({
  existsSync: (...args) => mockExistsSync(...args),
  readFileSync: (...args) => mockReadFileSync(...args),
  writeFileSync: (...args) => mockWriteFileSync(...args),
  appendFileSync: (...args) => mockAppendFileSync(...args),
  mkdirSync: (...args) => mockMkdirSync(...args),
}));

const {
  checkStaleness,
  readManifestTimestamp,
  updateManifest,
  runReindexBackground,
  runReindexSync,
  parseEnvInt,
  safeLog,
} = await import('./pre-dispatch-freshness-hook.mjs');

describe('pre-dispatch-freshness-hook', () => {
  beforeEach(() => {
    mockSpawn.mockClear();
    mockSpawnSync.mockClear();
    mockChild.unref.mockClear();
    mockChild.on.mockClear();
    mockExistsSync.mockClear();
    mockReadFileSync.mockClear();
    mockWriteFileSync.mockClear();
    mockAppendFileSync.mockClear();
    mockMkdirSync.mockClear();
    mockReadFileSync.mockReturnValue('{}');
  });

  describe('parseEnvInt', () => {
    it('returns the default when env var is not set', () => {
      delete process.env.TEST_GRACE;
      assert.strictEqual(parseEnvInt('TEST_GRACE', 300), 300);
    });

    it('parses a valid integer', () => {
      process.env.TEST_GRACE = '600';
      assert.strictEqual(parseEnvInt('TEST_GRACE', 300), 600);
      delete process.env.TEST_GRACE;
    });

    it('returns default for non-numeric values', () => {
      process.env.TEST_GRACE = 'abc';
      assert.strictEqual(parseEnvInt('TEST_GRACE', 300), 300);
      delete process.env.TEST_GRACE;
    });

    it('returns default for negative values', () => {
      process.env.TEST_GRACE = '-5';
      assert.strictEqual(parseEnvInt('TEST_GRACE', 300), 300);
      delete process.env.TEST_GRACE;
    });
  });

  describe('readManifestTimestamp', () => {
    it('returns null when manifest does not exist', () => {
      mockExistsSync.mockReturnValue(false);
      assert.strictEqual(readManifestTimestamp(), null);
    });

    it('returns timestamp from manifest', () => {
      mockExistsSync.mockReturnValue(true);
      mockReadFileSync.mockReturnValue(
        JSON.stringify({ lastReindex: 1700000000000 }),
      );
      assert.strictEqual(readManifestTimestamp(), 1700000000000);
    });

    it('returns null on malformed JSON', () => {
      mockExistsSync.mockReturnValue(true);
      mockReadFileSync.mockReturnValue('{not json');
      assert.strictEqual(readManifestTimestamp(), null);
    });

    it('returns null when lastReindex is not finite', () => {
      mockExistsSync.mockReturnValue(true);
      mockReadFileSync.mockReturnValue(JSON.stringify({ lastReindex: 'NaN' }));
      assert.strictEqual(readManifestTimestamp(), null);
    });
  });

  describe('checkStaleness', () => {
    it('reports stale (Infinity age) when no manifest exists', () => {
      mockExistsSync.mockReturnValue(false);
      const result = checkStaleness(300, 300);
      assert.strictEqual(result.shouldReindex, true);
      assert.strictEqual(result.ageS, Infinity);
      assert.strictEqual(result.thresholdS, 600);
    });

    it('reports fresh when within grace window', () => {
      const recent = Date.now() - 100_000; // 100s ago
      mockExistsSync.mockReturnValue(true);
      mockReadFileSync.mockReturnValue(JSON.stringify({ lastReindex: recent }));
      const result = checkStaleness(300, 300);
      assert.strictEqual(result.shouldReindex, false);
      assert.ok(result.ageS >= 99 && result.ageS <= 102);
    });

    it('reports stale when beyond grace + threshold', () => {
      const old = Date.now() - 700_000; // 700s ago > 600s threshold
      mockExistsSync.mockReturnValue(true);
      mockReadFileSync.mockReturnValue(JSON.stringify({ lastReindex: old }));
      const result = checkStaleness(300, 300);
      assert.strictEqual(result.shouldReindex, true);
      assert.ok(result.ageS >= 698);
    });

    it('reports not stale when exactly at threshold boundary', () => {
      const boundary = Date.now() - 600_000; // exactly 600s
      mockExistsSync.mockReturnValue(true);
      mockReadFileSync.mockReturnValue(
        JSON.stringify({ lastReindex: boundary }),
      );
      const result = checkStaleness(300, 300);
      assert.strictEqual(result.shouldReindex, false);
    });
  });

  describe('updateManifest', () => {
    it('writes a manifest with the current timestamp', () => {
      updateManifest();
      assert.ok(mockMkdirSync.mock.calls.length >= 1);
      assert.ok(mockWriteFileSync.mock.calls.length >= 1);
      const content = mockWriteFileSync.mock.calls[0][1];
      const parsed = JSON.parse(content);
      assert.ok(parsed.lastReindex > 0);
      assert.strictEqual(parsed.updatedBy, 'pre-dispatch-freshness-hook');
    });

    it('does not throw when writeFileSync fails', () => {
      mockWriteFileSync.mockImplementationOnce(() => {
        throw new Error('EACCES');
      });
      assert.doesNotThrow(() => updateManifest());
    });
  });

  describe('runReindexBackground', () => {
    it('spawns a detached build-index process and unrefs it', () => {
      runReindexBackground();
      assert.ok(mockSpawn.mock.calls.length >= 1);
      assert.ok(mockChild.unref.mock.calls.length >= 1);
      assert.ok(mockChild.on.mock.calls.length >= 1);

      const spawnArgs = mockSpawn.mock.calls[0];
      assert.strictEqual(spawnArgs[0], process.execPath);
      assert.ok(spawnArgs[1].includes('--force'));
      assert.strictEqual(spawnArgs[2].detached, true);
      assert.strictEqual(spawnArgs[2].stdio, 'ignore');
    });

    it('does not throw when spawn throws', () => {
      mockSpawn.mockImplementationOnce(() => {
        throw new Error('EAGAIN');
      });
      assert.doesNotThrow(() => runReindexBackground());
    });

    it('logs question mark when child pid is null', () => {
      const originalPid = mockChild.pid;
      mockChild.pid = null;
      runReindexBackground();
      mockChild.pid = originalPid;
      assert.ok(
        mockAppendFileSync.mock.calls.some((call) => call[1].includes('pid ?')),
      );
    });

    it('spawns embed-index when build-index exits with code 0', () => {
      runReindexBackground();
      const exitCall = mockChild.on.mock.calls.find(
        (call) => call[0] === 'exit',
      );
      exitCall[1](0);
      assert.ok(mockSpawn.mock.calls.length >= 2);
    });

    it('logs question mark for embed child pid when null on exit code 0', () => {
      const originalPid = mockChild.pid;
      runReindexBackground();
      mockChild.pid = null;
      const exitCall = mockChild.on.mock.calls.find(
        (call) => call[0] === 'exit',
      );
      exitCall[1](0);
      mockChild.pid = originalPid;
      assert.ok(
        mockAppendFileSync.mock.calls.some((call) => call[1].includes('pid ?')),
      );
    });

    it('logs exit code when build-index exits with non-zero code', () => {
      runReindexBackground();
      const exitCall = mockChild.on.mock.calls.find(
        (call) => call[0] === 'exit',
      );
      exitCall[1](1);
      assert.ok(
        mockAppendFileSync.mock.calls.some((call) =>
          call[1].includes('code 1'),
        ),
      );
    });

    it('logs embed-index spawn failure when spawn throws in exit callback', () => {
      runReindexBackground();
      mockSpawn.mockImplementationOnce(() => {
        throw new Error('embed spawn failed');
      });
      const exitCall = mockChild.on.mock.calls.find(
        (call) => call[0] === 'exit',
      );
      exitCall[1](0);
      assert.ok(
        mockAppendFileSync.mock.calls.some((call) =>
          call[1].includes('embed-index spawn failed'),
        ),
      );
    });
  });

  describe('runReindexSync', () => {
    it('calls spawnSync for build-index and embed-index', () => {
      runReindexSync();
      assert.ok(mockSpawnSync.mock.calls.length >= 2);
      const buildArgs = mockSpawnSync.mock.calls[0][1];
      assert.ok(buildArgs.includes('--force'));
      const embedArgs = mockSpawnSync.mock.calls[1][1];
      assert.ok(embedArgs.includes('--json'));
    });

    it('does not throw when spawnSync throws', () => {
      mockSpawnSync.mockImplementationOnce(() => {
        throw new Error('ENOENT');
      });
      assert.doesNotThrow(() => runReindexSync());
    });

    it('logs question mark when build status is null in sync reindex', () => {
      mockSpawnSync.mockReturnValueOnce({
        status: null,
        stdout: '',
        stderr: '',
      });
      runReindexSync();
      assert.ok(
        mockAppendFileSync.mock.calls.some((call) =>
          call[1].includes('status ?'),
        ),
      );
    });

    it('logs question mark when embed status is null in sync reindex', () => {
      mockSpawnSync
        .mockReturnValueOnce({ status: 0, stdout: '', stderr: '' })
        .mockReturnValueOnce({ status: null, stdout: '', stderr: '' });
      runReindexSync();
      assert.ok(
        mockAppendFileSync.mock.calls.some((call) =>
          call[1].includes('status ?'),
        ),
      );
    });
  });

  describe('safeLog', () => {
    it('appends a timestamped message', () => {
      safeLog('test event');
      assert.strictEqual(mockAppendFileSync.mock.calls.length, 1);
      assert.ok(mockAppendFileSync.mock.calls[0][1].includes('test event'));
    });

    it('does not throw when appendFileSync fails', () => {
      mockAppendFileSync.mockImplementationOnce(() => {
        throw new Error('ENOSPC');
      });
      assert.doesNotThrow(() => safeLog('test'));
    });
  });

  describe('main() flow', () => {
    it('logs fresh when index is within grace window', async () => {
      mockExistsSync.mockReturnValue(true);
      mockReadFileSync.mockImplementation((filePath) => {
        if (filePath === 0) return '{}';
        return JSON.stringify({ lastReindex: Date.now() });
      });
      const origWrite = process.stdout.write;
      const messages = [];
      process.stdout.write = (chunk) => {
        messages.push(String(chunk));
        return true;
      };
      jest.resetModules();
      await import('./pre-dispatch-freshness-hook.mjs');
      await new Promise((r) => setTimeout(r, 200));
      process.stdout.write = origWrite;
      assert.ok(
        mockAppendFileSync.mock.calls.some((call) => call[1].includes('fresh')),
      );
    });

    it('runs sync reindex when wait_for_reindex is true', async () => {
      mockExistsSync.mockReturnValue(false);
      mockReadFileSync.mockImplementation((filePath) => {
        if (filePath === 0) return JSON.stringify({ wait_for_reindex: true });
        return '{}';
      });
      const origWrite = process.stdout.write;
      process.stdout.write = () => true;
      jest.resetModules();
      await import('./pre-dispatch-freshness-hook.mjs');
      await new Promise((r) => setTimeout(r, 200));
      process.stdout.write = origWrite;
      assert.ok(mockSpawnSync.mock.calls.length >= 2);
    });

    it('logs fatal error message when main rejects with Error', async () => {
      mockExistsSync.mockReturnValue(false);
      mockReadFileSync.mockReturnValue('{}');
      const origWrite = process.stdout.write;
      let firstCall = true;
      process.stdout.write = () => {
        if (firstCall) {
          firstCall = false;
          throw new Error('write failed');
        }
        return true;
      };
      jest.resetModules();
      await import('./pre-dispatch-freshness-hook.mjs');
      await new Promise((r) => setTimeout(r, 200));
      process.stdout.write = origWrite;
      assert.ok(
        mockAppendFileSync.mock.calls.some((call) =>
          call[1].includes('fatal: write failed'),
        ),
      );
    });

    it('logs fatal message when main rejects with non-Error', async () => {
      mockExistsSync.mockReturnValue(false);
      mockReadFileSync.mockReturnValue('{}');
      const origWrite = process.stdout.write;
      let firstCall = true;
      process.stdout.write = () => {
        if (firstCall) {
          firstCall = false;
          throw 'string error';
        }
        return true;
      };
      jest.resetModules();
      await import('./pre-dispatch-freshness-hook.mjs');
      await new Promise((r) => setTimeout(r, 200));
      process.stdout.write = origWrite;
      assert.ok(
        mockAppendFileSync.mock.calls.some((call) =>
          call[1].includes('fatal: string error'),
        ),
      );
    });

    it('returns empty object when stdin is empty', async () => {
      mockExistsSync.mockReturnValue(false);
      mockReadFileSync.mockReturnValue('');
      const origWrite = process.stdout.write;
      const messages = [];
      process.stdout.write = (chunk) => {
        messages.push(String(chunk));
        return true;
      };
      jest.resetModules();
      await import('./pre-dispatch-freshness-hook.mjs');
      await new Promise((r) => setTimeout(r, 200));
      process.stdout.write = origWrite;
      const output = JSON.parse(messages[messages.length - 1].trim());
      assert.strictEqual(output.continue, true);
    });

    it('parses hook input when stdin is non-empty', async () => {
      mockExistsSync.mockReturnValue(false);
      mockReadFileSync.mockReturnValue('{}');
      const origWrite = process.stdout.write;
      const messages = [];
      process.stdout.write = (chunk) => {
        messages.push(String(chunk));
        return true;
      };
      jest.resetModules();
      await import('./pre-dispatch-freshness-hook.mjs');
      await new Promise((r) => setTimeout(r, 200));
      process.stdout.write = origWrite;
      const output = JSON.parse(messages[messages.length - 1].trim());
      assert.strictEqual(output.continue, true);
    });
  });
});
