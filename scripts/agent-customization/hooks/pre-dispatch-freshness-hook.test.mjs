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
    mockSpawnSync(...args);
    return { status: 0, stdout: '', stderr: '' };
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
      mockReadFileSync.mockReturnValue(
        JSON.stringify({ lastReindex: 'NaN' }),
      );
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
      mockReadFileSync.mockReturnValue(
        JSON.stringify({ lastReindex: recent }),
      );
      const result = checkStaleness(300, 300);
      assert.strictEqual(result.shouldReindex, false);
      assert.ok(result.ageS >= 99 && result.ageS <= 102);
    });

    it('reports stale when beyond grace + threshold', () => {
      const old = Date.now() - 700_000; // 700s ago > 600s threshold
      mockExistsSync.mockReturnValue(true);
      mockReadFileSync.mockReturnValue(
        JSON.stringify({ lastReindex: old }),
      );
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
});