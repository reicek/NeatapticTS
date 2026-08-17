/**
 * @module embed-readiness.test
 * @description 100% coverage tests for embed-readiness.mjs.
 */

import { jest } from '@jest/globals';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const sourceFilePath = path.join(__dirname, 'embed-readiness.mjs');

// Shared mock functions
const mockCheckDenseReadiness = jest.fn();
const mockFail = jest.fn();
const mockParseCliArgs = jest.fn();
const mockPrintHelp = jest.fn();
const mockWriteJsonOrText = jest.fn();

jest.unstable_mockModule('./dense-readiness.mjs', () => ({
  checkDenseReadiness: mockCheckDenseReadiness,
}));
jest.unstable_mockModule('./cli-utils.mjs', () => ({
  fail: mockFail,
  parseCliArgs: mockParseCliArgs,
  printHelp: mockPrintHelp,
  writeJsonOrText: mockWriteJsonOrText,
}));

beforeEach(() => {
  jest.clearAllMocks();
  mockCheckDenseReadiness.mockReset();
});

// ---------------------------------------------------------------------------
// getEmbedReadiness
// ---------------------------------------------------------------------------

describe('embed-readiness: getEmbedReadiness', () => {
  it('probes on first call and returns uncached report', async () => {
    mockCheckDenseReadiness.mockResolvedValue({
      ready: true,
      state: 'warm',
      reason: 'all good',
      chunk_count: 100,
      embedding_count: 100,
    });
    await jest.isolateModulesAsync(async () => {
      const { getEmbedReadiness } = await import('./embed-readiness.mjs');
      const result = await getEmbedReadiness({ modelDirectory: '/models' });
      expect(result.cached).toBe(false);
      expect(result.ready).toBe(true);
      expect(result.state).toBe('warm');
      expect(result.reason).toBe('all good');
      expect(result.chunk_count).toBe(100);
      expect(result.embedding_count).toBe(100);
      expect(result.latency_ms).toBeGreaterThanOrEqual(0);
      expect(mockCheckDenseReadiness).toHaveBeenCalledTimes(1);
    });
  });

  it('returns cached result on second call with same options key', async () => {
    mockCheckDenseReadiness.mockResolvedValue({
      ready: true,
      state: 'warm',
      reason: 'cached test',
      chunk_count: 50,
      embedding_count: 50,
    });
    await jest.isolateModulesAsync(async () => {
      const { getEmbedReadiness } = await import('./embed-readiness.mjs');
      const first = await getEmbedReadiness({ modelDirectory: '/models' });
      expect(first.cached).toBe(false);
      const second = await getEmbedReadiness({ modelDirectory: '/models' });
      expect(second.cached).toBe(true);
      expect(second.ready).toBe(true);
      expect(second.state).toBe('warm');
      expect(mockCheckDenseReadiness).toHaveBeenCalledTimes(1);
    });
  });

  it('probes again when options key changes', async () => {
    mockCheckDenseReadiness.mockResolvedValue({
      ready: false,
      state: 'cold',
      reason: 'different',
      chunk_count: 0,
      embedding_count: 0,
    });
    await jest.isolateModulesAsync(async () => {
      const { getEmbedReadiness } = await import('./embed-readiness.mjs');
      await getEmbedReadiness({ modelDirectory: '/models' });
      await getEmbedReadiness({ modelDirectory: '/other' });
      expect(mockCheckDenseReadiness).toHaveBeenCalledTimes(2);
    });
  });

  it('uses databasePath as fallback for corpusDatabasePath in cache key', async () => {
    mockCheckDenseReadiness.mockResolvedValue({
      ready: true,
      state: 'warm',
      reason: '',
      chunk_count: 10,
      embedding_count: 10,
    });
    await jest.isolateModulesAsync(async () => {
      const { getEmbedReadiness } = await import('./embed-readiness.mjs');
      // databasePath is used as fallback when corpusDatabasePath is absent,
      // so these two calls produce the same cache key (second is cached).
      await getEmbedReadiness({ databasePath: '/db' });
      const second = await getEmbedReadiness({ corpusDatabasePath: '/db' });
      expect(second.cached).toBe(true);
      expect(mockCheckDenseReadiness).toHaveBeenCalledTimes(1);
    });
  });

  it('caches when databasePath is used consistently', async () => {
    mockCheckDenseReadiness.mockResolvedValue({
      ready: true,
      state: 'warm',
      reason: '',
      chunk_count: 5,
      embedding_count: 5,
    });
    await jest.isolateModulesAsync(async () => {
      const { getEmbedReadiness } = await import('./embed-readiness.mjs');
      await getEmbedReadiness({ databasePath: '/db' });
      const second = await getEmbedReadiness({ databasePath: '/db' });
      expect(second.cached).toBe(true);
      expect(mockCheckDenseReadiness).toHaveBeenCalledTimes(1);
    });
  });

  it('handles null chunk_count and embedding_count', async () => {
    mockCheckDenseReadiness.mockResolvedValue({
      ready: false,
      state: 'cold',
      reason: 'no data',
      chunk_count: undefined,
      embedding_count: undefined,
    });
    await jest.isolateModulesAsync(async () => {
      const { getEmbedReadiness } = await import('./embed-readiness.mjs');
      const result = await getEmbedReadiness({});
      expect(result.chunk_count).toBeNull();
      expect(result.embedding_count).toBeNull();
    });
  });

  it('handles missing reason in probe report', async () => {
    mockCheckDenseReadiness.mockResolvedValue({
      ready: true,
      state: 'model-only',
      chunk_count: 5,
      embedding_count: 0,
    });
    await jest.isolateModulesAsync(async () => {
      const { getEmbedReadiness } = await import('./embed-readiness.mjs');
      const result = await getEmbedReadiness({});
      expect(result.reason).toBe('');
    });
  });

  it('handles non-boolean ready value', async () => {
    mockCheckDenseReadiness.mockResolvedValue({
      ready: 1,
      state: 'warm',
      reason: 'truthy',
      chunk_count: 10,
      embedding_count: 10,
    });
    await jest.isolateModulesAsync(async () => {
      const { getEmbedReadiness } = await import('./embed-readiness.mjs');
      const result = await getEmbedReadiness({});
      expect(result.ready).toBe(true);
    });
  });
});

// ---------------------------------------------------------------------------
// main() — CLI guard
// ---------------------------------------------------------------------------

describe('embed-readiness: main()', () => {
  const origArgv1 = process.argv[1];

  afterEach(() => {
    process.argv[1] = origArgv1;
  });

  it('prints help when --help is passed', async () => {
    mockParseCliArgs.mockReturnValue({ help: true });
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./embed-readiness.mjs');
    });
    expect(mockPrintHelp).toHaveBeenCalledTimes(1);
    expect(mockWriteJsonOrText).not.toHaveBeenCalled();
  });

  it('runs readiness probe and writes output in text mode', async () => {
    mockParseCliArgs.mockReturnValue({ json: false });
    mockCheckDenseReadiness.mockResolvedValue({
      ready: true,
      state: 'warm',
      reason: 'ready',
      chunk_count: 10,
      embedding_count: 10,
    });
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./embed-readiness.mjs');
    });
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
    const payload = mockWriteJsonOrText.mock.calls[0][0];
    expect(payload.state).toBe('warm');
    expect(mockFail).not.toHaveBeenCalled();
  });

  it('runs readiness probe with database and model options in json mode', async () => {
    mockParseCliArgs.mockReturnValue({
      json: true,
      database: '/db',
      'model-directory': '/m',
      'model-id': 'id',
    });
    mockCheckDenseReadiness.mockResolvedValue({
      ready: true,
      state: 'warm',
      reason: 'ready',
      chunk_count: 10,
      embedding_count: 10,
    });
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./embed-readiness.mjs');
    });
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
    const jsonFlag = mockWriteJsonOrText.mock.calls[0][1];
    expect(jsonFlag).toBe(true);
  });

  it('calls fail on Error thrown by probe', async () => {
    mockParseCliArgs.mockReturnValue({ json: false });
    mockCheckDenseReadiness.mockRejectedValue(new Error('probe failed'));
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./embed-readiness.mjs');
    });
    expect(mockFail).toHaveBeenCalledTimes(1);
    expect(mockFail.mock.calls[0][0]).toBe('probe failed');
  });

  it('calls fail with String(error) for non-Error throws', async () => {
    mockParseCliArgs.mockReturnValue({ json: true });
    mockCheckDenseReadiness.mockRejectedValue('string error');
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./embed-readiness.mjs');
    });
    expect(mockFail).toHaveBeenCalledTimes(1);
    expect(mockFail.mock.calls[0][0]).toBe('string error');
  });
});