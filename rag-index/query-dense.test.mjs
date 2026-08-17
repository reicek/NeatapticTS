/**
 * @module query-dense.test
 * @description 100% coverage tests for query-dense.mjs.
 */

import { jest } from '@jest/globals';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const sourceFilePath = path.join(__dirname, 'query-dense.mjs');

// Shared mock functions
const mockFail = jest.fn();
const mockParseCliArgs = jest.fn();
const mockPrintHelp = jest.fn();
const mockWriteJsonOrText = jest.fn();

const mockCreateOnnxTextEmbedder = jest.fn();
const mockNormalizeEmbeddingVector = jest.fn();
const mockReadModelMeta = jest.fn();

const mockRankRRFResults = jest.fn();

const mockGetTursoClient = jest.fn();
const mockNormalizeLimit = jest.fn();
const mockReadChunkRow = jest.fn();
const mockSanitizeFtsQuery = jest.fn();

const mockCompileFilterToSqlAliased = jest.fn();
const mockValidateFilter = jest.fn();

jest.unstable_mockModule('./cli-utils.mjs', () => ({
  fail: mockFail,
  parseCliArgs: mockParseCliArgs,
  printHelp: mockPrintHelp,
  writeJsonOrText: mockWriteJsonOrText,
}));
jest.unstable_mockModule('./embed-index.mjs', () => ({
  DEFAULT_MODEL_DIRECTORY: '/fake/model',
  DEFAULT_MODEL_ID: 'fake-model-id',
  createOnnxTextEmbedder: mockCreateOnnxTextEmbedder,
  normalizeEmbeddingVector: mockNormalizeEmbeddingVector,
  readModelMeta: mockReadModelMeta,
}));
jest.unstable_mockModule('./hybrid-rank.mjs', () => ({
  rankRRFResults: mockRankRRFResults,
}));
jest.unstable_mockModule('../scripts/mcp-semantic/tools/cortex-db.mjs', () => ({
  getTursoClient: mockGetTursoClient,
  normalizeLimit: mockNormalizeLimit,
  readChunkRow: mockReadChunkRow,
  sanitizeFtsQuery: mockSanitizeFtsQuery,
}));
jest.unstable_mockModule('./metadata-filter.mjs', () => ({
  compileFilterToSqlAliased: mockCompileFilterToSqlAliased,
  validateFilter: mockValidateFilter,
}));

const { queryDenseIndex } = await import('./query-dense.mjs');

function setupDefaults() {
  mockSanitizeFtsQuery.mockImplementation((q) => q);
  mockNormalizeLimit.mockImplementation((value, fallback = 10) => {
    const n = Number(value ?? fallback);
    if (!Number.isFinite(n) || n < 1) return 1;
    return Math.min(Math.trunc(n), 50);
  });
  mockReadChunkRow.mockImplementation((row) => ({
    chunk_id: row.chunk_id,
    file_path: row.file_path,
    family: row.doc_family,
    heading_path: row.heading_path,
    body_text: row.body_text,
  }));
  mockRankRRFResults.mockImplementation(({ bm25Results, denseResults }) => {
    const merged = [...(bm25Results || []), ...(denseResults || [])];
    return merged.map((r, i) => ({
      ...r,
      rrf_score: 1 / (60 + i + 1),
      bm25_score: r.score,
      distance: r.distance,
    }));
  });
}

beforeEach(() => {
  jest.clearAllMocks();
  setupDefaults();
});

function makeMockClient(rows = []) {
  return {
    execute: jest.fn().mockResolvedValue({ rows }),
    close: jest.fn().mockResolvedValue(undefined),
  };
}

// ---------------------------------------------------------------------------
// Empty query
// ---------------------------------------------------------------------------

describe('query-dense: empty query returns empty results', () => {
  it('returns empty results for empty string', async () => {
    const result = await queryDenseIndex({ query: '' });
    expect(result.results).toEqual([]);
    expect(result.use_dense).toBe(false);
  });

  it('returns empty results for whitespace-only query', async () => {
    mockSanitizeFtsQuery.mockReturnValue('');
    const result = await queryDenseIndex({ query: '   ' });
    expect(result.results).toEqual([]);
  });

  it('returns empty results for undefined query', async () => {
    const result = await queryDenseIndex({});
    expect(result.results).toEqual([]);
  });

  it('includes family when provided with empty query', async () => {
    const result = await queryDenseIndex({ query: '', family: 'docs' });
    expect(result.family).toBe('docs');
  });

  it('includes alpha and use_dense when dense enabled with empty query', async () => {
    const result = await queryDenseIndex({ query: '', dense: true });
    expect(result.use_dense).toBe(true);
    expect(result.alpha).toBe(0.5);
  });
});

// ---------------------------------------------------------------------------
// BM25-only mode (default)
// ---------------------------------------------------------------------------

describe('query-dense: BM25-only mode', () => {
  it('queries with client and no family', async () => {
    const rows = [
      { chunk_id: 1, file_path: 'a.ts', doc_family: 'docs', heading_path: 'h', body_text: 'b', score: 1.5 },
    ];
    const client = makeMockClient(rows);
    const result = await queryDenseIndex({ query: 'neat', client });
    expect(result.use_dense).toBe(false);
    expect(result.results).toHaveLength(1);
    expect(client.execute).toHaveBeenCalledTimes(1);
  });

  it('queries with client and family', async () => {
    const rows = [
      { chunk_id: 1, file_path: 'a.ts', doc_family: 'docs', heading_path: 'h', body_text: 'b', score: 1.5 },
    ];
    const client = makeMockClient(rows);
    const result = await queryDenseIndex({ query: 'neat', client, family: 'docs' });
    expect(result.family).toBe('docs');
    expect(client.execute).toHaveBeenCalledTimes(1);
    const callArgs = client.execute.mock.calls[0][0];
    expect(callArgs.args).toContain('docs');
  });

  it('queries without client (uses getTursoClient)', async () => {
    const rows = [
      { chunk_id: 1, file_path: 'a.ts', doc_family: 'docs', heading_path: 'h', body_text: 'b', score: 1.5 },
    ];
    const dbClient = makeMockClient(rows);
    mockGetTursoClient.mockResolvedValue(dbClient);
    const result = await queryDenseIndex({ query: 'neat', databasePath: '/fake/db' });
    expect(result.results).toHaveLength(1);
    expect(mockGetTursoClient).toHaveBeenCalledWith('/fake/db');
  });

  it('queries without client and with family', async () => {
    const rows = [
      { chunk_id: 1, file_path: 'a.ts', doc_family: 'docs', heading_path: 'h', body_text: 'b', score: 1.5 },
    ];
    const dbClient = makeMockClient(rows);
    mockGetTursoClient.mockResolvedValue(dbClient);
    const result = await queryDenseIndex({ query: 'neat', databasePath: '/fake/db', family: 'docs' });
    expect(result.family).toBe('docs');
  });

  it('respects limit and slices results', async () => {
    const rows = Array.from({ length: 20 }, (_, i) => ({
      chunk_id: i, file_path: `f${i}.ts`, doc_family: 'docs', heading_path: '', body_text: '', score: 1,
    }));
    const client = makeMockClient(rows);
    const result = await queryDenseIndex({ query: 'neat', client, limit: 5 });
    expect(result.results).toHaveLength(5);
  });

  it('uses compiledFilter when provided', async () => {
    const rows = [
      { chunk_id: 1, file_path: 'a.ts', doc_family: 'docs', heading_path: 'h', body_text: 'b', score: 1.5 },
    ];
    const client = makeMockClient(rows);
    const compiledFilter = { sql: 'd.lang = ?', params: ['ts'] };
    const result = await queryDenseIndex({ query: 'neat', client, compiledFilter });
    expect(result.results).toHaveLength(1);
    const callArgs = client.execute.mock.calls[0][0];
    expect(callArgs.args).toContain('ts');
  });

  it('validates and compiles metadataFilter when provided', async () => {
    const rows = [
      { chunk_id: 1, file_path: 'a.ts', doc_family: 'docs', heading_path: 'h', body_text: 'b', score: 1.5 },
    ];
    const client = makeMockClient(rows);
    mockValidateFilter.mockReturnValue(true);
    mockCompileFilterToSqlAliased.mockReturnValue({ sql: 'd.lang = ?', params: ['ts'] });
    const result = await queryDenseIndex({ query: 'neat', client, metadataFilter: { type: 'eq', field: 'lang', value: 'ts' } });
    expect(result.results).toHaveLength(1);
    expect(mockValidateFilter).toHaveBeenCalledTimes(1);
    expect(mockCompileFilterToSqlAliased).toHaveBeenCalledTimes(1);
  });

  it('trims and validates family (empty string → null)', async () => {
    const rows = [];
    const client = makeMockClient(rows);
    const result = await queryDenseIndex({ query: 'neat', client, family: '   ' });
    expect(result.family).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Dense mode
// ---------------------------------------------------------------------------

describe('query-dense: dense mode', () => {
  function makeMockEmbedText(withRelease = true) {
    const fn = jest.fn().mockResolvedValue(new Float32Array([0.1, 0.2, 0.3, 0.4]));
    if (withRelease) fn.release = jest.fn().mockResolvedValue(undefined);
    return fn;
  }

  beforeEach(() => {
    mockReadModelMeta.mockResolvedValue({ dimension: 4, model_id: 'test-model', model_sha256: 'abc' });
    mockNormalizeEmbeddingVector.mockImplementation((vec) => new Float32Array([0.1, 0.2, 0.3, 0.4]));
    mockCreateOnnxTextEmbedder.mockResolvedValue(makeMockEmbedText(true));
  });

  it('throws when dimension is invalid (< 1)', async () => {
    mockReadModelMeta.mockResolvedValue({ dimension: 0, model_id: 'test', model_sha256: 'abc' });
    const client = makeMockClient([]);
    await expect(queryDenseIndex({ query: 'neat', client, dense: true })).rejects.toThrow(
      'Embedding dimension is required',
    );
  });

  it('throws when dimension is not an integer', async () => {
    mockReadModelMeta.mockResolvedValue({ dimension: 4.5, model_id: 'test', model_sha256: 'abc' });
    const client = makeMockClient([]);
    await expect(queryDenseIndex({ query: 'neat', client, dense: true })).rejects.toThrow(
      'Embedding dimension is required',
    );
  });

  it('uses provided embedText and does NOT release it (has release method)', async () => {
    const embedText = makeMockEmbedText(true);
    const bm25Rows = [
      { chunk_id: 1, file_path: 'a.ts', doc_family: 'docs', heading_path: '', body_text: '', score: 1 },
    ];
    const denseRows = [
      { chunk_id: 2, file_path: 'b.ts', doc_family: 'docs', heading_path: '', body_text: '', distance: 0.5 },
    ];
    const client = makeMockClient(bm25Rows);
    // First call returns BM25 rows, second call (ANN) returns dense rows
    client.execute
      .mockResolvedValueOnce({ rows: bm25Rows })
      .mockResolvedValueOnce({ rows: denseRows });
    const result = await queryDenseIndex({ query: 'neat', client, dense: true, embedText });
    expect(result.use_dense).toBe(true);
    expect(embedText.release).not.toHaveBeenCalled();
  });

  it('creates embedder via createOnnxTextEmbedder and releases it', async () => {
    const mockEmbed = makeMockEmbedText(true);
    mockCreateOnnxTextEmbedder.mockResolvedValue(mockEmbed);
    const bm25Rows = [
      { chunk_id: 1, file_path: 'a.ts', doc_family: 'docs', heading_path: '', body_text: '', score: 1 },
    ];
    const denseRows = [
      { chunk_id: 2, file_path: 'b.ts', doc_family: 'docs', heading_path: '', body_text: '', distance: 0.5 },
    ];
    const client = makeMockClient([]);
    client.execute
      .mockResolvedValueOnce({ rows: bm25Rows })
      .mockResolvedValueOnce({ rows: denseRows });
    const result = await queryDenseIndex({ query: 'neat', client, dense: true });
    expect(result.use_dense).toBe(true);
    expect(mockCreateOnnxTextEmbedder).toHaveBeenCalledTimes(1);
    expect(mockEmbed.release).toHaveBeenCalledTimes(1);
  });

  it('uses provided embedText without release (does not create new one)', async () => {
    const embedTextNoRelease = jest.fn().mockResolvedValue(new Float32Array([0.1, 0.2, 0.3, 0.4]));
    // No release method on embedText
    const bm25Rows = [
      { chunk_id: 1, file_path: 'a.ts', doc_family: 'docs', heading_path: '', body_text: '', score: 1 },
    ];
    const denseRows = [
      { chunk_id: 2, file_path: 'b.ts', doc_family: 'docs', heading_path: '', body_text: '', distance: 0.5 },
    ];
    const client = makeMockClient([]);
    client.execute
      .mockResolvedValueOnce({ rows: bm25Rows })
      .mockResolvedValueOnce({ rows: denseRows });
    const result = await queryDenseIndex({ query: 'neat', client, dense: true, embedText: embedTextNoRelease });
    expect(result.use_dense).toBe(true);
    expect(mockCreateOnnxTextEmbedder).not.toHaveBeenCalled();
  });

  it('passes family filter to dense query', async () => {
    const embedText = makeMockEmbedText(true);
    const bm25Rows = [];
    const denseRows = [];
    const client = makeMockClient([]);
    client.execute
      .mockResolvedValueOnce({ rows: bm25Rows })
      .mockResolvedValueOnce({ rows: denseRows });
    await queryDenseIndex({ query: 'neat', client, dense: true, family: 'docs', embedText });
    const denseCallArgs = client.execute.mock.calls[1][0];
    expect(denseCallArgs.args).toContain('docs');
  });

  it('passes compiledFilter to dense query', async () => {
    const embedText = makeMockEmbedText(true);
    const bm25Rows = [];
    const denseRows = [];
    const client = makeMockClient([]);
    client.execute
      .mockResolvedValueOnce({ rows: bm25Rows })
      .mockResolvedValueOnce({ rows: denseRows });
    const compiledFilter = { sql: 'd.lang = ?', params: ['ts'] };
    await queryDenseIndex({ query: 'neat', client, dense: true, compiledFilter, embedText });
    const denseCallArgs = client.execute.mock.calls[1][0];
    expect(denseCallArgs.args).toContain('ts');
  });

  it('uses options.dimension when provided', async () => {
    const embedText = makeMockEmbedText(true);
    const bm25Rows = [];
    const denseRows = [];
    const client = makeMockClient([]);
    client.execute
      .mockResolvedValueOnce({ rows: bm25Rows })
      .mockResolvedValueOnce({ rows: denseRows });
    await queryDenseIndex({ query: 'neat', client, dense: true, dimension: 8, embedText });
    expect(mockNormalizeEmbeddingVector).toHaveBeenCalledWith(expect.any(Float32Array), 8);
  });

  it('uses options.modelId when provided', async () => {
    const embedText = makeMockEmbedText(true);
    const bm25Rows = [];
    const denseRows = [];
    const client = makeMockClient([]);
    client.execute
      .mockResolvedValueOnce({ rows: bm25Rows })
      .mockResolvedValueOnce({ rows: denseRows });
    await queryDenseIndex({ query: 'neat', client, dense: true, modelId: 'custom-id', embedText });
    // The modelId should appear in the ANN query args
    const denseCallArgs = client.execute.mock.calls[1][0];
    expect(denseCallArgs.args).toContain('custom-id');
  });
});

// ---------------------------------------------------------------------------
// loadDenseRows — ANN-first with brute-force fallback
// ---------------------------------------------------------------------------

describe('query-dense: ANN-first with brute-force fallback', () => {
  function makeMockEmbedText() {
    const fn = jest.fn().mockResolvedValue(new Float32Array([0.1, 0.2, 0.3, 0.4]));
    fn.release = jest.fn().mockResolvedValue(undefined);
    return fn;
  }

  beforeEach(() => {
    mockReadModelMeta.mockResolvedValue({ dimension: 4, model_id: 'test', model_sha256: 'abc' });
    mockNormalizeEmbeddingVector.mockReturnValue(new Float32Array([0.1, 0.2, 0.3, 0.4]));
  });

  it('uses ANN path when it succeeds (with client)', async () => {
    const embedText = makeMockEmbedText();
    const bm25Rows = [];
    const annRows = [
      { chunk_id: 1, file_path: 'a.ts', doc_family: 'docs', heading_path: '', body_text: '', distance: 0.1 },
    ];
    const client = makeMockClient([]);
    client.execute
      .mockResolvedValueOnce({ rows: bm25Rows })
      .mockResolvedValueOnce({ rows: annRows });
    const result = await queryDenseIndex({ query: 'neat', client, dense: true, embedText });
    expect(client.execute).toHaveBeenCalledTimes(2);
    expect(result.use_dense).toBe(true);
  });

  it('falls back to brute-force when ANN fails (with client)', async () => {
    const embedText = makeMockEmbedText();
    const bm25Rows = [];
    const bfRows = [
      { chunk_id: 2, file_path: 'b.ts', doc_family: 'docs', heading_path: '', body_text: '', distance: 0.2 },
    ];
    const client = makeMockClient([]);
    client.execute
      .mockResolvedValueOnce({ rows: bm25Rows })
      .mockRejectedValueOnce(new Error('ANN index cold'))
      .mockResolvedValueOnce({ rows: bfRows });
    const result = await queryDenseIndex({ query: 'neat', client, dense: true, embedText });
    expect(client.execute).toHaveBeenCalledTimes(3);
    expect(result.use_dense).toBe(true);
  });

  it('uses getTursoClient when no client provided (ANN path)', async () => {
    const embedText = makeMockEmbedText();
    const bm25Rows = [];
    const annRows = [
      { chunk_id: 1, file_path: 'a.ts', doc_family: 'docs', heading_path: '', body_text: '', distance: 0.1 },
    ];
    const dbClient = makeMockClient([]);
    dbClient.execute
      .mockResolvedValueOnce({ rows: bm25Rows })
      .mockResolvedValueOnce({ rows: annRows });
    mockGetTursoClient.mockResolvedValue(dbClient);
    const result = await queryDenseIndex({ query: 'neat', databasePath: '/db', dense: true, embedText });
    expect(mockGetTursoClient).toHaveBeenCalled();
    expect(result.use_dense).toBe(true);
  });

  it('falls back to brute-force via getTursoClient when no client', async () => {
    const embedText = makeMockEmbedText();
    const bm25Rows = [];
    const bfRows = [
      { chunk_id: 2, file_path: 'b.ts', doc_family: 'docs', heading_path: '', body_text: '', distance: 0.2 },
    ];
    const dbClient = makeMockClient([]);
    dbClient.execute
      .mockResolvedValueOnce({ rows: bm25Rows })
      .mockRejectedValueOnce(new Error('ANN fail'))
      .mockResolvedValueOnce({ rows: bfRows });
    mockGetTursoClient.mockResolvedValue(dbClient);
    const result = await queryDenseIndex({ query: 'neat', databasePath: '/db', dense: true, embedText });
    expect(result.use_dense).toBe(true);
  });

  it('passes family to ANN and brute-force paths', async () => {
    const embedText = makeMockEmbedText();
    const bm25Rows = [];
    const annRows = [];
    const client = makeMockClient([]);
    client.execute
      .mockResolvedValueOnce({ rows: bm25Rows })
      .mockResolvedValueOnce({ rows: annRows });
    await queryDenseIndex({ query: 'neat', client, dense: true, family: 'docs', embedText });
    const annArgs = client.execute.mock.calls[1][0];
    expect(annArgs.args).toContain('docs');
  });

  it('passes compiledFilter to ANN path', async () => {
    const embedText = makeMockEmbedText();
    const bm25Rows = [];
    const annRows = [];
    const client = makeMockClient([]);
    client.execute
      .mockResolvedValueOnce({ rows: bm25Rows })
      .mockResolvedValueOnce({ rows: annRows });
    const compiledFilter = { sql: 'd.lang = ?', params: ['ts'] };
    await queryDenseIndex({ query: 'neat', client, dense: true, compiledFilter, embedText });
    const annArgs = client.execute.mock.calls[1][0];
    expect(annArgs.args).toContain('ts');
  });

  it('passes compiledFilter to brute-force fallback', async () => {
    const embedText = makeMockEmbedText();
    const bm25Rows = [];
    const bfRows = [];
    const client = makeMockClient([]);
    client.execute
      .mockResolvedValueOnce({ rows: bm25Rows })
      .mockRejectedValueOnce(new Error('ANN fail'))
      .mockResolvedValueOnce({ rows: bfRows });
    const compiledFilter = { sql: 'd.lang = ?', params: ['ts'] };
    await queryDenseIndex({ query: 'neat', client, dense: true, compiledFilter, embedText });
    const bfArgs = client.execute.mock.calls[2][0];
    expect(bfArgs.args).toContain('ts');
  });

  it('passes family to brute-force fallback', async () => {
    const embedText = makeMockEmbedText();
    const bm25Rows = [];
    const bfRows = [];
    const client = makeMockClient([]);
    client.execute
      .mockResolvedValueOnce({ rows: bm25Rows })
      .mockRejectedValueOnce(new Error('ANN fail'))
      .mockResolvedValueOnce({ rows: bfRows });
    await queryDenseIndex({ query: 'neat', client, dense: true, family: 'docs', embedText });
    const bfArgs = client.execute.mock.calls[2][0];
    expect(bfArgs.args).toContain('docs');
  });
});

// ---------------------------------------------------------------------------
// main() — CLI guard
// ---------------------------------------------------------------------------

describe('query-dense: main()', () => {
  const origArgv1 = process.argv[1];

  afterEach(() => {
    process.argv[1] = origArgv1;
  });

  it('prints help when --help is passed', async () => {
    mockParseCliArgs.mockReturnValue({ help: true });
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./query-dense.mjs');
    });
    expect(mockPrintHelp).toHaveBeenCalledTimes(1);
  });

  it('runs query and writes text output', async () => {
    mockParseCliArgs.mockReturnValue({
      query: 'neat',
      json: false,
      _: [],
    });
    mockGetTursoClient.mockResolvedValue(makeMockClient([
      { chunk_id: 1, file_path: 'a.ts', doc_family: 'docs', heading_path: 'h', body_text: 'b', score: 1 },
    ]));
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./query-dense.mjs');
    });
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
  });

  it('runs query and writes JSON output', async () => {
    mockParseCliArgs.mockReturnValue({
      query: 'neat',
      json: true,
      _: [],
    });
    mockGetTursoClient.mockResolvedValue(makeMockClient([
      { chunk_id: 1, file_path: 'a.ts', doc_family: 'docs', heading_path: 'h', body_text: 'b', score: 1 },
    ]));
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./query-dense.mjs');
    });
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
    // The json flag should be true
    const jsonArg = mockWriteJsonOrText.mock.calls[0][1];
    expect(jsonArg).toBe(true);
  });

  it('uses args._.join(" ") when no query', async () => {
    mockParseCliArgs.mockReturnValue({
      query: undefined,
      _: ['neat', 'activation'],
      json: true,
    });
    mockGetTursoClient.mockResolvedValue(makeMockClient([]));
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./query-dense.mjs');
    });
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
  });

  it('calls fail on Error', async () => {
    mockParseCliArgs.mockReturnValue({
      query: 'neat',
      _: [],
      json: false,
    });
    mockGetTursoClient.mockRejectedValue(new Error('DB connection failed'));
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./query-dense.mjs');
    });
    expect(mockFail).toHaveBeenCalledTimes(1);
    expect(mockFail.mock.calls[0][0]).toBe('DB connection failed');
  });

  it('calls fail with String(error) for non-Error throws', async () => {
    mockParseCliArgs.mockReturnValue({
      query: 'neat',
      _: [],
      json: true,
    });
    mockGetTursoClient.mockRejectedValue('string error');
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./query-dense.mjs');
    });
    expect(mockFail).toHaveBeenCalledTimes(1);
    expect(mockFail.mock.calls[0][0]).toBe('string error');
  });
});