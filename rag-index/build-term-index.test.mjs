/**
 * @module build-term-index.test
 * @description 100% coverage tests for build-term-index.mjs (extends existing red tests).
 */

import { jest } from '@jest/globals';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const sourceFilePath = path.join(__dirname, 'build-term-index.mjs');

// Shared mock functions
const mockCreateClient = jest.fn();
const mockFail = jest.fn();
const mockParseCliArgs = jest.fn();
const mockPrintHelp = jest.fn();
const mockWriteJsonOrText = jest.fn();
const mockReadModelMeta = jest.fn();
const mockDefaultDatabasePath = '/fake/default-db.sqlite';

jest.unstable_mockModule('@libsql/client', () => ({
  createClient: mockCreateClient,
}));
jest.unstable_mockModule('./cli-utils.mjs', () => ({
  fail: mockFail,
  parseCliArgs: mockParseCliArgs,
  printHelp: mockPrintHelp,
  writeJsonOrText: mockWriteJsonOrText,
}));
jest.unstable_mockModule('./init-schema.mjs', () => ({
  defaultDatabasePath: mockDefaultDatabasePath,
  repoRoot: '/fake/repo',
  initSemanticIndex: jest.fn(),
}));
jest.unstable_mockModule('./embed-index.mjs', () => ({
  DEFAULT_MODEL_DIRECTORY: '/fake/model',
  DEFAULT_MODEL_ID: 'fake-model-id',
  buildEmbeddingIndex: jest.fn(),
  createOnnxTextEmbedder: jest.fn(),
  normalizeEmbeddingVector: jest.fn(),
  readModelMeta: mockReadModelMeta,
}));
jest.unstable_mockModule('./hybrid-rank.mjs', () => ({
  computeCosineSimilarity: jest.fn(),
  rankRRFResults: jest.fn(),
}));

const {
  buildTermEmbeddings,
  extractQualifyingTerms,
  applyPorterStem,
  DEFAULT_MIN_FREQUENCY,
  DEFAULT_MAX_FREQUENCY_RATIO,
  DEFAULT_MIN_TERM_LENGTH,
} = await import('./build-term-index.mjs');

beforeEach(() => {
  jest.clearAllMocks();
});

// ---------------------------------------------------------------------------
// Helper: create embedding buffer
// ---------------------------------------------------------------------------

function createEmbeddingBuffer(values) {
  const float32 = new Float32Array(values);
  return Buffer.from(float32.buffer, float32.byteOffset, float32.byteLength);
}

function makeMockClient(embeddingRows = []) {
  const execute = jest.fn().mockImplementation((params) => {
    const sql = params.sql || params;
    if (sql.includes('SELECT chunk_id, embedding')) {
      return Promise.resolve({ rows: embeddingRows });
    }
    return Promise.resolve({ rows: [] });
  });
  return { execute, close: jest.fn().mockResolvedValue(undefined) };
}

// ---------------------------------------------------------------------------
// buildTermEmbeddings
// ---------------------------------------------------------------------------

describe('build-term-index: buildTermEmbeddings', () => {
  it('builds term embeddings from chunk embeddings', async () => {
    const embeddingRows = [
      { chunk_id: 1, embedding: createEmbeddingBuffer([1, 0, 0, 0]) },
      { chunk_id: 2, embedding: createEmbeddingBuffer([0, 1, 0, 0]) },
    ];
    const client = makeMockClient(embeddingRows);

    const qualifyingTerms = new Map([
      ['neat', { frequency: 2, doc_family_count: 1, chunkIds: [1, 2] }],
    ]);

    const result = await buildTermEmbeddings(client, qualifyingTerms, {
      modelId: 'test-model',
      modelSha256: 'abc123',
      dimension: 4,
    });

    expect(result.built).toBe(1);
    expect(result.skipped).toBe(0);
    expect(result.purged).toBe(0);
    // Should have: 1 SELECT, 1 DELETE, 1 INSERT
    expect(client.execute).toHaveBeenCalledTimes(3);
  });

  it('skips terms with no available embeddings', async () => {
    const embeddingRows = [
      { chunk_id: 1, embedding: createEmbeddingBuffer([1, 0, 0, 0]) },
    ];
    const client = makeMockClient(embeddingRows);

    const qualifyingTerms = new Map([
      ['neat', { frequency: 1, doc_family_count: 1, chunkIds: [1] }],
      ['missing', { frequency: 1, doc_family_count: 1, chunkIds: [999] }],
    ]);

    const result = await buildTermEmbeddings(client, qualifyingTerms, {
      modelId: 'test-model',
      modelSha256: 'abc123',
      dimension: 4,
    });

    expect(result.built).toBe(1);
  });

  it('skips all terms when no chunk embeddings are available', async () => {
    const embeddingRows = [];
    const client = makeMockClient(embeddingRows);

    const qualifyingTerms = new Map([
      ['neat', { frequency: 1, doc_family_count: 1, chunkIds: [1] }],
    ]);

    const result = await buildTermEmbeddings(client, qualifyingTerms, {
      modelId: 'test-model',
      modelSha256: 'abc123',
      dimension: 4,
    });

    expect(result.built).toBe(0);
  });

  it('handles zero-magnitude embeddings (normalizeL2 zero branch)', async () => {
    const embeddingRows = [
      { chunk_id: 1, embedding: createEmbeddingBuffer([0, 0, 0, 0]) },
    ];
    const client = makeMockClient(embeddingRows);

    const qualifyingTerms = new Map([
      ['zero', { frequency: 1, doc_family_count: 1, chunkIds: [1] }],
    ]);

    const result = await buildTermEmbeddings(client, qualifyingTerms, {
      modelId: 'test-model',
      modelSha256: 'abc123',
      dimension: 4,
    });

    expect(result.built).toBe(1);
  });

  it('handles empty qualifyingTerms map', async () => {
    const embeddingRows = [
      { chunk_id: 1, embedding: createEmbeddingBuffer([1, 0, 0, 0]) },
    ];
    const client = makeMockClient(embeddingRows);

    const result = await buildTermEmbeddings(client, new Map(), {
      modelId: 'test-model',
      modelSha256: 'abc123',
      dimension: 4,
    });

    expect(result.built).toBe(0);
  });

  it('purges existing term embeddings before inserting', async () => {
    const embeddingRows = [
      { chunk_id: 1, embedding: createEmbeddingBuffer([1, 0, 0, 0]) },
    ];
    const client = makeMockClient(embeddingRows);

    const qualifyingTerms = new Map([
      ['neat', { frequency: 1, doc_family_count: 1, chunkIds: [1] }],
    ]);

    await buildTermEmbeddings(client, qualifyingTerms, {
      modelId: 'test-model',
      modelSha256: 'abc123',
      dimension: 4,
    });

    const deleteCall = client.execute.mock.calls.find(
      (call) => call[0].sql && call[0].sql.includes('DELETE FROM term_embeddings'),
    );
    expect(deleteCall).toBeDefined();
    expect(deleteCall[0].args).toEqual(['test-model']);
  });

  it('mean-pools multiple chunk embeddings correctly', async () => {
    const embeddingRows = [
      { chunk_id: 1, embedding: createEmbeddingBuffer([2, 0, 0, 0]) },
      { chunk_id: 2, embedding: createEmbeddingBuffer([0, 2, 0, 0]) },
    ];
    const client = makeMockClient(embeddingRows);

    const qualifyingTerms = new Map([
      ['neat', { frequency: 2, doc_family_count: 1, chunkIds: [1, 2] }],
    ]);

    await buildTermEmbeddings(client, qualifyingTerms, {
      modelId: 'test-model',
      modelSha256: 'abc123',
      dimension: 4,
    });

    const insertCall = client.execute.mock.calls.find(
      (call) => call[0].sql && call[0].sql.includes('INSERT INTO term_embeddings'),
    );
    expect(insertCall).toBeDefined();
    // The embedding buffer should be L2-normalized mean of [2,0,0,0] and [0,2,0,0]
    // Mean = [1,1,0,0], L2-normalized = [0.707, 0.707, 0, 0]
    const insertedBuffer = insertCall[0].args[1];
    const float32 = new Float32Array(
      insertedBuffer.buffer,
      insertedBuffer.byteOffset,
      insertedBuffer.byteLength / 4,
    );
    expect(float32[0]).toBeCloseTo(0.7071, 3);
    expect(float32[1]).toBeCloseTo(0.7071, 3);
    expect(float32[2]).toBeCloseTo(0, 5);
    expect(float32[3]).toBeCloseTo(0, 5);
  });
});

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

describe('build-term-index: constants', () => {
  it('exports DEFAULT_MIN_FREQUENCY as 5', () => {
    expect(DEFAULT_MIN_FREQUENCY).toBe(5);
  });
  it('exports DEFAULT_MAX_FREQUENCY_RATIO as 0.3', () => {
    expect(DEFAULT_MAX_FREQUENCY_RATIO).toBe(0.3);
  });
  it('exports DEFAULT_MIN_TERM_LENGTH as 3', () => {
    expect(DEFAULT_MIN_TERM_LENGTH).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// main() — CLI guard
// ---------------------------------------------------------------------------

describe('build-term-index: main()', () => {
  const origArgv1 = process.argv[1];

  afterEach(() => {
    process.argv[1] = origArgv1;
  });

  it('prints help when --help is passed', async () => {
    mockParseCliArgs.mockReturnValue({ help: true });
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./build-term-index.mjs');
    });
    expect(mockPrintHelp).toHaveBeenCalledTimes(1);
  });

  it('calls fail when dimension and sha256 are missing', async () => {
    mockParseCliArgs.mockReturnValue({
      database: '/fake/db',
      _: [],
    });
    mockReadModelMeta.mockResolvedValue({ dimension: 0, model_sha256: '' });
    mockCreateClient.mockReturnValue(makeMockClient([]));
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./build-term-index.mjs');
    });
    expect(mockFail).toHaveBeenCalledTimes(1);
  });

  it('runs dry-run mode and writes summary', async () => {
    mockParseCliArgs.mockReturnValue({
      'dry-run': true,
      database: '/fake/db',
      json: false,
      _: [],
    });
    mockReadModelMeta.mockResolvedValue({ dimension: 4, model_sha256: 'abc' });
    const mockClient = makeMockClient([]);
    mockClient.execute.mockImplementation((params) => {
      const sql = params.sql || params;
      if (sql.includes('SELECT c.chunk_id')) {
        return Promise.resolve({ rows: [] });
      }
      return Promise.resolve({ rows: [] });
    });
    mockCreateClient.mockReturnValue(mockClient);
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./build-term-index.mjs');
    });
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
    expect(mockClient.close).toHaveBeenCalledTimes(1);
    const payload = mockWriteJsonOrText.mock.calls[0][0];
    expect(payload.dryRun).toBe(true);
  });

  it('runs non-dry-run mode and writes summary', async () => {
    mockParseCliArgs.mockReturnValue({
      database: '/fake/db',
      json: true,
      _: [],
    });
    mockReadModelMeta.mockResolvedValue({ dimension: 4, model_sha256: 'abc' });
    const embeddingRows = [
      { chunk_id: 1, embedding: createEmbeddingBuffer([1, 0, 0, 0]) },
    ];
    const mockClient = makeMockClient(embeddingRows);
    mockClient.execute.mockImplementation((params) => {
      const sql = params.sql || params;
      if (sql.includes('SELECT c.chunk_id')) {
        return Promise.resolve({
          rows: [
            { chunk_id: 1, heading_path: 'neat', body_text: 'activation', doc_family: 'docs' },
          ],
        });
      }
      if (sql.includes('SELECT chunk_id, embedding')) {
        return Promise.resolve({ rows: embeddingRows });
      }
      return Promise.resolve({ rows: [] });
    });
    mockCreateClient.mockReturnValue(mockClient);
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./build-term-index.mjs');
    });
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
    expect(mockClient.close).toHaveBeenCalledTimes(1);
    const payload = mockWriteJsonOrText.mock.calls[0][0];
    expect(payload.dryRun).toBe(false);
  });
});

describe('applyPorterStem', () => {
  it('returns stem without vowel when word ends with ed', () => {
    // "flyed" → stem "fly" has no [aeiou] → returns stem (line 203)
    expect(applyPorterStem('flyed')).toBe('fly');
  });

  it('returns stem without vowel when word ends with ing', () => {
    // "bbbing" → stem "bbb" has no [aeiou] → returns stem (line 211)
    expect(applyPorterStem('bbbing')).toBe('bbb');
  });
});