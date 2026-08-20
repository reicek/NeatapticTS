/**
 * @module query-dense.test
 * @description Coverage tests for rag-index/query-dense.mjs — BM25 + hybrid dense query runner.
 */

import { jest } from '@jest/globals';
import path from 'node:path';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { queryDenseIndex } from '../../../rag-index/query-dense.mjs';

let tempDir;

beforeEach(async () => {
  tempDir = await mkdtemp(path.join(tmpdir(), 'query-dense-test-'));
});

afterEach(async () => {
  await rm(tempDir, { recursive: true, force: true });
});

// ---------------------------------------------------------------------------
// Mock client factory
// ---------------------------------------------------------------------------

function createMockClient({ rows = [], failOnAnn = false } = {}) {
  const client = {
    async execute({ sql, args }) {
      if (sql.includes('vector_top_k')) {
        if (failOnAnn) throw new Error('ANN index not available');
        return {
          rows: rows.map((r) => ({
            ...r,
            distance: 0.1,
          })),
        };
      }
      if (
        sql.includes('vector_distance_cos') &&
        sql.includes('ORDER BY distance')
      ) {
        // brute-force path
        return {
          rows: rows.map((r) => ({
            ...r,
            distance: 0.2,
          })),
        };
      }
      // BM25 query
      return { rows };
    },
  };
  return client;
}

// ---------------------------------------------------------------------------
// queryDenseIndex: empty query
// ---------------------------------------------------------------------------

describe('query-dense: empty query', () => {
  it('returns empty results for empty query', async () => {
    const result = await queryDenseIndex({ query: '', limit: 10 });
    expect(result.results).toEqual([]);
    expect(result.use_dense).toBe(false);
  });

  it('returns empty results for whitespace-only query', async () => {
    const result = await queryDenseIndex({ query: '   ', limit: 10 });
    expect(result.results).toEqual([]);
  });

  it('returns empty results with no query property', async () => {
    const result = await queryDenseIndex({ limit: 10 });
    expect(result.results).toEqual([]);
    expect(result.query).toBe('');
  });

  it('includes family and alpha in empty result when dense and family are set', async () => {
    const result = await queryDenseIndex({
      query: '',
      family: 'ts-source',
      dense: true,
      alpha: 0.7,
    });
    expect(result.results).toEqual([]);
    expect(result.family).toBe('ts-source');
    expect(result.use_dense).toBe(true);
    expect(result.alpha).toBe(0.7);
  });
});

// ---------------------------------------------------------------------------
// queryDenseIndex: BM25-only mode (no dense)
// ---------------------------------------------------------------------------

describe('query-dense: BM25-only mode', () => {
  it('returns BM25 results without dense', async () => {
    const mockRows = [
      {
        file_path: 'src/network.ts',
        doc_family: 'ts-source',
        chunk_id: 1,
        chunk_index: 0,
        heading_path: 'activate',
        symbol_name: 'activate',
        body_text: 'activation function',
        char_start: 0,
        char_end: 20,
        score: 5.0,
      },
    ];
    const client = createMockClient({ rows: mockRows });
    const result = await queryDenseIndex({
      query: 'activation function',
      limit: 10,
      client,
      dense: false,
    });
    expect(result.results.length).toBe(1);
    expect(result.use_dense).toBe(false);
    expect(result.results[0].chunk_id).toBe(1);
  });

  it('includes family in result when family option is set', async () => {
    const mockRows = [];
    const client = createMockClient({ rows: mockRows });
    const result = await queryDenseIndex({
      query: 'test query',
      limit: 10,
      family: 'ts-source',
      client,
      dense: false,
    });
    expect(result.family).toBe('ts-source');
  });

  it('does not include family when family option is not set', async () => {
    const mockRows = [];
    const client = createMockClient({ rows: mockRows });
    const result = await queryDenseIndex({
      query: 'test query',
      limit: 10,
      client,
      dense: false,
    });
    expect(result.family).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// queryDenseIndex: dense mode
// ---------------------------------------------------------------------------

describe('query-dense: dense mode', () => {
  it('throws when dimension is invalid and no modelMeta', async () => {
    const mockRows = [
      {
        file_path: 'src/network.ts',
        doc_family: 'ts-source',
        chunk_id: 1,
        chunk_index: 0,
        heading_path: 'activate',
        symbol_name: 'activate',
        body_text: 'activation function',
        char_start: 0,
        char_end: 20,
        score: 5.0,
      },
    ];
    const client = createMockClient({ rows: mockRows });
    await expect(
      queryDenseIndex({
        query: 'activation function',
        limit: 10,
        client,
        dense: true,
        dimension: 0,
        modelMeta: { dimension: 0, model_id: 'test' },
        embedText: async () => new Float32Array([0.1]),
      }),
    ).rejects.toThrow('Embedding dimension is required');
  });

  it('returns dense results with RRF ranking', async () => {
    const mockRows = [
      {
        file_path: 'src/network.ts',
        doc_family: 'ts-source',
        chunk_id: 1,
        chunk_index: 0,
        heading_path: 'activate',
        symbol_name: 'activate',
        body_text: 'activation function',
        char_start: 0,
        char_end: 20,
        score: 5.0,
      },
      {
        file_path: 'src/mutation.ts',
        doc_family: 'ts-source',
        chunk_id: 2,
        chunk_index: 0,
        heading_path: 'mutate',
        symbol_name: 'mutate',
        body_text: 'mutation function',
        char_start: 0,
        char_end: 20,
        score: 3.0,
      },
    ];
    const client = createMockClient({ rows: mockRows });
    const result = await queryDenseIndex({
      query: 'activation function',
      limit: 10,
      client,
      dense: true,
      dimension: 3,
      modelMeta: { dimension: 3, model_id: 'test-model' },
      embedText: async () => new Float32Array([0.1, 0.2, 0.3]),
    });
    expect(result.use_dense).toBe(true);
    expect(result.alpha).toBe(0.5);
    expect(result.results.length).toBeGreaterThan(0);
  });

  it('does not call release when options.embedText has release method', async () => {
    let released = false;
    const mockRows = [];
    const client = createMockClient({ rows: mockRows });
    const mockEmbed = async () => new Float32Array([0.1, 0.2, 0.3]);
    mockEmbed.release = async () => {
      released = true;
    };
    await queryDenseIndex({
      query: 'test',
      limit: 5,
      client,
      dense: true,
      dimension: 3,
      modelMeta: { dimension: 3, model_id: 'test-model' },
      embedText: mockEmbed,
    });
    // Release should NOT be called because options.embedText has release
    expect(released).toBe(false);
  });

  it('falls back to brute-force when ANN fails', async () => {
    const mockRows = [
      {
        file_path: 'src/network.ts',
        doc_family: 'ts-source',
        chunk_id: 1,
        chunk_index: 0,
        heading_path: 'activate',
        symbol_name: 'activate',
        body_text: 'activation function',
        char_start: 0,
        char_end: 20,
        score: 5.0,
      },
    ];
    const client = createMockClient({ rows: mockRows, failOnAnn: true });
    const result = await queryDenseIndex({
      query: 'activation',
      limit: 10,
      client,
      dense: true,
      dimension: 3,
      modelMeta: { dimension: 3, model_id: 'test-model' },
      embedText: async () => new Float32Array([0.1, 0.2, 0.3]),
    });
    expect(result.use_dense).toBe(true);
    expect(result.results.length).toBeGreaterThan(0);
  });

  it('returns dense results with family set', async () => {
    const mockRows = [
      {
        file_path: 'src/network.ts',
        doc_family: 'ts-source',
        chunk_id: 1,
        chunk_index: 0,
        heading_path: 'activate',
        symbol_name: 'activate',
        body_text: 'activation function',
        char_start: 0,
        char_end: 20,
        score: 5.0,
      },
    ];
    const client = createMockClient({ rows: mockRows });
    const result = await queryDenseIndex({
      query: 'activation function',
      limit: 10,
      family: 'ts-source',
      client,
      dense: true,
      dimension: 3,
      modelMeta: { dimension: 3, model_id: 'test-model' },
      embedText: async () => new Float32Array([0.1, 0.2, 0.3]),
    });
    expect(result.use_dense).toBe(true);
    expect(result.family).toBe('ts-source');
  });

  it('uses modelMeta dimension when options.dimension is not provided', async () => {
    const mockRows = [];
    const client = createMockClient({ rows: mockRows });
    const result = await queryDenseIndex({
      query: 'test',
      limit: 5,
      client,
      dense: true,
      modelMeta: { dimension: 3, model_id: 'test-model' },
      embedText: async () => new Float32Array([0.1, 0.2, 0.3]),
    });
    expect(result.use_dense).toBe(true);
  });

  it('uses modelMeta model_id when options.modelId is not provided', async () => {
    const mockRows = [];
    const client = createMockClient({ rows: mockRows });
    const result = await queryDenseIndex({
      query: 'test',
      limit: 5,
      client,
      dense: true,
      dimension: 3,
      modelMeta: { dimension: 3, model_id: 'meta-model-id' },
      embedText: async () => new Float32Array([0.1, 0.2, 0.3]),
    });
    expect(result.use_dense).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// queryDenseIndex: compiled filter
// ---------------------------------------------------------------------------

describe('query-dense: metadata filter', () => {
  it('passes compiledFilter to BM25 query', async () => {
    const mockRows = [];
    const client = createMockClient({ rows: mockRows });
    const result = await queryDenseIndex({
      query: 'test',
      limit: 10,
      client,
      dense: false,
      compiledFilter: { sql: 'd.file_path LIKE ?', params: ['%test%'] },
    });
    expect(result.results).toEqual([]);
  });

  it('compiles metadataFilter when no compiledFilter provided', async () => {
    const mockRows = [];
    const client = createMockClient({ rows: mockRows });
    const result = await queryDenseIndex({
      query: 'test',
      limit: 10,
      client,
      dense: false,
      metadataFilter: { op: 'eq', field: 'family', value: 'ts-source' },
    });
    expect(result.results).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// queryDenseIndex: default parameter and ?? fallback coverage
// ---------------------------------------------------------------------------

describe('query-dense: default parameter and fallback coverage', () => {
  it('works with no arguments at all (uses default options = {})', async () => {
    const result = await queryDenseIndex();
    expect(result.results).toEqual([]);
    expect(result.query).toBe('');
  });

  it('throws when dimension falls back to 0 via modelMeta without dimension', async () => {
    const client = createMockClient({ rows: [] });
    await expect(
      queryDenseIndex({
        query: 'test',
        limit: 5,
        client,
        dense: true,
        modelMeta: { model_id: 'test-model' },
        embedText: async () => new Float32Array([0.1, 0.2, 0.3]),
      }),
    ).rejects.toThrow('Embedding dimension is required');
  });

  it('uses DEFAULT_MODEL_ID when neither options.modelId nor modelMeta.model_id provided', async () => {
    const mockRows = [];
    const client = createMockClient({ rows: mockRows });
    const result = await queryDenseIndex({
      query: 'test',
      limit: 5,
      client,
      dense: true,
      dimension: 3,
      modelMeta: { dimension: 3 },
      embedText: async () => new Float32Array([0.1, 0.2, 0.3]),
    });
    expect(result.use_dense).toBe(true);
  });

  it('passes compiledFilter to dense ANN query', async () => {
    const mockRows = [
      {
        file_path: 'src/network.ts',
        doc_family: 'ts-source',
        chunk_id: 1,
        chunk_index: 0,
        heading_path: 'activate',
        symbol_name: 'activate',
        body_text: 'activation function',
        char_start: 0,
        char_end: 20,
        score: 5.0,
      },
    ];
    const client = createMockClient({ rows: mockRows });
    const result = await queryDenseIndex({
      query: 'activation',
      limit: 10,
      client,
      dense: true,
      dimension: 3,
      modelMeta: { dimension: 3, model_id: 'test-model' },
      embedText: async () => new Float32Array([0.1, 0.2, 0.3]),
      compiledFilter: { sql: 'd.file_path LIKE ?', params: ['%test%'] },
    });
    expect(result.use_dense).toBe(true);
  });

  it('falls back to brute-force with family and compiledFilter when ANN fails', async () => {
    const mockRows = [
      {
        file_path: 'src/network.ts',
        doc_family: 'ts-source',
        chunk_id: 1,
        chunk_index: 0,
        heading_path: 'activate',
        symbol_name: 'activate',
        body_text: 'activation function',
        char_start: 0,
        char_end: 20,
        score: 5.0,
      },
    ];
    const client = createMockClient({ rows: mockRows, failOnAnn: true });
    const result = await queryDenseIndex({
      query: 'activation',
      limit: 10,
      family: 'ts-source',
      client,
      dense: true,
      dimension: 3,
      modelMeta: { dimension: 3, model_id: 'test-model' },
      embedText: async () => new Float32Array([0.1, 0.2, 0.3]),
      compiledFilter: { sql: 'd.file_path LIKE ?', params: ['%test%'] },
    });
    expect(result.use_dense).toBe(true);
  });
});
