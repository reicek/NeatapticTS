/**
 * @module eval-runner.test
 * @description Comprehensive tests for eval-runner.mjs targeting 100% coverage.
 */
import { jest } from '@jest/globals';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

const mockReadFile = jest.fn();
const mockWriteFile = jest.fn();
const mockReadModelMeta = jest.fn();
const mockCreateOnnxTextEmbedder = jest.fn();
const mockNormalizeEmbeddingVector = jest.fn();
const mockGetTursoClient = jest.fn();

jest.unstable_mockModule('node:fs/promises', () => ({
  ...jest.requireActual('node:fs/promises'),
  readFile: mockReadFile,
  writeFile: mockWriteFile,
}));
jest.unstable_mockModule('./embed-index.mjs', () => ({
  DEFAULT_MODEL_DIRECTORY: '/mock/model',
  DEFAULT_MODEL_ID: 'mock-model',
  createOnnxTextEmbedder: mockCreateOnnxTextEmbedder,
  readModelMeta: mockReadModelMeta,
  normalizeEmbeddingVector: mockNormalizeEmbeddingVector,
}));
jest.unstable_mockModule('../scripts/mcp-semantic/tools/cortex-db.mjs', () => ({
  getTursoClient: mockGetTursoClient,
}));

// Import real init-schema for repoRoot
const { repoRoot } = await import('./init-schema.mjs');

// Set up search module mocks at computed paths
const searchCorpusPath = path.join(
  repoRoot, 'scripts', 'mcp-semantic', 'tools', 'search-corpus.mjs',
);
const searchCorpusUrl = pathToFileURL(searchCorpusPath).href;
const searchAdvancedPath = path.join(
  repoRoot, 'scripts', 'mcp-semantic', 'tools', 'search-advanced.mjs',
);
const searchAdvancedUrl = pathToFileURL(searchAdvancedPath).href;

const mockSearchCorpus = jest.fn();
const mockSearchAdvanced = jest.fn();

jest.unstable_mockModule(searchCorpusUrl, () => ({ searchCorpus: mockSearchCorpus }));
jest.unstable_mockModule(searchAdvancedUrl, () => ({ searchAdvanced: mockSearchAdvanced }));

// Now import eval-runner with all mocks in place
const {
  validateQuerySchema,
  resolveSearchFn,
  buildConditionOptions,
  runEval,
  runSelfTest,
  runAllConditions,
  runCli,
  bootstrap,
  computeVectorRecall,
  runRecallBenchmark,
  DEFAULT_QUERY_FILE_PATH,
} = await import('./eval-runner.mjs');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeValidQuery(overrides = {}) {
  return {
    query_id: 'q1',
    query: 'network activate',
    class: 'simple_lookup',
    difficulty: 'easy',
    expected_doc_families: ['ts-source'],
    expected_heading_contains: 'activate',
    expected_symbol_contains: null,
    expected_chunk_ids: [],
    relevance_grades: [],
    ...overrides,
  };
}

function makeTestQueries() {
  return [
    makeValidQuery({ query_id: 'q1', query: 'Network activate', class: 'simple_lookup', expected_heading_contains: 'activate' }),
    makeValidQuery({ query_id: 'q2', query: 'NEAT crossover', class: 'simple_lookup', expected_heading_contains: 'crossover', expected_doc_families: ['readme'] }),
    makeValidQuery({ query_id: 'q3', query: 'genome speciation', class: 'cross_boundary', difficulty: 'medium', expected_heading_contains: 'speciation', expected_doc_families: ['readme', 'ts-source'] }),
    makeValidQuery({ query_id: 'q4', query: 'population fitness', class: 'code_specific', expected_heading_contains: 'fitness', expected_doc_families: ['ts-source'] }),
    makeValidQuery({ query_id: 'q5', query: 'neuron mutate', class: 'plan_specific', expected_heading_contains: 'mutate', expected_doc_families: ['ts-source'] }),
  ];
}

// ---------------------------------------------------------------------------

describe('eval-runner', () => {
  let origEvalForce;

  beforeAll(() => {
    origEvalForce = process.env.EVAL_FORCE_SYNTHETIC;
    process.env.EVAL_FORCE_SYNTHETIC = '1';
  });

  afterAll(() => {
    if (origEvalForce !== undefined) process.env.EVAL_FORCE_SYNTHETIC = origEvalForce;
    else delete process.env.EVAL_FORCE_SYNTHETIC;
  });

  beforeEach(() => {
    jest.clearAllMocks();
    mockReadFile.mockImplementation((filePath) => {
      const fp = String(filePath);
      if (fp.includes('eval-queries')) {
        return Promise.resolve(JSON.stringify(makeTestQueries()));
      }
      return Promise.resolve('');
    });
    mockReadModelMeta.mockResolvedValue({ model_id: 'mock-model', dimension: 10 });
    mockCreateOnnxTextEmbedder.mockResolvedValue(async () => new Float32Array([1, 2, 3]));
    mockNormalizeEmbeddingVector.mockReturnValue(new Float32Array([1, 2, 3]));
  });

  // -------------------------------------------------------------------------
  // validateQuerySchema
  // -------------------------------------------------------------------------

  describe('validateQuerySchema', () => {
    it('returns query for valid input', () => {
      const q = makeValidQuery();
      expect(validateQuerySchema(q)).toBe(q);
    });

    it('throws for null query', () => {
      expect(() => validateQuerySchema(null)).toThrow('Query must be an object');
    });

    it('throws for non-object query', () => {
      expect(() => validateQuerySchema('string')).toThrow('Query must be an object');
    });

    it('throws for non-string query_id', () => {
      expect(() => validateQuerySchema(makeValidQuery({ query_id: 123 })))
        .toThrow('query_id must be a non-empty string');
    });

    it('throws for empty query_id', () => {
      expect(() => validateQuerySchema(makeValidQuery({ query_id: '' })))
        .toThrow('query_id must be a non-empty string');
    });

    it('throws for non-string query', () => {
      expect(() => validateQuerySchema(makeValidQuery({ query: 123 })))
        .toThrow('query text must be a non-empty string');
    });

    it('throws for empty query', () => {
      expect(() => validateQuerySchema(makeValidQuery({ query: '' })))
        .toThrow('query text must be a non-empty string');
    });

    it('throws for invalid class', () => {
      expect(() => validateQuerySchema(makeValidQuery({ class: 'invalid' })))
        .toThrow('class must be one of');
    });

    it('throws for invalid difficulty', () => {
      expect(() => validateQuerySchema(makeValidQuery({ difficulty: 'impossible' })))
        .toThrow('difficulty must be easy, medium, or hard');
    });

    it('throws for non-array expected_doc_families', () => {
      expect(() => validateQuerySchema(makeValidQuery({ expected_doc_families: 'ts-source' })))
        .toThrow('expected_doc_families must be an array');
    });

    it('accepts null relevance_grades', () => {
      expect(() => validateQuerySchema(makeValidQuery({ relevance_grades: null })))
        .not.toThrow();
    });

    it('throws for null grade item in relevance_grades', () => {
      expect(() => validateQuerySchema(makeValidQuery({ relevance_grades: [null] })))
        .toThrow('Each relevance_grade must be an object');
    });

    it('throws for non-integer grade', () => {
      expect(() => validateQuerySchema(makeValidQuery({ relevance_grades: [{ grade: 1.5 }] })))
        .toThrow('relevance grade must be an integer in [0, 3]');
    });

    it('throws for grade below 0', () => {
      expect(() => validateQuerySchema(makeValidQuery({ relevance_grades: [{ grade: -1 }] })))
        .toThrow('relevance grade must be an integer in [0, 3]');
    });

    it('throws for grade above 3', () => {
      expect(() => validateQuerySchema(makeValidQuery({ relevance_grades: [{ grade: 4 }] })))
        .toThrow('relevance grade must be an integer in [0, 3]');
    });

    it('accepts valid grade 0-3', () => {
      expect(() => validateQuerySchema(makeValidQuery({ relevance_grades: [{ grade: 0 }, { grade: 3 }] })))
        .not.toThrow();
    });

    it('throws for non-array expected_chunk_ids when present', () => {
      expect(() => validateQuerySchema(makeValidQuery({ expected_chunk_ids: 'abc' })))
        .toThrow('expected_chunk_ids must be an array when present');
    });

    it('accepts undefined expected_chunk_ids', () => {
      const q = makeValidQuery();
      delete q.expected_chunk_ids;
      expect(() => validateQuerySchema(q)).not.toThrow();
    });
  });

  // -------------------------------------------------------------------------
  // buildConditionOptions
  // -------------------------------------------------------------------------

  describe('buildConditionOptions', () => {
    it('returns bm25_only options', () => {
      const opts = buildConditionOptions('bm25_only', {}, {});
      expect(opts).toEqual({ useDense: false, useRerank: false, alpha: 1, limit: 10 });
    });

    it('returns hybrid options', () => {
      const opts = buildConditionOptions('hybrid', {}, { alpha: 0.7 });
      expect(opts).toEqual({ useDense: true, useRerank: false, alpha: 0.7, limit: 10 });
    });

    it('returns hybrid_rerank options', () => {
      const opts = buildConditionOptions('hybrid_rerank', {}, { limit: 5 });
      expect(opts).toEqual({ useDense: true, useRerank: true, alpha: 0.5, limit: 5 });
    });

    it('returns advanced_default options with isCodeSpecific=false', () => {
      const opts = buildConditionOptions('advanced_default', { class: 'simple_lookup' }, {});
      expect(opts.useDense).toBe(true);
      expect(opts.useRerank).toBe(true);
      expect(opts.compact).toBe(true);
      expect(opts.read_top_result).toBe(true);
      expect(opts.auto_fallback).toBe(true);
      expect(opts.include_code_only).toBe(false);
      expect(opts.contextBudget).toBe(4096);
    });

    it('returns advanced_default options with isCodeSpecific=true', () => {
      const opts = buildConditionOptions('advanced_default', { class: 'code_specific' }, {});
      expect(opts.include_code_only).toBe(true);
    });

    it('throws for unsupported condition', () => {
      expect(() => buildConditionOptions('invalid', {}, {}))
        .toThrow('Unsupported condition: invalid');
    });

    it('uses default options when options is null', () => {
      const opts = buildConditionOptions('hybrid', {}, null);
      expect(opts.alpha).toBe(0.5);
      expect(opts.limit).toBe(10);
    });
  });

  // -------------------------------------------------------------------------
  // resolveSearchFn
  // -------------------------------------------------------------------------

  describe('resolveSearchFn', () => {
    it('returns provided searchFn', async () => {
      const fn = async () => [];
      const result = await resolveSearchFn('hybrid', { searchFn: fn });
      expect(result).toBe(fn);
    });

    it('returns syntheticSearch when EVAL_FORCE_SYNTHETIC=1', async () => {
      const result = await resolveSearchFn('hybrid', {});
      const searchResults = await result({ expected_doc_families: ['ts-source'] }, {});
      expect(Array.isArray(searchResults)).toBe(true);
      expect(searchResults[0].family).toBe('ts-source');
    });

    it('returns syntheticSearch fallback for code_specific class', async () => {
      const result = await resolveSearchFn('hybrid', {});
      const searchResults = await result({ class: 'code_specific' }, {});
      expect(searchResults[0].family).toBe('ts-source');
    });

    it('returns syntheticSearch fallback for non-code-specific class', async () => {
      const result = await resolveSearchFn('hybrid', {});
      const searchResults = await result({ class: 'simple_lookup' }, {});
      expect(searchResults[0].family).toBe('readme');
    });

    it('imports real search module for non-advanced condition', async () => {
      const origForce = process.env.EVAL_FORCE_SYNTHETIC;
      delete process.env.EVAL_FORCE_SYNTHETIC;
      mockSearchCorpus.mockResolvedValue([{ chunk_id: 1, family: 'ts-source', heading_path: 'test' }]);

      try {
        const searchFn = await resolveSearchFn('hybrid', {});
        const results = await searchFn({ query: 'test' }, { limit: 5, useDense: false, useRerank: false, alpha: 0.5 });
        expect(mockSearchCorpus).toHaveBeenCalled();
        expect(results).toHaveLength(1);
      } finally {
        if (origForce !== undefined) process.env.EVAL_FORCE_SYNTHETIC = origForce;
      }
    });

    it('imports real search-advanced module for advanced_default condition', async () => {
      const origForce = process.env.EVAL_FORCE_SYNTHETIC;
      delete process.env.EVAL_FORCE_SYNTHETIC;
      mockSearchAdvanced.mockResolvedValue([{ chunk_id: 1, family: 'ts-source', heading_path: 'test' }]);

      try {
        const searchFn = await resolveSearchFn('advanced_default', {});
        const results = await searchFn(
          { query: 'test', class: 'simple_lookup' },
          { limit: 5, contextBudget: 4096 },
        );
        expect(mockSearchAdvanced).toHaveBeenCalled();
        expect(results).toHaveLength(1);
      } finally {
        if (origForce !== undefined) process.env.EVAL_FORCE_SYNTHETIC = origForce;
      }
    });

    it('falls back to syntheticSearch when import fails', async () => {
      const origForce = process.env.EVAL_FORCE_SYNTHETIC;
      delete process.env.EVAL_FORCE_SYNTHETIC;

      try {
        const searchFn = await resolveSearchFn('hybrid', {
          searchModulePath: 'C:\\nonexistent\\path\\search.mjs',
        });
        const results = await searchFn({ expected_doc_families: ['ts-source'] }, {});
        expect(Array.isArray(results)).toBe(true);
      } finally {
        if (origForce !== undefined) process.env.EVAL_FORCE_SYNTHETIC = origForce;
      }
    });
  });

  // -------------------------------------------------------------------------
  // runEval
  // -------------------------------------------------------------------------

  describe('runEval', () => {
    it('throws for non-object options', async () => {
      await expect(runEval(null)).rejects.toThrow('runEval requires an options object');
    });

    it('throws for invalid condition', async () => {
      await expect(runEval({ queries: [], condition: 'invalid' }))
        .rejects.toThrow('condition must be one of');
    });

    it('throws for undefined condition', async () => {
      await expect(runEval({ queries: [] }))
        .rejects.toThrow('condition must be one of');
    });

    it('runs eval with searchFn returning array', async () => {
      const result = await runEval({
        queries: [makeValidQuery()],
        condition: 'hybrid',
        searchFn: async () => [{ chunk_id: 1, family: 'ts-source', heading_path: 'activate' }],
      });
      expect(result.condition).toBe('hybrid');
      expect(result.query_count).toBe(1);
      expect(result.metrics).toBeDefined();
      expect(result.per_query).toHaveLength(1);
    });

    it('runs eval with searchFn returning { results }', async () => {
      const result = await runEval({
        queries: [makeValidQuery()],
        condition: 'hybrid',
        searchFn: async () => ({ results: [{ chunk_id: 1, family: 'ts-source', heading_path: 'activate' }] }),
      });
      expect(result.query_count).toBe(1);
    });

    it('runs eval with searchFn returning { context: { results, selectedChunks } }', async () => {
      const result = await runEval({
        queries: [makeValidQuery()],
        condition: 'hybrid',
        searchFn: async () => ({
          context: {
            results: [{ chunk_id: 1, family: 'ts-source', heading_path: 'activate' }],
            selectedChunks: [{ chunk_id: 1, family: 'ts-source', heading_path: 'activate' }],
          },
        }),
      });
      expect(result.query_count).toBe(1);
      expect(result.per_query[0].context_relevance).toBeDefined();
    });

    it('runs eval with searchFn returning { results, selectedChunks }', async () => {
      const result = await runEval({
        queries: [makeValidQuery()],
        condition: 'hybrid',
        searchFn: async () => ({
          results: [{ chunk_id: 1, family: 'ts-source', heading_path: 'activate' }],
          selectedChunks: [{ chunk_id: 1, family: 'ts-source', heading_path: 'activate' }],
        }),
      });
      expect(result.query_count).toBe(1);
    });

    it('runs eval with searchFn returning empty object', async () => {
      const result = await runEval({
        queries: [makeValidQuery()],
        condition: 'hybrid',
        searchFn: async () => ({}),
      });
      expect(result.query_count).toBe(1);
      expect(result.metrics.zero_hit_queries).toBe(1);
    });

    it('runs eval for all conditions', async () => {
      for (const condition of ['bm25_only', 'hybrid', 'hybrid_rerank', 'advanced_default']) {
        const result = await runEval({
          queries: [makeValidQuery()],
          condition,
          searchFn: async () => [{ chunk_id: 1, family: 'ts-source', heading_path: 'activate' }],
        });
        expect(result.condition).toBe(condition);
        expect(result.premium_defaults_applied).toBe(condition === 'advanced_default');
      }
    });

    it('sorts queries by class then query_id', async () => {
      const queries = [
        makeValidQuery({ query_id: 'z-query', class: 'multi_hop' }),
        makeValidQuery({ query_id: 'a-query', class: 'simple_lookup' }),
        makeValidQuery({ query_id: 'b-query', class: 'simple_lookup' }),
      ];
      const result = await runEval({
        queries,
        condition: 'hybrid',
        searchFn: async () => [],
      });
      expect(result.per_query[0].query_id).toBe('a-query');
      expect(result.per_query[1].query_id).toBe('b-query');
      expect(result.per_query[2].query_id).toBe('z-query');
    });
  });

  // -------------------------------------------------------------------------
  // runSelfTest
  // -------------------------------------------------------------------------

  describe('runSelfTest', () => {
    it('runs self-test for all conditions', async () => {
      const result = await runSelfTest();
      expect(result.query_count).toBe(5);
      expect(result.conditions).toEqual(['bm25_only', 'hybrid', 'hybrid_rerank', 'advanced_default']);
      expect(result.results).toHaveLength(4);
      expect(typeof result.pass).toBe('boolean');
    });
  });

  // -------------------------------------------------------------------------
  // runAllConditions
  // -------------------------------------------------------------------------

  describe('runAllConditions', () => {
    it('runs all conditions with provided queries', async () => {
      const result = await runAllConditions({
        queries: [makeValidQuery()],
        searchFn: async () => [{ chunk_id: 1, family: 'ts-source', heading_path: 'activate' }],
      });
      expect(Object.keys(result)).toEqual(
        expect.arrayContaining(['bm25_only', 'hybrid', 'hybrid_rerank', 'advanced_default']),
      );
    });

    it('loads queries from file when not provided', async () => {
      const result = await runAllConditions({
        searchFn: async () => [{ chunk_id: 1, family: 'ts-source', heading_path: 'activate' }],
      });
      expect(Object.keys(result)).toHaveLength(4);
      expect(mockReadFile).toHaveBeenCalled();
    });

    it('throws when query file contains non-array', async () => {
      mockReadFile.mockResolvedValue(JSON.stringify({ not: 'array' }));
      await expect(runAllConditions({
        searchFn: async () => [],
      })).rejects.toThrow('Query file must contain an array');
    });
  });

  // -------------------------------------------------------------------------
  // computeVectorRecall (supplemental coverage)
  // -------------------------------------------------------------------------

  describe('computeVectorRecall', () => {
    it('returns 0 for empty ANN set', () => {
      expect(computeVectorRecall([], [1, 2, 3], 3)).toBe(0);
    });

    it('returns 0 for empty brute-force set', () => {
      expect(computeVectorRecall([1, 2, 3], [], 3)).toBe(0);
    });

    it('computes partial recall', () => {
      expect(computeVectorRecall([1, 2, 3, 4, 5], [1, 2, 3, 4, 6], 5)).toBe(0.8);
    });

    it('computes full recall', () => {
      expect(computeVectorRecall([1, 2, 3], [1, 2, 3], 3)).toBe(1);
    });
  });

  // -------------------------------------------------------------------------
  // runRecallBenchmark
  // -------------------------------------------------------------------------

  describe('runRecallBenchmark', () => {
    function makeMockClient(annThrows = false) {
      return {
        execute: jest.fn().mockImplementation((params) => {
          const sql = typeof params === 'string' ? params : params.sql;
          if (sql.includes('vector_top_k')) {
            if (annThrows) throw new Error('no such table');
            return Promise.resolve({ rows: [{ chunk_id: 1 }, { chunk_id: 2 }] });
          }
          if (sql.includes('vector_distance_cos')) {
            return Promise.resolve({ rows: [{ chunk_id: 1 }, { chunk_id: 2 }] });
          }
          return Promise.resolve({ rows: [] });
        }),
      };
    }

    it('runs benchmark with provided client and embedText', async () => {
      const result = await runRecallBenchmark({
        queries: [{ query: 'test query' }],
        client: makeMockClient(false),
        embedText: async () => new Float32Array([1, 2, 3]),
      });
      expect(result.benchmark).toBe('recall-benchmark');
      expect(result.query_count).toBe(1);
      expect(result.ann_available).toBe(true);
      expect(result.pass).toBe(true);
    });

    it('handles ANN unavailable', async () => {
      const result = await runRecallBenchmark({
        queries: [{ query: 'test query' }],
        client: makeMockClient(true),
        embedText: async () => new Float32Array([1, 2, 3]),
      });
      expect(result.ann_available).toBe(false);
      expect(result.criteria.recall_pass).toBeNull();
    });

    it('skips empty query text', async () => {
      const result = await runRecallBenchmark({
        queries: [{ query: '   ' }, { query: 'test' }],
        client: makeMockClient(false),
        embedText: async () => new Float32Array([1, 2, 3]),
      });
      expect(result.query_count).toBe(1);
    });

    it('handles empty queries (div by zero guard)', async () => {
      const result = await runRecallBenchmark({
        queries: [],
        client: makeMockClient(false),
        embedText: async () => new Float32Array([1, 2, 3]),
      });
      expect(result.query_count).toBe(0);
      expect(result.recall_at_k).toBe(0);
      expect(result.ann_latency_ms).toBe(0);
    });

    it('creates embedText via createOnnxTextEmbedder when not provided', async () => {
      const mockEmbedFn = async () => new Float32Array([1, 2, 3]);
      mockEmbedFn.release = jest.fn().mockResolvedValue(undefined);
      mockCreateOnnxTextEmbedder.mockResolvedValue(mockEmbedFn);

      await runRecallBenchmark({
        queries: [{ query: 'test' }],
        client: makeMockClient(false),
      });

      expect(mockCreateOnnxTextEmbedder).toHaveBeenCalled();
      expect(mockEmbedFn.release).toHaveBeenCalled();
    });

    it('does not call release when embedText provided with release', async () => {
      const releaseFn = jest.fn().mockResolvedValue(undefined);
      const mockEmbedFn = async () => new Float32Array([1, 2, 3]);
      mockEmbedFn.release = releaseFn;

      await runRecallBenchmark({
        queries: [{ query: 'test' }],
        client: makeMockClient(false),
        embedText: mockEmbedFn,
      });

      expect(releaseFn).not.toHaveBeenCalled();
    });

    it('uses getTursoClient when no client provided', async () => {
      mockGetTursoClient.mockResolvedValue(makeMockClient(false));

      const result = await runRecallBenchmark({
        queries: [{ query: 'test' }],
        embedText: async () => new Float32Array([1, 2, 3]),
        databasePath: 'C:\\mock\\db.sqlite',
      });

      expect(mockGetTursoClient).toHaveBeenCalledWith('C:\\mock\\db.sqlite');
      expect(result.query_count).toBe(1);
    });

    it('loads queries from file when not provided', async () => {
      mockGetTursoClient.mockResolvedValue(makeMockClient(false));

      const result = await runRecallBenchmark({
        embedText: async () => new Float32Array([1, 2, 3]),
      });

      expect(result.query_count).toBeGreaterThan(0);
      expect(mockReadFile).toHaveBeenCalled();
    });

    it('respects custom k, minRecall, maxLatency, maxHeapDelta', async () => {
      const result = await runRecallBenchmark({
        queries: [{ query: 'test' }],
        client: makeMockClient(false),
        embedText: async () => new Float32Array([1, 2, 3]),
        k: 5,
        minRecall: 0.5,
        maxLatencyMs: 100,
        maxHeapDeltaMb: 100,
      });
      expect(result.k).toBe(5);
      expect(result.thresholds.min_recall).toBe(0.5);
      expect(result.thresholds.max_latency_ms).toBe(100);
      expect(result.thresholds.max_heap_delta_mb).toBe(100);
    });

    it('fails when latency exceeds threshold', async () => {
      const result = await runRecallBenchmark({
        queries: [{ query: 'test' }],
        client: makeMockClient(false),
        embedText: async () => new Float32Array([1, 2, 3]),
        maxLatencyMs: 0,
      });
      expect(result.criteria.latency_pass).toBe(false);
      expect(result.pass).toBe(false);
    });
  });

  // -------------------------------------------------------------------------
  // runCli
  // -------------------------------------------------------------------------

  describe('runCli', () => {
    let logSpy;
    let stderrSpy;
    let origArgv;
    let origExitCode;

    beforeEach(() => {
      logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      stderrSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
      origArgv = process.argv;
      origExitCode = process.exitCode;
      mockReadFile.mockImplementation((filePath) => {
        const fp = String(filePath);
        if (fp.includes('eval-queries')) {
          return Promise.resolve(JSON.stringify(makeTestQueries()));
        }
        if (fp.includes('baseline')) {
          return Promise.resolve(JSON.stringify({
            conditions: {
              bm25_only: { mrr_at_5: 0.0, ndcg_at_5: 0.0, recall_at_5: 0.0, latency_ms: { p50: 0, p95: 0, avg: 0 } },
            },
          }));
        }
        return Promise.resolve('');
      });
    });

    afterEach(() => {
      logSpy.mockRestore();
      stderrSpy.mockRestore();
      process.argv = origArgv;
      process.exitCode = origExitCode;
    });

    it('shows help with --help', async () => {
      await runCli(['--help']);
      expect(logSpy).toHaveBeenCalled();
    });

    it('runs normal eval in text mode', async () => {
      await runCli(['--condition', 'bm25_only']);
      expect(logSpy).toHaveBeenCalled();
    });

    it('runs normal eval in JSON mode', async () => {
      await runCli(['--condition', 'bm25_only', '--json']);
      const jsonOut = logSpy.mock.calls.find((c) => {
        try { JSON.parse(c[0]); return true; } catch { return false; }
      });
      expect(jsonOut).toBeDefined();
    });

    it('runs all conditions', async () => {
      await runCli(['--condition', 'all', '--json']);
      const jsonOut = logSpy.mock.calls.find((c) => {
        try { return JSON.parse(c[0]).conditions; } catch { return false; }
      });
      expect(jsonOut).toBeDefined();
      const parsed = JSON.parse(jsonOut[0]);
      expect(parsed.conditions).toEqual(
        expect.arrayContaining(['bm25_only', 'hybrid', 'hybrid_rerank', 'advanced_default']),
      );
    });

    it('throws for unsupported conditions', async () => {
      await runCli(['--condition', 'invalid']);
      expect(stderrSpy).toHaveBeenCalled();
    });

    it('runs --compare with two conditions in JSON mode', async () => {
      await runCli(['--condition', 'bm25_only', '--condition', 'hybrid', '--compare', '--json']);
      const anyJson = logSpy.mock.calls.find((c) => {
        try { JSON.parse(c[0]); return true; } catch { return false; }
      });
      expect(anyJson).toBeDefined();
    });

    it('runs --compare with two conditions in text mode', async () => {
      await runCli(['--condition', 'bm25_only', '--condition', 'hybrid', '--compare']);
      expect(logSpy).toHaveBeenCalled();
    });

    it('runs --alpha-sweep in JSON mode', async () => {
      await runCli(['--alpha-sweep', '0.3,0.5,0.7', '--json']);
      const anyJson = logSpy.mock.calls.find((c) => {
        try { JSON.parse(c[0]); return true; } catch { return false; }
      });
      expect(anyJson).toBeDefined();
    });

    it('runs --alpha-sweep in text mode', async () => {
      await runCli(['--alpha-sweep', '0.3,0.5']);
      expect(logSpy).toHaveBeenCalled();
    });

    it('writes output to file with --output', async () => {
      await runCli(['--condition', 'bm25_only', '--output', 'C:\\mock\\output.json', '--json']);
      expect(mockWriteFile).toHaveBeenCalled();
    });

    it('runs --baseline regression check with failures', async () => {
      // Baseline has mrr_at_5: 0.0, current will be higher → pass
      await runCli(['--condition', 'bm25_only', '--baseline', 'C:\\mock\\baseline.json', '--json']);
      const anyJson = logSpy.mock.calls.find((c) => {
        try { return JSON.parse(c[0]).regression; } catch { return false; }
      });
      expect(anyJson).toBeDefined();
      const parsed = JSON.parse(anyJson[0]);
      expect(parsed.regression.checked).toBe(true);
    });

    it('runs --baseline with regression failures (exitCode=1)', async () => {
      // Set baseline mrr_at_5 very high so current can't beat it
      mockReadFile.mockImplementation((filePath) => {
        const fp = String(filePath);
        if (fp.includes('eval-queries')) {
          return Promise.resolve(JSON.stringify(makeTestQueries()));
        }
        if (fp.includes('baseline')) {
          return Promise.resolve(JSON.stringify({
            conditions: {
              bm25_only: { mrr_at_5: 999.0, ndcg_at_5: 999.0, recall_at_5: 999.0, latency_ms: { p50: 0, p95: 0, avg: 0 } },
            },
          }));
        }
        return Promise.resolve('');
      });
      await runCli(['--condition', 'bm25_only', '--baseline', 'C:\\mock\\baseline.json', '--json']);
      expect(process.exitCode).toBe(1);
    });

    it('handles error in runCli', async () => {
      mockReadFile.mockRejectedValue(new Error('file not found'));
      await runCli(['--condition', 'bm25_only']);
      expect(stderrSpy).toHaveBeenCalled();
    });

    it('handles error in runCli with --json', async () => {
      mockReadFile.mockRejectedValue('string error');
      await runCli(['--condition', 'bm25_only', '--json']);
      expect(logSpy).toHaveBeenCalled();
    });

    it('runs --recall-benchmark in text mode', async () => {
      mockGetTursoClient.mockResolvedValue({
        execute: jest.fn().mockImplementation((params) => {
          const sql = typeof params === 'string' ? params : params.sql;
          if (sql.includes('vector_top_k')) throw new Error('no index');
          if (sql.includes('vector_distance_cos')) {
            return Promise.resolve({ rows: [{ chunk_id: 1 }] });
          }
          return Promise.resolve({ rows: [] });
        }),
      });

      await runCli(['--recall-benchmark', '--query-file', 'C:\\mock\\eval-queries.json']);
      expect(logSpy).toHaveBeenCalled();
    });

    it('runs --recall-benchmark in JSON mode', async () => {
      mockGetTursoClient.mockResolvedValue({
        execute: jest.fn().mockResolvedValue({ rows: [{ chunk_id: 1 }] }),
      });

      await runCli(['--recall-benchmark', '--json']);
      const anyJson = logSpy.mock.calls.find((c) => {
        try { return JSON.parse(c[0]).benchmark === 'recall-benchmark'; } catch { return false; }
      });
      expect(anyJson).toBeDefined();
    });
  });

  // -------------------------------------------------------------------------
  // bootstrap
  // -------------------------------------------------------------------------

  describe('bootstrap', () => {
    let stdoutSpy;
    let logSpy;
    let origArgv;

    beforeEach(() => {
      stdoutSpy = jest.spyOn(process.stdout, 'write').mockImplementation(() => {});
      logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      origArgv = process.argv;
    });

    afterEach(() => {
      stdoutSpy.mockRestore();
      logSpy.mockRestore();
      process.argv = origArgv;
    });

    it('runs runCli when process.argv[1] matches module path', async () => {
      const modulePath = fileURLToPath(new URL('./eval-runner.mjs', import.meta.url));
      process.argv = ['node', modulePath, '--help'];
      await bootstrap();
      expect(logSpy).toHaveBeenCalled();
    });

    it('does nothing when process.argv[1] does not match', async () => {
      process.argv = ['node', 'other-script.mjs'];
      await bootstrap();
      expect(logSpy).not.toHaveBeenCalled();
    });
  });
});