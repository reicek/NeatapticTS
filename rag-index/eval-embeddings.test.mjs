/**
 * @module eval-embeddings.test
 * @description Comprehensive tests for eval-embeddings.mjs targeting 100% coverage.
 */
import { jest } from '@jest/globals';
import { fileURLToPath } from 'node:url';

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

const mockReadFile = jest.fn();
const mockCreateClient = jest.fn();
const mockQueryDenseIndex = jest.fn();
const mockReadModelMeta = jest.fn();
const mockCreateOnnxTextEmbedder = jest.fn();

jest.unstable_mockModule('node:fs/promises', () => ({
  ...jest.requireActual('node:fs/promises'),
  readFile: mockReadFile,
}));
jest.unstable_mockModule('@libsql/client', () => ({
  createClient: mockCreateClient,
}));
jest.unstable_mockModule('./query-dense.mjs', () => ({
  queryDenseIndex: mockQueryDenseIndex,
}));
jest.unstable_mockModule('./embed-index.mjs', () => ({
  DEFAULT_MODEL_DIRECTORY: '/mock/model-dir',
  DEFAULT_MODEL_ID: 'mock-model-id',
  createOnnxTextEmbedder: mockCreateOnnxTextEmbedder,
  readModelMeta: mockReadModelMeta,
}));

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeQuerySpecs(overrides = {}) {
  const base = [
    {
      query: 'network activate',
      family: null,
      expected_doc_families: ['ts-source'],
      expected_heading_contains: 'activate',
      expected_symbol_contains: null,
    },
    {
      query: 'neuron mutate',
      family: null,
      expected_doc_families: ['ts-source'],
      expected_heading_contains: 'mutate',
      expected_symbol_contains: null,
    },
    {
      query: 'genome evolve',
      family: null,
      expected_doc_families: ['ts-source'],
      expected_heading_contains: 'evolve',
      expected_symbol_contains: null,
    },
    {
      query: 'population fitness',
      family: null,
      expected_doc_families: ['ts-source'],
      expected_heading_contains: 'fitness',
      expected_symbol_contains: null,
    },
    {
      query: 'crossover mix',
      family: null,
      expected_doc_families: ['ts-source'],
      expected_heading_contains: 'crossover',
      expected_symbol_contains: null,
    },
  ];
  return base.map((q, i) => ({ ...q, ...overrides[i] }));
}

function makeResult(family, heading) {
  return { family, heading_path: heading };
}

// ---------------------------------------------------------------------------

describe('eval-embeddings', () => {
  let evaluateEmbeddings;

  beforeAll(async () => {
    const mod = await import('./eval-embeddings.mjs');
    evaluateEmbeddings = mod.evaluateEmbeddings;
  });

  beforeEach(() => {
    jest.clearAllMocks();
    mockReadFile.mockImplementation((filePath) => {
      if (typeof filePath === 'string' && filePath.includes('eval-queries')) {
        return Promise.resolve(JSON.stringify(makeQuerySpecs()));
      }
      return Promise.resolve('');
    });
    mockReadModelMeta.mockResolvedValue({ model_id: 'mock-model', dimension: 10 });
    mockCreateOnnxTextEmbedder.mockResolvedValue(async () => new Float32Array([1, 2, 3]));
  });

  // -------------------------------------------------------------------------
  // evaluateEmbeddings — basic usage with client provided
  // -------------------------------------------------------------------------

  describe('evaluateEmbeddings', () => {
    it('computes MRR@5 for BM25 and hybrid with client provided', async () => {
      const querySpecs = makeQuerySpecs();
      mockReadFile.mockResolvedValue(JSON.stringify(querySpecs));
      mockQueryDenseIndex.mockImplementation((opts) => {
        const isHybrid = opts.dense;
        const heading = isHybrid ? 'activate' : 'nope';
        // For hybrid: first query matches, for bm25: no match
        if (opts.query === 'network activate') {
          return Promise.resolve({
            results: isHybrid
              ? [makeResult('ts-source', 'activate')]
              : [makeResult('ts-source', 'nope')],
          });
        }
        return Promise.resolve({
          results: isHybrid
            ? [makeResult('ts-source', opts.query.split(' ')[1] || 'match')]
            : [makeResult('ts-source', 'nope')],
        });
      });

      const mockClient = {
        execute: jest.fn()
          .mockResolvedValueOnce({ rows: [{ count: 100 }] })
          .mockResolvedValueOnce({ rows: [{ count: 50 }] }),
      };

      const result = await evaluateEmbeddings({
        client: mockClient,
        embedText: async () => new Float32Array([1, 2, 3]),
      });

      expect(result.queryCount).toBe(5);
      expect(result.chunkCount).toBe(100);
      expect(result.embeddingCount).toBe(50);
      expect(result.tsSourceQueries).toBe(5);
      expect(result.pass).toBe(true);
    });

    it('returns pass=false when hybrid improvement is below threshold', async () => {
      const querySpecs = makeQuerySpecs();
      mockReadFile.mockResolvedValue(JSON.stringify(querySpecs));
      // Both BM25 and hybrid produce same results → no improvement
      mockQueryDenseIndex.mockResolvedValue({
        results: [makeResult('ts-source', 'match')],
      });

      const mockClient = {
        execute: jest.fn()
          .mockResolvedValueOnce({ rows: [{ count: 100 }] })
          .mockResolvedValueOnce({ rows: [{ count: 50 }] }),
      };

      const result = await evaluateEmbeddings({
        client: mockClient,
        embedText: async () => new Float32Array([1, 2, 3]),
      });

      expect(result.improvement).toBe(0);
      expect(result.pass).toBe(false);
    });

    it('returns pass=false when tsSourceQueries < 5', async () => {
      const querySpecs = makeQuerySpecs({
        4: { expected_doc_families: ['md-docs'] },
      });
      mockReadFile.mockResolvedValue(JSON.stringify(querySpecs));
      mockQueryDenseIndex.mockImplementation((opts) => {
        const heading = opts.query.split(' ')[1] || 'match';
        return Promise.resolve({
          results: opts.dense
            ? [makeResult('ts-source', heading)]
            : [makeResult('ts-source', 'nope')],
        });
      });

      const mockClient = {
        execute: jest.fn()
          .mockResolvedValueOnce({ rows: [{ count: 100 }] })
          .mockResolvedValueOnce({ rows: [{ count: 50 }] }),
      };

      const result = await evaluateEmbeddings({
        client: mockClient,
        embedText: async () => new Float32Array([1, 2, 3]),
      });

      expect(result.tsSourceQueries).toBe(4);
      expect(result.pass).toBe(false);
    });

    it('handles empty querySpecs (div by zero guard)', async () => {
      mockReadFile.mockResolvedValue(JSON.stringify([]));
      const mockClient = {
        execute: jest.fn()
          .mockResolvedValueOnce({ rows: [{ count: 0 }] })
          .mockResolvedValueOnce({ rows: [{ count: 0 }] }),
      };

      const result = await evaluateEmbeddings({
        client: mockClient,
        embedText: async () => new Float32Array([1, 2, 3]),
      });

      expect(result.queryCount).toBe(0);
      expect(result.bm25MrrAt5).toBe(0);
      expect(result.hybridMrrAt5).toBe(0);
      expect(result.pass).toBe(false);
    });

    it('uses createClient path when no client provided', async () => {
      const querySpecs = makeQuerySpecs();
      mockReadFile.mockResolvedValue(JSON.stringify(querySpecs));
      mockQueryDenseIndex.mockResolvedValue({
        results: [makeResult('ts-source', 'match')],
      });

      const mockCorpusClient = {
        execute: jest.fn()
          .mockResolvedValueOnce({ rows: [{ count: 200 }] })
          .mockResolvedValueOnce({ rows: [{ count: 100 }] }),
        close: jest.fn().mockResolvedValue(undefined),
      };
      mockCreateClient.mockReturnValue(mockCorpusClient);

      const result = await evaluateEmbeddings({
        embedText: async () => new Float32Array([1, 2, 3]),
      });

      expect(mockCreateClient).toHaveBeenCalled();
      expect(mockCorpusClient.close).toHaveBeenCalled();
      expect(result.chunkCount).toBe(200);
      expect(result.embeddingCount).toBe(100);
    });

    it('creates embedText via createOnnxTextEmbedder when not provided', async () => {
      const querySpecs = makeQuerySpecs();
      mockReadFile.mockResolvedValue(JSON.stringify(querySpecs));
      mockQueryDenseIndex.mockResolvedValue({
        results: [makeResult('ts-source', 'match')],
      });
      const mockEmbedFn = async () => new Float32Array([1, 2, 3]);
      mockEmbedFn.release = jest.fn().mockResolvedValue(undefined);
      mockCreateOnnxTextEmbedder.mockResolvedValue(mockEmbedFn);

      const mockClient = {
        execute: jest.fn()
          .mockResolvedValueOnce({ rows: [{ count: 100 }] })
          .mockResolvedValueOnce({ rows: [{ count: 50 }] }),
      };

      await evaluateEmbeddings({ client: mockClient });

      expect(mockCreateOnnxTextEmbedder).toHaveBeenCalled();
      // release should be called since embedText was created internally
      expect(mockEmbedFn.release).toHaveBeenCalled();
    });

    it('does not call release when embedText provided with release function', async () => {
      const querySpecs = makeQuerySpecs();
      mockReadFile.mockResolvedValue(JSON.stringify(querySpecs));
      mockQueryDenseIndex.mockResolvedValue({
        results: [makeResult('ts-source', 'match')],
      });
      const releaseFn = jest.fn().mockResolvedValue(undefined);
      const mockEmbedFn = async () => new Float32Array([1, 2, 3]);
      mockEmbedFn.release = releaseFn;

      const mockClient = {
        execute: jest.fn()
          .mockResolvedValueOnce({ rows: [{ count: 100 }] })
          .mockResolvedValueOnce({ rows: [{ count: 50 }] }),
      };

      await evaluateEmbeddings({ client: mockClient, embedText: mockEmbedFn });

      expect(releaseFn).not.toHaveBeenCalled();
    });

    it('does not call release when embedText has no release function', async () => {
      const querySpecs = makeQuerySpecs();
      mockReadFile.mockResolvedValue(JSON.stringify(querySpecs));
      mockQueryDenseIndex.mockResolvedValue({
        results: [makeResult('ts-source', 'match')],
      });
      const mockEmbedFn = async () => new Float32Array([1, 2, 3]);
      // no release property

      const mockClient = {
        execute: jest.fn()
          .mockResolvedValueOnce({ rows: [{ count: 100 }] })
          .mockResolvedValueOnce({ rows: [{ count: 50 }] }),
      };

      await evaluateEmbeddings({ client: mockClient, embedText: mockEmbedFn });
      // Should not throw
    });

    it('handles matchesQueryExpectation family mismatch', async () => {
      const querySpecs = [
        {
          query: 'test',
          family: null,
          expected_doc_families: ['ts-source'],
          expected_heading_contains: 'activate',
          expected_symbol_contains: null,
        },
        ...makeQuerySpecs().slice(1),
      ];
      mockReadFile.mockResolvedValue(JSON.stringify(querySpecs));
      // Results have wrong family
      mockQueryDenseIndex.mockResolvedValue({
        results: [makeResult('md-docs', 'activate')],
      });

      const mockClient = {
        execute: jest.fn()
          .mockResolvedValueOnce({ rows: [{ count: 100 }] })
          .mockResolvedValueOnce({ rows: [{ count: 50 }] }),
      };

      const result = await evaluateEmbeddings({
        client: mockClient,
        embedText: async () => new Float32Array([1, 2, 3]),
      });

      expect(result.queryReports[0].bm25_rank).toBeNull();
    });

    it('handles matchesQueryExpectation with no expected_doc_families', async () => {
      const querySpecs = [
        {
          query: 'test',
          family: null,
          expected_doc_families: [],
          expected_heading_contains: 'activate',
          expected_symbol_contains: null,
        },
        ...makeQuerySpecs().slice(1),
      ];
      mockReadFile.mockResolvedValue(JSON.stringify(querySpecs));
      mockQueryDenseIndex.mockResolvedValue({
        results: [makeResult('any-family', 'activate')],
      });

      const mockClient = {
        execute: jest.fn()
          .mockResolvedValueOnce({ rows: [{ count: 100 }] })
          .mockResolvedValueOnce({ rows: [{ count: 50 }] }),
      };

      const result = await evaluateEmbeddings({
        client: mockClient,
        embedText: async () => new Float32Array([1, 2, 3]),
      });

      // Should match since no family restriction
      expect(result.queryReports[0].bm25_rank).toBe(1);
    });

    it('handles expected_doc_families not an array', async () => {
      const querySpecs = [
        {
          query: 'test',
          family: null,
          expected_doc_families: 'not-an-array',
          expected_heading_contains: 'activate',
          expected_symbol_contains: null,
        },
        ...makeQuerySpecs().slice(1),
      ];
      mockReadFile.mockResolvedValue(JSON.stringify(querySpecs));
      mockQueryDenseIndex.mockResolvedValue({
        results: [makeResult('any-family', 'activate')],
      });

      const mockClient = {
        execute: jest.fn()
          .mockResolvedValueOnce({ rows: [{ count: 100 }] })
          .mockResolvedValueOnce({ rows: [{ count: 50 }] }),
      };

      const result = await evaluateEmbeddings({
        client: mockClient,
        embedText: async () => new Float32Array([1, 2, 3]),
      });

      expect(result.queryReports[0].bm25_rank).toBe(1);
    });

    it('handles symbol needle match and mismatch', async () => {
      const querySpecs = [
        {
          query: 'test',
          family: null,
          expected_doc_families: ['ts-source'],
          expected_heading_contains: null,
          expected_symbol_contains: 'activatenode',
        },
        ...makeQuerySpecs().slice(1),
      ];
      mockReadFile.mockResolvedValue(JSON.stringify(querySpecs));
      // First call: heading contains symbol needle → match
      // Second call (hybrid): heading doesn't contain → no match
      mockQueryDenseIndex.mockImplementation((opts) => {
        if (opts.query === 'test') {
          return Promise.resolve({
            results: opts.dense
              ? [makeResult('ts-source', 'mutate')]
              : [makeResult('ts-source', 'activatenode')],
          });
        }
        return Promise.resolve({
          results: [makeResult('ts-source', 'match')],
        });
      });

      const mockClient = {
        execute: jest.fn()
          .mockResolvedValueOnce({ rows: [{ count: 100 }] })
          .mockResolvedValueOnce({ rows: [{ count: 50 }] }),
      };

      const result = await evaluateEmbeddings({
        client: mockClient,
        embedText: async () => new Float32Array([1, 2, 3]),
      });

      expect(result.queryReports[0].bm25_rank).toBe(1);
      expect(result.queryReports[0].hybrid_rank).toBeNull();
    });

    it('handles heading needle mismatch (heading does not include needle)', async () => {
      const querySpecs = [
        {
          query: 'test',
          family: null,
          expected_doc_families: ['ts-source'],
          expected_heading_contains: 'activate',
          expected_symbol_contains: null,
        },
        ...makeQuerySpecs().slice(1),
      ];
      mockReadFile.mockResolvedValue(JSON.stringify(querySpecs));
      mockQueryDenseIndex.mockResolvedValue({
        results: [makeResult('ts-source', 'somethingelse')],
      });

      const mockClient = {
        execute: jest.fn()
          .mockResolvedValueOnce({ rows: [{ count: 100 }] })
          .mockResolvedValueOnce({ rows: [{ count: 50 }] }),
      };

      const result = await evaluateEmbeddings({
        client: mockClient,
        embedText: async () => new Float32Array([1, 2, 3]),
      });

      expect(result.queryReports[0].bm25_rank).toBeNull();
    });

    it('handles findHitRank with min_rank_of_hit set and match beyond it', async () => {
      const querySpecs = [
        {
          query: 'test',
          family: null,
          expected_doc_families: ['ts-source'],
          expected_heading_contains: 'match',
          expected_symbol_contains: null,
          min_rank_of_hit: 1,
        },
        ...makeQuerySpecs().slice(1),
      ];
      mockReadFile.mockResolvedValue(JSON.stringify(querySpecs));
      // First result doesn't contain 'match', second does at rank 2 but min_rank_of_hit=1 → null
      mockQueryDenseIndex.mockImplementation((opts) => {
        if (opts.query === 'test') {
          return Promise.resolve({
            results: [
              makeResult('ts-source', 'nope'),
              makeResult('ts-source', 'match'),
            ],
          });
        }
        return Promise.resolve({ results: [makeResult('ts-source', 'match')] });
      });

      const mockClient = {
        execute: jest.fn()
          .mockResolvedValueOnce({ rows: [{ count: 100 }] })
          .mockResolvedValueOnce({ rows: [{ count: 50 }] }),
      };

      const result = await evaluateEmbeddings({
        client: mockClient,
        embedText: async () => new Float32Array([1, 2, 3]),
      });

      expect(result.queryReports[0].bm25_rank).toBeNull();
    });

    it('handles empty results (no hits)', async () => {
      const querySpecs = makeQuerySpecs();
      mockReadFile.mockResolvedValue(JSON.stringify(querySpecs));
      mockQueryDenseIndex.mockResolvedValue({ results: [] });

      const mockClient = {
        execute: jest.fn()
          .mockResolvedValueOnce({ rows: [{ count: 100 }] })
          .mockResolvedValueOnce({ rows: [{ count: 50 }] }),
      };

      const result = await evaluateEmbeddings({
        client: mockClient,
        embedText: async () => new Float32Array([1, 2, 3]),
      });

      expect(result.queryReports[0].bm25_rank).toBeNull();
      expect(result.bm25MrrAt5).toBe(0);
    });

    it('uses custom alpha, modelDirectory, modelId, and minHybridImprovement', async () => {
      const querySpecs = makeQuerySpecs();
      mockReadFile.mockResolvedValue(JSON.stringify(querySpecs));
      mockQueryDenseIndex.mockResolvedValue({
        results: [makeResult('ts-source', 'match')],
      });

      const mockClient = {
        execute: jest.fn()
          .mockResolvedValueOnce({ rows: [{ count: 100 }] })
          .mockResolvedValueOnce({ rows: [{ count: 50 }] }),
      };

      const result = await evaluateEmbeddings({
        alpha: 0.3,
        client: mockClient,
        embedText: async () => new Float32Array([1, 2, 3]),
        minHybridImprovement: 0.5,
        modelDirectory: '/custom/model',
        modelId: 'custom-model',
      });

      expect(result.alpha).toBe(0.3);
      expect(result.minHybridImprovement).toBe(0.5);
      expect(result.modelId).toBe('custom-model');
    });

    it('uses options.databasePath when corpusDatabasePath not provided', async () => {
      const querySpecs = makeQuerySpecs();
      mockReadFile.mockResolvedValue(JSON.stringify(querySpecs));
      mockQueryDenseIndex.mockResolvedValue({
        results: [makeResult('ts-source', 'match')],
      });

      const mockClient = {
        execute: jest.fn()
          .mockResolvedValueOnce({ rows: [{ count: 100 }] })
          .mockResolvedValueOnce({ rows: [{ count: 50 }] }),
      };

      await evaluateEmbeddings({
        databasePath: 'C:\\custom\\db.sqlite',
        client: mockClient,
        embedText: async () => new Float32Array([1, 2, 3]),
      });

      // Should have called queryDenseIndex with resolved corpusDatabasePath
      expect(mockQueryDenseIndex).toHaveBeenCalled();
      const callArgs = mockQueryDenseIndex.mock.calls[0][0];
      expect(callArgs.corpusDatabasePath).toContain('db.sqlite');
    });
  });

  // -------------------------------------------------------------------------
  // main() — guarded entry point
  // -------------------------------------------------------------------------

  describe('main() via import', () => {
    let exitSpy;
    let logSpy;
    let stdoutSpy;
    let origArgv1;

    beforeEach(() => {
      exitSpy = jest.spyOn(process, 'exit').mockImplementation(() => {});
      logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      stdoutSpy = jest.spyOn(process.stdout, 'write').mockImplementation(() => {});
      origArgv1 = process.argv[1];
      jest.resetModules();
      mockReadFile.mockImplementation((filePath) => {
        if (typeof filePath === 'string' && filePath.includes('eval-queries')) {
          return Promise.resolve(JSON.stringify(makeQuerySpecs()));
        }
        return Promise.resolve('');
      });
      mockReadModelMeta.mockResolvedValue({ model_id: 'mock-model', dimension: 10 });
      mockCreateOnnxTextEmbedder.mockResolvedValue(async () => new Float32Array([1, 2, 3]));
    });

    afterEach(() => {
      exitSpy.mockRestore();
      logSpy.mockRestore();
      stdoutSpy.mockRestore();
      process.argv[1] = origArgv1;
    });

    it('shows help with --help flag', async () => {
      const modulePath = fileURLToPath(new URL('./eval-embeddings.mjs', import.meta.url));
      process.argv = ['node', modulePath, '--help'];
      await import('./eval-embeddings.mjs');
      expect(logSpy).toHaveBeenCalled();
    });

    it('runs evaluation in text mode (pass)', async () => {
      const modulePath = fileURLToPath(new URL('./eval-embeddings.mjs', import.meta.url));
      process.argv = ['node', modulePath];
      mockQueryDenseIndex.mockImplementation((opts) => {
        const heading = opts.query.split(' ')[1] || 'match';
        return Promise.resolve({
          results: opts.dense
            ? [makeResult('ts-source', heading)]
            : [makeResult('ts-source', 'nope')],
        });
      });
      const mockClient = {
        execute: jest.fn()
          .mockResolvedValueOnce({ rows: [{ count: 100 }] })
          .mockResolvedValueOnce({ rows: [{ count: 50 }] }),
      };
      // Need to mock getTursoClient? No, eval-embeddings uses createClient directly
      // But main() doesn't pass client, so it uses createClient path
      mockCreateClient.mockReturnValue({
        execute: jest.fn()
          .mockResolvedValueOnce({ rows: [{ count: 100 }] })
          .mockResolvedValueOnce({ rows: [{ count: 50 }] }),
        close: jest.fn().mockResolvedValue(undefined),
      });

      await import('./eval-embeddings.mjs');

      // Should have written to stdout (text mode)
      expect(logSpy).toHaveBeenCalled();
    });

    it('runs evaluation in JSON mode (fail)', async () => {
      const modulePath = fileURLToPath(new URL('./eval-embeddings.mjs', import.meta.url));
      process.argv = ['node', modulePath, '--json'];
      mockQueryDenseIndex.mockResolvedValue({
        results: [makeResult('ts-source', 'match')],
      });
      mockCreateClient.mockReturnValue({
        execute: jest.fn()
          .mockResolvedValueOnce({ rows: [{ count: 100 }] })
          .mockResolvedValueOnce({ rows: [{ count: 50 }] }),
        close: jest.fn().mockResolvedValue(undefined),
      });

      await import('./eval-embeddings.mjs');

      expect(logSpy).toHaveBeenCalled();
      const jsonOutput = logSpy.mock.calls.find((c) => {
        try {
          JSON.parse(c[0]);
          return true;
        } catch {
          return false;
        }
      });
      expect(jsonOutput).toBeDefined();
    });

    it('handles error in main() with non-Error throw', async () => {
      const modulePath = fileURLToPath(new URL('./eval-embeddings.mjs', import.meta.url));
      process.argv = ['node', modulePath, '--json'];
      mockReadFile.mockRejectedValue('string error');

      await import('./eval-embeddings.mjs');

      expect(logSpy).toHaveBeenCalled();
    });
  });
});