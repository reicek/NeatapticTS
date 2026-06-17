/**
 * @module eval-metrics.red.test
 * @description Red tests for the RAG evaluation metrics module.
 *
 * The metrics module implements MRR@k, nDCG@k, Recall@k, context relevance,
 * and latency computations used by the Step 22 eval suite.
 */

const METRICS_PATH = '../../semantic-index/eval-metrics.mjs';

function loadMetrics() {
  return import(METRICS_PATH);
}

function makeChunk(overrides = {}) {
  return {
    chunk_id: 1,
    file_path: 'src/foo.ts',
    doc_family: 'ts-source',
    heading_path: 'Foo',
    ...overrides,
  };
}

describe('eval-metrics', () => {
  describe('MRR@k', () => {
    it('computes MRR@1 for a single relevant result at rank 1', async () => {
      const { computeMrr } = await loadMetrics();
      const results = [makeChunk({ chunk_id: 1, doc_family: 'ts-source' })];
      const query = {
        expected_doc_families: ['ts-source'],
        expected_heading_contains: null,
        expected_symbol_contains: null,
      };
      expect(computeMrr(results, query, 1)).toBe(1);
    });

    it('computes MRR@5 when the first relevant result is at rank 3', async () => {
      const { computeMrr } = await loadMetrics();
      const results = [
        makeChunk({ chunk_id: 1, doc_family: 'readme' }),
        makeChunk({ chunk_id: 2, doc_family: 'readme' }),
        makeChunk({ chunk_id: 3, doc_family: 'ts-source' }),
      ];
      const query = {
        expected_doc_families: ['ts-source'],
        expected_heading_contains: null,
        expected_symbol_contains: null,
      };
      expect(computeMrr(results, query, 5)).toBeCloseTo(1 / 3, 6);
    });

    it('returns 0 when no relevant result is within the top-k', async () => {
      const { computeMrr } = await loadMetrics();
      const results = [
        makeChunk({ chunk_id: 1, doc_family: 'readme' }),
        makeChunk({ chunk_id: 2, doc_family: 'readme' }),
      ];
      const query = {
        expected_doc_families: ['ts-source'],
        expected_heading_contains: null,
        expected_symbol_contains: null,
      };
      expect(computeMrr(results, query, 5)).toBe(0);
    });

    it('uses heading match to determine relevance', async () => {
      const { computeMrr } = await loadMetrics();
      const results = [
        makeChunk({
          chunk_id: 1,
          doc_family: 'ts-source',
          heading_path: 'Bar',
        }),
        makeChunk({
          chunk_id: 2,
          doc_family: 'ts-source',
          heading_path: 'Foo',
        }),
      ];
      const query = {
        expected_doc_families: ['ts-source'],
        expected_heading_contains: 'Foo',
        expected_symbol_contains: null,
      };
      expect(computeMrr(results, query, 5)).toBeCloseTo(1 / 2, 6);
    });

    it('rejects a non-positive k', async () => {
      const { computeMrr } = await loadMetrics();
      expect(() => computeMrr([], {}, 0)).toThrow();
    });
  });

  describe('nDCG@k', () => {
    it('computes nDCG@5 with graded relevance', async () => {
      const { computeNdcg } = await loadMetrics();
      const results = [
        makeChunk({
          chunk_id: 1,
          doc_family: 'ts-source',
          heading_path: 'Foo',
        }),
        makeChunk({
          chunk_id: 2,
          doc_family: 'ts-source',
          heading_path: 'Bar',
        }),
        makeChunk({ chunk_id: 3, doc_family: 'readme' }),
      ];
      const query = {
        expected_doc_families: ['ts-source'],
        expected_heading_contains: null,
        expected_symbol_contains: null,
        relevance_grades: [
          { heading_path_contains: 'Foo', family: 'ts-source', grade: 3 },
          { heading_path_contains: 'Bar', family: 'ts-source', grade: 2 },
        ],
      };
      expect(computeNdcg(results, query, 5)).toBeGreaterThan(0);
      expect(computeNdcg(results, query, 5)).toBeLessThanOrEqual(1);
    });

    it('returns 1 for perfectly ordered graded results', async () => {
      const { computeNdcg } = await loadMetrics();
      const results = [
        makeChunk({
          chunk_id: 1,
          doc_family: 'ts-source',
          heading_path: 'Foo',
        }),
        makeChunk({
          chunk_id: 2,
          doc_family: 'ts-source',
          heading_path: 'Bar',
        }),
      ];
      const query = {
        expected_doc_families: ['ts-source'],
        expected_heading_contains: null,
        expected_symbol_contains: null,
        relevance_grades: [
          { heading_path_contains: 'Foo', family: 'ts-source', grade: 3 },
          { heading_path_contains: 'Bar', family: 'ts-source', grade: 2 },
        ],
      };
      expect(computeNdcg(results, query, 5)).toBe(1);
    });

    it('falls back to binary relevance when grades are absent', async () => {
      const { computeNdcg } = await loadMetrics();
      const results = [
        makeChunk({ chunk_id: 1, doc_family: 'ts-source' }),
        makeChunk({ chunk_id: 2, doc_family: 'readme' }),
      ];
      const query = {
        expected_doc_families: ['ts-source'],
        expected_heading_contains: null,
        expected_symbol_contains: null,
      };
      expect(computeNdcg(results, query, 5)).toBeGreaterThan(0);
    });

    it('returns 0 when no results are relevant', async () => {
      const { computeNdcg } = await loadMetrics();
      const results = [makeChunk({ chunk_id: 1, doc_family: 'readme' })];
      const query = {
        expected_doc_families: ['ts-source'],
        expected_heading_contains: null,
        expected_symbol_contains: null,
      };
      expect(computeNdcg(results, query, 5)).toBe(0);
    });

    it('rejects invalid k', async () => {
      const { computeNdcg } = await loadMetrics();
      expect(() => computeNdcg([], {}, -1)).toThrow();
    });
  });

  describe('Recall@k', () => {
    it('computes Recall@k using expected_chunk_ids', async () => {
      const { computeRecall } = await loadMetrics();
      const results = [
        makeChunk({ chunk_id: 1 }),
        makeChunk({ chunk_id: 2 }),
        makeChunk({ chunk_id: 3 }),
      ];
      const query = {
        expected_chunk_ids: [1, 3, 4],
      };
      expect(computeRecall(results, query, 5)).toBeCloseTo(2 / 3, 6);
    });

    it('falls back to family+heading match when expected_chunk_ids is empty', async () => {
      const { computeRecall } = await loadMetrics();
      const results = [
        makeChunk({ chunk_id: 1, doc_family: 'ts-source' }),
        makeChunk({ chunk_id: 2, doc_family: 'readme' }),
      ];
      const query = {
        expected_doc_families: ['ts-source', 'readme'],
        expected_chunk_ids: [],
      };
      expect(computeRecall(results, query, 5)).toBe(1);
    });

    it('returns 0 when no relevant results are retrieved', async () => {
      const { computeRecall } = await loadMetrics();
      const results = [makeChunk({ chunk_id: 1, doc_family: 'readme' })];
      const query = {
        expected_chunk_ids: [99],
      };
      expect(computeRecall(results, query, 5)).toBe(0);
    });

    it('caps recall at 1', async () => {
      const { computeRecall } = await loadMetrics();
      const results = [makeChunk({ chunk_id: 1 })];
      const query = {
        expected_chunk_ids: [1],
      };
      expect(computeRecall(results, query, 5)).toBe(1);
    });
  });

  describe('context relevance', () => {
    it('measures fraction of relevant chunks in assembled context', async () => {
      const { computeContextRelevance } = await loadMetrics();
      const contextChunks = [
        makeChunk({ chunk_id: 1, doc_family: 'ts-source' }),
        makeChunk({ chunk_id: 2, doc_family: 'ts-source' }),
        makeChunk({ chunk_id: 3, doc_family: 'readme' }),
      ];
      const query = {
        expected_doc_families: ['ts-source'],
        expected_heading_contains: null,
        expected_symbol_contains: null,
      };
      expect(computeContextRelevance(contextChunks, query)).toBeCloseTo(
        2 / 3,
        6,
      );
    });

    it('returns 0 for an empty context', async () => {
      const { computeContextRelevance } = await loadMetrics();
      const query = {
        expected_doc_families: ['ts-source'],
        expected_heading_contains: null,
        expected_symbol_contains: null,
      };
      expect(computeContextRelevance([], query)).toBe(0);
    });

    it('returns 1 when all chunks are relevant', async () => {
      const { computeContextRelevance } = await loadMetrics();
      const contextChunks = [
        makeChunk({ chunk_id: 1, doc_family: 'ts-source' }),
      ];
      const query = {
        expected_doc_families: ['ts-source'],
        expected_heading_contains: null,
        expected_symbol_contains: null,
      };
      expect(computeContextRelevance(contextChunks, query)).toBe(1);
    });
  });

  describe('latency', () => {
    it('tracks end-to-end query latency in milliseconds', async () => {
      const { measureLatency } = await loadMetrics();
      const { latency_ms: latency } = await measureLatency(async () => {
        await new Promise((resolve) => setTimeout(resolve, 5));
        return 'done';
      });
      expect(typeof latency).toBe('number');
      expect(latency).toBeGreaterThanOrEqual(0);
    });

    it('returns the async result alongside latency', async () => {
      const { measureLatency } = await loadMetrics();
      const { result, latency_ms: latency } = await measureLatency(
        async () => 'payload',
      );
      expect(result).toBe('payload');
      expect(typeof latency).toBe('number');
    });

    it('rejects non-function inputs', async () => {
      const { measureLatency } = await loadMetrics();
      await expect(measureLatency('not a function')).rejects.toThrow();
    });
  });

  describe('invalid inputs', () => {
    it('throws when results is not an array', async () => {
      const { computeMrr } = await loadMetrics();
      expect(() => computeMrr(null, {}, 5)).toThrow();
    });

    it('throws when query is missing', async () => {
      const { computeMrr } = await loadMetrics();
      expect(() => computeMrr([], undefined, 5)).toThrow();
    });

    it('throws when k is not an integer', async () => {
      const { computeRecall } = await loadMetrics();
      expect(() => computeRecall([], {}, 5.5)).toThrow();
    });
  });
});
