/**
 * @module eval-runner.red.test
 * @description Red tests for the RAG evaluation runner module.
 *
 * The runner orchestrates query execution, metric computation, and
 * condition-level aggregation for the Step 22 eval suite.
 */

const RUNNER_PATH = '../../semantic-index/eval-runner.mjs';
const COMPARE_PATH = '../../semantic-index/eval-compare.mjs';

function loadRunner() {
  return import(RUNNER_PATH);
}

function loadCompare() {
  return import(COMPARE_PATH);
}

function makeQuery(overrides = {}) {
  return {
    query_id: 'sl-001',
    query: 'how does NEAT crossover work',
    class: 'simple_lookup',
    difficulty: 'easy',
    expected_doc_families: ['readme'],
    expected_heading_contains: 'crossover',
    expected_symbol_contains: null,
    expected_chunk_ids: [],
    relevance_grades: [],
    notes: '',
    ...overrides,
  };
}

describe('eval-runner', () => {
  beforeEach(() => {
    // ONNX Runtime is incompatible with the Jest VM-modules runner in this
    // repository, so the runner tests use the documented synthetic-search
    // fallback to keep the evaluation contract tests deterministic.
    process.env.EVAL_FORCE_SYNTHETIC = '1';
  });

  afterEach(() => {
    delete process.env.EVAL_FORCE_SYNTHETIC;
  });

  describe('runner entrypoint', () => {
    it('accepts a query set and a baseline condition', async () => {
      const { runEval } = await loadRunner();
      const queries = [makeQuery()];
      const result = await runEval({
        queries,
        condition: 'bm25_only',
      });
      expect(result).toEqual(
        expect.objectContaining({
          condition: 'bm25_only',
          query_count: 1,
          metrics: expect.any(Object),
        }),
      );
    });

    it('executes queries and collects per-query results', async () => {
      const { runEval } = await loadRunner();
      const queries = [makeQuery(), makeQuery({ query_id: 'sl-002' })];
      const result = await runEval({
        queries,
        condition: 'hybrid',
      });
      expect(result.per_query).toHaveLength(2);
      expect(result.per_query[0]).toEqual(
        expect.objectContaining({
          query_id: expect.any(String),
          latency_ms: expect.any(Number),
        }),
      );
    });

    it('computes metrics across all four baseline conditions', async () => {
      const { runEval } = await loadRunner();
      const queries = [makeQuery()];
      const conditions = [
        'bm25_only',
        'hybrid',
        'hybrid_rerank',
        'advanced_default',
      ];
      const results = await Promise.all(
        conditions.map((condition) => runEval({ queries, condition })),
      );
      expect(results).toHaveLength(4);
      results.forEach((result) => {
        expect(result.metrics).toEqual(
          expect.objectContaining({
            mrr_at_5: expect.any(Number),
            ndcg_at_5: expect.any(Number),
            recall_at_5: expect.any(Number),
          }),
        );
      });
    });

    it('supports a self-test with a small query set', async () => {
      const { runSelfTest } = await loadRunner();
      const result = await runSelfTest();
      expect(result).toEqual(
        expect.objectContaining({
          query_count: expect.any(Number),
          conditions: expect.any(Array),
          pass: expect.any(Boolean),
        }),
      );
    });

    it('rejects an invalid query schema', async () => {
      const { runEval } = await loadRunner();
      await expect(
        runEval({
          queries: [{ invalid: true }],
          condition: 'bm25_only',
        }),
      ).rejects.toThrow();
    });

    it('rejects an unsupported condition', async () => {
      const { runEval } = await loadRunner();
      await expect(
        runEval({
          queries: [makeQuery()],
          condition: 'unsupported_condition',
        }),
      ).rejects.toThrow();
    });
  });

  describe('metric aggregation', () => {
    it('computes MRR@k for k = 1, 3, 5, 10', async () => {
      const { runEval } = await loadRunner();
      const result = await runEval({
        queries: [makeQuery()],
        condition: 'hybrid',
      });
      expect(result.metrics).toEqual(
        expect.objectContaining({
          mrr_at_1: expect.any(Number),
          mrr_at_3: expect.any(Number),
          mrr_at_5: expect.any(Number),
          mrr_at_10: expect.any(Number),
        }),
      );
    });

    it('computes nDCG@k for k = 5, 10', async () => {
      const { runEval } = await loadRunner();
      const result = await runEval({
        queries: [makeQuery()],
        condition: 'hybrid',
      });
      expect(result.metrics).toEqual(
        expect.objectContaining({
          ndcg_at_5: expect.any(Number),
          ndcg_at_10: expect.any(Number),
        }),
      );
    });

    it('computes Recall@k for k = 5, 10, 20', async () => {
      const { runEval } = await loadRunner();
      const result = await runEval({
        queries: [makeQuery()],
        condition: 'hybrid',
      });
      expect(result.metrics).toEqual(
        expect.objectContaining({
          recall_at_5: expect.any(Number),
          recall_at_10: expect.any(Number),
          recall_at_20: expect.any(Number),
        }),
      );
    });

    it('reports latency percentiles', async () => {
      const { runEval } = await loadRunner();
      const result = await runEval({
        queries: [makeQuery()],
        condition: 'hybrid',
      });
      expect(result.metrics.latency_ms).toEqual(
        expect.objectContaining({
          p50: expect.any(Number),
          p95: expect.any(Number),
          max: expect.any(Number),
        }),
      );
    });

    it('groups metrics by query class', async () => {
      const { runEval } = await loadRunner();
      const result = await runEval({
        queries: [
          makeQuery({ class: 'simple_lookup' }),
          makeQuery({ class: 'cross_boundary' }),
        ],
        condition: 'hybrid',
      });
      expect(result.per_class).toEqual(
        expect.objectContaining({
          simple_lookup: expect.any(Object),
          cross_boundary: expect.any(Object),
        }),
      );
    });
  });

  describe('A/B comparison integration', () => {
    it('compares two result sets', async () => {
      const { compareResults } = await loadCompare();
      const resultA = {
        condition: 'hybrid',
        query_count: 2,
        metrics: { mrr_at_5: 0.3 },
        per_query: [
          { query_id: 'sl-001', mrr_at_5: 0.5 },
          { query_id: 'sl-002', mrr_at_5: 0.1 },
        ],
      };
      const resultB = {
        condition: 'hybrid_rerank',
        query_count: 2,
        metrics: { mrr_at_5: 0.4 },
        per_query: [
          { query_id: 'sl-001', mrr_at_5: 0.6 },
          { query_id: 'sl-002', mrr_at_5: 0.2 },
        ],
      };
      const comparison = compareResults(resultA, resultB);
      expect(comparison).toEqual(
        expect.objectContaining({
          condition_a: 'hybrid',
          condition_b: 'hybrid_rerank',
          metrics: expect.any(Object),
        }),
      );
    });

    it('returns a p-value from the Wilcoxon signed-rank test', async () => {
      const { wilcoxonSignedRankTest } = await loadCompare();
      const seriesA = [0.1, 0.2, 0.3, 0.4];
      const seriesB = [0.2, 0.3, 0.4, 0.5];
      const stats = wilcoxonSignedRankTest(seriesA, seriesB);
      expect(stats).toEqual(
        expect.objectContaining({
          p_value: expect.any(Number),
          significant: expect.any(Boolean),
        }),
      );
    });

    it('produces valid statistical results from alpha sweep', async () => {
      const { alphaSweep } = await loadCompare();
      const queries = [makeQuery(), makeQuery({ query_id: 'sl-002' })];
      const sweep = await alphaSweep({
        queries,
        alphas: [0.0, 0.5, 1.0],
      });
      expect(sweep).toBeInstanceOf(Array);
      expect(sweep[0]).toEqual(
        expect.objectContaining({
          alpha: expect.any(Number),
          mrr_at_5: expect.any(Number),
        }),
      );
    });

    it('handles invalid comparison inputs', async () => {
      const { compareResults } = await loadCompare();
      expect(() => compareResults(null, {})).toThrow();
    });
  });
});
