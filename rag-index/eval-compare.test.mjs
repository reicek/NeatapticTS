/**
 * @module eval-compare.test
 * @description Comprehensive tests for eval-compare.mjs targeting 100% coverage.
 */
import { jest } from '@jest/globals';

// Mock eval-runner so alphaSweep uses a controlled runEval.
const mockRunEval = jest.fn();
jest.unstable_mockModule('./eval-runner.mjs', () => ({ runEval: mockRunEval }));

const {
  normalCdf,
  wilcoxonSignedRankTest,
  compareResults,
  alphaSweep,
} = await import('./eval-compare.mjs');

// ---------------------------------------------------------------------------
// normalCdf
// ---------------------------------------------------------------------------

describe('normalCdf', () => {
  it('returns 0.5 for x=0', () => {
    expect(normalCdf(0)).toBeCloseTo(0.5, 4);
  });

  it('returns value close to 1 for large positive x', () => {
    expect(normalCdf(5)).toBeCloseTo(1, 4);
  });

  it('returns value close to 0 for large negative x', () => {
    expect(normalCdf(-5)).toBeCloseTo(0, 4);
  });

  it('returns >0.5 for positive x', () => {
    expect(normalCdf(1)).toBeGreaterThan(0.5);
  });

  it('returns <0.5 for negative x', () => {
    expect(normalCdf(-1)).toBeLessThan(0.5);
  });
});

// ---------------------------------------------------------------------------
// wilcoxonSignedRankTest
// ---------------------------------------------------------------------------

describe('wilcoxonSignedRankTest', () => {
  it('throws when seriesA is not an array', () => {
    expect(() => wilcoxonSignedRankTest('not array', [1, 2])).toThrow(
      'wilcoxonSignedRankTest requires two numeric arrays.',
    );
  });

  it('throws when seriesB is not an array', () => {
    expect(() => wilcoxonSignedRankTest([1, 2], 'not array')).toThrow(
      'wilcoxonSignedRankTest requires two numeric arrays.',
    );
  });

  it('throws on length mismatch', () => {
    expect(() => wilcoxonSignedRankTest([1, 2, 3], [1, 2])).toThrow(
      'Wilcoxon test requires paired samples of equal length.',
    );
  });

  it('returns p_value=1 for empty arrays', () => {
    const result = wilcoxonSignedRankTest([], []);
    expect(result.p_value).toBe(1);
    expect(result.significant).toBe(false);
    expect(result.statistic).toBe(0);
  });

  it('returns p_value=1 when all diffs are zero', () => {
    const result = wilcoxonSignedRankTest([1, 2, 3], [1, 2, 3]);
    expect(result.p_value).toBe(1);
    expect(result.significant).toBe(false);
    expect(result.statistic).toBe(0);
  });

  it('computes test for non-zero diffs without ties', () => {
    // diffs: 5, -7, 2, -6 → no ties, mixed positive/negative
    const result = wilcoxonSignedRankTest([10, 5, 30, 2], [5, 12, 28, 8]);
    expect(result.p_value).toBeGreaterThanOrEqual(0);
    expect(result.p_value).toBeLessThanOrEqual(1);
    expect(result.statistic).toBeGreaterThan(0);
  });

  it('handles ties in absolute differences', () => {
    // diffs: 2, 2, 5 → ties at abs=2
    const result = wilcoxonSignedRankTest([4, 7, 10], [2, 5, 5]);
    expect(result.p_value).toBeGreaterThanOrEqual(0);
    expect(result.p_value).toBeLessThanOrEqual(1);
  });

  it('handles negative diffs', () => {
    const result = wilcoxonSignedRankTest([1, 2, 3], [5, 10, 15]);
    expect(result.p_value).toBeGreaterThanOrEqual(0);
    expect(result.p_value).toBeLessThanOrEqual(1);
  });

  it('handles mixed positive and negative diffs with ties', () => {
    // diffs: 3, -1, 3, -1 → ties at abs=1 and abs=3
    const result = wilcoxonSignedRankTest([5, 1, 8, 2], [2, 2, 5, 3]);
    expect(result.p_value).toBeGreaterThanOrEqual(0);
    expect(result.p_value).toBeLessThanOrEqual(1);
  });

  it('clamps p_value to [0, 1]', () => {
    // Use a large enough sample to get a meaningful p-value
    const a = Array.from({ length: 20 }, (_, i) => i + 10);
    const b = Array.from({ length: 20 }, (_, i) => i);
    const result = wilcoxonSignedRankTest(a, b);
    expect(result.p_value).toBeGreaterThanOrEqual(0);
    expect(result.p_value).toBeLessThanOrEqual(1);
  });
});

// ---------------------------------------------------------------------------
// compareResults
// ---------------------------------------------------------------------------

describe('compareResults', () => {
  it('throws when resultA is null', () => {
    expect(() => compareResults(null, {})).toThrow('resultA is required.');
  });

  it('throws when resultA is not an object', () => {
    expect(() => compareResults('string', {})).toThrow('resultA is required.');
  });

  it('throws when resultB is null', () => {
    expect(() => compareResults({}, null)).toThrow('resultB is required.');
  });

  it('throws when resultB is not an object', () => {
    expect(() => compareResults({}, 'string')).toThrow('resultB is required.');
  });

  it('compares two results with metrics and per_query', () => {
    const resultA = {
      condition: 'bm25_only',
      metrics: { mrr_at_5: 0.3, ndcg_at_5: 0.4, recall_at_5: 0.5 },
      per_query: [
        { mrr_at_5: 0.2, ndcg_at_5: 0.3, recall_at_5: 0.4 },
        { mrr_at_5: 0.4, ndcg_at_5: 0.5, recall_at_5: 0.6 },
      ],
    };
    const resultB = {
      condition: 'hybrid',
      metrics: { mrr_at_5: 0.35, ndcg_at_5: 0.45, recall_at_5: 0.55 },
      per_query: [
        { mrr_at_5: 0.25, ndcg_at_5: 0.35, recall_at_5: 0.45 },
        { mrr_at_5: 0.45, ndcg_at_5: 0.55, recall_at_5: 0.65 },
      ],
    };
    const comparison = compareResults(resultA, resultB);
    expect(comparison.condition_a).toBe('bm25_only');
    expect(comparison.condition_b).toBe('hybrid');
    expect(comparison.query_count).toBe(2);
    expect(comparison.metrics.mrr_at_5.a).toBe(0.3);
    expect(comparison.metrics.mrr_at_5.b).toBe(0.35);
    expect(comparison.metrics.mrr_at_5.delta).toMatch(/^\+/);
    expect(comparison.per_query_wins).toEqual({
      a_wins: 0,
      b_wins: 2,
      ties: 0,
    });
  });

  it('handles missing per_query arrays', () => {
    const resultA = { condition: 'a', metrics: { mrr_at_5: 0.3 } };
    const resultB = { condition: 'b', metrics: { mrr_at_5: 0.4 } };
    const comparison = compareResults(resultA, resultB);
    expect(comparison.query_count).toBe(0);
    expect(comparison.per_query_wins).toEqual({
      a_wins: 0,
      b_wins: 0,
      ties: 0,
    });
  });

  it('handles per_query rows with undefined metrics', () => {
    const resultA = {
      condition: 'a',
      metrics: {},
      per_query: [{}, { mrr_at_5: 0.5 }],
    };
    const resultB = {
      condition: 'b',
      metrics: {},
      per_query: [{ mrr_at_5: 0.3 }, {}],
    };
    const comparison = compareResults(resultA, resultB);
    // First: a=0, b=0.3 → b wins; Second: a=0.5, b=0 → a wins
    expect(comparison.per_query_wins).toEqual({
      a_wins: 1,
      b_wins: 1,
      ties: 0,
    });
  });

  it('handles per_class delta', () => {
    const resultA = {
      condition: 'a',
      metrics: {},
      per_class: {
        simple_lookup: { mrr_at_5: 0.3 },
        cross_boundary: { mrr_at_5: 0.4 },
        multi_hop: { mrr_at_5: 0.5 },
      },
    };
    const resultB = {
      condition: 'b',
      metrics: {},
      per_class: {
        simple_lookup: { mrr_at_5: 0.35 },
        multi_hop: { mrr_at_5: 0.2 },
      },
    };
    const comparison = compareResults(resultA, resultB);
    expect(comparison.per_class_delta.simple_lookup.mrr_at_5_delta).toMatch(
      /^\+/,
    );
    expect(comparison.per_class_delta.cross_boundary.mrr_at_5_delta).toMatch(
      /^-/,
    );
    expect(comparison.per_class_delta.multi_hop.mrr_at_5_delta).toMatch(
      /^-/,
    );
  });

  it('handles null per_class', () => {
    const resultA = { condition: 'a', metrics: {}, per_class: null };
    const resultB = { condition: 'b', metrics: {}, per_class: null };
    const comparison = compareResults(resultA, resultB);
    expect(comparison.per_class_delta).toEqual({});
  });

  it('handles undefined per_class', () => {
    const resultA = { condition: 'a', metrics: {} };
    const resultB = { condition: 'b', metrics: {} };
    const comparison = compareResults(resultA, resultB);
    expect(comparison.per_class_delta).toEqual({});
  });

  it('generates a comparison_id', () => {
    const resultA = { condition: 'a', metrics: {} };
    const resultB = { condition: 'b', metrics: {} };
    const comparison = compareResults(resultA, resultB);
    expect(comparison.comparison_id).toMatch(/^comparison-/);
  });

  it('handles ties in per_query_wins', () => {
    const resultA = {
      condition: 'a',
      metrics: {},
      per_query: [{ mrr_at_5: 0.5 }],
    };
    const resultB = {
      condition: 'b',
      metrics: {},
      per_query: [{ mrr_at_5: 0.5 }],
    };
    const comparison = compareResults(resultA, resultB);
    expect(comparison.per_query_wins.ties).toBe(1);
  });

  it('handles per_query arrays with same length but different content', () => {
    const resultA = {
      condition: 'a',
      metrics: {},
      per_query: [{ mrr_at_5: 0.5 }, { mrr_at_5: 0.3 }],
    };
    const resultB = {
      condition: 'b',
      metrics: {},
      per_query: [{ mrr_at_5: 0.4 }, { mrr_at_5: 0.3 }],
    };
    const comparison = compareResults(resultA, resultB);
    expect(comparison.query_count).toBe(2);
    expect(comparison.per_query_wins.a_wins).toBe(1);
    expect(comparison.per_query_wins.ties).toBe(1);
  });

  it('handles per_class entries with undefined mrr_at_5', () => {
    const resultA = {
      condition: 'a',
      metrics: {},
      per_class: { class_a: {} },
    };
    const resultB = {
      condition: 'b',
      metrics: {},
      per_class: { class_a: {} },
    };
    const comparison = compareResults(resultA, resultB);
    expect(comparison.per_class_delta.class_a.mrr_at_5_delta).toBe('+0.000');
  });
});

// ---------------------------------------------------------------------------
// alphaSweep
// ---------------------------------------------------------------------------

describe('alphaSweep', () => {
  beforeEach(() => {
    mockRunEval.mockReset();
  });

  it('returns empty array when no alphas', async () => {
    const result = await alphaSweep({ queries: [], alphas: [] });
    expect(result).toEqual([]);
  });

  it('returns empty array when alphas is not an array', async () => {
    const result = await alphaSweep({ queries: [] });
    expect(result).toEqual([]);
  });

  it('sweeps across alpha values', async () => {
    mockRunEval.mockResolvedValue({
      metrics: {
        mrr_at_5: 0.3,
        ndcg_at_5: 0.4,
        recall_at_5: 0.5,
        latency_ms: { p50: 10 },
      },
    });
    const result = await alphaSweep({
      queries: [{ query_id: 'q1', query: 'test', class: 'simple_lookup', difficulty: 'easy', expected_doc_families: [] }],
      alphas: [0, 0.5, 1],
    });
    expect(result).toHaveLength(3);
    expect(result[0].alpha).toBe(0);
    expect(result[0].mrr_at_5).toBe(0.3);
    expect(mockRunEval).toHaveBeenCalledTimes(3);
  });

  it('uses default condition when not specified', async () => {
    mockRunEval.mockResolvedValue({
      metrics: { mrr_at_5: 0, ndcg_at_5: 0, recall_at_5: 0, latency_ms: 0 },
    });
    await alphaSweep({ alphas: [0.5] });
    expect(mockRunEval).toHaveBeenCalledWith({
      queries: [],
      condition: 'hybrid',
      alpha: 0.5,
    });
  });

  it('uses custom condition when specified', async () => {
    mockRunEval.mockResolvedValue({
      metrics: { mrr_at_5: 0, ndcg_at_5: 0, recall_at_5: 0, latency_ms: 0 },
    });
    await alphaSweep({ alphas: [0.5], condition: 'bm25_only' });
    expect(mockRunEval).toHaveBeenCalledWith({
      queries: [],
      condition: 'bm25_only',
      alpha: 0.5,
    });
  });

  it('handles non-array queries', async () => {
    mockRunEval.mockResolvedValue({
      metrics: { mrr_at_5: 0, ndcg_at_5: 0, recall_at_5: 0, latency_ms: 0 },
    });
    await alphaSweep({ alphas: [0.5], queries: 'not array' });
    expect(mockRunEval).toHaveBeenCalledWith({
      queries: [],
      condition: 'hybrid',
      alpha: 0.5,
    });
  });
});