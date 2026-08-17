/**
 * @module eval-metrics.test
 * @description Comprehensive tests for eval-metrics.mjs targeting 100% statement,
 * branch, and function coverage.
 */

import { jest } from '@jest/globals';
import {
  matchesQueryExpectation,
  computeMrr,
  computeNdcg,
  computeRecall,
  computeContextRelevance,
  measureLatency,
  computeLatency,
  aggregateLatency,
  aggregateMetrics,
  aggregateByClass,
} from './eval-metrics.mjs';

// ---------------------------------------------------------------------------
// matchesQueryExpectation
// ---------------------------------------------------------------------------

describe('matchesQueryExpectation', () => {
  it('matches when family is in expected list and heading contains needle', () => {
    const result = { family: 'ts-source', heading_path: 'Network activate' };
    const querySpec = {
      expected_doc_families: ['ts-source'],
      expected_heading_contains: 'activate',
    };
    expect(matchesQueryExpectation(result, querySpec)).toBe(true);
  });

  it('does not match when family is not in expected list', () => {
    const result = { family: 'readme', heading_path: 'Network activate' };
    const querySpec = {
      expected_doc_families: ['ts-source'],
      expected_heading_contains: 'activate',
    };
    expect(matchesQueryExpectation(result, querySpec)).toBe(false);
  });

  it('matches any family when expected_doc_families is empty', () => {
    const result = { family: 'readme', heading_path: 'some heading' };
    const querySpec = { expected_doc_families: [] };
    expect(matchesQueryExpectation(result, querySpec)).toBe(true);
  });

  it('matches any family when expected_doc_families is absent', () => {
    const result = { family: 'readme', heading_path: 'some heading' };
    expect(matchesQueryExpectation(result, {})).toBe(true);
  });

  it('falls back to doc_family when family is absent', () => {
    const result = { doc_family: 'ts-source', heading_path: 'Network activate' };
    const querySpec = {
      expected_doc_families: ['ts-source'],
      expected_heading_contains: 'activate',
    };
    expect(matchesQueryExpectation(result, querySpec)).toBe(true);
  });

  it('returns null family when neither family nor doc_family present', () => {
    const result = { heading_path: 'test' };
    const querySpec = { expected_doc_families: ['ts-source'] };
    expect(matchesQueryExpectation(result, querySpec)).toBe(false);
  });

  it('matches on symbol_name when heading_path is absent', () => {
    const result = { family: 'ts-source', symbol_name: 'activateFunction' };
    const querySpec = {
      expected_doc_families: ['ts-source'],
      expected_heading_contains: 'activate',
    };
    expect(matchesQueryExpectation(result, querySpec)).toBe(true);
  });

  it('matches on symbol needle alone', () => {
    const result = { family: 'ts-source', symbol_name: 'mySymbol' };
    const querySpec = {
      expected_doc_families: ['ts-source'],
      expected_symbol_contains: 'mySymbol',
    };
    expect(matchesQueryExpectation(result, querySpec)).toBe(true);
  });

  it('does not match on symbol needle when absent', () => {
    const result = { family: 'ts-source', symbol_name: 'otherSymbol' };
    const querySpec = {
      expected_doc_families: ['ts-source'],
      expected_symbol_contains: 'mySymbol',
    };
    expect(matchesQueryExpectation(result, querySpec)).toBe(false);
  });

  it('returns true when both heading and symbol needle match (either sufficient)', () => {
    const result = { family: 'ts-source', heading_path: 'activate', symbol_name: 'other' };
    const querySpec = {
      expected_doc_families: ['ts-source'],
      expected_heading_contains: 'activate',
      expected_symbol_contains: 'mySymbol',
    };
    expect(matchesQueryExpectation(result, querySpec)).toBe(true);
  });

  it('returns true when symbol matches but heading does not (both needles set)', () => {
    const result = { family: 'ts-source', heading_path: 'other', symbol_name: 'mySymbol' };
    const querySpec = {
      expected_doc_families: ['ts-source'],
      expected_heading_contains: 'activate',
      expected_symbol_contains: 'mySymbol',
    };
    expect(matchesQueryExpectation(result, querySpec)).toBe(true);
  });

  it('returns false when neither heading nor symbol matches (both needles set)', () => {
    const result = { family: 'ts-source', heading_path: 'other', symbol_name: 'other' };
    const querySpec = {
      expected_doc_families: ['ts-source'],
      expected_heading_contains: 'activate',
      expected_symbol_contains: 'mySymbol',
    };
    expect(matchesQueryExpectation(result, querySpec)).toBe(false);
  });

  it('handles null/undefined querySpec gracefully', () => {
    const result = { family: 'ts-source', heading_path: 'test' };
    expect(matchesQueryExpectation(result, null)).toBe(true);
    expect(matchesQueryExpectation(result, undefined)).toBe(true);
  });

  it('handles null/undefined needles (normalize to empty string)', () => {
    const result = { family: 'ts-source', heading_path: 'test' };
    const querySpec = {
      expected_doc_families: ['ts-source'],
      expected_heading_contains: null,
      expected_symbol_contains: undefined,
    };
    expect(matchesQueryExpectation(result, querySpec)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// computeMrr
// ---------------------------------------------------------------------------

describe('computeMrr', () => {
  const querySpec = {
    expected_doc_families: ['ts-source'],
    expected_heading_contains: 'activate',
  };

  it('returns reciprocal rank of first relevant result', () => {
    const results = [
      { family: 'readme', heading_path: 'other' },
      { family: 'ts-source', heading_path: 'activate' },
    ];
    expect(computeMrr(results, querySpec, 5)).toBe(0.5);
  });

  it('returns 1 when first result is relevant', () => {
    const results = [{ family: 'ts-source', heading_path: 'activate' }];
    expect(computeMrr(results, querySpec, 5)).toBe(1);
  });

  it('returns 0 when no relevant result in top-k', () => {
    const results = [{ family: 'readme', heading_path: 'other' }];
    expect(computeMrr(results, querySpec, 5)).toBe(0);
  });

  it('returns 0 when results array is empty', () => {
    expect(computeMrr([], querySpec, 5)).toBe(0);
  });

  it('throws when results is not an array', () => {
    expect(() => computeMrr('notarray', querySpec, 5)).toThrow('results must be an array.');
  });

  it('throws when querySpec is undefined', () => {
    expect(() => computeMrr([], undefined, 5)).toThrow('querySpec is required.');
  });

  it('throws when querySpec is null', () => {
    expect(() => computeMrr([], null, 5)).toThrow('querySpec is required.');
  });

  it('throws for invalid k (non-integer)', () => {
    expect(() => computeMrr([], querySpec, 2.5)).toThrow('MRR k must be one of');
  });

  it('throws for invalid k (zero)', () => {
    expect(() => computeMrr([], querySpec, 0)).toThrow('MRR k must be one of');
  });

  it('throws for invalid k (not in allowed set)', () => {
    expect(() => computeMrr([], querySpec, 7)).toThrow('MRR k must be one of');
  });

  it('respects the k cutoff', () => {
    const results = [
      { family: 'readme', heading_path: 'other' },
      { family: 'readme', heading_path: 'other' },
      { family: 'ts-source', heading_path: 'activate' },
    ];
    expect(computeMrr(results, querySpec, 1)).toBe(0);
    expect(computeMrr(results, querySpec, 3)).toBe(1 / 3);
  });

  it('works with k=10', () => {
    const results = Array(9).fill({ family: 'readme', heading_path: 'other' });
    results.push({ family: 'ts-source', heading_path: 'activate' });
    expect(computeMrr(results, querySpec, 10)).toBe(0.1);
  });
});

// ---------------------------------------------------------------------------
// computeNdcg
// ---------------------------------------------------------------------------

describe('computeNdcg', () => {
  it('returns 0 when all grades are zero', () => {
    const results = [{ family: 'readme', heading_path: 'other' }];
    const querySpec = { expected_doc_families: ['ts-source'] };
    expect(computeNdcg(results, querySpec, 5)).toBe(0);
  });

  it('returns 1 when results are perfectly ordered by relevance', () => {
    const querySpec = {
      expected_doc_families: ['ts-source'],
      expected_heading_contains: 'activate',
      relevance_grades: [
        { family: 'ts-source', heading_path_contains: 'activate', grade: 3 },
      ],
    };
    const results = [
      { family: 'ts-source', heading_path: 'activate' },
      { family: 'readme', heading_path: 'other' },
    ];
    expect(computeNdcg(results, querySpec, 5)).toBeCloseTo(1, 5);
  });

  it('returns value between 0 and 1 for imperfect ordering', () => {
    const querySpec = {
      expected_doc_families: [],
      relevance_grades: [
        { family: '', heading_path_contains: 'good', grade: 3 },
        { family: '', heading_path_contains: 'bad', grade: 1 },
      ],
    };
    const results = [
      { family: 'readme', heading_path: 'bad result' },
      { family: 'readme', heading_path: 'good result' },
    ];
    const ndcg = computeNdcg(results, querySpec, 5);
    expect(ndcg).toBeGreaterThan(0);
    expect(ndcg).toBeLessThan(1);
  });

  it('uses binary relevance fallback when no relevance_grades match', () => {
    const querySpec = {
      expected_doc_families: ['ts-source'],
      expected_heading_contains: 'activate',
    };
    const results = [
      { family: 'ts-source', heading_path: 'activate' },
      { family: 'readme', heading_path: 'other' },
    ];
    const ndcg = computeNdcg(results, querySpec, 5);
    expect(ndcg).toBeGreaterThan(0);
  });

  it('handles grade rule with empty family (matches any family)', () => {
    const querySpec = {
      expected_doc_families: [],
      relevance_grades: [
        { family: '', heading_path_contains: '', grade: 2 },
      ],
    };
    const results = [{ family: 'anything', heading_path: 'anything' }];
    const ndcg = computeNdcg(results, querySpec, 5);
    expect(ndcg).toBe(1);
  });

  it('handles grade rule with specific family match', () => {
    const querySpec = {
      expected_doc_families: [],
      relevance_grades: [
        { family: 'ts-source', heading_path_contains: '', grade: 3 },
      ],
    };
    const results = [{ family: 'ts-source', heading_path: 'anything' }];
    expect(computeNdcg(results, querySpec, 5)).toBe(1);
  });

  it('handles grade rule with specific family non-match (falls through)', () => {
    const querySpec = {
      expected_doc_families: ['ts-source'],
      relevance_grades: [
        { family: 'ts-source', heading_path_contains: '', grade: 3 },
      ],
    };
    const results = [{ family: 'readme', heading_path: 'anything' }];
    // No grade rule matches (family readme != ts-source),
    // binary relevance also fails (family readme not in ['ts-source']) → 0
    expect(computeNdcg(results, querySpec, 5)).toBe(0);
  });

  it('handles non-finite grade by returning 0', () => {
    const querySpec = {
      expected_doc_families: [],
      relevance_grades: [
        { family: '', heading_path_contains: '', grade: NaN },
      ],
    };
    const results = [{ family: 'anything', heading_path: 'anything' }];
    // NaN grade → Number.isFinite is false → grade 0 → all zeros → nDCG 0
    expect(computeNdcg(results, querySpec, 5)).toBe(0);
  });

  it('handles grade rule with heading_path_contains match', () => {
    const querySpec = {
      expected_doc_families: [],
      relevance_grades: [
        { family: '', heading_path_contains: 'activate', grade: 3 },
      ],
    };
    const results = [{ family: 'anything', heading_path: 'Network activate' }];
    expect(computeNdcg(results, querySpec, 5)).toBe(1);
  });

  it('handles grade rule with heading_path_contains non-match', () => {
    const querySpec = {
      expected_doc_families: [],
      expected_heading_contains: 'nonexistent',
      relevance_grades: [
        { family: '', heading_path_contains: 'nonexistent', grade: 3 },
      ],
    };
    const results = [{ family: 'anything', heading_path: 'Network activate' }];
    // Rule doesn't match (heading has 'nonexistent' needle but result has 'activate'),
    // binary relevance also fails (expected_heading_contains 'nonexistent' not found) → 0
    expect(computeNdcg(results, querySpec, 5)).toBe(0);
  });

  it('clamps grade to [0, 3] range', () => {
    const querySpec = {
      expected_doc_families: [],
      relevance_grades: [
        { family: '', heading_path_contains: '', grade: 10 },
      ],
    };
    const results = [{ family: 'anything', heading_path: 'anything' }];
    // Grade 10 clamped to 3
    expect(computeNdcg(results, querySpec, 5)).toBe(1);
  });

  it('clamps negative grade to 0', () => {
    const querySpec = {
      expected_doc_families: [],
      relevance_grades: [
        { family: '', heading_path_contains: '', grade: -5 },
      ],
    };
    const results = [{ family: 'anything', heading_path: 'anything' }];
    // Grade -5 clamped to 0 → all zeros → nDCG 0
    expect(computeNdcg(results, querySpec, 5)).toBe(0);
  });

  it('uses default grade 0 when grade field is absent', () => {
    const querySpec = {
      expected_doc_families: [],
      relevance_grades: [{ family: '', heading_path_contains: '' }],
    };
    const results = [{ family: 'anything', heading_path: 'anything' }];
    expect(computeNdcg(results, querySpec, 5)).toBe(0);
  });

  it('uses doc_family fallback in gradeResult', () => {
    const querySpec = {
      expected_doc_families: [],
      relevance_grades: [
        { family: 'ts-source', heading_path_contains: '', grade: 3 },
      ],
    };
    const results = [{ doc_family: 'ts-source', heading_path: 'anything' }];
    expect(computeNdcg(results, querySpec, 5)).toBe(1);
  });

  it('throws when results is not an array', () => {
    expect(() => computeNdcg('nope', {}, 5)).toThrow('results must be an array.');
  });

  it('throws when querySpec is undefined', () => {
    expect(() => computeNdcg([], undefined, 5)).toThrow('querySpec is required.');
  });

  it('throws when querySpec is null', () => {
    expect(() => computeNdcg([], null, 5)).toThrow('querySpec is required.');
  });

  it('throws for invalid k', () => {
    expect(() => computeNdcg([], {}, 3)).toThrow('nDCG k must be one of');
  });

  it('throws for invalid k (non-integer)', () => {
    expect(() => computeNdcg([], {}, 5.5)).toThrow('nDCG k must be one of');
  });

  it('works with k=10', () => {
    const querySpec = {
      expected_doc_families: ['ts-source'],
      expected_heading_contains: 'activate',
    };
    const results = Array(10).fill({ family: 'ts-source', heading_path: 'activate' });
    expect(computeNdcg(results, querySpec, 10)).toBeCloseTo(1, 5);
  });
});

// ---------------------------------------------------------------------------
// computeRecall
// ---------------------------------------------------------------------------

describe('computeRecall', () => {
  it('computes recall using expected_chunk_ids when provided', () => {
    const results = [
      { chunk_id: 1, family: 'ts-source' },
      { chunk_id: 2, family: 'ts-source' },
      { chunk_id: 3, family: 'readme' },
    ];
    const querySpec = { expected_chunk_ids: [1, 2, 4] };
    expect(computeRecall(results, querySpec, 5)).toBeCloseTo(2 / 3, 5);
  });

  it('caps recall at 1.0', () => {
    const results = [
      { chunk_id: 1, family: 'ts-source' },
      { chunk_id: 2, family: 'ts-source' },
    ];
    const querySpec = { expected_chunk_ids: [1, 2] };
    expect(computeRecall(results, querySpec, 5)).toBe(1);
  });

  it('computes recall using family+heading when no expected_chunk_ids', () => {
    const results = [
      { family: 'ts-source', heading_path: 'activate' },
      { family: 'ts-source', heading_path: 'activate' },
      { family: 'readme', heading_path: 'other' },
    ];
    const querySpec = {
      expected_doc_families: ['ts-source'],
      expected_heading_contains: 'activate',
    };
    // 2 relevant in top-5, 2 relevant in total → recall 1.0
    expect(computeRecall(results, querySpec, 5)).toBe(1);
  });

  it('returns 0 when relevantPool is 0 (no matches in full set)', () => {
    const results = [{ family: 'readme', heading_path: 'other' }];
    const querySpec = {
      expected_doc_families: ['ts-source'],
      expected_heading_contains: 'activate',
    };
    expect(computeRecall(results, querySpec, 5)).toBe(0);
  });

  it('returns 0 when expected_chunk_ids is empty and no family matches', () => {
    const results = [{ family: 'readme', heading_path: 'other' }];
    const querySpec = { expected_doc_families: ['ts-source'] };
    expect(computeRecall(results, querySpec, 5)).toBe(0);
  });

  it('returns 0 when results is empty', () => {
    expect(computeRecall([], { expected_chunk_ids: [1] }, 5)).toBe(0);
  });

  it('throws when results is not an array', () => {
    expect(() => computeRecall('nope', {}, 5)).toThrow('results must be an array.');
  });

  it('throws when querySpec is undefined', () => {
    expect(() => computeRecall([], undefined, 5)).toThrow('querySpec is required.');
  });

  it('throws when querySpec is null', () => {
    expect(() => computeRecall([], null, 5)).toThrow('querySpec is required.');
  });

  it('throws for invalid k', () => {
    expect(() => computeRecall([], {}, 7)).toThrow('Recall k must be one of');
  });

  it('throws for invalid k (non-integer)', () => {
    expect(() => computeRecall([], {}, 5.5)).toThrow('Recall k must be one of');
  });

  it('works with k=20', () => {
    const results = Array(20).fill({ chunk_id: 1, family: 'ts-source', heading_path: 'activate' });
    const querySpec = {
      expected_doc_families: ['ts-source'],
      expected_heading_contains: 'activate',
    };
    expect(computeRecall(results, querySpec, 20)).toBe(1);
  });

  it('works with k=10', () => {
    const results = Array(10).fill({ chunk_id: 1, family: 'ts-source', heading_path: 'activate' });
    const querySpec = {
      expected_doc_families: ['ts-source'],
      expected_heading_contains: 'activate',
    };
    expect(computeRecall(results, querySpec, 10)).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// computeContextRelevance
// ---------------------------------------------------------------------------

describe('computeContextRelevance', () => {
  it('returns fraction of relevant chunks', () => {
    const chunks = [
      { family: 'ts-source', heading_path: 'activate' },
      { family: 'readme', heading_path: 'other' },
    ];
    const querySpec = {
      expected_doc_families: ['ts-source'],
      expected_heading_contains: 'activate',
    };
    expect(computeContextRelevance(chunks, querySpec)).toBe(0.5);
  });

  it('returns 1 when all chunks are relevant', () => {
    const chunks = [{ family: 'ts-source', heading_path: 'activate' }];
    const querySpec = {
      expected_doc_families: ['ts-source'],
      expected_heading_contains: 'activate',
    };
    expect(computeContextRelevance(chunks, querySpec)).toBe(1);
  });

  it('returns 0 when assembledChunks is empty', () => {
    expect(computeContextRelevance([], {})).toBe(0);
  });

  it('throws when assembledChunks is not an array', () => {
    expect(() => computeContextRelevance('nope', {})).toThrow('assembledChunks must be an array.');
  });

  it('throws when querySpec is undefined', () => {
    expect(() => computeContextRelevance([], undefined)).toThrow('querySpec is required.');
  });

  it('throws when querySpec is null', () => {
    expect(() => computeContextRelevance([], null)).toThrow('querySpec is required.');
  });
});

// ---------------------------------------------------------------------------
// measureLatency
// ---------------------------------------------------------------------------

describe('measureLatency', () => {
  it('measures latency of an async function', async () => {
    const { result, latency_ms } = await measureLatency(async () => 42);
    expect(result).toBe(42);
    expect(latency_ms).toBeGreaterThanOrEqual(0);
  });

  it('throws when asyncFn is not a function', async () => {
    await expect(measureLatency('not a function')).rejects.toThrow(
      'measureLatency requires a function.',
    );
  });

  it('throws when asyncFn is undefined', async () => {
    await expect(measureLatency(undefined)).rejects.toThrow(
      'measureLatency requires a function.',
    );
  });
});

// ---------------------------------------------------------------------------
// computeLatency
// ---------------------------------------------------------------------------

describe('computeLatency', () => {
  it('computes difference between end and start', () => {
    expect(computeLatency(100, 250)).toBe(150);
  });

  it('returns 0 when end is before start (clamped)', () => {
    expect(computeLatency(200, 100)).toBe(0);
  });

  it('handles null endMs (defaults to 0)', () => {
    expect(computeLatency(100, null)).toBe(0);
  });

  it('handles undefined endMs (defaults to 0)', () => {
    expect(computeLatency(100, undefined)).toBe(0);
  });

  it('handles null startMs (defaults to 0)', () => {
    expect(computeLatency(null, 100)).toBe(100);
  });

  it('handles undefined startMs (defaults to 0)', () => {
    expect(computeLatency(undefined, 100)).toBe(100);
  });

  it('handles both null', () => {
    expect(computeLatency(null, null)).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// aggregateLatency
// ---------------------------------------------------------------------------

describe('aggregateLatency', () => {
  it('aggregates p50, p95, and max', () => {
    const latencies = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const result = aggregateLatency(latencies);
    expect(result.p50).toBeGreaterThan(0);
    expect(result.p95).toBeGreaterThan(0);
    expect(result.max).toBe(10);
  });

  it('returns zeros for empty array', () => {
    const result = aggregateLatency([]);
    expect(result.p50).toBe(0);
    expect(result.p95).toBe(0);
    expect(result.max).toBe(0);
  });

  it('returns zeros for non-array input', () => {
    const result = aggregateLatency(null);
    expect(result.p50).toBe(0);
    expect(result.p95).toBe(0);
    expect(result.max).toBe(0);
  });

  it('handles single-element array', () => {
    const result = aggregateLatency([42]);
    expect(result.p50).toBe(42);
    expect(result.p95).toBe(42);
    expect(result.max).toBe(42);
  });
});

// ---------------------------------------------------------------------------
// aggregateMetrics
// ---------------------------------------------------------------------------

describe('aggregateMetrics', () => {
  it('aggregates per-query rows into summary metrics', () => {
    const rows = [
      {
        mrr_at_1: 1, mrr_at_3: 1, mrr_at_5: 1, mrr_at_10: 1,
        ndcg_at_5: 1, ndcg_at_10: 1,
        recall_at_5: 1, recall_at_10: 1, recall_at_20: 1,
        latency_ms: 5,
      },
      {
        mrr_at_1: 0, mrr_at_3: 0, mrr_at_5: 0, mrr_at_10: 0,
        ndcg_at_5: 0, ndcg_at_10: 0,
        recall_at_5: 0, recall_at_10: 0, recall_at_20: 0,
        latency_ms: 10,
      },
    ];
    const result = aggregateMetrics(rows);
    expect(result.mrr_at_5).toBe(0.5);
    expect(result.ndcg_at_5).toBe(0.5);
    expect(result.recall_at_5).toBe(0.5);
    expect(result.zero_hit_queries).toBe(1);
    expect(result.latency_ms.p50).toBeGreaterThan(0);
  });

  it('returns zeros for empty array', () => {
    const result = aggregateMetrics([]);
    expect(result.mrr_at_5).toBe(0);
    expect(result.ndcg_at_5).toBe(0);
    expect(result.recall_at_5).toBe(0);
    expect(result.zero_hit_queries).toBe(0);
  });

  it('returns zeros for non-array input', () => {
    const result = aggregateMetrics(null);
    expect(result.mrr_at_5).toBe(0);
  });

  it('handles rows with missing fields (defaults to 0)', () => {
    const result = aggregateMetrics([{}]);
    expect(result.mrr_at_5).toBe(0);
    expect(result.zero_hit_queries).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// aggregateByClass
// ---------------------------------------------------------------------------

describe('aggregateByClass', () => {
  it('groups rows by class and aggregates', () => {
    const rows = [
      { class: 'simple_lookup', mrr_at_5: 1, latency_ms: 5 },
      { class: 'simple_lookup', mrr_at_5: 0, latency_ms: 10 },
      { class: 'cross_boundary', mrr_at_5: 0.5, latency_ms: 7 },
    ];
    const result = aggregateByClass(rows);
    expect(Object.keys(result)).toEqual(['cross_boundary', 'simple_lookup']);
    expect(result.simple_lookup.mrr_at_5).toBe(0.5);
    expect(result.cross_boundary.mrr_at_5).toBe(0.5);
  });

  it('assigns unknown class when class is absent', () => {
    const rows = [{ mrr_at_5: 1, latency_ms: 5 }];
    const result = aggregateByClass(rows);
    expect(result.unknown).toBeDefined();
    expect(result.unknown.mrr_at_5).toBe(1);
  });

  it('returns empty object for empty array', () => {
    expect(Object.keys(aggregateByClass([]))).toEqual([]);
  });

  it('returns empty object for non-array input', () => {
    expect(Object.keys(aggregateByClass(null))).toEqual([]);
  });
});