/**
 * @module parallel-search.test
 * @description 100% coverage tests for parallel-search.mjs.
 */

import { jest } from '@jest/globals';

const { runParallelQueries } = await import('./parallel-search.mjs');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function createMockClient(results) {
  return {
    async execute(query) {
      const idx = query._idx;
      if (results[idx]?.error) throw new Error(results[idx].error);
      return { rows: results[idx]?.rows ?? [] };
    },
  };
}

// ---------------------------------------------------------------------------
// runParallelQueries: empty / invalid queries
// ---------------------------------------------------------------------------

describe('parallel-search: runParallelQueries empty/invalid', () => {
  it('returns empty array with errors when queries is empty', async () => {
    const client = createMockClient([]);
    const result = await runParallelQueries({ client, queries: [] });
    expect(result).toEqual([]);
    expect(result.errors).toEqual([]);
  });

  it('returns empty array with errors when queries is not an array', async () => {
    const client = createMockClient([]);
    const result = await runParallelQueries({ client, queries: null });
    expect(result).toEqual([]);
    expect(result.errors).toEqual([]);
  });

  it('returns empty array with errors when queries is undefined', async () => {
    const client = createMockClient([]);
    const result = await runParallelQueries({ client });
    expect(result).toEqual([]);
    expect(result.errors).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// runParallelQueries: all queries succeed (RRF default)
// ---------------------------------------------------------------------------

describe('parallel-search: runParallelQueries RRF fusion', () => {
  it('merges multiple result lists with RRF and sorts by rrf_score', async () => {
    const client = {
      async execute(query) {
        if (query.sql === 'q1') {
          return {
            rows: [
              { chunk_id: 1, score: 5 },
              { chunk_id: 2, score: 3 },
            ],
          };
        }
        return {
          rows: [
            { chunk_id: 2, score: 8 },
            { chunk_id: 3, score: 1 },
          ],
        };
      },
    };
    const result = await runParallelQueries({
      client,
      queries: [{ sql: 'q1', args: [] }, { sql: 'q2', args: [] }],
    });
    // chunk 2 appears in both lists (rank 0 in q2, rank 1 in q1) → highest RRF
    expect(result[0].chunk_id).toBe(2);
    expect(result[0].rrf_score).toBeDefined();
    expect(result.errors).toEqual([]);
  });

  it('respects custom k parameter for RRF', async () => {
    const client = {
      async execute() {
        return { rows: [{ chunk_id: 1, score: 5 }] };
      },
    };
    const result = await runParallelQueries({
      client,
      queries: [{ sql: 'q1', args: [] }],
      k: 30,
    });
    // Single list, sorted by score; chunk 1 at rank 0 → 1/(30+0)
    expect(result[0].chunk_id).toBe(1);
  });

  it('caps results to limit when provided', async () => {
    const client = {
      async execute() {
        return {
          rows: [
            { chunk_id: 1, score: 5 },
            { chunk_id: 2, score: 3 },
            { chunk_id: 3, score: 1 },
          ],
        };
      },
    };
    const result = await runParallelQueries({
      client,
      queries: [{ sql: 'q1', args: [] }],
      limit: 2,
    });
    expect(result.length).toBe(2);
  });

  it('does not cap when limit is not a positive integer', async () => {
    const client = {
      async execute() {
        return { rows: [{ chunk_id: 1, score: 5 }] };
      },
    };
    const result = await runParallelQueries({
      client,
      queries: [{ sql: 'q1', args: [] }],
      limit: 0,
    });
    expect(result.length).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// runParallelQueries: alpha fusion
// ---------------------------------------------------------------------------

describe('parallel-search: runParallelQueries alpha fusion', () => {
  it('merges multiple result lists with alpha-blend and sorts by alpha_score', async () => {
    const client = {
      async execute(query) {
        if (query.sql === 'q1') {
          return {
            rows: [
              { chunk_id: 1, score: 10 },
              { chunk_id: 2, score: 5 },
            ],
          };
        }
        return {
          rows: [
            { chunk_id: 2, score: 8 },
            { chunk_id: 3, score: 4 },
          ],
        };
      },
    };
    const result = await runParallelQueries({
      client,
      queries: [{ sql: 'q1', args: [] }, { sql: 'q2', args: [] }],
      fusion: 'alpha',
    });
    // All results should have alpha_score
    expect(result.every((r) => r.alpha_score !== undefined)).toBe(true);
    // Sorted descending by alpha_score
    for (let i = 1; i < result.length; i += 1) {
      expect(result[i - 1].alpha_score).toBeGreaterThanOrEqual(
        result[i].alpha_score,
      );
    }
  });
});

// ---------------------------------------------------------------------------
// runParallelQueries: graceful degradation
// ---------------------------------------------------------------------------

describe('parallel-search: runParallelQueries graceful degradation', () => {
  it('records errors for failed queries and returns surviving results', async () => {
    const client = {
      async execute(query) {
        if (query.sql === 'fail') throw new Error('query failed');
        return { rows: [{ chunk_id: 1, score: 5 }] };
      },
    };
    const result = await runParallelQueries({
      client,
      queries: [
        { sql: 'ok', args: [] },
        { sql: 'fail', args: ['x'] },
      ],
    });
    expect(result.length).toBe(1);
    expect(result[0].chunk_id).toBe(1);
    expect(result.errors).toHaveLength(1);
    expect(result.errors[0].sql).toBe('fail');
    expect(result.errors[0].message).toBe('query failed');
  });

  it('returns empty array with errors when all queries fail', async () => {
    const client = {
      async execute() {
        throw new Error('all failed');
      },
    };
    const result = await runParallelQueries({
      client,
      queries: [{ sql: 'q1', args: [] }, { sql: 'q2', args: [] }],
    });
    expect(result).toEqual([]);
    expect(result.errors).toHaveLength(2);
  });
});

// ---------------------------------------------------------------------------
// runParallelQueries: single surviving list
// ---------------------------------------------------------------------------

describe('parallel-search: runParallelQueries single list', () => {
  it('sorts a single surviving list by score descending', async () => {
    const client = {
      async execute() {
        return {
          rows: [
            { chunk_id: 3, score: 1 },
            { chunk_id: 1, score: 5 },
            { chunk_id: 2, score: 3 },
          ],
        };
      },
    };
    const result = await runParallelQueries({
      client,
      queries: [{ sql: 'q1', args: [] }],
    });
    expect(result.map((r) => r.chunk_id)).toEqual([1, 2, 3]);
  });
});

// ---------------------------------------------------------------------------
// runParallelQueries: concurrency
// ---------------------------------------------------------------------------

describe('parallel-search: runParallelQueries concurrency', () => {
  const origEnv = { ...process.env };

  afterEach(() => {
    process.env = { ...origEnv };
  });

  it('uses TURSO_CONCURRENCY env var when set to a positive integer', async () => {
    process.env.TURSO_CONCURRENCY = '5';
    const client = {
      async execute() {
        return { rows: [{ chunk_id: 1, score: 1 }] };
      },
    };
    const result = await runParallelQueries({
      client,
      queries: [{ sql: 'q1', args: [] }],
    });
    expect(result.length).toBe(1);
  });

  it('falls back to default when TURSO_CONCURRENCY is invalid', async () => {
    process.env.TURSO_CONCURRENCY = 'not-a-number';
    const client = {
      async execute() {
        return { rows: [{ chunk_id: 1, score: 1 }] };
      },
    };
    const result = await runParallelQueries({
      client,
      queries: [{ sql: 'q1', args: [] }],
    });
    expect(result.length).toBe(1);
  });

  it('falls back to default when TURSO_CONCURRENCY is zero', async () => {
    process.env.TURSO_CONCURRENCY = '0';
    const client = {
      async execute() {
        return { rows: [{ chunk_id: 1, score: 1 }] };
      },
    };
    const result = await runParallelQueries({
      client,
      queries: [{ sql: 'q1', args: [] }],
    });
    expect(result.length).toBe(1);
  });

  it('falls back to default when TURSO_CONCURRENCY is negative', async () => {
    process.env.TURSO_CONCURRENCY = '-5';
    const client = {
      async execute() {
        return { rows: [{ chunk_id: 1, score: 1 }] };
      },
    };
    const result = await runParallelQueries({
      client,
      queries: [{ sql: 'q1', args: [] }],
    });
    expect(result.length).toBe(1);
  });

  it('falls back to default when TURSO_CONCURRENCY is unset', async () => {
    delete process.env.TURSO_CONCURRENCY;
    const client = {
      async execute() {
        return { rows: [{ chunk_id: 1, score: 1 }] };
      },
    };
    const result = await runParallelQueries({
      client,
      queries: [{ sql: 'q1', args: [] }],
    });
    expect(result.length).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// runParallelQueries: use_dense flag (no-op, just passes through)
// ---------------------------------------------------------------------------

describe('parallel-search: runParallelQueries use_dense', () => {
  it('accepts use_dense flag without error', async () => {
    const client = {
      async execute() {
        return { rows: [{ chunk_id: 1, score: 1 }] };
      },
    };
    const result = await runParallelQueries({
      client,
      queries: [{ sql: 'q1', args: [] }],
      use_dense: true,
    });
    expect(result.length).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// normalizeScores edge cases (via alpha-blend)
// ---------------------------------------------------------------------------

describe('parallel-search: alpha-blend edge cases', () => {
  it('handles empty result lists in alpha-blend', async () => {
    const client = {
      async execute() {
        return { rows: [] };
      },
    };
    const result = await runParallelQueries({
      client,
      queries: [{ sql: 'q1', args: [] }],
      fusion: 'alpha',
    });
    // Single surviving list with 0 rows → empty result
    expect(result).toEqual([]);
  });

  it('handles all-equal scores in alpha-blend (range=0 → normalized to 0)', async () => {
    const client = {
      async execute() {
        return {
          rows: [
            { chunk_id: 1, score: 5 },
            { chunk_id: 2, score: 5 },
          ],
        };
      },
    };
    const result = await runParallelQueries({
      client,
      queries: [{ sql: 'q1', args: [] }, { sql: 'q2', args: [] }],
      fusion: 'alpha',
    });
    expect(result.length).toBe(2);
    // All normalized to 0 → alpha_score = 0
    expect(result[0].alpha_score).toBe(0);
  });

  it('handles missing score field (defaults to 0)', async () => {
    const client = {
      async execute() {
        return {
          rows: [
            { chunk_id: 1 },
            { chunk_id: 2, score: 5 },
          ],
        };
      },
    };
    const result = await runParallelQueries({
      client,
      queries: [{ sql: 'q1', args: [] }],
      fusion: 'alpha',
    });
    expect(result.length).toBe(2);
    // chunk 2 has higher normalized score
    expect(result[0].chunk_id).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// RRF edge cases
// ---------------------------------------------------------------------------

describe('parallel-search: RRF edge cases', () => {
  it('handles missing score field in RRF (defaults to 0)', async () => {
    const client = {
      async execute() {
        return {
          rows: [{ chunk_id: 1 }, { chunk_id: 2 }],
        };
      },
    };
    const result = await runParallelQueries({
      client,
      queries: [{ sql: 'q1', args: [] }, { sql: 'q2', args: [] }],
    });
    expect(result.length).toBe(2);
    expect(result[0].rrf_score).toBeDefined();
  });

  it('handles chunk appearing in only one of multiple lists', async () => {
    const client = {
      async execute(query) {
        if (query.sql === 'q1') {
          return { rows: [{ chunk_id: 1, score: 5 }] };
        }
        return { rows: [{ chunk_id: 2, score: 3 }] };
      },
    };
    const result = await runParallelQueries({
      client,
      queries: [{ sql: 'q1', args: [] }, { sql: 'q2', args: [] }],
    });
    const chunkIds = result.map((r) => r.chunk_id).sort();
    expect(chunkIds).toEqual([1, 2]);
  });
});

// ---------------------------------------------------------------------------
// attachErrors: non-enumerable property
// ---------------------------------------------------------------------------

describe('parallel-search: errors property is non-enumerable', () => {
  it('errors array is non-enumerable on result', async () => {
    const client = {
      async execute() {
        return { rows: [{ chunk_id: 1, score: 1 }] };
      },
    };
    const result = await runParallelQueries({
      client,
      queries: [{ sql: 'q1', args: [] }],
    });
    expect(Object.keys(result)).not.toContain('errors');
    expect(result.errors).toEqual([]);
  });
});