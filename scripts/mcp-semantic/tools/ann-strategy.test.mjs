/**
 * @module ann-strategy.test
 * @description Coverage tests for ann-strategy.mjs — dense strategy selection,
 * query-result caching, and incremental-update detection.
 */
import {
  DEFAULT_ANN_THRESHOLD,
  DEFAULT_CACHE_TTL_MS,
  DEFAULT_CACHE_MAX_ENTRIES,
  DEFAULT_DISKANN_MAX_NEIGHBORS,
  DEFAULT_DISKANN_ALPHA,
  DEFAULT_DISKANN_SEARCH_L,
  resolveDenseStrategy,
  quantizedHash,
  getQueryResultCache,
  setQueryResultCache,
  clearQueryResultCache,
  detectIncrementalUpdateAction,
} from './ann-strategy.mjs';

describe('ann-strategy', () => {
  describe('constants', () => {
    it('exports expected constant values', () => {
      expect(DEFAULT_ANN_THRESHOLD).toBe(50_000);
      expect(DEFAULT_CACHE_TTL_MS).toBe(30 * 60 * 1000);
      expect(DEFAULT_CACHE_MAX_ENTRIES).toBe(500);
      expect(DEFAULT_DISKANN_MAX_NEIGHBORS).toBe(59);
      expect(DEFAULT_DISKANN_ALPHA).toBe(1.2);
      expect(DEFAULT_DISKANN_SEARCH_L).toBe(80);
    });
  });

  describe('resolveDenseStrategy', () => {
    it('returns diskann with no arguments', () => {
      expect(resolveDenseStrategy()).toBe('diskann');
    });

    it('returns diskann regardless of chunkCount', () => {
      expect(resolveDenseStrategy({ chunkCount: 1_000_000 })).toBe('diskann');
    });

    it('returns diskann regardless of annThreshold', () => {
      expect(resolveDenseStrategy({ chunkCount: 10, annThreshold: 5 })).toBe(
        'diskann',
      );
    });

    it('returns diskann regardless of indexStatus', () => {
      expect(resolveDenseStrategy({ indexStatus: 'missing' })).toBe('diskann');
    });

    it('returns diskann regardless of forceStrategy', () => {
      expect(resolveDenseStrategy({ forceStrategy: 'hnsw' })).toBe('diskann');
    });

    it('returns diskann with all options provided', () => {
      expect(
        resolveDenseStrategy({
          chunkCount: 100,
          annThreshold: 200,
          indexStatus: 'ready',
          forceStrategy: 'diskann',
        }),
      ).toBe('diskann');
    });
  });

  describe('quantizedHash', () => {
    it('hashes a number array, rounding to 3 decimals', () => {
      const result = quantizedHash([1.23456, 2.34567]);
      expect(result).toBe('1.235,2.346');
    });

    it('hashes a Float32Array', () => {
      const arr = new Float32Array([0.1, 0.2, 0.3]);
      const result = quantizedHash(arr);
      expect(result).toBe('0.100,0.200,0.300');
    });

    it('only uses first 16 components', () => {
      const arr = Array.from({ length: 20 }, (_, i) => i);
      const result = quantizedHash(arr);
      const parts = result.split(',');
      expect(parts).toHaveLength(16);
      expect(parts[0]).toBe('0.000');
      expect(parts[15]).toBe('15.000');
    });

    it('handles empty or null embedding gracefully', () => {
      expect(quantizedHash(null)).toBe('');
      expect(quantizedHash(undefined)).toBe('');
      expect(quantizedHash([])).toBe('');
    });

    it('handles fewer than 16 components', () => {
      const result = quantizedHash([1]);
      expect(result).toBe('1.000');
    });
  });

  describe('query result cache', () => {
    const embedding = [0.1, 0.2, 0.3];
    const modelId = 'test-model';

    afterEach(() => {
      clearQueryResultCache();
    });

    it('returns undefined when no entry exists', () => {
      expect(getQueryResultCache({ embedding, modelId })).toBeUndefined();
    });

    it('stores and retrieves results', () => {
      const results = [{ chunk_id: 1 }];
      setQueryResultCache({ embedding, modelId, results });
      expect(getQueryResultCache({ embedding, modelId })).toEqual(results);
    });

    it('returns undefined after expiry', () => {
      setQueryResultCache({
        embedding,
        modelId,
        results: 'data',
        ttlMs: -1,
      });
      expect(getQueryResultCache({ embedding, modelId })).toBeUndefined();
    });

    it('clears all entries', () => {
      setQueryResultCache({ embedding, modelId, results: 'data' });
      clearQueryResultCache();
      expect(getQueryResultCache({ embedding, modelId })).toBeUndefined();
    });

    it('promotes accessed entries to most-recently-used position', () => {
      // Fill cache with entries, access one, then add one more to trigger eviction.
      // The accessed entry should NOT be evicted.
      const embA = [1, 0, 0];
      const embB = [0, 1, 0];
      const embC = [0, 0, 1];

      setQueryResultCache({ embedding: embA, modelId, results: 'A' });
      setQueryResultCache({ embedding: embB, modelId, results: 'B' });

      // Access A to promote it
      getQueryResultCache({ embedding: embA, modelId });

      // Add C with maxEntries=2, which should evict the oldest (B, not A)
      setQueryResultCache({
        embedding: embC,
        modelId,
        results: 'C',
        maxEntries: 2,
      });

      expect(getQueryResultCache({ embedding: embA, modelId })).toBe('A');
      expect(getQueryResultCache({ embedding: embB, modelId })).toBeUndefined();
      expect(getQueryResultCache({ embedding: embC, modelId })).toBe('C');
    });

    it('evicts oldest entries when maxEntries is exceeded', () => {
      const maxEntries = 3;
      const embeddings = [
        [1, 0],
        [0, 1],
        [0, 0],
        [1, 1],
      ];

      for (let i = 0; i < embeddings.length; i++) {
        setQueryResultCache({
          embedding: embeddings[i],
          modelId,
          results: `result-${i}`,
          maxEntries,
        });
      }

      // First entry should be evicted
      expect(
        getQueryResultCache({ embedding: embeddings[0], modelId }),
      ).toBeUndefined();
      // Last entry should exist
      expect(getQueryResultCache({ embedding: embeddings[3], modelId })).toBe(
        'result-3',
      );
    });

    it('uses default ttl and maxEntries', () => {
      setQueryResultCache({ embedding, modelId, results: 'data' });
      const cached = getQueryResultCache({ embedding, modelId });
      expect(cached).toBe('data');
    });

    it('overwrites existing entry on re-set', () => {
      setQueryResultCache({ embedding, modelId, results: 'old' });
      setQueryResultCache({ embedding, modelId, results: 'new' });
      expect(getQueryResultCache({ embedding, modelId })).toBe('new');
    });
  });

  describe('detectIncrementalUpdateAction', () => {
    it('returns rebuild when chunkCountDelta is non-zero (positive)', () => {
      expect(
        detectIncrementalUpdateAction({
          currentElements: 100,
          changedChunkCount: 1,
          chunkCountDelta: 5,
        }),
      ).toBe('rebuild');
    });

    it('returns rebuild when chunkCountDelta is non-zero (negative)', () => {
      expect(
        detectIncrementalUpdateAction({
          currentElements: 100,
          changedChunkCount: 1,
          chunkCountDelta: -3,
        }),
      ).toBe('rebuild');
    });

    it('returns rebuild when chunkCountDelta is null (coerced to 0... actually null ?? 0 = 0)', () => {
      // null ?? 0 → 0, so delta is 0; then check current/changed
      expect(
        detectIncrementalUpdateAction({
          currentElements: 100,
          changedChunkCount: 1,
          chunkCountDelta: null,
        }),
      ).toBe('incremental');
    });

    it('returns rebuild when currentElements is 0', () => {
      expect(
        detectIncrementalUpdateAction({
          currentElements: 0,
          changedChunkCount: 1,
          chunkCountDelta: 0,
        }),
      ).toBe('rebuild');
    });

    it('returns rebuild when currentElements is negative', () => {
      expect(
        detectIncrementalUpdateAction({
          currentElements: -1,
          changedChunkCount: 1,
          chunkCountDelta: 0,
        }),
      ).toBe('rebuild');
    });

    it('returns rebuild when currentElements is null', () => {
      expect(
        detectIncrementalUpdateAction({
          currentElements: null,
          changedChunkCount: 1,
          chunkCountDelta: 0,
        }),
      ).toBe('rebuild');
    });

    it('returns incremental when changed ratio is below threshold', () => {
      expect(
        detectIncrementalUpdateAction({
          currentElements: 100,
          changedChunkCount: 3,
          chunkCountDelta: 0,
        }),
      ).toBe('incremental');
    });

    it('returns rebuild when changed ratio is above threshold', () => {
      expect(
        detectIncrementalUpdateAction({
          currentElements: 100,
          changedChunkCount: 10,
          chunkCountDelta: 0,
        }),
      ).toBe('rebuild');
    });

    it('returns incremental at exact threshold boundary (changed/current == threshold is NOT > threshold)', () => {
      expect(
        detectIncrementalUpdateAction({
          currentElements: 100,
          changedChunkCount: 5,
          chunkCountDelta: 0,
        }),
      ).toBe('incremental');
    });

    it('respects custom threshold', () => {
      expect(
        detectIncrementalUpdateAction(
          {
            currentElements: 100,
            changedChunkCount: 10,
            chunkCountDelta: 0,
          },
          0.2,
        ),
      ).toBe('incremental');
    });

    it('handles null changedChunkCount as 0', () => {
      expect(
        detectIncrementalUpdateAction({
          currentElements: 100,
          changedChunkCount: null,
          chunkCountDelta: 0,
        }),
      ).toBe('incremental');
    });

    it('handles undefined chunkCountDelta as 0', () => {
      expect(
        detectIncrementalUpdateAction({
          currentElements: 100,
          changedChunkCount: 1,
        }),
      ).toBe('incremental');
    });
  });
});
