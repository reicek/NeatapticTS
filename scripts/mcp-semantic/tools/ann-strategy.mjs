/**
 * @module ann-strategy
 * @description Dense retrieval strategy selection, query-result caching, and
 * incremental-update detection for the Repo Cortex ANN index.
 *
 * Uses DiskANN (libsql_vector_idx) as the sole dense-retrieval strategy.
 * The old HNSW and fallback strategies have been removed in favor of
 * server-side DiskANN indexes built into libSQL.
 */

/** Default chunk-count threshold at which DiskANN index build is recommended. */
export const DEFAULT_ANN_THRESHOLD = 50_000;

/** Default TTL for cached dense query results (30 minutes). */
export const DEFAULT_CACHE_TTL_MS = 30 * 60 * 1000;

/** Default maximum number of cached query-result entries. */
export const DEFAULT_CACHE_MAX_ENTRIES = 500;

/** DiskANN construction parameter: max neighbors per node (~3*sqrt(dimension)). */
export const DEFAULT_DISKANN_MAX_NEIGHBORS = 59;

/** DiskANN construction parameter: controls search/build tradeoff. */
export const DEFAULT_DISKANN_ALPHA = 1.2;

/** DiskANN query-time parameter: search list size (lower = faster, higher = better recall). */
export const DEFAULT_DISKANN_SEARCH_L = 80;

/**
 * Select the dense-retrieval strategy for a corpus.
 *
 * With DiskANN as the sole strategy, this always returns the diskann strategy
 * name. The threshold and force parameters are retained for API compatibility
 * with existing callers but no longer select between multiple strategies.
 *
 * @param {object} options - Strategy inputs.
 * @param {number} [options.chunkCount] - Number of chunks in the corpus.
 * @param {number} [options.annThreshold=50000] - Chunk-count threshold (retained for compat).
 * @param {string} [options.indexStatus] - ANN index status (retained for compat).
 * @param {string} [options.forceStrategy] - Force override (retained for compat, only diskann valid).
 * @returns {'diskann'} Selected strategy name.
 */
export function resolveDenseStrategy({
  chunkCount,
  annThreshold = DEFAULT_ANN_THRESHOLD,
  indexStatus,
  forceStrategy,
} = {}) {
  return 'diskann';
}

/**
 * Build a compact, fuzzy cache key from an embedding vector.
 *
 * Rounds the first 16 components to 3 decimals so nearly-identical vectors
 * (e.g. from the same query with minor floating-point drift) share a key.
 *
 * @param {Float32Array | number[]} embedding - Input embedding vector.
 * @returns {string} Quantized hash string.
 */
export function quantizedHash(embedding) {
  const sampleSize = Math.min(embedding?.length ?? 0, 16);
  const parts = [];
  for (let i = 0; i < sampleSize; i++) {
    parts.push(Number(embedding[i]).toFixed(3));
  }
  return parts.join(',');
}

/** @type {Map<string, { results: unknown, expiresAt: number }>} */
const queryResultCache = new Map();

/**
 * Build a cache key from a model identifier and a quantized embedding hash.
 *
 * @param {Float32Array | number[]} embedding - Query embedding.
 * @param {string} modelId - Embedding model identifier.
 * @returns {string} Cache key.
 */
function buildCacheKey(embedding, modelId) {
  return `${String(modelId)}:${quantizedHash(embedding)}`;
}

/**
 * Retrieve cached dense-query results for a query embedding.
 *
 * Returns `undefined` when no entry exists or the entry has expired. Hits are
 * promoted to the most-recently-used position.
 *
 * @param {object} options - Cache lookup inputs.
 * @param {Float32Array | number[]} options.embedding - Query embedding.
 * @param {string} options.modelId - Embedding model identifier.
 * @returns {unknown | undefined} Cached results or `undefined`.
 */
export function getQueryResultCache({ embedding, modelId }) {
  const key = buildCacheKey(embedding, modelId);
  const entry = queryResultCache.get(key);
  if (!entry) return undefined;

  if (Date.now() > entry.expiresAt) {
    queryResultCache.delete(key);
    return undefined;
  }

  // Promote to most-recently-used position.
  queryResultCache.delete(key);
  queryResultCache.set(key, entry);

  return entry.results;
}

/**
 * Store dense-query results in the LRU cache.
 *
 * Evicts the oldest entries when the cache grows beyond `maxEntries`.
 *
 * @param {object} options - Cache store inputs.
 * @param {Float32Array | number[]} options.embedding - Query embedding.
 * @param {string} options.modelId - Embedding model identifier.
 * @param {unknown} options.results - Results to cache.
 * @param {number} [options.ttlMs=1800000] - Time-to-live in milliseconds.
 * @param {number} [options.maxEntries=500] - Maximum cache size.
 */
export function setQueryResultCache({
  embedding,
  modelId,
  results,
  ttlMs = DEFAULT_CACHE_TTL_MS,
  maxEntries = DEFAULT_CACHE_MAX_ENTRIES,
}) {
  const key = buildCacheKey(embedding, modelId);
  const entry = { results, expiresAt: Date.now() + ttlMs };

  queryResultCache.set(key, entry);

  while (queryResultCache.size > maxEntries) {
    const firstKey = queryResultCache.keys().next().value;
    queryResultCache.delete(firstKey);
  }
}

/**
 * Clear all cached dense-query results.
 *
 * Useful in tests and after an index rebuild to avoid serving stale results.
 */
export function clearQueryResultCache() {
  queryResultCache.clear();
}

/**
 * Decide whether an index change can be applied incrementally or needs a full rebuild.
 *
 * An update is incremental when the ratio of changed chunks to current elements is at
 * or below the threshold and the total chunk count has not changed. Any chunk-count
 * change forces a rebuild because external-id to chunk-id mappings may shift.
 *
 * @param {object} options - Detection inputs.
 * @param {number} options.currentElements - Number of elements currently in the index.
 * @param {number} options.changedChunkCount - Number of chunks that changed.
 * @param {number} [options.chunkCountDelta=0] - Net change in total chunk count.
 * @param {number} [threshold=0.05] - Ratio threshold for incremental updates.
 * @returns {'incremental' | 'rebuild'} Recommended update action.
 */
export function detectIncrementalUpdateAction(
  { currentElements, changedChunkCount, chunkCountDelta },
  threshold = 0.05,
) {
  const delta = Number(chunkCountDelta ?? 0);
  if (delta !== 0) return 'rebuild';

  const current = Number(currentElements ?? 0);
  if (current <= 0) return 'rebuild';

  const changed = Number(changedChunkCount ?? 0);
  return changed / current > threshold ? 'rebuild' : 'incremental';
}
