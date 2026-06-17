/**
 * @module ann-strategy
 * @description Dense retrieval strategy selection, query-result caching, and
 * incremental-update detection for the Repo Cortex ANN index.
 *
 * Selects among three strategies:
 *   - `brute_force_cached`: default below the 50K-chunk threshold; uses an
 *     in-memory LRU + TTL cache for repeated exact/near-exact queries.
 *   - `hnsw`: used at or above the threshold when an HNSW index is ready and
 *     `hnswlib-node` is available.
 *   - `brute_force`: baseline strategy selectable only via `forceStrategy` for
 *     recall evaluation or diagnostics.
 */

/** Default chunk-count threshold at which HNSW becomes the preferred strategy. */
export const DEFAULT_ANN_THRESHOLD = 50_000;

/** Default TTL for cached brute-force query results (30 minutes). */
export const DEFAULT_CACHE_TTL_MS = 30 * 60 * 1000;

/** Default maximum number of cached query-result entries. */
export const DEFAULT_CACHE_MAX_ENTRIES = 500;

/** HNSW construction parameter: number of bi-directional links per node. */
export const DEFAULT_HNSW_M = 32;

/** HNSW construction parameter: size of the dynamic candidate list during build. */
export const DEFAULT_HNSW_EF_CONSTRUCTION = 200;

/** HNSW search parameter: size of the dynamic candidate list during query. */
export const DEFAULT_HNSW_EF_SEARCH = 100;

/**
 * Detect whether the optional `hnswlib-node` dependency is installed.
 *
 * The value is determined once at module load by attempting a dynamic import.
 * It is always a boolean so callers can use it directly in conditions.
 */
export let isHnswAvailable = false;

/**
 * Internal test seam for injecting a mock `hnswlib-node` implementation.
 * The default loader attempts the real optional dependency. Tests may replace
 * `importFn` and call `__refreshHnswAvailability()` to exercise the HNSW code
 * paths without installing the native package.
 */
export const __hnswTestSeam = {
  importFn: () => import('hnswlib-node'),
};

async function refreshHnswAvailability() {
  try {
    await __hnswTestSeam.importFn();
    isHnswAvailable = true;
  } catch {
    isHnswAvailable = false;
  }
}

/**
 * Re-evaluate HNSW availability using the current test seam loader.
 * Exported for coverage tests; no production caller should use this.
 * @returns {Promise<void>}
 */
export async function __refreshHnswAvailability() {
  await refreshHnswAvailability();
}

await refreshHnswAvailability();

/**
 * Select the dense-retrieval strategy for a corpus.
 *
 * Force overrides everything. Below the configured chunk threshold the
 * cache-assisted brute-force strategy is always preferred because the HNSW
 * build overhead outweighs the query benefit. At or above the threshold HNSW
 * is used only when the index is `ready` and HNSW support is available.
 *
 * @param {object} options - Strategy inputs.
 * @param {number} options.chunkCount - Number of chunks in the corpus.
 * @param {number} [options.annThreshold=50000] - Chunk-count threshold for HNSW.
 * @param {string} [options.indexStatus] - ANN index status (`ready`, `stale`, `error`, etc.).
 * @param {string} [options.forceStrategy] - Force a specific strategy (`hnsw`, `brute_force_cached`, `brute_force`).
 * @param {boolean} [options.hnswAvailable] - Whether `hnswlib-node` is available.
 * @returns {'brute_force_cached' | 'hnsw' | 'brute_force'} Selected strategy name.
 */
export function resolveDenseStrategy({
  chunkCount,
  annThreshold = DEFAULT_ANN_THRESHOLD,
  indexStatus,
  forceStrategy,
  hnswAvailable,
}) {
  if (forceStrategy === 'brute_force') return 'brute_force';
  if (forceStrategy === 'brute_force_cached') return 'brute_force_cached';
  if (forceStrategy === 'hnsw') return 'hnsw';

  const count = Number(chunkCount ?? 0);
  const threshold = Number(annThreshold ?? DEFAULT_ANN_THRESHOLD);
  if (count < threshold) return 'brute_force_cached';

  if (indexStatus === 'ready' && hnswAvailable !== false) return 'hnsw';

  return 'brute_force_cached';
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
 * Retrieve cached brute-force results for a query embedding.
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
 * Store brute-force results in the LRU cache.
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
 * Clear all cached brute-force query results.
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
