/**
 * @module parallel-search
 * @description Parallel query execution for the NeatapticTS semantic index.
 *
 * Runs multiple libSQL queries concurrently using a worker-pool pattern with
 * Promise.all, respecting the TURSO_CONCURRENCY env var (default 20). Results
 * from multiple queries are merged using Reciprocal Ranked Fusion (RRF) with
 * the standard formula `score = sum(1/(k + rank_i))` and default k=60.
 *
 * Graceful degradation: when individual queries fail, surviving query results
 * are still returned. A non-enumerable `errors` array is attached to the
 * returned result array so callers can inspect per-query failures without a
 * rejected batch promise.
 *
 * @example
 * ```ts
 * const results = await runParallelQueries({
 *   client,
 *   queries: [
 *     { sql: 'SELECT * FROM chunks_fts WHERE chunks_fts MATCH ?', args: ['neat'] },
 *     { sql: 'SELECT * FROM chunks_fts WHERE chunks_fts MATCH ?', args: ['evolution'] },
 *   ],
 * });
 * console.log(results.map((r) => r.chunk_id));
 * ```
 */

/**
 * Default RRF constant (k). Standard value from the original RRF paper
 * (Cormack, Clarke, Buettcher, 2009).
 * @type {number}
 */
const DEFAULT_RRF_K = 60;

/**
 * Default maximum number of concurrent in-flight database requests when
 * the TURSO_CONCURRENCY env var is not set or invalid.
 * @type {number}
 */
const DEFAULT_TURSO_CONCURRENCY = 20;

/**
 * Default fusion strategy when none is specified.
 * @type {'rrf'|'alpha'}
 */
const DEFAULT_FUSION = 'rrf';

/**
 * Normalize the `score` values in a single result list to the [0, 1] range.
 *
 * The highest score maps to 1 and the lowest to 0. When all scores are equal
 * (or the list is empty), every row receives a normalized score of 0 so the
 * blend does not artificially inflate chunks from a uniform list.
 *
 * @param {Array<{ chunk_id: number, score: number }>} list - Result rows.
 * @returns {Map<number, number>} Map of chunk_id to normalized score.
 */
function normalizeScores(list) {
  if (list.length === 0) {
    return new Map();
  }
  const scores = list.map((row) => Number(row.score ?? 0));
  const max = Math.max(...scores);
  const min = Math.min(...scores);
  const range = max - min;
  return new Map(
    list.map((row) => {
      const raw = Number(row.score ?? 0);
      const normalized = range > 0 ? (raw - min) / range : 0;
      return [row.chunk_id, normalized];
    }),
  );
}

/**
 * Merge multiple result lists using a score-based weighted alpha-blend.
 *
 * Each result list has its `score` values normalized to the [0, 1] range.
 * For every unique `chunk_id` in the union of all lists, the alpha-blend
 * score is the average of the normalized scores across all lists where the
 * chunk appears. Results are returned sorted by descending blend score.
 *
 * Unlike RRF (rank-based), alpha-blend is score-based: the magnitude of the
 * original score influences the final ranking, not just the rank position.
 *
 * @param {Array<Array<{ chunk_id: number, score: number }>>} resultLists -
 *   Array of result row arrays from successful queries.
 * @returns {Array} Merged results sorted by descending alpha-blend score.
 */
function mergeWithAlphaBlend(resultLists) {
  // Step 1: Build normalized score maps for each result list.
  const scoreMaps = resultLists.map((list) => normalizeScores(list));

  // Step 2: Union all chunk_ids across every result list.
  const allChunkIds = new Set();
  for (const scoreMap of scoreMaps) {
    for (const chunkId of scoreMap.keys()) {
      allChunkIds.add(chunkId);
    }
  }

  // Step 3: Compute alpha-blend score (average of normalized scores) for each chunk.
  const merged = [...allChunkIds].map((chunkId) => {
    let total = 0;
    let count = 0;
    for (const scoreMap of scoreMaps) {
      const normalized = scoreMap.get(chunkId);
      if (normalized !== undefined) {
        total += normalized;
        count += 1;
      }
    }
    const blendScore = count > 0 ? total / count : 0;

    // Find the base result row from the first list containing this chunk.
    let base = null;
    for (const list of resultLists) {
      const found = list.find((row) => row.chunk_id === chunkId);
      if (found) {
        base = found;
        break;
      }
    }

    return { ...base, chunk_id: chunkId, alpha_score: blendScore };
  });

  // Step 4: Sort by descending alpha-blend score.
  return merged.toSorted(
    (left, right) =>
      Number(right.alpha_score ?? 0) - Number(left.alpha_score ?? 0),
  );
}

/**
 * Merge multiple result lists using the configured fusion strategy.
 *
 * @param {Array<Array<{ chunk_id: number, score: number }>>} resultLists -
 *   Array of result row arrays from successful queries.
 * @param {object} options - Fusion options.
 * @param {'rrf'|'alpha'} [options.fusion='rrf'] - Fusion strategy.
 * @param {number} [options.k=60] - RRF constant (ignored for alpha-blend).
 * @returns {Array} Merged results sorted by descending fusion score.
 */
function mergeResults(
  resultLists,
  { fusion = DEFAULT_FUSION, k = DEFAULT_RRF_K } = {},
) {
  if (fusion === 'alpha') {
    return mergeWithAlphaBlend(resultLists);
  }
  return mergeWithRRF(resultLists, k);
}

/**
 * Read the TURSO_CONCURRENCY env var and return a positive integer.
 *
 * Falls back to the default (20) when the env var is unset, empty, or
 * parses to a non-positive value.
 *
 * @returns {number} Concurrency limit (minimum 1).
 */
function readConcurrency() {
  const raw = process.env.TURSO_CONCURRENCY;
  const parsed = Number.parseInt(raw, 10);
  if (Number.isInteger(parsed) && parsed > 0) {
    return parsed;
  }
  return DEFAULT_TURSO_CONCURRENCY;
}

/**
 * Merge multiple result lists using Reciprocal Ranked Fusion (RRF).
 *
 * Each result list is sorted by descending `score` and assigned 0-indexed
 * rank positions. For every unique `chunk_id` in the union of all lists,
 * the RRF score is computed as `sum(1/(k + rank_i))` across all lists
 * where the chunk appears. Results are returned sorted by descending
 * RRF score.
 *
 * @param {Array<Array<{ chunk_id: number, score: number }>>} resultLists -
 *   Array of result row arrays from successful queries.
 * @param {number} [k=60] - RRF constant.
 * @returns {Array} Merged results sorted by descending RRF score.
 */
function mergeWithRRF(resultLists, k = DEFAULT_RRF_K) {
  // Step 1: Build rank maps for each result list (sorted by descending score).
  const rankMaps = resultLists.map((list) => {
    const sorted = list.toSorted(
      (left, right) => Number(right.score ?? 0) - Number(left.score ?? 0),
    );
    return new Map(sorted.map((row, index) => [row.chunk_id, index]));
  });

  // Step 2: Union all chunk_ids across every result list.
  const allChunkIds = new Set();
  for (const rankMap of rankMaps) {
    for (const chunkId of rankMap.keys()) {
      allChunkIds.add(chunkId);
    }
  }

  // Step 3: Compute RRF score for each chunk in the union.
  const merged = [...allChunkIds].map((chunkId) => {
    let rrfScore = 0;
    for (const rankMap of rankMaps) {
      const rank = rankMap.get(chunkId);
      if (rank !== undefined) {
        rrfScore += 1 / (k + rank);
      }
    }

    // Find the base result row from the first list containing this chunk.
    let base = null;
    for (const list of resultLists) {
      const found = list.find((row) => row.chunk_id === chunkId);
      if (found) {
        base = found;
        break;
      }
    }

    return { ...base, chunk_id: chunkId, rrf_score: rrfScore };
  });

  // Step 4: Sort by descending RRF score.
  return merged.toSorted(
    (left, right) => Number(right.rrf_score ?? 0) - Number(left.rrf_score ?? 0),
  );
}

/**
 * Attach a non-enumerable `errors` property to a result array.
 *
 * Keeping the property non-enumerable preserves the array-like behavior
 * expected by existing callers (for example Jest `toEqual` on array elements)
 * while still allowing callers that know about the contract to read
 * `results.errors`.
 *
 * @param {Array} array - Result array to annotate.
 * @param {Array<object>} errors - Per-query failure descriptors.
 * @returns {Array} The same array instance.
 */
function attachErrors(array, errors) {
  Object.defineProperty(array, 'errors', {
    value: errors,
    enumerable: false,
    writable: true,
    configurable: true,
  });
  return array;
}

/**
 * Run multiple libSQL queries concurrently using a worker-pool pattern.
 *
 * Uses Promise.all with a bounded worker pool to execute queries
 * concurrently, respecting the TURSO_CONCURRENCY env var (default 20).
 * Results from all successful queries are merged using Reciprocal Ranked
 * Fusion (RRF) with default k=60.
 *
 * Graceful degradation: when individual queries fail, the error is recorded
 * in the non-enumerable `errors` array on the returned result list and only
 * surviving query results are merged and returned. A single bad query never
 * rejects the entire batch.
 *
 * @param {object} options - Parallel query options.
 * @param {object} options.client - libSQL client with an `execute({ sql, args })` method.
 * @param {Array<{ sql: string, args: Array }>} options.queries - Queries to run concurrently.
 * @param {'rrf'|'alpha'} [options.fusion='rrf'] - Fusion strategy for merging multi-query results.
 * @param {number} [options.k=60] - RRF constant for result fusion (ignored when fusion is 'alpha').
 * @param {number} [options.limit] - Maximum number of merged results to return.
 * @param {boolean} [options.use_dense] - When true, signals that dense queries are included in the batch.
 * @returns {Promise<Array>} Merged results sorted by descending fusion score, capped to `limit` when provided. The returned array carries a non-enumerable `errors` property listing any per-query failures.
 *
 * @example
 * ```ts
 * const results = await runParallelQueries({
 *   client,
 *   queries: [
 *     { sql: 'SELECT * FROM chunks_fts WHERE chunks_fts MATCH ?', args: ['neat'] },
 *     { sql: 'SELECT * FROM chunks_fts WHERE chunks_fts MATCH ?', args: ['evolution'] },
 *   ],
 *   fusion: 'rrf',
 *   limit: 10,
 * });
 * ```
 */
export async function runParallelQueries({
  client,
  queries,
  fusion = DEFAULT_FUSION,
  k = DEFAULT_RRF_K,
  limit,
  use_dense,
} = {}) {
  if (!Array.isArray(queries) || queries.length === 0) {
    return attachErrors([], []);
  }

  const errors = [];

  const concurrency = Math.min(readConcurrency(), queries.length);

  // Step 1: Launch worker pool — each worker pulls the next query from the
  // shared index and executes it. Per-query errors are caught so a single
  // failure does not reject the entire batch.
  let nextIndex = 0;
  const slots = new Array(queries.length);

  async function worker() {
    while (nextIndex < queries.length) {
      const idx = nextIndex;
      nextIndex += 1;

      try {
        const result = await client.execute(queries[idx]);
        slots[idx] = { ok: true, rows: result.rows };
      } catch (error) {
        errors.push({
          queryIndex: idx,
          sql: queries[idx].sql,
          args: queries[idx].args,
          message: error.message,
        });
        slots[idx] = { ok: false, error };
      }
    }
  }

  const workers = Array.from({ length: concurrency }, () => worker());
  await Promise.all(workers);

  // Step 2: Collect successful result lists (graceful degradation).
  const resultLists = slots
    .filter((slot) => slot && slot.ok)
    .map((slot) => slot.rows);

  if (resultLists.length === 0) {
    return attachErrors([], errors);
  }

  // Step 3: Merge results using the configured fusion strategy.
  // For a single surviving list, no cross-query fusion is needed — but we
  // still sort by score so the optional `limit` cap is deterministic.
  const merged =
    resultLists.length === 1
      ? resultLists[0].toSorted(
          (left, right) => Number(right.score ?? 0) - Number(left.score ?? 0),
        )
      : mergeResults(resultLists, { fusion, k });

  // Step 4: Cap result count when a limit is provided.
  const capped =
    Number.isInteger(limit) && limit > 0 ? merged.slice(0, limit) : merged;
  return attachErrors(capped, errors);
}
