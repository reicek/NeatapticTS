import {
  buildPairKey,
  compareInnovationLists,
  computeCompatibilityDistance,
  ensureGenerationCache,
  getDistanceCacheMap,
  getSortedInnovationCache,
} from './core/compat.core';
import type {
  ConnectionLike,
  GenomeLike,
  NeatLikeForCompat,
} from './core/compat.types';

/**
 * Compatibility-distance orchestration for the NEAT controller.
 *
 * The root compatibility chapter stays intentionally small so readers can find
 * the public distance entrypoints first, then drill into the lower-level
 * comparison mechanics inside `core/`.
 *
 * - `core/` explains the cache lifecycle, innovation-list comparison, and the
 *   narrow runtime types used by compatibility checks.
 */

/**
 * Generate a deterministic fallback innovation id for a connection when the
 * connection does not provide an explicit innovation number.
 *
 * This fallback encodes the `(from.index, to.index)` pair into one stable
 * number so compatibility distance can still compare legacy or partially
 * normalized genomes. Explicit innovation numbers remain the preferred source
 * of truth.
 *
 * @param this - NEAT context kept for symmetry with the other compatibility helpers.
 * @param connection - Connection object expected to contain `from.index` and `to.index`.
 * @returns Numeric innovation id derived from the directional endpoint pair.
 */
export const _fallbackInnov = function (
  this: NeatLikeForCompat,
  connection: ConnectionLike,
): number {
  // Step 1: Resolve directional endpoint indices with safe defaults.
  const fromIndex = connection.from?.index ?? 0;
  const toIndex = connection.to?.index ?? 0;

  // Step 2: Encode the pair deterministically using a large multiplier.
  return fromIndex * 100_000 + toIndex;
};

/**
 * Compute the NEAT compatibility distance between two genomes.
 *
 * The helper keeps the top-level flow deliberately linear: refresh
 * generation-scoped caches, resolve sorted innovation lists, compare them, and
 * fold the resulting metrics into the final distance. The detailed list logic
 * lives in `core/` so this surface reads like the speciation contract rather
 * than an implementation dump.
 *
 * @param this - NEAT context holding generation state, options, and caches.
 * @param genomeA - First genome to compare.
 * @param genomeB - Second genome to compare.
 * @returns Compatibility distance where lower values mean more similar genomes.
 */
export const _compatibilityDistance = function (
  this: NeatLikeForCompat,
  genomeA: GenomeLike,
  genomeB: GenomeLike,
): number {
  // Step 1: Ensure generation-scoped caches exist.
  ensureGenerationCache(this);

  // Step 2: Resolve the ordered pair key and cache map.
  const cacheKey = buildPairKey(genomeA, genomeB);
  const cacheMap = getDistanceCacheMap(this);

  // Step 3: Return cached distance when available.
  const cachedDistance = cacheMap.get(cacheKey);
  if (cachedDistance !== undefined) {
    return cachedDistance;
  }

  // Step 4: Resolve the sorted innovation lists for both genomes.
  const genomeAInnovations = getSortedInnovationCache(this, genomeA);
  const genomeBInnovations = getSortedInnovationCache(this, genomeB);

  // Step 5: Compare the sorted lists and compute the final distance.
  const comparison = compareInnovationLists(
    genomeAInnovations,
    genomeBInnovations,
  );
  const distance = computeCompatibilityDistance(this, comparison);

  // Step 6: Cache and return the result.
  cacheMap.set(cacheKey, distance);
  return distance;
};
