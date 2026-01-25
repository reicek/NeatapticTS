import {
  buildPairKey,
  compareInnovationLists,
  computeCompatibilityDistance,
  ensureGenerationCache,
  getDistanceCacheMap,
  getSortedInnovationCache,
} from './neat.compat.utils';
import type {
  ConnectionLike,
  GenomeLike,
  NeatLikeForCompat,
} from './neat.compat.utils';

/**
 * Generate a deterministic fallback innovation id for a connection when the
 * connection does not provide an explicit innovation number.
 *
 * This function encodes the (from.index, to.index) pair into a single number
 * by multiplying the `from` index by a large base and adding the `to` index.
 * The large base reduces collisions between different pairs and keeps the id
 * stable and deterministic across runs. It is intended as a fallback only —
 * explicit innovation numbers (when present) should be preferred.
 *
 * Example:
 * const conn = { from: { index: 2 }, to: { index: 5 } };
 * const id = _fallbackInnov.call(neatContext, conn); // 200005
 *
 * Notes:
 * - Not globally guaranteed unique, but deterministic for the same indices.
 * - Useful during compatibility checks when some connections are missing innovation ids.
 *
 * @param this - The NEAT instance / context (kept for symmetry with other helpers).
 * @param connection - Connection object expected to contain `from.index` and `to.index`.
 * @returns A numeric innovation id derived from the (from, to) index pair.
 */
// eslint-disable-next-line prefer-arrow/prefer-arrow-functions
export const _fallbackInnov = function (
  this: NeatLikeForCompat,
  connection: ConnectionLike,
): number {
  // Read the source and target node indices, defaulting to 0 if missing.
  const fromIndex = connection.from?.index ?? 0;
  const toIndex = connection.to?.index ?? 0;

  // Encode the pair deterministically using a large multiplier to reduce collisions.
  return fromIndex * 100000 + toIndex;
};

/**
 * Compute the NEAT compatibility distance between two genomes (networks).
 *
 * The compatibility distance is used for speciation in NEAT. It combines the
 * number of excess and disjoint genes with the average weight difference of
 * matching genes. A generation-scoped cache is used to avoid recomputing the
 * same pair distances repeatedly within a generation.
 *
 * Formula:
 * distance = (c1 * excess + c2 * disjoint) / N + c3 * avgWeightDiff
 * where N = max(number of genes in genomeA, number of genes in genomeB)
 * and c1,c2,c3 are coefficients provided in `this.options`.
 *
 * Example:
 * const d = _compatibilityDistance.call(neatInstance, genomeA, genomeB);
 * if (d < neatInstance.options.compatibilityThreshold) { // same species }
 *
 * @param this - The NEAT instance / context which holds generation, options, and caches.
 * @param genomeA - First genome (network) to compare. Expected to expose `_id` and `connections`.
 * @param genomeB - Second genome (network) to compare. Expected to expose `_id` and `connections`.
 * @returns A numeric compatibility distance; lower means more similar.
 */
export const _compatibilityDistance = function (
  this: NeatLikeForCompat,
  genomeA: GenomeLike,
  genomeB: GenomeLike,
): number {
  // Step 1: Ensure generation-scoped caches are initialized.
  ensureGenerationCache(this);

  // Step 2: Resolve cache key and shared cache map.
  const cacheKey = buildPairKey(genomeA, genomeB);
  const cacheMap = getDistanceCacheMap(this);

  // Step 3: Return cached distance when available.
  const cachedDistance = cacheMap.get(cacheKey);
  if (cachedDistance !== undefined) return cachedDistance;

  // Step 4: Build sorted innovation lists for both genomes.
  const genomeAInnovations = getSortedInnovationCache(this, genomeA);
  const genomeBInnovations = getSortedInnovationCache(this, genomeB);

  // Step 5: Compare lists to derive match/excess/disjoint metrics.
  const comparison = compareInnovationLists(
    genomeAInnovations,
    genomeBInnovations,
  );

  // Step 6: Fold metrics into final distance and store in cache.
  const distance = computeCompatibilityDistance(this, comparison);
  cacheMap.set(cacheKey, distance);

  return distance;
};
