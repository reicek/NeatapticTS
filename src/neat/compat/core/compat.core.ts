import type {
  ComparisonMetrics,
  GenomeLike,
  NeatLikeForCompat,
} from './compat.types';

/**
 * Compatibility-distance mechanics used by NEAT speciation.
 *
 * This chapter holds the reusable inner loop: generation cache setup,
 * innovation-list caching, linear list comparison, and the final distance
 * calculation.
 */

/**
 * Ensure generation-scoped compatibility caches exist.
 *
 * @param neatContext - Current NEAT context holding generation and caches.
 * @returns Nothing. The helper resets caches when the generation changes.
 */
export function ensureGenerationCache(neatContext: NeatLikeForCompat): void {
  // Step 1: Reset caches when the generation changes.
  if (
    !neatContext._compatCacheGen ||
    neatContext._compatCacheGen !== neatContext.generation
  ) {
    neatContext._compatCacheGen = neatContext.generation;
    neatContext._compatDistCache = new Map<string, number>();
  }
}

/**
 * Build a stable cache key for a genome pair.
 *
 * @param firstGenome - First genome in the pair.
 * @param secondGenome - Second genome in the pair.
 * @returns Stable cache key in the form `minId|maxId`.
 */
export function buildPairKey(
  firstGenome: GenomeLike,
  secondGenome: GenomeLike,
): string {
  // Step 1: Normalize ids to stable numeric values.
  const firstId = firstGenome._id ?? 0;
  const secondId = secondGenome._id ?? 0;

  // Step 2: Order ids deterministically for cache stability.
  return firstId < secondId
    ? `${firstId}|${secondId}`
    : `${secondId}|${firstId}`;
}

/**
 * Retrieve the generation-scoped cache map for pairwise distances.
 *
 * @param neatContext - Current NEAT context with the cache map.
 * @returns Map storing cached distances for genome pairs this generation.
 */
export function getDistanceCacheMap(
  neatContext: NeatLikeForCompat,
): Map<string, number> {
  // Step 1: Return the already-initialized cache map.
  return neatContext._compatDistCache!;
}

/**
 * Retrieve or build a sorted innovation list for a genome.
 *
 * @param neatContext - NEAT context used for fallback innovation numbers.
 * @param genome - Genome to derive a sorted innovation list for.
 * @returns Array of `[innovationNumber, weight]` sorted by innovation number.
 */
export function getSortedInnovationCache(
  neatContext: NeatLikeForCompat,
  genome: GenomeLike,
): [number, number][] {
  // Step 1: Return the existing cache when present.
  if (genome._compatCache) {
    return genome._compatCache as [number, number][];
  }

  // Step 2: Build innovation-weight pairs from the connection list.
  const innovationPairs: [number, number][] = genome.connections.map(
    (connection) => [
      connection.innovation ?? neatContext._fallbackInnov(connection),
      connection.weight,
    ],
  );

  // Step 3: Sort by innovation id for linear merge comparisons.
  const sortedPairs = innovationPairs.toSorted(
    ([innovationA], [innovationB]) => innovationA - innovationB,
  );

  // Step 4: Store and return the cache.
  genome._compatCache = sortedPairs;
  return sortedPairs;
}

/**
 * Compare two sorted innovation lists and derive compatibility metrics.
 *
 * @param firstList - Sorted innovation list for the first genome.
 * @param secondList - Sorted innovation list for the second genome.
 * @returns Aggregated comparison metrics for distance computation.
 */
export function compareInnovationLists(
  firstList: [number, number][],
  secondList: [number, number][],
): ComparisonMetrics {
  // Step 1: Initialize comparison state.
  let firstIndex = 0;
  let secondIndex = 0;
  let matchingCount = 0;
  let disjointCount = 0;
  let excessCount = 0;
  let weightDifferenceSum = 0;

  // Step 2: Resolve max innovation ids for excess detection.
  const maxInnovFirst = resolveMaxInnovation(firstList);
  const maxInnovSecond = resolveMaxInnovation(secondList);

  // Step 3: Merge-walk the lists to classify matching, disjoint, and excess genes.
  while (firstIndex < firstList.length && secondIndex < secondList.length) {
    const [innovationFirst, weightFirst] = firstList[firstIndex];
    const [innovationSecond, weightSecond] = secondList[secondIndex];

    if (innovationFirst === innovationSecond) {
      matchingCount++;
      weightDifferenceSum += Math.abs(weightFirst - weightSecond);
      firstIndex++;
      secondIndex++;
      continue;
    }

    if (innovationFirst < innovationSecond) {
      excessCount += innovationFirst > maxInnovSecond ? 1 : 0;
      disjointCount += innovationFirst > maxInnovSecond ? 0 : 1;
      firstIndex++;
      continue;
    }

    excessCount += innovationSecond > maxInnovFirst ? 1 : 0;
    disjointCount += innovationSecond > maxInnovFirst ? 0 : 1;
    secondIndex++;
  }

  // Step 4: Remaining genes after one list ends are excess.
  excessCount += Math.max(0, firstList.length - firstIndex);
  excessCount += Math.max(0, secondList.length - secondIndex);

  return {
    firstGenomeSize: firstList.length,
    secondGenomeSize: secondList.length,
    matchingCount,
    disjointCount,
    excessCount,
    weightDifferenceSum,
  };
}

/**
 * Resolve the highest innovation id from a sorted list.
 *
 * @param list - Sorted innovation list for a genome.
 * @returns Highest innovation id or `0` when the list is empty.
 */
export function resolveMaxInnovation(list: [number, number][]): number {
  // Step 1: Return the last entry's innovation id, or 0 when empty.
  return list.length ? list.at(-1)![0] : 0;
}

/**
 * Compute the compatibility distance from precomputed metrics.
 *
 * @param neatContext - NEAT context providing compatibility coefficients.
 * @param metrics - Aggregated comparison metrics.
 * @returns Final compatibility distance for the genome pair.
 */
export function computeCompatibilityDistance(
  neatContext: NeatLikeForCompat,
  metrics: ComparisonMetrics,
): number {
  // Step 1: Resolve the normalization factor.
  const normalizationFactor = Math.max(
    1,
    Math.max(metrics.firstGenomeSize, metrics.secondGenomeSize),
  );

  // Step 2: Resolve the average weight difference for matching genes.
  const averageWeightDifference = metrics.matchingCount
    ? metrics.weightDifferenceSum / metrics.matchingCount
    : 0;

  // Step 3: Compute the weighted distance components.
  const options = neatContext.options;
  const excessComponent =
    (options.excessCoeff! * metrics.excessCount) / normalizationFactor;
  const disjointComponent =
    (options.disjointCoeff! * metrics.disjointCount) / normalizationFactor;
  const weightComponent = options.weightDiffCoeff! * averageWeightDifference;

  // Step 4: Fold the components into the final distance.
  return excessComponent + disjointComponent + weightComponent;
}
