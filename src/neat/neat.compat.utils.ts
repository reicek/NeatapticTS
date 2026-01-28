/**
 * Compatibility-distance helper utilities.
 *
 * @remarks
 * This module is intentionally dependency-free and provides small, focused
 * helpers for use by the compatibility orchestration layer.
 */

/**
 * Shape of a connection entry used during compatibility checks.
 *
 * @remarks
 * This is intentionally minimal and tolerant of missing fields. When
 * `innovation` is absent, a deterministic fallback id is derived from
 * `from.index` and `to.index`.
 */
export interface ConnectionLike {
  /** Optional source node descriptor (may be missing index). */
  from?: { index?: number };
  /** Optional target node descriptor (may be missing index). */
  to?: { index?: number };
  /** Optional innovation id; preferred when available. */
  innovation?: number;
  /** Connection weight for matching-gene comparisons. */
  weight: number;
}

/**
 * Minimal genome shape used for compatibility distance calculations.
 *
 * @remarks
 * The `_compatCache` stores sorted pairs of `[innovation, weight]` to avoid
 * recomputing the same derived list across repeated comparisons.
 */
export interface GenomeLike {
  /** Optional stable id for cache keys; defaults to 0 if absent. */
  _id?: number;
  /** Raw connection list for the genome/network. */
  connections: ConnectionLike[];
  /** Optional cached, sorted innovation list. */
  _compatCache?: Array<[number, number]>;
}

/**
 * Minimal NEAT context required by compatibility helpers.
 *
 * @remarks
 * This interface captures only what `_compatibilityDistance` and
 * `_fallbackInnov` depend on, including generation-scoped cache state and
 * coefficient options.
 */
export interface NeatLikeForCompat {
  /** Current generation number used to scope caches. */
  generation: number;
  /** Coefficients for compatibility distance components. */
  options: {
    /** Excess-gene coefficient used in the distance formula. */
    excessCoeff?: number;
    /** Disjoint-gene coefficient used in the distance formula. */
    disjointCoeff?: number;
    /** Average weight-difference coefficient used in the distance formula. */
    weightDiffCoeff?: number;
  };
  /** Generation id for which caches are valid. */
  _compatCacheGen?: number;
  /** Pairwise distance cache for the current generation. */
  _compatDistCache?: Map<string, number>;
  /** Deterministic fallback innovation id generator. */
  _fallbackInnov: (connection: ConnectionLike) => number;
}

/**
 * Aggregated comparison metrics for compatibility calculations.
 */
export type ComparisonMetrics = {
  /** Connection count for the first genome. */
  firstGenomeSize: number;
  /** Connection count for the second genome. */
  secondGenomeSize: number;
  /** Number of matching innovation ids. */
  matchingCount: number;
  /** Number of disjoint genes. */
  disjointCount: number;
  /** Number of excess genes. */
  excessCount: number;
  /** Sum of absolute weight differences for matching genes. */
  weightDifferenceSum: number;
};

/**
 * Ensure generation-scoped compatibility caches exist.
 *
 * @param neatContext - Current NEAT context holding generation and caches.
 * @returns void
 */
export function ensureGenerationCache(neatContext: NeatLikeForCompat): void {
  // Step 1: Reset caches when generation changes.
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
 * @returns Stable cache key in the form "minId|maxId".
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
 * @param genome - Genome to derive sorted innovation list for.
 * @returns Array of [innovationNumber, weight] sorted by innovationNumber.
 */
export function getSortedInnovationCache(
  neatContext: NeatLikeForCompat,
  genome: GenomeLike,
): [number, number][] {
  // Step 1: Return existing cache when present.
  if (genome._compatCache) return genome._compatCache as [number, number][];

  // Step 2: Build list of innovation-weight pairs.
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
 * Compare two sorted innovation lists and derive comparison metrics.
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

  // Step 3: Merge-walk the lists to classify genes.
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

  // Step 4: Remaining genes after exhaustion are excess.
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
 * @returns Highest innovation id or 0 if list is empty.
 */
export function resolveMaxInnovation(list: [number, number][]): number {
  // Step 1: Return the last entry's innovation id, or 0 if empty.
  return list.length ? list.at(-1)![0] : 0;
}

/**
 * Compute the compatibility distance from comparison metrics.
 *
 * @param neatContext - NEAT context providing coefficients.
 * @param metrics - Aggregated comparison metrics.
 * @returns Final compatibility distance for the genome pair.
 */
export function computeCompatibilityDistance(
  neatContext: NeatLikeForCompat,
  metrics: ComparisonMetrics,
): number {
  // Step 1: Resolve normalization factor.
  const normalizationFactor = Math.max(
    1,
    Math.max(metrics.firstGenomeSize, metrics.secondGenomeSize),
  );

  // Step 2: Resolve average weight difference.
  const averageWeightDifference = metrics.matchingCount
    ? metrics.weightDifferenceSum / metrics.matchingCount
    : 0;

  // Step 3: Compute weighted distance components.
  const options = neatContext.options;
  const excessComponent =
    (options.excessCoeff! * metrics.excessCount) / normalizationFactor;
  const disjointComponent =
    (options.disjointCoeff! * metrics.disjointCount) / normalizationFactor;
  const weightComponent = options.weightDiffCoeff! * averageWeightDifference;

  // Step 4: Fold components into final distance.
  return excessComponent + disjointComponent + weightComponent;
}
