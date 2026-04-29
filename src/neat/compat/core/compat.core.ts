import type {
  CompatibilityInnovationMode,
  ComparisonMetrics,
  GenomeLike,
  NeatLikeForCompat,
} from './compat.types';

const explicitCompatibilityCacheGenomes = new WeakSet<GenomeLike>();

/**
 * Compatibility-distance mechanics used by NEAT speciation.
 *
 * This file holds the reusable inner loop beneath the controller-facing
 * compatibility chapter: generation cache setup, innovation-list caching,
 * linear list comparison, and the final distance calculation.
 *
 * The root compatibility chapter answers what the distance means. This core
 * chapter answers how that distance is assembled efficiently and deterministically.
 * The helpers are deliberately small and ordered to match the real execution
 * path: stabilize caches, normalize genomes, compare aligned innovations, then
 * fold the discovered evidence into the NEAT distance formula.
 */

/**
 * Ensure generation-scoped compatibility caches exist.
 *
 * The compatibility layer keeps pairwise distance results only for the current
 * generation because the population can mutate between generations. Once that
 * happens, earlier distances are no longer trustworthy. This helper provides
 * the safety boundary that drops stale cache state before later helpers assume
 * a cache map exists.
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
 * Compatibility distance is symmetric, so the cache key must be symmetric too.
 * Ordering the genome ids ensures the pair `(A, B)` lands in the same cache
 * slot as `(B, A)` instead of duplicating work or storing conflicting entries.
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
 * This helper exists mainly to make the orchestration path read cleanly after
 * `ensureGenerationCache()` has established the invariant that the map exists.
 * It keeps the later flow focused on comparison logic rather than repeated null
 * checks or map initialization details.
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
 * Raw connection arrays are not ideal for repeated pairwise comparison because
 * they may arrive unsorted and may mix explicit innovations with fallback-only
 * connections. This helper normalizes that surface into sorted
 * `[innovationNumber, weight]` pairs once, caches the result on the genome, and
 * returns the stable view that the merge comparison depends on.
 *
 * @example
 * ```ts
 * const innovationPairs = getSortedInnovationCache(neat, genome);
 * // [[3, 0.12], [8, -0.7], [11, 0.44]]
 * ```
 *
 * @param neatContext - NEAT context used for fallback innovation numbers.
 * @param genome - Genome to derive a sorted innovation list for.
 * @returns Array of `[innovationNumber, weight]` sorted by innovation number.
 */
export function getSortedInnovationCache(
  neatContext: NeatLikeForCompat,
  genome: GenomeLike,
): [number, number][] {
  const innovationMode = resolveCompatibilityInnovationMode(genome);

  // Step 1: Reuse the cached explicit-innovation view when it is still valid.
  if (
    innovationMode === 'require-explicit' &&
    genome._compatCache &&
    explicitCompatibilityCacheGenomes.has(genome)
  ) {
    return genome._compatCache as [number, number][];
  }

  // Step 2: Drop any non-canonical cached view before rebuilding.
  if (genome._compatCache && !explicitCompatibilityCacheGenomes.has(genome)) {
    delete genome._compatCache;
  }

  // Step 3: Build and sort the current innovation-weight pairs.
  const innovationPairs: [number, number][] = genome.connections.map(
    (connection, connectionIndex) => [
      resolveConnectionInnovation(
        neatContext,
        genome,
        connection,
        connectionIndex,
        innovationMode,
      ),
      connection.weight,
    ],
  );

  // Step 4: Sort by innovation id for linear merge comparisons.
  const sortedPairs = innovationPairs.toSorted(
    ([innovationA], [innovationB]) => innovationA - innovationB,
  );

  // Step 5: Keep fallback-derived views transient so native caches stay canonical.
  if (innovationMode === 'allow-fallback') {
    return sortedPairs;
  }

  // Step 6: Store and return the canonical explicit-innovation cache.
  genome._compatCache = sortedPairs;
  explicitCompatibilityCacheGenomes.add(genome);
  return sortedPairs;
}

/**
 * Compare two sorted innovation lists and derive compatibility metrics.
 *
 * This is the heart of the compatibility read. Because both lists are sorted,
 * the helper can walk them once like a merge step: matching innovations count
 * toward aligned genes, gaps inside the shared innovation range become disjoint
 * genes, and the remaining tail genes become excess. Weight differences are
 * only measured for matching genes because that is the only case where the two
 * genomes clearly refer to the same structural gene.
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
      disjointCount++;
      firstIndex++;
      continue;
    }

    disjointCount++;
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
 * The merge comparison uses this as the boundary between disjoint and excess
 * genes. Once one genome's innovations extend past the other's maximum seen id,
 * the remaining unmatched genes are no longer in-range mismatches; they are the
 * structural tail that NEAT treats as excess.
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
 * This fold turns the raw comparison evidence into the familiar NEAT distance:
 * excess structure penalty, disjoint structure penalty, and average matching
 * weight drift. Structural counts are normalized by the larger genome size so
 * larger topologies do not inflate distance merely because they contain more
 * possible genes.
 *
 * @example
 * ```ts
 * const distance = computeCompatibilityDistance(neat, {
 *   firstGenomeSize: 12,
 *   secondGenomeSize: 10,
 *   matchingCount: 8,
 *   disjointCount: 1,
 *   excessCount: 2,
 *   weightDifferenceSum: 0.9,
 * });
 * ```
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

/**
 * Resolve the compatibility-innovation mode for one genome.
 *
 * Native controller genomes default to `require-explicit`. Legacy or partial
 * genomes must opt into `allow-fallback` deliberately before compatibility can
 * synthesize alignment ids from endpoints.
 *
 * @param genome - Genome whose compatibility mode should be read.
 * @returns Effective compatibility-innovation mode.
 */
function resolveCompatibilityInnovationMode(
  genome: GenomeLike,
): CompatibilityInnovationMode {
  return genome._compatInnovationMode ?? 'require-explicit';
}

/**
 * Resolve the comparison innovation id for one connection.
 *
 * Native compatibility reads require explicit innovations. The fallback path is
 * reserved for genomes that deliberately opt into legacy or partial comparison.
 *
 * @param neatContext - NEAT context providing the fallback innovation resolver.
 * @param genome - Genome currently being normalized for comparison.
 * @param connection - Connection whose innovation id must be resolved.
 * @param connectionIndex - Stable connection position used for diagnostics.
 * @param innovationMode - Effective compatibility mode for the genome.
 * @returns Finite innovation id used for compatibility alignment.
 */
function resolveConnectionInnovation(
  neatContext: NeatLikeForCompat,
  genome: GenomeLike,
  connection: GenomeLike['connections'][number],
  connectionIndex: number,
  innovationMode: CompatibilityInnovationMode,
): number {
  if (Number.isFinite(connection.innovation)) {
    return connection.innovation!;
  }

  if (innovationMode === 'allow-fallback') {
    return neatContext._fallbackInnov(connection);
  }

  throw createMissingInnovationError(genome, connection, connectionIndex);
}

/**
 * Build the fail-fast error for native genomes missing explicit innovations.
 *
 * @param genome - Genome that failed compatibility normalization.
 * @param connection - Connection missing its explicit innovation id.
 * @param connectionIndex - Stable connection position used for diagnostics.
 * @returns Error describing why native compatibility refused fallback ids.
 */
function createMissingInnovationError(
  genome: GenomeLike,
  connection: GenomeLike['connections'][number],
  connectionIndex: number,
): Error {
  return new Error(
    'Compatibility distance requires explicit connection innovations for native genomes. Use `_compatInnovationMode = "allow-fallback"` only for legacy, imported, or deliberately partial genomes.',
    {
      cause: {
        genomeId: genome._id ?? null,
        connectionIndex,
        fromIndex: connection.from?.index ?? null,
        toIndex: connection.to?.index ?? null,
      },
    },
  );
}
