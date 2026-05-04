import type {
  AncestorQueueEntry,
  GenomeIndexPair,
  GenomeLike,
} from './lineage.types';

/** Depth value used for direct parents in ancestor traversal. */
const DIRECT_PARENT_DEPTH = 1;
/** Common zero value for counters and defaults. */
const ZERO_VALUE = 0;
/** Increment step for ancestor depth. */
const DEPTH_INCREMENT = 1;
/** Minimum population size needed to form a pair. */
const MIN_POPULATION_FOR_PAIR = 2;
/** Pair denominator for nC2 = n(n-1)/2. */
const PAIR_COUNT_DENOMINATOR = 2;
/** Offset used to avoid identical random indices. */
const INDEX_OFFSET = 1;
/** Default empty set size sentinel used in guards. */
const EMPTY_SET_SIZE = 0;
/** Jaccard distance base (1 - similarity). */
const JACCARD_DISTANCE_BASE = 1;
/** Decimal precision for reported uniqueness. */
const UNIQUENESS_DECIMAL_PLACES = 3;
/** Maximum number of distinct genome pairs to sample when computing uniqueness. */
const MAX_UNIQUENESS_SAMPLE_PAIRS = 30;
/** Depth window used when gathering ancestor IDs. */
const ANCESTOR_DEPTH_WINDOW = 4;

/**
 * Lineage-analysis mechanics used by NEAT ancestry helpers.
 *
 * This chapter is the mechanics layer beneath the controller-facing lineage
 * helpers. Its job is to turn stored parent ids into a bounded ancestry signal
 * that is cheap enough for telemetry and adaptive policy to read during a run.
 *
 * The pipeline has four stages:
 *
 * 1. normalize parent links into one consistent ancestry input shape,
 * 2. walk outward through recent ancestors with a bounded breadth-first queue,
 * 3. sample a limited set of genome pairs instead of exhaustively comparing the
 *    full population,
 * 4. measure ancestor-set overlap with Jaccard distance and fold the result
 *    into one mean uniqueness signal.
 *
 * That staged split is what keeps the public lineage chapter readable. The
 * root helpers can describe what lineage metrics mean, while this file owns the
 * exact traversal, sampling, and distance math that makes those metrics stable.
 *
 * Read the chapter in this order: start with `normalizeParentIds()` and
 * `createInitialQueue()`, continue through `collectAncestorIds()` for the
 * breadth-first walk, then `sampleGenomePairs()` for the comparison budget, and
 * end with `computePairDistances()` plus `computeAverageDistance()` for the
 * final Jaccard-based fold.
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   parents[Genome parent ids]:::base --> queue[Initial breadth-first queue]:::accent
 *   queue --> traversal[Bounded ancestor traversal]:::base
 *   traversal --> sets[Per-genome ancestor sets]:::base
 *   population[Population plus deterministic RNG]:::base --> pairs[Sampled genome pairs]:::accent
 *   pairs --> compare[Jaccard distance per pair]:::base
 *   sets --> compare
 *   compare --> mean[Mean ancestor uniqueness]:::accent
 * ```
 */

/**
 * Normalize the parent ID list for a genome.
 *
 * This helper is the first seam in the pipeline: it turns "maybe has lineage
 * metadata" into a guaranteed array shape so the traversal code never needs to
 * branch on missing parent storage.
 *
 * @param value - Genome to read parents from.
 * @returns Parent ID list, or an empty array when absent.
 */
export function normalizeParentIds(value: GenomeLike): number[] {
  return Array.isArray(value._parents) ? value._parents : [];
}

/**
 * Create the initial breadth-first queue from direct parent IDs.
 *
 * This queue is the bridge between stored lineage metadata and the traversal
 * loop. Depth starts at the direct-parent layer because lineage uniqueness is a
 * recent-family metric first; later queue expansion can then walk outward while
 * still respecting the bounded depth window.
 *
 * @param parentIds - Direct parent IDs to seed the queue.
 * @param population - Current population for ID lookups.
 * @returns Queue entries at depth 1.
 */
export function createInitialQueue(
  parentIds: number[],
  population: GenomeLike[],
): AncestorQueueEntry[] {
  return parentIds.map((parentId) => ({
    ancestorId: parentId,
    depth: DIRECT_PARENT_DEPTH,
    genomeRef: findGenomeById(population, parentId),
  }));
}

/**
 * Collect ancestor IDs encountered within the configured depth window.
 *
 * This is the core ancestry walk. It uses a queue-backed breadth-first pass so
 * near ancestors are explored before deeper ones, which matches the meaning of
 * the metric: recent family overlap should dominate the signal more than very
 * old shared history.
 *
 * The traversal stays bounded on purpose. Runtime lineage metrics do not need a
 * full genealogy of the run; they need a stable recent-neighborhood view that
 * remains cheap to recompute.
 *
 * @param queueEntries - Breadth-first queue seeded with direct parents.
 * @param population - Current population for ID lookups.
 * @returns Unique ancestor IDs encountered within the depth window.
 */
export function collectAncestorIds(
  queueEntries: AncestorQueueEntry[],
  population: GenomeLike[],
): Set<number> {
  const ancestorIds = new Set<number>();
  let queueIndex = ZERO_VALUE;

  while (queueIndex < queueEntries.length) {
    const currentEntry = queueEntries[queueIndex];
    queueIndex += INDEX_OFFSET;

    if (!isWithinDepthWindow(currentEntry.depth)) {
      continue;
    }

    ancestorIds.add(currentEntry.ancestorId);
    enqueueParentEntries(queueEntries, currentEntry, population);
  }

  return ancestorIds;
}

/**
 * Check whether the population is large enough to form a sampled pair.
 *
 * This guard exists because ancestor uniqueness is defined over genome
 * comparisons, not individual genomes. A population smaller than two can still
 * have ancestry data, but it cannot produce a meaningful pairwise distance.
 *
 * @param size - Population size.
 * @returns `true` when at least two genomes exist.
 */
export function hasMinimumPopulation(size: number): boolean {
  return size >= MIN_POPULATION_FOR_PAIR;
}

/**
 * Compute the upper bound on sampled genome pairs.
 *
 * The uniqueness metric is intentionally sampled rather than exhaustive. This
 * helper keeps that budget honest by capping the requested sample count to both
 * the true combinatorial maximum and the chapter's global runtime limit.
 *
 * @param size - Population size.
 * @returns Sample cap respecting both the combinatorial count and the global limit.
 */
export function calculateMaxSamplePairs(size: number): number {
  const totalPairCount =
    (size * (size - INDEX_OFFSET)) / PAIR_COUNT_DENOMINATOR;
  return Math.min(MAX_UNIQUENESS_SAMPLE_PAIRS, totalPairCount);
}

/**
 * Sample genome index pairs for ancestor uniqueness.
 *
 * The sampling policy chooses a bounded set of pair comparisons while keeping
 * replay deterministic through the caller-supplied RNG factory. The result is
 * not a canonical population ordering; it is a reproducible comparison budget
 * for the current generation.
 *
 * @param sampleCount - Number of pairs to sample.
 * @param size - Population size for index bounds.
 * @param rngFactory - RNG provider used to obtain a random function.
 * @returns Array of sampled index pairs.
 */
export function sampleGenomePairs(
  sampleCount: number,
  size: number,
  rngFactory: () => () => number,
): GenomeIndexPair[] {
  const randomNumber = rngFactory();
  const pairs: GenomeIndexPair[] = [];

  for (
    let sampleIndex = ZERO_VALUE;
    sampleIndex < sampleCount;
    sampleIndex += INDEX_OFFSET
  ) {
    const firstIndex = pickRandomIndex(randomNumber, size);
    const secondIndex = pickDistinctIndex(randomNumber, size, firstIndex);
    pairs.push({ firstIndex, secondIndex });
  }

  return pairs;
}

/**
 * Compute Jaccard distances for sampled genome pairs.
 *
 * This stage turns sampled pair indices into actual lineage evidence. Each pair
 * is resolved to two genomes, each genome is expanded into a shallow ancestor
 * set, and those sets are compared using Jaccard distance so the final metric
 * reflects overlap rather than raw ancestor counts.
 *
 * @param pairs - Sampled index pairs.
 * @param population - Current population.
 * @param buildAncestorSet - Helper that builds ancestor sets for genomes.
 * @returns Jaccard distances for valid pairs.
 */
export function computePairDistances(
  pairs: GenomeIndexPair[],
  population: GenomeLike[],
  buildAncestorSet: (genome: GenomeLike) => Set<number>,
): number[] {
  const distances: number[] = [];

  for (const pair of pairs) {
    const distance = computePairDistance(pair, population, buildAncestorSet);
    if (distance == null) {
      continue;
    }
    distances.push(distance);
  }

  return distances;
}

/**
 * Compute the mean ancestor distance across sampled pairs.
 *
 * This is the final fold from many local comparisons into one controller-facing
 * scalar. Rounding is intentional: the value is meant to be a stable telemetry
 * and adaptive-policy signal rather than a high-precision scientific output.
 *
 * @param distances - Pairwise Jaccard distances.
 * @returns Mean distance rounded to the configured decimal precision.
 */
export function computeAverageDistance(distances: number[]): number {
  if (distances.length === ZERO_VALUE) {
    return ZERO_VALUE;
  }

  const distanceSum = distances.reduce(
    (accumulator, distance) => accumulator + distance,
    ZERO_VALUE,
  );

  return +(distanceSum / distances.length).toFixed(UNIQUENESS_DECIMAL_PLACES);
}

function isWithinDepthWindow(depth: number): boolean {
  return depth <= ANCESTOR_DEPTH_WINDOW;
}

function enqueueParentEntries(
  queueEntries: AncestorQueueEntry[],
  currentEntry: AncestorQueueEntry,
  population: GenomeLike[],
): void {
  const parentIds = resolveParentIds(currentEntry.genomeRef);
  if (parentIds.length === ZERO_VALUE) {
    return;
  }

  for (const parentId of parentIds) {
    queueEntries.push({
      ancestorId: parentId,
      depth: currentEntry.depth + DEPTH_INCREMENT,
      genomeRef: findGenomeById(population, parentId),
    });
  }
}

function findGenomeById(
  population: GenomeLike[],
  genomeId: number,
): GenomeLike | undefined {
  return population.find((genomeItem) => genomeItem._id === genomeId);
}

function resolveParentIds(value: GenomeLike | undefined): number[] {
  if (!value) {
    return [];
  }

  return normalizeParentIds(value);
}

function pickRandomIndex(randomNumber: () => number, size: number): number {
  return Math.floor(randomNumber() * size);
}

function pickDistinctIndex(
  randomNumber: () => number,
  size: number,
  firstIndex: number,
): number {
  let secondIndex = pickRandomIndex(randomNumber, size);
  if (secondIndex === firstIndex) {
    secondIndex = (secondIndex + INDEX_OFFSET) % size;
  }
  return secondIndex;
}

function computePairDistance(
  pair: GenomeIndexPair,
  population: GenomeLike[],
  buildAncestorSet: (genome: GenomeLike) => Set<number>,
): number | undefined {
  const ancestorSetA = buildAncestorSet(population[pair.firstIndex]);
  const ancestorSetB = buildAncestorSet(population[pair.secondIndex]);

  if (isEmptyAncestorPair(ancestorSetA, ancestorSetB)) {
    return undefined;
  }

  return computeJaccardDistance(ancestorSetA, ancestorSetB);
}

function isEmptyAncestorPair(
  ancestorSetA: Set<number>,
  ancestorSetB: Set<number>,
): boolean {
  return (
    ancestorSetA.size === EMPTY_SET_SIZE && ancestorSetB.size === EMPTY_SET_SIZE
  );
}

function computeJaccardDistance(
  ancestorSetA: Set<number>,
  ancestorSetB: Set<number>,
): number {
  const intersectionCount = countIntersection(ancestorSetA, ancestorSetB);
  const unionSize = ancestorSetA.size + ancestorSetB.size - intersectionCount;
  return JACCARD_DISTANCE_BASE - intersectionCount / unionSize;
}

function countIntersection(
  ancestorSetA: Set<number>,
  ancestorSetB: Set<number>,
): number {
  const [smallerSet, largerSet] =
    ancestorSetA.size <= ancestorSetB.size
      ? [ancestorSetA, ancestorSetB]
      : [ancestorSetB, ancestorSetA];

  let intersectionCount = ZERO_VALUE;
  for (const ancestorId of smallerSet) {
    if (largerSet.has(ancestorId)) {
      intersectionCount += INDEX_OFFSET;
    }
  }

  return intersectionCount;
}
