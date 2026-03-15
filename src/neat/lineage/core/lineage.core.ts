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
/** Fallback union size to avoid divide-by-zero. */
const UNION_SIZE_FALLBACK = 1;
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
 * This chapter holds the breadth-first ancestor traversal, sampled pair
 * generation, and Jaccard-distance aggregation logic behind the public lineage
 * metrics.
 */

/**
 * Normalize the parent ID list for a genome.
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
 * @param size - Population size.
 * @returns `true` when at least two genomes exist.
 */
export function hasMinimumPopulation(size: number): boolean {
  return size >= MIN_POPULATION_FOR_PAIR;
}

/**
 * Compute the upper bound on sampled genome pairs.
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
  const unionSize =
    ancestorSetA.size + ancestorSetB.size - intersectionCount ||
    UNION_SIZE_FALLBACK;
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
