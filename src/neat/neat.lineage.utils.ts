/**
 * Lineage / ancestry helper utilities for NEAT populations.
 *
 * This module centralizes helper logic used by the public lineage APIs to keep
 * the main entry file small and orchestration-focused.
 */

/**
 * Minimal shape assumed for a genome inside the NEAT population. Additional properties are
 * intentionally left open (index signature) because user implementations may extend genomes.
 */
export interface GenomeLike {
  /** Unique numeric identifier assigned when the genome is created. */
  _id: number;
  /** Optional list of parent genome IDs (could be 1 or 2 for sexual reproduction, or more in custom ops). */
  _parents?: number[];
  /** Allow arbitrary extra properties without forcing casts. */
  [key: string]: any; // eslint-disable-line @typescript-eslint/no-explicit-any
}

/** Expected `this` context for lineage helpers (a subset of the NEAT instance). */
export interface NeatLineageContext {
  /** Current evolutionary population (array of genomes). */
  population: GenomeLike[];
  /** RNG provider returning a PRNG function; shape taken from core NEAT implementation. */
  _getRNG: () => () => number;
}

/** Index pair representing a sampled genome pair. */
interface GenomeIndexPair {
  firstIndex: number;
  secondIndex: number;
}

/** Queue entry for ancestor traversal. */
interface AncestorQueueEntry {
  ancestorId: number;
  depth: number;
  genomeRef?: GenomeLike;
}

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

/**
 * Depth window (in breadth-first layers) used when gathering ancestor IDs.
 * A small window keeps the metric inexpensive while still capturing recent lineage diversity.
 *
 * Rationale: Deep full ancestry can grow quickly and become O(N * lineage depth). Empirically,
 * a window of 4 gives a stable signal about short‑term innovation mixing without large cost.
 *
 * You can fork and increase this constant if you need deeper lineage metrics, but note that
 * performance will degrade roughly proportionally to the number of enqueued ancestor nodes.
 *
 * Example (changing the window):
 *   // (NOT exported) – modify locally before building docs
 *   // const ANCESTOR_DEPTH_WINDOW = 6; // capture deeper history
 */
const ANCESTOR_DEPTH_WINDOW = 4;

/**
 * @param value Genome to read parents from.
 * @returns Parent ID list (empty when absent).
 */
export function normalizeParentIds(value: GenomeLike): number[] {
  // Step 1: Return a stable array for downstream logic.
  return Array.isArray(value._parents) ? value._parents : [];
}

/**
 * @param parentIds Direct parent IDs to seed the queue.
 * @param population Current population for lookups.
 * @returns Queue entries at depth 1.
 */
export function createInitialQueue(
  parentIds: number[],
  population: GenomeLike[],
): AncestorQueueEntry[] {
  // Step 1: Map each parent ID into a queue entry with lookup.
  return parentIds.map((parentId) => ({
    ancestorId: parentId,
    depth: DIRECT_PARENT_DEPTH,
    genomeRef: findGenomeById(population, parentId),
  }));
}

/**
 * @param queueEntries BFS queue seeded with direct parents.
 * @param population Current population for lookups.
 * @returns Unique ancestor IDs encountered within the window.
 */
export function collectAncestorIds(
  queueEntries: AncestorQueueEntry[],
  population: GenomeLike[],
): Set<number> {
  // Step 1: Traverse the queue with a stable index (FIFO behavior).
  const ancestorIds = new Set<number>();
  let queueIndex = ZERO_VALUE;

  // Step 2: Expand each entry within the allowed depth.
  while (queueIndex < queueEntries.length) {
    const currentEntry = queueEntries[queueIndex];
    queueIndex += INDEX_OFFSET;

    if (!isWithinDepthWindow(currentEntry.depth)) continue;

    ancestorIds.add(currentEntry.ancestorId);
    enqueueParentEntries(queueEntries, currentEntry, population);
  }

  // Step 3: Return the collected IDs.
  return ancestorIds;
}

/**
 * @param size Population size.
 * @returns True when at least two genomes exist.
 */
export function hasMinimumPopulation(size: number): boolean {
  // Step 1: Validate the minimum size requirement.
  return size >= MIN_POPULATION_FOR_PAIR;
}

/**
 * @param size Population size.
 * @returns Upper bound on the number of sampled pairs.
 */
export function calculateMaxSamplePairs(size: number): number {
  // Step 1: Compute total possible unordered pairs.
  const totalPairCount =
    (size * (size - INDEX_OFFSET)) / PAIR_COUNT_DENOMINATOR;
  // Step 2: Respect the sampling cap.
  return Math.min(MAX_UNIQUENESS_SAMPLE_PAIRS, totalPairCount);
}

/**
 * @param sampleCount Number of pairs to sample.
 * @param size Population size for index bounds.
 * @param rngFactory RNG provider to obtain a random function.
 * @returns Array of sampled index pairs.
 */
export function sampleGenomePairs(
  sampleCount: number,
  size: number,
  rngFactory: () => () => number,
): GenomeIndexPair[] {
  // Step 1: Build the RNG once per call.
  const randomNumber = rngFactory();
  const pairs: GenomeIndexPair[] = [];

  // Step 2: Sample each pair with self-pair avoidance.
  for (
    let sampleIndex = ZERO_VALUE;
    sampleIndex < sampleCount;
    sampleIndex += INDEX_OFFSET
  ) {
    const firstIndex = pickRandomIndex(randomNumber, size);
    const secondIndex = pickDistinctIndex(randomNumber, size, firstIndex);
    pairs.push({ firstIndex, secondIndex });
  }

  // Step 3: Return the sampled pairs.
  return pairs;
}

/**
 * @param pairs Sampled index pairs.
 * @param population Current population.
 * @param buildAncestorSet Helper to build ancestor sets.
 * @returns Jaccard distances for valid pairs.
 */
export function computePairDistances(
  pairs: GenomeIndexPair[],
  population: GenomeLike[],
  buildAncestorSet: (genome: GenomeLike) => Set<number>,
): number[] {
  // Step 1: Map pairs to distances, filtering invalid pairs.
  const distances: number[] = [];
  for (const pair of pairs) {
    const distance = computePairDistance(pair, population, buildAncestorSet);
    if (distance == null) continue;
    distances.push(distance);
  }
  return distances;
}

/**
 * @param distances Jaccard distances to average.
 * @returns Mean distance rounded to the configured decimal places.
 */
export function computeAverageDistance(distances: number[]): number {
  // Step 1: Guard on empty inputs.
  if (distances.length === ZERO_VALUE) return ZERO_VALUE;

  // Step 2: Sum the distances.
  const distanceSum = distances.reduce(
    (accumulator, distance) => accumulator + distance,
    ZERO_VALUE,
  );

  // Step 3: Average and round to the requested precision.
  return +(distanceSum / distances.length).toFixed(UNIQUENESS_DECIMAL_PLACES);
}

/**
 * @param depth Current depth value.
 * @returns True when within the configured depth window.
 */
function isWithinDepthWindow(depth: number): boolean {
  // Step 1: Compare against the configured window.
  return depth <= ANCESTOR_DEPTH_WINDOW;
}

/**
 * @param queueEntries Mutable queue array to append to.
 * @param currentEntry Current ancestor entry being expanded.
 * @param population Current population for lookups.
 */
function enqueueParentEntries(
  queueEntries: AncestorQueueEntry[],
  currentEntry: AncestorQueueEntry,
  population: GenomeLike[],
): void {
  // Step 1: Resolve parent IDs for the current entry.
  const parentIds = resolveParentIds(currentEntry.genomeRef);
  if (parentIds.length === ZERO_VALUE) return;

  // Step 2: Enqueue each parent with incremented depth.
  for (const parentId of parentIds) {
    queueEntries.push({
      ancestorId: parentId,
      depth: currentEntry.depth + DEPTH_INCREMENT,
      genomeRef: findGenomeById(population, parentId),
    });
  }
}

/**
 * @param population Current population for lookup.
 * @param genomeId Genome identifier to match.
 * @returns Genome reference if found.
 */
function findGenomeById(
  population: GenomeLike[],
  genomeId: number,
): GenomeLike | undefined {
  // Step 1: Resolve the genome by ID.
  return population.find((genomeItem) => genomeItem._id === genomeId);
}

/**
 * @param value Optional genome reference.
 * @returns Parent IDs when available.
 */
function resolveParentIds(value: GenomeLike | undefined): number[] {
  // Step 1: Normalize when the genome exists.
  if (!value) return [];
  return normalizeParentIds(value);
}

/**
 * @param randomNumber RNG function returning [0,1).
 * @param size Population size for bounds.
 * @returns Random index within bounds.
 */
function pickRandomIndex(randomNumber: () => number, size: number): number {
  // Step 1: Convert random float to integer index.
  return Math.floor(randomNumber() * size);
}

/**
 * @param randomNumber RNG function returning [0,1).
 * @param size Population size for bounds.
 * @param firstIndex Index to avoid.
 * @returns Random index not equal to the first index.
 */
function pickDistinctIndex(
  randomNumber: () => number,
  size: number,
  firstIndex: number,
): number {
  // Step 1: Draw an initial candidate.
  let secondIndex = pickRandomIndex(randomNumber, size);
  // Step 2: Offset when the indices collide.
  if (secondIndex === firstIndex)
    secondIndex = (secondIndex + INDEX_OFFSET) % size;
  return secondIndex;
}

/**
 * @param pair Index pair for comparison.
 * @param population Current population.
 * @param buildAncestorSet Helper to build ancestor sets.
 * @returns Jaccard distance or undefined when skipped.
 */
function computePairDistance(
  pair: GenomeIndexPair,
  population: GenomeLike[],
  buildAncestorSet: (genome: GenomeLike) => Set<number>,
): number | undefined {
  // Step 1: Build ancestor sets for each genome.
  const ancestorSetA = buildAncestorSet(population[pair.firstIndex]);
  const ancestorSetB = buildAncestorSet(population[pair.secondIndex]);

  // Step 2: Skip if both sets are empty.
  if (isEmptyAncestorPair(ancestorSetA, ancestorSetB)) return undefined;

  // Step 3: Compute Jaccard distance for the pair.
  return computeJaccardDistance(ancestorSetA, ancestorSetB);
}

/**
 * @param ancestorSetA First ancestor set.
 * @param ancestorSetB Second ancestor set.
 * @returns True when both sets are empty.
 */
function isEmptyAncestorPair(
  ancestorSetA: Set<number>,
  ancestorSetB: Set<number>,
): boolean {
  // Step 1: Check size for both sets.
  return (
    ancestorSetA.size === EMPTY_SET_SIZE && ancestorSetB.size === EMPTY_SET_SIZE
  );
}

/**
 * @param ancestorSetA First ancestor set.
 * @param ancestorSetB Second ancestor set.
 * @returns Jaccard distance for the two sets.
 */
function computeJaccardDistance(
  ancestorSetA: Set<number>,
  ancestorSetB: Set<number>,
): number {
  // Step 1: Count intersection size.
  const intersectionCount = countIntersection(ancestorSetA, ancestorSetB);
  // Step 2: Compute union size with fallback.
  const unionSize =
    ancestorSetA.size + ancestorSetB.size - intersectionCount ||
    UNION_SIZE_FALLBACK;
  // Step 3: Return distance as 1 - similarity.
  return JACCARD_DISTANCE_BASE - intersectionCount / unionSize;
}

/**
 * @param ancestorSetA First ancestor set.
 * @param ancestorSetB Second ancestor set.
 * @returns Size of intersection between the sets.
 */
function countIntersection(
  ancestorSetA: Set<number>,
  ancestorSetB: Set<number>,
): number {
  // Step 1: Fold over the smaller set.
  const [smallerSet, largerSet] =
    ancestorSetA.size <= ancestorSetB.size
      ? [ancestorSetA, ancestorSetB]
      : [ancestorSetB, ancestorSetA];

  let intersectionCount = ZERO_VALUE;
  for (const ancestorId of smallerSet)
    if (largerSet.has(ancestorId)) intersectionCount += INDEX_OFFSET;

  return intersectionCount;
}
