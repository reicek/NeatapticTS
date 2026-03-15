import type Network from '../../../architecture/network';
import type {
  NetworkWithMOAnnotations,
  ObjectiveDescriptor,
} from '../shared/multiobjective.types';

/**
 * Builds a stable mapping from genome object references to their population
 * index.
 *
 * This relies on object identity (reference equality), not structural
 * equality. It is used to resolve objective values from a values matrix when
 * working with reordered views (e.g., sorted fronts).
 *
 * @param population - Genomes in population order.
 * @returns Map from genome references to their index.
 */
export function buildGenomeIndexByReference(
  population: Network[],
): Map<Network, number> {
  // Step 1: map each genome to its stable index.
  return new Map(
    population.map((genomeItem, genomeIndex) => [genomeItem, genomeIndex]),
  );
}

/**
 * Resolves a genome’s index using a reference-based map.
 *
 * @param genomeIndexByReference - Lookup map created by
 * {@link buildGenomeIndexByReference}.
 * @param genomeItem - Genome to resolve.
 * @returns The population index of the genome.
 * @throws If the genome is not present in the map.
 */
export function resolveGenomeIndex(
  genomeIndexByReference: Map<Network, number>,
  genomeItem: Network,
): number {
  // Step 1: return the stored index or throw if missing.
  const resolvedIndex = genomeIndexByReference.get(genomeItem);
  if (resolvedIndex === undefined) {
    throw new Error('Genome index lookup failed in crowding distance.');
  }

  return resolvedIndex;
}

/**
 * Resolves an objective value for a genome from a values matrix.
 *
 * This is a convenience helper for working with sorted/reordered views of the
 * population while keeping objective values in a dense matrix.
 *
 * @param valuesMatrixInput - Values matrix indexed by population index.
 * @param genomeIndexByReference - Lookup map from genome reference to index.
 * @param genomeItem - Genome to resolve.
 * @param objectiveIndex - Objective column index.
 * @returns The objective value for the genome.
 * @throws If the genome is not present in the lookup map.
 */
export function resolveObjectiveValue(
  valuesMatrixInput: number[][],
  genomeIndexByReference: Map<Network, number>,
  genomeItem: Network,
  objectiveIndex: number,
): number {
  const genomeIndex = resolveGenomeIndex(genomeIndexByReference, genomeItem);
  return valuesMatrixInput[genomeIndex][objectiveIndex];
}

/**
 * Initializes crowding-distance annotations for a front.
 *
 * This sets each genome’s `_moCrowd` to `0`. Later steps accumulate per-
 * objective spacing deltas.
 *
 * @param front - Pareto front.
 */
export function initializeCrowding(front: Network[]): void {
  // Step 1: initialize crowding distance to zero.
  for (const genomeItem of front) {
    (genomeItem as NetworkWithMOAnnotations)._moCrowd = 0;
  }
}

/**
 * Marks the boundary genomes of a sorted front as infinitely crowded.
 *
 * In NSGA-II style crowding distance, boundary solutions (extremes for the
 * objective) are assigned an infinite crowding distance to ensure they are
 * always preferred when ranks tie.
 *
 * @param sortedFront - Front sorted by the current objective.
 */
export function markBoundaryCrowding(sortedFront: Network[]): void {
  const firstGenome = sortedFront[0];
  const lastGenome = sortedFront.at(-1);
  if (!firstGenome || !lastGenome) return;

  // Step 1: mark boundary genomes as infinitely crowded.
  (firstGenome as NetworkWithMOAnnotations)._moCrowd = Infinity;
  (lastGenome as NetworkWithMOAnnotations)._moCrowd = Infinity;
}

/**
 * Resolves a non-zero objective range used to normalize crowding deltas.
 *
 * If all genomes have the same objective value, the raw range is `0`. This
 * returns `1` in that case to avoid division by zero while still producing a
 * well-defined crowding delta of `0`.
 *
 * @param minValue - Minimum objective value.
 * @param maxValue - Maximum objective value.
 * @returns Normalized range with a non-zero floor.
 */
export function resolveObjectiveRange(
  minValue: number,
  maxValue: number,
): number {
  const rawRange = maxValue - minValue;
  if (rawRange === 0) return 1;
  return rawRange;
}

/**
 * Accumulates crowding distance contributions for a single objective.
 *
 * Pre-conditions / expectations:
 * - `sortedFront` must be sorted ascending by the selected objective.
 * - {@link initializeCrowding} has already set `_moCrowd = 0` for the front.
 * - {@link markBoundaryCrowding} is typically called before this to set the
 *   boundary genomes to `Infinity`.
 *
 * Edge cases:
 * - If the front has fewer than 2 genomes, this is a no-op.
 * - If the objective range is `0`, a range of `1` is used (see
 *   {@link resolveObjectiveRange}).
 *
 * @param sortedFront - Front sorted by objective.
 * @param valuesMatrixInput - Values matrix.
 * @param genomeIndexByReference - Lookup map.
 * @param objectiveIndex - Objective column index.
 */
export function accumulateCrowdingForObjective(
  sortedFront: Network[],
  valuesMatrixInput: number[][],
  genomeIndexByReference: Map<Network, number>,
  objectiveIndex: number,
): void {
  const boundaryGenomes = resolveBoundaryGenomes(sortedFront);
  if (!boundaryGenomes) return;

  const valueRange = resolveObjectiveRangeFromBoundaries(
    valuesMatrixInput,
    genomeIndexByReference,
    boundaryGenomes,
    objectiveIndex,
  );

  // Step 1: accumulate distances between neighbors for non-boundary genomes.
  accumulateInteriorCrowding(
    sortedFront,
    valuesMatrixInput,
    genomeIndexByReference,
    objectiveIndex,
    valueRange,
  );
}

/**
 * Resolves the boundary (first/last) genomes for a sorted front.
 *
 * @param sortedFront - Front sorted by objective.
 * @returns Boundary genomes, or `null` if the front is empty.
 */
function resolveBoundaryGenomes(
  sortedFront: Network[],
): { firstGenome: Network; lastGenome: Network } | null {
  const firstGenome = sortedFront[0];
  const lastGenome = sortedFront.at(-1);
  if (!firstGenome || !lastGenome) return null;

  // Step 1: return boundary genomes.
  return { firstGenome, lastGenome };
}

/**
 * Resolves the normalized objective range for a front from its boundary
 * genomes.
 *
 * Because `sortedFront` is sorted by objective, the first and last genomes are
 * the extrema used for range normalization.
 *
 * @param valuesMatrixInput - Values matrix.
 * @param genomeIndexByReference - Lookup map.
 * @param boundaryGenomes - Boundary genomes for the front.
 * @param objectiveIndex - Objective column index.
 * @returns Normalized value range for the objective.
 */
function resolveObjectiveRangeFromBoundaries(
  valuesMatrixInput: number[][],
  genomeIndexByReference: Map<Network, number>,
  boundaryGenomes: { firstGenome: Network; lastGenome: Network },
  objectiveIndex: number,
): number {
  // Step 1: resolve objective values for boundary genomes.
  const minValue = resolveObjectiveValue(
    valuesMatrixInput,
    genomeIndexByReference,
    boundaryGenomes.firstGenome,
    objectiveIndex,
  );
  const maxValue = resolveObjectiveValue(
    valuesMatrixInput,
    genomeIndexByReference,
    boundaryGenomes.lastGenome,
    objectiveIndex,
  );

  // Step 2: normalize range to avoid division by zero.
  return resolveObjectiveRange(minValue, maxValue);
}

/**
 * Accumulates crowding deltas for the interior genomes of a sorted front.
 *
 * Interior genomes receive a normalized spacing delta:
 * `delta = (nextValue - previousValue) / valueRange`.
 *
 * @param sortedFront - Front sorted by objective.
 * @param valuesMatrixInput - Values matrix.
 * @param genomeIndexByReference - Lookup map.
 * @param objectiveIndex - Objective column index.
 * @param valueRange - Normalized objective range.
 */
function accumulateInteriorCrowding(
  sortedFront: Network[],
  valuesMatrixInput: number[][],
  genomeIndexByReference: Map<Network, number>,
  objectiveIndex: number,
  valueRange: number,
): void {
  const interiorIndices = buildInteriorIndexRange(sortedFront.length);

  // Step 1: accumulate distances between neighbors for interior genomes.
  for (const sortedIndex of interiorIndices) {
    const neighborPair = resolveNeighborPair(sortedFront, sortedIndex);
    const currentGenome = sortedFront[sortedIndex] as NetworkWithMOAnnotations;
    const previousValue = resolveObjectiveValue(
      valuesMatrixInput,
      genomeIndexByReference,
      neighborPair.previousGenome,
      objectiveIndex,
    );
    const nextValue = resolveObjectiveValue(
      valuesMatrixInput,
      genomeIndexByReference,
      neighborPair.nextGenome,
      objectiveIndex,
    );

    applyCrowdingDelta(currentGenome, previousValue, nextValue, valueRange);
  }
}

/**
 * Builds the index range for interior genomes of a front.
 *
 * Boundary genomes are excluded because their crowding distance is treated as
 * infinite.
 *
 * @param frontLength - Length of the sorted front.
 * @returns Interior indices excluding boundary genomes.
 */
function buildInteriorIndexRange(frontLength: number): number[] {
  // Step 1: skip boundary indices when front is too small.
  if (frontLength <= 2) return [];

  // Step 2: return indices excluding first and last elements.
  return Array.from({ length: frontLength - 2 }, (_, index) => index + 1);
}

/**
 * Resolves the neighbor genomes for an interior element of a sorted front.
 *
 * @param sortedFront - Front sorted by objective.
 * @param sortedIndex - Current index in sorted front.
 * @returns Previous and next neighbor genomes.
 */
function resolveNeighborPair(
  sortedFront: Network[],
  sortedIndex: number,
): { previousGenome: Network; nextGenome: Network } {
  // Step 1: resolve neighbor genomes from the sorted front.
  return {
    previousGenome: sortedFront[sortedIndex - 1],
    nextGenome: sortedFront[sortedIndex + 1],
  };
}

/**
 * Applies a normalized crowding-distance delta to a genome.
 *
 * If the genome’s crowding distance is `Infinity`, it will remain `Infinity`.
 * This helper only updates when `_moCrowd` is initialized.
 *
 * @param currentGenome - Genome to update.
 * @param previousValue - Objective value of previous genome.
 * @param nextValue - Objective value of next genome.
 * @param valueRange - Normalized objective range.
 */
function applyCrowdingDelta(
  currentGenome: NetworkWithMOAnnotations,
  previousValue: number,
  nextValue: number,
  valueRange: number,
): void {
  // Step 1: apply delta when crowding has been initialized.
  if (currentGenome._moCrowd !== undefined) {
    currentGenome._moCrowd += (nextValue - previousValue) / valueRange;
  }
}

/**
 * Assigns crowding-distance annotations for each Pareto front.
 *
 * This implements the crowding distance component of NSGA-II selection. Each
 * genome in each front receives a `_moCrowd` value representing how isolated
 * it is in objective space within its front.
 *
 * Notes:
 * - This function sorts each front by each objective (ascending raw values).
 *   Objective direction (min vs max) does not affect the computed spacing
 *   magnitude; extrema are treated as boundaries either way.
 * - Empty fronts are skipped.
 *
 * Side effects:
 * - Writes `_moCrowd` on each genome in each front.
 *
 * @param fronts - Pareto fronts.
 * @param valuesMatrixInput - Values matrix.
 * @param descriptors - Objective descriptors (provides objective count).
 * @param population - Population to resolve indices.
 */
export function assignCrowdingDistances(
  fronts: Network[][],
  valuesMatrixInput: number[][],
  descriptors: ObjectiveDescriptor[],
  population: Network[],
): void {
  const genomeIndexByReference = buildGenomeIndexByReference(population);
  const objectiveIndices = buildObjectiveIndexRange(descriptors.length);

  // Step 1: iterate each front and accumulate crowding per objective.
  for (const front of fronts) {
    if (shouldSkipCrowdingFront(front)) continue;
    initializeCrowding(front);
    assignCrowdingForFront(
      front,
      valuesMatrixInput,
      genomeIndexByReference,
      objectiveIndices,
    );
  }
}

/**
 * Builds a stable objective index range.
 *
 * @param objectiveCount - Number of objectives.
 * @returns Objective indices `0..objectiveCount-1`.
 */
function buildObjectiveIndexRange(objectiveCount: number): number[] {
  // Step 1: create a stable objective index range.
  return Array.from({ length: objectiveCount }, (_, index) => index);
}

/**
 * Determines whether crowding-distance processing should be skipped for a
 * front.
 *
 * @param front - Pareto front.
 * @returns `true` if the front should be skipped.
 */
function shouldSkipCrowdingFront(front: Network[]): boolean {
  // Step 1: skip empty fronts.
  return front.length === 0;
}

/**
 * Assigns crowding distances for a single front across all objectives.
 *
 * @param front - Pareto front.
 * @param valuesMatrixInput - Values matrix.
 * @param genomeIndexByReference - Lookup map.
 * @param objectiveIndices - Objective indices to process.
 */
function assignCrowdingForFront(
  front: Network[],
  valuesMatrixInput: number[][],
  genomeIndexByReference: Map<Network, number>,
  objectiveIndices: number[],
): void {
  // Step 1: process each objective in order.
  for (const objectiveIndex of objectiveIndices) {
    applyCrowdingForObjective(
      front,
      valuesMatrixInput,
      genomeIndexByReference,
      objectiveIndex,
    );
  }
}

/**
 * Applies crowding-distance accumulation for a single objective within a
 * single front.
 *
 * @param front - Pareto front.
 * @param valuesMatrixInput - Values matrix.
 * @param genomeIndexByReference - Lookup map.
 * @param objectiveIndex - Objective column index.
 */
function applyCrowdingForObjective(
  front: Network[],
  valuesMatrixInput: number[][],
  genomeIndexByReference: Map<Network, number>,
  objectiveIndex: number,
): void {
  // Step 1: sort the front by the current objective.
  const sortedByObjective = buildSortedFrontByObjective(
    front,
    valuesMatrixInput,
    genomeIndexByReference,
    objectiveIndex,
  );

  // Step 2: mark boundary genomes and accumulate interior distances.
  markBoundaryCrowding(sortedByObjective);
  accumulateCrowdingForObjective(
    sortedByObjective,
    valuesMatrixInput,
    genomeIndexByReference,
    objectiveIndex,
  );
}

/**
 * Builds a copy of the front sorted by the specified objective.
 *
 * Sorting is ascending by the raw objective value. This ordering is used for
 * computing neighbor spacing in objective space.
 *
 * @param front - Pareto front.
 * @param valuesMatrixInput - Values matrix.
 * @param genomeIndexByReference - Lookup map.
 * @param objectiveIndex - Objective column index.
 * @returns Front sorted by objective value.
 */
function buildSortedFrontByObjective(
  front: Network[],
  valuesMatrixInput: number[][],
  genomeIndexByReference: Map<Network, number>,
  objectiveIndex: number,
): Network[] {
  // Step 1: sort in ascending order of objective value.
  return front.toSorted((leftGenome, rightGenome) =>
    compareObjectiveValuesForCrowding(
      valuesMatrixInput,
      genomeIndexByReference,
      objectiveIndex,
      leftGenome,
      rightGenome,
    ),
  );
}

/**
 * Comparator used to sort genomes by a specific objective value.
 *
 * @param valuesMatrixInput - Values matrix.
 * @param genomeIndexByReference - Lookup map.
 * @param objectiveIndex - Objective column index.
 * @param leftGenome - Left genome.
 * @param rightGenome - Right genome.
 * @returns Numeric sort comparison value (ascending).
 */
function compareObjectiveValuesForCrowding(
  valuesMatrixInput: number[][],
  genomeIndexByReference: Map<Network, number>,
  objectiveIndex: number,
  leftGenome: Network,
  rightGenome: Network,
): number {
  // Step 1: resolve objective values.
  const leftValue = resolveObjectiveValue(
    valuesMatrixInput,
    genomeIndexByReference,
    leftGenome,
    objectiveIndex,
  );
  const rightValue = resolveObjectiveValue(
    valuesMatrixInput,
    genomeIndexByReference,
    rightGenome,
    objectiveIndex,
  );

  // Step 2: return ascending comparison.
  return leftValue - rightValue;
}
