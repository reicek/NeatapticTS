import type Network from '../../../architecture/network';
import type { NetworkWithMOAnnotations } from '../shared/multiobjective.types';
import type { DominanceState } from '../dominance/multiobjective.dominance';

/**
 * Maximum number of Pareto fronts to allow during ranking before aborting.
 *
 * This is a defensive guard against pathological conditions (e.g., corrupted
 * dominance bookkeeping) that could otherwise cause long/infinite loops.
 */
export const MAX_PARETO_FRONT_RANK_GUARD = 50;

/**
 * Builds Pareto fronts from a precomputed dominance state.
 *
 * This performs the “peeling” phase of fast non-dominated sorting:
 * - Start with the first front (all non-dominated genomes).
 * - For each front, reduce domination counts of the genomes it dominates.
 * - Any genome whose domination count becomes zero moves to the next front.
 *
 * Side effects:
 * - Annotates each genome in `population` with `_moRank` (0 = best front).
 *
 * Guard:
 * - Stops when `currentFrontRank > maxFrontRankGuard` to avoid pathological
 *   infinite/degenerate runs. If the guard triggers, the returned fronts may
 *   be incomplete.
 *
 * @param population - Genome population (same ordering used by dominance
 * bookkeeping).
 * @param dominanceState - Dominance bookkeeping.
 * @param maxFrontRankGuard - Safety guard for ranking iterations.
 * @returns Ordered Pareto fronts (rank order).
 */
export function buildParetoFronts(
  population: Network[],
  dominanceState: DominanceState,
  maxFrontRankGuard: number,
): Network[][] {
  const paretoFronts: Network[][] = [];
  let currentFrontIndices = dominanceState.firstFrontIndices;
  let currentFrontRank = 0;

  // Step 1: peel off fronts breadth-first by dominance counts.
  while (currentFrontIndices.length > 0) {
    // Step 1.1: annotate current front and collect the next front.
    const nextFrontIndices = buildNextFrontIndices(
      population,
      dominanceState,
      currentFrontIndices,
      currentFrontRank,
    );

    // Step 1.2: add current front to result set.
    appendFront(paretoFronts, population, currentFrontIndices);

    // Step 1.3: advance to the next front.
    currentFrontIndices = nextFrontIndices;
    currentFrontRank = incrementFrontRank(currentFrontRank);

    // Step 1.4: stop if guard threshold is exceeded.
    if (shouldStopFrontRanking(currentFrontRank, maxFrontRankGuard)) break;
  }

  return paretoFronts;
}

/**
 * Builds the next front by applying rank annotations and dominance updates.
 *
 * @param population - Genome population.
 * @param dominanceState - Dominance bookkeeping.
 * @param currentFrontIndices - Indices for the current front.
 * @param currentFrontRank - Rank to assign to the current front.
 * @returns Indices for the next front.
 */
function buildNextFrontIndices(
  population: Network[],
  dominanceState: DominanceState,
  currentFrontIndices: number[],
  currentFrontRank: number,
): number[] {
  const nextFrontIndices: number[] = [];

  // Step 1: assign ranks and update domination counts.
  for (const genomeIndex of currentFrontIndices) {
    annotateGenomeRank(population, genomeIndex, currentFrontRank);
    collectNextFrontIndices(dominanceState, genomeIndex, nextFrontIndices);
  }

  return nextFrontIndices;
}

/**
 * Annotates a genome with its Pareto front rank.
 *
 * @param population - Genome population.
 * @param genomeIndex - Index of the genome to annotate.
 * @param frontRank - Pareto front rank (0 = best front).
 */
function annotateGenomeRank(
  population: Network[],
  genomeIndex: number,
  frontRank: number,
): void {
  // Step 1: record rank on the genome annotation.
  (population[genomeIndex] as NetworkWithMOAnnotations)._moRank = frontRank;
}

/**
 * Collects indices that become non-dominated after removing the current
 * genome’s dominance influence.
 *
 * @param dominanceState - Dominance bookkeeping.
 * @param genomeIndex - Index of the current genome.
 * @param nextFrontIndices - Accumulator for the next front.
 */
function collectNextFrontIndices(
  dominanceState: DominanceState,
  genomeIndex: number,
  nextFrontIndices: number[],
): void {
  // Step 1: update dominated indices and collect newly non-dominated entries.
  for (const dominatedIndex of dominanceState.dominatedIndicesByIndex[
    genomeIndex
  ]) {
    dominanceState.dominationCounts[dominatedIndex]--;
    if (dominanceState.dominationCounts[dominatedIndex] === 0) {
      nextFrontIndices.push(dominatedIndex);
    }
  }
}

/**
 * Appends the current front (index list) as genome references to the
 * `paretoFronts` accumulator.
 *
 * @param paretoFronts - Accumulator for Pareto fronts.
 * @param population - Genome population.
 * @param currentFrontIndices - Indices for the current front.
 */
function appendFront(
  paretoFronts: Network[][],
  population: Network[],
  currentFrontIndices: number[],
): void {
  // Step 1: map indices to genomes and append.
  paretoFronts.push(
    currentFrontIndices.map((genomeIndex) => population[genomeIndex]),
  );
}

/**
 * Increments the front rank counter.
 *
 * @param currentFrontRank - Current front rank.
 * @returns Incremented front rank.
 */
function incrementFrontRank(currentFrontRank: number): number {
  // Step 1: increment rank.
  return currentFrontRank + 1;
}

/**
 * Determines whether ranking should stop due to a safety guard.
 *
 * @param currentFrontRank - Current front rank after increment.
 * @param maxFrontRankGuard - Safety guard for ranking iterations.
 * @returns `true` if ranking should stop.
 */
function shouldStopFrontRanking(
  currentFrontRank: number,
  maxFrontRankGuard: number,
): boolean {
  // Step 1: check guard threshold.
  return currentFrontRank > maxFrontRankGuard;
}
