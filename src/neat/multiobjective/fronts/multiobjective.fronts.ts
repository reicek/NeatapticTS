import type Network from '../../../architecture/network/network';
import type { NetworkWithMOAnnotations } from '../shared/multiobjective.types';
import type { DominanceState } from '../dominance/multiobjective.dominance';

/**
 * Frontier construction for NEAT multi-objective ranking.
 *
 * This chapter picks up where `dominance/` stops. Once the pairwise pass has
 * recorded how many times each genome is dominated and which genomes each row
 * blocks, this file peels the population into ordered Pareto fronts and writes
 * the `_moRank` annotations that later selection and telemetry reads expect.
 *
 * The boundary is intentionally narrow:
 * - `objectives/` owns safe value extraction and matrix assembly
 * - `dominance/` owns pairwise comparison and first-front discovery
 * - this file owns frontier peeling and rank annotation
 * - `crowding/` owns within-front spacing after ranks already exist
 *
 * That split matters because front construction depends on one stable contract:
 * matrix row order, population order, and dominance-state indices must all keep
 * referring to the same genome while fronts are peeled layer by layer. This
 * file therefore works only with index lists and one shared population array;
 * it does not rebuild the matrix or recompute pairwise dominance.
 *
 * Conceptually, the pass answers three questions:
 * 1. Which indices already belong to the first front?
 * 2. When a front is removed, which domination counts should drop to zero next?
 * 3. Which genomes should be annotated with the current Pareto rank before the
 *    next frontier is collected?
 *
 * ```mermaid
 * flowchart TD
 *   A[Dominance state with firstFrontIndices] --> B[Take current front indices]
 *   B --> C[Annotate genomes with current _moRank]
 *   C --> D[Relax domination counts for dominated neighbors]
 *   D --> E[Collect indices whose counts reach zero]
 *   E --> F[Append current front to ordered results]
 *   F --> G{Guard exceeded?}
 *   G -->|No| B
 *   G -->|Yes| H[Stop with bounded result]
 * ```
 *
 * Read this chapter when the missing question is "how did the bookkeeping from
 * the dominance pass become ordered Pareto fronts?" Read `dominance/` first if
 * the missing context is how `firstFrontIndices` or domination counts were
 * computed in the first place.
 */

// The separator below keeps the generated README chapter intro separate from
// the first exported symbol description.

/**
 * Maximum number of Pareto fronts to allow during ranking before aborting.
 *
 * This is a defensive guard against pathological conditions (e.g., corrupted
 * dominance bookkeeping) that could otherwise cause long/infinite loops.
 * A healthy ranking pass should terminate well before this threshold, so the
 * constant exists as a safety rail rather than a normal control knob.
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
 * The implementation is intentionally breadth-first. It never revisits pairwise
 * comparisons; instead it trusts the dominance-state handoff and repeatedly
 * relaxes domination counts until the next layer becomes visible.
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
 * This helper is the inner loop of frontier peeling: mark every genome in the
 * current front with the same rank, then remove each genome's blocking
 * influence so newly non-dominated neighbors can surface as the next front.
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
 * Ranking is stored directly on the genome so later selection, telemetry, and
 * archive helpers can read one stable annotation instead of carrying a parallel
 * rank table beside the population.
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
 * Every dominated neighbor starts this pass with one known blocker: the genome
 * that just joined the current front. Decrementing its domination count models
 * removing that blocker. When the count reaches zero, the neighbor has no
 * remaining dominating opponents and can join the next frontier.
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
 * The accumulator stores genomes, not indices, because the returned fronts are
 * meant to be consumed by later crowding and archive code. The conversion from
 * stable indices to genome references happens only after the current layer has
 * been fully identified.
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
 * Keeping rank advancement in its own helper makes the frontier loop easier to
 * read and keeps the guard check phrased in terms of the next rank value.
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
 * The guard is intentionally blunt: if frontier peeling advances beyond the
 * configured maximum rank, the function stops and returns the fronts collected
 * so far rather than risking a pathological or corrupted infinite loop.
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
