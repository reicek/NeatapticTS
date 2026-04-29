/**
 * Root orchestration for NEAT multi-objective ranking.
 *
 * This chapter keeps the public ranking flow small and readable: collect the
 * active objective schema, precompute the population value matrix, resolve
 * pairwise dominance into fronts, assign NSGA-II style crowding distances, and
 * archive the leading fronts for later inspection.
 *
 * The neighboring `objectives/`, `dominance/`, `fronts/`, and `crowding/`
 * chapters own the narrow mechanics. This file exists so a reader can learn
 * the ranking pipeline from top to bottom without digging through those lower-
 * level helpers first.
 *
 * Ownership boundary: this ranking flow is controller-owned policy, not
 * canonical genome identity. `_moRank` and `_moCrowd` are temporary annotations
 * for selection, telemetry, and export; they do not become compatibility inputs
 * or historical markings unless a later opt-in policy explicitly says so.
 *
 * Multi-objective ranking answers a different question than ordinary single-
 * score selection. Instead of asking "which genome has the highest score?",
 * this chapter asks "which genomes are still competitive once several goals
 * must be satisfied at the same time?"
 * The result is a layered view of the population:
 * - Pareto fronts separate clearly dominated genomes from still-competitive ones
 * - crowding distances prefer spread along the frontier instead of collapsing to one region
 * - optional archiving preserves the leading fronts for later telemetry and inspection
 *
 * A useful reading order is:
 * 1. this orchestration file for the top-level ranking flow
 * 2. `objectives/` for value extraction and direction handling
 * 3. `dominance/` for pairwise comparison rules
 * 4. `fronts/` and `crowding/` for frontier construction and diversity on the frontier
 */
import type Network from '../../architecture/network/network';
import { archiveParetoFrontsIfEnabled } from './archive/multiobjective.archive';
import { buildValuesMatrix } from './objectives/multiobjective.objectives';
import { buildDominanceState } from './dominance/multiobjective.dominance';
import {
  buildParetoFronts,
  MAX_PARETO_FRONT_RANK_GUARD,
} from './fronts/multiobjective.fronts';
import { assignCrowdingDistances } from './crowding/multiobjective.crowding';
import type {
  NeatLikeWithMultiObjective,
  ObjectiveDescriptor,
} from './shared/multiobjective.types';

/**
 * Perform fast non-dominated sorting and compute crowding distances for a
 * population of networks (genomes). This implements a standard NSGA-II style
 * non-dominated sorting followed by crowding distance assignment.
 *
 * Conceptually, the function runs in four stages:
 * 1. read the active objective schema from the controller,
 * 2. build one objective-value vector per genome,
 * 3. resolve dominance relationships into ordered Pareto fronts,
 * 4. assign crowding distances so selection can prefer spread within each front.
 *
 * That final crowding step matters because a frontier alone only tells you that
 * several genomes are non-dominated. It does not tell you whether those genomes
 * represent a broad tradeoff surface or a tightly clustered patch of nearly
 * identical solutions.
 *
 * The function annotates genomes with two controller-owned fields used
 * elsewhere in the codebase:
 * - `_moRank`: integer Pareto front rank (0 = best/frontier)
 * - `_moCrowd`: numeric crowding distance (higher is better; Infinity for
 *   boundary solutions)
 *
 * Treat both fields as current-ranking metadata rather than as canonical genome
 * traits.
 *
 * This orchestration layer also decides when the leading fronts should be
 * archived for later telemetry or inspection. That keeps the ranking story in
 * one place: compute the competitive ordering now, and optionally preserve the
 * resulting frontier snapshot for later analysis.
 *
 * Example:
 * ```ts
 * // inside a Neat class that exposes `_getObjectives()` and Pareto archiving options
 * const fronts = fastNonDominated.call(neatInstance, population);
 *
 * // fronts[0] contains the current Pareto-optimal genomes
 * // genomes inside each front now also carry `_moRank` and `_moCrowd`
 * ```
 *
 * Read the return value like this:
 * - `fronts[0]` is the current non-dominated frontier
 * - `fronts[1]` contains genomes dominated only by the first front
 * - larger `_moCrowd` values indicate genomes that sit in less crowded regions of the same front
 *
 * Important assumptions:
 * - Each objective descriptor returned by `_getObjectives()` must have an
 *   `accessor(genome: Network): number` function and may include
 *   `direction: 'max' | 'min'` to indicate optimization direction.
 * - Accessor failures are guarded and will yield a default value of 0.
 *
 * @param this - Neat instance providing `_getObjectives()`, `options` and
 *   `_paretoArchive` fields (function is meant to be invoked using `.call`)
 * @param pop - population array of `Network` genomes to be ranked
 * @returns Array of Pareto fronts; each front is an array of `Network` genomes.
 */
export function fastNonDominated(
  this: NeatLikeWithMultiObjective,
  pop: Network[],
): Network[][] {
  /**
   * const: objective descriptors array
   * Short description: descriptors returned by the Neat instance that define
   * how to extract objective values from genomes and whether each objective is
   * maximized or minimized.
   *
   * Each descriptor must provide:
   * - `accessor(genome: Network): number` — returns numeric score for genome
   * - `direction?: 'max' | 'min'` — optional optimization direction (default 'max')
   */
  const objectiveDescriptors: ObjectiveDescriptor[] = this._getObjectives();

  /**
   * const: objective values matrix
   * Short description: precomputed numeric values of each objective for every
   * genome in the population. This avoids repeated accessor calls during
   * pairwise domination checks.
   *
   * Shape: `[population.length][objectives.length]` where row i contains the
   * objective vector for `pop[i]`.
   */
  const valuesMatrix: number[][] = buildValuesMatrix(pop, objectiveDescriptors);

  const dominanceState = buildDominanceState(
    valuesMatrix,
    objectiveDescriptors,
  );
  const paretoFronts = buildParetoFronts(
    pop,
    dominanceState,
    MAX_PARETO_FRONT_RANK_GUARD,
  );

  assignCrowdingDistances(
    paretoFronts,
    valuesMatrix,
    objectiveDescriptors,
    pop,
  );
  archiveParetoFrontsIfEnabled(this, paretoFronts);

  return paretoFronts;
}
