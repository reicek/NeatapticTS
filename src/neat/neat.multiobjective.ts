/**
 * Multi-objective helpers (fast non-dominated sorting + crowding distance).
 * Extracted from `neat.ts` to keep the core class slimmer.
 */
import type Network from '../architecture/network';
import {
  archiveParetoFrontsIfEnabled,
  assignCrowdingDistances,
  buildDominanceState,
  buildParetoFronts,
  buildValuesMatrix,
  MAX_PARETO_FRONT_RANK_GUARD,
} from './neat.multiobjective.utils';
import type {
  NeatLikeWithMultiObjective,
  ObjectiveDescriptor,
} from './neat.multiobjective.utils';

/**
 * Perform fast non-dominated sorting and compute crowding distances for a
 * population of networks (genomes). This implements a standard NSGA-II style
 * non-dominated sorting followed by crowding distance assignment.
 *
 * The function annotates genomes with two fields used elsewhere in the codebase:
 * - `_moRank`: integer Pareto front rank (0 = best/frontier)
 * - `_moCrowd`: numeric crowding distance (higher is better; Infinity for
 *   boundary solutions)
 *
 * Example
 * ```ts
 * // inside a Neat class that exposes `_getObjectives()` and `options`
 * const fronts = fastNonDominated.call(neatInstance, population);
 * // fronts[0] is the Pareto-optimal set
 * ```
 *
 * Notes for documentation generation:
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
