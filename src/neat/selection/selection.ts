import type { NeatLike } from '../neat.types';
import {
  calculateTotalScore,
  DEFAULT_SCORE,
  ensurePopulationEvaluated,
  ensurePopulationSortedDescending,
  FIRST_INDEX,
  selectParentByStrategy,
} from './core/selection.core';
import type {
  GenomeWithScore,
  NeatLikeWithSelection,
} from './core/selection.types';

export {
  DEFAULT_POWER,
  DEFAULT_TOURNAMENT_SIZE,
  DEFAULT_TOURNAMENT_PROBABILITY,
  DEFAULT_SCORE,
  FIRST_INDEX,
  SECOND_INDEX,
  LAST_INDEX_OFFSET,
  LOOP_INDEX_INCREMENT,
  LAST_ELEMENT_INDEX,
  INITIAL_TOTAL_FITNESS,
  INITIAL_MOST_NEGATIVE_SCORE,
  INITIAL_CUMULATIVE_FITNESS,
} from './core/selection.core';

/**
 * Parent-selection helpers for the NEAT controller.
 *
 * The root selection chapter keeps the public controller-facing methods small
 * and readable, while `core/` holds the selection strategy mechanics, constants,
 * and narrow runtime contracts.
 *
 * - `core/` explains score defaults, ordering checks, and parent-selection strategies.
 */

/**
 * Sort the internal population in place by descending fitness.
 *
 * @this NeatLike NEAT instance whose population should be sorted.
 * @returns Nothing. The population array is reordered in place.
 */
export function sort(this: NeatLike): void {
  const internal = this as unknown as NeatLikeWithSelection;
  internal.population.sort(
    (left, right) =>
      (right.score ?? DEFAULT_SCORE) - (left.score ?? DEFAULT_SCORE),
  );
}

/**
 * Select a parent genome according to the configured selection strategy.
 *
 * @this NeatLike NEAT instance containing population, selection options, and RNG access.
 * @returns Genome chosen according to the active selection strategy.
 */
export function getParent(this: NeatLike): GenomeWithScore {
  const internal = this as unknown as NeatLikeWithSelection;

  // Step 1: Resolve parent selection via the core strategy helpers.
  return selectParentByStrategy(internal);
}

/**
 * Return the fittest genome in the population.
 *
 * @this NeatLike NEAT instance containing population and evaluation support.
 * @returns Genome with the highest current score.
 */
export function getFittest(this: NeatLike): GenomeWithScore {
  const internal = this as unknown as NeatLikeWithSelection;
  const population = internal.population;

  // Step 1: Ensure the population has scores.
  ensurePopulationEvaluated(internal);

  // Step 2: Ensure the population is ordered by descending score.
  ensurePopulationSortedDescending(internal);

  // Step 3: Return the best genome.
  return population[FIRST_INDEX];
}

/**
 * Compute the average fitness across the population.
 *
 * @this NeatLike NEAT instance containing population and evaluation support.
 * @returns Mean fitness across the current population.
 */
export function getAverage(this: NeatLike): number {
  const internal = this as unknown as NeatLikeWithSelection;
  const population = internal.population;

  // Step 1: Ensure the population has scores.
  ensurePopulationEvaluated(internal);

  // Step 2: Fold total fitness into the arithmetic mean.
  const totalScore = calculateTotalScore(population);
  return totalScore / population.length;
}