import type { NeatLike } from './neat.types';
import {
  calculateTotalScore,
  DEFAULT_SCORE,
  ensurePopulationEvaluated,
  ensurePopulationSortedDescending,
  FIRST_INDEX,
  type GenomeWithScore,
  type NeatLikeWithSelection,
  selectParentByStrategy,
} from './neat.selection.utils';

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
} from './neat.selection.utils';

/**
 * Sorts the internal population in place by descending fitness.
 *
 * This method mutates the `population` array on the Neat instance so that
 * the genome with the highest `score` appears at index 0. It treats missing
 * scores as 0.
 *
 * Example:
 * const neat = new Neat(...);
 * neat.sort();
 * console.log(neat.population[0].score); // highest score
 *
 * Notes for documentation generators: this is a small utility used by many
 * selection and evaluation routines; it intentionally sorts in-place for
 * performance and to preserve references to genome objects.
 *
 * @this NeatLike - the Neat instance with `population` to sort
 */
export function sort(this: NeatLike): void {
  // Sort population descending by score (highest score first). Missing
  // scores (undefined/null) are treated as 0 using the nullish coalescing operator.
  const internal = this as unknown as NeatLikeWithSelection;
  internal.population.sort(
    (left, right) =>
      (right.score ?? DEFAULT_SCORE) - (left.score ?? DEFAULT_SCORE),
  );
}

/**
 * Select a parent genome according to the configured selection strategy.
 *
 * Supported strategies (via `options.selection.name`):
 * - 'POWER'              : biased power-law selection (exploits best candidates)
 * - 'FITNESS_PROPORTIONATE': roulette-wheel style selection proportional to fitness
 * - 'TOURNAMENT'         : pick N random competitors and select the best with probability p
 *
 * This function intentionally makes no changes to the population except in
 * the POWER path where a quick sort may be triggered to ensure descending
 * order.
 *
 * Examples:
 * // POWER selection (higher power => more exploitation)
 * neat.options.selection = { name: 'POWER', power: 2 };
 * const parent = neat.getParent();
 *
 * // Tournament selection (size 3, 75% probability to take top of tournament)
 * neat.options.selection = { name: 'TOURNAMENT', size: 3, probability: 0.75 };
 * const parent2 = neat.getParent();
 *
 * @this NeatLike - the Neat instance containing `population`, `options`, and `_getRNG`
 * @returns A genome object chosen as the parent according to the selection strategy
 */
export function getParent(this: NeatLike): GenomeWithScore {
  const internal = this as unknown as NeatLikeWithSelection;

  // Step 1: resolve parent selection via utility helpers.
  const selectedParent = selectParentByStrategy(internal);

  // Step 2: return the selected parent.
  return selectedParent;
}

/**
 * Return the fittest genome in the population.
 *
 * This will trigger an `evaluate()` if genomes have not been scored yet, and
 * will ensure the population is sorted so index 0 contains the fittest.
 *
 * Example:
 * const best = neat.getFittest();
 * console.log(best.score);
 *
 * @this NeatLike - the Neat instance containing `population` and `evaluate`.
 * @returns The genome object judged to be the fittest (highest score).
 */
export function getFittest(this: NeatLike): GenomeWithScore {
  const internal = this as unknown as NeatLikeWithSelection;
  const population = internal.population;

  // Step 1: ensure the population has scores.
  ensurePopulationEvaluated(internal);

  // Step 2: ensure the population is sorted descending by score.
  ensurePopulationSortedDescending(internal);

  // Step 3: return the fittest genome (index 0).
  return population[FIRST_INDEX];
}

/**
 * Compute the average (mean) fitness across the population.
 *
 * If genomes have not been evaluated yet this will call `evaluate()` so
 * that scores exist. Missing scores are treated as 0.
 *
 * Example:
 * const avg = neat.getAverage();
 * console.log(`Average fitness: ${avg}`);
 *
 * @this NeatLike - the Neat instance containing `population` and `evaluate`.
 * @returns The mean fitness as a number.
 */
export function getAverage(this: NeatLike): number {
  const internal = this as unknown as NeatLikeWithSelection;
  const population = internal.population;

  // Step 1: ensure scores exist.
  ensurePopulationEvaluated(internal);

  // Step 2: compute total fitness.
  const totalScore = calculateTotalScore(population);

  // Step 3: fold into the mean.
  return totalScore / population.length;
}
