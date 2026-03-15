import type Network from '../../../architecture/network';
import {
  getAverage as getAverageScore,
  getFittest as getFittestGenome,
  sort as sortPopulationByScore,
} from '../selection';
import type { NeatLikeWithSelection } from '../core/selection.types';

/**
 * Public population-summary facade helpers for the stable `Neat` entrypoint.
 *
 * The broader selection chapter already owns ordering and parent-choice
 * behavior. This facade keeps only the stable class-friendly wrappers that the
 * top-level [src/neat.ts](src/neat.ts) entrypoint exposes: sorting the current
 * population, reading the fittest genome, and reading the average score.
 * Keeping that wrapper surface in `selection/facade/` makes the ownership story
 * match the generated docs and the direct-path chapter layout used by the newer
 * RNG, pruning, and telemetry facades.
 *
 * Invariant: this boundary only summarizes or reorders the current population.
 * It does not change parent-selection strategy, crossover policy, or mutation
 * behavior.
 */

/**
 * Narrow `Neat` host surface required by the public population-summary facade.
 *
 * The host keeps the contract small: a population, selection-aware options,
 * the existing sort hook, and the evaluation entrypoint used when callers ask
 * for summary data before scores have been computed.
 */
export interface NeatPopulationSummaryFacadeHost extends NeatLikeWithSelection {
  evaluate: () => void;
}

/**
 * Sort the population in descending fitness order.
 *
 * This preserves the historical `neat.sort()` behavior while keeping the
 * top-level class free of inline ordering details.
 *
 * @param host - `Neat` instance exposing population sorting state.
 * @returns Nothing. The population array is reordered in place.
 *
 * @example
 * ```ts
 * neat.sort();
 * console.log(neat.population[0].score);
 * ```
 */
export function sort(host: NeatPopulationSummaryFacadeHost): void {
  sortPopulationByScore.call(host as never);
}

/**
 * Return the fittest genome in the current population.
 *
 * If scores are missing, the underlying selection helper will trigger the
 * existing evaluation path before resolving the best genome.
 *
 * @param host - `Neat` instance exposing population and evaluation state.
 * @returns Genome with the highest current score.
 */
export function getFittest(host: NeatPopulationSummaryFacadeHost): Network {
  return getFittestGenome.call(host as never) as Network;
}

/**
 * Compute the average score across the current population.
 *
 * This is a compact inspection helper for telemetry, tests, and quick
 * debugging where the caller only needs the current mean fitness.
 *
 * @param host - `Neat` instance exposing population and evaluation state.
 * @returns Mean score across the current population.
 */
export function getAverage(host: NeatPopulationSummaryFacadeHost): number {
  return getAverageScore.call(host as never);
}
