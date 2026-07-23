/**
 * Deterministic variant selection for the Neatenstein asymmetric co-evolution
 * harness.
 *
 * Selection is index-stable and fully deterministic: the same population and
 * fitness scores always produce the same champion. When multiple variants share
 * the highest fitness, the variant with the lowest stable id wins. This keeps
 * reproduction reproducible from a seed and avoids relying on insertion order
 * or random tie-breaking.
 *
 * @module
 */

import type { FitnessScore, Individual } from './types.ts';

/**
 * Select the fittest individual from a population.
 *
 * The selection is deterministic: it always returns the same individual for
 * the same input array. Ties are resolved by choosing the variant with the
 * lowest {@link Individual.id}. Variants without a cached fitness score are
 * treated as `-Infinity` and therefore never win against scored variants.
 *
 * @param variants - Ordered population of evaluated individuals. Must contain
 *   at least one variant.
 * @returns The selected champion individual.
 * @throws Error when `variants` is empty.
 *
 * @example
 * ```ts
 * const champion = selectVariant([
 *   { id: 0, variant: genome0, fitness: 10 },
 *   { id: 1, variant: genome1, fitness: 30 },
 *   { id: 2, variant: genome2, fitness: 20 },
 * ]);
 * console.log(champion.id); // 1
 * ```
 */
export function selectVariant<TVariant>(
  variants: readonly Individual<TVariant>[],
): Individual<TVariant> {
  if (variants.length === 0) {
    throw new Error('Cannot select a variant from an empty population.');
  }

  return variants.reduce((best, current) => {
    const currentFitness: FitnessScore = current.fitness ?? -Infinity;
    const bestFitness: FitnessScore = best.fitness ?? -Infinity;

    const isFitter = currentFitness > bestFitness;
    const tiesAndLowerId =
      currentFitness === bestFitness && current.id < best.id;

    return isFitter || tiesAndLowerId ? current : best;
  });
}
