import type Network from '../architecture/network';
import type { ObjectiveDescriptor } from './neat.multiobjective.utils.types';

/**
 * Safely reads a single objective value for a given genome.
 *
 * This wraps the descriptor `accessor` in a `try/catch` so that a buggy
 * objective function cannot crash multi-objective ranking.
 *
 * Notes:
 * - If the accessor throws, this returns `0` (a neutral-ish fallback).
 * - Callers should prefer to surface accessor errors during development;
 *   this helper is intentionally defensive for long-running training loops.
 *
 * @param genomeItem - Genome to evaluate.
 * @param descriptor - Objective descriptor providing an accessor.
 * @returns Numeric objective value; `0` if the accessor throws.
 *
 * @example
 * ```ts
 * const score = readObjectiveValue(genome, { accessor: (g) => g.score ?? 0 });
 * ```
 */
export function readObjectiveValue(
  genomeItem: Network,
  descriptor: ObjectiveDescriptor,
): number {
  // Step 1: guard against accessor exceptions.
  try {
    return descriptor.accessor(genomeItem);
  } catch {
    return 0;
  }
}

/**
 * Builds an objective vector for a single genome.
 *
 * The resulting array order matches the `descriptors` order exactly.
 * Each component is read via {@link readObjectiveValue} so individual
 * objective accessors are fault-tolerant.
 *
 * @param genomeItem - Genome to evaluate.
 * @param descriptors - Objective descriptors (vector schema).
 * @returns Objective value vector (length equals `descriptors.length`).
 */
export function buildGenomeValues(
  genomeItem: Network,
  descriptors: ObjectiveDescriptor[],
): number[] {
  // Step 1: map descriptors to safe values.
  return descriptors.map((descriptor) =>
    readObjectiveValue(genomeItem, descriptor),
  );
}

/**
 * Builds a population-wide objective value matrix.
 *
 * The resulting matrix is indexed as `[genomeIndex][objectiveIndex]` where
 * `genomeIndex` matches the input `population` order.
 *
 * @param population - Genomes to evaluate (population order is preserved).
 * @param descriptors - Objective descriptors (column schema).
 * @returns Objective values matrix.
 */
export function buildValuesMatrix(
  population: Network[],
  descriptors: ObjectiveDescriptor[],
): number[][] {
  // Step 1: map population to vectors of objective values.
  return population.map((genomeItem) =>
    buildGenomeValues(genomeItem, descriptors),
  );
}
