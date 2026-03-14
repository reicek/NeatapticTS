import {
  calculateMaxSamplePairs,
  collectAncestorIds,
  computeAverageDistance,
  computePairDistances,
  createInitialQueue,
  hasMinimumPopulation,
  normalizeParentIds,
  sampleGenomePairs,
} from './core/lineage.core';
import type {
  GenomeLike,
  NeatLineageContext,
} from './core/lineage.types';

/** Common zero value for counters and defaults. */
const ZERO_VALUE = 0;

export type { GenomeLike, NeatLineageContext } from './core/lineage.types';

/**
 * Lineage and ancestry analysis helpers for NEAT populations.
 *
 * The root lineage chapter keeps the two public ancestry metrics together while
 * moving the queue mechanics, sampling helpers, and narrow runtime types into
 * `core/`.
 *
 * - `core/` explains ancestor traversal, sampled pair generation, and Jaccard-distance aggregation.
 */

/**
 * Build the shallow ancestor ID set for a genome using breadth-first traversal.
 *
 * @param this - NEAT lineage context providing the current population.
 * @param genome - Genome whose shallow ancestor set should be computed.
 * @returns Set of ancestor IDs within the configured depth window.
 */
export function buildAnc(
  this: NeatLineageContext,
  genome: GenomeLike,
): Set<number> {
  const ancestorSet = new Set<number>();
  const directParentIds = normalizeParentIds(genome);

  if (directParentIds.length === ZERO_VALUE) {
    return ancestorSet;
  }

  // Step 1: Seed the breadth-first queue from the direct parents.
  const queueEntries = createInitialQueue(directParentIds, this.population);

  // Step 2: Collect ancestor IDs within the configured depth window.
  const ancestorIds = collectAncestorIds(queueEntries, this.population);

  // Step 3: Fold the collected IDs into the output set.
  for (const ancestorId of ancestorIds) {
    ancestorSet.add(ancestorId);
  }

  return ancestorSet;
}

/**
 * Compute the ancestor uniqueness metric for the current population.
 *
 * @param this - NEAT lineage context exposing the population and RNG provider.
 * @returns Mean sampled Jaccard distance across shallow ancestor sets.
 */
export function computeAncestorUniqueness(
  this: NeatLineageContext,
): number {
  const buildAncestorSet = buildAnc.bind(this);
  const populationSize = this.population.length;
  const maxSamplePairs = calculateMaxSamplePairs(populationSize);

  if (!hasMinimumPopulation(populationSize)) {
    return ZERO_VALUE;
  }

  // Step 1: Sample candidate genome pairs.
  const sampledPairs = sampleGenomePairs(
    maxSamplePairs,
    populationSize,
    this._getRNG,
  );

  // Step 2: Compute pairwise ancestor distances.
  const pairDistances = computePairDistances(
    sampledPairs,
    this.population,
    buildAncestorSet,
  );

  // Step 3: Fold the sampled distances into the final average.
  return computeAverageDistance(pairDistances);
}