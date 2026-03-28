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
import type { GenomeLike, NeatLineageContext } from './core/lineage.types';

/** Common zero value for counters and defaults. */
const ZERO_VALUE = 0;

export type { GenomeLike, NeatLineageContext } from './core/lineage.types';

/**
 * Lineage and ancestry analysis helpers for NEAT populations.
 *
 * The lineage boundary answers a different question from the broader diversity
 * chapter: not just whether genomes look different now, but whether they come
 * from meaningfully different recent families. That ancestry view matters when
 * telemetry, diagnostics, or adaptive policy need to detect whether a run is
 * still exploring multiple lineages or quietly collapsing onto descendants of a
 * small ancestor set.
 *
 * The root chapter keeps two public read models together:
 *
 * - `buildAnc()` builds one genome's shallow ancestor set.
 * - `computeAncestorUniqueness()` turns many shallow ancestor sets into one
 *   sampled population signal.
 *
 * Read the chapter in this order:
 *
 * - `buildAnc()` when you want the raw ancestry evidence for one genome.
 * - `computeAncestorUniqueness()` when you want the controller-facing summary
 *   used by telemetry, diagnostics, or adaptive lineage pressure.
 * - `core/` when you need the breadth-first traversal, pair sampling, or
 *   Jaccard-distance mechanics behind the public helpers.
 *
 * ```mermaid
 * flowchart TD
 *   Genome[Genome with parent ids] --> Ancestors[Shallow ancestor set]
 *   Population[Population genomes] --> Pairs[Sample genome pairs]
 *   Ancestors --> Distance[Compare ancestor overlap]
 *   Pairs --> Distance
 *   Distance --> Uniqueness[Ancestor uniqueness summary]
 *   Uniqueness --> Consumers[Telemetry diagnostics and adaptive lineage policy]
 * ```
 *
 * The root chapter stays compact on purpose. `core/` owns the queue mechanics,
 * sampled pair generation, and distance aggregation so this file can stay
 * focused on what the ancestry reads mean at the controller surface.
 */

/**
 * Build the shallow ancestor ID set for a genome using breadth-first traversal.
 *
 * "Shallow" means this helper intentionally stops after a small ancestry
 * window instead of walking the entire historical tree. That keeps the result
 * useful for runtime telemetry: it captures the recent family neighborhood that
 * most directly explains current convergence or branching without turning every
 * read into an unbounded genealogy crawl.
 *
 * Use this when you need the raw ancestry evidence behind later population
 * summaries. The returned set is most helpful for pairwise overlap checks,
 * debugging parent tracking, or validating that speciation and reproduction are
 * still producing multiple recent family branches.
 *
 * @example
 * ```ts
 * const ancestorIds = neat.buildAnc(neat.population[0]);
 *
 * console.log(ancestorIds.has(42));
 * ```
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
 * This is the controller-facing lineage summary. It samples genome pairs,
 * builds a shallow ancestor set for each genome in the pair, then measures how
 * different those ancestor sets are using Jaccard distance.
 *
 * Interpret the returned value as a bounded trend signal:
 *
 * - lower values mean many genomes still share recent ancestors,
 * - higher values mean recent ancestry is spread across more distinct family
 *   branches.
 *
 * The helper is intentionally sampled rather than exhaustive so telemetry and
 * adaptive controllers can reuse it during a run without paying the full cost
 * of comparing every genome pair. It complements the diversity chapter by
 * focusing on ancestry overlap rather than structural size or compatibility
 * distance.
 *
 * @example
 * ```ts
 * const ancestorUniqueness = neat.computeAncestorUniqueness();
 *
 * if (ancestorUniqueness < 0.2) {
 *   console.log('Recent ancestry is collapsing into a narrow family band.');
 * }
 * ```
 *
 * @param this - NEAT lineage context exposing the population and RNG provider.
 * @returns Mean sampled Jaccard distance across shallow ancestor sets.
 */
export function computeAncestorUniqueness(this: NeatLineageContext): number {
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
