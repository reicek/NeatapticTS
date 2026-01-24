/**
 * Lineage / ancestry analysis helpers for NEAT populations.
 *
 * These utilities were migrated from the historical implementation inside `src/neat.ts`
 * to keep core NEAT logic lean while still exposing educational metrics for users who
 * want to introspect evolutionary diversity.
 *
 * Glossary:
 *  - Genome: An individual network encoding (has a unique `_id` and optional `_parents`).
 *  - Ancestor Window: A shallow breadth‑first window (default depth = 4) over the lineage graph.
 *  - Jaccard Distance: 1 - |A ∩ B| / |A ∪ B|, measuring dissimilarity between two sets.
 */

import {
  calculateMaxSamplePairs,
  collectAncestorIds,
  computeAverageDistance,
  computePairDistances,
  createInitialQueue,
  hasMinimumPopulation,
  normalizeParentIds,
  sampleGenomePairs,
  type GenomeLike,
  type NeatLineageContext,
} from './neat.lineage.utils';

/** Common zero value for counters and defaults. */
const ZERO_VALUE = 0;

export type { GenomeLike, NeatLineageContext } from './neat.lineage.utils';

/**
 * Build the (shallow) ancestor ID set for a single genome using breadth‑first traversal.
 *
 * Traversal Strategy:
 * 1. Seed queue with the genome's parent IDs (depth = 1).
 * 2. Repeatedly dequeue, record its ID, and enqueue its parents with incremented depth.
 * 3. Stop exploring a branch once the configured depth window is exceeded.
 *
 * This bounded BFS gives a quick, memory‑friendly approximation of a genome's lineage neighborhood
 * that works well for diversity/uniqueness metrics without the expense of full historical graphs.
 *
 * Edge Cases:
 *  - Missing or empty `_parents` array ⇒ returns an empty set.
 *  - Orphan parent IDs (not found in population) are still added (their ID), but no further expansion occurs.
 *
 * Complexity (worst case): O(B^D) where B is average branching factor of parent links (usually <= 2)
 * and D = ANCESTOR_DEPTH_WINDOW (default 4) – so effectively constant for typical NEAT usage.
 *
 * @param this NEAT / evolutionary context; must provide `population` (array) for ID lookups.
 * @param genome Genome whose shallow ancestor set you want to compute.
 * @returns A Set of numeric ancestor IDs (deduplicated).
 *
 * @example
 * // Assuming `neat` is your NEAT instance and `g` a genome inside `neat.population`:
 * import { buildAnc } from 'neataptic';
 * const ancestorIds = buildAnc.call(neat, g);
 * console.log([...ancestorIds]); // -> e.g. [12, 4, 9]
 */
export function buildAnc(
  this: NeatLineageContext,
  genome: GenomeLike,
): Set<number> {
  // Local accumulator and normalized inputs.
  const ancestorSet = new Set<number>();
  const directParentIds = normalizeParentIds(genome);

  // Guard: no parents means no ancestors to collect.
  if (directParentIds.length === ZERO_VALUE) return ancestorSet;

  // 1) Seed the breadth-first queue.
  const queueEntries = createInitialQueue(directParentIds, this.population);
  // 2) Collect ancestor IDs within the configured depth window.
  const ancestorIds = collectAncestorIds(queueEntries, this.population);
  // 3) Fold into the output set and return.
  for (const ancestorId of ancestorIds) ancestorSet.add(ancestorId);

  return ancestorSet;
}

/**
 * Compute an "ancestor uniqueness" diversity metric for the current population.
 *
 * The metric = mean Jaccard distance between shallow ancestor sets of randomly sampled genome pairs.
 * A higher value indicates that individuals trace back to more distinct recent lineages (i.e. less
 * overlap in their ancestor windows), while a lower value indicates convergence toward similar ancestry.
 *
 * Why Jaccard Distance? It is scale‑independent: adding unrelated ancestors to both sets simultaneously
 * does not change the proportion of shared ancestry, and distance stays within [0,1].
 *
 * Sampling Strategy:
 *  - Uniformly sample up to N = min(30, populationPairs) distinct unordered pairs (with replacement on pair selection, but indices are adjusted to avoid self‑pairs).
 *  - For each pair, construct ancestor sets via `buildAnc` and accumulate their Jaccard distance.
 *  - Return the average (rounded to 3 decimal places) or 0 if insufficient samples.
 *
 * Edge Cases:
 *  - Population < 2 ⇒ returns 0 (cannot form pairs).
 *  - Both ancestor sets empty ⇒ pair skipped (no information about uniqueness).
 *
 * Performance: O(S * W) where S is sampled pair count (≤ 30) and W is bounded ancestor set size
 * (kept small by the depth window). This is intentionally lightweight for per‑generation telemetry.
 *
 * @param this NEAT context (`population` and `_getRNG` must exist).
 * @returns Mean Jaccard distance in [0,1]. Higher ⇒ more lineage uniqueness / diversity.
 *
 * @example
 * import { computeAncestorUniqueness } from 'neataptic';
 * // inside an evolutionary loop, with `neat` as your NEAT instance:
 * const uniqueness = computeAncestorUniqueness.call(neat);
 * console.log('Ancestor uniqueness:', uniqueness); // e.g. 0.742
 */
export function computeAncestorUniqueness(this: NeatLineageContext): number {
  // Local references and derived values.
  const buildAncestorSet = buildAnc.bind(this);
  const populationSize = this.population.length;
  const maxSamplePairs = calculateMaxSamplePairs(populationSize);

  // Guard: cannot sample pairs when the population is too small.
  if (!hasMinimumPopulation(populationSize)) return ZERO_VALUE;

  // 1) Sample candidate pairs.
  const sampledPairs = sampleGenomePairs(
    maxSamplePairs,
    populationSize,
    this._getRNG,
  );
  // 2) Compute distances for valid pairs.
  const pairDistances = computePairDistances(
    sampledPairs,
    this.population,
    buildAncestorSet,
  );
  // 3) Fold into the final average.
  return computeAverageDistance(pairDistances);
}
