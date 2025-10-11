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

/**
 * Minimal shape assumed for a genome inside the NEAT population. Additional properties are
 * intentionally left open (index signature) because user implementations may extend genomes.
 */
export interface GenomeLike {
  /** Unique numeric identifier assigned when the genome is created. */
  _id: number;
  /** Optional list of parent genome IDs (could be 1 or 2 for sexual reproduction, or more in custom ops). */
  _parents?: number[];
  /** Allow arbitrary extra properties without forcing casts. */
  [key: string]: any; // eslint-disable-line @typescript-eslint/no-explicit-any
}

/** Expected `this` context for lineage helpers (a subset of the NEAT instance). */
export interface NeatLineageContext {
  /** Current evolutionary population (array of genomes). */
  population: GenomeLike[];
  /** RNG provider returning a PRNG function; shape taken from core NEAT implementation. */
  _getRNG: () => () => number;
}

/**
 * Depth window (in breadth-first layers) used when gathering ancestor IDs.
 * A small window keeps the metric inexpensive while still capturing recent lineage diversity.
 *
 * Rationale: Deep full ancestry can grow quickly and become O(N * lineage depth). Empirically,
 * a window of 4 gives a stable signal about short‑term innovation mixing without large cost.
 *
 * You can fork and increase this constant if you need deeper lineage metrics, but note that
 * performance will degrade roughly proportionally to the number of enqueued ancestor nodes.
 *
 * Example (changing the window):
 *   // (NOT exported) – modify locally before building docs
 *   // const ANCESTOR_DEPTH_WINDOW = 6; // capture deeper history
 */
const ANCESTOR_DEPTH_WINDOW = 4;

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
  genome: GenomeLike
): Set<number> {
  // Initialize ancestor ID accumulator.
  const ancestorSet = new Set<number>();

  // Fast exit if the genome has no recorded parents.
  if (!Array.isArray(genome._parents)) return ancestorSet;

  /**
   * BFS queue entries carrying the ancestor ID, current depth within the window,
   * and a direct reference to the ancestor genome (if located) so we can expand its parents.
   */
  const queue: { id: number; depth: number; genomeRef?: GenomeLike }[] = [];

  // Seed: enqueue each direct parent at depth = 1.
  for (const parentId of genome._parents) {
    queue.push({
      id: parentId,
      depth: 1,
      genomeRef: this.population.find((gm) => gm._id === parentId),
    });
  }

  // Breadth‑first expansion within the fixed depth window.
  while (queue.length) {
    // Dequeue (FIFO) to ensure breadth‑first order.
    const current = queue.shift()!;

    // Skip nodes that exceed the depth window limit.
    if (current.depth > ANCESTOR_DEPTH_WINDOW) continue;

    // Record ancestor ID (dedup automatically handled by Set semantics).
    if (current.id != null) ancestorSet.add(current.id);

    // If we have a concrete genome reference with parents, enqueue them for the next layer.
    if (current.genomeRef && Array.isArray(current.genomeRef._parents)) {
      for (const parentId of current.genomeRef._parents) {
        queue.push({
          id: parentId,
          // Depth increases as we move one layer further away from the focal genome.
          depth: current.depth + 1,
          genomeRef: this.population.find((gm) => gm._id === parentId),
        });
      }
    }
  }
  return ancestorSet;
}

/** Maximum number of distinct genome pairs to sample when computing uniqueness. */
const MAX_UNIQUENESS_SAMPLE_PAIRS = 30;

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
  // Bind builder once for clarity & micro‑efficiency.
  const buildAncestorSet = buildAnc.bind(this);

  // Accumulators for (distance sum, sampled pair count).
  let sampledPairCount = 0;
  let jaccardDistanceSum = 0;

  /**
   * Maximum number of pair samples respecting both the cap constant and the total
   * possible distinct unordered pairs nC2 = n(n-1)/2.
   */
  const maxSamplePairs = Math.min(
    MAX_UNIQUENESS_SAMPLE_PAIRS,
    (this.population.length * (this.population.length - 1)) / 2
  );

  // Main sampling loop.
  for (let t = 0; t < maxSamplePairs; t++) {
    if (this.population.length < 2) break; // not enough genomes to form pairs

    // Randomly pick first genome index.
    const indexA = Math.floor(this._getRNG()() * this.population.length);
    // Pick second index (avoid identical -> simple offset if collision).
    let indexB = Math.floor(this._getRNG()() * this.population.length);
    if (indexB === indexA) indexB = (indexB + 1) % this.population.length;

    // Build ancestor sets for the pair.
    const ancestorSetA = buildAncestorSet(this.population[indexA]);
    const ancestorSetB = buildAncestorSet(this.population[indexB]);

    // Skip if both sets are empty (no lineage info to compare yet).
    if (ancestorSetA.size === 0 && ancestorSetB.size === 0) continue;

    // Compute intersection size.
    let intersectionCount = 0;
    for (const id of ancestorSetA)
      if (ancestorSetB.has(id)) intersectionCount++;

    // Union size = |A| + |B| - |A ∩ B| (guard against divide-by-zero).
    const unionSize =
      ancestorSetA.size + ancestorSetB.size - intersectionCount || 1;

    // Jaccard distance = 1 - similarity.
    const jaccardDistance = 1 - intersectionCount / unionSize;

    // Accumulate for averaging.
    jaccardDistanceSum += jaccardDistance;
    sampledPairCount++;
  }

  // Average (3 decimal places) or 0 if no valid samples.
  const ancestorUniqueness = sampledPairCount
    ? +(jaccardDistanceSum / sampledPairCount).toFixed(3)
    : 0;
  return ancestorUniqueness;
}
