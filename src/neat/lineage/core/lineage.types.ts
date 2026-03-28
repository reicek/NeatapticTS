/**
 * Core contracts for lineage-analysis mechanics.
 *
 * This file defines the narrow data shapes behind the controller-facing
 * lineage helpers. The public lineage chapter talks about recent family
 * overlap as a runtime signal; this core contract layer explains which pieces
 * of state the mechanics actually need to produce that signal.
 *
 * The important constraint is intentional minimalism. Ancestor traversal does
 * not need the full genome implementation, and sampled uniqueness does not need
 * the whole `Neat` controller. By describing only ids, parent links,
 * population access, and RNG access, this layer keeps the lineage utilities
 * reusable and easy to reason about.
 *
 * Read the contracts in this order:
 *
 * 1. `GenomeLike` describes the smallest ancestry-bearing genome shape,
 * 2. `NeatLineageContext` shows what the controller must supply for sampled
 *    lineage reads,
 * 3. `AncestorQueueEntry` models one step in breadth-first ancestor expansion,
 * 4. `GenomeIndexPair` models one sampled comparison between genomes.
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   genome[GenomeLike id plus parent ids]:::base --> queue[AncestorQueueEntry breadth-first work item]:::accent
 *   context[NeatLineageContext population plus RNG]:::base --> pairs[GenomeIndexPair sampled comparison]:::base
 *   queue --> ancestors[Ancestor sets]:::base
 *   pairs --> ancestors
 *   ancestors --> distance[Jaccard lineage distance]:::accent
 * ```
 */

/**
 * Minimal genome shape used by lineage helpers.
 *
 * Lineage analysis only needs two structural facts from each genome: a stable
 * identifier and the identifiers of its recorded parents. Everything else is
 * intentionally left open-ended so ancestry helpers can run against richer
 * runtime objects without importing or depending on all of their fields.
 *
 * In practice this interface is the bridge between reproduction-time lineage
 * bookkeeping and read-side lineage metrics. If those ids are present and
 * stable, the rest of the ancestry pipeline can stay decoupled from mutation,
 * evaluation, telemetry, and speciation internals.
 */
export interface GenomeLike {
  /** Unique numeric identifier assigned when the genome is created. */
  _id: number;
  /** Optional list of parent genome IDs. */
  _parents?: number[];
  /** Additional runtime metadata carried by the genome. */
  [key: string]: unknown;
}

/**
 * Minimal NEAT context required by lineage helpers.
 *
 * The lineage boundary only needs the current population and the RNG provider
 * used for sampled ancestor uniqueness. That small host contract makes the
 * ownership model explicit: lineage reporting is a read-side controller
 * concern, not a stateful subsystem with its own storage or mutation rules.
 *
 * The population supplies the ancestry graph to inspect. The RNG provider keeps
 * sampled uniqueness deterministic so the same run can replay the same sampled
 * comparisons during tests or exported-state debugging.
 */
export interface NeatLineageContext {
  /** Current evolutionary population. */
  population: GenomeLike[];
  /** RNG provider returning a PRNG function. */
  _getRNG: () => () => number;
}

/**
 * Index pair representing one sampled genome comparison.
 *
 * The lineage uniqueness metric does not compare every possible pair in large
 * populations. Instead it samples a bounded set of pairs, then asks how much
 * recent ancestry overlaps inside each comparison.
 */
export interface GenomeIndexPair {
  /** Population index of the first genome in the sampled comparison. */
  firstIndex: number;
  /** Population index of the second genome in the sampled comparison. */
  secondIndex: number;
}

/**
 * Queue entry used during breadth-first ancestor traversal.
 *
 * Each entry records which ancestor id is being explored, how deep that
 * ancestor sits relative to the original genome, and an optional cached genome
 * reference so later traversal steps can enqueue that ancestor's parents.
 */
export interface AncestorQueueEntry {
  /** Identifier of the ancestor currently being visited. */
  ancestorId: number;
  /** Breadth-first depth relative to the genome whose ancestry is being built. */
  depth: number;
  /** Optional resolved genome object for the ancestor id. */
  genomeRef?: GenomeLike;
}
