/**
 * Minimal genome shape used by lineage helpers.
 *
 * Lineage analysis only needs the genome id and optional parent ids, so this
 * boundary intentionally leaves the rest of the genome open-ended. That keeps
 * the ancestry helpers reusable anywhere the controller can provide stable
 * genome identifiers and recorded parentage, even if the surrounding runtime
 * object carries much richer state.
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
 * used for sampled ancestor uniqueness. The small host contract makes it clear
 * that lineage reporting is a read-side controller concern: it inspects the
 * current population and sampling behavior without owning mutation, speciation,
 * or telemetry storage itself.
 */
export interface NeatLineageContext {
  /** Current evolutionary population. */
  population: GenomeLike[];
  /** RNG provider returning a PRNG function. */
  _getRNG: () => () => number;
}

/** Index pair representing a sampled genome pair. */
export interface GenomeIndexPair {
  firstIndex: number;
  secondIndex: number;
}

/** Queue entry used during breadth-first ancestor traversal. */
export interface AncestorQueueEntry {
  ancestorId: number;
  depth: number;
  genomeRef?: GenomeLike;
}
