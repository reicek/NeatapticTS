/**
 * Core contracts for compatibility-distance mechanics.
 *
 * This chapter explains how the root compatibility helper turns two genomes
 * into one stable distance signal without dragging the full controller surface
 * into every comparison. The boundary stays deliberately small: these types
 * describe only the data needed to align genes, classify mismatches, and fold
 * the final score.
 *
 * Read this core layer when you want the "how" behind the root chapter:
 * how generation-scoped caches stay valid, how sorted innovation lists are
 * derived, how a linear merge walk decides whether genes are matching,
 * disjoint, or excess, and how those counts become the distance used by
 * speciation and diversity summaries.
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   shapes[Minimal genome and connection contracts]:::base --> cache[Generation and per-genome caches]:::base
 *   cache --> lists[Sorted innovation-weight lists]:::accent
 *   lists --> metrics[Matching, disjoint, excess, weight metrics]:::base
 *   metrics --> distance[Final compatibility distance]:::base
 * ```
 *
 * Practical reading order:
 *
 * 1. Start with `GenomeLike` and `NeatLikeForCompat` to see which state the
 *    core layer actually depends on.
 * 2. Read `ComparisonMetrics` as the bridge between raw list comparison and the
 *    final NEAT distance formula.
 * 3. Continue into `compat.core.ts` for cache setup, list comparison, and the
 *    distance fold itself.
 */

// Compatibility core symbol contracts begin below.

/**
 * Compatibility-innovation policy for one genome comparison surface.
 *
 * Native controller genomes should stay on `require-explicit`, which means
 * compatibility reads expect every connection gene to carry a finite
 * innovation number. Legacy, imported, or deliberately partial genomes may opt
 * into `allow-fallback` so comparison can still proceed with endpoint-derived
 * synthetic ids.
 */
export type CompatibilityInnovationMode = 'require-explicit' | 'allow-fallback';

/**
 * Shape of a connection entry used during compatibility checks.
 *
 * The mechanics layer only needs endpoint indices, an optional innovation id,
 * and the connection weight. Keeping this shape small makes the comparison code
 * easier to reuse in tests and controller helpers without importing the full
 * network or genome implementation.
 */
export interface ConnectionLike {
  /** Optional source node descriptor used by fallback innovation resolution. */
  from?: { index?: number };
  /** Optional target node descriptor used by fallback innovation resolution. */
  to?: { index?: number };
  /** Explicit innovation id when available and preferred over synthetic fallback ids. */
  innovation?: number;
  /** Connection weight used when matching genes compare structural alignment and parameter drift together. */
  weight: number;
}

/**
 * Minimal genome shape used for compatibility distance calculations.
 *
 * The `_compatCache` stores sorted `[innovation, weight]` pairs so repeated
 * comparisons within a generation can reuse the same derived list instead of
 * repeatedly normalizing the raw connection array.
 */
export interface GenomeLike {
  /** Optional stable id used to build an order-independent cache key for pairwise distances. */
  _id?: number;
  /** Raw connection list that will be normalized into innovation-weight pairs. */
  connections: ConnectionLike[];
  /**
   * Optional cached sorted innovation list for the current generation's native comparisons.
   *
   * This cache is reserved for explicit-innovation views only. Any connection
   * insertion, removal, innovation rewrite, weight rewrite, or compatibility
   * mode change must invalidate `_compatCache` before the next comparison.
   */
  _compatCache?: Array<[number, number]>;
  /** Optional compatibility mode for legacy/imported/partial genomes. */
  _compatInnovationMode?: CompatibilityInnovationMode;
}

/**
 * Minimal NEAT context required by compatibility helpers.
 *
 * This keeps the boundary tightly focused on generation-scoped caches,
 * compatibility coefficients, and the fallback innovation resolver instead of
 * coupling the mechanics layer to the full `Neat` surface.
 */
export interface NeatLikeForCompat {
  /** Current generation number used to invalidate stale pairwise distance caches. */
  generation: number;
  /** Coefficients used by the final compatibility distance formula. */
  options: {
    /** Excess-gene coefficient controlling how strongly post-alignment tail genes increase distance. */
    excessCoeff?: number;
    /** Disjoint-gene coefficient controlling how strongly mid-list structural mismatches increase distance. */
    disjointCoeff?: number;
    /** Average weight-difference coefficient controlling how much aligned genes can still drift apart numerically. */
    weightDiffCoeff?: number;
  };
  /** Generation id for which the current compatibility caches remain valid. */
  _compatCacheGen?: number;
  /** Pairwise distance cache for the current generation's already-compared genome pairs. */
  _compatDistCache?: Map<string, number>;
  /** Deterministic fallback innovation id generator used only by fallback-allowed genomes. */
  _fallbackInnov: (connection: ConnectionLike) => number;
}

/**
 * Aggregated comparison metrics for compatibility calculations.
 *
 * `compareInnovationLists()` produces this structure so the final distance fold
 * can stay separate from the merge walk that discovered the evidence. That
 * split keeps the algorithm easier to explain: one pass classifies the gene
 * relationship, a later pass applies the NEAT weighting policy.
 */
export type ComparisonMetrics = {
  /** Connection count for the first genome after normalization into sorted innovation pairs. */
  firstGenomeSize: number;
  /** Connection count for the second genome after normalization into sorted innovation pairs. */
  secondGenomeSize: number;
  /** Number of aligned genes with the same innovation id in both genomes. */
  matchingCount: number;
  /** Number of non-matching genes that fall within the shared innovation range. */
  disjointCount: number;
  /** Number of non-matching genes that extend beyond the other genome's highest innovation id. */
  excessCount: number;
  /** Sum of absolute weight differences across only the matching aligned genes. */
  weightDifferenceSum: number;
};
