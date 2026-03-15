/**
 * Shape of a connection entry used during compatibility checks.
 *
 * The compatibility helpers only need endpoint indices, an optional
 * innovation number, and the connection weight. Keeping this shape small makes
 * the distance chapter easier to reuse in tests and controller helpers.
 */
export interface ConnectionLike {
  /** Optional source node descriptor. */
  from?: { index?: number };
  /** Optional target node descriptor. */
  to?: { index?: number };
  /** Explicit innovation id when available. */
  innovation?: number;
  /** Connection weight used for matching-gene comparisons. */
  weight: number;
}

/**
 * Minimal genome shape used for compatibility distance calculations.
 *
 * The `_compatCache` stores sorted `[innovation, weight]` pairs so repeated
 * comparisons within a generation can reuse the same derived list.
 */
export interface GenomeLike {
  /** Optional stable id used for pair-cache keys. */
  _id?: number;
  /** Raw connection list for the genome. */
  connections: ConnectionLike[];
  /** Optional cached sorted innovation list. */
  _compatCache?: Array<[number, number]>;
}

/**
 * Minimal NEAT context required by compatibility helpers.
 *
 * This keeps the boundary tightly focused on generation-scoped caches,
 * compatibility coefficients, and the fallback innovation resolver.
 */
export interface NeatLikeForCompat {
  /** Current generation number used to scope caches. */
  generation: number;
  /** Coefficients used by the compatibility distance formula. */
  options: {
    /** Excess-gene coefficient. */
    excessCoeff?: number;
    /** Disjoint-gene coefficient. */
    disjointCoeff?: number;
    /** Average weight-difference coefficient. */
    weightDiffCoeff?: number;
  };
  /** Generation id for which compatibility caches remain valid. */
  _compatCacheGen?: number;
  /** Pairwise distance cache for the current generation. */
  _compatDistCache?: Map<string, number>;
  /** Deterministic fallback innovation id generator. */
  _fallbackInnov: (connection: ConnectionLike) => number;
}

/**
 * Aggregated comparison metrics for compatibility calculations.
 */
export type ComparisonMetrics = {
  /** Connection count for the first genome. */
  firstGenomeSize: number;
  /** Connection count for the second genome. */
  secondGenomeSize: number;
  /** Number of matching innovation ids. */
  matchingCount: number;
  /** Number of disjoint genes. */
  disjointCount: number;
  /** Number of excess genes. */
  excessCount: number;
  /** Sum of absolute weight differences for matching genes. */
  weightDifferenceSum: number;
};
