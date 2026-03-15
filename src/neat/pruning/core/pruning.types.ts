/**
 * Minimal NEAT host contract required by pruning helpers.
 *
 * The pruning boundary only needs schedule settings, population structure, and
 * two shared adaptive fields, so this contract stays intentionally narrow.
 */
export interface NeatLikeForPruning {
  /** Pruning-related configuration options. */
  options: {
    /** Configuration for scheduled pruning during evolution. */
    evolutionPruning?: {
      /** Generation index at which pruning begins. */
      startGeneration?: number;
      /** Number of generations between pruning events. */
      interval?: number;
      /** Number of generations used to ramp the target sparsity in. */
      rampGenerations?: number;
      /** Target sparsity fraction in the range `[0, 1]`. */
      targetSparsity?: number;
      /** Pruning method identifier forwarded to `pruneToSparsity()`. */
      method?: string;
    };
    /** Configuration for adaptive pruning driven by population metrics. */
    adaptivePruning?: {
      /** Enables adaptive pruning when true. */
      enabled?: boolean;
      /** Metric name used for adaptation, such as `nodes` or `connections`. */
      metric?: string;
      /** Target sparsity fraction in the range `[0, 1]`. */
      targetSparsity?: number;
      /** Legacy learning rate retained for compatibility. */
      learningRate?: number;
      /** Tolerance band before adjusting pruning. */
      tolerance?: number;
      /** Step size used to change the adaptive prune level. */
      adjustRate?: number;
    };
  };
  /** Current generation counter. */
  generation: number;
  /** Population containing genomes with optional pruning support. */
  population: Array<{
    /** Node list for the genome. */
    nodes: unknown[];
    /** Connection list for the genome. */
    connections: unknown[];
    /** Optional method that prunes the genome to a target sparsity. */
    pruneToSparsity?: (sparsity: number, method?: string) => void;
  }>;
  /** Shared adaptive pruning level across the population. */
  _adaptivePruneLevel?: number;
  /** Baseline metric captured when adaptive pruning begins. */
  _adaptivePruneBaseline?: number;
}

/** Evolution pruning options extracted from the NEAT host. */
export type EvolutionPruningOptions = NonNullable<
  NeatLikeForPruning['options']['evolutionPruning']
>;

/** Adaptive pruning options extracted from the NEAT host. */
export type AdaptivePruningOptions = NonNullable<
  NeatLikeForPruning['options']['adaptivePruning']
>;

/** Summary of population metrics used by adaptive pruning. */
export type PopulationMetrics = {
  meanNodeCount: number;
  meanConnectionCount: number;
};
