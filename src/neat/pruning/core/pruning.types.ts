/**
 * Minimal NEAT host contract required by pruning helpers.
 *
 * The pruning boundary only needs schedule settings, population structure, and
 * two shared adaptive fields, so this contract stays intentionally narrow.
 *
 * This file is the contract map for the pruning core. It exists so scheduled
 * pruning and adaptive pruning can share one metric-and-policy layer without
 * depending on the full `Neat` controller surface.
 *
 * The contracts divide into three roles:
 *
 * 1. host runtime state: `NeatLikeForPruning`,
 * 2. extracted policy blocks: `EvolutionPruningOptions` and
 *    `AdaptivePruningOptions`,
 * 3. shared measured evidence: `PopulationMetrics`.
 *
 * Read this chapter before `pruning.core.ts` when you need to know which state
 * the pruning helpers are allowed to read, which state they may write back, and
 * which configuration knobs belong to scheduled versus adaptive control.
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

/**
 * Evolution pruning options extracted from the NEAT host.
 *
 * These options drive the calendar-like pruning path: when pruning starts, how
 * often it repeats, how quickly the target sparsity ramps in, and which pruning
 * method is forwarded to compatible genomes.
 */
export type EvolutionPruningOptions = NonNullable<
  NeatLikeForPruning['options']['evolutionPruning']
>;

/**
 * Adaptive pruning options extracted from the NEAT host.
 *
 * These options drive the feedback-controller path: which population metric is
 * observed, what sparsity target should remain, how large a drift is tolerated,
 * and how quickly the shared prune level is allowed to move.
 */
export type AdaptivePruningOptions = NonNullable<
  NeatLikeForPruning['options']['adaptivePruning']
>;

/**
 * Summary of population metrics used by adaptive pruning.
 *
 * Adaptive pruning intentionally works on aggregated evidence rather than on
 * per-genome detail. These means are the small shared measurement surface used
 * to decide whether the population has drifted far enough from the desired
 * complexity level to justify changing the prune level.
 */
export type PopulationMetrics = {
  meanNodeCount: number;
  meanConnectionCount: number;
};
