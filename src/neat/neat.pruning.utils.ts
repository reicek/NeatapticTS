/**
 * Minimal Neat instance contract required by pruning helpers.
 *
 * @example
 * const host: NeatLikeForPruning = {
 *   options: { evolutionPruning: { startGeneration: 5, targetSparsity: 0.4 } },
 *   generation: 10,
 *   population: [],
 * } as NeatLikeForPruning;
 */
export interface NeatLikeForPruning {
  /**
   * Pruning-related configuration options.
   */
  options: {
    /**
     * Configuration for scheduled pruning during evolution.
     */
    evolutionPruning?: {
      /**
       * Generation index at which pruning begins.
       */
      startGeneration?: number;
      /**
       * Number of generations between pruning events.
       */
      interval?: number;
      /**
       * Number of generations to ramp the target sparsity in.
       */
      rampGenerations?: number;
      /**
       * Target sparsity fraction in the range $[0,1]$.
       */
      targetSparsity?: number;
      /**
       * Pruning method identifier passed to `pruneToSparsity()`.
       */
      method?: string;
    };
    /**
     * Configuration for adaptive pruning driven by population metrics.
     */
    adaptivePruning?: {
      /**
       * Enables adaptive pruning when true.
       */
      enabled?: boolean;
      /**
       * Metric name used for adaptation (e.g., `nodes` or `connections`).
       */
      metric?: string;
      /**
       * Target sparsity fraction in the range $[0,1]$.
       */
      targetSparsity?: number;
      /**
       * Legacy learning rate (unused by current implementation).
       */
      learningRate?: number;
      /**
       * Tolerance band for metric deviation before adjusting pruning.
       */
      tolerance?: number;
      /**
       * Step size for adjusting the adaptive prune level.
       */
      adjustRate?: number;
    };
  };
  /**
   * Current generation counter.
   */
  generation: number;
  /**
   * Population containing genomes with node/connection sets and optional pruning method.
   */
  population: Array<{
    /**
     * Node list for the genome.
     */
    nodes: unknown[];
    /**
     * Connection list for the genome.
     */
    connections: unknown[];
    /**
     * Optional method that prunes the genome to a target sparsity.
     */
    pruneToSparsity?: (sparsity: number, method?: string) => void;
  }>;
  /**
   * Shared adaptive pruning level across the population.
   */
  _adaptivePruneLevel?: number;
  /**
   * Baseline metric captured when adaptive pruning begins.
   */
  _adaptivePruneBaseline?: number;
}

/** Evolution pruning options extracted from the Neat instance. */
export type EvolutionPruningOptions = NonNullable<
  NeatLikeForPruning['options']['evolutionPruning']
>;

/** Adaptive pruning options extracted from the Neat instance. */
export type AdaptivePruningOptions = NonNullable<
  NeatLikeForPruning['options']['adaptivePruning']
>;

/** Summary of population metrics used by adaptive pruning. */
export type PopulationMetrics = {
  meanNodeCount: number;
  meanConnectionCount: number;
};

/**
 * @param host - Neat instance with generation state.
 * @returns Evolution pruning options when active, otherwise null.
 */
export function resolveActiveEvolutionPruningOptions(
  host: NeatLikeForPruning,
): EvolutionPruningOptions | null {
  // Step 1: Skip when pruning is not configured.
  const options = host.options.evolutionPruning ?? null;
  if (!options) return null;

  // Step 2: Skip until the configured start generation is reached.
  const startGeneration = options.startGeneration ?? 0;
  if (host.generation < startGeneration) return null;

  // Step 3: Skip when not on the configured interval.
  const interval = options.interval ?? 1;
  if ((host.generation - startGeneration) % interval !== 0) return null;

  return options;
}

/**
 * @param host - Neat instance with generation state.
 * @param options - Evolution pruning options.
 * @returns Target sparsity to apply for this generation.
 */
export function computeTargetSparsityNow(
  host: NeatLikeForPruning,
  options: EvolutionPruningOptions,
): number {
  // Step 1: Resolve ramp progress fraction.
  const rampFraction = computeRampFraction(host, options);

  // Step 2: Scale configured target sparsity by the ramp fraction.
  const configuredTarget = options.targetSparsity ?? 0;
  return configuredTarget * rampFraction;
}

/**
 * @param host - Neat instance with generation state.
 * @param options - Evolution pruning options.
 * @returns Fraction in [0,1] indicating ramp completion.
 */
export function computeRampFraction(
  host: NeatLikeForPruning,
  options: EvolutionPruningOptions,
): number {
  // Step 1: Return full ramp when ramping is disabled.
  const rampGenerations = options.rampGenerations ?? 0;
  if (rampGenerations <= 0) return 1;

  // Step 2: Compute normalized progress through the ramp window.
  const startGeneration = options.startGeneration ?? 0;
  const rawProgress = (host.generation - startGeneration) / rampGenerations;
  return Math.min(1, Math.max(0, rawProgress));
}

/**
 * @param host - Neat instance with population.
 * @param options - Evolution pruning options.
 * @param targetSparsity - Target sparsity to apply.
 */
export function applyPruningToPopulation(
  host: NeatLikeForPruning,
  options: EvolutionPruningOptions,
  targetSparsity: number,
): void {
  // Step 1: Resolve pruning method.
  const method = options.method ?? 'magnitude';

  // Step 2: Apply pruning to each genome that supports it.
  for (const genome of host.population) {
    if (!genome || typeof genome.pruneToSparsity !== 'function') continue;
    genome.pruneToSparsity(targetSparsity, method);
  }
}

/**
 * @param host - Neat instance with adaptive pruning options.
 * @returns Adaptive pruning options when enabled, otherwise null.
 */
export function resolveActiveAdaptivePruningOptions(
  host: NeatLikeForPruning,
): AdaptivePruningOptions | null {
  // Step 1: Exit when not configured or disabled.
  const options = host.options.adaptivePruning ?? null;
  if (!options?.enabled) return null;

  return options;
}

/**
 * @param host - Neat instance with adaptive pruning state.
 */
export function initializeAdaptivePruningState(host: NeatLikeForPruning): void {
  // Step 1: Initialize shared prune level if needed.
  if (host._adaptivePruneLevel === undefined) host._adaptivePruneLevel = 0;
}

/**
 * @param host - Neat instance with population.
 * @returns Population metric summary.
 */
export function computePopulationMetrics(
  host: NeatLikeForPruning,
): PopulationMetrics {
  // Step 1: Compute mean node count across the population.
  const meanNodeCount = computeMeanNodeCount(host);

  // Step 2: Compute mean connection count across the population.
  const meanConnectionCount = computeMeanConnectionCount(host);

  return { meanNodeCount, meanConnectionCount };
}

/**
 * @param host - Neat instance with population.
 * @returns Average number of nodes per genome.
 */
export function computeMeanNodeCount(host: NeatLikeForPruning): number {
  // Step 1: Sum node counts across genomes.
  const totalNodes = host.population.reduce(
    (accumulator: number, genome) => accumulator + genome.nodes.length,
    0,
  );

  // Step 2: Normalize by population size.
  return totalNodes / (host.population.length || 1);
}

/**
 * @param host - Neat instance with population.
 * @returns Average number of connections per genome.
 */
export function computeMeanConnectionCount(host: NeatLikeForPruning): number {
  // Step 1: Sum connection counts across genomes.
  const totalConnections = host.population.reduce(
    (accumulator: number, genome) => accumulator + genome.connections.length,
    0,
  );

  // Step 2: Normalize by population size.
  return totalConnections / (host.population.length || 1);
}

/**
 * @param options - Adaptive pruning options.
 * @param metrics - Population metric summary.
 * @returns Current observed metric value used for adaptation.
 */
export function resolveObservedMetricValue(
  options: AdaptivePruningOptions,
  metrics: PopulationMetrics,
): number {
  // Step 1: Resolve which metric is observed.
  const metricName = options.metric ?? 'connections';
  return metricName === 'nodes'
    ? metrics.meanNodeCount
    : metrics.meanConnectionCount;
}

/**
 * @param host - Neat instance with adaptive baseline state.
 * @param currentMetricValue - Current observed metric value.
 * @returns Baseline metric value used for adaptation.
 */
export function resolveAdaptivePruneBaseline(
  host: NeatLikeForPruning,
  currentMetricValue: number,
): number {
  // Step 1: Initialize baseline on first run.
  if (host._adaptivePruneBaseline === undefined)
    host._adaptivePruneBaseline = currentMetricValue;

  return host._adaptivePruneBaseline;
}

/**
 * @param options - Adaptive pruning options.
 * @param adaptivePruneBaseline - Baseline metric value.
 * @returns Target remaining metric value.
 */
export function computeTargetRemainingMetric(
  options: AdaptivePruningOptions,
  adaptivePruneBaseline: number,
): number {
  // Step 1: Resolve target sparsity.
  const desiredSparsity = options.targetSparsity ?? 0.5;

  // Step 2: Compute remaining metric target.
  return adaptivePruneBaseline * (1 - desiredSparsity);
}

/**
 * @param options - Adaptive pruning options.
 * @param currentMetricValue - Current observed metric value.
 * @param targetRemainingMetric - Target remaining metric value.
 * @param adaptivePruneBaseline - Baseline metric value.
 * @returns True when pruning should be adjusted.
 */
export function shouldAdjustAdaptivePruning(
  options: AdaptivePruningOptions,
  currentMetricValue: number,
  targetRemainingMetric: number,
  adaptivePruneBaseline: number,
): boolean {
  // Step 1: Resolve tolerance.
  const tolerance = options.tolerance ?? 0.05;

  // Step 2: Compute normalized difference from target.
  const normalizedDifference =
    (currentMetricValue - targetRemainingMetric) / (adaptivePruneBaseline || 1);

  return Math.abs(normalizedDifference) > tolerance;
}

/**
 * @param options - Adaptive pruning options.
 * @param currentPruneLevel - Current global prune level.
 * @param currentMetricValue - Current observed metric value.
 * @param targetRemainingMetric - Target remaining metric value.
 * @returns Updated prune level.
 */
export function computeNextAdaptivePruneLevel(
  options: AdaptivePruningOptions,
  currentPruneLevel: number,
  currentMetricValue: number,
  targetRemainingMetric: number,
): number {
  // Step 1: Resolve adjust rate and target sparsity.
  const adjustRate = options.adjustRate ?? 0.02;
  const desiredSparsity = options.targetSparsity ?? 0.5;

  // Step 2: Determine adjustment direction.
  const adjustmentDirection =
    currentMetricValue > targetRemainingMetric ? 1 : -1;

  // Step 3: Apply adjustment and clamp.
  const proposedPruneLevel =
    currentPruneLevel + adjustRate * adjustmentDirection;
  return Math.max(0, Math.min(desiredSparsity, proposedPruneLevel));
}

/**
 * @param host - Neat instance with population.
 * @param pruneLevel - Prune level to apply.
 */
export function applyAdaptivePruneLevelToPopulation(
  host: NeatLikeForPruning,
  pruneLevel: number,
): void {
  // Step 1: Apply pruning to each genome that supports it.
  for (const genome of host.population) {
    if (typeof genome.pruneToSparsity !== 'function') continue;
    genome.pruneToSparsity(pruneLevel, 'magnitude');
  }
}
