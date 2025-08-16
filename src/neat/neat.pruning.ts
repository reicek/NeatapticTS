/**
 * Apply evolution-time pruning to the current population.
 *
 * This method is intended to be called from the evolve loop. It reads
 * pruning parameters from `this.options.evolutionPruning` and, when
 * appropriate for the current generation, instructs each genome to
 * prune its connections/nodes to reach a target sparsity.
 *
 * The pruning target can be ramped in over a number of generations so
 * sparsification happens gradually instead of abruptly.
 *
 * Example (in a Neat instance):
 * ```ts
 * // options.evolutionPruning = { startGeneration: 10, targetSparsity: 0.5 }
 * neat.applyEvolutionPruning();
 * ```
 *
 * Notes for docs:
 * - `method` is passed through to each genome's `pruneToSparsity` and
 *   commonly is `'magnitude'` (prune smallest-weight connections first).
 * - This function performs no changes if pruning options are not set or
 *   the generation is before `startGeneration`.
 *
 * @this any A Neat instance (expects `options`, `generation` and `population`).
 */
export function applyEvolutionPruning(this: any) {
  // Read configured evolution pruning options from the Neat instance.
  /** Evolution pruning options configured on the Neat instance. */
  const evolutionPruningOpts = this.options.evolutionPruning;

  // Abort early when pruning is not configured or not yet started.
  if (
    !evolutionPruningOpts ||
    this.generation < (evolutionPruningOpts.startGeneration || 0)
  )
    return;

  /**
   * Interval (in generations) between pruning operations.
   * @default 1
   */
  const interval = evolutionPruningOpts.interval || 1;

  // Only run at configured interval.
  if ((this.generation - evolutionPruningOpts.startGeneration) % interval !== 0)
    return;

  /**
   * How many generations to ramp the pruning in over. If 0, pruning is immediate.
   * @default 0
   */
  const rampGenerations = evolutionPruningOpts.rampGenerations || 0;

  // Fraction in [0,1] indicating how far through the ramp we currently are.
  /** Fraction of ramp completed (0 -> 1). */
  let rampFraction = 1;

  // Compute ramp fraction when ramping is enabled.
  if (rampGenerations > 0) {
    // Step: compute normalized progress through the ramp window.
    const progressThroughRamp = Math.min(
      1,
      Math.max(
        0,
        (this.generation - evolutionPruningOpts.startGeneration) /
          rampGenerations
      )
    );
    rampFraction = progressThroughRamp;
  }

  /**
   * The target sparsity to apply at this generation (0..1), scaled by rampFraction.
   * Example: a configured targetSparsity of 0.5 and rampFraction 0.5 => target 0.25.
   */
  const targetSparsityNow =
    (evolutionPruningOpts.targetSparsity || 0) * rampFraction;

  // Instruct each genome to prune itself to the calculated sparsity.
  for (const genome of this.population) {
    if (genome && typeof genome.pruneToSparsity === 'function') {
      // Step: call the genome's pruning routine. Method defaults to 'magnitude'.
      genome.pruneToSparsity(
        targetSparsityNow,
        evolutionPruningOpts.method || 'magnitude'
      );
    }
  }
}
/**
 * Adaptive pruning controller.
 *
 * This function monitors a population-level metric (average nodes or
 * average connections) and adjusts a global pruning level so the
 * population converges to a target sparsity automatically.
 *
 * It updates `this._adaptivePruneLevel` on the Neat instance and calls
 * each genome's `pruneToSparsity` with the new level when adjustment
 * is required.
 *
 * Example:
 * ```ts
 * // options.adaptivePruning = { enabled: true, metric: 'connections', targetSparsity: 0.6 }
 * neat.applyAdaptivePruning();
 * ```
 *
 * @this any A Neat instance (expects `options` and `population`).
 */
export function applyAdaptivePruning(this: any) {
  // Skip when adaptive pruning is disabled.
  if (!this.options.adaptivePruning?.enabled) return;

  /** Adaptive pruning options from the Neat instance. */
  const adaptivePruningOpts = this.options.adaptivePruning;

  // Initialize the shared prune level if needed.
  if (this._adaptivePruneLevel === undefined) this._adaptivePruneLevel = 0;

  /**
   * Which population-level metric to observe when deciding pruning adjustments.
   * Supported values: 'nodes' | 'connections'
   * @default 'connections'
   */
  const metricName = adaptivePruningOpts.metric || 'connections';

  // Compute average node count across the population.
  /** Average number of nodes per genome in the population (float). */
  const meanNodeCount =
    this.population.reduce((acc: number, g: any) => acc + g.nodes.length, 0) /
    (this.population.length || 1);

  // Compute average connection count across the population.
  /** Average number of connections per genome in the population (float). */
  const meanConnectionCount =
    this.population.reduce(
      (acc: number, g: any) => acc + g.connections.length,
      0
    ) / (this.population.length || 1);

  // Select the current observed metric value.
  /** Current observed metric value used for adaptation. */
  const currentMetricValue =
    metricName === 'nodes' ? meanNodeCount : meanConnectionCount;

  // Initialize baseline if it's the first run.
  if (this._adaptivePruneBaseline === undefined)
    this._adaptivePruneBaseline = currentMetricValue;

  /** Baseline metric value captured when adaptive pruning started. */
  const adaptivePruneBaseline = this._adaptivePruneBaseline;

  /** Target sparsity fraction to aim for (0..1). */
  const desiredSparsity = adaptivePruningOpts.targetSparsity ?? 0.5;

  /**
   * Target remaining metric value (nodes or connections) computed from baseline
   * and desiredSparsity. For example, baseline=100, desiredSparsity=0.5 => targetRemaining=50
   */
  const targetRemainingMetric = adaptivePruneBaseline * (1 - desiredSparsity);

  /** Tolerance to ignore small fluctuations in the observed metric. */
  const tolerance = adaptivePruningOpts.tolerance ?? 0.05;

  /** Rate at which to adjust the global prune level each step (0..1). */
  const adjustRate = adaptivePruningOpts.adjustRate ?? 0.02;

  // Normalized difference between current metric and where we want to be.
  /** Normalized difference: (current - targetRemaining) / baseline. */
  const normalizedDifference =
    (currentMetricValue - targetRemainingMetric) / (adaptivePruneBaseline || 1);

  // Only adjust prune level if deviation exceeds tolerance.
  if (Math.abs(normalizedDifference) > tolerance) {
    // Step: move the prune level up or down by adjustRate in the right direction
    // and clamp it between 0 and desiredSparsity.
    this._adaptivePruneLevel = Math.max(
      0,
      Math.min(
        desiredSparsity,
        this._adaptivePruneLevel +
          adjustRate * (normalizedDifference > 0 ? 1 : -1)
      )
    );

    // Propagate new prune level to each genome using magnitude pruning.
    for (const g of this.population)
      if (typeof g.pruneToSparsity === 'function')
        g.pruneToSparsity(this._adaptivePruneLevel, 'magnitude');
  }
}
