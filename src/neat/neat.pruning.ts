import type { NeatLikeForPruning } from './neat.pruning.utils';
import {
  applyAdaptivePruneLevelToPopulation,
  applyPruningToPopulation,
  computeNextAdaptivePruneLevel,
  computePopulationMetrics,
  computeTargetRemainingMetric,
  computeTargetSparsityNow,
  initializeAdaptivePruningState,
  resolveActiveAdaptivePruningOptions,
  resolveActiveEvolutionPruningOptions,
  resolveAdaptivePruneBaseline,
  resolveObservedMetricValue,
  shouldAdjustAdaptivePruning,
} from './neat.pruning.utils';

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
 * @this NeatLikeForPruning A Neat instance (expects `options`, `generation` and `population`).
 */
export function applyEvolutionPruning(this: NeatLikeForPruning) {
  // Step 1: Resolve the active evolution pruning options for this generation.
  const evolutionPruningOpts = resolveActiveEvolutionPruningOptions(this);

  // Step 2: Exit early when pruning should not run for this generation.
  if (!evolutionPruningOpts) return;

  // Step 3: Calculate the target sparsity for this generation.
  const targetSparsityNow = computeTargetSparsityNow(
    this,
    evolutionPruningOpts,
  );

  // Step 4: Apply pruning to each genome using the selected method.
  applyPruningToPopulation(this, evolutionPruningOpts, targetSparsityNow);
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
 *
 * ```ts
 * // options.adaptivePruning = { enabled: true, metric: 'connections', targetSparsity: 0.6 }
 * neat.applyAdaptivePruning();
 * ```
 *
 * @this NeatLikeForPruning A Neat instance (expects `options` and `population`).
 */
export function applyAdaptivePruning(this: NeatLikeForPruning) {
  // Step 1: Resolve adaptive pruning options when enabled.
  const adaptivePruningOpts = resolveActiveAdaptivePruningOptions(this);

  // Step 2: Exit early when adaptive pruning is disabled.
  if (!adaptivePruningOpts) return;

  // Step 3: Ensure adaptive pruning state is initialized.
  initializeAdaptivePruningState(this);

  // Step 4: Compute population metrics needed for adaptation.
  const populationMetrics = computePopulationMetrics(this);

  // Step 5: Resolve the current observed metric value.
  const currentMetricValue = resolveObservedMetricValue(
    adaptivePruningOpts,
    populationMetrics,
  );

  // Step 6: Resolve and persist the baseline metric.
  const adaptivePruneBaseline = resolveAdaptivePruneBaseline(
    this,
    currentMetricValue,
  );

  // Step 7: Compute the target remaining metric value.
  const targetRemainingMetric = computeTargetRemainingMetric(
    adaptivePruningOpts,
    adaptivePruneBaseline,
  );

  // Step 8: Evaluate whether adjustment is required.
  const shouldAdjust = shouldAdjustAdaptivePruning(
    adaptivePruningOpts,
    currentMetricValue,
    targetRemainingMetric,
    adaptivePruneBaseline,
  );

  // Step 9: Update the prune level and propagate when adjustment is needed.
  if (shouldAdjust) {
    const updatedPruneLevel = computeNextAdaptivePruneLevel(
      adaptivePruningOpts,
      this._adaptivePruneLevel ?? 0,
      currentMetricValue,
      targetRemainingMetric,
    );
    this._adaptivePruneLevel = updatedPruneLevel;
    applyAdaptivePruneLevelToPopulation(this, updatedPruneLevel);
  }
}
