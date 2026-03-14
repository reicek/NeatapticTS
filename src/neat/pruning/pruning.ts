import type { NeatLikeForPruning } from './core/pruning.types';
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
} from './core/pruning.core';

/**
 * Population-pruning orchestration for the NEAT controller.
 *
 * The root pruning chapter keeps the two controller entrypoints together while
 * pushing the lower-level option resolution and metric math into `core/` and
 * the stable `Neat` wrappers into `facade/`.
 *
 * - `core/` explains pruning options, population metrics, and adjustment math.
 * - `facade/` explains the lazy public `Neat` method wrappers.
 */

/**
 * Apply evolution-time pruning to the current population.
 *
 * This entrypoint is intended for the evolve loop. It reads scheduled pruning
 * options from the controller, computes the current target sparsity, and asks
 * each genome to prune itself using the configured method.
 *
 * @this NeatLikeForPruning NEAT host exposing pruning options, generation state, and population.
 * @returns Nothing. Genomes are pruned in place when the schedule is active.
 */
export function applyEvolutionPruning(this: NeatLikeForPruning): void {
  // Step 1: Resolve the active evolution pruning options for this generation.
  const evolutionPruningOptions = resolveActiveEvolutionPruningOptions(this);

  // Step 2: Exit early when scheduled pruning is inactive.
  if (!evolutionPruningOptions) {
    return;
  }

  // Step 3: Compute the target sparsity for the current generation.
  const targetSparsityNow = computeTargetSparsityNow(
    this,
    evolutionPruningOptions,
  );

  // Step 4: Apply pruning to each genome using the selected method.
  applyPruningToPopulation(this, evolutionPruningOptions, targetSparsityNow);
}

/**
 * Run the adaptive pruning controller.
 *
 * Adaptive pruning monitors a population-level metric and nudges a shared prune
 * level toward a configured sparsity target. This makes pruning responsive to
 * the actual population rather than a fixed generation schedule.
 *
 * @this NeatLikeForPruning NEAT host exposing adaptive pruning options, state, and population.
 * @returns Nothing. The shared prune level and the genomes may be updated in place.
 */
export function applyAdaptivePruning(this: NeatLikeForPruning): void {
  // Step 1: Resolve adaptive pruning options when enabled.
  const adaptivePruningOptions = resolveActiveAdaptivePruningOptions(this);

  // Step 2: Exit early when adaptive pruning is disabled.
  if (!adaptivePruningOptions) {
    return;
  }

  // Step 3: Ensure the shared adaptive pruning state exists.
  initializeAdaptivePruningState(this);

  // Step 4: Compute population metrics and the currently observed metric value.
  const populationMetrics = computePopulationMetrics(this);
  const currentMetricValue = resolveObservedMetricValue(
    adaptivePruningOptions,
    populationMetrics,
  );

  // Step 5: Resolve the baseline and target remaining metric.
  const adaptivePruneBaseline = resolveAdaptivePruneBaseline(
    this,
    currentMetricValue,
  );
  const targetRemainingMetric = computeTargetRemainingMetric(
    adaptivePruningOptions,
    adaptivePruneBaseline,
  );

  // Step 6: Adjust and apply the prune level when the population drifts enough.
  if (
    shouldAdjustAdaptivePruning(
      adaptivePruningOptions,
      currentMetricValue,
      targetRemainingMetric,
      adaptivePruneBaseline,
    )
  ) {
    const updatedPruneLevel = computeNextAdaptivePruneLevel(
      adaptivePruningOptions,
      this._adaptivePruneLevel ?? 0,
      currentMetricValue,
      targetRemainingMetric,
    );
    this._adaptivePruneLevel = updatedPruneLevel;
    applyAdaptivePruneLevelToPopulation(this, updatedPruneLevel);
  }
}