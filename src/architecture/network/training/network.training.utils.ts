/**
 * Training pipeline utilities (migrated from legacy architecture/network.train.ts).
 *
 * Provides:
 *  - Gradient clipping (global / layerwise; norm / percentile variants).
 *  - Mini & micro-batch gradient accumulation.
 *  - Optimizer step dispatch (SGD + adaptive optimizers + lookahead wrapper).
 *  - Simple mixed precision dynamic loss scaling (overflow detection heuristic).
 *  - Multiple moving-average smoothing strategies for error monitoring (SMA, EMA, adaptive EMA,
 *    median, gaussian, trimmed mean, WMA) plus separate plateau averaging.
 *  - Early stopping, schedule hooks, pruning hooks, and checkpoint callbacks.
 *
 * Notes:
 *  - This module intentionally keeps imperative style for clarity/perf (avoids heap churn in hot loops).
 *  - Refactor changes here are documentation & naming only; numerical behavior preserved.
 */
import type Network from '../../network';
export { propagate, clearState } from './network.training.backprop.utils';
import type {
  CostFunction,
  CostFunctionOrObject,
  OptimizerConfigBase,
  RegularizationConfig,
  TrainingOptions,
} from '../network.types';
import {
  computeMonitoredError,
  computePlateauMetric,
} from './network.training.smoothing.utils';
import { applyGradientClippingCore } from './network.training.gradient-clip.utils';
import { trainSetCore } from './network.training.loop.utils';
import { trainFinalizeCore } from './network.training.finalize.utils';
import type {
  GradientClipRuntimeConfig,
  TrainingSample,
} from './network.training.utils.types';
export type {
  CheckpointConfig,
  CostFunction,
  GradientClipConfig,
  MetricsHook,
  MixedPrecisionConfig,
  MixedPrecisionDynamicConfig,
  MovingAverageType,
  OptimizerConfigBase,
  ScheduleConfig,
  SerializedNetwork,
  TrainingOptions,
} from '../network.types';

/**
 * Test-only internal helper bundle.
 *
 * This is exported so unit tests can cover edge-cases in the smoothing logic without
 * running full end-to-end training loops.
 *
 * Important: this is **not** considered stable public API. It may change between releases.
 */
export const __trainingInternals = {
  computeMonitoredError,
  computePlateauMetric,
};

/**
 * Apply gradient clipping to a network using a normalized runtime configuration.
 *
 * This is a small wrapper that forwards to the concrete implementation used by training.
 *
 * @param net - Network instance to update.
 * @param cfg - Normalized clipping settings.
 */
export function applyGradientClippingImpl(
  net: Network,
  cfg: GradientClipRuntimeConfig,
): void {
  // Step 1: Delegate clipping work to concrete gradient helper module.
  applyGradientClippingCore(net, cfg);
}

/**
 * Execute one full pass over dataset (epoch) with optional accumulation & adaptive optimizer.
 * Returns mean cost across processed samples.
 *
 * This is the core "one epoch" primitive used by higher-level training orchestration.
 *
 * @param net - Network instance receiving training updates.
 * @param set - Training samples.
 * @param batchSize - Mini-batch size (use 1 for pure SGD).
 * @param accumulationSteps - Micro-batch accumulation steps.
 * @param currentRate - Current learning rate (may be scheduled by caller).
 * @param momentum - Momentum used by some optimizers (when applicable).
 * @param regularization - Regularization configuration passed down to nodes.
 * @param costFunction - Cost function selector (function or compatible object).
 * @param optimizer - Optional optimizer configuration.
 * @returns Mean cost across the processed samples.
 */
export function trainSetImpl(
  net: Network,
  set: TrainingSample[],
  batchSize: number,
  accumulationSteps: number,
  currentRate: number,
  momentum: number,
  regularization: RegularizationConfig,
  costFunction: CostFunction | CostFunctionOrObject,
  optimizer?: OptimizerConfigBase,
): number {
  // Step 1: Delegate dataset training loop to concrete loop helper module.
  return trainSetCore(
    net,
    set,
    batchSize,
    accumulationSteps,
    currentRate,
    momentum,
    regularization,
    costFunction,
    optimizer,
  );
}

/**
 * High-level training orchestration with early stopping, smoothing & callbacks.
 *
 * This is the main entrypoint used by `Network.train(...)`-style APIs.
 *
 * @param net - Network instance to train.
 * @param set - Training dataset.
 * @param options - Training options (stopping conditions, optimizer, hooks, etc.).
 * @returns Summary payload containing final error, iteration count, and elapsed time.
 * @example
 * ```ts
 * const result = net.train(set, { iterations: 500, rate: 0.3 });
 * console.log(result.error);
 * ```
 */
export function trainImpl(
  net: Network,
  set: TrainingSample[],
  options: TrainingOptions,
): { error: number; iterations: number; time: number } {
  // Step 1: Delegate training orchestration to concrete finalize helper module.
  return trainFinalizeCore(net, set, options);
}
