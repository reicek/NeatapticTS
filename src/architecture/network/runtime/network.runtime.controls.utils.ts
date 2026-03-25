import type Network from '../network';
import { getRegularizationStats } from '../stats/network.stats.utils';
import type {
  NetworkRuntimeControlInternals,
  PruningMethod,
} from '../network.types';

type PruningConfiguration = {
  start: number;
  end: number;
  targetSparsity: number;
  regrowFraction?: number;
  frequency?: number;
  method?: PruningMethod;
};

type WeightNoiseConfiguration = number | { perHiddenLayer: number[] };

type StochasticDepthSchedule = (step: number, current: number[]) => number[];

/**
 * Runtime control helpers for the public `Network` class.
 *
 * This chapter owns the public knobs that change how a network behaves at
 * training or activation time without changing its long-lived topology
 * contract. Keeping these controls here makes the main `network.ts` file read
 * more like orchestration while this file documents the regularization and
 * schedule policies callers can tune directly.
 */

/**
 * Configure scheduled pruning during training.
 *
 * This stores the pruning window and target policy on the network so the
 * training loop can opportunistically apply structured sparsification later.
 *
 * @param this Target network instance.
 * @param configuration Pruning schedule and ranking configuration.
 * @returns Nothing.
 */
export function configurePruning(
  this: Network,
  configuration: PruningConfiguration,
): void {
  const runtimeNetwork = this as unknown as NetworkRuntimeControlInternals;
  const { start, end, targetSparsity } = configuration;

  if (start < 0 || end < start) {
    throw new Error('Invalid pruning schedule window');
  }

  if (targetSparsity <= 0 || targetSparsity >= 1) {
    throw new Error('targetSparsity must be in (0,1)');
  }

  runtimeNetwork._pruningConfig = {
    start,
    end,
    targetSparsity,
    regrowFraction: configuration.regrowFraction ?? 0,
    frequency: configuration.frequency ?? 1,
    method: configuration.method ?? 'magnitude',
    lastPruneIter: undefined,
  };
  runtimeNetwork._initialConnectionCount = runtimeNetwork.connections.length;
}

/**
 * Enable weight noise using either one global standard deviation or per-hidden-layer values.
 *
 * A single global value is useful for quick experiments, while the per-hidden
 * schedule keeps layered models explicit about which hidden stage receives how
 * much perturbation.
 *
 * @param this Target network instance.
 * @param configuration Global standard deviation or one value per hidden layer.
 * @returns Nothing.
 */
export function enableWeightNoise(
  this: Network,
  configuration: WeightNoiseConfiguration,
): void {
  const runtimeNetwork = this as unknown as NetworkRuntimeControlInternals;

  if (typeof configuration === 'number') {
    if (configuration < 0) {
      throw new Error('Weight noise stdDev must be >= 0');
    }

    runtimeNetwork._weightNoiseStd = configuration;
    runtimeNetwork._weightNoisePerHidden = [];
    return;
  }

  if (!configuration || !Array.isArray(configuration.perHiddenLayer)) {
    throw new Error('Invalid weight noise configuration');
  }

  if (!runtimeNetwork.layers || runtimeNetwork.layers.length < 3) {
    throw new Error(
      'Per-hidden-layer weight noise requires a layered network with at least one hidden layer',
    );
  }

  const hiddenLayerCount = runtimeNetwork.layers.length - 2;
  if (configuration.perHiddenLayer.length !== hiddenLayerCount) {
    throw new Error(
      `Expected ${hiddenLayerCount} std dev entries (one per hidden layer), got ${configuration.perHiddenLayer.length}`,
    );
  }

  if (configuration.perHiddenLayer.some((standardDeviation) => standardDeviation < 0)) {
    throw new Error('Weight noise std devs must be >= 0');
  }

  runtimeNetwork._weightNoiseStd = 0;
  runtimeNetwork._weightNoisePerHidden = configuration.perHiddenLayer.slice();
}

/**
 * Disable all configured weight noise.
 *
 * @param this Target network instance.
 * @returns Nothing.
 */
export function disableWeightNoise(this: Network): void {
  const runtimeNetwork = this as unknown as NetworkRuntimeControlInternals;
  runtimeNetwork._weightNoiseStd = 0;
  runtimeNetwork._weightNoisePerHidden = [];
}

/**
 * Set a dynamic scheduler for global weight noise.
 *
 * @param this Target network instance.
 * @param schedule Function mapping the current training step to a standard deviation.
 * @returns Nothing.
 */
export function setWeightNoiseSchedule(
  this: Network,
  schedule: (step: number) => number,
): void {
  const runtimeNetwork = this as unknown as NetworkRuntimeControlInternals;
  runtimeNetwork._weightNoiseSchedule = schedule;
}

/**
 * Clear the dynamic global weight-noise schedule.
 *
 * @param this Target network instance.
 * @returns Nothing.
 */
export function clearWeightNoiseSchedule(this: Network): void {
  const runtimeNetwork = this as unknown as NetworkRuntimeControlInternals;
  runtimeNetwork._weightNoiseSchedule = undefined;
}

/**
 * Replace the network random number generator.
 *
 * This lets advanced callers share one deterministic source across mutation,
 * stochastic depth, DropConnect, and other runtime randomness.
 *
 * @param this Target network instance.
 * @param randomFunction RNG function returning values in $[0,1)$.
 * @returns Nothing.
 */
export function setRandom(this: Network, randomFunction: () => number): void {
  const runtimeNetwork = this as unknown as NetworkRuntimeControlInternals;
  runtimeNetwork._rand = randomFunction;
}

/**
 * Force the next mixed-precision overflow path.
 *
 * This is a test-oriented hook used to exercise loss-scale recovery logic
 * without waiting for a real floating-point overflow.
 *
 * @param this Target network instance.
 * @returns Nothing.
 */
export function testForceOverflow(this: Network): void {
  const runtimeNetwork = this as unknown as NetworkRuntimeControlInternals;
  runtimeNetwork._forceNextOverflow = true;
}

/**
 * Read the current training-step counter.
 *
 * @param this Target network instance.
 * @returns Current training step.
 */
export function getTrainingStep(this: Network): number {
  const runtimeNetwork = this as unknown as NetworkRuntimeControlInternals;
  return runtimeNetwork._trainingStep;
}

/**
 * Read the last hidden-layer indices skipped by stochastic depth.
 *
 * @param this Target network instance.
 * @returns Snapshot of the last skipped hidden-layer indices.
 */
export function getLastSkippedLayers(this: Network): number[] {
  const runtimeNetwork = this as unknown as NetworkRuntimeControlInternals;
  return runtimeNetwork._lastSkippedLayers ?? [];
}

/**
 * Set the stochastic-depth schedule function.
 *
 * @param this Target network instance.
 * @param schedule Function mapping the current step and schedule to a new schedule.
 * @returns Nothing.
 */
export function setStochasticDepthSchedule(
  this: Network,
  schedule: StochasticDepthSchedule,
): void {
  const runtimeNetwork = this as unknown as NetworkRuntimeControlInternals;
  runtimeNetwork._stochasticDepthSchedule = schedule;
}

/**
 * Clear the stochastic-depth schedule function.
 *
 * @param this Target network instance.
 * @returns Nothing.
 */
export function clearStochasticDepthSchedule(this: Network): void {
  const runtimeNetwork = this as unknown as NetworkRuntimeControlInternals;
  runtimeNetwork._stochasticDepthSchedule = undefined;
}

/**
 * Read regularization statistics collected during training.
 *
 * @param this Target network instance.
 * @returns Last regularization stats payload or `null` when none exists yet.
 */
export function getRuntimeRegularizationStats(this: Network) {
  return getRegularizationStats.call(this);
}

/**
 * Configure stochastic depth with one survival probability per hidden layer.
 *
 * Matching survival values to hidden layers keeps the runtime contract explicit
 * and avoids silently applying one layer's policy to another.
 *
 * @param this Target network instance.
 * @param survivalProbabilities Survival probabilities for each hidden layer.
 * @returns Nothing.
 */
export function setStochasticDepth(
  this: Network,
  survivalProbabilities: number[],
): void {
  const runtimeNetwork = this as unknown as NetworkRuntimeControlInternals;

  if (!Array.isArray(survivalProbabilities)) {
    throw new Error('survival must be an array');
  }

  if (survivalProbabilities.some((survivalProbability) => survivalProbability <= 0 || survivalProbability > 1)) {
    throw new Error('Stochastic depth survival probs must be in (0,1]');
  }

  if (!runtimeNetwork.layers || runtimeNetwork.layers.length === 0) {
    throw new Error('Stochastic depth requires layer-based network');
  }

  const hiddenLayerCount = Math.max(0, runtimeNetwork.layers.length - 2);
  if (survivalProbabilities.length !== hiddenLayerCount) {
    throw new Error(
      `Expected ${hiddenLayerCount} survival probabilities for hidden layers, got ${survivalProbabilities.length}`,
    );
  }

  runtimeNetwork._stochasticDepth = survivalProbabilities.slice();
}

/**
 * Disable stochastic depth.
 *
 * @param this Target network instance.
 * @returns Nothing.
 */
export function disableStochasticDepth(this: Network): void {
  const runtimeNetwork = this as unknown as NetworkRuntimeControlInternals;
  runtimeNetwork._stochasticDepth = [];
}
