import type Network from '../../network/network';
import type { MonitoredSmoothingConfig } from '../network.types';

/**
 * Local node shape alias used by training utility modules.
 */
export type NetworkNode = Network['nodes'][number];

/**
 * Regularization argument accepted by node-level propagation.
 */
export type RegularizationArgument = Parameters<NetworkNode['propagate']>[3];

/**
 * Cost-derivative callback shape for output-node backpropagation.
 */
export type CostDerivative = (target: number, output: number) => number;

/**
 * Extended output-node contract that supports custom cost derivatives.
 */
export interface OutputNodeWithCostDerivative {
  propagate(
    rate: number,
    momentum: number,
    update: boolean,
    regularization: RegularizationArgument,
    target: number,
    costDerivative: CostDerivative,
  ): void;
}

/**
 * Shared immutable context for network propagation helpers.
 */
export interface PropagationContext {
  network: Network;
  rate: number;
  momentum: number;
  update: boolean;
  regularization: RegularizationArgument;
  costDerivative?: CostDerivative;
}

/**
 * Training sample consumed by training set loops.
 */
export type TrainingSample = {
  input: number[];
  output: number[];
};

/**
 * Runtime gradient clipping configuration normalized from training options.
 */
export type GradientClipRuntimeConfig = {
  mode: 'norm' | 'percentile' | 'layerwiseNorm' | 'layerwisePercentile';
  maxNorm?: number;
  percentile?: number;
};

/**
 * Set of supported optimizer identifiers accepted by training options.
 */
export const ALLOWED_OPTIMIZERS = new Set<string>([
  'sgd',
  'rmsprop',
  'adagrad',
  'adam',
  'adamw',
  'amsgrad',
  'adamax',
  'nadam',
  'radam',
  'lion',
  'adabelief',
  'lookahead',
]);

/**
 * Resolve default EMA alpha using a window length.
 *
 * @param smoothingWindow - Window length for moving average operations.
 * @param explicitAlpha - Optional user-provided alpha override.
 * @returns A valid EMA alpha in the range (0, 1].
 */
export function resolveEmaAlpha(
  smoothingWindow: number,
  explicitAlpha: number | undefined,
): number {
  if (explicitAlpha != null && explicitAlpha > 0 && explicitAlpha <= 1) {
    return explicitAlpha;
  }
  return 2 / (smoothingWindow + 1);
}

/**
 * Build monitored smoothing configuration from options and defaults.
 *
 * @param type - Selected monitored smoothing mode.
 * @param window - Monitored smoothing window length.
 * @param emaAlpha - Optional monitored EMA alpha.
 * @param trimmedRatio - Optional trimmed-mean ratio.
 * @returns Normalized monitored smoothing configuration.
 */
export function buildMonitoredSmoothingConfig(
  type: MonitoredSmoothingConfig['type'],
  window: number,
  emaAlpha: number | undefined,
  trimmedRatio: number | undefined,
): MonitoredSmoothingConfig {
  return {
    type,
    window,
    emaAlpha,
    trimmedRatio,
  };
}
