import type Network from '../../network/network';
import type { MonitoredSmoothingConfig } from '../network.types';

/**
 * Node instance type used by training helpers.
 *
 * This alias keeps helper signatures short while preserving the exact node
 * contract exposed by the owning `Network` instance.
 */
export type NetworkNode = Network['nodes'][number];

/**
 * Regularization payload accepted by `Node.propagate`.
 *
 * Helpers pass this through unchanged so callers can centralize L1/L2
 * configuration at the training entrypoint.
 */
export type RegularizationArgument = Parameters<NetworkNode['propagate']>[3];

/**
 * Derivative callback used by output-node backpropagation.
 *
 * Inputs are `(target, output)` so custom objectives can match the built-in
 * training loop without changing node internals.
 */
export type CostDerivative = (target: number, output: number) => number;

/**
 * Output-node contract for propagation paths that provide a custom derivative.
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
 * Immutable context shared by propagation helpers.
 *
 * Keeping these values in a single object avoids argument drift across helper
 * boundaries and keeps orchestration code declarative.
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
 * Input/output pair consumed by dataset training loops for one supervision
 * step during iterative optimization.
 */
export type TrainingSample = {
  input: number[];
  output: number[];
};

/**
 * Normalized runtime gradient clipping configuration.
 *
 * Optional fields are mode-dependent (`maxNorm` for norm modes, `percentile`
 * for percentile modes).
 */
export type GradientClipRuntimeConfig = {
  mode: 'norm' | 'percentile' | 'layerwiseNorm' | 'layerwisePercentile';
  maxNorm?: number;
  percentile?: number;
};

/**
 * Allow-list of optimizer identifiers accepted by training options before
 * optimizer-specific runtime state is initialized.
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
 * When the caller omits a valid explicit alpha, this helper applies the
 * standard EMA conversion `2 / (window + 1)`.
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
 * This keeps call sites declarative by normalizing all monitored-smoothing
 * fields into one explicit configuration object.
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
