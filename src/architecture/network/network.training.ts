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
import * as methods from '../../methods/methods';
import { config } from '../../config';
import type Network from '../network';

/**
 * -----------------------------------------------------------------------------
 * Internal Type Definitions (documentation only; optional for callers)
 * -----------------------------------------------------------------------------
 */
/** Cost function signature used by training. */
export type CostFunction = (target: number[], output: number[]) => number;

/** Gradient clipping configuration accepted by options.gradientClip. */
export interface GradientClipConfig {
  mode?: 'norm' | 'percentile' | 'layerwiseNorm' | 'layerwisePercentile';
  /** Max L2 norm (for *Norm modes). */
  maxNorm?: number;
  /** Percentile threshold (0-100) for *Percentile modes (clamps absolute values). */
  percentile?: number;
  /** Whether to treat bias separately (currently informational flag – behavior parity preserved). */
  separateBias?: boolean;
}

/** Mixed precision configuration. */
export interface MixedPrecisionDynamicConfig {
  /** Minimum loss scale when scaling down after overflows. */
  minScale?: number;
  /** Maximum allowed loss scale for automatic increases. */
  maxScale?: number;
  /** Steps of stable (non-overflow) updates before doubling loss scale. */
  increaseEvery?: number; // alias stableStepsForIncrease
  /** Legacy alias: stable steps threshold for increase. */
  stableStepsForIncrease?: number;
}
export interface MixedPrecisionConfig {
  /** Initial loss scale (larger -> more mantissa preservation but higher overflow risk). */
  lossScale?: number;
  /** Enable dynamic (auto increase/decrease) logic. */
  dynamic?: MixedPrecisionDynamicConfig;
}

/** Optimizer configuration (subset – delegated to node.applyBatchUpdatesWithOptimizer). */
export interface OptimizerConfigBase {
  type: string; // normalized to lowercase
  baseType?: string; // for lookahead
  beta1?: number;
  beta2?: number;
  eps?: number;
  weightDecay?: number;
  momentum?: number;
  la_k?: number; // lookahead sync interval
  la_alpha?: number; // lookahead interpolation factor
}

/** Checkpoint callback spec. */
export interface CheckpointConfig {
  /** Save final state each iteration. */
  last?: boolean;
  /** Save best (lowest error) state. */
  best?: boolean;
  /** Persist function invoked with metadata + serialized network. */
  save: (payload: {
    type: 'last' | 'best';
    iteration: number;
    error: number;
    network: any;
  }) => void;
}

/** Schedule hook executed every N iterations. */
export interface ScheduleConfig {
  iterations: number; // frequency
  function: (info: { error: number; iteration: number }) => void;
}

/** Metrics hook signature. */
export type MetricsHook = (m: {
  iteration: number;
  error: number;
  plateauError?: number;
  gradNorm: number;
}) => void;

/** Moving average strategy identifiers. */
export type MovingAverageType =
  | 'sma'
  | 'ema'
  | 'adaptive-ema'
  | 'median'
  | 'gaussian'
  | 'trimmed'
  | 'wma';

/** Primary training options object (public shape). */
export interface TrainingOptions {
  iterations?: number; // stopping condition: max passes
  error?: number; // stopping condition: target monitored (smoothed) error
  rate?: number; // base learning rate
  momentum?: number; // momentum for SGD / sometimes consumed by wrappers
  optimizer?: string | OptimizerConfigBase; // adaptive optimizer choice
  dropout?: number; // dropout probability applied per forward (mutable net.dropout)
  batchSize?: number; // mini-batch size; if > dataset length => error
  accumulationSteps?: number; // gradient accumulation factor (micro-batches per optimizer step)
  accumulationReduction?: 'average' | 'sum'; // scaling mode for accumulated gradients
  gradientClip?: GradientClipConfig; // gradient clipping configuration
  mixedPrecision?: boolean | MixedPrecisionConfig; // enable FP16-like scaling logic
  cost?: CostFunction | { fn?: CostFunction; calculate?: CostFunction }; // cost interface variants
  movingAverageWindow?: number; // smoothing window size
  movingAverageType?: MovingAverageType; // smoothing algorithm
  emaAlpha?: number; // override alpha for EMA
  adaptiveEmaBaseAlpha?: number; // (not currently used – placeholder)
  trimmedRatio?: number; // fraction dropped from each tail for trimmed mean (0..0.49)
  plateauMovingAverageWindow?: number; // independent plateau window
  plateauMovingAverageType?: MovingAverageType; // independent plateau strategy
  plateauEmaAlpha?: number; // plateau EMA alpha override
  earlyStopPatience?: number; // iterations with no improvement before stop
  earlyStopMinDelta?: number; // required improvement beyond previous best
  checkpoint?: CheckpointConfig; // persistence callbacks
  schedule?: ScheduleConfig; // periodic hook
  metricsHook?: MetricsHook; // telemetry per iteration
}

/** ---------------------------------------------------------------------------
 * Internal Helper Utilities (non-exported)
 * ---------------------------------------------------------------------------
 * These functions encapsulate cohesive sub-steps of the training pipeline so the
 * main exported functions remain readable while preserving original behavior.
 * Each helper is intentionally pure where reasonable or documents its side-effects.
 */

/** State container for EMA / Adaptive EMA smoothing values. */
interface PrimarySmoothingState {
  /** Classic EMA value (when movingAverageType === 'ema'). */
  emaValue?: number;
  /** Baseline EMA part of adaptive EMA (slower). */
  adaptiveBaseEmaValue?: number;
  /** Fast adaptive EMA (higher alpha under variance). */
  adaptiveEmaValue?: number;
}

/** State container for plateau EMA smoothing. */
interface PlateauSmoothingState {
  plateauEmaValue?: number;
}

/** Configuration passed to monitored (primary) smoothing computation. */
interface MonitoredSmoothingConfig {
  type: MovingAverageType;
  window: number;
  emaAlpha?: number; // optional override (only for EMA types)
  trimmedRatio?: number; // for trimmed mean strategy
}

/** Configuration for plateau smoothing computation. */
interface PlateauSmoothingConfig {
  type: MovingAverageType;
  window: number;
  emaAlpha?: number;
}

/**
 * Compute the monitored (primary) smoothed error given recent raw errors.
 *
 * Behavior:
 *  - For SMA-like strategies uses the supplied window slice directly.
 *  - For EMA it mutates state.emaValue.
 *  - For adaptive-ema maintains dual EMA tracks inside state and returns the min for stability.
 *  - For median / gaussian / trimmed / wma applies algorithmic weighting as documented inline.
 *
 * Inputs:
 *  - trainError: Current raw mean error for this iteration.
 *  - recentErrors: Chronological array (oldest->newest) of last N raw errors.
 *  - cfg: Algorithm selection + parameters.
 *  - state: Mutable smoothing state (ema / adaptive fields updated in-place).
 *
 * Returns: Smoothed/monitored error metric (may equal trainError if no smoothing active).
 */
function computeMonitoredError(
  trainError: number,
  recentErrors: number[],
  cfg: MonitoredSmoothingConfig,
  state: PrimarySmoothingState
): number {
  // Fast path: no smoothing window / algorithm requiring history.
  if (cfg.window <= 1 && cfg.type !== 'ema' && cfg.type !== 'adaptive-ema') {
    return trainError;
  }
  const type = cfg.type;
  if (type === 'median') {
    const sorted = [...recentErrors].sort((a, b) => a - b);
    const midIndex = Math.floor(sorted.length / 2);
    return sorted.length % 2
      ? sorted[midIndex]
      : (sorted[midIndex - 1] + sorted[midIndex]) / 2;
  }
  if (type === 'ema') {
    // Standard exponential moving average.
    if (state.emaValue == null) state.emaValue = trainError;
    else
      state.emaValue =
        state.emaValue + cfg.emaAlpha! * (trainError - state.emaValue);
    return state.emaValue;
  }
  if (type === 'adaptive-ema') {
    // Adaptive EMA: baseline alpha + volatility-inflated alpha, final metric is more conservative (min).
    const mean = recentErrors.reduce((a, b) => a + b, 0) / recentErrors.length;
    const variance =
      recentErrors.reduce((a, b) => a + (b - mean) * (b - mean), 0) /
      recentErrors.length;
    const baseAlpha = cfg.emaAlpha || 2 / (cfg.window + 1);
    const varianceScaled = variance / Math.max(mean * mean, 1e-8);
    const adaptiveAlpha = Math.min(
      0.95,
      Math.max(baseAlpha, baseAlpha * (1 + 2 * varianceScaled))
    );
    if (state.adaptiveBaseEmaValue == null) {
      state.adaptiveBaseEmaValue = trainError;
      state.adaptiveEmaValue = trainError;
    } else {
      state.adaptiveBaseEmaValue =
        state.adaptiveBaseEmaValue +
        baseAlpha * (trainError - state.adaptiveBaseEmaValue);
      state.adaptiveEmaValue =
        state.adaptiveEmaValue! +
        adaptiveAlpha * (trainError - state.adaptiveEmaValue!);
    }
    return Math.min(state.adaptiveEmaValue!, state.adaptiveBaseEmaValue!);
  }
  if (type === 'gaussian') {
    // Gaussian kernel weights centered at newest element (index length-1).
    const sigma = cfg.window / 3 || 1; // heuristic: cover window ~3 sigma
    let weightSum = 0;
    let weightedAccumulator = 0;
    const length = recentErrors.length;
    for (let i = 0; i < length; i++) {
      const weight = Math.exp(-0.5 * Math.pow((i - (length - 1)) / sigma, 2));
      weightSum += weight;
      weightedAccumulator += weight * recentErrors[i];
    }
    return weightedAccumulator / (weightSum || 1);
  }
  if (type === 'trimmed') {
    // Trim symmetric tails before averaging to reduce outlier influence.
    const ratio = Math.min(0.49, Math.max(0, cfg.trimmedRatio || 0.1));
    const sorted = [...recentErrors].sort((a, b) => a - b);
    const drop = Math.floor(sorted.length * ratio);
    const trimmed = sorted.slice(drop, sorted.length - drop);
    return trimmed.reduce((a, b) => a + b, 0) / (trimmed.length || 1);
  }
  if (type === 'wma') {
    // Linear weighting (oldest weight=1 ... newest weight=n).
    let weightSum = 0;
    let weightedAccumulator = 0;
    for (let i = 0; i < recentErrors.length; i++) {
      const weight = i + 1;
      weightSum += weight;
      weightedAccumulator += weight * recentErrors[i];
    }
    return weightedAccumulator / (weightSum || 1);
  }
  // Default: arithmetic mean (SMA).
  return recentErrors.reduce((a, b) => a + b, 0) / recentErrors.length;
}

/**
 * Compute plateau metric (may differ in strategy from primary monitored error).
 * Only algorithms actually supported for plateau in current pipeline are SMA, median and EMA.
 * Provided flexibility keeps room for extension; unsupported types silently fallback to mean.
 */
function computePlateauMetric(
  trainError: number,
  plateauErrors: number[],
  cfg: PlateauSmoothingConfig,
  state: PlateauSmoothingState
): number {
  if (cfg.window <= 1 && cfg.type !== 'ema') return trainError;
  if (cfg.type === 'median') {
    const sorted = [...plateauErrors].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2
      ? sorted[mid]
      : (sorted[mid - 1] + sorted[mid]) / 2;
  }
  if (cfg.type === 'ema') {
    if (state.plateauEmaValue == null) state.plateauEmaValue = trainError;
    else
      state.plateauEmaValue =
        state.plateauEmaValue +
        cfg.emaAlpha! * (trainError - state.plateauEmaValue);
    return state.plateauEmaValue;
  }
  // Fallback default mean.
  return plateauErrors.reduce((a, b) => a + b, 0) / plateauErrors.length;
}

// Internal export bundle (test-only usage) to enable direct branch coverage of smoothing helpers.
// Marked with double underscore to discourage production use.
export const __trainingInternals = {
  computeMonitoredError,
  computePlateauMetric,
};

/**
 * Detect mixed precision overflow (NaN / Inf) in bias values if mixed precision enabled.
 * Side-effect: may clear internal trigger _forceNextOverflow.
 */
function detectMixedPrecisionOverflow(net: Network, internalNet: any): boolean {
  if (!internalNet._mixedPrecision.enabled) return false;
  if (internalNet._forceNextOverflow) {
    internalNet._forceNextOverflow = false;
    return true;
  }
  let overflow = false;
  net.nodes.forEach((node) => {
    if ((node as any)._fp32Bias !== undefined) {
      if (!Number.isFinite((node as any).bias)) overflow = true;
    }
  });
  return overflow;
}

/** Zero-out accumulated gradient buffers after an overflow to discard invalid updates. */
function zeroAccumulatedGradients(net: Network) {
  net.nodes.forEach((node) => {
    (node as any).connections.in.forEach((c: any) => {
      c.totalDeltaWeight = 0;
    });
    (node as any).connections.self.forEach((c: any) => {
      c.totalDeltaWeight = 0;
    });
    if (typeof (node as any).totalDeltaBias === 'number')
      (node as any).totalDeltaBias = 0;
    (node as any).previousDeltaBias = 0;
  });
}

/** Divide accumulated gradients by accumulationSteps (average reduction mode). */
function averageAccumulatedGradients(net: Network, accumulationSteps: number) {
  if (accumulationSteps <= 1) return;
  net.nodes.forEach((node) => {
    (node as any).connections.in.forEach((c: any) => {
      if (typeof c.totalDeltaWeight === 'number')
        c.totalDeltaWeight /= accumulationSteps;
    });
    (node as any).connections.self.forEach((c: any) => {
      if (typeof c.totalDeltaWeight === 'number')
        c.totalDeltaWeight /= accumulationSteps;
    });
    if (typeof (node as any).totalDeltaBias === 'number')
      (node as any).totalDeltaBias /= accumulationSteps;
  });
}

/** Apply optimizer update step across all nodes; returns gradient L2 norm (approx). */
function applyOptimizerStep(
  net: Network,
  optimizer: any,
  currentRate: number,
  momentum: number,
  internalNet: any
): number {
  let sumSq = 0;
  net.nodes.forEach((node) => {
    if (node.type === 'input') return;
    (node as any).applyBatchUpdatesWithOptimizer({
      type: optimizer.type,
      baseType: optimizer.baseType,
      beta1: optimizer.beta1,
      beta2: optimizer.beta2,
      eps: optimizer.eps,
      weightDecay: optimizer.weightDecay,
      momentum: optimizer.momentum ?? momentum,
      lrScale: currentRate,
      t: internalNet._optimizerStep,
      la_k: optimizer.la_k,
      la_alpha: optimizer.la_alpha,
    });
    (node as any).connections.in.forEach((c: any) => {
      if (typeof c.previousDeltaWeight === 'number')
        sumSq += c.previousDeltaWeight * c.previousDeltaWeight;
    });
    (node as any).connections.self.forEach((c: any) => {
      if (typeof c.previousDeltaWeight === 'number')
        sumSq += c.previousDeltaWeight * c.previousDeltaWeight;
    });
  });
  return Math.sqrt(sumSq);
}

/** Update dynamic loss scaling after a successful (non-overflow) optimizer step. */
function maybeIncreaseLossScale(internalNet: any) {
  internalNet._mixedPrecisionState.goodSteps++;
  const incEvery = internalNet._mpIncreaseEvery || 200;
  if (
    internalNet._mixedPrecisionState.goodSteps >= incEvery &&
    internalNet._mixedPrecision.lossScale <
      internalNet._mixedPrecisionState.maxLossScale
  ) {
    internalNet._mixedPrecision.lossScale *= 2;
    internalNet._mixedPrecisionState.goodSteps = 0;
    internalNet._mixedPrecisionState.scaleUpEvents =
      (internalNet._mixedPrecisionState.scaleUpEvents || 0) + 1;
  }
}

/** Respond to a mixed precision overflow by shrinking loss scale & bookkeeping. */
function handleOverflow(internalNet: any) {
  internalNet._mixedPrecisionState.badSteps++;
  internalNet._mixedPrecisionState.goodSteps = 0;
  internalNet._mixedPrecision.lossScale = Math.max(
    internalNet._mixedPrecisionState.minLossScale,
    Math.floor(internalNet._mixedPrecision.lossScale / 2) || 1
  );
  internalNet._mixedPrecisionState.overflowCount =
    (internalNet._mixedPrecisionState.overflowCount || 0) + 1;
  internalNet._mixedPrecisionState.scaleDownEvents =
    (internalNet._mixedPrecisionState.scaleDownEvents || 0) + 1;
  internalNet._lastOverflowStep = internalNet._optimizerStep;
}

/**
 * Apply gradient clipping to accumulated connection deltas / bias deltas.
 *
 * Modes:
 *  - norm / layerwiseNorm: L2 norm scaling (global vs per group).
 *  - percentile / layerwisePercentile: element-wise clamp at absolute percentile threshold.
 *
 * Grouping:
 *  - If layerwise* and net.layers exists -> each defined layer is a group.
 *  - Else if layerwise* -> each non-input node becomes its own group.
 *  - Otherwise a single global group containing all learnable params.
 */
export function applyGradientClippingImpl(
  net: Network,
  cfg: {
    mode: 'norm' | 'percentile' | 'layerwiseNorm' | 'layerwisePercentile';
    maxNorm?: number;
    percentile?: number;
  }
) {
  const internalNet = net as any;
  /**
   * Build arrays of gradient values grouped according to chosen clipping mode.
   * Each group is later processed independently (layerwise modes) or as a single global set.
   */
  const collectGroups = () => {
    const collected: number[][] = [];
    if (cfg.mode.startsWith('layerwise')) {
      if ((net as any).layers && (net as any).layers.length > 0) {
        for (let li = 0; li < (net as any).layers.length; li++) {
          const layer = (net as any).layers[li];
          if (!layer || !layer.nodes) continue;
          const groupVals: number[] = [];
          layer.nodes.forEach((node: any) => {
            if (!node || node.type === 'input') return;
            node.connections.in.forEach((c: any) => {
              if (typeof c.totalDeltaWeight === 'number')
                groupVals.push(c.totalDeltaWeight);
            });
            node.connections.self.forEach((c: any) => {
              if (typeof c.totalDeltaWeight === 'number')
                groupVals.push(c.totalDeltaWeight);
            });
            if (typeof node.totalDeltaBias === 'number')
              groupVals.push(node.totalDeltaBias);
          });
          if (groupVals.length) collected.push(groupVals);
        }
      } else {
        net.nodes.forEach((node) => {
          if (node.type === 'input') return;
          const groupVals: number[] = [];
          (node as any).connections.in.forEach((c: any) => {
            if (typeof c.totalDeltaWeight === 'number')
              groupVals.push(c.totalDeltaWeight);
          });
          (node as any).connections.self.forEach((c: any) => {
            if (typeof c.totalDeltaWeight === 'number')
              groupVals.push(c.totalDeltaWeight);
          });
          if (typeof (node as any).totalDeltaBias === 'number')
            groupVals.push((node as any).totalDeltaBias);
          if (groupVals.length) collected.push(groupVals);
        });
      }
    } else {
      const globalVals: number[] = [];
      net.nodes.forEach((node) => {
        (node as any).connections.in.forEach((c: any) => {
          if (typeof c.totalDeltaWeight === 'number')
            globalVals.push(c.totalDeltaWeight);
        });
        (node as any).connections.self.forEach((c: any) => {
          if (typeof c.totalDeltaWeight === 'number')
            globalVals.push(c.totalDeltaWeight);
        });
        if (typeof (node as any).totalDeltaBias === 'number')
          globalVals.push((node as any).totalDeltaBias);
      });
      if (globalVals.length) collected.push(globalVals);
    }
    return collected;
  };
  /**
   * Gradient groups discovered for clipping (size: 1 for global modes).
   * Each entry is an array of parameter delta values belonging to a logical group (layer or node level).
   */
  const groups = collectGroups();
  /** Tracking for diagnostics / potential external tooling. */
  internalNet._lastGradClipGroupCount = groups.length;
  /**
   * Compute absolute percentile threshold (e.g. percentile=99 => value whose |value| is at the 99th percentile).
   * Sorting by absolute value guarantees consistent clipping for symmetric distributions.
   */
  const computeAbsolutePercentileThreshold = (
    values: number[],
    percentile: number
  ) => {
    if (!values.length) return 0;
    const sortedByAbs = [...values].sort((a, b) => Math.abs(a) - Math.abs(b));
    const rank = Math.min(
      sortedByAbs.length - 1,
      Math.max(0, Math.floor((percentile / 100) * sortedByAbs.length - 1))
    );
    return Math.abs(sortedByAbs[rank]);
  };
  /**
   * Iterate all learnable parameters applying a transform function.
   * The transform receives the current value and the owning group so it can selectively scale only
   * the active group (when computing per-group scaling factor yet iterating entire model).
   */
  const applyScale = (
    scaleFn: (currentValue: number, owningGroup: number[]) => number
  ) => {
    let groupIndex = 0; // advances only for layerwise modes
    net.nodes.forEach((node) => {
      if (cfg.mode.startsWith('layerwise') && node.type === 'input') return; // skip input nodes in layerwise grouping
      const activeGroup = cfg.mode.startsWith('layerwise')
        ? groups[groupIndex++]
        : groups[0];
      (node as any).connections.in.forEach((c: any) => {
        if (typeof c.totalDeltaWeight === 'number')
          c.totalDeltaWeight = scaleFn(c.totalDeltaWeight, activeGroup);
      });
      (node as any).connections.self.forEach((c: any) => {
        if (typeof c.totalDeltaWeight === 'number')
          c.totalDeltaWeight = scaleFn(c.totalDeltaWeight, activeGroup);
      });
      if (typeof (node as any).totalDeltaBias === 'number')
        (node as any).totalDeltaBias = scaleFn(
          (node as any).totalDeltaBias,
          activeGroup
        );
    });
  };
  if (cfg.mode === 'norm' || cfg.mode === 'layerwiseNorm') {
    /** Maximum allowed L2 norm per group (or global). */
    const maxAllowedNorm = cfg.maxNorm || 1;
    groups.forEach((groupValues) => {
      /** Current group L2 norm. */
      const groupL2Norm = Math.sqrt(
        groupValues.reduce((sum, v) => sum + v * v, 0)
      );
      if (groupL2Norm > maxAllowedNorm && groupL2Norm > 0) {
        /** Scaling factor applied uniformly to bring norm to boundary. */
        const normScaleFactor = maxAllowedNorm / groupL2Norm;
        applyScale((currentValue, owningGroup) =>
          owningGroup === groupValues
            ? currentValue * normScaleFactor
            : currentValue
        );
      }
    });
  } else if (cfg.mode === 'percentile' || cfg.mode === 'layerwisePercentile') {
    /** Percentile specifying absolute magnitude cutoff (values above are clamped). */
    const percentileSetting = cfg.percentile || 99;
    groups.forEach((groupValues) => {
      const percentileThreshold = computeAbsolutePercentileThreshold(
        groupValues,
        percentileSetting
      );
      if (percentileThreshold <= 0) return;
      applyScale((currentValue, owningGroup) =>
        owningGroup === groupValues &&
        Math.abs(currentValue) > percentileThreshold
          ? percentileThreshold * Math.sign(currentValue)
          : currentValue
      );
    });
  }
}

/**
 * Execute one full pass over dataset (epoch) with optional accumulation & adaptive optimizer.
 * Returns mean cost across processed samples.
 */
export function trainSetImpl(
  net: Network,
  set: { input: number[]; output: number[] }[],
  batchSize: number,
  accumulationSteps: number,
  currentRate: number,
  momentum: number,
  regularization: any,
  costFunction: (target: number[], output: number[]) => number,
  optimizer?: any
): number {
  const internalNet = net as any;
  /** Sum of raw (unsmoothed) cost values across valid samples. */
  let cumulativeError = 0;
  /** Number of samples processed in current mini-batch (resets after potential optimizer step). */
  let batchSampleCount = 0;
  /** Counter of micro-batches contributing to current accumulated gradient set. */
  internalNet._gradAccumMicroBatches = 0;
  /** Total number of dataset samples actually processed (dimension-valid). */
  let totalProcessedSamples = 0;
  /** Cached list of output layer nodes (backprop order requires targets). */
  const outputNodes = net.nodes.filter((n) => n.type === 'output');
  /** Unified cost evaluation function resolved from provided cost variant. */
  let computeError: (t: number[], o: number[]) => number;
  if (typeof costFunction === 'function') computeError = costFunction as any;
  else if (
    (costFunction as any) &&
    typeof (costFunction as any).fn === 'function'
  )
    computeError = (costFunction as any).fn;
  else if (
    (costFunction as any) &&
    typeof (costFunction as any).calculate === 'function'
  )
    computeError = (costFunction as any).calculate;
  else computeError = () => 0;

  for (let sampleIndex = 0; sampleIndex < set.length; sampleIndex++) {
    /** Current training sample record (input + target). */
    const dataPoint = set[sampleIndex];
    /** Input feature vector (validated for dimension). */
    const input = dataPoint.input;
    /** Target output vector (validated for dimension). */
    const target = dataPoint.output;
    if (input.length !== net.input || target.length !== net.output) {
      if (config.warnings)
        console.warn(
          `Data point ${sampleIndex} has incorrect dimensions (input: ${input.length}/${net.input}, output: ${target.length}/${net.output}), skipping.`
        );
      continue;
    }
    try {
      // Forward pass with training flag (enables dropout / any stochastic layers).
      const output = (net as any).activate(input, true);
      if (optimizer && optimizer.type && optimizer.type !== 'sgd') {
        // Accumulate gradients for adaptive optimizers (no immediate weight update inside propagate).
        for (let outIndex = 0; outIndex < outputNodes.length; outIndex++)
          (outputNodes[outIndex] as any).propagate(
            currentRate,
            momentum,
            false,
            regularization,
            target[outIndex]
          );
        for (
          let reverseIndex = net.nodes.length - 1;
          reverseIndex >= 0;
          reverseIndex--
        ) {
          const node = net.nodes[reverseIndex];
          if (node.type === 'output' || node.type === 'input') continue;
          (node as any).propagate(currentRate, momentum, false, regularization);
        }
      } else {
        // SGD mode: propagate performs immediate parameter updates using deltas.
        for (let outIndex = 0; outIndex < outputNodes.length; outIndex++)
          (outputNodes[outIndex] as any).propagate(
            currentRate,
            momentum,
            true,
            regularization,
            target[outIndex]
          );
        for (
          let reverseIndex = net.nodes.length - 1;
          reverseIndex >= 0;
          reverseIndex--
        ) {
          const node = net.nodes[reverseIndex];
          if (node.type === 'output' || node.type === 'input') continue;
          (node as any).propagate(currentRate, momentum, true, regularization);
        }
      }
      cumulativeError += computeError(target, output);
      batchSampleCount++;
      totalProcessedSamples++;
    } catch (e: any) {
      if (config.warnings)
        console.warn(
          `Error processing data point ${sampleIndex} (input: ${JSON.stringify(
            input
          )}): ${e.message}. Skipping.`
        );
    }
    // Mini-batch / end-of-dataset flush condition.
    if (
      batchSampleCount > 0 &&
      ((sampleIndex + 1) % batchSize === 0 || sampleIndex === set.length - 1)
    ) {
      if (optimizer && optimizer.type && optimizer.type !== 'sgd') {
        // Only adaptive optimizers delay the step; vanilla SGD already updated weights per sample.
        internalNet._gradAccumMicroBatches++;
        /** True when we have accumulated sufficient micro-batches or reached dataset end. */
        const readyForStep =
          internalNet._gradAccumMicroBatches % accumulationSteps === 0 ||
          sampleIndex === set.length - 1;
        if (readyForStep) {
          /** 1-based optimizer step counter (used for bias-correction terms by adaptive methods). */
          internalNet._optimizerStep = (internalNet._optimizerStep || 0) + 1;
          /** Detect overflow under mixed precision (NaN/Inf). */
          const overflowDetected = detectMixedPrecisionOverflow(
            net,
            internalNet
          );
          if (overflowDetected) {
            // Discard invalid gradients & shrink loss scale.
            zeroAccumulatedGradients(net);
            if (internalNet._mixedPrecision.enabled)
              handleOverflow(internalNet);
            internalNet._lastGradNorm = 0;
          } else {
            // Optional gradient clipping before optimizer math.
            if (internalNet._currentGradClip)
              applyGradientClippingImpl(net, internalNet._currentGradClip);
            // Average accumulated micro-batch gradients if configured.
            if (
              accumulationSteps > 1 &&
              internalNet._accumulationReduction === 'average'
            ) {
              averageAccumulatedGradients(net, accumulationSteps);
            }
            // Apply optimizer updates and compute gradient norm.
            internalNet._lastGradNorm = applyOptimizerStep(
              net,
              optimizer,
              currentRate,
              momentum,
              internalNet
            );
            // Dynamic loss scaling increase if conditions satisfied.
            if (internalNet._mixedPrecision.enabled)
              maybeIncreaseLossScale(internalNet);
          }
        }
        batchSampleCount = 0; // reset mini-batch sample counter
      }
    }
  }
  if (internalNet._lastGradNorm == null) internalNet._lastGradNorm = 0;
  return totalProcessedSamples > 0
    ? cumulativeError / totalProcessedSamples
    : 0;
}

/**
 * High-level training orchestration with early stopping, smoothing & callbacks.
 */
export function trainImpl(
  net: Network,
  set: { input: number[]; output: number[] }[],
  options: TrainingOptions
): { error: number; iterations: number; time: number } {
  const internalNet = net as any;
  if (
    !set ||
    set.length === 0 ||
    set[0].input.length !== net.input ||
    set[0].output.length !== net.output
  ) {
    throw new Error(
      'Dataset is invalid or dimensions do not match network input/output size!'
    );
  }
  options = options || {};
  if (
    typeof options.iterations === 'undefined' &&
    typeof options.error === 'undefined'
  ) {
    if (config.warnings)
      console.warn('Missing `iterations` or `error` option.');
    throw new Error(
      'Missing `iterations` or `error` option. Training requires a stopping condition.'
    );
  }
  if (config.warnings) {
    if (typeof options.rate === 'undefined') {
      console.warn('Missing `rate` option');
      console.warn('Missing `rate` option, using default learning rate 0.3.');
    }
    if (typeof options.iterations === 'undefined')
      console.warn(
        'Missing `iterations` option. Training will run potentially indefinitely until `error` threshold is met.'
      );
  }
  /** Target monitored (smoothed) error threshold for early termination. */
  let targetError = options.error ?? -Infinity;
  /** Cost function (defaults to MSE) resolved from provided variant. */
  const cost = options.cost || methods.Cost.mse;
  if (
    typeof cost !== 'function' &&
    !(
      typeof cost === 'object' &&
      (typeof (cost as any).fn === 'function' ||
        typeof (cost as any).calculate === 'function')
    )
  ) {
    throw new Error('Invalid cost function provided to Network.train.');
  }
  /** Base learning rate used as scaling factor for optimizer weight updates. */
  const baseRate = options.rate ?? 0.3;
  /** Dropout probability applied each forward pass (0 disables). */
  const dropout = options.dropout || 0;
  if (dropout < 0 || dropout >= 1) throw new Error('dropout must be in [0,1)');
  /** Momentum factor for SGD or reused by optimizers expecting momentum param. */
  const momentum = options.momentum || 0;
  /** Mini-batch size (#samples per gradient accumulation flush). */
  const batchSize = options.batchSize || 1;
  if (batchSize > set.length)
    throw new Error('Batch size cannot be larger than the dataset length.');
  /** Gradient accumulation factor (micro-batches per optimizer step). */
  const accumulationSteps = options.accumulationSteps || 1;
  internalNet._accumulationReduction =
    options.accumulationReduction === 'sum' ? 'sum' : 'average';
  if (accumulationSteps < 1 || !Number.isFinite(accumulationSteps))
    throw new Error('accumulationSteps must be >=1');
  if (options.gradientClip) {
    const gc = options.gradientClip;
    if (gc.mode)
      internalNet._currentGradClip = {
        mode: gc.mode,
        maxNorm: gc.maxNorm,
        percentile: gc.percentile,
      } as any;
    else if (typeof gc.maxNorm === 'number')
      internalNet._currentGradClip = { mode: 'norm', maxNorm: gc.maxNorm };
    else if (typeof gc.percentile === 'number')
      internalNet._currentGradClip = {
        mode: 'percentile',
        percentile: gc.percentile,
      } as any;
    internalNet._gradClipSeparateBias = !!gc.separateBias;
  } else {
    internalNet._currentGradClip = undefined;
    internalNet._gradClipSeparateBias = false;
  }
  if (options.mixedPrecision) {
    const mp =
      options.mixedPrecision === true
        ? { lossScale: 1024 }
        : options.mixedPrecision;
    internalNet._mixedPrecision.enabled = true;
    internalNet._mixedPrecision.lossScale = mp.lossScale || 1024;
    const dyn = mp.dynamic || {};
    internalNet._mixedPrecisionState.minLossScale = dyn.minScale || 1;
    internalNet._mixedPrecisionState.maxLossScale = dyn.maxScale || 65536;
    internalNet._mpIncreaseEvery =
      dyn.increaseEvery || dyn.stableStepsForIncrease || 200;
    net.connections.forEach((c) => {
      (c as any)._fp32Weight = c.weight;
    });
    net.nodes.forEach((n) => {
      if (n.type !== 'input') (n as any)._fp32Bias = n.bias;
    });
  } else {
    internalNet._mixedPrecision.enabled = false;
    internalNet._mixedPrecision.lossScale = 1;
    internalNet._mpIncreaseEvery = 200;
  }
  /** Supported optimizer algorithm identifiers (lowercased). */
  const allowedOptimizers = new Set([
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
  /** Normalized optimizer configuration or undefined for pure SGD mode. */
  let optimizerConfig: any = undefined;
  if (typeof options.optimizer !== 'undefined') {
    if (typeof options.optimizer === 'string')
      optimizerConfig = { type: options.optimizer.toLowerCase() };
    else if (
      typeof options.optimizer === 'object' &&
      options.optimizer !== null
    ) {
      optimizerConfig = { ...options.optimizer };
      if (typeof optimizerConfig.type === 'string')
        optimizerConfig.type = optimizerConfig.type.toLowerCase();
    } else
      throw new Error('Invalid optimizer option; must be string or object');
    if (!allowedOptimizers.has(optimizerConfig.type))
      throw new Error(`Unknown optimizer type: ${optimizerConfig.type}`);
    if (optimizerConfig.type === 'lookahead') {
      if (!optimizerConfig.baseType) optimizerConfig.baseType = 'adam';
      if (optimizerConfig.baseType === 'lookahead')
        throw new Error(
          'Nested lookahead (baseType lookahead) is not supported'
        );
      if (!allowedOptimizers.has(optimizerConfig.baseType))
        throw new Error(
          `Unknown baseType for lookahead: ${optimizerConfig.baseType}`
        );
      optimizerConfig.la_k = optimizerConfig.la_k || 5;
      optimizerConfig.la_alpha = optimizerConfig.la_alpha ?? 0.5;
    }
  }
  /** Maximum training iterations permitted (guard against infinite loops w/ only error criterion). */
  const iterations = options.iterations ?? Number.MAX_SAFE_INTEGER;
  /** Wall-clock start time for duration metric. */
  const start = Date.now();
  /** Most recent monitored (smoothed) error value. */
  let finalError = Infinity;
  /** Window length for primary moving average smoothing. */
  const movingAverageWindow = Math.max(1, options.movingAverageWindow || 1);
  /** Selected smoothing algorithm kind. */
  const movingAverageType = options.movingAverageType || 'sma';
  /** EMA alpha (if EMA selected) computed via CMA formula unless explicitly overridden. */
  const emaAlpha = (() => {
    if (movingAverageType !== 'ema') return undefined;
    if (options.emaAlpha && options.emaAlpha > 0 && options.emaAlpha <= 1)
      return options.emaAlpha;
    return 2 / (movingAverageWindow + 1);
  })();
  /** Separate window for plateau detection (defaults to primary window). */
  const plateauWindow = Math.max(
    1,
    options.plateauMovingAverageWindow || movingAverageWindow
  );
  /** Smoothing algorithm used specifically for plateau (scheduler / early-stop) metrics. */
  const plateauType = options.plateauMovingAverageType || movingAverageType;
  /** EMA alpha for plateau smoothing if needed. */
  const plateauEmaAlpha = (() => {
    if (plateauType !== 'ema') return undefined;
    if (
      options.plateauEmaAlpha &&
      options.plateauEmaAlpha > 0 &&
      options.plateauEmaAlpha <= 1
    )
      return options.plateauEmaAlpha;
    return 2 / (plateauWindow + 1);
  })();
  /** Max consecutive non-improving iterations tolerated before early stop (undefined => disabled). */
  const earlyStopPatience = options.earlyStopPatience;
  /** Minimal decrease required to qualify as improvement. */
  const earlyStopMinDelta = options.earlyStopMinDelta || 0;
  /** Best (lowest) monitored error observed so far. */
  let bestError = Infinity;
  /** Count of successive iterations without sufficient improvement. */
  let noImproveCount = 0;
  /** Capacity of circular buffer for recent errors. */
  const recentErrorsCapacity = movingAverageWindow;
  /** Circular buffer holding recent raw training errors (for smoothing). */
  const recentErrorsBuf: number[] = new Array(recentErrorsCapacity);
  /** Current number of valid entries in buffer (grows until capacity). */
  let recentErrorsCount = 0;
  /** Next write index within circular buffer. */
  let recentErrorsWriteIdx = 0;
  /** Push a new error value into circular buffer (overwriting oldest when full). */
  const recentErrorsPush = (value: number) => {
    if (recentErrorsCapacity === 1) {
      recentErrorsBuf[0] = value;
      recentErrorsCount = 1;
      recentErrorsWriteIdx = 0;
      return;
    }
    recentErrorsBuf[recentErrorsWriteIdx] = value;
    recentErrorsWriteIdx = (recentErrorsWriteIdx + 1) % recentErrorsCapacity;
    if (recentErrorsCount < recentErrorsCapacity) recentErrorsCount++;
  };
  /** Produce chronologically ordered snapshot of buffered errors. */
  const recentErrorsChrono = (): number[] => {
    if (recentErrorsCount === 0) return [];
    if (recentErrorsCount < recentErrorsCapacity)
      return recentErrorsBuf.slice(0, recentErrorsCount);
    const out = new Array(recentErrorsCount);
    const start = recentErrorsWriteIdx;
    for (let i = 0; i < recentErrorsCount; i++)
      out[i] = recentErrorsBuf[(start + i) % recentErrorsCapacity];
    return out;
  };
  /** Exponential moving average state for classic EMA smoothing. */
  let emaValue: number | undefined = undefined;
  /** Base EMA state for adaptive EMA (lower variance baseline). */
  let adaptiveBaseEmaValue: number | undefined = undefined;
  /** Adaptive EMA state (higher alpha when volatility detected). */
  let adaptiveEmaValue: number | undefined = undefined;
  /** Capacity of plateau circular buffer. */
  const plateauCapacity = plateauWindow;
  /** Raw errors buffer for plateau smoothing. */
  const plateauBuf: number[] = new Array(plateauCapacity);
  /** Current number of plateau entries filled. */
  let plateauCount = 0;
  /** Next write index for plateau buffer. */
  let plateauWriteIdx = 0;
  /** Insert new training error into plateau buffer. */
  const plateauPush = (value: number) => {
    if (plateauCapacity === 1) {
      plateauBuf[0] = value;
      plateauCount = 1;
      plateauWriteIdx = 0;
      return;
    }
    plateauBuf[plateauWriteIdx] = value;
    plateauWriteIdx = (plateauWriteIdx + 1) % plateauCapacity;
    if (plateauCount < plateauCapacity) plateauCount++;
  };
  /** Chronologically ordered plateau buffer snapshot. */
  const plateauChrono = (): number[] => {
    if (plateauCount === 0) return [];
    if (plateauCount < plateauCapacity)
      return plateauBuf.slice(0, plateauCount);
    const out = new Array(plateauCount);
    const start = plateauWriteIdx;
    for (let i = 0; i < plateauCount; i++)
      out[i] = plateauBuf[(start + i) % plateauCapacity];
    return out;
  };
  /** Plateau-specific EMA state (if plateauType === 'ema'). */
  let plateauEmaValue: number | undefined = undefined;
  /** Mutate network dropout probability for upcoming epoch iterations. */
  net.dropout = dropout;
  /** Number of iterations actually executed (in case of early stopping). */
  let performedIterations = 0;
  for (let iter = 1; iter <= iterations; iter++) {
    // -----------------------------
    // Iteration prologue
    // -----------------------------
    // 'iter' is 1-based to align with common optimizer bias-correction formulae (Adam etc.).
    if ((net as any)._maybePrune) {
      (net as any)._maybePrune((internalNet._globalEpoch || 0) + iter);
    }
    // Run one epoch pass over dataset (mini-batching handled internally) and obtain raw mean error.
    const trainError = trainSetImpl(
      net,
      set,
      batchSize,
      accumulationSteps,
      baseRate,
      momentum,
      {},
      cost as any,
      optimizerConfig
    );
    // Record that this iteration was fully executed (used if we early break afterwards).
    performedIterations = iter;
    // Push raw error into smoothing buffer(s) for subsequent moving-average computation.
    recentErrorsPush(trainError);
    /** Monitored error value after smoothing strategy is applied (initially raw). */
    let monitored = trainError;
    // -----------------------------
    // Primary moving-average smoothing block
    // -----------------------------
    // Conditions: apply if window > 1 or a strategy that inherently disregards window size (ema/adaptive).
    if (
      movingAverageWindow > 1 ||
      movingAverageType === 'ema' ||
      movingAverageType === 'adaptive-ema'
    ) {
      const recentArr = recentErrorsChrono();
      if (movingAverageType === 'median') {
        // Robust central tendency; reduces influence of transient spikes.
        const sorted = [...recentArr].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2); // middle index
        monitored =
          sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
      } else if (movingAverageType === 'ema') {
        // Classic exponentially weighted moving average (constant alpha).
        if (emaValue == null) emaValue = trainError;
        else emaValue = emaValue + emaAlpha! * (trainError - emaValue);
        monitored = emaValue;
      } else if (movingAverageType === 'adaptive-ema') {
        // Dual EMA: baseline + adaptive alpha that expands under variance to speed reaction, then we keep min.
        const mean = recentArr.reduce((a, b) => a + b, 0) / recentArr.length;
        const variance =
          recentArr.reduce((a, b) => a + (b - mean) * (b - mean), 0) /
          recentArr.length;
        const baseAlpha = emaAlpha || 2 / (movingAverageWindow + 1);
        const varScaled = variance / Math.max(mean * mean, 1e-8);
        const adaptAlpha = Math.min(
          0.95,
          Math.max(baseAlpha, baseAlpha * (1 + 2 * varScaled))
        );
        if (adaptiveBaseEmaValue == null) {
          adaptiveBaseEmaValue = trainError;
          adaptiveEmaValue = trainError;
        } else {
          adaptiveBaseEmaValue =
            adaptiveBaseEmaValue +
            baseAlpha * (trainError - adaptiveBaseEmaValue);
          adaptiveEmaValue =
            adaptiveEmaValue! + adaptAlpha * (trainError - adaptiveEmaValue!);
        }
        monitored = Math.min(adaptiveEmaValue!, adaptiveBaseEmaValue!);
      } else if (movingAverageType === 'gaussian') {
        // Weighted by Gaussian kernel centered at newest point; older (earlier) points get progressively less weight.
        const gaussianWindow = recentArr;
        const windowLength = gaussianWindow.length;
        const sigma = movingAverageWindow / 3 || 1; // heuristic: cover window with ~3 sigma
        let gaussianWeightSum = 0;
        let gaussianWeightedAccumulator = 0;
        for (let gi = 0; gi < windowLength; gi++) {
          const weight = Math.exp(
            -0.5 * Math.pow((gi - (windowLength - 1)) / sigma, 2)
          );
          gaussianWeightSum += weight;
          gaussianWeightedAccumulator += weight * gaussianWindow[gi];
        }
        monitored = gaussianWeightedAccumulator / (gaussianWeightSum || 1);
      } else if (movingAverageType === 'trimmed') {
        // Trim symmetrical tails to damp outliers before averaging.
        const tailTrimRatio = Math.min(
          0.49,
          Math.max(0, options.trimmedRatio || 0.1)
        );
        const sorted = [...recentArr].sort((a, b) => a - b);
        const elementsToDropEachSide = Math.floor(
          sorted.length * tailTrimRatio
        );
        const trimmedSegment = sorted.slice(
          elementsToDropEachSide,
          sorted.length - elementsToDropEachSide
        );
        monitored =
          trimmedSegment.reduce((a, b) => a + b, 0) /
          (trimmedSegment.length || 1);
      } else if (movingAverageType === 'wma') {
        // Linear weights: newer samples more influential.
        let linearWeightSum = 0;
        let linearWeightedAccumulator = 0;
        for (let li = 0; li < recentArr.length; li++) {
          const weight = li + 1; // oldest gets 1, newest gets N
          linearWeightSum += weight;
          linearWeightedAccumulator += weight * recentArr[li];
        }
        monitored = linearWeightedAccumulator / (linearWeightSum || 1);
      } else {
        // Simple arithmetic mean (SMA).
        monitored = recentArr.reduce((a, b) => a + b, 0) / recentArr.length;
      }
    }
    // Update finalError with the smoothed/selected monitored metric.
    finalError = monitored;
    // Store raw trainError (not smoothed) for plateau evaluation buffer.
    plateauPush(trainError);
    /** Plateau-smoothed error (could use different smoothing strategy than monitored). */
    let plateauError: number | undefined = trainError;
    if (plateauWindow > 1 || plateauType === 'ema') {
      if (plateauType === 'median') {
        // Median for plateau stability over variable noise.
        const sorted = [...plateauChrono()].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        plateauError =
          sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
      } else if (plateauType === 'ema') {
        // EMA variant for plateau detection (faster adaptation with controlled lag).
        if (plateauEmaValue == null) plateauEmaValue = trainError;
        else
          plateauEmaValue =
            plateauEmaValue + plateauEmaAlpha! * (trainError - plateauEmaValue);
        plateauError = plateauEmaValue;
      } else {
        // Default plateau = arithmetic mean over plateau window.
        const arr = plateauChrono();
        plateauError = arr.reduce((a, b) => a + b, 0) / arr.length;
      }
    }
    if (typeof options.metricsHook === 'function') {
      try {
        // User hook for live metrics logging / dashboards / adaptive schedulers.
        options.metricsHook({
          iteration: iter,
          error: finalError,
          plateauError,
          gradNorm: internalNet._lastGradNorm ?? 0,
        });
      } catch {}
    }
    if (options.checkpoint && typeof options.checkpoint.save === 'function') {
      if (options.checkpoint.last) {
        try {
          // Always save most recent network state.
          options.checkpoint.save({
            type: 'last',
            iteration: iter,
            error: finalError,
            network: net.toJSON(),
          });
        } catch {}
      }
      if (options.checkpoint.best) {
        if (
          finalError < (net as any)._checkpointBestError ||
          (net as any)._checkpointBestError == null
        ) {
          // New best model discovered under monitored error metric.
          (net as any)._checkpointBestError = finalError;
          try {
            options.checkpoint.save({
              type: 'best',
              iteration: iter,
              error: finalError,
              network: net.toJSON(),
            });
          } catch {}
        }
      }
    }
    if (
      options.schedule &&
      options.schedule.iterations &&
      iter % options.schedule.iterations === 0
    ) {
      try {
        // Periodic user-defined callback (e.g., adjust LR, print status, inject curriculum changes).
        options.schedule.function({ error: finalError, iteration: iter });
      } catch {}
    }
    // -----------------------------
    // Early stopping logic
    // -----------------------------
    if (finalError < bestError - earlyStopMinDelta) {
      // Sufficient improvement: update best and reset stagnation counter.
      bestError = finalError;
      noImproveCount = 0;
    } else if (earlyStopPatience) {
      // Track consecutive non-improving iterations.
      noImproveCount++;
    }
    // Patience exhaustion: terminate.
    if (earlyStopPatience && noImproveCount >= earlyStopPatience) break;
    // Target error reached: terminate.
    if (finalError <= targetError) break;
  }
  net.nodes.forEach((n) => {
    if (n.type === 'hidden') n.mask = 1;
  });
  // Clear dropout for inference after training completes.
  net.dropout = 0;
  internalNet._globalEpoch =
    (internalNet._globalEpoch || 0) + performedIterations;
  return {
    /** Final monitored (possibly smoothed) error achieved at termination. */
    error: finalError,
    /** Number of iterations actually executed (could be < requested iterations due to early stop). */
    iterations: performedIterations,
    /** Wall-clock training duration in milliseconds. */
    time: Date.now() - start,
  };
}
