import type {
  MonitoredSmoothingConfig,
  PlateauSmoothingConfig,
  PlateauSmoothingState,
  PrimarySmoothingState,
} from '../network.types';

/**
 * Compute monitored training error using the configured smoothing strategy.
 *
 * @param trainError - Raw training error for the current iteration.
 * @param recentErrors - Chronological recent error window (oldest to newest).
 * @param cfg - Monitored smoothing configuration.
 * @param state - Mutable smoothing state for EMA-based modes.
 * @returns Smoothed monitored error.
 */
export const computeMonitoredError = (
  trainError: number,
  recentErrors: number[],
  cfg: MonitoredSmoothingConfig,
  state: PrimarySmoothingState,
): number => {
  if (cfg.window <= 1 && cfg.type !== 'ema' && cfg.type !== 'adaptive-ema') {
    return trainError;
  }
  const type = cfg.type;
  if (type === 'median') {
    const sorted = recentErrors.toSorted((a, b) => a - b);
    const midIndex = Math.floor(sorted.length / 2);
    return sorted.length % 2
      ? sorted[midIndex]
      : (sorted[midIndex - 1] + sorted[midIndex]) / 2;
  }
  if (type === 'ema') {
    if (state.emaValue == null) state.emaValue = trainError;
    else {
      state.emaValue =
        state.emaValue + cfg.emaAlpha! * (trainError - state.emaValue);
    }
    return state.emaValue;
  }
  if (type === 'adaptive-ema') {
    const mean = recentErrors.reduce((a, b) => a + b, 0) / recentErrors.length;
    const variance =
      recentErrors.reduce((a, b) => a + (b - mean) * (b - mean), 0) /
      recentErrors.length;
    const baseAlpha = cfg.emaAlpha || 2 / (cfg.window + 1);
    const varianceScaled = variance / Math.max(mean * mean, 1e-8);
    const adaptiveAlpha = Math.min(
      0.95,
      Math.max(baseAlpha, baseAlpha * (1 + 2 * varianceScaled)),
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
    const sigma = cfg.window / 3;
    let weightSum = 0;
    let weightedAccumulator = 0;
    const length = recentErrors.length;
    for (let sampleIndex = 0; sampleIndex < length; sampleIndex++) {
      const weight = Math.exp(
        -0.5 * Math.pow((sampleIndex - (length - 1)) / sigma, 2),
      );
      weightSum += weight;
      weightedAccumulator += weight * recentErrors[sampleIndex];
    }
    return weightedAccumulator / weightSum;
  }
  if (type === 'trimmed') {
    const ratio = Math.min(0.49, Math.max(0, cfg.trimmedRatio || 0.1));
    const sorted = recentErrors.toSorted((a, b) => a - b);
    const drop = Math.floor(sorted.length * ratio);
    const trimmed = sorted.slice(drop, sorted.length - drop);
    return trimmed.reduce((a, b) => a + b, 0) / trimmed.length;
  }
  if (type === 'wma') {
    let weightSum = 0;
    let weightedAccumulator = 0;
    for (
      let sampleIndex = 0;
      sampleIndex < recentErrors.length;
      sampleIndex++
    ) {
      const weight = sampleIndex + 1;
      weightSum += weight;
      weightedAccumulator += weight * recentErrors[sampleIndex];
    }
    return weightedAccumulator / weightSum;
  }
  return recentErrors.reduce((a, b) => a + b, 0) / recentErrors.length;
};

/**
 * Compute plateau metric using the configured plateau smoothing strategy.
 *
 * @param trainError - Raw training error for the current iteration.
 * @param plateauErrors - Plateau window of recent raw errors.
 * @param cfg - Plateau smoothing configuration.
 * @param state - Mutable state for plateau EMA.
 * @returns Smoothed plateau metric.
 */
export const computePlateauMetric = (
  trainError: number,
  plateauErrors: number[],
  cfg: PlateauSmoothingConfig,
  state: PlateauSmoothingState,
): number => {
  if (cfg.window <= 1 && cfg.type !== 'ema') return trainError;
  if (cfg.type === 'median') {
    const sorted = plateauErrors.toSorted((a, b) => a - b);
    const middleIndex = Math.floor(sorted.length / 2);
    return sorted.length % 2
      ? sorted[middleIndex]
      : (sorted[middleIndex - 1] + sorted[middleIndex]) / 2;
  }
  if (cfg.type === 'ema') {
    if (state.plateauEmaValue == null) state.plateauEmaValue = trainError;
    else {
      state.plateauEmaValue =
        state.plateauEmaValue +
        cfg.emaAlpha! * (trainError - state.plateauEmaValue);
    }
    return state.plateauEmaValue;
  }
  return plateauErrors.reduce((a, b) => a + b, 0) / plateauErrors.length;
};
