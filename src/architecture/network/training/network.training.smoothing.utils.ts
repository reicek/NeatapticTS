import type {
  MonitoredSmoothingConfig,
  PlateauSmoothingConfig,
  PlateauSmoothingState,
  PrimarySmoothingState,
} from '../network.types';

/**
 * Compute monitored training error using the configured smoothing strategy.
 *
 * The helper returns the raw error when smoothing is effectively disabled.
 * For stateful modes (`ema`, `adaptive-ema`), the provided state object is
 * updated in place so callers can keep continuity across iterations.
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
  if (shouldBypassMonitoredSmoothing(cfg)) {
    return trainError;
  }
  if (cfg.type === 'median') {
    return computeMedianMonitoredError(recentErrors);
  }
  if (cfg.type === 'ema') {
    return computeEmaMonitoredError(trainError, cfg, state);
  }
  if (cfg.type === 'adaptive-ema') {
    return computeAdaptiveEmaMonitoredError(
      trainError,
      recentErrors,
      cfg,
      state,
    );
  }
  if (cfg.type === 'gaussian') {
    return computeGaussianMonitoredError(recentErrors, cfg.window);
  }
  if (cfg.type === 'trimmed') {
    return computeTrimmedMonitoredError(recentErrors, cfg.trimmedRatio);
  }
  if (cfg.type === 'wma') {
    return computeWeightedMonitoredError(recentErrors);
  }
  return computeAverageMonitoredError(recentErrors);
};

const shouldBypassMonitoredSmoothing = (
  cfg: MonitoredSmoothingConfig,
): boolean =>
  cfg.window <= 1 && cfg.type !== 'ema' && cfg.type !== 'adaptive-ema';

const computeMedianMonitoredError = (recentErrors: number[]): number => {
  const sorted = recentErrors.toSorted((a, b) => a - b);
  const middleIndex = Math.floor(sorted.length / 2);

  return sorted.length % 2
    ? sorted[middleIndex]
    : (sorted[middleIndex - 1] + sorted[middleIndex]) / 2;
};

const computeEmaMonitoredError = (
  trainError: number,
  cfg: MonitoredSmoothingConfig,
  state: PrimarySmoothingState,
): number => {
  if (state.emaValue == null) {
    state.emaValue = trainError;
  } else {
    state.emaValue =
      state.emaValue + cfg.emaAlpha! * (trainError - state.emaValue);
  }

  return state.emaValue;
};

const computeAdaptiveEmaMonitoredError = (
  trainError: number,
  recentErrors: number[],
  cfg: MonitoredSmoothingConfig,
  state: PrimarySmoothingState,
): number => {
  const mean = computeAverageMonitoredError(recentErrors);
  const variance = computeAverageMonitoredError(
    recentErrors.map((error) => {
      const errorDelta = error - mean;
      return errorDelta * errorDelta;
    }),
  );
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
};

const computeGaussianMonitoredError = (
  recentErrors: number[],
  window: number,
): number => {
  const sigma = window / 3;
  let weightSum = 0;
  let weightedAccumulator = 0;
  const latestSampleIndex = recentErrors.length - 1;

  for (let sampleIndex = 0; sampleIndex < recentErrors.length; sampleIndex++) {
    const weight = Math.exp(
      -0.5 * Math.pow((sampleIndex - latestSampleIndex) / sigma, 2),
    );
    weightSum += weight;
    weightedAccumulator += weight * recentErrors[sampleIndex];
  }

  return weightedAccumulator / weightSum;
};

const computeTrimmedMonitoredError = (
  recentErrors: number[],
  trimmedRatio: number | undefined,
): number => {
  const ratio = Math.min(0.49, Math.max(0, trimmedRatio || 0.1));
  const sorted = recentErrors.toSorted((a, b) => a - b);
  const droppedSampleCount = Math.floor(sorted.length * ratio);
  const trimmed = sorted.slice(
    droppedSampleCount,
    sorted.length - droppedSampleCount,
  );

  return computeAverageMonitoredError(trimmed);
};

const computeWeightedMonitoredError = (recentErrors: number[]): number => {
  let weightSum = 0;
  let weightedAccumulator = 0;

  for (let sampleIndex = 0; sampleIndex < recentErrors.length; sampleIndex++) {
    const weight = sampleIndex + 1;
    weightSum += weight;
    weightedAccumulator += weight * recentErrors[sampleIndex];
  }

  return weightedAccumulator / weightSum;
};

const computeAverageMonitoredError = (recentErrors: number[]): number =>
  recentErrors.reduce((sum, value) => sum + value, 0) / recentErrors.length;

/**
 * Compute plateau metric using the configured plateau smoothing strategy.
 *
 * This metric is intentionally independent from the primary monitored metric so
 * plateau detection can use a different noise profile.
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
