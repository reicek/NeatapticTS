import type { NeatControllerForEval } from './neat.evaluate.types.utils';
import {
  AUTO_COEFF_ADJUST_DEFAULT,
  AUTO_COEFF_MAX_DEFAULT,
  AUTO_COEFF_MIN_DEFAULT,
  DISTANCE_COEFF_DEFAULT,
  VARIANCE_DECREASE_THRESHOLD,
  VARIANCE_INCREASE_THRESHOLD,
} from './neat.evaluate.constants.utils';

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 * @returns void.
 */
export function runAutoDistanceCoefficientTuning(
  controller: NeatControllerForEval,
  evaluationOptions: NeatControllerForEval['options'],
): void {
  // Step 1: Apply variance-driven coefficient tuning.
  try {
    const autoDistanceCoeffOptions = evaluationOptions.autoDistanceCoeffTuning;
    if (!autoDistanceCoeffOptions?.enabled || !evaluationOptions.speciation) {
      return;
    }
    const connectionSizes = controller.population.map(
      (genome) => genome.connections.length,
    );
    const meanConnectionSize = computeMean(connectionSizes);
    const connectionVariance = computeVariance(
      connectionSizes,
      meanConnectionSize,
    );
    applyAutoDistanceCoefficientTuning(
      controller,
      autoDistanceCoeffOptions,
      connectionVariance,
    );
  } catch {
    // Intentionally ignore auto-distance coefficient tuning errors
  }
}

/**
 * @param values - Input values.
 * @returns Mean of the values.
 */
function computeMean(values: number[]): number {
  // Step 1: Guard empty arrays.
  if (values.length === 0) return 0;
  return (
    values.reduce((accumulated, value) => accumulated + value, 0) /
    values.length
  );
}

/**
 * @param values - Input values.
 * @param meanValue - Precomputed mean.
 * @returns Variance of the values.
 */
function computeVariance(values: number[], meanValue: number): number {
  // Step 1: Guard empty arrays.
  if (values.length === 0) return 0;
  const squaredSum = values.reduce(
    (accumulated, value) => accumulated + (value - meanValue) ** 2,
    0,
  );
  return squaredSum / values.length;
}

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param autoDistanceCoeffOptions - Tuning options.
 * @param connectionVariance - Variance of connection counts.
 * @returns void.
 */
function applyAutoDistanceCoefficientTuning(
  controller: NeatControllerForEval,
  autoDistanceCoeffOptions: NonNullable<
    NeatControllerForEval['options']['autoDistanceCoeffTuning']
  >,
  connectionVariance: number,
): void {
  // Step 1: Initialize and bootstrap if needed.
  const tuningBounds = getDistanceCoefficientBounds(autoDistanceCoeffOptions);
  const adjustRate =
    autoDistanceCoeffOptions.adjustRate ?? AUTO_COEFF_ADJUST_DEFAULT;
  if (
    controller._lastConnVar === undefined ||
    controller._lastConnVar === null
  ) {
    initializeConnectionVarianceBootstrap(
      controller,
      connectionVariance,
      tuningBounds,
      adjustRate,
    );
  }

  // Step 2: Apply tuning based on variance delta.
  if (
    connectionVariance <
    (controller._lastConnVar ?? 0) * VARIANCE_DECREASE_THRESHOLD
  ) {
    applyDistanceCoefficientIncrease(controller, tuningBounds, adjustRate);
  } else if (
    connectionVariance >
    (controller._lastConnVar ?? 0) * VARIANCE_INCREASE_THRESHOLD
  ) {
    applyDistanceCoefficientDecrease(controller, tuningBounds, adjustRate);
  }
  controller._lastConnVar = connectionVariance;
}

/**
 * @param autoDistanceCoeffOptions - Tuning options.
 * @returns Bounds for coefficients.
 */
function getDistanceCoefficientBounds(
  autoDistanceCoeffOptions: NonNullable<
    NeatControllerForEval['options']['autoDistanceCoeffTuning']
  >,
): { minCoeff: number; maxCoeff: number } {
  // Step 1: Resolve bounds.
  return {
    minCoeff: autoDistanceCoeffOptions.minCoeff ?? AUTO_COEFF_MIN_DEFAULT,
    maxCoeff: autoDistanceCoeffOptions.maxCoeff ?? AUTO_COEFF_MAX_DEFAULT,
  };
}

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param connectionVariance - Current connection variance.
 * @param bounds - Min/max coefficients.
 * @param adjustRate - Adjustment rate.
 * @returns void.
 */
function initializeConnectionVarianceBootstrap(
  controller: NeatControllerForEval,
  connectionVariance: number,
  bounds: { minCoeff: number; maxCoeff: number },
  adjustRate: number,
): void {
  // Step 1: Record baseline variance.
  controller._lastConnVar = connectionVariance;
  // Step 2: Apply a deterministic nudge so tuning has a visible effect.
  try {
    controller.options.excessCoeff = Math.min(
      bounds.maxCoeff,
      (controller.options.excessCoeff ?? DISTANCE_COEFF_DEFAULT) *
        (1 + adjustRate),
    );
    controller.options.disjointCoeff = Math.min(
      bounds.maxCoeff,
      (controller.options.disjointCoeff ?? DISTANCE_COEFF_DEFAULT) *
        (1 + adjustRate),
    );
  } catch {
    // Intentionally ignore coefficient adjustment errors during bootstrap
  }
}

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param bounds - Min/max coefficients.
 * @param adjustRate - Adjustment rate.
 * @returns void.
 */
function applyDistanceCoefficientIncrease(
  controller: NeatControllerForEval,
  bounds: { minCoeff: number; maxCoeff: number },
  adjustRate: number,
): void {
  // Step 1: Increase coefficients within bounds.
  controller.options.excessCoeff = Math.min(
    bounds.maxCoeff,
    (controller.options.excessCoeff ?? DISTANCE_COEFF_DEFAULT) *
      (1 + adjustRate),
  );
  controller.options.disjointCoeff = Math.min(
    bounds.maxCoeff,
    (controller.options.disjointCoeff ?? DISTANCE_COEFF_DEFAULT) *
      (1 + adjustRate),
  );
}

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param bounds - Min/max coefficients.
 * @param adjustRate - Adjustment rate.
 * @returns void.
 */
function applyDistanceCoefficientDecrease(
  controller: NeatControllerForEval,
  bounds: { minCoeff: number; maxCoeff: number },
  adjustRate: number,
): void {
  // Step 1: Decrease coefficients within bounds.
  controller.options.excessCoeff = Math.max(
    bounds.minCoeff,
    (controller.options.excessCoeff ?? DISTANCE_COEFF_DEFAULT) *
      (1 - adjustRate),
  );
  controller.options.disjointCoeff = Math.max(
    bounds.minCoeff,
    (controller.options.disjointCoeff ?? DISTANCE_COEFF_DEFAULT) *
      (1 - adjustRate),
  );
}
