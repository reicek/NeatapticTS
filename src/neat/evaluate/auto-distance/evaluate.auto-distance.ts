import type { NeatControllerForEval } from '../shared/evaluate.types';
import {
  AUTO_COEFF_ADJUST_DEFAULT,
  AUTO_COEFF_MAX_DEFAULT,
  AUTO_COEFF_MIN_DEFAULT,
  DISTANCE_COEFF_DEFAULT,
  VARIANCE_DECREASE_THRESHOLD,
  VARIANCE_INCREASE_THRESHOLD,
} from '../shared/evaluate.constants';

/**
 * Auto-distance tuning helpers for the NEAT evaluate chapter.
 *
 * This chapter owns the variance-driven adjustment of compatibility-distance
 * coefficients. It watches connection-count variance across the evaluated
 * population and nudges the controller's excess and disjoint coefficients up or
 * down to keep structural diversity from collapsing.
 */

/**
 * Apply variance-driven tuning to the controller's distance coefficients.
 *
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 */
export function runAutoDistanceCoefficientTuning(
  controller: NeatControllerForEval,
  evaluationOptions: NeatControllerForEval['options'],
): void {
  // Step 1: Exit unless speciation and auto tuning are both enabled.
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
    // Distance-coefficient tuning is optional and should not fail evaluation.
  }
}

/**
 * Compute the arithmetic mean of a number list.
 *
 * @param values - Input values.
 * @returns Mean of the values.
 */
function computeMean(values: number[]): number {
  if (values.length === 0) return 0;
  return (
    values.reduce((accumulated, value) => accumulated + value, 0) /
    values.length
  );
}

/**
 * Compute the variance of a number list from a precomputed mean.
 *
 * @param values - Input values.
 * @param meanValue - Precomputed mean.
 * @returns Variance of the values.
 */
function computeVariance(values: number[], meanValue: number): number {
  if (values.length === 0) return 0;
  const squaredSum = values.reduce(
    (accumulated, value) => accumulated + (value - meanValue) ** 2,
    0,
  );
  return squaredSum / values.length;
}

/**
 * Apply the coefficient-tuning policy using the observed connection variance.
 *
 * @param controller - NEAT controller instance for evaluation.
 * @param autoDistanceCoeffOptions - Tuning options.
 * @param connectionVariance - Variance of connection counts.
 */
function applyAutoDistanceCoefficientTuning(
  controller: NeatControllerForEval,
  autoDistanceCoeffOptions: NonNullable<
    NeatControllerForEval['options']['autoDistanceCoeffTuning']
  >,
  connectionVariance: number,
): void {
  const tuningBounds = getDistanceCoefficientBounds(autoDistanceCoeffOptions);
  const adjustRate =
    autoDistanceCoeffOptions.adjustRate ?? AUTO_COEFF_ADJUST_DEFAULT;

  // Step 1: Bootstrap the moving baseline the first time tuning runs.
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

  // Step 2: Compare the new variance to the moving baseline and adjust.
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
 * Resolve the minimum and maximum coefficient bounds for tuning.
 *
 * @param autoDistanceCoeffOptions - Tuning options.
 * @returns Min and max coefficient bounds.
 */
function getDistanceCoefficientBounds(
  autoDistanceCoeffOptions: NonNullable<
    NeatControllerForEval['options']['autoDistanceCoeffTuning']
  >,
): { minCoeff: number; maxCoeff: number } {
  return {
    minCoeff: autoDistanceCoeffOptions.minCoeff ?? AUTO_COEFF_MIN_DEFAULT,
    maxCoeff: autoDistanceCoeffOptions.maxCoeff ?? AUTO_COEFF_MAX_DEFAULT,
  };
}

/**
 * Bootstrap connection-variance tuning the first time the policy runs.
 *
 * @param controller - NEAT controller instance for evaluation.
 * @param connectionVariance - Current connection variance.
 * @param bounds - Min and max coefficient bounds.
 * @param adjustRate - Adjustment rate.
 */
function initializeConnectionVarianceBootstrap(
  controller: NeatControllerForEval,
  connectionVariance: number,
  bounds: { minCoeff: number; maxCoeff: number },
  adjustRate: number,
): void {
  controller._lastConnVar = connectionVariance;

  // Step 1: Apply a deterministic bootstrap nudge so the policy has an effect.
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
    // Bootstrap is best-effort; failed writes should not stop evaluation.
  }
}

/**
 * Increase distance coefficients within the configured bounds.
 *
 * @param controller - NEAT controller instance for evaluation.
 * @param bounds - Min and max coefficient bounds.
 * @param adjustRate - Adjustment rate.
 */
function applyDistanceCoefficientIncrease(
  controller: NeatControllerForEval,
  bounds: { minCoeff: number; maxCoeff: number },
  adjustRate: number,
): void {
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
 * Decrease distance coefficients within the configured bounds.
 *
 * @param controller - NEAT controller instance for evaluation.
 * @param bounds - Min and max coefficient bounds.
 * @param adjustRate - Adjustment rate.
 */
function applyDistanceCoefficientDecrease(
  controller: NeatControllerForEval,
  bounds: { minCoeff: number; maxCoeff: number },
  adjustRate: number,
): void {
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
