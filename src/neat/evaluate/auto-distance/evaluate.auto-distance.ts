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
 *
 * The boundary is deliberately narrower than the surrounding speciation story.
 * By the time these helpers run, evaluation has already produced fresh scores
 * and diversity evidence. Auto-distance tuning does not change current species
 * membership or recompute pairwise compatibility distances in place. Instead it
 * adjusts the coefficients that later compatibility reads will use, so the next
 * cycle can respond to whether topology size is converging too quickly or
 * spreading too wildly.
 *
 * Read this chapter when you want to answer questions such as:
 * - Why does evaluation tune excess and disjoint coefficients from connection
 *   variance instead of hard-coding them forever?
 * - Why does this policy compare against a moving baseline instead of a single
 *   target number?
 * - What does the first-run bootstrap do?
 * - Which controller assumptions stay stable for later selection and
 *   speciation reads?
 *
 * The mental model is a four-step loop:
 * 1. collect connection counts from the freshly evaluated population,
 * 2. reduce them into a mean and variance,
 * 3. compare the new variance with the last stored baseline,
 * 4. raise or lower excess and disjoint coefficients inside safe bounds.
 *
 * This boundary preserves the currently evaluated population as-is. Scores,
 * population order, and live species membership stay untouched; only the
 * distance coefficients are prepared for later compatibility and speciation
 * work.
 *
 * ```mermaid
 * flowchart TD
 *   Population[Freshly evaluated population] --> Sizes[Collect connection counts]
 *   Sizes --> Stats[Compute mean and variance]
 *   Stats --> Baseline[Compare with stored variance baseline]
 *   Baseline --> Bootstrap[First run: seed baseline and bootstrap coefficients]
 *   Baseline --> Lower[Variance lower than baseline band: increase coefficients]
 *   Baseline --> Higher[Variance higher than baseline band: decrease coefficients]
 *   Baseline --> Stable[Inside bands: keep coefficients]
 *   Bootstrap --> Next[Next compatibility reads use updated coefficients]
 *   Lower --> Next
 *   Higher --> Next
 *   Stable --> Next
 * ```
 */

/**
 * Apply variance-driven tuning to the controller's distance coefficients.
 *
 * This is the controller-facing entrypoint for auto-distance tuning. It only
 * runs when both speciation and auto coefficient tuning are enabled because the
 * resulting coefficients are only meaningful when later compatibility reads are
 * still part of the runtime policy.
 *
 * The helper preserves several important controller assumptions:
 * - genome scores are already complete and are not recomputed here,
 * - live species assignments are left intact,
 * - population order is left intact,
 * - only the structural-distance coefficients and their variance baseline are
 *   updated for future passes.
 *
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 *
 * @example
 * ```ts
 * runAutoDistanceCoefficientTuning(controller, controller.options);
 * console.log(controller.options.excessCoeff, controller.options.disjointCoeff);
 * ```
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
 * This small reducer keeps the higher-level tuning flow declarative: collect
 * connection counts first, then summarize them before making policy
 * decision.
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
 * The tuning policy watches variance rather than just the raw mean so it can
 * notice when topology sizes are collapsing toward one narrow shape or
 * spreading apart more aggressively than before.
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
 * This helper compares the freshly observed variance with the controller's
 * stored baseline. Lower-than-expected variance means topology sizes are
 * converging, so excess and disjoint coefficients are increased to make future
 * structural differences matter more. Higher-than-expected variance means the
 * population is already spreading structurally, so the coefficients are eased
 * downward.
 *
 * The comparison is deliberately relative instead of target-based. The policy
 * tracks the population's recent structural spread and responds to drift,
 * rather than forcing every problem domain toward one global variance number.
 *
 * @param controller - NEAT controller instance for evaluation.
 * @param autoDistanceCoeffOptions - Tuning options that define adjustment rate
 * and coefficient bounds.
 * @param connectionVariance - Freshly observed variance of population
 * connection counts.
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
 * These bounds keep automatic tuning from shrinking structural-distance
 * pressure until compatibility becomes toothless, or increasing it until small
 * topology edits dominate every comparison.
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
 * The first run has no historical baseline yet, so this helper seeds the last
 * observed variance and applies one deterministic coefficient nudge. That makes
 * the policy immediately visible instead of waiting one extra generation before
 * a coefficient change is possible.
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
 * Increasing these coefficients makes later compatibility reads treat excess
 * and disjoint structural differences as more important, which helps push back
 * when topology sizes are collapsing toward one narrow profile.
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
 * Decreasing these coefficients softens the structural-distance penalty when
 * topology sizes are already spreading, which helps keep the controller from
 * over-fragmenting species on the next pass.
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
