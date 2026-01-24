import {
  ensureDiversityStatsContainer,
  runAutoDistanceCoefficientTuning,
  runAutoEntropyObjectiveInjection,
  runEntropyCompatibilityTuning,
  runEntropySharingTuning,
  runFitnessEvaluation,
  runLightweightSpeciation,
  runNoveltyBlendAndArchive,
} from './neat.evaluate.utils';
import type { NeatControllerForEval } from './neat.evaluate.utils';

export {
  NOVELTY_DEFAULT_NEIGHBORS,
  NOVELTY_DEFAULT_BLEND,
  NOVELTY_ARCHIVE_CAP,
  ENTROPY_VAR_TARGET_DEFAULT,
  ENTROPY_VAR_ADJUST_DEFAULT,
  ENTROPY_VAR_MIN_SIGMA_DEFAULT,
  ENTROPY_VAR_MAX_SIGMA_DEFAULT,
  ENTROPY_VAR_LOW_BAND,
  ENTROPY_VAR_HIGH_BAND,
  ENTROPY_TARGET_DEFAULT,
  ENTROPY_DEADBAND_DEFAULT,
  ENTROPY_ADJUST_DEFAULT,
  COMPAT_THRESHOLD_DEFAULT,
  COMPAT_MIN_THRESHOLD_DEFAULT,
  COMPAT_MAX_THRESHOLD_DEFAULT,
  AUTO_COEFF_ADJUST_DEFAULT,
  AUTO_COEFF_MIN_DEFAULT,
  AUTO_COEFF_MAX_DEFAULT,
  DISTANCE_COEFF_DEFAULT,
  VARIANCE_DECREASE_THRESHOLD,
  VARIANCE_INCREASE_THRESHOLD,
} from './neat.evaluate.utils';

/**
 * Evaluate the population or population-wide fitness delegate.
 *
 * This function mirrors the legacy `evaluate` behaviour used by NeatapticTS
 * but adds documentation and clearer local variable names for readability.
 *
 * Top-level responsibilities (method steps descriptions):
 * 1) Run fitness either on each genome or once for the population depending
 *    on `options.fitnessPopulation`.
 * 2) Optionally clear genome internal state before evaluation when
 *    `options.clear` is set.
 * 3) After scoring, apply optional novelty blending using a user-supplied
 *    descriptor function. Novelty is blended into scores using a blend
 *    factor and may be archived.
 * 4) Apply several adaptive tuning behaviors (entropy-sharing, compatibility
 *    threshold tuning, auto-distance coefficient tuning) guarded by options.
 * 5) Trigger light-weight speciation when speciation-related controller
 *    options are enabled so tests that only call evaluate still exercise
 *    threshold tuning.
 *
 * Example usage:
 * // await evaluate.call(controller); // where controller has `population`, `fitness` etc.
 *
 * @returns Promise<void> resolves after evaluation and adaptive updates complete.
 */
export async function evaluate(this: NeatControllerForEval): Promise<void> {
  // Delegate-evaluated version of the fallback in src/neat.ts
  /**
   * The options object for the running NEAT controller.
   *
   * This is a shallow accessor to `this.options` that guarantees an object
   * is available during evaluation. Options control behaviour such as whether
   * the fitness delegate is called per-genome or once for the population,
   * whether various automatic tuning features are enabled, and novelty
   * behaviour.
   *
   * Example:
   * const controller = { options: { fitnessPopulation: false } };
   * await evaluate.call(controller);
   */
  const options = this.options || {};

  // === Declarative evaluation flow ===
  await runFitnessEvaluation(this, options);
  runNoveltyBlendAndArchive(this, options);
  ensureDiversityStatsContainer(this);
  runEntropySharingTuning(this, options);
  runEntropyCompatibilityTuning(this, options);
  runLightweightSpeciation(this, options);
  runAutoDistanceCoefficientTuning(this, options);
  runAutoEntropyObjectiveInjection(this, options);
}

export default { evaluate };
