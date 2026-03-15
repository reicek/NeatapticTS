import {
  ensureDiversityStatsContainer,
  runEntropySharingTuning,
} from './entropy-sharing/evaluate.entropy-sharing';
import { runAutoDistanceCoefficientTuning } from './auto-distance/evaluate.auto-distance';
import { runEntropyCompatibilityTuning } from './entropy-compat/evaluate.entropy-compat';
import { runFitnessEvaluation } from './fitness/evaluate.fitness';
import { runNoveltyBlendAndArchive } from './novelty/evaluate.novelty';
import { runAutoEntropyObjectiveInjection } from './objectives/evaluate.objectives';
import { runLightweightSpeciation } from './speciation/evaluate.speciation';
import type { NeatControllerForEval } from './shared/evaluate.types';

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
} from './shared/evaluate.constants';
export type { NeatControllerForEval } from './shared/evaluate.types';

/**
 * Root orchestration for NEAT population evaluation.
 *
 * This chapter keeps the evaluation pipeline readable at the top level: run
 * fitness, optionally blend novelty into scores, then apply the small adaptive
 * tuning passes that depend on freshly computed population statistics.
 *
 * The neighboring chapters own the narrower mechanics:
 * - `fitness/` drives per-genome or population-level scoring.
 * - `novelty/` computes behavioral novelty, blending, and archive writes.
 * - `entropy-sharing/`, `entropy-compat/`, and `auto-distance/` tune the
 *   controller's diversity-related parameters.
 * - `speciation/` keeps lightweight post-evaluation species maintenance small.
 * - `objectives/` handles opt-in automatic entropy-objective registration.
 */

/**
 * Evaluate the population or population-wide fitness delegate.
 *
 * This preserves the long-standing `Neat.evaluate()` flow while moving the
 * implementation into a chaptered boundary that is easier to discover in the
 * generated docs.
 *
 * Top-level responsibilities:
 * 1. Run fitness either on each genome or once for the whole population.
 * 2. Blend novelty into scores when novelty search is enabled.
 * 3. Ensure diversity statistics storage exists before adaptive tuning.
 * 4. Apply entropy-sharing, entropy-compatibility, speciation, and automatic
 *    distance-coefficient adjustments.
 * 5. Register the entropy objective when multi-objective mode asks for it.
 *
 * @returns Promise that resolves after evaluation and adaptive follow-up steps.
 *
 * @example
 * ```ts
 * await evaluate.call(controller);
 * ```
 */
export async function evaluate(this: NeatControllerForEval): Promise<void> {
  const evaluationOptions = this.options || {};

  // Step 1: Score the population through the configured fitness delegate.
  await runFitnessEvaluation(this, evaluationOptions);

  // Step 2: Blend novelty and archive descriptors when novelty search is active.
  runNoveltyBlendAndArchive(this, evaluationOptions);

  // Step 3: Ensure downstream tuning has a diversity container to write into.
  ensureDiversityStatsContainer(this);

  // Step 4: Apply the lightweight adaptive post-evaluation maintenance steps.
  runEntropySharingTuning(this, evaluationOptions);
  runEntropyCompatibilityTuning(this, evaluationOptions);
  runLightweightSpeciation(this, evaluationOptions);
  runAutoDistanceCoefficientTuning(this, evaluationOptions);
  runAutoEntropyObjectiveInjection(this, evaluationOptions);
}

export default { evaluate };
