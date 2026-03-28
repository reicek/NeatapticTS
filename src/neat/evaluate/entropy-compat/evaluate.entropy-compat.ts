import type { NeatControllerForEval } from '../shared/evaluate.types';
import {
  COMPAT_MAX_THRESHOLD_DEFAULT,
  COMPAT_MIN_THRESHOLD_DEFAULT,
  COMPAT_THRESHOLD_DEFAULT,
  ENTROPY_ADJUST_DEFAULT,
  ENTROPY_DEADBAND_DEFAULT,
  ENTROPY_TARGET_DEFAULT,
} from '../shared/evaluate.constants';

/**
 * Entropy-compatibility tuning helpers for the NEAT evaluate chapter.
 *
 * This chapter owns the adaptive compatibility-threshold adjustment that keeps
 * the controller's speciation threshold moving with the observed mean entropy
 * of the evaluated population.
 *
 * The boundary stays intentionally small. Evaluation has already produced
 * fresh scores, novelty signals, and diversity measurements by the time these
 * helpers run. This chapter does not reassign species, rescore genomes, or
 * reinterpret the whole speciation policy. Its job is narrower: read the fresh
 * mean entropy signal and decide whether the next pass should loosen, tighten,
 * or preserve the compatibility threshold.
 *
 * Read this chapter when you want to answer questions such as:
 * - Why does evaluation tune compatibility threshold from mean entropy instead
 *   of folding that logic into speciation itself?
 * - What does the deadband do, and why is it safer than reacting to every
 *   small entropy fluctuation?
 * - How do adjustment rate and threshold bounds interact?
 * - Which controller assumptions stay stable for later selection and
 *   speciation reads?
 *
 * The mental model is a short three-step control loop:
 * 1. read the fresh mean entropy after evaluation,
 * 2. compare it with the configured target and deadband,
 * 3. nudge the compatibility threshold up or down inside its safe bounds.
 *
 * The preserved assumptions matter as much as the update. This boundary leaves
 * current genome scores, current species membership, and population order
 * untouched so downstream chapters can still reason about one stable evaluated
 * state while the next cycle inherits a better threshold.
 *
 * ```mermaid
 * flowchart TD
 *   Stats[Fresh mean entropy after evaluation] --> Guard[Require enabled tuning and numeric meanEntropy]
 *   Guard --> Compare[Compare mean entropy with target and deadband]
 *   Compare --> Low[Below band: tighten threshold]
 *   Compare --> Mid[Inside deadband: keep threshold]
 *   Compare --> High[Above band: widen threshold]
 *   Low --> Clamp[Clamp to configured min or max]
 *   Mid --> Clamp
 *   High --> Clamp
 *   Clamp --> Next[Next speciation pass reads updated threshold]
 * ```
 */

/**
 * Adjust the compatibility threshold when entropy-compatibility tuning is enabled.
 *
 * This is the controller-facing entrypoint for the entropy-compatibility
 * chapter. It behaves like best-effort maintenance after fresh diversity
 * evidence has been written, not like a required scoring or speciation phase.
 *
 * The helper preserves several important controller assumptions:
 * - genome scores are already complete and are not recomputed here,
 * - current species assignments are left intact,
 * - population order is left intact,
 * - only `compatibilityThreshold` is prepared for later passes.
 *
 * That narrow scope lets the evaluation chapter adjust future speciation
 * pressure without widening into a full species-rebuild workflow.
 *
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 *
 * @example
 * ```ts
 * controller._diversityStats = { meanEntropy: 0.42 };
 *
 * runEntropyCompatibilityTuning(controller, controller.options);
 * console.log(controller.options.compatibilityThreshold);
 * ```
 */
export function runEntropyCompatibilityTuning(
  controller: NeatControllerForEval,
  evaluationOptions: NeatControllerForEval['options'],
): void {
  // Step 1: Read mean entropy and update the threshold safely.
  try {
    const entropyCompatOptions = evaluationOptions.entropyCompatTuning;
    if (!entropyCompatOptions?.enabled) return;

    const meanEntropy = controller._diversityStats?.meanEntropy;
    if (typeof meanEntropy !== 'number') return;

    controller.options.compatibilityThreshold =
      computeNextCompatibilityThreshold(
        entropyCompatOptions,
        meanEntropy,
        controller.options.compatibilityThreshold ?? COMPAT_THRESHOLD_DEFAULT,
      );
  } catch {
    // Compatibility-threshold tuning is optional and should not fail evaluation.
  }
}

/**
 * Compute the next compatibility threshold from the observed mean entropy.
 *
 * The rule uses a target-plus-deadband shape instead of reacting to every
 * small change in mean entropy. When entropy falls below the lower edge of the
 * band, the threshold is reduced so the next speciation pass becomes stricter.
 * When entropy rises above the upper edge, the threshold is increased so the
 * next pass tolerates a wider neighborhood. Readings inside the deadband keep
 * the threshold unchanged to avoid jitter.
 *
 * The result is always clamped to the configured minimum and maximum so the
 * controller cannot drift into an unusably tiny or overly permissive threshold.
 *
 * @param entropyCompatOptions - Tuning options that define the target entropy,
 * deadband width, adjustment rate, and clamp bounds.
 * @param meanEntropy - Freshly observed mean structural entropy for the current
 * population.
 * @param currentThreshold - Current compatibility threshold before adjustment.
 * @returns Next compatibility threshold to carry into later controller passes.
 *
 * @example
 * ```ts
 * const nextThreshold = computeNextCompatibilityThreshold(
 *   { enabled: true, targetEntropy: 0.4, deadband: 0.02, adjustRate: 0.1, minThreshold: 1, maxThreshold: 6 },
 *   0.46,
 *   3,
 * );
 *
 * console.log(nextThreshold);
 * ```
 */
function computeNextCompatibilityThreshold(
  entropyCompatOptions: NonNullable<
    NeatControllerForEval['options']['entropyCompatTuning']
  >,
  meanEntropy: number,
  currentThreshold: number,
): number {
  const targetEntropy =
    entropyCompatOptions.targetEntropy ?? ENTROPY_TARGET_DEFAULT;
  const deadband = entropyCompatOptions.deadband ?? ENTROPY_DEADBAND_DEFAULT;
  const adjustRate = entropyCompatOptions.adjustRate ?? ENTROPY_ADJUST_DEFAULT;
  const minThreshold =
    entropyCompatOptions.minThreshold ?? COMPAT_MIN_THRESHOLD_DEFAULT;
  const maxThreshold =
    entropyCompatOptions.maxThreshold ?? COMPAT_MAX_THRESHOLD_DEFAULT;

  if (meanEntropy < targetEntropy - deadband) {
    return Math.max(minThreshold, currentThreshold * (1 - adjustRate));
  }
  if (meanEntropy > targetEntropy + deadband) {
    return Math.min(maxThreshold, currentThreshold * (1 + adjustRate));
  }

  return currentThreshold;
}
