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
 */

/**
 * Adjust the compatibility threshold when entropy-compatibility tuning is enabled.
 *
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
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
 * @param entropyCompatOptions - Tuning options.
 * @param meanEntropy - Current mean entropy.
 * @param currentThreshold - Current compatibility threshold.
 * @returns Next compatibility threshold.
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
