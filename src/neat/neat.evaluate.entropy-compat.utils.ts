import type { NeatControllerForEval } from './neat.evaluate.types.utils';
import {
  COMPAT_MAX_THRESHOLD_DEFAULT,
  COMPAT_MIN_THRESHOLD_DEFAULT,
  COMPAT_THRESHOLD_DEFAULT,
  ENTROPY_ADJUST_DEFAULT,
  ENTROPY_DEADBAND_DEFAULT,
  ENTROPY_TARGET_DEFAULT,
} from './neat.evaluate.constants.utils';

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 * @returns void.
 */
export function runEntropyCompatibilityTuning(
  controller: NeatControllerForEval,
  evaluationOptions: NeatControllerForEval['options'],
): void {
  // Step 1: Adjust compatibility threshold when entropy tuning is enabled.
  try {
    const entropyCompatOptions = evaluationOptions.entropyCompatTuning;
    if (!entropyCompatOptions?.enabled) return;
    const meanEntropy = controller._diversityStats?.meanEntropy;
    if (typeof meanEntropy !== 'number') return;
    const nextThreshold = computeNextCompatibilityThreshold(
      entropyCompatOptions,
      meanEntropy,
      controller.options.compatibilityThreshold ?? COMPAT_THRESHOLD_DEFAULT,
    );
    controller.options.compatibilityThreshold = nextThreshold;
  } catch {
    // Intentionally ignore entropy-compatibility tuning errors
  }
}

/**
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
  // Step 1: Prepare tuning constants.
  const targetEntropy =
    entropyCompatOptions.targetEntropy ?? ENTROPY_TARGET_DEFAULT;
  const deadband = entropyCompatOptions.deadband ?? ENTROPY_DEADBAND_DEFAULT;
  const adjustRate = entropyCompatOptions.adjustRate ?? ENTROPY_ADJUST_DEFAULT;
  const minThreshold =
    entropyCompatOptions.minThreshold ?? COMPAT_MIN_THRESHOLD_DEFAULT;
  const maxThreshold =
    entropyCompatOptions.maxThreshold ?? COMPAT_MAX_THRESHOLD_DEFAULT;
  // Step 2: Adjust within the deadband.
  if (meanEntropy < targetEntropy - deadband) {
    return Math.max(minThreshold, currentThreshold * (1 - adjustRate));
  }
  if (meanEntropy > targetEntropy + deadband) {
    return Math.min(maxThreshold, currentThreshold * (1 + adjustRate));
  }
  return currentThreshold;
}
