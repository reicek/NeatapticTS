import type { NeatControllerForEval } from './neat.evaluate.utils.types';
import {
  ENTROPY_VAR_ADJUST_DEFAULT,
  ENTROPY_VAR_HIGH_BAND,
  ENTROPY_VAR_LOW_BAND,
  ENTROPY_VAR_MAX_SIGMA_DEFAULT,
  ENTROPY_VAR_MIN_SIGMA_DEFAULT,
  ENTROPY_VAR_TARGET_DEFAULT,
} from './neat.evaluate.constants.utils';

/**
 * @param controller - NEAT controller instance for evaluation.
 * @returns void.
 */
export function ensureDiversityStatsContainer(
  controller: NeatControllerForEval,
): void {
  // Step 1: Ensure a diversity stats container exists for tuning.
  if (!controller._diversityStats) controller._diversityStats = {};
}

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 * @returns void.
 */
export function runEntropySharingTuning(
  controller: NeatControllerForEval,
  evaluationOptions: NeatControllerForEval['options'],
): void {
  // Step 1: Adjust sharing sigma when entropy sharing tuning is enabled.
  try {
    const entropySharingOptions = evaluationOptions.entropySharingTuning;
    if (!entropySharingOptions?.enabled) return;
    const currentVarEntropy = controller._diversityStats?.varEntropy;
    if (typeof currentVarEntropy !== 'number') return;
    const nextSigma = computeNextSharingSigma(
      entropySharingOptions,
      currentVarEntropy,
      controller.options.sharingSigma ?? 0,
    );
    controller.options.sharingSigma = nextSigma;
  } catch {
    // Intentionally ignore entropy sharing tuning errors
  }
}

/**
 * @param entropySharingOptions - Tuning options.
 * @param currentVarEntropy - Current variance of entropy.
 * @param currentSigma - Current sigma value.
 * @returns Next sigma value.
 */
function computeNextSharingSigma(
  entropySharingOptions: NonNullable<
    NeatControllerForEval['options']['entropySharingTuning']
  >,
  currentVarEntropy: number,
  currentSigma: number,
): number {
  // Step 1: Prepare tuning constants.
  const targetVar =
    entropySharingOptions.targetEntropyVar ?? ENTROPY_VAR_TARGET_DEFAULT;
  const adjustRate =
    entropySharingOptions.adjustRate ?? ENTROPY_VAR_ADJUST_DEFAULT;
  const minSigma =
    entropySharingOptions.minSigma ?? ENTROPY_VAR_MIN_SIGMA_DEFAULT;
  const maxSigma =
    entropySharingOptions.maxSigma ?? ENTROPY_VAR_MAX_SIGMA_DEFAULT;
  // Step 2: Adjust sigma based on variance band.
  if (currentVarEntropy < targetVar * ENTROPY_VAR_LOW_BAND) {
    return Math.max(minSigma, currentSigma * (1 - adjustRate));
  }
  if (currentVarEntropy > targetVar * ENTROPY_VAR_HIGH_BAND) {
    return Math.min(maxSigma, currentSigma * (1 + adjustRate));
  }
  return currentSigma;
}
