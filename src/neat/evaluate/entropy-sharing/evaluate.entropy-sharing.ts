import type { NeatControllerForEval } from '../shared/evaluate.types';
import {
  ENTROPY_VAR_ADJUST_DEFAULT,
  ENTROPY_VAR_HIGH_BAND,
  ENTROPY_VAR_LOW_BAND,
  ENTROPY_VAR_MAX_SIGMA_DEFAULT,
  ENTROPY_VAR_MIN_SIGMA_DEFAULT,
  ENTROPY_VAR_TARGET_DEFAULT,
} from '../shared/evaluate.constants';

/**
 * Entropy-sharing tuning helpers for the NEAT evaluate chapter.
 *
 * This chapter owns the small post-evaluation adjustment that nudges
 * `sharingSigma` based on the observed variance of structural entropy across
 * the current population.
 */

/**
 * Ensure diversity statistics storage exists before tuning writes into it.
 *
 * @param controller - NEAT controller instance for evaluation.
 */
export function ensureDiversityStatsContainer(
  controller: NeatControllerForEval,
): void {
  if (!controller._diversityStats) controller._diversityStats = {};
}

/**
 * Adjust the entropy-sharing sigma when entropy sharing tuning is enabled.
 *
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 */
export function runEntropySharingTuning(
  controller: NeatControllerForEval,
  evaluationOptions: NeatControllerForEval['options'],
): void {
  // Step 1: Read the current entropy variance and update sharing sigma safely.
  try {
    const entropySharingOptions = evaluationOptions.entropySharingTuning;
    if (!entropySharingOptions?.enabled) return;

    const currentVarEntropy = controller._diversityStats?.varEntropy;
    if (typeof currentVarEntropy !== 'number') return;

    controller.options.sharingSigma = computeNextSharingSigma(
      entropySharingOptions,
      currentVarEntropy,
      controller.options.sharingSigma ?? 0,
    );
  } catch {
    // Entropy-sharing tuning is opportunistic and should not fail evaluation.
  }
}

/**
 * Compute the next sharing sigma value from the observed entropy variance.
 *
 * @param entropySharingOptions - Tuning options.
 * @param currentVarEntropy - Current variance of entropy.
 * @param currentSigma - Current sigma value.
 * @returns Next sharing sigma value.
 */
function computeNextSharingSigma(
  entropySharingOptions: NonNullable<
    NeatControllerForEval['options']['entropySharingTuning']
  >,
  currentVarEntropy: number,
  currentSigma: number,
): number {
  const targetVar =
    entropySharingOptions.targetEntropyVar ?? ENTROPY_VAR_TARGET_DEFAULT;
  const adjustRate =
    entropySharingOptions.adjustRate ?? ENTROPY_VAR_ADJUST_DEFAULT;
  const minSigma =
    entropySharingOptions.minSigma ?? ENTROPY_VAR_MIN_SIGMA_DEFAULT;
  const maxSigma =
    entropySharingOptions.maxSigma ?? ENTROPY_VAR_MAX_SIGMA_DEFAULT;

  if (currentVarEntropy < targetVar * ENTROPY_VAR_LOW_BAND) {
    return Math.max(minSigma, currentSigma * (1 - adjustRate));
  }
  if (currentVarEntropy > targetVar * ENTROPY_VAR_HIGH_BAND) {
    return Math.min(maxSigma, currentSigma * (1 + adjustRate));
  }

  return currentSigma;
}
