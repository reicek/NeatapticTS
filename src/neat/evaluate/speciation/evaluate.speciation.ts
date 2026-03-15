import type { NeatControllerForEval } from '../shared/evaluate.types';

/**
 * Lightweight speciation helpers for the NEAT evaluate chapter.
 *
 * This chapter keeps the post-evaluation speciation trigger small: only run the
 * controller's speciation maintenance hook when evaluation-time features such
 * as target-species tuning or extended species history actually need it.
 */

/**
 * Run lightweight post-evaluation speciation when related controller features are enabled.
 *
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 */
export function runLightweightSpeciation(
  controller: NeatControllerForEval,
  evaluationOptions: NeatControllerForEval['options'],
): void {
  // Step 1: Run speciation only when evaluation-time features require it.
  try {
    if (!shouldRunSpeciation(evaluationOptions)) return;
    controller._speciate?.();
  } catch {
    // Speciation maintenance is best-effort during evaluation.
  }
}

/**
 * Check whether the evaluation pass should trigger the speciation hook.
 *
 * @param evaluationOptions - Options object for the current evaluation pass.
 * @returns True when speciation should run.
 */
function shouldRunSpeciation(
  evaluationOptions: NeatControllerForEval['options'],
): boolean {
  if (!evaluationOptions.speciation) return false;

  return Boolean(
    evaluationOptions.targetSpecies ||
    evaluationOptions.compatAdjust ||
    evaluationOptions.speciesAllocation?.extendedHistory,
  );
}
