import type { NeatControllerForEval } from './neat.evaluate.utils.types';

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 * @returns void.
 */
export function runLightweightSpeciation(
  controller: NeatControllerForEval,
  evaluationOptions: NeatControllerForEval['options'],
): void {
  // Step 1: Run speciation when controller features are enabled.
  try {
    if (!shouldRunSpeciation(evaluationOptions)) return;
    controller._speciate?.();
  } catch {
    // Intentionally ignore speciation errors during evaluation
  }
}

/**
 * @param evaluationOptions - Options object for the current evaluation pass.
 * @returns Whether speciation should be run.
 */
function shouldRunSpeciation(
  evaluationOptions: NeatControllerForEval['options'],
): boolean {
  // Step 1: Require speciation and at least one related feature enabled.
  if (!evaluationOptions.speciation) return false;
  return Boolean(
    evaluationOptions.targetSpecies ||
    evaluationOptions.compatAdjust ||
    evaluationOptions.speciesAllocation?.extendedHistory,
  );
}
