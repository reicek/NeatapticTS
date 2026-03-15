import type { NeatControllerForEval } from '../shared/evaluate.types';

/**
 * Objective-registration helpers for the NEAT evaluate chapter.
 *
 * This chapter owns the narrow policy that automatically registers an entropy
 * objective during evaluation when multi-objective mode is active and the
 * controller requests auto-entropy behavior.
 */

/**
 * Inject the entropy objective when multi-objective evaluation requests it.
 *
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 */
export function runAutoEntropyObjectiveInjection(
  controller: NeatControllerForEval,
  evaluationOptions: NeatControllerForEval['options'],
): void {
  // Step 1: Register entropy only when the auto-entropy policy is active.
  try {
    if (!shouldAutoInjectEntropy(evaluationOptions)) return;

    const objectiveKeys =
      controller._getObjectives?.()?.map((objective) => objective.key) ?? [];
    if (objectiveKeys.includes('entropy')) return;

    registerEntropyObjective(controller);
  } catch {
    // Auto-objective injection is optional and should not fail evaluation.
  }
}

/**
 * Check whether the evaluation pass should auto-register the entropy objective.
 *
 * @param evaluationOptions - Options object for the current evaluation pass.
 * @returns True when entropy should be injected.
 */
function shouldAutoInjectEntropy(
  evaluationOptions: NeatControllerForEval['options'],
): boolean {
  if (!evaluationOptions.multiObjective?.enabled) return false;
  if (!evaluationOptions.multiObjective.autoEntropy) return false;
  return !evaluationOptions.multiObjective.dynamic?.enabled;
}

/**
 * Register the entropy objective and invalidate the cached objective list.
 *
 * @param controller - NEAT controller instance for evaluation.
 */
function registerEntropyObjective(controller: NeatControllerForEval): void {
  controller.registerObjective?.(
    'entropy',
    'max',
    (genome) => controller._structuralEntropy?.(genome) ?? 0,
  );
  controller._pendingObjectiveAdds?.push('entropy');
  controller._objectivesList = undefined;
}
