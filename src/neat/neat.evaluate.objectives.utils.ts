import type { NeatControllerForEval } from './neat.evaluate.utils.types';

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 * @returns void.
 */
export function runAutoEntropyObjectiveInjection(
  controller: NeatControllerForEval,
  evaluationOptions: NeatControllerForEval['options'],
): void {
  // Step 1: Inject entropy objective when requested.
  try {
    if (!shouldAutoInjectEntropy(evaluationOptions)) return;
    const objectiveKeys =
      controller._getObjectives?.()?.map((objective) => objective.key) ?? [];
    if (objectiveKeys.includes('entropy')) return;
    registerEntropyObjective(controller);
  } catch {
    // Intentionally ignore auto-entropy objective injection errors
  }
}

/**
 * @param evaluationOptions - Options object for the current evaluation pass.
 * @returns Whether entropy objective should be injected.
 */
function shouldAutoInjectEntropy(
  evaluationOptions: NeatControllerForEval['options'],
): boolean {
  // Step 1: Require multi-objective and auto-entropy without dynamic override.
  if (!evaluationOptions.multiObjective?.enabled) return false;
  if (!evaluationOptions.multiObjective.autoEntropy) return false;
  return !evaluationOptions.multiObjective.dynamic?.enabled;
}

/**
 * @param controller - NEAT controller instance for evaluation.
 * @returns void.
 */
function registerEntropyObjective(controller: NeatControllerForEval): void {
  // Step 1: Register entropy objective and invalidate cache.
  controller.registerObjective?.(
    'entropy',
    'max',
    (genome) => controller._structuralEntropy?.(genome) ?? 0,
  );
  controller._pendingObjectiveAdds?.push('entropy');
  controller._objectivesList = undefined;
}
