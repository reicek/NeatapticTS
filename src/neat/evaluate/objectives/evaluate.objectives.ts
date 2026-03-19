import type { NeatControllerForEval } from '../shared/evaluate.types';

/**
 * Objective-registration helpers for the NEAT evaluate chapter.
 *
 * This chapter owns the narrow policy that automatically registers an entropy
 * objective during evaluation when multi-objective mode is active and the
 * controller requests auto-entropy behavior.
 *
 * The boundary is intentionally policy-sized rather than subsystem-sized.
 * Evaluation does not try to own all objective management here. Instead it
 * answers one narrow question: when the controller is already running in
 * multi-objective mode, should this pass ensure that structural entropy is one
 * of the active objectives?
 *
 * Read this chapter when you want to answer questions such as:
 * - Why does auto-entropy registration happen during evaluation instead of in
 *   constructor setup or the broader objectives chapter?
 * - What conditions must be true before entropy is injected automatically?
 * - Why does this helper invalidate the cached objective list after
 *   registration?
 * - Which controller assumptions stay stable for later ranking and selection
 *   reads?
 *
 * The mental model is a three-step policy gate:
 * 1. check whether multi-objective mode and auto-entropy are both active,
 * 2. skip if entropy is already present or dynamic-objective mode owns the
 *   policy,
 * 3. register entropy and invalidate cached objective resolution.
 *
 * This boundary preserves the current scores, novelty evidence, and species
 * state. It only updates the objective vocabulary that later ranking stages may
 * consume.
 *
 * ```mermaid
 * flowchart TD
 *   Evaluate[Evaluation pass reaches auto-objective stage] --> Gate[Check multi-objective and auto-entropy flags]
 *   Gate --> Skip[Skip when disabled or dynamic mode owns objectives]
 *   Gate --> Existing[Check whether entropy objective already exists]
 *   Existing --> Register[Register entropy objective]
 *   Register --> Invalidate[Invalidate cached objective list]
 *   Invalidate --> Next[Later ranking reads updated objective set]
 * ```
 */

/**
 * Inject the entropy objective when multi-objective evaluation requests it.
 *
 * This is the controller-facing entrypoint for the evaluate-objectives stage.
 * It behaves like best-effort policy maintenance after the rest of the
 * evaluation evidence has been gathered.
 *
 * The helper preserves several important controller assumptions:
 * - current scores are left intact,
 * - novelty and diversity evidence are left intact,
 * - current species state is left intact,
 * - only the objective-registration layer and its cache are updated.
 *
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 *
 * @example
 * ```ts
 * runAutoEntropyObjectiveInjection(controller, controller.options);
 * console.log(controller._getObjectives?.().map((objective) => objective.key));
 * ```
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
 * Automatic injection is intentionally conservative. Dynamic-objective mode is
 * treated as the stronger owner when it is enabled, which prevents this helper
 * from silently competing with a more explicit objective-management policy.
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
 * Registering the objective is only half of the job. The cached objective list
 * must also be invalidated so later ranking stages resolve the updated
 * objective set instead of continuing to use stale ordering information.
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
