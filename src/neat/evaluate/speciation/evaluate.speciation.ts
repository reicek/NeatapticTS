import type { NeatControllerForEval } from '../shared/evaluate.types';

/**
 * Lightweight speciation helpers for the NEAT evaluate chapter.
 *
 * This chapter keeps the post-evaluation speciation trigger small: only run the
 * controller's speciation maintenance hook when evaluation-time features such
 * as target-species tuning or extended species history actually need it.
 *
 * The boundary is intentionally thin. Evaluation does not take over the full
 * species-assignment lifecycle here; the main speciation chapter still owns
 * the heavy lifting. This helper exists because some evaluation-time policies
 * depend on refreshed species state immediately after new evidence lands.
 *
 * Read this chapter when you want to answer questions such as:
 * - Why can evaluation trigger speciation at all if speciation already has its
 *   own chapter?
 * - Which option flags are strong enough to justify a lightweight post-score
 *   speciation refresh?
 * - Why does this boundary only trigger the hook instead of duplicating the
 *   broader species-maintenance logic?
 * - Which controller assumptions stay stable for later selection and objective
 *   reads?
 *
 * The mental model is a two-step gate:
 * 1. check whether evaluation-time features currently depend on refreshed
 *    species state,
 * 2. run the controller's existing speciation hook when they do.
 *
 * This chapter preserves score evidence, novelty annotations, and objective
 * policy. It only decides whether the existing speciation subsystem should be
 * asked to refresh after evaluation.
 *
 * ```mermaid
 * flowchart TD
 *   Evaluate[Evaluation has produced fresh evidence] --> Gate[Check speciation flag and dependent features]
 *   Gate --> Skip[Skip when no evaluation-time speciation feature is active]
 *   Gate --> Trigger[Run controller's existing _speciate hook]
 *   Trigger --> Next[Later reads see refreshed species state]
 * ```
 */

/**
 * Run lightweight post-evaluation speciation when related controller features are enabled.
 *
 * This is the controller-facing entrypoint for the evaluate-speciation stage.
 * It behaves like best-effort maintenance: evaluation may request a refresh of
 * species state when nearby tuning or history features rely on it, but failure
 * here should not invalidate the newly gathered score evidence.
 *
 * The helper preserves several important controller assumptions:
 * - current score and novelty evidence are left intact,
 * - objective policy is left intact,
 * - only the existing speciation hook is triggered when needed.
 *
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 *
 * @example
 * ```ts
 * runLightweightSpeciation(controller, controller.options);
 * ```
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
 * The trigger is deliberately narrow. Evaluation asks for a speciation refresh
 * only when a feature such as target-species tuning, compatibility adjustment,
 * or extended species history would otherwise be reasoning from stale species
 * state.
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
