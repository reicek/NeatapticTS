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
 *
 * The boundary stays intentionally narrow. By the time these helpers run, the
 * evaluation stage has already produced fresh scores and diversity evidence.
 * Entropy-sharing tuning does not rescore genomes, rebuild species, or widen
 * into another compatibility-control loop. Its only job is to read the newly
 * observed entropy spread and decide whether the next generation should apply
 * stronger or weaker sharing pressure.
 *
 * Read this chapter when you want to answer questions such as:
 * - Why does evaluation keep `sharingSigma` adaptation separate from the rest
 *   of the scoring pipeline?
 * - What evidence is considered trustworthy enough to tune sigma?
 * - How do the variance bands, adjustment rate, and min or max bounds interact?
 * - Which controller assumptions stay stable for later selection and
 *   speciation reads?
 *
 * The mental model is a short three-step loop:
 * 1. make sure diversity-stat storage exists,
 * 2. read the fresh entropy-variance measurement,
 * 3. nudge `sharingSigma` up or down inside its configured bounds.
 *
 * The important preserved assumptions are just as valuable as the update
 * itself. This boundary leaves genome scores, population ordering, species
 * membership, and compatibility-threshold policy untouched so later selection
 * and speciation chapters can still reason from stable post-evaluation state.
 *
 * ```mermaid
 * flowchart TD
 *   Stats[Fresh diversity stats after evaluation] --> Guard[Require enabled tuning and numeric varEntropy]
 *   Guard --> Compare[Compare observed variance with target bands]
 *   Compare --> Low[Too low: reduce sharing sigma]
 *   Compare --> Mid[Within band: keep sigma]
 *   Compare --> High[Too high: increase sharing sigma]
 *   Low --> Clamp[Clamp to configured min or max]
 *   Mid --> Clamp
 *   High --> Clamp
 *   Clamp --> Next[Next evaluation and selection cycle reads updated sigma]
 * ```
 */

/**
 * Ensure diversity statistics storage exists before tuning writes into it.
 *
 * Evaluation uses this as a small guardrail before an adaptive helper writes
 * post-score measurements. The container is created lazily so callers do not
 * need to pre-seed optional diversity state during controller construction.
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
 * This is the controller-facing entrypoint for the entropy-sharing chapter.
 * It runs only after evaluation has produced fresh diversity evidence, and it
 * intentionally behaves like best-effort maintenance rather than a required
 * scoring step.
 *
 * The helper preserves several important controller assumptions:
 * - genome scores are treated as complete and are not recomputed here,
 * - population order is not changed here,
 * - species assignment is not refreshed here,
 * - only `sharingSigma` is updated for future passes.
 *
 * That narrow scope lets later selection and speciation reads consume the same
 * evaluated population while still benefiting from a tuned sharing radius on
 * the next cycle.
 *
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 *
 * @example
 * ```ts
 * ensureDiversityStatsContainer(controller);
 * controller._diversityStats!.varEntropy = 0.18;
 *
 * runEntropySharingTuning(controller, controller.options);
 * console.log(controller.options.sharingSigma);
 * ```
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
 * The tuning rule is intentionally small and band-driven instead of trying to
 * build a full controller inside evaluation. When observed variance drops below
 * the low band, sigma is reduced so the next pass applies a tighter sharing
 * radius. When variance rises above the high band, sigma is increased so the
 * next pass smooths pressure across a wider neighborhood. Values inside the
 * band keep the current sigma unchanged.
 *
 * The returned value is always clamped to the configured minimum and maximum,
 * which keeps tuning predictable even when entropy measurements swing sharply.
 *
 * @param entropySharingOptions - Tuning options that define the target
 * entropy variance, adjustment rate, and clamp bounds.
 * @param currentVarEntropy - Freshly observed variance of structural entropy
 * for the current population.
 * @param currentSigma - Current sharing sigma before this adjustment.
 * @returns Next sharing sigma value to carry into later controller passes.
 *
 * @example
 * ```ts
 * const nextSigma = computeNextSharingSigma(
 *   { enabled: true, targetEntropyVar: 0.2, adjustRate: 0.1, minSigma: 0.5, maxSigma: 3 },
 *   0.28,
 *   1,
 * );
 *
 * console.log(nextSigma);
 * ```
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
