import {
  ACCEPTANCE_LOWER_MULTIPLIER,
  ACCEPTANCE_UPPER_MULTIPLIER,
  ADJUST_RATE_DEFAULT,
  TARGET_ACCEPTANCE_DEFAULT,
  ONE,
  ZERO,
} from '../core/adaptive.core.constants';
import type {
  MinimalCriterionAdaptiveConfig,
  NeatLikeWithAdaptive,
} from '../core/adaptive.core.types';

/**
 * Minimal-criterion helpers for adaptive acceptance.
 *
 * This file owns the small evidence-to-threshold loop behind adaptive
 * acceptance. It stays separate from the root adaptive controller entrypoint so
 * the generated chapter can explain acceptance pressure as a readable pipeline
 * instead of burying the mechanics inside one long method.
 *
 * The helper flow is intentionally compact:
 *
 * 1. ensure there is a starting threshold,
 * 2. snapshot current scores,
 * 3. measure how much of the population clears the bar,
 * 4. retune the threshold and reject genomes that still miss it.
 *
 * That last step is deliberately a controller-policy overlay. Rejection may
 * rewrite the current generation's `score` field so later selection can treat
 * weak genomes as filtered out, but it does not redefine the raw evaluation
 * evidence or the canonical genome contract.
 */

/* Module introduction boundary for generated README output. */

/**
 * Initialize MC threshold if missing.
 *
 * Threshold initialization is lazy because many runs never enable adaptive
 * acceptance at all. The first invocation seeds the long-lived threshold, and
 * later generations reuse the updated value rather than restarting from the
 * original configuration each time.
 *
 * @param engine - NEAT engine instance.
 * @param config - Minimal-criterion adaptive configuration.
 * @returns Nothing.
 */
export function initializeThreshold(
  engine: NeatLikeWithAdaptive,
  config: MinimalCriterionAdaptiveConfig,
): void {
  if (engine._mcThreshold !== undefined) return;

  const initialThreshold = config.initialThreshold ?? ZERO;
  engine._mcThreshold = initialThreshold;
}

/**
 * Collect population scores into a snapshot array.
 *
 * The acceptance controller works from one stable view of the current
 * generation rather than mixing threshold updates with in-place rejection while
 * it is still counting. Missing scores are treated as zero so unevaluated or
 * explicitly rejected genomes remain part of the acceptance picture.
 *
 * @param engine - NEAT engine instance.
 * @returns Array of scores (missing scores treated as 0).
 */
export function collectScores(engine: NeatLikeWithAdaptive): number[] {
  return engine.population.map((genome) => genome.score ?? ZERO);
}

/**
 * Compute acceptance metrics for the current threshold.
 *
 * Acceptance is deliberately reduced to one proportion: how much of the current
 * generation still clears the bar. That single number is enough for the caller
 * to decide whether the threshold is too lenient, too strict, or already close
 * enough to the configured target band.
 *
 * @param scores - Population score snapshot.
 * @param threshold - Current MC threshold.
 * @returns Acceptance proportion.
 */
export function computeAcceptance(scores: number[], threshold: number): number {
  if (!scores.length) return ZERO;

  const acceptedCount = scores.filter((score) => score >= threshold).length;
  return acceptedCount / scores.length;
}

/**
 * Resolve target acceptance and adjust rate settings.
 *
 * This helper centralizes the defaulting rules so later threshold-updating code
 * can focus on policy instead of configuration fallback noise.
 *
 * @param config - Minimal-criterion adaptive configuration.
 * @returns Target settings.
 */
export function resolveTargetSettings(config: MinimalCriterionAdaptiveConfig): {
  targetAcceptance: number;
  adjustRate: number;
} {
  const targetAcceptance = config.targetAcceptance ?? TARGET_ACCEPTANCE_DEFAULT;
  const adjustRate = config.adjustRate ?? ADJUST_RATE_DEFAULT;
  return { targetAcceptance, adjustRate };
}

/**
 * Update the MC threshold based on acceptance proportion.
 *
 * The threshold only moves when observed acceptance drifts outside the target
 * band. Staying inside the band is treated as success, so the current threshold
 * persists and the controller avoids oscillating every generation.
 *
 * @param engine - NEAT engine instance.
 * @param acceptance - Observed acceptance proportion.
 * @param tuning - Target acceptance and adjustment settings.
 * @returns Nothing.
 */
export function updateThreshold(
  engine: NeatLikeWithAdaptive,
  acceptance: number,
  tuning: { targetAcceptance: number; adjustRate: number },
): void {
  const upperBound = tuning.targetAcceptance * ACCEPTANCE_UPPER_MULTIPLIER;
  const lowerBound = tuning.targetAcceptance * ACCEPTANCE_LOWER_MULTIPLIER;

  if (acceptance > upperBound) {
    engine._mcThreshold =
      (engine._mcThreshold ?? ZERO) * (ONE + tuning.adjustRate);
    return;
  }

  if (acceptance < lowerBound) {
    engine._mcThreshold =
      (engine._mcThreshold ?? ZERO) * (ONE - tuning.adjustRate);
  }
}

/**
 * Rewrite controller-visible scores below the final threshold.
 *
 * Rejection is the acceptance chapter's most direct intervention. Instead of
 * queuing a future policy change, it rewrites the current generation's scores so
 * the same selection pass immediately treats low-performing genomes as filtered
 * out. Treat that write as current-generation controller state, not as a claim
 * that the task evaluator itself literally returned zero.
 *
 * Example:
 *
 * ```ts
 * const scores = collectScores(engine);
 * const acceptance = computeAcceptance(scores, engine._mcThreshold ?? 0);
 * updateThreshold(engine, acceptance, resolveTargetSettings(config));
 * applyRejection(engine, engine._mcThreshold ?? 0);
 * ```
 *
 * @param engine - NEAT engine instance.
 * @param threshold - Final MC threshold.
 * @returns Nothing.
 */
export function applyRejection(
  engine: NeatLikeWithAdaptive,
  threshold: number,
): void {
  for (const genome of engine.population) {
    if ((genome.score ?? ZERO) < threshold) genome.score = ZERO;
  }
}
