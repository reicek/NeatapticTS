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
 * Initialize MC threshold if missing.
 *
 * @param engine - NEAT engine instance.
 * @param config - Minimal-criterion adaptive configuration.
 * @returns {void}
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
 * @param engine - NEAT engine instance.
 * @returns Array of scores (missing scores treated as 0).
 */
export function collectScores(engine: NeatLikeWithAdaptive): number[] {
  return engine.population.map((genome) => genome.score ?? ZERO);
}

/**
 * Compute acceptance metrics for the current threshold.
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
 * @param engine - NEAT engine instance.
 * @param acceptance - Observed acceptance proportion.
 * @param tuning - Target acceptance and adjustment settings.
 * @returns {void}
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
 * Zero scores below the final threshold.
 *
 * @param engine - NEAT engine instance.
 * @param threshold - Final MC threshold.
 * @returns {void}
 */
export function applyRejection(
  engine: NeatLikeWithAdaptive,
  threshold: number,
): void {
  for (const genome of engine.population) {
    if ((genome.score ?? ZERO) < threshold) genome.score = ZERO;
  }
}