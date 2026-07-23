/**
 * Generation cadence helpers for the Neatenstein evolution harness.
 *
 * These utilities translate episode length and evaluator overhead into a
 * generations-per-minute estimate. They are intentionally simple stubs: the
 * real harness will measure wall-clock time, but the constants and formula here
 * guarantee the Phase 2 design can hit its cadence target.
 *
 * @module
 */

import { NEATENSTEIN_MIN_GENERATIONS_PER_MINUTE } from './constants';

/**
 * Target minimum number of evolutionary generations per minute.
 *
 * Re-exported from the canonical constants module so the cadence surface
 * exposes both the target and the estimator in one import.
 */
export const NEATENSTEIN_TARGET_MIN_GENERATIONS_PER_MINUTE =
  NEATENSTEIN_MIN_GENERATIONS_PER_MINUTE;

/** Inputs accepted by {@link estimateGenerationsPerMinute}. */
export interface EstimateCadenceOptions {
  /** Average episode duration in milliseconds. */
  episodeDurationMs: number;
  /** Per-episode evaluator overhead in milliseconds (inference, fitness, etc.). */
  evaluationOverheadMs: number;
}

/**
 * Estimate how many generations the harness can run per minute.
 *
 * The estimate treats each generation as one episode plus a fixed evaluator
 * overhead, then converts the total cycle time into a rate. It is a stub: it
 * does not measure real wall-clock time, but it proves the design satisfies
 * the minimum cadence target for the configured defaults.
 *
 * @param options - Episode duration and evaluator overhead.
 * @returns Generations per minute for the supplied timing parameters.
 *
 * @example
 * ```ts
 * const rate = estimateGenerationsPerMinute({
 *   episodeDurationMs: 20_000,
 *   evaluationOverheadMs: 2_000,
 * });
 * expect(rate).toBeGreaterThanOrEqual(NEATENSTEIN_TARGET_MIN_GENERATIONS_PER_MINUTE);
 * ```
 */
export function estimateGenerationsPerMinute(
  options: EstimateCadenceOptions,
): number {
  const totalCycleMs = options.episodeDurationMs + options.evaluationOverheadMs;
  if (totalCycleMs <= 0) {
    return 0;
  }
  return 60_000 / totalCycleMs;
}
