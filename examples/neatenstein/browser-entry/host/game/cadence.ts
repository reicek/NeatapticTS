/**
 * Generation cadence helpers for the Neatenstein evolution harness.
 *
 * These utilities translate episode length and evaluator overhead into a
 * generations-per-minute estimate. They are intentionally lightweight: the real
 * harness can measure wall-clock time, but this module provides deterministic
 * planning helpers for validating cadence targets during design and tests.
 *
 * @module
 */

import { NEATENSTEIN_MIN_GENERATIONS_PER_MINUTE } from './constants';
import type { EstimateCadenceOptions } from '../types';

// Re-export consolidated types so existing imports from this module remain valid.
export type { EstimateCadenceOptions } from '../types';

/**
 * Milliseconds in one minute.
 *
 * Named to avoid scattering the `60_000` conversion constant through cadence
 * formulas.
 */
const MILLISECONDS_PER_MINUTE = 60_000;

/**
 * Minimum valid cycle duration.
 *
 * A generation with zero or negative total duration is physically impossible,
 * so invalid or non-positive cycles produce a cadence estimate of `0`.
 */
const MIN_GENERATION_CYCLE_MS = 0;

/**
 * Target minimum number of evolutionary generations per minute.
 *
 * Re-exported from the canonical constants module so callers can import both
 * the target and cadence helpers from this module.
 */
export const NEATENSTEIN_TARGET_MIN_GENERATIONS_PER_MINUTE =
  NEATENSTEIN_MIN_GENERATIONS_PER_MINUTE;

/** Inputs accepted by {@link estimateGenerationsPerMinute}. */
// Type is defined in ../types and re-exported above.

/**
 * Return whether a duration can be used in cadence calculations.
 *
 * @param value - Candidate duration in milliseconds.
 * @returns Whether the duration is finite and non-negative.
 */
function isValidDurationMs(value: number): boolean {
  return Number.isFinite(value) && value >= 0;
}

/**
 * Resolve the total estimated cycle duration for one generation.
 *
 * @param options - Episode duration and evaluator overhead.
 * @returns Total cycle duration in milliseconds, or `null` when invalid.
 */
function resolveGenerationCycleMs(
  options: EstimateCadenceOptions,
): number | null {
  if (
    !isValidDurationMs(options.episodeDurationMs) ||
    !isValidDurationMs(options.evaluationOverheadMs)
  ) {
    return null;
  }

  const totalCycleMs = options.episodeDurationMs + options.evaluationOverheadMs;

  if (
    !Number.isFinite(totalCycleMs) ||
    totalCycleMs <= MIN_GENERATION_CYCLE_MS
  ) {
    return null;
  }

  return totalCycleMs;
}

/**
 * Estimate how many generations the harness can run per minute.
 *
 * The estimate treats each generation as one episode plus fixed evaluator
 * overhead, then converts the total cycle time into a per-minute rate.
 *
 * Invalid inputs return `0` rather than throwing, which keeps this helper safe
 * for dashboards, tests, and design probes.
 *
 * @param options - Episode duration and evaluator overhead.
 * @returns Estimated generations per minute.
 *
 * @example
 * ```ts
 * const rate = estimateGenerationsPerMinute({
 *   episodeDurationMs: 20_000,
 *   evaluationOverheadMs: 2_000,
 * });
 *
 * expect(rate).toBeGreaterThanOrEqual(
 *   NEATENSTEIN_TARGET_MIN_GENERATIONS_PER_MINUTE,
 * );
 * ```
 */
export function estimateGenerationsPerMinute(
  options: EstimateCadenceOptions,
): number {
  const totalCycleMs = resolveGenerationCycleMs(options);

  if (totalCycleMs === null) {
    return 0;
  }

  return MILLISECONDS_PER_MINUTE / totalCycleMs;
}

/**
 * Return whether the supplied cadence estimate satisfies the configured
 * Neatenstein minimum generation rate.
 *
 * @param options - Episode duration and evaluator overhead.
 * @returns Whether the estimated cadence meets the minimum target.
 *
 * @example
 * ```ts
 * const ok = meetsGenerationCadenceTarget({
 *   episodeDurationMs: 20_000,
 *   evaluationOverheadMs: 2_000,
 * });
 * ```
 */
export function meetsGenerationCadenceTarget(
  options: EstimateCadenceOptions,
): boolean {
  return (
    estimateGenerationsPerMinute(options) >=
    NEATENSTEIN_TARGET_MIN_GENERATIONS_PER_MINUTE
  );
}
