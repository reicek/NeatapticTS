/**
 * Timestep resolution executors for the Neatenstein tick pipeline.
 *
 * @module
 */

import { NEATENSTEIN_FIXED_TIMESTEP_MS } from './constants';

/**
 * Resolve a safe tick duration.
 *
 * The simulation is designed around a fixed positive timestep. If an invalid
 * value reaches this boundary, falling back to the canonical fixed timestep is
 * safer than propagating `NaN`, infinities, or negative time into subsystems.
 *
 * @param dtMs - Candidate tick duration in milliseconds.
 * @returns Positive finite tick duration.
 */
export function resolveTickDurationMs(dtMs: number): number {
  return Number.isFinite(dtMs) && dtMs > 0
    ? dtMs
    : NEATENSTEIN_FIXED_TIMESTEP_MS;
}
