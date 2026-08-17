/**
 * Impact-spot aging executors for the Neatenstein tick pipeline.
 *
 * @module
 */

import { resolveTickDurationMs } from './tick.time.utils';
import type { EnemyImpactSpot, ImpactSpot } from './types';

/**
 * Age active wall-impact spots by one tick and remove any that have expired.
 *
 * Impact spots are immutable snapshots; each surviving impact gets its
 * remaining lifetime reduced by the elapsed timestep.
 *
 * @param impacts - Active wall-impact snapshots before this tick.
 * @param dtMs - Elapsed time in milliseconds.
 * @returns New array of impact spots still visible after aging.
 */
export function ageImpacts(impacts: ImpactSpot[], dtMs: number): ImpactSpot[] {
  const resolvedDtMs = resolveTickDurationMs(dtMs);

  return impacts
    .map((impact) => ({
      ...impact,
      lifetimeMs: impact.lifetimeMs - resolvedDtMs,
    }))
    .filter((impact) => impact.lifetimeMs > 0);
}

/**
 * Age active enemy-impact spots by one tick and remove any that have expired.
 *
 * Follows the same deterministic pattern as {@link ageImpacts}: each surviving
 * spot gets its remaining lifetime reduced by the elapsed timestep (derived
 * from `simTimeMs`, never `Date.now()`), and spots with lifetime ≤ 0 are
 * removed.
 *
 * @param impacts - Active enemy-impact snapshots before this tick.
 * @param dtMs - Elapsed time in milliseconds.
 * @returns New array of enemy-impact spots still visible after aging.
 */
export function ageEnemyImpacts(
  impacts: EnemyImpactSpot[],
  dtMs: number,
): EnemyImpactSpot[] {
  const resolvedDtMs = resolveTickDurationMs(dtMs);

  return impacts
    .map((impact) => ({
      ...impact,
      lifetimeMs: impact.lifetimeMs - resolvedDtMs,
    }))
    .filter((impact) => impact.lifetimeMs > 0);
}
