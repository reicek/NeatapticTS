/**
 * Death feedback loop for the Neatenstein asymmetric co-evolution harness.
 *
 * Computes an adaptation signal from consecutive generation snapshots so the
 * harness can communicate whether the enemy population is getting stronger,
 * weaker, or shifting strategy between generations. The signal is consumed by
 * the host HUD to give the player a visual read on co-evolution pressure.
 *
 * @module
 */

import type { GenerationSnapshot, AdaptationSignal } from './types';
import {
  ADAPTATION_STRONGER,
  ADAPTATION_WEAKER,
  ADAPTATION_SHIFTED,
} from '../constants';

/**
 * One generation snapshot accepted by {@link computeAdaptationSignal}.
 *
 * @deprecated Import from `./types` instead. This re-export preserves the
 *   public API for existing consumers.
 */
export type { GenerationSnapshot } from './types';

/**
 * Adaptation signal emitted by {@link computeAdaptationSignal}.
 *
 * @deprecated Import from `./types` instead. This re-export preserves the
 *   public API for existing consumers.
 */
export type { AdaptationSignal } from './types';

/**
 * Magnitude threshold above which an aggression delta is classified as
 * `'stronger'` or `'weaker'` rather than `'shifted'`.
 *
 * Deltas within `[-THRESHOLD, +THRESHOLD]` are considered too small to be a
 * clear strength change and fall back to the `'shifted'` label.
 */
const AGGRESSION_DELTA_THRESHOLD = 0.1;

/**
 * Compute the adaptation signal between two consecutive generation snapshots.
 *
 * The function diffs the enemy behavior metrics of `currGeneration` against
 * `prevGeneration` and produces an {@link AdaptationSignal} with:
 * - `direction` — `'stronger'` when aggression rose beyond the threshold,
 *   `'weaker'` when it fell beyond the threshold, or `'shifted'` when the
 *   aggression change is small but other axes moved.
 * - `aggressionDelta`, `movementDelta`, `positioningDelta` — the raw numeric
 *   differences for each behavior axis.
 *
 * @param prevGeneration - Previous generation snapshot.
 * @param currGeneration - Current generation snapshot.
 * @returns The adaptation signal describing the behavioral delta.
 *
 * @example
 * ```ts
 * const signal = computeAdaptationSignal(
 *   { generation: 1, enemyBehaviorMetrics: { aggression: 0.3, movementPattern: 0.5, positioning: 0.2 } },
 *   { generation: 2, enemyBehaviorMetrics: { aggression: 0.7, movementPattern: 0.4, positioning: 0.6 } },
 * );
 * console.log(signal.direction);        // 'stronger'
 * console.log(signal.aggressionDelta);  // 0.4
 * ```
 */
export function computeAdaptationSignal(
  prevGeneration: GenerationSnapshot,
  currGeneration: GenerationSnapshot,
): AdaptationSignal {
  const aggressionDelta =
    currGeneration.enemyBehaviorMetrics.aggression -
    prevGeneration.enemyBehaviorMetrics.aggression;
  const movementDelta =
    currGeneration.enemyBehaviorMetrics.movementPattern -
    prevGeneration.enemyBehaviorMetrics.movementPattern;
  const positioningDelta =
    currGeneration.enemyBehaviorMetrics.positioning -
    prevGeneration.enemyBehaviorMetrics.positioning;

  let direction: 'stronger' | 'weaker' | 'shifted';
  if (aggressionDelta > AGGRESSION_DELTA_THRESHOLD) {
    direction = ADAPTATION_STRONGER;
  } else if (aggressionDelta < -AGGRESSION_DELTA_THRESHOLD) {
    direction = ADAPTATION_WEAKER;
  } else {
    direction = ADAPTATION_SHIFTED;
  }

  return { direction, aggressionDelta, movementDelta, positioningDelta };
}
