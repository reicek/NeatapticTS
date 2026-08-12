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

import type { EnemyBehaviorMetrics } from './types';

/**
 * One generation snapshot accepted by {@link computeAdaptationSignal}.
 *
 * Pairs a generation number with the enemy behavior metrics observed during
 * that generation so the adaptation signal can diff consecutive generations.
 */
export interface GenerationSnapshot {
  /** Generation number. */
  generation: number;
  /** Enemy behavior metrics observed during this generation. */
  enemyBehaviorMetrics: EnemyBehaviorMetrics;
}

/**
 * Adaptation signal emitted by {@link computeAdaptationSignal}.
 *
 * Summarises the behavioral delta between two consecutive generations into a
 * coarse `direction` label plus the raw numeric deltas for each behavior axis.
 */
export interface AdaptationSignal {
  /** Coarse direction label: `'stronger'`, `'weaker'`, or `'shifted'`. */
  direction: 'stronger' | 'weaker' | 'shifted';
  /** Change in enemy aggression between the two generations. */
  aggressionDelta: number;
  /** Change in enemy movement pattern between the two generations. */
  movementDelta: number;
  /** Change in enemy positioning between the two generations. */
  positioningDelta: number;
}

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
    direction = 'stronger';
  } else if (aggressionDelta < -AGGRESSION_DELTA_THRESHOLD) {
    direction = 'weaker';
  } else {
    direction = 'shifted';
  }

  return { direction, aggressionDelta, movementDelta, positioningDelta };
}
