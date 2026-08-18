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

import type {
  GenerationSnapshot,
  AdaptationSignal,
  EnemyBehaviorMetrics,
} from './types';
import {
  ADAPTATION_STRONGER,
  ADAPTATION_WEAKER,
  ADAPTATION_SHIFTED,
} from '../constants';
import { clamp01 } from '../shared/math-guards.utils';

/**
 * One generation snapshot accepted by {@link computeAdaptationSignal}.
 *
 */
export type { GenerationSnapshot } from './types';

/**
 * Adaptation signal emitted by {@link computeAdaptationSignal}.
 *
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

// ---------------------------------------------------------------------------
// Real telemetry behavior metrics (AC-A4-003)
// ---------------------------------------------------------------------------

/**
 * Raw gameplay telemetry used to compute normalized behavior metrics.
 *
 * Each field is measured directly from the simulation during an enemy's
 * lifetime, not drawn from an RNG. The normalized descriptors are used by
 * MAP-Elites grid indexing and the adaptation-signal diff.
 */
export interface EnemyBehaviorTelemetry {
  /** Total damage dealt to the player during this variant's lifetime. */
  damageDealt: number;
  /** Total ticks the variant survived before death. */
  survivalTicks: number;
  /** Mean distance to the player across the variant's lifetime. */
  meanDistanceToPlayer: number;
  /** Number of direction changes during the variant's lifetime. */
  dirChangeCount: number;
  /** Maximum possible map distance (used to normalize positioning). */
  maxMapDistance: number;
}

/**
 * Survival-tick scalar used in the aggression denominator.
 *
 * The formula `damageDealt / (damageDealt + survivalTicks * 0.1)` balances
 * raw damage output against how long the enemy survived, so a long-lived
 * enemy that dealt little damage is not classified as highly aggressive.
 */
const AGGRESSION_SURVIVAL_SCALAR = 0.1;

/**
 * Compute normalized enemy behavior metrics from real gameplay telemetry.
 *
 * Replaces the previous RNG-drawn `EnemyBehaviorMetrics` with descriptors
 * measured from actual gameplay events:
 * - `aggression` — `clamp01(damageDealt / (damageDealt + survivalTicks * 0.1))`
 * - `positioning` — `clamp01(meanDistanceToPlayer / maxMapDistance)`
 * - `movementPattern` — `clamp01(dirChangeCount / max(1, survivalTicks))`
 *
 * All three descriptors are bounded to [0, 1] for MAP-Elites grid indexing.
 *
 * @param telemetry - Raw gameplay telemetry measured during an enemy's
 *   lifetime.
 * @returns Normalized behavior metrics with each field in [0, 1].
 *
 * @example
 * ```ts
 * const metrics = computeEnemyBehaviorMetrics({
 *   damageDealt: 10,
 *   survivalTicks: 50,
 *   meanDistanceToPlayer: 5,
 *   dirChangeCount: 3,
 *   maxMapDistance: 20,
 * });
 * console.log(metrics.aggression);       // ≈ 0.667
 * console.log(metrics.positioning);       // 0.25
 * console.log(metrics.movementPattern);   // 0.06
 * ```
 */
export function computeEnemyBehaviorMetrics(
  telemetry: EnemyBehaviorTelemetry,
): EnemyBehaviorMetrics {
  const aggression = computeAggression(telemetry);
  const positioning = computePositioning(telemetry);
  const movementPattern = computeMovementPattern(telemetry);

  return { aggression, movementPattern, positioning };
}

/**
 * Compute the aggression descriptor from damage dealt and survival ticks.
 *
 * @param telemetry - Raw gameplay telemetry.
 * @returns Aggression in [0, 1].
 */
function computeAggression(telemetry: EnemyBehaviorTelemetry): number {
  const denominator =
    telemetry.damageDealt + telemetry.survivalTicks * AGGRESSION_SURVIVAL_SCALAR;
  if (denominator <= 0) {
    return 0;
  }
  return clamp01(telemetry.damageDealt / denominator);
}

/**
 * Compute the positioning descriptor from mean distance to the player.
 *
 * @param telemetry - Raw gameplay telemetry.
 * @returns Positioning in [0, 1].
 */
function computePositioning(telemetry: EnemyBehaviorTelemetry): number {
  if (telemetry.maxMapDistance <= 0) {
    return 0;
  }
  return clamp01(telemetry.meanDistanceToPlayer / telemetry.maxMapDistance);
}

/**
 * Compute the movement-pattern descriptor from direction-change frequency.
 *
 * @param telemetry - Raw gameplay telemetry.
 * @returns Movement pattern in [0, 1].
 */
function computeMovementPattern(telemetry: EnemyBehaviorTelemetry): number {
  return clamp01(telemetry.dirChangeCount / Math.max(1, telemetry.survivalTicks));
}