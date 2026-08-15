/**
 * Hit-stun executor for the enemy controller pipeline.
 *
 * Extracted from {@link enemy-controller.ts} as a pure leaf executor. While
 * the enemy is stunned (stunTimerMs > 0), movement, MLP, and fire are
 * skipped and the enemy adopts the pushed-back position from EnemyState.
 *
 * @module
 */

import { NEATENSTEIN_FIXED_TIMESTEP_MS } from '../browser-entry/host/game/constants';
import type { ControlledEnemy, EnemyUpdateContext } from './enemy-controller.types';
import { PREVIOUS_STEP_DISTANCE_SENTINEL } from './enemy-controller.constants';

/**
 * Handle the hit-stun branch.
 *
 * Computes the remaining stun timer (decremented by the fixed simulation
 * timestep, not the variable rAF dtMs, for deterministic stun duration).
 * On zero-timestep sync passes (dtMs === 0) the timer is not decremented.
 *
 * While stunned or transitioning out of stun, the enemy adopts the
 * pushed-back position from {@link EnemyState} (not the stale
 * {@link ControlledEnemy} position) and returns a terminal descriptor with
 * the `'damage'` animation state.
 *
 * @param ctx - Mutable pipeline context.
 * @returns Terminal controlled enemy descriptor, or `null` to continue the
 *   alive-path pipeline.
 */
export function handleStun(ctx: EnemyUpdateContext): ControlledEnemy | null {
  ctx.stunTimerMs = Math.max(
    0,
    (ctx.enemyState.stunTimerMs ?? 0) -
      (ctx.dtMs > 0 ? NEATENSTEIN_FIXED_TIMESTEP_MS : 0),
  );

  if (ctx.stunTimerMs <= 0 && (ctx.enemyState.stunTimerMs ?? 0) <= 0) {
    return null;
  }

  // Adopt the pushed-back position from EnemyState, not the stale
  // ControlledEnemy position.
  const stunPosition = { ...ctx.enemyState.position };

  return {
    index: ctx.index,
    position: stunPosition,
    health: ctx.enemyState.health,
    yawRad: ctx.yawRad,
    animationState: 'damage',
    ammo: Math.max(0, ctx.ammo),
    fireCooldownMs: ctx.fireCooldownMs,
    deRezElapsedMs: ctx.deRezElapsedMs,
    active: true,
    walkTick: 0,
    shootBlinkTicks: 0,
    flankStallTicks: 0,
    bfsStallTicks: 0,
    weights: ctx.weights,
    variantId: ctx.variantId,
    previousStepDistance: PREVIOUS_STEP_DISTANCE_SENTINEL,
    stunTimerMs: ctx.stunTimerMs,
  };
}
