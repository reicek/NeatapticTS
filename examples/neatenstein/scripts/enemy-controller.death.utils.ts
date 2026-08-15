/**
 * Death / de-rez executor for the enemy controller pipeline.
 *
 * Extracted from {@link enemy-controller.ts} as a pure leaf executor. When
 * the source enemy's health has reached zero, advances the de-rez animation
 * timer and returns a terminal {@link ControlledEnemy} descriptor.
 *
 * @module
 */

import type { ControlledEnemy, EnemyUpdateContext } from './enemy-controller.types';
import { ENEMY_CONTROLLER_DE_REZ_DURATION_MS, PREVIOUS_STEP_DISTANCE_SENTINEL } from './enemy-controller.constants';

/**
 * Handle the death / de-rez branch.
 *
 * When health is zero or below, advances the de-rez elapsed timer by the
 * tick duration and returns a terminal {@link ControlledEnemy} in the
 * `'death'` animation state. The `active` flag is `true` until the de-rez
 * duration elapses, after which it becomes `false` (signaling the display
 * worker to compact the enemy roster).
 *
 * @param ctx - Mutable pipeline context.
 * @returns Terminal controlled enemy descriptor, or `null` to continue the
 *   alive-path pipeline.
 */
export function handleDeath(ctx: EnemyUpdateContext): ControlledEnemy | null {
  if (ctx.enemyState.health > 0) return null;

  ctx.deRezElapsedMs += ctx.dtMs;
  const active = ctx.deRezElapsedMs < ENEMY_CONTROLLER_DE_REZ_DURATION_MS;

  return {
    index: ctx.index,
    position: ctx.position,
    health: ctx.enemyState.health,
    yawRad: ctx.yawRad,
    animationState: 'death',
    ammo: Math.max(0, ctx.ammo),
    fireCooldownMs: ctx.fireCooldownMs,
    deRezElapsedMs: ctx.deRezElapsedMs,
    active,
    walkTick: 0,
    shootBlinkTicks: 0,
    flankStallTicks: 0,
    bfsStallTicks: 0,
    weights: ctx.weights,
    variantId: ctx.variantId,
    previousStepDistance: PREVIOUS_STEP_DISTANCE_SENTINEL,
    stunTimerMs: 0,
  };
}
