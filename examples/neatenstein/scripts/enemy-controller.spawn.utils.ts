/**
 * Spawn / respawn executor for the enemy controller pipeline.
 *
 * Extracted from {@link enemy-controller.ts} as a pure leaf executor. Reads
 * the previous controlled state and source enemy snapshot, detects respawn,
 * and initializes all mutable pipeline state on the context object.
 *
 * @module
 */

import type { EnemyUpdateContext } from './enemy-controller.types';
import {
  ENEMY_CONTROLLER_STARTING_AMMO,
  PREVIOUS_STEP_DISTANCE_SENTINEL,
} from './enemy-controller.constants';

/**
 * Detect respawn and initialize all mutable pipeline state on the context.
 *
 * A respawn occurs when a previous controlled enemy exists, is no longer
 * active (fully de-rezzed), and the source enemy has positive health. On
 * respawn, all per-enemy state is reset to fresh defaults; otherwise, state
 * carries forward from the previous tick.
 *
 * @param ctx - Mutable pipeline context.
 */
export function resolveRespawnState(ctx: EnemyUpdateContext): void {
  ctx.isRespawn =
    ctx.previous !== undefined &&
    !ctx.previous.active &&
    ctx.enemyState.health > 0;

  ctx.previousOrDefault =
    ctx.previous ??
    ({
      index: ctx.index,
      position: { ...ctx.enemyState.position },
      health: ctx.enemyState.health,
      yawRad: 0,
      animationState: 'idle',
      ammo: ENEMY_CONTROLLER_STARTING_AMMO,
      fireCooldownMs: 0,
      deRezElapsedMs: 0,
      active: true,
      walkTick: 0,
      shootBlinkTicks: 0,
      flankStallTicks: 0,
      bfsStallTicks: 0,
      weights: undefined,
      variantId: 0,
      previousStepDistance: PREVIOUS_STEP_DISTANCE_SENTINEL,
      stunTimerMs: 0,
    } as typeof ctx.previousOrDefault);

  ctx.ammo = ctx.isRespawn
    ? ENEMY_CONTROLLER_STARTING_AMMO
    : ctx.previousOrDefault.ammo;
  ctx.fireCooldownMs = Math.max(
    0,
    (ctx.isRespawn ? 0 : ctx.previousOrDefault.fireCooldownMs) - ctx.dtMs,
  );
  ctx.deRezElapsedMs = ctx.isRespawn ? 0 : ctx.previousOrDefault.deRezElapsedMs;
  ctx.walkTick = ctx.isRespawn ? 0 : ctx.previousOrDefault.walkTick;
  ctx.shootBlinkTicks = ctx.isRespawn
    ? 0
    : ctx.previousOrDefault.shootBlinkTicks;
  ctx.flankStallTicks = ctx.isRespawn
    ? 0
    : ctx.previousOrDefault.flankStallTicks;
  ctx.bfsStallTicks = ctx.isRespawn ? 0 : ctx.previousOrDefault.bfsStallTicks;
  ctx.weights =
    ctx.injectedWeights ??
    (ctx.isRespawn ? undefined : ctx.previousOrDefault.weights);
  ctx.variantId = ctx.isRespawn ? 0 : ctx.previousOrDefault.variantId;
  ctx.position = ctx.isRespawn
    ? { ...(ctx.enemyState.initialPosition ?? ctx.enemyState.position) }
    : { ...ctx.previousOrDefault.position };
  ctx.yawRad = ctx.isRespawn ? 0 : ctx.previousOrDefault.yawRad;
  ctx.moved = false;
  ctx.animationState = 'idle';
  ctx.shouldMoveByBfs = false;
  ctx.shouldMoveByFlank = false;
  ctx.slotTarget = { x: 0, y: 0 };
  ctx.isFiring = false;
  ctx.stunTimerMs = 0;
  ctx.distToPlayer = 0;
}
