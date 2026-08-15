/**
 * Hitscan fire decision and shoot-blink executor for the enemy controller
 * pipeline.
 *
 * Extracted from {@link enemy-controller.ts} as a pure leaf executor. Determines
 * whether the enemy fires this tick, pushes a hitscan event, updates ammunition
 * and fire cooldown, decrements the shoot blink counter, and sets the animation
 * state.
 *
 * @module
 */

import {
  ENEMY_CONTROLLER_FIRE_COOLDOWN_MS,
  ENEMY_CONTROLLER_FIRE_RANGE_CELLS,
  ENEMY_CONTROLLER_HITSCAN_DAMAGE,
  ENEMY_CONTROLLER_SHOOT_BLINK_TICKS,
} from './enemy-controller.constants';
import type { EnemyUpdateContext } from './enemy-controller.types';
import { hasLineOfSight } from './enemy-controller.collision.utils';

/**
 * Resolve the fire decision, shoot blink, and animation state for one tick.
 *
 * The enemy fires when its fire cooldown has elapsed, it has ammunition
 * remaining, the player is within fire range, and there is an unobstructed
 * line of sight. On firing, a {@link HitscanEvent} is pushed, ammunition is
 * decremented, and the fire cooldown is reset. The shoot blink counter is
 * decremented from the previous tick and refreshed if firing. The animation
 * state is set to `'fire'` if firing, `'move'` if moved, or `'idle'`
 * otherwise.
 *
 * @param ctx - Mutable pipeline context.
 */
export function resolveFire(ctx: EnemyUpdateContext): void {
  let isFiring = false;
  if (
    ctx.fireCooldownMs <= 0 &&
    ctx.ammo > 0 &&
    ctx.distToPlayer <= ENEMY_CONTROLLER_FIRE_RANGE_CELLS &&
    hasLineOfSight(
      ctx.position,
      ctx.gameState.player.position,
      ctx.collisionMap,
    )
  ) {
    isFiring = true;
    ctx.hitscanEvents.push({
      enemyIndex: ctx.index,
      origin: { ...ctx.position },
      direction: { x: Math.cos(ctx.yawRad), y: Math.sin(ctx.yawRad) },
      damage: ENEMY_CONTROLLER_HITSCAN_DAMAGE,
    });
  }
  ctx.isFiring = isFiring;

  // Decrement shoot blink from the previous tick, then refresh if firing.
  if (ctx.shootBlinkTicks > 0) {
    ctx.shootBlinkTicks -= 1;
  }
  if (isFiring) {
    ctx.shootBlinkTicks = ENEMY_CONTROLLER_SHOOT_BLINK_TICKS;
  }

  // Update ammo and fire cooldown.
  if (isFiring) {
    ctx.ammo = Math.max(0, ctx.ammo - 1);
    ctx.fireCooldownMs = ENEMY_CONTROLLER_FIRE_COOLDOWN_MS;
  }

  // Set animation state.
  if (isFiring) {
    ctx.animationState = 'fire';
  } else if (ctx.moved) {
    ctx.animationState = 'move';
  }
}
