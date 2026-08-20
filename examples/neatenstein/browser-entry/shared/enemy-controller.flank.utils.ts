/**
 * Flanking slot and movement-mode resolver for the enemy controller pipeline.
 *
 * Extracted from {@link enemy-controller.ts} as a pure leaf executor. Assigns
 * each enemy a slot angle around the player so they spread out, performs
 * wall-aware slot placement, and determines whether the enemy should use BFS
 * or flanking movement this tick.
 *
 * @module
 */

import {
  ENEMY_CONTROLLER_FLANKING_RADIUS_CELLS,
  ENEMY_CONTROLLER_STOP_DISTANCE_CELLS,
  ENEMY_CONTROLLER_WALL_COLLISION_RADIUS_CELLS,
} from './enemy-controller.constants';
import type { EnemyUpdateContext } from './enemy-controller.types';
import { isPositionBlockedByWall } from './enemy-controller.collision.utils';
import { isFiniteNumber } from './math-guards.utils';

/** Small angle offsets (in radians) tried when the slot target is in a wall. */
const SLOT_ANGLE_OFFSETS = [
  Math.PI / 12,
  -Math.PI / 12,
  Math.PI / 6,
  -Math.PI / 6,
  Math.PI / 4,
  -Math.PI / 4,
  Math.PI / 3,
  -Math.PI / 3,
  Math.PI / 2,
  -Math.PI / 2,
];

/**
 * Resolve flanking slot position and select BFS vs flanking movement mode.
 *
 * Assigns each enemy an evenly-spaced slot angle around the player. If the
 * slot target lands inside a wall, tries shifting the angle by small
 * increments to find a nearby non-solid slot. Sets `shouldMoveByBfs` and
 * `shouldMoveByFlank` on the context based on distance to player, flanking
 * range, and stall-recovery fallback logic.
 *
 * @param ctx - Mutable pipeline context.
 */
export function resolveFlankState(ctx: EnemyUpdateContext): void {
  const { gameState, index, position, distToPlayer, collisionMap } = ctx;

  // Each enemy gets an evenly-spaced angle; with one enemy the slot is at
  // the player's position (no flanking).
  const numEnemies = gameState.enemies.length;
  const slotAngle = numEnemies > 1 ? (index * 2 * Math.PI) / numEnemies : 0;
  // Mutate slotTarget in place (A2 Fix 8: no new {x,y} allocation).
  ctx.slotTarget.x =
    gameState.player.position.x +
    Math.cos(slotAngle) * ENEMY_CONTROLLER_STOP_DISTANCE_CELLS;
  ctx.slotTarget.y =
    gameState.player.position.y +
    Math.sin(slotAngle) * ENEMY_CONTROLLER_STOP_DISTANCE_CELLS;

  // Wall-aware slot placement: if the slot target lands inside a wall,
  // try shifting the angle by small increments to find a nearby non-solid
  // slot. If no valid slot is found, fall back to BFS mode.
  let slotInWall = isPositionBlockedByWall(
    collisionMap,
    ctx.slotTarget.x,
    ctx.slotTarget.y,
    ENEMY_CONTROLLER_WALL_COLLISION_RADIUS_CELLS,
  );
  if (slotInWall) {
    for (const offset of SLOT_ANGLE_OFFSETS) {
      const shiftedAngle = slotAngle + offset;
      const candidateX =
        gameState.player.position.x +
        Math.cos(shiftedAngle) * ENEMY_CONTROLLER_STOP_DISTANCE_CELLS;
      const candidateY =
        gameState.player.position.y +
        Math.sin(shiftedAngle) * ENEMY_CONTROLLER_STOP_DISTANCE_CELLS;
      if (
        !isPositionBlockedByWall(
          collisionMap,
          candidateX,
          candidateY,
          ENEMY_CONTROLLER_WALL_COLLISION_RADIUS_CELLS,
        )
      ) {
        ctx.slotTarget.x = candidateX;
        ctx.slotTarget.y = candidateY;
        slotInWall = false;
        break;
      }
    }
  }

  const dxToSlot = ctx.slotTarget.x - position.x;
  const dyToSlot = ctx.slotTarget.y - position.y;
  const distToSlot = Math.hypot(dxToSlot, dyToSlot);
  const inFlankingRange =
    numEnemies > 1 &&
    isFiniteNumber(distToPlayer) &&
    distToPlayer <= ENEMY_CONTROLLER_FLANKING_RADIUS_CELLS;

  ctx.shouldMoveByBfs =
    (!inFlankingRange || slotInWall) &&
    isFiniteNumber(distToPlayer) &&
    distToPlayer > ENEMY_CONTROLLER_STOP_DISTANCE_CELLS;
  ctx.shouldMoveByFlank =
    inFlankingRange &&
    !slotInWall &&
    isFiniteNumber(distToSlot) &&
    distToSlot > 0.15;

  // Greedy descent stall fallback: if the enemy has been stalled in
  // flanking mode for too many consecutive ticks, temporarily switch to
  // BFS mode to prevent permanent stalls against walls.
  if (inFlankingRange && ctx.flankStallTicks > 3) {
    ctx.shouldMoveByBfs =
      isFiniteNumber(distToPlayer) &&
      distToPlayer > ENEMY_CONTROLLER_STOP_DISTANCE_CELLS;
    ctx.shouldMoveByFlank = false;
  }
}
