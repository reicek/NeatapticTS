/**
 * Collision, line-of-sight, and separation helper functions for the enemy
 * controller.
 *
 * Extracted from {@link enemy-controller.ts} as pure leaf executors. These
 * functions own no module-level mutable state and are safe to call from any
 * context.
 *
 * @module
 */

import type { CollisionMap } from '../renderer/map';
import type { Vector2 } from '../host/game/types';
import {
  ENEMY_CONTROLLER_RADIUS_CELLS,
  ENEMY_CONTROLLER_WALL_COLLISION_RADIUS_CELLS,
} from './enemy-controller.constants';
import type { ControlledEnemy } from './enemy-controller.types';
import { isFiniteNumber } from './math-guards.utils';

/** Line-of-sight sampling step in world cells. */
const LINE_OF_SIGHT_STEP_CELLS = 0.5;

/**
 * Check whether a circle at the given position overlaps any solid cell.
 *
 * For a circle of radius R centered at (x, y), this checks every grid cell
 * whose bounding box could overlap the circle. The cell range is computed
 * as floor(x − R) to ceil(x + R) − 1 on each axis, then each cell in that
 * bounding box is tested for solidity. This catches all wall cells the
 * enemy circle could overlap, including diagonal cells, preventing enemies
 * from clipping into walls or getting stuck inside them.
 *
 * @param collisionMap - Map queried for solid cells.
 * @param x - Desired center X in world cells.
 * @param y - Desired center Y in world cells.
 * @param radius - Collision radius in world cells.
 * @returns Whether any solid cell overlaps the circle's bounding box.
 */
export function isPositionBlockedByWall(
  collisionMap: CollisionMap,
  x: number,
  y: number,
  radius: number,
): boolean {
  const minX = Math.floor(x - radius);
  const maxX = Math.ceil(x + radius) - 1;
  const minY = Math.floor(y - radius);
  const maxY = Math.ceil(y + radius) - 1;
  for (let cy = minY; cy <= maxY; cy += 1) {
    for (let cx = minX; cx <= maxX; cx += 1) {
      if (collisionMap.isSolid(cx, cy)) {
        return true;
      }
    }
  }
  return false;
}

/**
 * Return whether a straight line from `from` to `to` is free of solid cells.
 *
 * Samples the grid at regular intervals. The start cell is skipped because the
 * caller is expected to be in an open cell.
 *
 * @param from - Ray origin in world cells.
 * @param to - Ray target in world cells.
 * @param collisionMap - Map queried for solid cells.
 * @returns Whether the target is visible from the origin.
 */
export function hasLineOfSight(
  from: Vector2,
  to: Vector2,
  collisionMap: CollisionMap,
): boolean {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const distance = Math.hypot(dx, dy);

  if (!isFiniteNumber(distance) || distance === 0) {
    return true;
  }

  const steps = Math.max(1, Math.ceil(distance / LINE_OF_SIGHT_STEP_CELLS));
  const stepX = dx / steps;
  const stepY = dy / steps;

  for (let i = 1; i <= steps; i += 1) {
    const x = from.x + stepX * i;
    const y = from.y + stepY * i;

    if (collisionMap.isSolid(Math.floor(x), Math.floor(y))) {
      return false;
    }
  }

  return true;
}

/**
 * Push active enemies apart so their 192×192-block footprints do not overlap.
 *
 * A single pairwise pass is sufficient because the AI moves slowly and the
 * collision radius is small. The separation is applied symmetrically, so two
 * overlapping enemies each move half the overlap distance.
 *
 * After the separation pass, each active enemy's position is re-checked
 * against the wall collision map. If separation pushed an enemy inside a
 * wall, its position is reverted to the pre-separation position (which was
 * already verified wall-free by the movement code).
 *
 * @param enemies - Resolved enemy descriptors produced by
 *   {@link updateControlledEnemy} this tick.
 * @param collisionMap - Map queried for solid cells, used for the
 *   post-separation wall re-check.
 */
export function separateEnemies(
  enemies: ControlledEnemy[],
  collisionMap: CollisionMap,
): void {
  const combinedDiameter = ENEMY_CONTROLLER_RADIUS_CELLS * 2;

  // Save pre-separation positions for the post-separation wall re-check.
  const prePositions = enemies.map((e) => ({
    x: e.position.x,
    y: e.position.y,
  }));

  for (let i = 0; i < enemies.length; i += 1) {
    const a = enemies[i];
    if (!a.active || a.stunTimerMs > 0) {
      continue;
    }

    for (let j = i + 1; j < enemies.length; j += 1) {
      const b = enemies[j];
      if (!b.active || b.stunTimerMs > 0) {
        continue;
      }

      const dx = b.position.x - a.position.x;
      const dy = b.position.y - a.position.y;
      const distanceSquared = dx * dx + dy * dy;
      const combinedRadius = combinedDiameter;
      const combinedRadiusSquared = combinedRadius * combinedRadius;

      if (distanceSquared >= combinedRadiusSquared) {
        continue;
      }

      const distance = Math.sqrt(distanceSquared) || 1;
      const overlap = combinedRadius - distance;
      const offsetX = (dx / distance) * (overlap * 0.5);
      const offsetY = (dy / distance) * (overlap * 0.5);

      a.position.x -= offsetX;
      a.position.y -= offsetY;
      b.position.x += offsetX;
      b.position.y += offsetY;
    }
  }

  // Wall re-check: if separation pushed an enemy inside a wall, revert to
  // the pre-separation position (already verified wall-free by the move).
  for (let i = 0; i < enemies.length; i += 1) {
    const e = enemies[i];
    if (!e.active || e.stunTimerMs > 0) {
      continue;
    }
    if (
      isPositionBlockedByWall(
        collisionMap,
        e.position.x,
        e.position.y,
        ENEMY_CONTROLLER_WALL_COLLISION_RADIUS_CELLS,
      )
    ) {
      e.position.x = prePositions[i].x;
      e.position.y = prePositions[i].y;
    }
  }
}
