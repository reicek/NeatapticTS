/**
 * Player plasma-bolt movement and enemy-collision executors for the
 * Neatenstein tick pipeline.
 *
 * @module
 */

import {
  NEATENSTEIN_BOLT_HIT_RADIUS_CELLS,
  NEATENSTEIN_BOLT_MAX_RANGE_CELLS,
  NEATENSTEIN_BOLT_TRAVEL_DURATION_MS,
  NEATENSTEIN_MAP_SIZE,
  NEATENSTEIN_MS_PER_SECOND,
} from './constants';
import { resolveTickDurationMs } from './tick.time.utils';
import type { CollisionMap } from '../../renderer/map';
import type { BoltState, EnemyState, Vector2 } from './types';

/**
 * Project a point onto a bolt ray and return the signed distance from origin.
 *
 * @param origin - Bolt origin.
 * @param direction - Normalized bolt direction.
 * @param point - Enemy position to test.
 * @returns Signed distance along the bolt from origin to closest approach.
 */
function projectOntoBoltRay(
  origin: Vector2,
  direction: Vector2,
  point: Vector2,
): number {
  return (
    (point.x - origin.x) * direction.x + (point.y - origin.y) * direction.y
  );
}

/**
 * Return the perpendicular distance from a point to a bolt ray.
 *
 * @param origin - Bolt origin.
 * @param direction - Normalized bolt direction.
 * @param point - Enemy position to test.
 * @param distanceAlong - Signed distance along the ray already computed.
 * @returns Shortest distance from point to the ray line.
 */
function perpendicularDistanceToBoltRay(
  origin: Vector2,
  direction: Vector2,
  point: Vector2,
  distanceAlong: number,
): number {
  return Math.hypot(
    origin.x + direction.x * distanceAlong - point.x,
    origin.y + direction.y * distanceAlong - point.y,
  );
}

/**
 * Find the nearest active enemy that this bolt will hit during the tick.
 *
 * @param bolt - Bolt snapshot before movement.
 * @param step - Distance the bolt will travel this tick.
 * @param enemies - Active enemy snapshots.
 * @returns Tuple of `[impactDistance, enemyIndex]` for the first hit, or
 *   `null` when no enemy is struck.
 */
function findBoltEnemyImpact(
  bolt: BoltState,
  step: number,
  enemies: EnemyState[],
): [number, number] | null {
  if (!bolt.origin || enemies.length === 0) {
    return null;
  }

  let best: [number, number] | null = null;

  for (let index = 0; index < enemies.length; index += 1) {
    const enemy = enemies[index];

    if (
      enemy.health <= 0 ||
      enemy.active === false ||
      !enemy.position ||
      !Number.isFinite(enemy.position.x) ||
      !Number.isFinite(enemy.position.y)
    ) {
      continue;
    }

    const hitRadius = bolt.radius ?? NEATENSTEIN_BOLT_HIT_RADIUS_CELLS;
    const distanceAlong = projectOntoBoltRay(
      bolt.position,
      bolt.direction,
      enemy.position,
    );

    if (distanceAlong < -hitRadius || distanceAlong > step + hitRadius) {
      continue;
    }

    const missDistance = perpendicularDistanceToBoltRay(
      bolt.position,
      bolt.direction,
      enemy.position,
      distanceAlong,
    );

    if (missDistance <= (bolt.radius ?? NEATENSTEIN_BOLT_HIT_RADIUS_CELLS)) {
      if (best === null || distanceAlong < best[0]) {
        best = [distanceAlong, index];
      }
    }
  }

  return best;
}

/**
 * Advance active plasma bolts by one tick.
 *
 * Each active bolt is moved along its direction by `speed * dt`. Bolts remain
 * active until their on-screen travel time reaches
 * {@link NEATENSTEIN_BOLT_TRAVEL_DURATION_MS} so a close target never makes
 * the bolt vanish before the 300 ms screen travel completes. Bolts that leave
 * the world bounds, hit a wall, or exceed the maximum travel range stop moving
 * but stay active and remain in the returned array until the visual travel
 * duration expires. Bolts that were already inactive are removed.
 *
 * @param bolts - Active bolt snapshots before this tick.
 * @param dtMs - Elapsed time in milliseconds.
 * @param currentTimeMs - Current simulation time in milliseconds, used to
 *   decide when the bolt's screen travel has finished.
 * @param collisionMap - Optional collision map used to stop bolts that hit a
 *   wall.
 * @param enemies - Optional active enemy snapshots used for bolt-enemy
 *   collision.
 * @returns New array of bolts after movement and deactivation.
 */
export function updateBolts(
  bolts: BoltState[],
  dtMs: number,
  currentTimeMs: number,
  collisionMap?: CollisionMap,
  enemies?: EnemyState[],
): BoltState[] {
  const resolvedDtMs = resolveTickDurationMs(dtMs);
  const dtSeconds = resolvedDtMs / NEATENSTEIN_MS_PER_SECOND;

  // In-place mutation (A2 Fix 5): mutate each bolt's position, active flag,
  // and hitEnemyIndex in place — no .filter().map() clone chains.
  // Compaction of inactive bolts is deferred to the caller (tick.ts).
  let writeIndex = 0;

  for (let readIndex = 0; readIndex < bolts.length; readIndex += 1) {
    const bolt = bolts[readIndex];

    if (!bolt.active) {
      continue;
    }

    const step = bolt.speedCellsPerSecond * dtSeconds;
    const enemyImpact =
      enemies && enemies.length > 0
        ? findBoltEnemyImpact(bolt, step, enemies)
        : null;

    // Compute next position.
    const nextX = enemyImpact
      ? bolt.position.x + bolt.direction.x * enemyImpact[0]
      : bolt.position.x + bolt.direction.x * step;
    const nextY = enemyImpact
      ? bolt.position.y + bolt.direction.y * enemyImpact[0]
      : bolt.position.y + bolt.direction.y * step;

    const outOfBounds =
      nextX < 0 ||
      nextX >= NEATENSTEIN_MAP_SIZE ||
      nextY < 0 ||
      nextY >= NEATENSTEIN_MAP_SIZE;
    const hitWall = collisionMap
      ? collisionMap.isSolid(Math.floor(nextX), Math.floor(nextY))
      : false;
    const distanceTraveled =
      bolt.origin &&
      Number.isFinite(bolt.origin.x) &&
      Number.isFinite(bolt.origin.y)
        ? Math.hypot(nextX - bolt.origin.x, nextY - bolt.origin.y)
        : 0;
    const beyondMaxRange = distanceTraveled >= NEATENSTEIN_BOLT_MAX_RANGE_CELLS;
    const reachedTarget =
      bolt.targetDistance !== undefined &&
      Number.isFinite(bolt.targetDistance) &&
      distanceTraveled >= bolt.targetDistance;
    const elapsedMs = Math.max(0, currentTimeMs - bolt.createdAtMs);
    const travelExpired = elapsedMs >= NEATENSTEIN_BOLT_TRAVEL_DURATION_MS;
    const hitEnemy = enemyImpact !== null;
    const movementStopped =
      outOfBounds || hitWall || beyondMaxRange || reachedTarget;
    const active = !travelExpired && !hitEnemy;

    // Update position in place.
    if (hitEnemy) {
      bolt.position.x = nextX;
      bolt.position.y = nextY;
    } else if (!movementStopped) {
      bolt.position.x = nextX;
      bolt.position.y = nextY;
    }
    // If movementStopped and !hitEnemy, position stays the same.

    bolt.active = active;
    bolt.hitEnemyIndex = enemyImpact ? enemyImpact[1] : bolt.hitEnemyIndex;

    // Keep the bolt in the array (compaction deferred to caller).
    bolts[writeIndex] = bolt;
    writeIndex += 1;
  }

  bolts.length = writeIndex;
  return bolts;
}
