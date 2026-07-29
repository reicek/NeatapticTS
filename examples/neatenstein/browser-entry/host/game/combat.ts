/**
 * Traveling plasma-bolt combat for the Neatenstein host-side simulation.
 *
 * This module owns the player's primary weapon. Gameplay remains deterministic:
 * firing resolves the nearest enemy or wall along the aim ray, consumes ammo,
 * applies damage, and spawns a traveling {@link BoltState} projectile plus a
 * wall-impact marker when the ray terminates on a wall.
 *
 * @module
 */

import {
  NEATENSTEIN_IMPACT_SPOT_LIFETIME_MS,
  NEATENSTEIN_MAP_SIZE,
} from '../../constants';
import {
  buildNeatensteinMap,
  castRayDDAFromFlatMap,
} from '../../renderer/raycast';
import {
  NEATENSTEIN_BOLT_DAMAGE,
  NEATENSTEIN_BOLT_HIT_RADIUS_CELLS,
  NEATENSTEIN_BOLT_MAX_RANGE_CELLS,
  NEATENSTEIN_BOLT_SPEED_CELLS_PER_SECOND,
  NEATENSTEIN_BOLT_TRAVEL_DURATION_MS,
  NEATENSTEIN_MUZZLE_OFFSET_CELLS,
} from './constants';
import { consumeAmmo } from './state';
import type { BoltState, GameState, ImpactSpot, Vector2 } from './types';

/** Re-export the muzzle offset so tests can assert bolt spawn position. */
export { NEATENSTEIN_MUZZLE_OFFSET_CELLS } from './constants';

/** Re-export the bolt damage constant for test contracts. */
export { NEATENSTEIN_BOLT_DAMAGE } from './constants';

/**
 * Result of attempting to fire the traveling plasma bolt.
 */
export interface FireBoltResult {
  /** Snapshot after the shot: ammo consumed, bolt appended, damage applied. */
  state: GameState;

  /** `true` when a shot was actually fired this frame. */
  fired: boolean;

  /** Spawned bolt for this frame, or `null` when the weapon did not fire. */
  bolt: BoltState | null;
}

/**
 * Cached deterministic combat map.
 *
 * Firing can happen frequently, and rebuilding the same seeded map for every
 * shot is unnecessary. The cache is intentionally tiny because combat only
 * needs the current state's seed.
 */
let cachedMapSeed: number | null = null;
let cachedFlatMap: Uint8Array | null = null;

/**
 * Return a deterministic flat map for the supplied seed.
 *
 * @param seed - Game-state seed.
 * @returns Cached or freshly generated flat wall map.
 */
function resolveCombatMap(seed: number): Uint8Array {
  if (cachedFlatMap !== null && cachedMapSeed === seed) {
    return cachedFlatMap;
  }

  cachedMapSeed = seed;
  cachedFlatMap = buildNeatensteinMap(seed);

  return cachedFlatMap;
}

/**
 * Build a world position along the shot ray.
 *
 * @param origin - Shot origin.
 * @param direction - Normalized shot direction.
 * @param distance - Distance along the ray.
 * @returns World position at `origin + direction * distance`.
 */
function pointAlongRay(
  origin: Vector2,
  direction: Vector2,
  distance: number,
): Vector2 {
  return {
    x: origin.x + direction.x * distance,
    y: origin.y + direction.y * distance,
  };
}

/**
 * Fire the traveling plasma bolt and return the updated state plus the bolt.
 *
 * Combat resolution remains immediate and deterministic:
 *
 * 1. Ammo is checked.
 * 2. The shot ray is built from player position and yaw.
 * 3. The nearest wall along the ray is found.
 * 4. Living enemies are tested against the bolt path cylinder.
 * 5. The nearest valid enemy before the wall takes damage.
 * 6. A traveling {@link BoltState} is appended.
 * 7. Wall-impact spots are emitted only when the ray terminates on a wall.
 *
 * @param state - Snapshot before firing.
 * @returns Immutable result with the new state, fire flag, and spawned bolt.
 *
 * @example
 * ```ts
 * const result = fireBolt(state);
 * if (result.fired && result.bolt) {
 *   state.bolts.push(result.bolt);
 * }
 * ```
 */
export function fireBolt(state: GameState): FireBoltResult {
  if (state.player.ammo <= 0) {
    return { state, fired: false, bolt: null };
  }

  const direction: Vector2 = {
    x: Math.cos(state.player.angleRad),
    y: Math.sin(state.player.angleRad),
  };

  const origin: Vector2 = {
    x: state.player.position.x + direction.x * NEATENSTEIN_MUZZLE_OFFSET_CELLS,
    y: state.player.position.y + direction.y * NEATENSTEIN_MUZZLE_OFFSET_CELLS,
  };

  const flatMap = resolveCombatMap(state.seed);
  const wallHit = castRayDDAFromFlatMap(
    flatMap,
    NEATENSTEIN_MAP_SIZE,
    origin.x,
    origin.y,
    direction.x,
    direction.y,
  );

  const rawWallDistance = Number.isFinite(wallHit.perpWallDist)
    ? wallHit.perpWallDist
    : Number.POSITIVE_INFINITY;

  let hitType: 'wall' | 'enemy' | 'range' =
    rawWallDistance <= NEATENSTEIN_BOLT_MAX_RANGE_CELLS ? 'wall' : 'range';
  let hitDistance = Math.min(rawWallDistance, NEATENSTEIN_BOLT_MAX_RANGE_CELLS);
  let hitEnemyIndex = -1;

  // Test every living enemy against the bolt path and keep the nearest valid
  // hit before the wall.
  for (let index = 0; index < state.enemies.length; index += 1) {
    const enemy = state.enemies[index];

    if (enemy.health <= 0) {
      continue;
    }

    const distanceAlongBolt = projectOntoRay(origin, direction, enemy.position);

    if (distanceAlongBolt <= 0 || distanceAlongBolt > hitDistance) {
      continue;
    }

    const missDistance = perpendicularDistance(
      origin,
      direction,
      enemy.position,
      distanceAlongBolt,
    );

    if (missDistance <= NEATENSTEIN_BOLT_HIT_RADIUS_CELLS) {
      hitType = 'enemy';
      hitDistance = distanceAlongBolt;
      hitEnemyIndex = index;
    }
  }

  const hit = pointAlongRay(origin, direction, hitDistance);

  let nextState = consumeAmmo(state);

  const bolt: BoltState = {
    position: { ...origin },
    direction: { ...direction },
    speedCellsPerSecond: NEATENSTEIN_BOLT_SPEED_CELLS_PER_SECOND,
    active: true,
    createdAtMs: state.simTimeMs,
    origin: { ...origin },
    targetDistance: Math.max(0, hitDistance),
  };

  nextState = {
    ...nextState,
    bolts: [...(nextState.bolts ?? []), bolt],
  };

  // Only create a wall impact when the shot truly terminated on a wall within
  // the bolt's maximum range. Beyond that range the bolt vanishes in mid-air.
  if (hitType === 'wall') {
    const wallHitCoordinate =
      wallHit.side === 0
        ? origin.y + wallHit.perpWallDist * direction.y
        : origin.x + wallHit.perpWallDist * direction.x;

    const impact: ImpactSpot = {
      wallHit: {
        mapX: wallHit.mapX,
        mapY: wallHit.mapY,
        side: wallHit.side,
        wallX: wallHitCoordinate - Math.floor(wallHitCoordinate),
      },
      position: { ...hit },
      createdAtMs: state.simTimeMs,
      lifetimeMs: NEATENSTEIN_IMPACT_SPOT_LIFETIME_MS,
      perpWallDist: wallHit.perpWallDist,
      boltTravelTimeMs: NEATENSTEIN_BOLT_TRAVEL_DURATION_MS,
    };

    nextState = {
      ...nextState,
      impacts: [...nextState.impacts, impact],
    };
  }

  if (hitType === 'enemy' && hitEnemyIndex >= 0) {
    nextState = applyEnemyDamage(nextState, hitEnemyIndex);
  }

  return {
    state: nextState,
    fired: true,
    bolt,
  };
}

/**
 * Project a point onto the bolt ray and return signed distance from origin.
 *
 * Negative values mean the point is behind the bolt origin and should be
 * ignored.
 *
 * @param origin - Bolt origin.
 * @param direction - Normalized bolt direction.
 * @param point - Enemy position to test.
 * @returns Signed distance along the bolt from origin to closest approach.
 */
function projectOntoRay(
  origin: Vector2,
  direction: Vector2,
  point: Vector2,
): number {
  return (
    (point.x - origin.x) * direction.x + (point.y - origin.y) * direction.y
  );
}

/**
 * Compute the perpendicular distance from a point to the bolt ray.
 *
 * @param origin - Bolt origin.
 * @param direction - Normalized bolt direction.
 * @param point - Enemy position to test.
 * @param t - Distance along the bolt to closest approach.
 * @returns Euclidean distance from the point to the bolt.
 */
function perpendicularDistance(
  origin: Vector2,
  direction: Vector2,
  point: Vector2,
  t: number,
): number {
  const closestX = origin.x + direction.x * t;
  const closestY = origin.y + direction.y * t;

  return Math.hypot(point.x - closestX, point.y - closestY);
}

/**
 * Apply bolt damage to the enemy at the given index.
 *
 * Health is clamped at zero and the kill counter increments only when an enemy
 * transitions from alive to dead in this shot.
 *
 * @param state - Snapshot with the bolt already appended.
 * @param enemyIndex - Index into {@link GameState.enemies}.
 * @returns New snapshot with updated enemy health and kill count.
 */
function applyEnemyDamage(state: GameState, enemyIndex: number): GameState {
  const enemy = state.enemies[enemyIndex];
  const newHealth = Math.max(0, enemy.health - NEATENSTEIN_BOLT_DAMAGE);
  const killedByThisShot = newHealth === 0;

  const newEnemies = state.enemies.map((existing, index) =>
    index === enemyIndex ? { ...existing, health: newHealth } : existing,
  );

  return {
    ...state,
    enemies: newEnemies,
    kills: killedByThisShot ? state.kills + 1 : state.kills,
  };
}
