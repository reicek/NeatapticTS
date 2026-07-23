/**
 * Hitscan neon beam combat for the Neatenstein host-side simulation.
 *
 * This module owns the only weapon in the demo — a deterministic hitscan
 * beam that intersects the nearest enemy or wall and produces a visible
 * tracer. It stays deliberately small: no weapon switching, no projectiles
 * with flight time, and no cooldown beyond the ammo gate.
 *
 * @module
 */

import { NEATENSTEIN_IMPACT_SPOT_LIFETIME_MS } from '../../constants';
import {
  buildNeatensteinMap,
  castRayDDAFromFlatMap,
} from '../../renderer/raycast';
import {
  NEATENSTEIN_BEAM_COLOR,
  NEATENSTEIN_BEAM_DAMAGE,
  NEATENSTEIN_BEAM_MAX_RANGE_CELLS,
  NEATENSTEIN_ENEMY_HIT_RADIUS_CELLS,
  NEATENSTEIN_MUZZLE_OFFSET_CELLS,
  NEATENSTEIN_TRACER_DURATION_MS,
} from './constants';
import { consumeAmmo } from './state';
import type { GameState, ImpactSpot, TracerState, Vector2 } from './types';

/** Result of attempting to fire the neon beam. */
export interface FireNeonBeamResult {
  /** Snapshot after the shot (ammo consumed, tracer appended, damage applied). */
  state: GameState;
  /** `true` when a shot was actually fired this frame. */
  fired: boolean;
  /** Visible tracer for this frame, or `null` when the weapon did not fire. */
  tracer: TracerState | null;
}

/**
 * Fire the neon beam and return the updated state plus a visible tracer.
 *
 * The beam is hitscan: it instantly tests every living enemy within range and
 * the wall grid, then terminates on whichever is closest to the player. A hit
 * enemy loses {@link NEATENSTEIN_BEAM_DAMAGE} hit points and the kill counter
 * increments when the enemy is reduced to zero. The tracer records the ray
 * origin, direction, endpoint, and hit type so the renderer can draw a short
 * neon flash.
 *
 * @param state - Snapshot before firing.
 * @returns Immutable result with the new state, fire flag, and tracer.
 *
 * @example
 * ```ts
 * const result = fireNeonBeam(state);
 * if (result.fired) {
 *   drawTracer(result.tracer);
 * }
 * ```
 */
export function fireNeonBeam(state: GameState): FireNeonBeamResult {
  if (state.player.ammo <= 0) {
    return { state, fired: false, tracer: null };
  }

  const direction: Vector2 = {
    x: Math.cos(state.player.angleRad),
    y: Math.sin(state.player.angleRad),
  };
  const origin: Vector2 = {
    x: state.player.position.x + direction.x * NEATENSTEIN_MUZZLE_OFFSET_CELLS,
    y: state.player.position.y + direction.y * NEATENSTEIN_MUZZLE_OFFSET_CELLS,
  };

  const flatMap = buildNeatensteinMap(state.seed);
  const side = Math.round(Math.sqrt(flatMap.length));
  const wallHit = castRayDDAFromFlatMap(
    flatMap,
    side,
    origin.x,
    origin.y,
    direction.x,
    direction.y,
  );

  let hitType: 'wall' | 'enemy' = 'wall';
  let hitDistance = Math.min(
    Math.max(0, wallHit.perpWallDist),
    NEATENSTEIN_BEAM_MAX_RANGE_CELLS,
  );
  let hitEnemyIndex = -1;

  for (let index = 0; index < state.enemies.length; index++) {
    const enemy = state.enemies[index];
    if (enemy.health <= 0) {
      continue;
    }

    const t = projectOntoBeam(origin, direction, enemy.position);
    if (t <= 0 || t > NEATENSTEIN_BEAM_MAX_RANGE_CELLS || t > hitDistance) {
      continue;
    }

    const perpDistance = perpendicularDistance(
      origin,
      direction,
      enemy.position,
      t,
    );
    if (perpDistance <= NEATENSTEIN_ENEMY_HIT_RADIUS_CELLS) {
      hitType = 'enemy';
      hitDistance = t;
      hitEnemyIndex = index;
    }
  }

  const hit: Vector2 = {
    x: origin.x + direction.x * hitDistance,
    y: origin.y + direction.y * hitDistance,
  };

  const tracer: TracerState = {
    origin,
    direction,
    hit,
    distance: hitDistance,
    hitType,
    durationMs: NEATENSTEIN_TRACER_DURATION_MS,
    color: NEATENSTEIN_BEAM_COLOR,
  };

  let nextState = consumeAmmo(state);
  nextState = { ...nextState, tracers: [...nextState.tracers, tracer] };

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
    };
    nextState = { ...nextState, impacts: [...nextState.impacts, impact] };
  }

  if (hitType === 'enemy' && hitEnemyIndex >= 0) {
    nextState = applyEnemyDamage(nextState, hitEnemyIndex);
  }

  return { state: nextState, fired: true, tracer };
}

/**
 * Project a point onto the beam ray and return the signed distance from origin.
 *
 * Negative values mean the point is behind the beam origin and should be
 * ignored.
 *
 * @param origin - Beam origin.
 * @param direction - Normalized beam direction.
 * @param point - Enemy position to test.
 * @returns Signed distance along the beam from origin to the closest approach.
 */
function projectOntoBeam(
  origin: Vector2,
  direction: Vector2,
  point: Vector2,
): number {
  return (
    (point.x - origin.x) * direction.x + (point.y - origin.y) * direction.y
  );
}

/**
 * Compute the perpendicular distance from a point to the beam ray.
 *
 * @param origin - Beam origin.
 * @param direction - Normalized beam direction.
 * @param point - Enemy position to test.
 * @param t - Distance along the beam to the closest approach.
 * @returns Euclidean distance from the point to its closest point on the beam.
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
 * Apply beam damage to the enemy at the given index.
 *
 * Health is clamped at zero and the kill counter increments only when an
 * enemy transitions from alive to dead in this shot.
 *
 * @param state - Snapshot with the tracer already appended.
 * @param enemyIndex - Index into {@link GameState.enemies}.
 * @returns New snapshot with updated enemy health and kill count.
 */
function applyEnemyDamage(state: GameState, enemyIndex: number): GameState {
  const enemy = state.enemies[enemyIndex];
  const newHealth = Math.max(0, enemy.health - NEATENSTEIN_BEAM_DAMAGE);
  const newEnemies = state.enemies.map((existing, index) =>
    index === enemyIndex ? { ...existing, health: newHealth } : existing,
  );
  const newKills = newHealth === 0 ? state.kills + 1 : state.kills;

  return { ...state, enemies: newEnemies, kills: newKills };
}
