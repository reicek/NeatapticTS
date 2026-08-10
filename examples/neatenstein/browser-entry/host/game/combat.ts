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
  NEATENSTEIN_ENEMY_IMPACT_LIFETIME_MS,
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
  NEATENSTEIN_ENEMY_BOLT_DAMAGE,
  NEATENSTEIN_ENEMY_BOLT_MAX_RANGE_CELLS,
  NEATENSTEIN_ENEMY_BOLT_SPEED_CELLS_PER_SECOND,
  NEATENSTEIN_ENEMY_IMPACT_MAX_CONCURRENT,
  NEATENSTEIN_ENEMY_PUSHBACK_DISTANCE_CELLS,
  NEATENSTEIN_ENEMY_STUN_DURATION_MS,
  NEATENSTEIN_MUZZLE_OFFSET_CELLS,
  NEATENSTEIN_AMMO_PICKUP_AMOUNT,
} from './constants';
import { consumeAmmo } from './state';
import type {
  AmmoPickupState,
  BoltState,
  EnemyBoltState,
  EnemyImpactSpot,
  EpisodeTelemetry,
  GameState,
  ImpactSpot,
  Vector2,
} from './types';

/** Re-export the muzzle offset so tests can assert bolt spawn position. */
export { NEATENSTEIN_MUZZLE_OFFSET_CELLS } from './constants';

/** Re-export the bolt damage constant for test contracts. */
export { NEATENSTEIN_BOLT_DAMAGE } from './constants';

/**
 * Default zero-valued telemetry for a fresh episode.
 *
 * Used when a {@link GameState} does not yet carry a `telemetry` field, which
 * happens for states created before any combat function has run.
 *
 * @returns A new `EpisodeTelemetry` with all counters at zero.
 */
function createDefaultTelemetry(): EpisodeTelemetry {
  return { damageDealt: 0, shotsFired: 0, shotsHit: 0, aimMissRate: 0 };
}

/**
 * Recompute `aimMissRate` from the raw shot counters.
 *
 * @param telemetry - Telemetry with updated raw counters.
 * @returns Telemetry with `aimMissRate` synchronized to the current counters.
 */
function withAimMissRate(telemetry: EpisodeTelemetry): EpisodeTelemetry {
  const { shotsFired, shotsHit } = telemetry;
  const aimMissRate = shotsFired > 0 ? (shotsFired - shotsHit) / shotsFired : 0;
  return { ...telemetry, aimMissRate };
}

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
 * 5. The nearest valid enemy before the wall takes damage; a non-lethal hit
 *    also applies stun, pushback, and an {@link EnemyImpactSpot} mark.
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

    if (enemy.health <= 0 || enemy.active === false) {
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
    radius: NEATENSTEIN_BOLT_HIT_RADIUS_CELLS,
    hitEnemyIndex: hitEnemyIndex,
  };

  nextState = {
    ...nextState,
    bolts: [...(nextState.bolts ?? []), bolt],
  };

  // Increment shotsFired telemetry for every successful shot.
  const telemetryBeforeShot = nextState.telemetry ?? createDefaultTelemetry();
  nextState = {
    ...nextState,
    telemetry: withAimMissRate({
      ...telemetryBeforeShot,
      shotsFired: telemetryBeforeShot.shotsFired + 1,
    }),
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
    const enemy = state.enemies[hitEnemyIndex];
    const enemyImpact: EnemyImpactSpot = {
      position: { ...enemy.position },
      createdAtMs: state.simTimeMs,
      lifetimeMs: NEATENSTEIN_ENEMY_IMPACT_LIFETIME_MS,
      boltTravelTimeMs: NEATENSTEIN_BOLT_TRAVEL_DURATION_MS,
    };

    nextState = {
      ...nextState,
      enemyImpacts: [...(nextState.enemyImpacts ?? []), enemyImpact].slice(
        -NEATENSTEIN_ENEMY_IMPACT_MAX_CONCURRENT,
      ),
    };

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
 * transitions from alive to dead in this shot. On a non-lethal hit the enemy
 * receives a hit-stun (stunTimerMs set to
 * {@link NEATENSTEIN_ENEMY_STUN_DURATION_MS}) and is pushed back away from the
 * player by {@link NEATENSTEIN_ENEMY_PUSHBACK_DISTANCE_CELLS}, wall-checked.
 * Lethal hits skip stun and pushback (the enemy enters the death/de-rez path).
 *
 * While already stunned (stunTimerMs > 0) damage is skipped entirely,
 * preventing damage stacking during the stun window.
 *
 * @param state - Snapshot with the bolt already appended.
 * @param enemyIndex - Index into {@link GameState.enemies}.
 * @returns New snapshot with updated enemy health, stun timer, pushback
 *   position, and kill count.
 */
export function applyEnemyDamage(
  state: GameState,
  enemyIndex: number,
): GameState {
  const enemy = state.enemies[enemyIndex];

  // Invincibility: skip damage entirely while stunned.
  if ((enemy.stunTimerMs ?? 0) > 0) {
    return state;
  }

  const newHealth = Math.max(0, enemy.health - NEATENSTEIN_BOLT_DAMAGE);
  const killedByThisShot = newHealth === 0;

  // On non-lethal hits: apply hit-stun and pushback.
  let newPosition = { ...enemy.position };
  let stunTimerMs = 0;

  if (!killedByThisShot) {
    stunTimerMs = NEATENSTEIN_ENEMY_STUN_DURATION_MS;

    // Pushback: move enemy away from the player, wall-checked.
    const dx = enemy.position.x - state.player.position.x;
    const dy = enemy.position.y - state.player.position.y;
    const dist = Math.hypot(dx, dy);

    if (dist > 0) {
      const pushX =
        enemy.position.x +
        (dx / dist) * NEATENSTEIN_ENEMY_PUSHBACK_DISTANCE_CELLS;
      const pushY =
        enemy.position.y +
        (dy / dist) * NEATENSTEIN_ENEMY_PUSHBACK_DISTANCE_CELLS;

      // Wall-check: only apply pushback if the target cell is not solid.
      const flatMap = resolveCombatMap(state.seed);
      const cellX = Math.floor(pushX);
      const cellY = Math.floor(pushY);
      if (
        cellX >= 0 &&
        cellX < NEATENSTEIN_MAP_SIZE &&
        cellY >= 0 &&
        cellY < NEATENSTEIN_MAP_SIZE &&
        flatMap[cellY * NEATENSTEIN_MAP_SIZE + cellX] === 0
      ) {
        newPosition = { x: pushX, y: pushY };
      }
    }
  }

  const newEnemies = state.enemies.map((existing, index) =>
    index === enemyIndex
      ? { ...existing, health: newHealth, position: newPosition, stunTimerMs }
      : existing,
  );

  // Increment damageDealt and shotsHit telemetry for this successful hit.
  const damageThisShot = Math.min(NEATENSTEIN_BOLT_DAMAGE, enemy.health);
  const telemetryBeforeHit = state.telemetry ?? createDefaultTelemetry();
  const telemetryAfterHit = withAimMissRate({
    ...telemetryBeforeHit,
    damageDealt: telemetryBeforeHit.damageDealt + damageThisShot,
    shotsHit: telemetryBeforeHit.shotsHit + 1,
  });

  // On kill: spawn an ammo pickup at the enemy's death position.
  const newAmmoPickups = killedByThisShot
    ? [
        ...(state.ammoPickups ?? []),
        {
          position: { ...enemy.position },
          amount: NEATENSTEIN_AMMO_PICKUP_AMOUNT,
          active: true,
          createdAtMs: state.simTimeMs,
        } satisfies AmmoPickupState,
      ]
    : (state.ammoPickups ?? []);

  return {
    ...state,
    enemies: newEnemies,
    kills: killedByThisShot ? state.kills + 1 : state.kills,
    ammoPickups: newAmmoPickups,
    telemetry: telemetryAfterHit,
  };
}

/**
 * Input shape for spawning an enemy bolt from a hitscan event.
 *
 * Mirrors the relevant fields of {@link HitscanEvent} without importing the
 * controller module, keeping the combat module dependency-free.
 */
export interface FireEnemyBoltInput {
  /** World-space origin of the enemy's hitscan ray. */
  origin: Vector2;
  /** Normalized direction toward the player at fire time. */
  direction: Vector2;
  /** Damage applied on hit. Defaults to {@link NEATENSTEIN_ENEMY_BOLT_DAMAGE}. */
  damage?: number;
}

/**
 * Spawn a traveling enemy plasma bolt from a pre-computed hitscan event.
 *
 * The bolt inherits the enemy's computed origin and direction directly — no
 * additional RNG is used, keeping the simulation fully deterministic. The
 * maximum travel distance is capped at {@link NEATENSTEIN_ENEMY_BOLT_MAX_RANGE_CELLS}.
 *
 * @param input - Hitscan event data (origin, direction, optional damage).
 * @param simTimeMs - Current simulation time in milliseconds.
 * @returns A new active {@link EnemyBoltState} ready to be appended to the
 *   game state's `enemyBolts` array.
 */
export function fireEnemyBolt(
  input: FireEnemyBoltInput,
  simTimeMs: number,
): EnemyBoltState {
  return {
    position: { ...input.origin },
    direction: { ...input.direction },
    speedCellsPerSecond: NEATENSTEIN_ENEMY_BOLT_SPEED_CELLS_PER_SECOND,
    active: true,
    createdAtMs: simTimeMs,
    origin: { ...input.origin },
    targetDistance: NEATENSTEIN_ENEMY_BOLT_MAX_RANGE_CELLS,
    damage: input.damage ?? NEATENSTEIN_ENEMY_BOLT_DAMAGE,
    hitPlayer: false,
  };
}
