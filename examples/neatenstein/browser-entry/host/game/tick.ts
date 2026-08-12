/**
 * Deterministic fixed-timestep game tick for the Neatenstein host-side
 * simulation.
 *
 * This module owns the top-level world update pipeline: one input snapshot in,
 * one immutable {@link GameState} out, advanced by one simulation timestep.
 *
 * The tick function deliberately remains thin. Movement, combat, collision,
 * dashing, and episode progression are delegated to focused subsystem helpers
 * so each part can be tested independently.
 *
 * @module
 */

import {
  buildNeatensteinMap,
  createCollisionMap,
  type CollisionMap,
} from '../../renderer/map';
import { applyEnemyDamage, fireBolt } from './combat';
import {
  NEATENSTEIN_AMMO_PICKUP_COLLECTION_RADIUS_CELLS,
  NEATENSTEIN_AMMO_PICKUP_LIFETIME_MS,
  NEATENSTEIN_BOLT_HIT_RADIUS_CELLS,
  NEATENSTEIN_BOLT_MAX_RANGE_CELLS,
  NEATENSTEIN_BOLT_SPEED_CELLS_PER_SECOND,
  NEATENSTEIN_BOLT_TRAVEL_DURATION_MS,
  NEATENSTEIN_CONTACT_IFRAME_MS,
  NEATENSTEIN_ENEMY_BOLT_HIT_RADIUS_CELLS,
  NEATENSTEIN_ENEMY_BOLT_LIFETIME_MS,
  NEATENSTEIN_ENEMY_BOLT_MAX_RANGE_CELLS,
  NEATENSTEIN_ENEMY_BOLT_SPEED_CELLS_PER_SECOND,
  NEATENSTEIN_ENEMY_IMPACT_MAX_CONCURRENT,
  NEATENSTEIN_FIXED_TIMESTEP_MS,
  NEATENSTEIN_GUN_RECOIL_DECAY_PX_PER_SECOND,
  NEATENSTEIN_GUN_RECOIL_MAX_OFFSET_PX,
  NEATENSTEIN_MAP_SIZE,
  NEATENSTEIN_PLAYER_MAX_AMMO,
  NEATENSTEIN_PLAYER_MAX_HEALTH,
  NEATENSTEIN_SPAWN_CENTER_X,
  NEATENSTEIN_SPAWN_CENTER_Y,
} from './constants';
import { updateEpisode } from './episode';
import { updatePlayerMovement } from './movement';
import { applyDamage, applyDash, createGameState, restoreAmmo } from './state';
import type {
  BoltState,
  EnemyBoltState,
  EnemyImpactSpot,
  EnemyState,
  GameState,
  GunState,
  ImpactSpot,
  Vector2,
} from './types';
import { NEATENSTEIN_ENEMY_IMPACT_LIFETIME_MS } from '../../constants';
import { MAX_TURN_RATE } from '../../harness/neat-io-config';

export { createGameState };
export {
  NEATENSTEIN_BOLT_SPEED_CELLS_PER_SECOND,
  NEATENSTEIN_BOLT_TRAVEL_DURATION_MS,
  NEATENSTEIN_ENEMY_BOLT_SPEED_CELLS_PER_SECOND,
  NEATENSTEIN_FIXED_TIMESTEP_MS,
};

/**
 * Normalized input snapshot consumed by {@link gameTick}.
 *
 * The shape is deliberately clone-safe so the same snapshot can be forwarded
 * from the host input router, across the worker boundary, and replayed for
 * deterministic simulation evaluation.
 */
export interface GameTickInputSnapshot {
  /** Player-local movement vector; `x=-1` left, `x=1` right, `y=1` forward. */
  move: Vector2;

  /** Horizontal look delta to apply this tick, in radians. */
  lookDelta: number;

  /** `true` when the fire action is requested or held this tick. */
  fire: boolean;

  /** `true` when the dash action is requested this tick. */
  dash: boolean;
}

/**
 * Fully normalized input used internally by the tick pipeline.
 */
interface NormalizedGameTickInputSnapshot {
  /** Sanitized player-local movement vector. */
  move: Vector2;

  /** Finite horizontal look delta in radians. */
  lookDelta: number;

  /** Whether fire is active this tick. */
  fire: boolean;

  /** Whether dash is active this tick. */
  dash: boolean;
}

/**
 * One-entry collision-map cache used only when callers omit an explicit map.
 *
 * Runtime callers should still pass a prebuilt {@link CollisionMap}; this cache
 * prevents accidental per-tick map rebuilding in tests or fallback paths while
 * keeping the module simple and deterministic.
 */
let cachedCollisionSeed: number | null = null;
let cachedCollisionMap: CollisionMap | null = null;

/**
 * Resolve the collision map for a tick.
 *
 * If the caller provides a map, it is used directly. Otherwise, a deterministic
 * map is built from the state seed and cached for later calls with the same
 * seed.
 *
 * @param state - Current game state.
 * @param collisionMap - Optional caller-provided collision map.
 * @returns Collision map for this tick.
 */
function resolveCollisionMap(
  state: GameState,
  collisionMap?: CollisionMap,
): CollisionMap {
  if (collisionMap) {
    return collisionMap;
  }

  if (cachedCollisionMap && cachedCollisionSeed === state.seed) {
    return cachedCollisionMap;
  }

  const flatMap = buildNeatensteinMap(state.seed);
  const nextCollisionMap = createCollisionMap(flatMap, NEATENSTEIN_MAP_SIZE);

  cachedCollisionSeed = state.seed;
  cachedCollisionMap = nextCollisionMap;

  return nextCollisionMap;
}

/**
 * Resolve a safe tick duration.
 *
 * The simulation is designed around a fixed positive timestep. If an invalid
 * value reaches this boundary, falling back to the canonical fixed timestep is
 * safer than propagating `NaN`, infinities, or negative time into subsystems.
 *
 * @param dtMs - Candidate tick duration in milliseconds.
 * @returns Positive finite tick duration.
 */
function resolveTickDurationMs(dtMs: number): number {
  return Number.isFinite(dtMs) && dtMs > 0
    ? dtMs
    : NEATENSTEIN_FIXED_TIMESTEP_MS;
}

/**
 * Convert an unknown or partial movement vector into finite numeric components.
 *
 * Missing and non-finite components are treated as zero.
 *
 * @param move - Optional movement vector from the input snapshot.
 * @returns Sanitized movement vector.
 */
function normalizeMoveVector(move?: Partial<Vector2>): Vector2 {
  return {
    x: typeof move?.x === 'number' && Number.isFinite(move.x) ? move.x : 0,
    y: typeof move?.y === 'number' && Number.isFinite(move.y) ? move.y : 0,
  };
}

/**
 * Normalize a partial tick input snapshot.
 *
 * Missing fields default to neutral input. Non-finite numeric fields are
 * ignored so malformed input cannot poison the deterministic game state.
 *
 * @param snapshot - Partial input snapshot supplied by host or worker.
 * @returns Fully normalized tick input.
 */
function normalizeGameTickInput(
  snapshot: Partial<GameTickInputSnapshot>,
): NormalizedGameTickInputSnapshot {
  return {
    move: normalizeMoveVector(snapshot.move),
    lookDelta:
      typeof snapshot.lookDelta === 'number' &&
      Number.isFinite(snapshot.lookDelta)
        ? snapshot.lookDelta
        : 0,
    fire: snapshot.fire === true,
    dash: snapshot.dash === true,
  };
}

/**
 * Advance the deterministic game state by one simulation timestep.
 *
 * Pipeline order:
 *
 * 1. Resolve the timestep, collision map, and normalized input.
 * 2. Advance episode systems such as spawning, timers, and contact damage.
 * 3. Apply player look.
 * 4. Apply dash if requested.
 * 5. Apply player movement against collision.
 * 6. Move active player plasma bolts and remove any that hit a wall or leave
 *    the map.
 * 7. Age wall-impact spots and enemy-impact spots.
 * 8. Move active enemy bolts, check player proximity, and apply damage.
 * 9. Decay gun recoil toward zero.
 * 10. Fire a traveling plasma bolt if requested.
 *
 * Bolts move before firing so newly spawned bolts start at the muzzle and are
 * not advanced until the following tick.
 *
 * @param state - Snapshot before this tick.
 * @param snapshot - Partial input snapshot for this tick.
 * @param collisionMap - Optional collision map; when omitted a deterministic
 *   map is resolved from {@link GameState.seed}.
 * @param dtMs - Duration of this tick in milliseconds.
 * @returns A new immutable {@link GameState} advanced by the resolved timestep.
 *
 * @example
 * ```ts
 * const state = createGameState({ seed: 7 });
 * const next = gameTick(state, {
 *   move: { x: 0, y: 1 },
 *   lookDelta: 0.05,
 *   fire: true,
 *   dash: false,
 * });
 * ```
 */
export function gameTick(
  state: GameState,
  snapshot: Partial<GameTickInputSnapshot>,
  collisionMap?: CollisionMap,
  dtMs: number = NEATENSTEIN_FIXED_TIMESTEP_MS,
): GameState {
  const resolvedDtMs = resolveTickDurationMs(dtMs);
  const input = normalizeGameTickInput(snapshot);
  const map = resolveCollisionMap(state, collisionMap);

  // Step 1: Advance deterministic world/episode systems.
  let next = updateEpisode(state, resolvedDtMs, map);

  // Step 2: Apply yaw input before movement so movement uses the new facing.
  next = applyLook(next, input.lookDelta);

  // Step 3: Apply dash before movement so movement can consume updated player
  // state such as dash velocity, cooldown, or flags.
  if (input.dash) {
    next = applyDash(next);
  }

  // Step 4: Resolve player movement against the collision map.
  next = updatePlayerMovement(
    next,
    snapshotToMovement(input.move),
    map,
    resolvedDtMs,
  );

  // Step 5: Move active plasma bolts, apply enemy damage on hit, and cull
  // inactive ones.
  let boltHitEnemy = false;
  const updatedBolts = updateBolts(
    next.bolts ?? [],
    resolvedDtMs,
    next.simTimeMs,
    map,
    next.enemies,
  );
  for (const bolt of updatedBolts) {
    if (
      !bolt.active &&
      bolt.hitEnemyIndex !== undefined &&
      bolt.hitEnemyIndex >= 0
    ) {
      const enemy = next.enemies[bolt.hitEnemyIndex];
      if (enemy) {
        boltHitEnemy = true;
        const enemyImpact: EnemyImpactSpot = {
          position: { ...enemy.position },
          createdAtMs: next.simTimeMs,
          lifetimeMs: NEATENSTEIN_ENEMY_IMPACT_LIFETIME_MS,
          boltTravelTimeMs: 0,
        };
        next = {
          ...next,
          enemyImpacts: [...(next.enemyImpacts ?? []), enemyImpact].slice(
            -NEATENSTEIN_ENEMY_IMPACT_MAX_CONCURRENT,
          ),
        };
        next = applyEnemyDamage(next, bolt.hitEnemyIndex);
      }
    }
  }
  next = {
    ...next,
    bolts: updatedBolts.filter((bolt) => bolt.active),
    impacts: ageImpacts(next.impacts, resolvedDtMs),
    enemyImpacts: ageEnemyImpacts(next.enemyImpacts ?? [], resolvedDtMs),
    gun: decayGunRecoil(
      next.gun ?? { recoilOffset: 0, firing: false },
      resolvedDtMs,
    ),
    lastShotHit: boltHitEnemy,
  };

  // Step 5b: Move active enemy bolts, check player proximity, apply damage,
  // and cull inactive ones.
  const enemyBoltResult = updateEnemyBolts(
    next.enemyBolts ?? [],
    resolvedDtMs,
    next.simTimeMs,
    map,
    next,
  );
  next = enemyBoltResult.state;
  next = {
    ...next,
    enemyBolts: enemyBoltResult.bolts.filter((bolt) => bolt.active),
  };

  // Step 5c: Respawn the hero at the map center if health has been depleted.
  if (next.player.health <= 0) {
    next = {
      ...next,
      player: {
        ...next.player,
        position: {
          x: NEATENSTEIN_SPAWN_CENTER_X,
          y: NEATENSTEIN_SPAWN_CENTER_Y,
        },
        previousPosition: {
          x: NEATENSTEIN_SPAWN_CENTER_X,
          y: NEATENSTEIN_SPAWN_CENTER_Y,
        },
        health: NEATENSTEIN_PLAYER_MAX_HEALTH,
        ammo: NEATENSTEIN_PLAYER_MAX_AMMO,
        dashTimeRemainingMs: 0,
        dashCooldownMs: 0,
        contactIFrameMs: 0,
      },
      deaths: (next.deaths ?? 0) + 1,
    };
  }

  // Step 5d: Update ammo pickups — collect by proximity and expire by lifetime.
  next = updateAmmoPickups(next, next.simTimeMs);

  // Step 6: Fire after updating bolts so newly spawned bolts start at the
  // muzzle and are advanced on the following tick.
  if (input.fire) {
    const fireResult = fireBolt(next);
    next = fireResult.state;

    if (fireResult.fired) {
      next = {
        ...next,
        gun: {
          ...(next.gun ?? { recoilOffset: 0, firing: false }),
          recoilOffset: NEATENSTEIN_GUN_RECOIL_MAX_OFFSET_PX,
          firing: true,
        },
      };
    }
  }

  return next;
}

/**
 * Update ammo pickups for one tick: collect by proximity and expire by lifetime.
 *
 * Active pickups within {@link NEATENSTEIN_AMMO_PICKUP_COLLECTION_RADIUS_CELLS}
 * of the player are collected — the player's ammo is restored via
 * {@link restoreAmmo} and the pickup is marked inactive. Pickups whose
 * lifetime has elapsed are also marked inactive. Inactive pickups are filtered
 * out of the returned state.
 *
 * @param state - Snapshot before the pickup update.
 * @param simTimeMs - Current simulation time in milliseconds.
 * @returns New snapshot with collected/expired pickups removed and ammo
 *   restored for any collected pickups.
 */
export function updateAmmoPickups(
  state: GameState,
  simTimeMs: number,
): GameState {
  const pickups = state.ammoPickups ?? [];
  if (pickups.length === 0) {
    return state;
  }

  let ammoGain = 0;
  const updatedPickups = pickups.map((pickup) => {
    if (!pickup.active) {
      return pickup;
    }

    const lifetimeMs = pickup.lifetimeMs ?? NEATENSTEIN_AMMO_PICKUP_LIFETIME_MS;
    const expired = simTimeMs - pickup.createdAtMs >= lifetimeMs;
    if (expired) {
      return { ...pickup, active: false };
    }

    const dx = pickup.position.x - state.player.position.x;
    const dy = pickup.position.y - state.player.position.y;
    const dist = Math.hypot(dx, dy);
    if (dist <= NEATENSTEIN_AMMO_PICKUP_COLLECTION_RADIUS_CELLS) {
      ammoGain += pickup.amount;
      return { ...pickup, active: false };
    }

    return pickup;
  });

  const activePickups = updatedPickups.filter((pickup) => pickup.active);

  let next: GameState = {
    ...state,
    ammoPickups: activePickups,
  };

  if (ammoGain > 0) {
    next = restoreAmmo(next, ammoGain);
    if (next.telemetry) {
      next = {
        ...next,
        telemetry: {
          ...next.telemetry,
          ammoPickupsCollected: (next.telemetry.ammoPickupsCollected ?? 0) + 1,
        },
      };
    }
  }

  return next;
}

/**
 * Rotate the player by a yaw delta for one tick.
 *
 * Non-finite or zero deltas leave the state unchanged. The `lookDelta` is
 * clamped to {@link MAX_TURN_RATE} so no caller can exceed the canonical
 * turn-rate cap, regardless of whether the input originated from the NEAT
 * network mapping or a direct call.
 *
 * @param state - Game state before look input.
 * @param lookDelta - Horizontal yaw delta in radians.
 * @returns Updated state, or the original state if no rotation is needed.
 */
function applyLook(state: GameState, lookDelta: number): GameState {
  if (!Number.isFinite(lookDelta) || lookDelta === 0) {
    return state;
  }

  const clampedDelta =
    lookDelta > MAX_TURN_RATE
      ? MAX_TURN_RATE
      : lookDelta < -MAX_TURN_RATE
        ? -MAX_TURN_RATE
        : lookDelta;

  return {
    ...state,
    player: {
      ...state.player,
      angleRad: state.player.angleRad + clampedDelta,
    },
  };
}

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
  const dtSeconds = resolvedDtMs / 1000;

  return bolts
    .filter((bolt) => bolt.active)
    .map((bolt) => {
      const step = bolt.speedCellsPerSecond * dtSeconds;
      const enemyImpact =
        enemies && enemies.length > 0
          ? findBoltEnemyImpact(bolt, step, enemies)
          : null;
      const nextPosition: Vector2 = enemyImpact
        ? {
            x: bolt.position.x + bolt.direction.x * enemyImpact[0],
            y: bolt.position.y + bolt.direction.y * enemyImpact[0],
          }
        : {
            x: bolt.position.x + bolt.direction.x * step,
            y: bolt.position.y + bolt.direction.y * step,
          };
      const outOfBounds =
        nextPosition.x < 0 ||
        nextPosition.x >= NEATENSTEIN_MAP_SIZE ||
        nextPosition.y < 0 ||
        nextPosition.y >= NEATENSTEIN_MAP_SIZE;
      const hitWall = collisionMap
        ? collisionMap.isSolid(
            Math.floor(nextPosition.x),
            Math.floor(nextPosition.y),
          )
        : false;
      const distanceTraveled =
        bolt.origin &&
        Number.isFinite(bolt.origin.x) &&
        Number.isFinite(bolt.origin.y)
          ? Math.hypot(
              nextPosition.x - bolt.origin.x,
              nextPosition.y - bolt.origin.y,
            )
          : 0;
      const beyondMaxRange =
        distanceTraveled >= NEATENSTEIN_BOLT_MAX_RANGE_CELLS;
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
      const nextPositionFinal = hitEnemy
        ? nextPosition
        : movementStopped
          ? bolt.position
          : nextPosition;

      return {
        ...bolt,
        position: nextPositionFinal,
        active,
        hitEnemyIndex: enemyImpact ? enemyImpact[1] : bolt.hitEnemyIndex,
      };
    });
}

/**
 * Result of advancing enemy bolts for one tick.
 */
interface UpdateEnemyBoltsResult {
  /** Updated game state (damage applied if any bolt hit the player). */
  state: GameState;
  /** Updated enemy bolt array (inactive bolts still present for culling). */
  bolts: EnemyBoltState[];
}

/**
 * Advance active enemy plasma bolts by one tick.
 *
 * Each active bolt is moved along its direction by `speed * dt`. Bolts that
 * come within {@link NEATENSTEIN_ENEMY_BOLT_HIT_RADIUS_CELLS} of the player
 * register a hit: damage is applied via {@link applyDamage} (which respects
 * dash i-frames and contact i-frames), the bolt is deactivated, and the
 * player's contact i-frame timer is set to
 * {@link NEATENSTEIN_CONTACT_IFRAME_MS}. Bolts that hit a wall, leave the map,
 * exceed their maximum range, or expire after
 * {@link NEATENSTEIN_ENEMY_BOLT_LIFETIME_MS} are also deactivated.
 *
 * @param bolts - Active enemy bolt snapshots before this tick.
 * @param dtMs - Elapsed time in milliseconds.
 * @param currentTimeMs - Current simulation time in milliseconds.
 * @param collisionMap - Optional collision map for wall-hit detection.
 * @param state - Current game state (used for player position and damage).
 * @returns Updated state and bolt array.
 */
export function updateEnemyBolts(
  bolts: EnemyBoltState[],
  dtMs: number,
  currentTimeMs: number,
  collisionMap: CollisionMap | undefined,
  state: GameState,
): UpdateEnemyBoltsResult {
  const resolvedDtMs = resolveTickDurationMs(dtMs);
  const dtSeconds = resolvedDtMs / 1000;
  let nextState = state;

  const updatedBolts = bolts
    .filter((bolt) => bolt.active)
    .map((bolt) => {
      const step = bolt.speedCellsPerSecond * dtSeconds;
      const nextPosition: Vector2 = {
        x: bolt.position.x + bolt.direction.x * step,
        y: bolt.position.y + bolt.direction.y * step,
      };

      // Check wall collision.
      const hitWall = collisionMap
        ? collisionMap.isSolid(
            Math.floor(nextPosition.x),
            Math.floor(nextPosition.y),
          )
        : false;

      // Check out of bounds.
      const outOfBounds =
        nextPosition.x < 0 ||
        nextPosition.x >= NEATENSTEIN_MAP_SIZE ||
        nextPosition.y < 0 ||
        nextPosition.y >= NEATENSTEIN_MAP_SIZE;

      // Check distance traveled from origin.
      const distanceTraveled =
        bolt.origin &&
        Number.isFinite(bolt.origin.x) &&
        Number.isFinite(bolt.origin.y)
          ? Math.hypot(
              nextPosition.x - bolt.origin.x,
              nextPosition.y - bolt.origin.y,
            )
          : 0;
      const beyondMaxRange =
        distanceTraveled >= NEATENSTEIN_ENEMY_BOLT_MAX_RANGE_CELLS;

      // Check lifetime expiry.
      const elapsedMs = Math.max(0, currentTimeMs - bolt.createdAtMs);
      const lifetimeExpired = elapsedMs >= NEATENSTEIN_ENEMY_BOLT_LIFETIME_MS;

      // Check player proximity.
      const playerDist = Math.hypot(
        nextPosition.x - nextState.player.position.x,
        nextPosition.y - nextState.player.position.y,
      );
      const hitPlayer =
        !hitWall &&
        !outOfBounds &&
        !beyondMaxRange &&
        playerDist <= NEATENSTEIN_ENEMY_BOLT_HIT_RADIUS_CELLS;

      const movementStopped = outOfBounds || hitWall || beyondMaxRange;
      const active = !lifetimeExpired && !hitPlayer;
      const nextPositionFinal = movementStopped ? bolt.position : nextPosition;

      if (hitPlayer && !bolt.hitPlayer) {
        nextState = applyDamage(nextState, bolt.damage);
        // Grant the same contact i-frame window as melee contact damage
        // so subsequent bolts and contact damage are blocked for 500ms.
        nextState = {
          ...nextState,
          player: {
            ...nextState.player,
            contactIFrameMs: NEATENSTEIN_CONTACT_IFRAME_MS,
          },
        };
      }

      return {
        ...bolt,
        position: nextPositionFinal,
        active,
        hitPlayer: hitPlayer || bolt.hitPlayer,
      };
    });

  return { state: nextState, bolts: updatedBolts };
}

/**
 * Decay gun recoil toward zero over time.
 *
 * The recoil offset is reduced by the configured decay rate each second and
 * clamped so it never becomes negative or exceeds the maximum offset.
 *
 * @param gun - Gun overlay state before decay.
 * @param dtMs - Elapsed time in milliseconds.
 * @returns Updated gun state with decayed recoil offset.
 */
export function decayGunRecoil(gun: GunState, dtMs: number): GunState {
  const resolvedDtMs = resolveTickDurationMs(dtMs);
  const decayPixels =
    NEATENSTEIN_GUN_RECOIL_DECAY_PX_PER_SECOND * (resolvedDtMs / 1000);
  const nextOffset = Math.max(0, gun.recoilOffset - decayPixels);

  return {
    ...gun,
    recoilOffset: Math.min(nextOffset, NEATENSTEIN_GUN_RECOIL_MAX_OFFSET_PX),
    firing: false,
  };
}

/**
 * Age active wall-impact spots by one tick and remove any that have expired.
 *
 * Impact spots are immutable snapshots; each surviving impact gets its
 * remaining lifetime reduced by the elapsed timestep.
 *
 * @param impacts - Active wall-impact snapshots before this tick.
 * @param dtMs - Elapsed time in milliseconds.
 * @returns New array of impact spots still visible after aging.
 */
export function ageImpacts(impacts: ImpactSpot[], dtMs: number): ImpactSpot[] {
  const resolvedDtMs = resolveTickDurationMs(dtMs);

  return impacts
    .map((impact) => ({
      ...impact,
      lifetimeMs: impact.lifetimeMs - resolvedDtMs,
    }))
    .filter((impact) => impact.lifetimeMs > 0);
}

/**
 * Age active enemy-impact spots by one tick and remove any that have expired.
 *
 * Follows the same deterministic pattern as {@link ageImpacts}: each surviving
 * spot gets its remaining lifetime reduced by the elapsed timestep (derived
 * from `simTimeMs`, never `Date.now()`), and spots with lifetime ≤ 0 are
 * removed.
 *
 * @param impacts - Active enemy-impact snapshots before this tick.
 * @param dtMs - Elapsed time in milliseconds.
 * @returns New array of enemy-impact spots still visible after aging.
 */
export function ageEnemyImpacts(
  impacts: EnemyImpactSpot[],
  dtMs: number,
): EnemyImpactSpot[] {
  const resolvedDtMs = resolveTickDurationMs(dtMs);

  return impacts
    .map((impact) => ({
      ...impact,
      lifetimeMs: impact.lifetimeMs - resolvedDtMs,
    }))
    .filter((impact) => impact.lifetimeMs > 0);
}

/**
 * Convert a movement vector into the directional boolean intent used by the
 * keyboard-oriented movement helper.
 *
 * A non-zero axis value is treated as pressed on that side. Diagonal vectors
 * preserve both components so {@link updatePlayerMovement} can normalize the
 * final world-space step.
 *
 * @param move - Sanitized movement vector.
 * @returns Directional movement intent.
 */
function snapshotToMovement(move?: Vector2): {
  forward: boolean;
  backward: boolean;
  left: boolean;
  right: boolean;
} {
  const normalizedMove = normalizeMoveVector(move);

  return {
    forward: normalizedMove.y > 0,
    backward: normalizedMove.y < 0,
    left: normalizedMove.x < 0,
    right: normalizedMove.x > 0,
  };
}
