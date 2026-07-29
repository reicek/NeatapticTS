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
import { fireBolt } from './combat';
import {
  NEATENSTEIN_BOLT_MAX_RANGE_CELLS,
  NEATENSTEIN_BOLT_TRAVEL_DURATION_MS,
  NEATENSTEIN_FIXED_TIMESTEP_MS,
  NEATENSTEIN_GUN_RECOIL_DECAY_PX_PER_SECOND,
  NEATENSTEIN_GUN_RECOIL_MAX_OFFSET_PX,
  NEATENSTEIN_MAP_SIZE,
} from './constants';
import { updateEpisode } from './episode';
import { updatePlayerMovement } from './movement';
import { applyDash, createGameState } from './state';
import type {
  BoltState,
  GameState,
  GunState,
  ImpactSpot,
  Vector2,
} from './types';

export { createGameState };
export {
  NEATENSTEIN_BOLT_SPEED_CELLS_PER_SECOND,
  NEATENSTEIN_BOLT_TRAVEL_DURATION_MS,
  NEATENSTEIN_FIXED_TIMESTEP_MS,
} from './constants';

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
 * 6. Move active plasma bolts and remove any that hit a wall or leave the map.
 * 7. Age wall-impact spots.
 * 8. Decay gun recoil toward zero.
 * 9. Fire a traveling plasma bolt if requested.
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
  let next = updateEpisode(state, resolvedDtMs);

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

  // Step 5: Move active plasma bolts and cull inactive ones.
  next = {
    ...next,
    bolts: updateBolts(
      next.bolts ?? [],
      resolvedDtMs,
      next.simTimeMs,
      map,
    ).filter((bolt) => bolt.active),
    impacts: ageImpacts(next.impacts, resolvedDtMs),
    gun: decayGunRecoil(next.gun ?? { recoilOffset: 0 }, resolvedDtMs),
  };

  // Step 6: Fire after updating bolts so newly spawned bolts start at the
  // muzzle and are advanced on the following tick.
  if (input.fire) {
    const fireResult = fireBolt(next);
    next = fireResult.state;

    if (fireResult.fired) {
      next = {
        ...next,
        gun: {
          ...next.gun,
          recoilOffset: NEATENSTEIN_GUN_RECOIL_MAX_OFFSET_PX,
        },
      };
    }
  }

  return next;
}

/**
 * Rotate the player by a yaw delta for one tick.
 *
 * Non-finite or zero deltas leave the state unchanged.
 *
 * @param state - Game state before look input.
 * @param lookDelta - Horizontal yaw delta in radians.
 * @returns Updated state, or the original state if no rotation is needed.
 */
function applyLook(state: GameState, lookDelta: number): GameState {
  if (!Number.isFinite(lookDelta) || lookDelta === 0) {
    return state;
  }

  return {
    ...state,
    player: {
      ...state.player,
      angleRad: state.player.angleRad + lookDelta,
    },
  };
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
 * @returns New array of bolts after movement and deactivation.
 */
export function updateBolts(
  bolts: BoltState[],
  dtMs: number,
  currentTimeMs: number,
  collisionMap?: CollisionMap,
): BoltState[] {
  const resolvedDtMs = resolveTickDurationMs(dtMs);
  const dtSeconds = resolvedDtMs / 1000;

  return bolts
    .filter((bolt) => bolt.active)
    .map((bolt) => {
      const step = bolt.speedCellsPerSecond * dtSeconds;
      const nextPosition: Vector2 = {
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
      const elapsedMs = Math.max(0, currentTimeMs - bolt.createdAtMs);
      const travelExpired = elapsedMs >= NEATENSTEIN_BOLT_TRAVEL_DURATION_MS;
      const movementStopped = outOfBounds || hitWall || beyondMaxRange;
      const active = !travelExpired;

      return {
        ...bolt,
        position: movementStopped ? bolt.position : nextPosition,
        active,
      };
    });
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
