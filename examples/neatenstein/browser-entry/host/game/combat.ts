/**
 * Hitscan plasma-burst combat for the Neatenstein host-side simulation.
 *
 * This module owns the player's primary weapon. Gameplay remains deterministic
 * and hitscan: firing instantly resolves the nearest enemy or wall along the
 * aim ray, consumes ammo, applies damage, and emits visual tracer state.
 *
 * A true moving plasma projectile cannot be implemented in this file alone
 * because projectile travel requires persistent projectile state, per-tick
 * movement, collision updates, and renderer support. Instead, this module
 * enhances the existing hitscan weapon visually by emitting a short segmented
 * plasma trail along the resolved beam path. The result reads less like a
 * single laser line while preserving the existing combat contract.
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
  NEATENSTEIN_BEAM_COLOR,
  NEATENSTEIN_BEAM_DAMAGE,
  NEATENSTEIN_BEAM_MAX_RANGE_CELLS,
  NEATENSTEIN_ENEMY_HIT_RADIUS_CELLS,
  NEATENSTEIN_MUZZLE_OFFSET_CELLS,
  NEATENSTEIN_TRACER_DURATION_MS,
} from './constants';
import { consumeAmmo } from './state';
import type { GameState, ImpactSpot, TracerState, Vector2 } from './types';

/**
 * Number of additional visual trail segments emitted for each shot.
 *
 * These extra tracers are purely visual. Damage and impact resolution still
 * happen once, at the nearest valid hit.
 */
const NEATENSTEIN_PLASMA_TRAIL_SEGMENTS = 3;

/**
 * Fraction of the total hit distance covered by the primary plasma core.
 *
 * A shorter leading segment makes the shot read more like a bright plasma bolt
 * than a full-length laser line.
 */
const NEATENSTEIN_PLASMA_CORE_DISTANCE_RATIO = 0.42;

/**
 * Fractional lifetime multiplier applied to each trailing plasma segment.
 *
 * Later trail segments live for a shorter time, creating a quick fading tail.
 */
const NEATENSTEIN_PLASMA_TRAIL_DURATION_FALLOFF = 0.72;

/**
 * Minimum visual segment distance in world cells.
 *
 * Prevents extremely close shots from producing zero-length visual tracers.
 */
const NEATENSTEIN_PLASMA_MIN_SEGMENT_DISTANCE_CELLS = 0.05;

/** Result of attempting to fire the neon beam/plasma burst. */
export interface FireNeonBeamResult {
  /** Snapshot after the shot: ammo consumed, tracers appended, damage applied. */
  state: GameState;

  /** `true` when a shot was actually fired this frame. */
  fired: boolean;

  /** Primary visible tracer for this frame, or `null` when the weapon did not fire. */
  tracer: TracerState | null;
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
 * Return whether a number is finite.
 *
 * @param value - Candidate number.
 * @returns Whether the value is finite.
 */
function isFiniteNumber(value: number): boolean {
  return Number.isFinite(value);
}

/**
 * Resolve a finite player angle.
 *
 * Invalid angles fall back to `0` so malformed state cannot propagate `NaN`
 * into ray direction, enemy projection, or tracer positions.
 *
 * @param angleRad - Candidate player angle in radians.
 * @returns Finite angle in radians.
 */
function resolvePlayerAngle(angleRad: number): number {
  return isFiniteNumber(angleRad) ? angleRad : 0;
}

/**
 * Resolve a finite positive beam range.
 *
 * @returns Safe maximum weapon range in world cells.
 */
function resolveBeamMaxRange(): number {
  return isFiniteNumber(NEATENSTEIN_BEAM_MAX_RANGE_CELLS) &&
    NEATENSTEIN_BEAM_MAX_RANGE_CELLS > 0
    ? NEATENSTEIN_BEAM_MAX_RANGE_CELLS
    : 1;
}

/**
 * Clamp a value to the inclusive `[min, max]` range.
 *
 * @param value - Value to clamp.
 * @param min - Lower bound.
 * @param max - Upper bound.
 * @returns Clamped value.
 */
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

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
 * Return whether a DDA wall distance represents a usable wall hit.
 *
 * @param distance - Candidate wall distance.
 * @param maxRange - Maximum weapon range.
 * @returns Whether the wall is finite, in front of the muzzle, and in range.
 */
function isWallHitInRange(distance: number, maxRange: number): boolean {
  return isFiniteNumber(distance) && distance >= 0 && distance <= maxRange;
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
 * Create a tracer segment between two distances along the same ray.
 *
 * The renderer consumes tracers as origin/hit pairs, so each plasma trail
 * segment is represented as its own short tracer.
 *
 * @param origin - Shot muzzle origin.
 * @param direction - Normalized shot direction.
 * @param startDistance - Segment start distance from the muzzle.
 * @param endDistance - Segment end distance from the muzzle.
 * @param hitType - Type of final combat hit for color/renderer semantics.
 * @param durationMs - Segment lifetime in milliseconds.
 * @returns Tracer state for one plasma segment.
 */
function createTracerSegment(
  origin: Vector2,
  direction: Vector2,
  startDistance: number,
  endDistance: number,
  hitType: 'wall' | 'enemy',
  durationMs: number,
): TracerState {
  const safeStart = Math.max(0, startDistance);
  const safeEnd = Math.max(
    safeStart + NEATENSTEIN_PLASMA_MIN_SEGMENT_DISTANCE_CELLS,
    endDistance,
  );

  const segmentOrigin = pointAlongRay(origin, direction, safeStart);
  const segmentHit = pointAlongRay(origin, direction, safeEnd);

  return {
    origin: segmentOrigin,
    direction,
    hit: segmentHit,
    distance: safeEnd - safeStart,
    hitType,
    durationMs,
    color: NEATENSTEIN_BEAM_COLOR,
  };
}

/**
 * Create a plasma-like visual trail for a hitscan shot.
 *
 * The first tracer is the primary bright plasma core. Additional tracers are
 * shorter fading segments behind it, giving the renderer multiple overlapping
 * glowing strokes without changing combat resolution.
 *
 * @param origin - Shot origin.
 * @param direction - Normalized shot direction.
 * @param hitDistance - Resolved combat hit distance.
 * @param hitType - Resolved combat hit type.
 * @returns Ordered tracer list, with the primary tracer first.
 */
function createPlasmaTracerTrail(
  origin: Vector2,
  direction: Vector2,
  hitDistance: number,
  hitType: 'wall' | 'enemy',
): TracerState[] {
  const safeDistance = Math.max(
    NEATENSTEIN_PLASMA_MIN_SEGMENT_DISTANCE_CELLS,
    hitDistance,
  );

  const coreLength = clamp(
    safeDistance * NEATENSTEIN_PLASMA_CORE_DISTANCE_RATIO,
    NEATENSTEIN_PLASMA_MIN_SEGMENT_DISTANCE_CELLS,
    safeDistance,
  );

  const coreStart = Math.max(0, safeDistance - coreLength);
  const tracers: TracerState[] = [
    createTracerSegment(
      origin,
      direction,
      coreStart,
      safeDistance,
      hitType,
      NEATENSTEIN_TRACER_DURATION_MS,
    ),
  ];

  // Add trailing visual segments behind the bright core. These are not separate
  // hits; they only make the shot feel more like a moving plasma burst.
  for (
    let segment = 1;
    segment <= NEATENSTEIN_PLASMA_TRAIL_SEGMENTS;
    segment += 1
  ) {
    const segmentEnd =
      coreStart * (1 - segment / (NEATENSTEIN_PLASMA_TRAIL_SEGMENTS + 1));
    const segmentStart = segmentEnd * 0.72;
    const duration =
      NEATENSTEIN_TRACER_DURATION_MS *
      Math.pow(NEATENSTEIN_PLASMA_TRAIL_DURATION_FALLOFF, segment);

    tracers.push(
      createTracerSegment(
        origin,
        direction,
        segmentStart,
        segmentEnd,
        hitType,
        duration,
      ),
    );
  }

  return tracers;
}

/**
 * Fire the neon plasma burst and return the updated state plus visible tracer.
 *
 * Gameplay remains hitscan:
 *
 * 1. Ammo is checked.
 * 2. The shot ray is built from player position and yaw.
 * 3. The nearest wall within range is found.
 * 4. Living enemies are tested against the beam cylinder.
 * 5. The nearest valid enemy before the wall takes damage.
 * 6. Plasma-like visual tracers are appended.
 * 7. Wall impacts are emitted only when a real in-range wall was hit.
 *
 * @param state - Snapshot before firing.
 * @returns Immutable result with the new state, fire flag, and primary tracer.
 *
 * @example
 * ```ts
 * const result = fireNeonBeam(state);
 * if (result.fired && result.tracer) {
 *   drawTracer(result.tracer);
 * }
 * ```
 */
export function fireNeonBeam(state: GameState): FireNeonBeamResult {
  if (state.player.ammo <= 0) {
    return { state, fired: false, tracer: null };
  }

  const maxRange = resolveBeamMaxRange();
  const angleRad = resolvePlayerAngle(state.player.angleRad);

  const direction: Vector2 = {
    x: Math.cos(angleRad),
    y: Math.sin(angleRad),
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

  const wallHitInRange = isWallHitInRange(wallHit.perpWallDist, maxRange);

  let hitType: 'wall' | 'enemy' = 'wall';
  let hitDistance = wallHitInRange ? wallHit.perpWallDist : maxRange;
  let hitEnemyIndex = -1;

  // Test every living enemy against the beam and keep the nearest valid hit
  // before the current wall/range endpoint.
  for (let index = 0; index < state.enemies.length; index += 1) {
    const enemy = state.enemies[index];

    if (enemy.health <= 0) {
      continue;
    }

    const distanceAlongBeam = projectOntoBeam(
      origin,
      direction,
      enemy.position,
    );

    if (
      distanceAlongBeam <= 0 ||
      distanceAlongBeam > maxRange ||
      distanceAlongBeam > hitDistance
    ) {
      continue;
    }

    const missDistance = perpendicularDistance(
      origin,
      direction,
      enemy.position,
      distanceAlongBeam,
    );

    if (missDistance <= NEATENSTEIN_ENEMY_HIT_RADIUS_CELLS) {
      hitType = 'enemy';
      hitDistance = distanceAlongBeam;
      hitEnemyIndex = index;
    }
  }

  const hit = pointAlongRay(origin, direction, hitDistance);
  const plasmaTracers = createPlasmaTracerTrail(
    origin,
    direction,
    hitDistance,
    hitType,
  );
  const primaryTracer = plasmaTracers[0] ?? null;

  let nextState = consumeAmmo(state);

  // Append all visual plasma segments in one immutable update.
  nextState = {
    ...nextState,
    tracers: [...nextState.tracers, ...plasmaTracers],
  };

  // Only create a wall impact when the shot truly terminated on an in-range
  // wall. If the beam reached max range without hitting a wall, no impact spot
  // should appear in empty space.
  if (hitType === 'wall' && wallHitInRange) {
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
    tracer: primaryTracer,
  };
}

/**
 * Project a point onto the beam ray and return signed distance from origin.
 *
 * Negative values mean the point is behind the beam origin and should be
 * ignored.
 *
 * @param origin - Beam origin.
 * @param direction - Normalized beam direction.
 * @param point - Enemy position to test.
 * @returns Signed distance along the beam from origin to closest approach.
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
 * @param t - Distance along the beam to closest approach.
 * @returns Euclidean distance from the point to the beam.
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
 * Health is clamped at zero and the kill counter increments only when an enemy
 * transitions from alive to dead in this shot.
 *
 * @param state - Snapshot with tracers already appended.
 * @param enemyIndex - Index into {@link GameState.enemies}.
 * @returns New snapshot with updated enemy health and kill count.
 */
function applyEnemyDamage(state: GameState, enemyIndex: number): GameState {
  const enemy = state.enemies[enemyIndex];

  if (!enemy || enemy.health <= 0) {
    return state;
  }

  const newHealth = Math.max(0, enemy.health - NEATENSTEIN_BEAM_DAMAGE);
  const killedByThisShot = enemy.health > 0 && newHealth === 0;

  const newEnemies = state.enemies.map((existing, index) =>
    index === enemyIndex ? { ...existing, health: newHealth } : existing,
  );

  return {
    ...state,
    enemies: newEnemies,
    kills: killedByThisShot ? state.kills + 1 : state.kills,
  };
}
