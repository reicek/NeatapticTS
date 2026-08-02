/**
 * Enemy AI controller for the Neatenstein NGE demo.
 *
 * Produces deterministic per-enemy animation/AI descriptors from a
 * {@link GameState} snapshot. Each controlled enemy seeks the player, resolves
 * wall collisions with axis-slide behavior, fires hitscan shots when the
 * player is in range and line-of-sight, and plays a 4-second de-rez death
 * animation when its health reaches zero or its hitscan ammunition is depleted.
 *
 * @module
 */

import type { CollisionMap } from '../browser-entry/renderer/map';
import {
  NEATENSTEIN_ENEMY_COLLISION_RADIUS_CELLS,
  NEATENSTEIN_FIXED_TIMESTEP_MS,
} from '../browser-entry/host/game/constants';
import { normalizeMoveVector } from '../browser-entry/host/game/movement';
import type {
  EnemyState,
  GameState,
  Vector2,
} from '../browser-entry/host/game/types';

/**
 * Number of sim ticks the muzzle-flash shoot blink lasts on the upper body.
 *
 * When an enemy fires, the upper body shows the shoot frame for this many
 * ticks before reverting to the walk frame. Must be in the 3–5 range per
 * AC-10f-005.
 */
export const ENEMY_CONTROLLER_SHOOT_BLINK_TICKS = 4;

/**
 * Animation and AI state for a single controlled enemy, produced by the
 * controller each tick and consumed by the sprite renderer.
 *
 * This type is intentionally local to the controller so the core
 * {@link EnemyState} can stay minimal while the renderer gets richer,
 * per-enemy behavior data.
 */
export interface ControlledEnemy {
  /** Index of this enemy in the source {@link GameState.enemies} array. */
  index: number;
  /** Current world position after movement and collision resolution. */
  position: Vector2;
  /** Current hit points, mirrored from the source {@link EnemyState}. */
  health: number;
  /** Facing angle in radians; 0 = +X axis. */
  yawRad: number;
  /** Animation state consumed by the sprite renderer. */
  animationState: 'idle' | 'move' | 'fire' | 'death';
  /** Remaining hitscan ammunition. Depletes to trigger de-rez. */
  ammo: number;
  /** Milliseconds until the enemy may fire again. */
  fireCooldownMs: number;
  /** Milliseconds spent in the death de-rez animation. */
  deRezElapsedMs: number;
  /** `true` while the enemy is still active (not fully de-rezzed). */
  active: boolean;
  /** Sim tick counter driving the walk cycle (stand → walk1 → stand → walk2). Increments each tick the enemy moves; resets to 0 when idle. */
  walkTick: number;
  /** Remaining sim ticks for the muzzle-flash shoot blink on the upper body. When > 0 the renderer composites the shoot upper body over the walk lower body. */
  shootBlinkTicks: number;
}

/**
 * One hitscan fire event emitted by an enemy this tick.
 */
export interface HitscanEvent {
  /** Source enemy index in {@link GameState.enemies}. */
  enemyIndex: number;
  /** World-space origin of the hitscan ray. */
  origin: Vector2;
  /** Normalized direction toward the player. */
  direction: Vector2;
  /** Hit points removed from the player on a confirmed hit. */
  damage: number;
}

/**
 * Output of a single controller tick, bundling per-enemy AI/animation
 * descriptors with any hitscan fire events produced this tick.
 */
export interface EnemyControllerState {
  /** Per-enemy AI/animation descriptors. */
  enemies: ControlledEnemy[];
  /** Hitscan fire events produced this tick. */
  hitscanEvents: HitscanEvent[];
}

/**
 * Enemy movement speed in world cells per second, tuned so evolved enemies
 * close distance quickly without overshooting the player at short range.
 */
export const ENEMY_CONTROLLER_SPEED_CELLS_PER_SECOND = 2.5;

/**
 * Collision radius in world cells.
 *
 * Enemies have a 192×192 block footprint and each floor cell spans 252 blocks,
 * giving a radius of 96 / 252 ≈ 0.381 cells.
 */
export const ENEMY_CONTROLLER_RADIUS_CELLS =
  NEATENSTEIN_ENEMY_COLLISION_RADIUS_CELLS;

/** Maximum cell distance at which an enemy will attempt to fire. */
export const ENEMY_CONTROLLER_FIRE_RANGE_CELLS = 8;

/**
 * Minimum cell distance the enemy tries to maintain from the player.
 *
 * Keeps the seek behavior from producing excessive overlap once the enemy is
 * already close enough for contact damage.
 */
export const ENEMY_CONTROLLER_STOP_DISTANCE_CELLS = 0.3;

/**
 * Cooldown between hitscan shots in milliseconds, preventing enemies from
 * firing continuously and giving the player a predictable rhythm.
 */
export const ENEMY_CONTROLLER_FIRE_COOLDOWN_MS = 1000;

/** Hit points removed from the player by a single enemy hitscan shot. */
export const ENEMY_CONTROLLER_HITSCAN_DAMAGE = 10;

/**
 * Starting ammunition for each freshly tracked enemy, depleted by hitscan
 * shots and used to trigger the de-rez death animation.
 */
export const ENEMY_CONTROLLER_STARTING_AMMO = 3;

/**
 * Duration of the death de-rez animation in milliseconds, kept identical to
 * the sprite-side timing so visual and AI states stay synchronized.
 */
export const ENEMY_CONTROLLER_DE_REZ_DURATION_MS = 4000;

/** Small epsilon subtracted from the max collision edge to avoid boundary bleed. */
const COLLISION_EDGE_EPSILON = 1e-9;

/** Line-of-sight sampling step in world cells. */
const LINE_OF_SIGHT_STEP_CELLS = 0.5;

/** Neutral vector reused for zero-length fallbacks. */
const ZERO_VECTOR: Vector2 = { x: 0, y: 0 };

/**
 * Return whether a value is a finite number.
 *
 * @param value - Candidate value.
 * @returns Whether the value is finite.
 */
function isFiniteNumber(value: number): boolean {
  return Number.isFinite(value);
}

/**
 * Clamp a timestep to a positive finite value.
 *
 * Invalid or non-positive timesteps fall back to the canonical fixed timestep
 * so malformed caller input cannot poison enemy behavior.
 *
 * @param dtMs - Candidate timestep in milliseconds.
 * @returns Positive finite timestep.
 */
function resolveTimestepMs(dtMs: number): number {
  return isFiniteNumber(dtMs) && dtMs > 0
    ? dtMs
    : NEATENSTEIN_FIXED_TIMESTEP_MS;
}

/**
 * Return the signed distance and normalized direction from `from` to `to`.
 *
 * @param from - Source position.
 * @param to - Target position.
 * @returns Object containing distance and direction vector.
 */
function vectorTo(
  from: Vector2,
  to: Vector2,
): { distance: number; direction: Vector2 } {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const distance = Math.hypot(dx, dy);

  if (!isFiniteNumber(distance) || distance === 0) {
    return { distance: 0, direction: { ...ZERO_VECTOR } };
  }

  return {
    distance,
    direction: { x: dx / distance, y: dy / distance },
  };
}

/**
 * Return whether a circular position overlaps any solid map cell.
 *
 * Out-of-bounds cells are treated as solid by the {@link CollisionMap}
 * implementation.
 *
 * @param position - Entity center in world cells.
 * @param radius - Entity radius in world cells.
 * @param collisionMap - Map queried for solid cells.
 * @returns Whether the position is blocked.
 */
function isPositionBlocked(
  position: Vector2,
  radius: number,
  collisionMap: CollisionMap,
): boolean {
  const minX = Math.floor(position.x - radius);
  const minY = Math.floor(position.y - radius);
  const maxX = Math.floor(position.x + radius - COLLISION_EDGE_EPSILON);
  const maxY = Math.floor(position.y + radius - COLLISION_EDGE_EPSILON);

  for (let x = minX; x <= maxX; x += 1) {
    for (let y = minY; y <= maxY; y += 1) {
      if (collisionMap.isSolid(x, y)) {
        return true;
      }
    }
  }

  return false;
}

/**
 * Resolve a desired enemy position against wall collisions.
 *
 * Resolution order mirrors the player wall-slide behavior:
 *
 * 1. Accept the new position if it is not blocked.
 * 2. Try X-only movement.
 * 3. Try Y-only movement.
 * 4. Revert to the previous position.
 *
 * @param previous - Position before the movement step.
 * @param desired - Desired position after the movement step.
 * @param radius - Entity radius in world cells.
 * @param collisionMap - Map queried for solid cells.
 * @returns Collision-resolved position.
 */
function resolveEnemyPosition(
  previous: Vector2,
  desired: Vector2,
  radius: number,
  collisionMap: CollisionMap,
): Vector2 {
  if (!isPositionBlocked(desired, radius, collisionMap)) {
    return desired;
  }

  const xOnly = { x: desired.x, y: previous.y };
  if (!isPositionBlocked(xOnly, radius, collisionMap)) {
    return xOnly;
  }

  const yOnly = { x: previous.x, y: desired.y };
  if (!isPositionBlocked(yOnly, radius, collisionMap)) {
    return yOnly;
  }

  return previous;
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
function hasLineOfSight(
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
 * Build a fresh controller state from a {@link GameState} snapshot.
 *
 * Each source enemy is initialized with default ammunition, no fire cooldown,
 * and the idle animation state. Callers must hold the returned state across
 * ticks so fire cooldowns, ammunition, and de-rez timing advance correctly.
 *
 * @param state - Source game snapshot.
 * @returns Fresh controller state.
 */
export function createEnemyControllerState(
  state: GameState,
): EnemyControllerState {
  return {
    enemies: state.enemies.map((enemy, index) => ({
      index,
      position: { ...enemy.position },
      health: enemy.health,
      yawRad: 0,
      animationState: 'idle',
      ammo: ENEMY_CONTROLLER_STARTING_AMMO,
      fireCooldownMs: 0,
      deRezElapsedMs: 0,
      active: true,
      walkTick: 0,
      shootBlinkTicks: 0,
    })),
    hitscanEvents: [],
  };
}

/**
 * Update one controlled enemy for a single tick.
 *
 * @param index - Index in the source enemy array.
 * @param enemyState - Source enemy snapshot.
 * @param previous - Previous controlled state, if any.
 * @param gameState - Current game snapshot.
 * @param collisionMap - Map queried for solid cells.
 * @param dtMs - Tick duration in milliseconds.
 * @param hitscanEvents - Array to append any fire event to.
 * @returns New controlled enemy descriptor.
 */
function updateControlledEnemy(
  index: number,
  enemyState: EnemyState,
  previous: ControlledEnemy | undefined,
  gameState: GameState,
  collisionMap: CollisionMap,
  dtMs: number,
  hitscanEvents: HitscanEvent[],
): ControlledEnemy {
  const isRespawn =
    previous !== undefined && !previous.active && enemyState.health > 0;

  const previousOrDefault: ControlledEnemy =
    previous ??
    ({
      index,
      position: { ...enemyState.position },
      health: enemyState.health,
      yawRad: 0,
      animationState: 'idle',
      ammo: ENEMY_CONTROLLER_STARTING_AMMO,
      fireCooldownMs: 0,
      deRezElapsedMs: 0,
      active: true,
      walkTick: 0,
      shootBlinkTicks: 0,
    } as ControlledEnemy);

  const ammo = isRespawn
    ? ENEMY_CONTROLLER_STARTING_AMMO
    : previousOrDefault.ammo;
  const fireCooldownMs = Math.max(
    0,
    (isRespawn ? 0 : previousOrDefault.fireCooldownMs) - dtMs,
  );
  let deRezElapsedMs = isRespawn ? 0 : previousOrDefault.deRezElapsedMs;
  let walkTick = isRespawn ? 0 : previousOrDefault.walkTick;
  let shootBlinkTicks = isRespawn ? 0 : previousOrDefault.shootBlinkTicks;
  let position = isRespawn
    ? { ...enemyState.position }
    : { ...previousOrDefault.position };
  let yawRad = isRespawn ? 0 : previousOrDefault.yawRad;

  // Death / de-rez path: health depleted or ammunition exhausted.
  if (enemyState.health <= 0 || ammo <= 0) {
    deRezElapsedMs += dtMs;
    const active = deRezElapsedMs < ENEMY_CONTROLLER_DE_REZ_DURATION_MS;

    if (enemyState.health > 0) {
      // When ammo depletion triggers de-rez, face the player one last time.
      const toPlayer = vectorTo(position, gameState.player.position);
      if (toPlayer.distance > 0) {
        yawRad = Math.atan2(toPlayer.direction.y, toPlayer.direction.x);
      }
    }

    return {
      index,
      position,
      health: enemyState.health,
      yawRad,
      animationState: 'death',
      ammo: Math.max(0, ammo),
      fireCooldownMs,
      deRezElapsedMs,
      active,
      walkTick: 0,
      shootBlinkTicks: 0,
    };
  }

  // Alive path: seek player and fire when able.
  const toPlayer = vectorTo(position, gameState.player.position);
  const distToPlayer = toPlayer.distance;

  if (toPlayer.distance > 0) {
    yawRad = Math.atan2(toPlayer.direction.y, toPlayer.direction.x);
  }

  let moved = false;
  let animationState: ControlledEnemy['animationState'] = 'idle';

  if (distToPlayer > ENEMY_CONTROLLER_STOP_DISTANCE_CELLS) {
    const moveDir = normalizeMoveVector(toPlayer.direction);
    const stepDistance =
      ENEMY_CONTROLLER_SPEED_CELLS_PER_SECOND * (dtMs / 1000);
    const desired = {
      x: position.x + moveDir.x * stepDistance,
      y: position.y + moveDir.y * stepDistance,
    };

    const resolved = resolveEnemyPosition(
      position,
      desired,
      ENEMY_CONTROLLER_RADIUS_CELLS,
      collisionMap,
    );

    moved = resolved.x !== position.x || resolved.y !== position.y;
    position = resolved;
  }

  // Update walk tick: increment each tick the enemy moves, reset when idle.
  walkTick = moved ? walkTick + 1 : 0;

  let isFiring = false;
  if (
    fireCooldownMs <= 0 &&
    ammo > 0 &&
    distToPlayer <= ENEMY_CONTROLLER_FIRE_RANGE_CELLS &&
    hasLineOfSight(position, gameState.player.position, collisionMap)
  ) {
    isFiring = true;
    hitscanEvents.push({
      enemyIndex: index,
      origin: { ...position },
      direction: { x: Math.cos(yawRad), y: Math.sin(yawRad) },
      damage: ENEMY_CONTROLLER_HITSCAN_DAMAGE,
    });
  }

  // Decrement shoot blink from the previous tick, then refresh if firing.
  if (shootBlinkTicks > 0) {
    shootBlinkTicks -= 1;
  }
  if (isFiring) {
    shootBlinkTicks = ENEMY_CONTROLLER_SHOOT_BLINK_TICKS;
  }

  if (isFiring) {
    animationState = 'fire';
  } else if (moved) {
    animationState = 'move';
  }

  return {
    index,
    position,
    health: enemyState.health,
    yawRad,
    animationState,
    ammo: isFiring ? Math.max(0, ammo - 1) : ammo,
    fireCooldownMs: isFiring
      ? ENEMY_CONTROLLER_FIRE_COOLDOWN_MS
      : fireCooldownMs,
    deRezElapsedMs,
    active: true,
    walkTick,
    shootBlinkTicks,
  };
}

/**
 * Push active enemies apart so their 192×192-block footprints do not overlap.
 *
 * A single pairwise pass is sufficient because the AI moves slowly and the
 * collision radius is small. The separation is applied symmetrically, so two
 * overlapping enemies each move half the overlap distance.
 *
 * @param enemies - Resolved enemy descriptors produced by
 *   {@link updateControlledEnemy} this tick.
 */
function separateEnemies(enemies: ControlledEnemy[]): void {
  const combinedDiameter = ENEMY_CONTROLLER_RADIUS_CELLS * 2;

  for (let i = 0; i < enemies.length; i += 1) {
    const a = enemies[i];
    if (!a.active) {
      continue;
    }

    for (let j = i + 1; j < enemies.length; j += 1) {
      const b = enemies[j];
      if (!b.active) {
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
}

/**
 * Advance the enemy AI controller by one simulation tick, producing new enemy
 * descriptors and any hitscan fire events produced this tick.
 *
 * The controller matches enemies by array index so it can preserve per-enemy
 * ammunition, fire cooldowns, and de-rez timing across calls. If an enemy is
 * replaced at the same index while the previous one was fully de-rezzed, the
 * new enemy receives fresh controller state.
 *
 * @param controller - Previous controller state.
 * @param state - Current game snapshot.
 * @param collisionMap - Map queried for solid cells.
 * @param dtMs - Tick duration in milliseconds.
 * @returns New controller state and any hitscan events produced this tick.
 */
export function updateEnemyController(
  controller: EnemyControllerState,
  state: GameState,
  collisionMap: CollisionMap,
  dtMs: number,
): EnemyControllerState {
  const resolvedDtMs = resolveTimestepMs(dtMs);
  const previousByIndex = new Map(
    controller.enemies.map((enemy) => [enemy.index, enemy]),
  );

  const enemies: ControlledEnemy[] = [];
  const hitscanEvents: HitscanEvent[] = [];

  for (let i = 0; i < state.enemies.length; i += 1) {
    const previous = previousByIndex.get(i);
    enemies.push(
      updateControlledEnemy(
        i,
        state.enemies[i],
        previous,
        state,
        collisionMap,
        resolvedDtMs,
        hitscanEvents,
      ),
    );
  }

  separateEnemies(enemies);

  return {
    enemies,
    hitscanEvents,
  };
}
