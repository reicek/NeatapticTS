/**
 * Player movement and wall-collision resolution for the Neatenstein host-side
 * simulation.
 *
 * This module owns the translation half of the player simulation:
 *
 * - reading directional movement intent
 * - rotating local movement into world space
 * - normalizing diagonal movement so it does not outrun cardinal movement
 * - stepping the player by a fixed timestep
 * - resolving wall collisions with simple axis-slide behavior
 *
 * @module
 */

import type { CollisionMap } from '../../renderer/map';
import type { InputSnapshot } from '../input';
import {
  NEATENSTEIN_ENEMY_COLLISION_RADIUS_CELLS,
  NEATENSTEIN_FIXED_TIMESTEP_MS,
  NEATENSTEIN_MS_PER_SECOND,
  NEATENSTEIN_PLAYER_RADIUS_CELLS,
  NEATENSTEIN_PLAYER_SPEED_CELLS_PER_SECOND,
} from './constants';
import { createGameState } from './state';
import type { EnemyState, GameState, Vector2 } from './types';

export { createGameState };

/**
 * Small epsilon used when converting a player radius AABB into occupied cells.
 *
 * Subtracting epsilon from the max edge prevents exact boundary contact from
 * being treated as penetration into the neighboring cell.
 */
const NEATENSTEIN_COLLISION_EDGE_EPSILON = 1e-9;

/**
 * Neutral movement vector.
 */
const ZERO_VECTOR: Vector2 = { x: 0, y: 0 };

/**
 * Return whether a number is finite.
 *
 * @param value - Candidate numeric value.
 * @returns Whether the value is a finite number.
 */
function isFiniteNumber(value: number): boolean {
  return Number.isFinite(value);
}

/**
 * Resolve a safe movement timestep.
 *
 * Invalid or non-positive timesteps fall back to the canonical fixed timestep
 * so malformed caller input cannot poison player position.
 *
 * @param dtMs - Candidate timestep in milliseconds.
 * @returns Positive finite timestep.
 */
function resolveMovementTimestepMs(dtMs: number): number {
  return isFiniteNumber(dtMs) && dtMs > 0
    ? dtMs
    : NEATENSTEIN_FIXED_TIMESTEP_MS;
}

/**
 * Return whether a vector has finite numeric components.
 *
 * @param vector - Vector to inspect.
 * @returns Whether both components are finite.
 */
function isFiniteVector(vector: Vector2): boolean {
  return isFiniteNumber(vector.x) && isFiniteNumber(vector.y);
}

/**
 * Normalize a movement vector so diagonal movement does not outrun cardinal
 * movement.
 *
 * A zero-length or non-finite vector is returned as `(0, 0)` to avoid division
 * by zero and prevent invalid input from contaminating player state.
 *
 * @param vector - Raw movement vector, typically built from `{-1, 0, 1}`
 *   per-axis inputs.
 * @returns A vector with the same direction and length `1`, or `(0, 0)`.
 *
 * @example
 * ```ts
 * const diagonal = normalizeMoveVector({ x: 1, y: 1 });
 * expect(Math.hypot(diagonal.x, diagonal.y)).toBeCloseTo(1);
 * ```
 */
export function normalizeMoveVector(vector: Vector2): Vector2 {
  if (!isFiniteVector(vector)) {
    return { ...ZERO_VECTOR };
  }

  const length = Math.hypot(vector.x, vector.y);

  if (!isFiniteNumber(length) || length === 0) {
    return { ...ZERO_VECTOR };
  }

  return {
    x: vector.x / length,
    y: vector.y / length,
  };
}

/**
 * Step the player by a movement vector for one timestep.
 *
 * The previous position is preserved so collision resolution can try
 * horizontal-only and vertical-only recovery before fully reverting the move.
 *
 * @param state - Snapshot before movement.
 * @param delta - Desired movement direction.
 * @param dtMs - Movement duration in milliseconds.
 * @returns New snapshot with the player moved and `previousPosition` recorded.
 */
export function movePlayer(
  state: GameState,
  delta: Vector2,
  dtMs: number = NEATENSTEIN_FIXED_TIMESTEP_MS,
): GameState {
  const move = normalizeMoveVector(delta);

  // Avoid needless state allocation when there is no movement.
  if (move.x === 0 && move.y === 0) {
    return state;
  }

  const resolvedDtMs = resolveMovementTimestepMs(dtMs);
  const previousPosition = { ...state.player.position };
  const stepDistance =
    NEATENSTEIN_PLAYER_SPEED_CELLS_PER_SECOND *
    (resolvedDtMs / NEATENSTEIN_MS_PER_SECOND);

  return {
    ...state,
    player: {
      ...state.player,
      previousPosition,
      position: {
        x: state.player.position.x + move.x * stepDistance,
        y: state.player.position.y + move.y * stepDistance,
      },
    },
  };
}

/**
 * Resolve wall collision by sliding along the first non-blocking axis or
 * reverting to the previous position.
 *
 * Resolution order:
 *
 * 1. Accept the new position if it is not blocked.
 * 2. Try X-only movement: new X, old Y.
 * 3. Try Y-only movement: old X, new Y.
 * 4. Revert to the previous position.
 *
 * This gives classic FPS wall-slide behavior for movement that grazes a single
 * wall face.
 *
 * @param state - Snapshot after an attempted move.
 * @param collisionMap - Map queried for solid cells.
 * @returns State with resolved player position.
 */
export function resolveWallCollision(
  state: GameState,
  collisionMap: CollisionMap,
): GameState {
  if (!isPositionBlocked(state.player.position, collisionMap, state.enemies)) {
    return state;
  }

  const previous = state.player.previousPosition ?? state.player.position;

  const xOnly = {
    x: state.player.position.x,
    y: previous.y,
  };

  if (!isPositionBlocked(xOnly, collisionMap, state.enemies)) {
    return updatePlayerPosition(state, xOnly, previous);
  }

  const yOnly = {
    x: previous.x,
    y: state.player.position.y,
  };

  if (!isPositionBlocked(yOnly, collisionMap, state.enemies)) {
    return updatePlayerPosition(state, yOnly, previous);
  }

  return updatePlayerPosition(state, previous, previous);
}

/**
 * Apply directional movement intent to the player for one tick.
 *
 * Movement booleans are rotated into world space using the player's current
 * look angle. The resulting vector is normalized, stepped, and collision
 * resolved.
 *
 * @param state - Snapshot before movement.
 * @param movement - Directional movement intent from the input router.
 * @param collisionMap - Map queried for solid cells.
 * @param dtMs - Movement duration in milliseconds.
 * @returns New snapshot with the player moved and collision-resolved.
 */
export function updatePlayerMovement(
  state: GameState,
  movement: InputSnapshot['movement'],
  collisionMap: CollisionMap,
  dtMs: number = NEATENSTEIN_FIXED_TIMESTEP_MS,
): GameState {
  const forward = movement.forward ? 1 : 0;
  const backward = movement.backward ? 1 : 0;
  const left = movement.left ? 1 : 0;
  const right = movement.right ? 1 : 0;

  const localForward = forward - backward;
  const localRight = right - left;

  // No movement input means no movement or collision allocation is required.
  if (localForward === 0 && localRight === 0) {
    return state;
  }

  const yaw = isFiniteNumber(state.player.angleRad) ? state.player.angleRad : 0;

  const forwardX = Math.cos(yaw);
  const forwardY = Math.sin(yaw);
  const rightX = -Math.sin(yaw);
  const rightY = Math.cos(yaw);

  const delta = normalizeMoveVector({
    x: localForward * forwardX + localRight * rightX,
    y: localForward * forwardY + localRight * rightY,
  });

  const moved = movePlayer(state, delta, dtMs);

  return resolveWallCollision(moved, collisionMap);
}

/**
 * Return whether a player position overlaps any solid map cell.
 *
 * The player is approximated as an axis-aligned square around its center. This
 * is intentionally simple and stable for grid collision. Out-of-bounds cells
 * are treated as solid by {@link CollisionMap}.
 *
 * @param position - Player center position in world/grid units.
 * @param collisionMap - Collision map queried for solid cells.
 * @returns Whether the position is blocked.
 */
export function isPositionBlocked(
  position: Vector2,
  collisionMap: CollisionMap,
  enemies: ReadonlyArray<EnemyState> = [],
): boolean {
  if (!isFiniteVector(position)) {
    return true;
  }

  const radius = NEATENSTEIN_PLAYER_RADIUS_CELLS;

  const minX = Math.floor(position.x - radius);
  const minY = Math.floor(position.y - radius);

  // Subtract epsilon from the max edge so exact boundary contact does not count
  // as being inside the neighboring cell.
  const maxX = Math.floor(
    position.x + radius - NEATENSTEIN_COLLISION_EDGE_EPSILON,
  );
  const maxY = Math.floor(
    position.y + radius - NEATENSTEIN_COLLISION_EDGE_EPSILON,
  );

  for (let x = minX; x <= maxX; x += 1) {
    for (let y = minY; y <= maxY; y += 1) {
      if (collisionMap.isSolid(x, y)) {
        return true;
      }
    }
  }

  // Hero cannot walk through living, active enemies. Treat each enemy as a
  // circle with the shared enemy collision radius, using the synced controller
  // position so the block matches the rendered enemy location.
  const playerRadius = radius;
  const enemyRadius = NEATENSTEIN_ENEMY_COLLISION_RADIUS_CELLS;
  const combinedRadius = playerRadius + enemyRadius;
  const combinedRadiusSquared = combinedRadius * combinedRadius;

  for (const enemy of enemies) {
    if (enemy.health <= 0 || enemy.active === false) {
      continue;
    }

    const enemyPosition = enemy.controllerPosition ?? enemy.position;
    const dx = position.x - enemyPosition.x;
    const dy = position.y - enemyPosition.y;
    if (dx * dx + dy * dy < combinedRadiusSquared) {
      return true;
    }
  }

  return false;
}

/**
 * Return a state with an updated player position.
 *
 * @param state - Source game state.
 * @param position - Resolved player position.
 * @param previousPosition - Previous player position preserved for future
 *   collision resolution.
 * @returns New state with updated player position fields.
 */
function updatePlayerPosition(
  state: GameState,
  position: Vector2,
  previousPosition: Vector2,
): GameState {
  return {
    ...state,
    player: {
      ...state.player,
      position,
      previousPosition,
    },
  };
}
