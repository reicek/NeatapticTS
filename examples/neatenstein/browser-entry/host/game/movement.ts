/**
 * Player movement and wall-collision resolution for the Neatenstein host-side
 * simulation.
 *
 * This module owns the translation half of AC-205: reading keyboard movement
 * intent, normalizing diagonal speed, stepping the player forward at a fixed
 * speed, and sliding or reverting when a wall is hit.
 *
 * @module
 */

import type { InputSnapshot } from '../input';
import type { CollisionMap } from '../../renderer/map';
import {
  NEATENSTEIN_FIXED_TIMESTEP_MS,
  NEATENSTEIN_MS_PER_SECOND,
  NEATENSTEIN_PLAYER_RADIUS_CELLS,
  NEATENSTEIN_PLAYER_SPEED_CELLS_PER_SECOND,
} from './constants';
import { createGameState } from './state';
import type { GameState, Vector2 } from './types';

export { createGameState };

/**
 * Normalize a movement vector so diagonal movement does not outrun cardinal
 * movement.
 *
 * A zero-length vector is returned unchanged as `(0, 0)` to avoid a division
 * by zero when the player releases all movement keys.
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
  const length = Math.hypot(vector.x, vector.y);
  if (length === 0) {
    return { x: 0, y: 0 };
  }
  return { x: vector.x / length, y: vector.y / length };
}

/**
 * Step the player forward by a raw movement vector for one fixed timestep.
 *
 * The previous position is remembered so wall-slide collision can attempt
 * horizontal-only and vertical-only recovery before fully reverting a move.
 *
 * @param state - Snapshot before movement.
 * @param delta - Desired movement direction; diagonal inputs are normalized by
 *   callers such as {@link updatePlayerMovement} before this function runs.
 * @param dtMs - Movement duration in milliseconds; defaults to the fixed
 *   simulation timestep.
 * @returns New snapshot with the player moved one step and `previousPosition`
 *   set to the pre-move location.
 */
export function movePlayer(
  state: GameState,
  delta: Vector2,
  dtMs: number = NEATENSTEIN_FIXED_TIMESTEP_MS,
): GameState {
  const previousPosition = { ...state.player.position };
  const move = normalizeMoveVector(delta);
  const speed =
    NEATENSTEIN_PLAYER_SPEED_CELLS_PER_SECOND *
    (dtMs / NEATENSTEIN_MS_PER_SECOND);

  return {
    ...state,
    player: {
      ...state.player,
      previousPosition,
      position: {
        x: state.player.position.x + move.x * speed,
        y: state.player.position.y + move.y * speed,
      },
    },
  };
}

/**
 * Resolve wall collision by sliding along the first non-blocking axis or
 * reverting to the previous position.
 *
 * The resolution order is:
 *   1. Accept the new position if it is not blocked.
 *   2. Try X-only movement (new X, old Y).
 *   3. Try Y-only movement (old X, new Y).
 *   4. Revert to the previous position.
 *
 * This gives classic FPS wall-slide behavior for movement that grazes a
 * single wall face.
 *
 * @param state - Snapshot after an attempted move.
 * @param collisionMap - Map queried for solid cells.
 * @returns New snapshot with the resolved player position and
 *   `previousPosition` preserved from the input snapshot.
 */
export function resolveWallCollision(
  state: GameState,
  collisionMap: CollisionMap,
): GameState {
  if (!isPositionBlocked(state.player.position, collisionMap)) {
    return { ...state, player: { ...state.player } };
  }

  const previous = state.player.previousPosition ?? state.player.position;

  const xOnly = { x: state.player.position.x, y: previous.y };
  if (!isPositionBlocked(xOnly, collisionMap)) {
    return updatePlayerPosition(state, xOnly, previous);
  }

  const yOnly = { x: previous.x, y: state.player.position.y };
  if (!isPositionBlocked(yOnly, collisionMap)) {
    return updatePlayerPosition(state, yOnly, previous);
  }

  return updatePlayerPosition(state, previous, previous);
}

/**
 * Apply keyboard movement intent to the player for one tick.
 *
 * W/A/S/D booleans are rotated into world space using the player's current
 * look angle, normalized to preserve diagonal speed, and then stepped and
 * collision-resolved.
 *
 * @param state - Snapshot before movement.
 * @param movement - Directional movement intent from the input router.
 * @param collisionMap - Map queried for solid cells.
 * @param dtMs - Movement duration in milliseconds; defaults to the fixed
 *   simulation timestep.
 * @returns New snapshot with the player moved, slid along walls if needed.
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

  const yaw = state.player.angleRad;
  const forwardX = Math.cos(yaw);
  const forwardY = Math.sin(yaw);
  const rightX = -Math.sin(yaw);
  const rightY = Math.cos(yaw);

  const moveX = (forward - backward) * forwardX + (right - left) * rightX;
  const moveY = (forward - backward) * forwardY + (right - left) * rightY;

  const delta = normalizeMoveVector({ x: moveX, y: moveY });
  const moved = movePlayer(state, delta, dtMs);
  return resolveWallCollision(moved, collisionMap);
}

function isPositionBlocked(
  position: Vector2,
  collisionMap: CollisionMap,
): boolean {
  const radius = NEATENSTEIN_PLAYER_RADIUS_CELLS;
  const minX = Math.floor(position.x - radius);
  const minY = Math.floor(position.y - radius);
  const maxX = Math.floor(position.x + radius);
  const maxY = Math.floor(position.y + radius);

  for (let x = minX; x <= maxX; x++) {
    for (let y = minY; y <= maxY; y++) {
      if (collisionMap.isSolid(x, y)) {
        return true;
      }
    }
  }

  return false;
}

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
