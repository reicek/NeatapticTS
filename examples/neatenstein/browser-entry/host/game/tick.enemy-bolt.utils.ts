/**
 * Enemy plasma-bolt movement and player-hit executors for the Neatenstein tick
 * pipeline.
 *
 * @module
 */

import {
  NEATENSTEIN_CONTACT_IFRAME_MS,
  NEATENSTEIN_ENEMY_BOLT_HIT_RADIUS_CELLS,
  NEATENSTEIN_ENEMY_BOLT_LIFETIME_MS,
  NEATENSTEIN_ENEMY_BOLT_MAX_RANGE_CELLS,
  NEATENSTEIN_MAP_SIZE,
  NEATENSTEIN_MS_PER_SECOND,
} from './constants';
import { applyDamage } from './state';
import { resolveTickDurationMs } from './tick.time.utils';
import type { CollisionMap } from '../../renderer/map';
import type { EnemyBoltState, GameState } from './types';
import type { UpdateEnemyBoltsResult } from '../types';

// Re-export consolidated types so existing imports from this module remain valid.
export type { UpdateEnemyBoltsResult } from '../types';

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
  const dtSeconds = resolvedDtMs / NEATENSTEIN_MS_PER_SECOND;
  let nextState = state;

  // Mutate bolt positions in place and compact into a result array of the
  // same bolt references — no .filter().map() clone chain (A2 Fix 5).
  const updatedBolts: EnemyBoltState[] = [];
  for (let i = 0; i < bolts.length; i += 1) {
    const bolt = bolts[i];
    if (!bolt.active) continue;

    // Use local scalars for the next position — no Vector2 allocation.
    const step = bolt.speedCellsPerSecond * dtSeconds;
    const nextX = bolt.position.x + bolt.direction.x * step;
    const nextY = bolt.position.y + bolt.direction.y * step;

    // Check wall collision.
    const hitWall = collisionMap
      ? collisionMap.isSolid(Math.floor(nextX), Math.floor(nextY))
      : false;

    // Check out of bounds.
    const outOfBounds =
      nextX < 0 ||
      nextX >= NEATENSTEIN_MAP_SIZE ||
      nextY < 0 ||
      nextY >= NEATENSTEIN_MAP_SIZE;

    // Check distance traveled from origin.
    const distanceTraveled =
      bolt.origin &&
      Number.isFinite(bolt.origin.x) &&
      Number.isFinite(bolt.origin.y)
        ? Math.hypot(nextX - bolt.origin.x, nextY - bolt.origin.y)
        : 0;
    const beyondMaxRange =
      distanceTraveled >= NEATENSTEIN_ENEMY_BOLT_MAX_RANGE_CELLS;

    // Check lifetime expiry.
    const elapsedMs = Math.max(0, currentTimeMs - bolt.createdAtMs);
    const lifetimeExpired = elapsedMs >= NEATENSTEIN_ENEMY_BOLT_LIFETIME_MS;

    // Check player proximity.
    const playerDist = Math.hypot(
      nextX - nextState.player.position.x,
      nextY - nextState.player.position.y,
    );
    const hitPlayer =
      !hitWall &&
      !outOfBounds &&
      !beyondMaxRange &&
      playerDist <= NEATENSTEIN_ENEMY_BOLT_HIT_RADIUS_CELLS;

    const movementStopped = outOfBounds || hitWall || beyondMaxRange;
    const active = !lifetimeExpired && !hitPlayer;

    // Capture previous hit state before mutation.
    const wasHit = bolt.hitPlayer;

    // Mutate bolt position in place — only update if the bolt moved.
    if (!movementStopped) {
      bolt.position.x = nextX;
      bolt.position.y = nextY;
    }
    bolt.active = active;
    bolt.hitPlayer = hitPlayer || bolt.hitPlayer;

    if (hitPlayer && !wasHit) {
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

    updatedBolts.push(bolt);
  }

  return { state: nextState, bolts: updatedBolts };
}
