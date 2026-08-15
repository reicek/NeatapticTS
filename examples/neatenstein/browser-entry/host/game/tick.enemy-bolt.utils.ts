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
import type { EnemyBoltState, GameState, Vector2 } from './types';
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
