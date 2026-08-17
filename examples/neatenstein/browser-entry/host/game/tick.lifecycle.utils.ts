/**
 * Lifecycle executors extracted from the gameTick pipeline: bolt-hit
 * application, hero respawn-on-death, and fire-recoil application.
 *
 * @module
 */

import { applyEnemyDamage, fireBolt } from './combat';
import {
  NEATENSTEIN_ENEMY_IMPACT_MAX_CONCURRENT,
  NEATENSTEIN_GUN_RECOIL_MAX_OFFSET_PX,
  NEATENSTEIN_PLAYER_MAX_AMMO,
  NEATENSTEIN_PLAYER_MAX_HEALTH,
  NEATENSTEIN_SPAWN_CENTER_X,
  NEATENSTEIN_SPAWN_CENTER_Y,
} from './constants';
import { NEATENSTEIN_ENEMY_IMPACT_LIFETIME_MS } from '../../constants';
import type { BoltState, EnemyImpactSpot, GameState } from './types';
import type { BoltImpactResult, FireRecoilResult } from '../types';

// Re-export consolidated types so existing imports from this module remain valid.
export type { BoltImpactResult, FireRecoilResult } from '../types';

/**
 * Process bolt-hit results: apply enemy damage and spawn enemy-impact spots.
 *
 * Iterates the updated bolt array and for each bolt that has been deactivated
 * by an enemy hit, applies damage via {@link applyEnemyDamage} and records an
 * {@link EnemyImpactSpot} at the enemy position.
 *
 * @param state - Game state before bolt-impact processing.
 * @param updatedBolts - Bolt array after movement and deactivation.
 * @param simTimeMs - Current simulation time in milliseconds.
 * @returns Updated state and whether any bolt hit an enemy.
 */
export function applyBoltImpactResults(
  state: GameState,
  updatedBolts: BoltState[],
  simTimeMs: number,
): BoltImpactResult {
  let next = state;
  let boltHitEnemy = false;

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
          createdAtMs: simTimeMs,
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

  return { state: next, boltHitEnemy };
}

/**
 * Respawn the hero at the map center if health has been depleted.
 *
 * Resets the player to spawn position with full health, ammo, and cleared
 * dash/i-frame timers. Increments the death counter.
 *
 * @param state - Game state to check for death.
 * @returns Updated state with respawned hero, or the original state if the
 *   hero is still alive.
 */
export function respawnHeroOnDeath(state: GameState): GameState {
  if (state.player.health > 0) {
    return state;
  }

  return {
    ...state,
    player: {
      ...state.player,
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
    deaths: (state.deaths ?? 0) + 1,
  };
}

/**
 * Fire a plasma bolt and apply gun recoil if the shot was fired.
 *
 * Delegates to {@link fireBolt} for the actual firing logic. When a bolt is
 * fired, sets the gun recoil offset to the maximum and marks the gun as
 * firing.
 *
 * @param state - Game state before firing.
 * @returns Updated state and whether a bolt was fired.
 */
export function applyFireAndRecoil(state: GameState): FireRecoilResult {
  const fireResult = fireBolt(state);
  let next = fireResult.state;

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

  return { state: next, fired: fireResult.fired };
}
