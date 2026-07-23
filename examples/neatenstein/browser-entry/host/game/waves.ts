/**
 * Deterministic enemy wave spawning for the Neatenstein host-side simulation.
 *
 * Enemies spawn as a continuous trickle: at most one new enemy per simulation
 * tick, and never more than {@link NEATENSTEIN_ENEMY_MAX_CONCURRENT} active at
 * once. All transitions return immutable state snapshots so replay stays
 * deterministic.
 *
 * @module
 */

import {
  NEATENSTEIN_ENEMY_MAX_CONCURRENT,
  NEATENSTEIN_ENEMY_SPAWN_RADIUS,
} from './constants';
import { createGameRng } from './state';
import type { EnemyState, GameState } from './types';

/** Result of a single spawn tick. */
export interface SpawnWaveTickResult {
  /** Number of enemies added this tick (always 0 or 1). */
  spawnedThisTick: number;
  /** New state snapshot with any spawned enemy included. */
  state: GameState;
}

/**
 * Advance enemy spawning by one fixed tick.
 *
 * Spawns at most one enemy per call and enforces the active-enemy concurrency
 * cap. The returned state is always a new immutable snapshot, even when no
 * enemy is spawned, so callers can replay history without accidental mutation.
 *
 * @param state - Snapshot before the spawn tick.
 * @param _dtMs - Elapsed simulation time in milliseconds (reserved for future
 *   spawn-rate modulation; currently each tick may spawn at most one enemy).
 * @returns Object reporting how many enemies spawned and the new state.
 *
 * @example
 * ```ts
 * const before = createGameState({ seed: 1 });
 * const result = spawnWaveTick(before, 16);
 * expect(result.spawnedThisTick).toBeLessThanOrEqual(1);
 * ```
 */
export function spawnWaveTick(
  state: GameState,
  _dtMs: number,
): SpawnWaveTickResult {
  // Reserved for future spawn-rate modulation; the public contract accepts
  // the elapsed tick time even though the current trickle policy is one enemy
  // per call below the concurrency cap.
  void _dtMs;

  if (state.enemies.length >= NEATENSTEIN_ENEMY_MAX_CONCURRENT) {
    return {
      spawnedThisTick: 0,
      state: { ...state, enemies: [...state.enemies] },
    };
  }

  // Use a dedicated monotonic spawn counter for the RNG seed so different
  // game histories (different kill counts / active enemy rosters) can never
  // produce the same deterministic spawn position.
  const rng = createGameRng(state.seed + state.spawnCount);
  const angle = rng() * 2 * Math.PI;
  const distance = rng() * NEATENSTEIN_ENEMY_SPAWN_RADIUS;
  const enemy: EnemyState = {
    position: {
      x: Math.cos(angle) * distance,
      y: Math.sin(angle) * distance,
    },
    health: 1,
  };

  return {
    spawnedThisTick: 1,
    state: {
      ...state,
      enemies: [...state.enemies, enemy],
      spawnCount: state.spawnCount + 1,
    },
  };
}
