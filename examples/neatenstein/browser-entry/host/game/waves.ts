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
  NEATENSTEIN_ENEMY_SPAWN_MIN_DISTANCE_CELLS,
  NEATENSTEIN_ENEMY_SPAWN_RADIUS,
  NEATENSTEIN_SPAWN_CENTER_X,
  NEATENSTEIN_SPAWN_CENTER_Y,
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
 * cap. Enemies appear at a random angle and a random distance between
 * {@link NEATENSTEIN_ENEMY_SPAWN_MIN_DISTANCE_CELLS} and
 * {@link NEATENSTEIN_ENEMY_SPAWN_RADIUS} cells from the map center so their
 * cell coordinates always fall inside the 60×60 grid and never start inside
 * the player's contact-damage range. The returned state is always a new
 * immutable snapshot, even when no enemy is spawned, so callers can replay
 * history without accidental mutation.
 *
 * @param state - Snapshot before the spawn tick.
 * @param _dtMs - Elapsed simulation time in milliseconds (reserved for future
 *   spawn-rate modulation; currently each tick may spawn at most one enemy).
 * @returns Object reporting how many enemies spawned and the new state.
 *
 * @example
 * ```ts
 * import { NEATENSTEIN_FIXED_TIMESTEP_MS } from './constants';
 * const before = createGameState({ seed: 1 });
 * const result = spawnWaveTick(before, NEATENSTEIN_FIXED_TIMESTEP_MS);
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
  const distance =
    NEATENSTEIN_ENEMY_SPAWN_MIN_DISTANCE_CELLS +
    rng() *
      (NEATENSTEIN_ENEMY_SPAWN_RADIUS -
        NEATENSTEIN_ENEMY_SPAWN_MIN_DISTANCE_CELLS);
  const enemy: EnemyState = {
    position: {
      x: NEATENSTEIN_SPAWN_CENTER_X + Math.cos(angle) * distance,
      y: NEATENSTEIN_SPAWN_CENTER_Y + Math.sin(angle) * distance,
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
