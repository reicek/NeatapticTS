/**
 * Host-level wave transition for the Neatenstein NGE demo.
 *
 * Orchestrates the "clear → evolve → spawn" cycle that starts a new enemy
 * wave in the live renderer. The module is intentionally thin: it clears the
 * arena of active enemies, projectiles and transient impact spots, advances the
 * co-evolution enemy population by one generation, and pre-populates the wave
 * with up to {@link NEATENSTEIN_ENEMY_MAX_CONCURRENT} enemies using the existing
 * deterministic trickle spawner.
 *
 * @module
 */

import {
  NEATENSTEIN_ENEMY_MAX_CONCURRENT,
  NEATENSTEIN_FIXED_TIMESTEP_MS,
} from './game/constants';
import { spawnWaveTick } from './game/waves';
import type { GameState } from './game/types';
import type { MlpEnemyPopulation } from '../harness/enemy-mlp';
import type { Snapshot } from '../harness/types';

/**
 * Options accepted by {@link advanceWave}, pairing the live enemy population
 * with an optional cap on how many enemies the next wave should spawn.
 */
export interface AdvanceWaveOptions {
  /** Live enemy population whose champion snapshot is advanced one generation. */
  population: MlpEnemyPopulation;
  /**
   * Number of enemies to spawn for the new wave.
   *
   * Defaults to {@link NEATENSTEIN_ENEMY_MAX_CONCURRENT}. Values below zero are
   * treated as zero; non-finite values fall back to the default cap; values
   * above the concurrency cap are clamped to the cap.
   */
  spawnCount?: number;
}

/**
 * Result returned by {@link advanceWave}, exposing the cleared arena state,
 * the advanced champion snapshot, and the actual number of spawned enemies.
 */
export interface AdvanceWaveResult {
  /** New game snapshot with a cleared arena and freshly spawned enemies. */
  state: GameState;
  /** Champion snapshot produced by advancing the enemy population. */
  snapshot: Snapshot;
  /** Number of enemies actually spawned (always in [0, cap]). */
  spawnedCount: number;
}

/**
 * Clamp a requested wave spawn count to a valid integer in [0, cap].
 *
 * @param requested - Raw caller value, which may be missing or invalid.
 * @param cap - Maximum concurrent enemies allowed.
 * @returns Safe integer spawn count.
 */
function resolveSpawnCount(requested: number | undefined, cap: number): number {
  if (requested === undefined || !Number.isFinite(requested)) {
    return cap;
  }
  const integer = Math.max(0, Math.floor(requested));
  return Math.min(integer, cap);
}

/**
 * Advance to the next enemy wave by clearing the arena, evolving the MLP
 * population one generation, and spawning the next generation of enemies.
 *
 * The function performs three deterministic steps:
 *
 * 1. **Clear the arena** — removes all active enemies, traveling bolts and
 *    persistent impact spots while preserving the player, score counters and
 *    episode timing.
 * 2. **Run the evolution harness** — advances the enemy population by one
 *    generation and returns its champion snapshot.
 * 3. **Spawn the wave** — calls the deterministic trickle spawner repeatedly
 *    until the requested number of enemies have entered the arena, without
 *    exceeding {@link NEATENSTEIN_ENEMY_MAX_CONCURRENT}.
 *
 * @param state - Snapshot before the wave transition.
 * @param options - Population and optional spawn count.
 * @returns New state, enemy snapshot and number of enemies spawned.
 *
 * @example
 * ```ts
 * const population = createMlpEnemyPopulation({ seed: 1 });
 * const before = createGameState({ seed: 1 });
 * const result = advanceWave(before, { population });
 * console.log(result.state.enemies.length); // <= 8
 * console.log(result.state.generation);     // before.generation + 1
 * ```
 */
export function advanceWave(
  state: GameState,
  options: AdvanceWaveOptions,
): AdvanceWaveResult {
  const cap = NEATENSTEIN_ENEMY_MAX_CONCURRENT;
  const nextGeneration = state.generation + 1;
  const snapshot = options.population.update({ generation: nextGeneration });

  let clearedState: GameState = {
    ...state,
    generation: nextGeneration,
    enemies: [],
    bolts: [],
    impacts: [],
  };

  const targetSpawnCount = resolveSpawnCount(options.spawnCount, cap);
  let spawnedCount = 0;

  for (let i = 0; i < targetSpawnCount; i++) {
    const result = spawnWaveTick(clearedState, NEATENSTEIN_FIXED_TIMESTEP_MS);
    clearedState = result.state;
    spawnedCount += result.spawnedThisTick;
  }

  return { state: clearedState, snapshot, spawnedCount };
}
