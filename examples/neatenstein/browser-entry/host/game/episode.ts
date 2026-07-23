/**
 * Deterministic episode lifecycle for the Neatenstein host-side simulation.
 *
 * An episode is a single deterministic play-through from a fixed seed to a
 * terminal condition (time limit, player death, or all enemies killed). The
 * functions in this module are pure: they return immutable state snapshots so
 * the same seed always produces the same final state.
 *
 * @module
 */

import {
  NEATENSTEIN_EPISODE_DEFAULT_DURATION_MS,
  NEATENSTEIN_FIXED_TIMESTEP_MS,
} from './constants';
import { resolveContactDamage } from './collision';
import { createGameState } from './state';
import { spawnWaveTick } from './waves';
import type { GameState } from './types';

/** Options accepted by {@link createEpisode}. */
export interface CreateEpisodeOptions {
  /** Deterministic seed used to initialize the episode. */
  seed?: number;
  /** Target episode duration in milliseconds; defaults to 20_000. */
  durationMs?: number;
}

/** In-memory handle for a single episode run. */
export interface Episode {
  /** Mutable snapshot of the running episode state. */
  state: GameState;
  /** Target duration in milliseconds used to decide episode completion. */
  durationMs: number;
}

/**
 * Create a fresh episode container for the requested seed and duration.
 *
 * The returned episode captures the initial {@link GameState} and the target
 * duration. Call {@link runEpisode} to advance the episode to completion, or
 * manually step it with {@link updateEpisode}.
 *
 * @param options - Optional seed and duration overrides.
 * @returns An {@link Episode} ready to run.
 *
 * @example
 * ```ts
 * const episode = createEpisode({ seed: 7 });
 * const final = runEpisode(episode);
 * console.log(final.kills, final.episodeTimeMs);
 * ```
 */
export function createEpisode(options: CreateEpisodeOptions = {}): Episode {
  const seed = options.seed ?? 1;
  const durationMs =
    options.durationMs ?? NEATENSTEIN_EPISODE_DEFAULT_DURATION_MS;
  const state = createGameState({ seed });
  return {
    state: { ...state, episodeDurationMs: durationMs },
    durationMs,
  };
}

/**
 * Build the canonical initial {@link GameState} for an episode seed.
 *
 * This is a thin wrapper around {@link createGameState} that makes the
 * episode lifecycle naming explicit.
 *
 * @param seed - Deterministic seed for the episode.
 * @returns Fresh episode state with default duration configured.
 */
export function startEpisode(seed: number): GameState {
  return createGameState({ seed });
}

/**
 * Advance the episode state by one simulation step.
 *
 * The step advances timers, spawns at most one enemy via
 * {@link spawnWaveTick}, and resolves contact damage. The returned state is a
 * new immutable snapshot.
 *
 * @param state - Snapshot before the update.
 * @param dtMs - Elapsed time in milliseconds for this step.
 * @returns New snapshot after the update.
 */
export function updateEpisode(state: GameState, dtMs: number): GameState {
  let next: GameState = {
    ...state,
    simTimeMs: state.simTimeMs + dtMs,
    episodeTimeMs: state.episodeTimeMs + dtMs,
    player: {
      ...state.player,
      dashTimeRemainingMs: Math.max(0, state.player.dashTimeRemainingMs - dtMs),
      dashCooldownMs: Math.max(0, state.player.dashCooldownMs - dtMs),
    },
  };

  const spawnResult = spawnWaveTick(next, dtMs);
  next = spawnResult.state;
  next = resolveContactDamage(next, dtMs);

  return next;
}

/**
 * Check whether the episode should end.
 *
 * An episode is complete when any of the following hold:
 *   - the episode time reaches the configured duration limit,
 *   - the player has died, or
 *   - all spawned enemies have been killed.
 *
 * @param state - Current episode snapshot.
 * @returns `true` when the episode has reached a terminal condition.
 */
export function isEpisodeComplete(state: GameState): boolean {
  const durationMs =
    state.episodeDurationMs ?? NEATENSTEIN_EPISODE_DEFAULT_DURATION_MS;
  const outOfTime = state.episodeTimeMs >= durationMs;
  const playerDead = state.player.health <= 0;
  const allEnemiesKilled =
    state.enemies.length > 0 &&
    state.enemies.every((enemy) => enemy.health <= 0);
  return outOfTime || playerDead || allEnemiesKilled;
}

/**
 * Finalize a completed episode state.
 *
 * Currently returns a fresh immutable copy of the terminal state. Future
 * slices may attach a fitness or score summary here once the evolution
 * harness defines its objective function.
 *
 * @param state - Terminal episode snapshot.
 * @returns Finalized episode snapshot.
 */
export function endEpisode(state: GameState): GameState {
  return { ...state };
}

/**
 * Run an episode from its initial state to completion using fixed timesteps.
 *
 * Controller inputs are intentionally ignored in this slice; the red-phase
 * contract only requires deterministic seed-based replay. Future slices may
 * introduce an input stream once the evolution harness wires controllers in.
 *
 * @param episode - Episode container returned by {@link createEpisode}.
 * @returns Final {@link GameState} after the episode ends.
 */
export function runEpisode(episode: Episode): GameState {
  let state = episode.state;
  while (!isEpisodeComplete(state)) {
    state = updateEpisode(state, NEATENSTEIN_FIXED_TIMESTEP_MS);
  }
  return endEpisode(state);
}
