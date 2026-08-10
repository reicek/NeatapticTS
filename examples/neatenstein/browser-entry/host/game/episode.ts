/**
 * Deterministic episode lifecycle for the Neatenstein host-side simulation.
 *
 * An episode is a single deterministic play-through from a fixed seed to a
 * terminal condition: player death or all spawned enemies killed. Episodes
 * are NOT time-bound — generations run until gameplay ends.
 *
 * The functions in this module return immutable state snapshots so the same
 * seed and input/update sequence produce the same final state.
 *
 * @module
 */

import {
  NEATENSTEIN_EPISODE_DEFAULT_DURATION_MS,
  NEATENSTEIN_FIXED_TIMESTEP_MS,
  NEATENSTEIN_MAP_SIZE,
} from './constants';
import { resolveContactDamage } from './collision';
import { buildNeatensteinMap, createCollisionMap } from '../../renderer/map';
import type { CollisionMap } from '../../renderer/map';
import { createGameState } from './state';
import { spawnWaveTick } from './waves';
import type { GameState } from './types';

/**
 * Minimum valid episode duration in milliseconds.
 */
const MIN_EPISODE_DURATION_MS = 1;

/**
 * Minimum valid fixed timestep in milliseconds.
 */
const MIN_TIMESTEP_MS = 1;

/**
 * Safety margin applied to the computed maximum episode step count.
 *
 * This prevents `runEpisode` from looping forever if future completion rules
 * change or malformed state slips through.
 */
const EPISODE_STEP_GUARD_MARGIN = 2;

/** Options accepted by {@link createEpisode}. */
export interface CreateEpisodeOptions {
  /** Deterministic seed used to initialize the episode. */
  seed?: number;

  /** Target episode duration in milliseconds. */
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
 * Return whether a value is a finite number.
 *
 * @param value - Candidate number.
 * @returns Whether the value is finite.
 */
function isFiniteNumber(value: number): boolean {
  return Number.isFinite(value);
}

/**
 * Resolve a deterministic episode seed.
 *
 * Non-finite seeds fall back to the canonical default seed.
 *
 * @param seed - Optional caller-provided seed.
 * @returns Finite episode seed.
 */
function resolveEpisodeSeed(seed: number | undefined): number {
  return typeof seed === 'number' && isFiniteNumber(seed) ? seed : 1;
}

/**
 * Resolve a valid positive episode duration.
 *
 * Invalid, zero, or negative durations fall back to the configured default.
 *
 * @param durationMs - Optional caller-provided duration.
 * @returns Positive finite episode duration in milliseconds.
 */
function resolveEpisodeDurationMs(durationMs: number | undefined): number {
  return typeof durationMs === 'number' &&
    isFiniteNumber(durationMs) &&
    durationMs >= MIN_EPISODE_DURATION_MS
    ? durationMs
    : NEATENSTEIN_EPISODE_DEFAULT_DURATION_MS;
}

/**
 * Resolve a valid positive timestep.
 *
 * Invalid, zero, or negative timesteps fall back to the configured fixed
 * timestep.
 *
 * @param dtMs - Candidate timestep.
 * @returns Positive finite timestep in milliseconds.
 */
function resolveEpisodeTimestepMs(dtMs: number): number {
  return isFiniteNumber(dtMs) && dtMs >= MIN_TIMESTEP_MS
    ? dtMs
    : NEATENSTEIN_FIXED_TIMESTEP_MS;
}

/**
 * Decrease a timer by the elapsed timestep and clamp it at zero.
 *
 * Invalid existing timers are treated as already expired.
 *
 * @param currentMs - Current timer value in milliseconds.
 * @param dtMs - Elapsed timestep in milliseconds.
 * @returns Updated non-negative timer value.
 */
function tickTimerMs(currentMs: number, dtMs: number): number {
  const current = isFiniteNumber(currentMs) && currentMs > 0 ? currentMs : 0;

  return Math.max(0, current - dtMs);
}

/**
 * Attach an episode duration to a freshly created game state.
 *
 * @param state - Source game state.
 * @param durationMs - Positive finite episode duration.
 * @returns State with `episodeDurationMs` configured.
 */
function withEpisodeDuration(state: GameState, durationMs: number): GameState {
  return {
    ...state,
    episodeDurationMs: durationMs,
  };
}

/**
 * Advance global episode timers and player dash timers.
 *
 * @param state - Snapshot before timer advancement.
 * @param dtMs - Positive finite timestep.
 * @returns State with advanced timers.
 */
function advanceEpisodeTimers(state: GameState, dtMs: number): GameState {
  const simTimeMs = isFiniteNumber(state.simTimeMs) ? state.simTimeMs : 0;
  const episodeTimeMs = isFiniteNumber(state.episodeTimeMs)
    ? state.episodeTimeMs
    : 0;

  return {
    ...state,
    simTimeMs: simTimeMs + dtMs,
    episodeTimeMs: episodeTimeMs + dtMs,
    player: {
      ...state.player,
      dashTimeRemainingMs: tickTimerMs(state.player.dashTimeRemainingMs, dtMs),
      dashCooldownMs: tickTimerMs(state.player.dashCooldownMs, dtMs),
    },
  };
}

/**
 * Compute a safe upper bound on update steps for an episode.
 *
 * @param durationMs - Episode duration in milliseconds.
 * @param timestepMs - Fixed timestep in milliseconds.
 * @returns Maximum update iterations before the run is forcibly finalized.
 */
function resolveMaxEpisodeSteps(
  durationMs: number,
  timestepMs: number,
): number {
  return Math.ceil(durationMs / timestepMs) + EPISODE_STEP_GUARD_MARGIN;
}

/**
 * Create a fresh episode container for the requested seed and duration.
 *
 * The returned episode captures the initial {@link GameState} and target
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
  const seed = resolveEpisodeSeed(options.seed);
  const durationMs = resolveEpisodeDurationMs(options.durationMs);
  const state = withEpisodeDuration(createGameState({ seed }), durationMs);

  return {
    state,
    durationMs,
  };
}

/**
 * Build the canonical initial {@link GameState} for an episode seed.
 *
 * This wrapper mirrors {@link createEpisode} by attaching the default episode
 * duration to the returned state.
 *
 * @param seed - Deterministic seed for the episode.
 * @returns Fresh episode state with default duration configured.
 */
export function startEpisode(seed: number): GameState {
  const resolvedSeed = resolveEpisodeSeed(seed);
  const durationMs = resolveEpisodeDurationMs(undefined);

  return withEpisodeDuration(
    createGameState({ seed: resolvedSeed }),
    durationMs,
  );
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
 * @param collisionMap - Optional collision map passed to the spawner so edge
 *   cells can be validated.
 * @returns New snapshot after the update.
 */
export function updateEpisode(
  state: GameState,
  dtMs: number,
  collisionMap?: CollisionMap,
): GameState {
  const resolvedDtMs = resolveEpisodeTimestepMs(dtMs);

  // Step 1: Advance deterministic timers before subsystems use the new time.
  let next = advanceEpisodeTimers(state, resolvedDtMs);

  // Step 2: Spawn enemies according to the wave schedule.
  const spawnResult = spawnWaveTick(next, resolvedDtMs, collisionMap);
  next = spawnResult.state;

  // Step 3: Apply player/enemy contact damage after spawning and timer updates.
  next = resolveContactDamage(next, resolvedDtMs);

  return next;
}

/**
 * Check whether the episode should end.
 *
 * The episode runs until the time-limit guard in {@link runEpisode} is
 * reached. Player death triggers a respawn (not episode end) and the game
 * features infinite waves, so there is no gameplay-based terminal condition.
 *
 * @param state - Current episode snapshot.
 * @returns Always `false`; the episode is terminated by the step guard.
 */
export function isEpisodeComplete(_state: GameState): boolean {
  void _state;
  return false;
}

/**
 * Finalize a completed episode state.
 *
 * Currently returns a fresh snapshot of the terminal state. Future slices may
 * attach a fitness or score summary here once the evolution harness defines its
 * objective function.
 *
 * @param state - Terminal episode snapshot.
 * @returns Finalized episode snapshot.
 */
export function endEpisode(state: GameState): GameState {
  return {
    ...state,
    player: {
      ...state.player,
      position: { ...state.player.position },
      previousPosition: state.player.previousPosition
        ? { ...state.player.previousPosition }
        : undefined,
    },
    enemies: state.enemies.map((enemy) => ({
      ...enemy,
      position: { ...enemy.position },
    })),
    bolts: (state.bolts ?? []).map((bolt) => ({
      ...bolt,
      position: { ...bolt.position },
      direction: { ...bolt.direction },
    })),
    impacts: state.impacts.map((impact) => ({
      ...impact,
      position: { ...impact.position },
      wallHit: { ...impact.wallHit },
    })),
  };
}

/**
 * Run an episode from its initial state to completion using fixed timesteps.
 *
 * The episode loops {@link updateEpisode} until the step guard is reached.
 * Player death triggers a respawn (not episode end) and the game features
 * infinite waves, so there is no gameplay-based terminal condition — the
 * step guard caps the loop at the configured episode duration.
 *
 * @param episode - Episode container returned by {@link createEpisode}.
 * @returns Final {@link GameState} after the episode ends.
 */
export function runEpisode(episode: Episode): GameState {
  const durationMs = resolveEpisodeDurationMs(episode.durationMs);
  const timestepMs = resolveEpisodeTimestepMs(NEATENSTEIN_FIXED_TIMESTEP_MS);
  const maxSteps = resolveMaxEpisodeSteps(durationMs, timestepMs);

  let state = withEpisodeDuration(episode.state, durationMs);
  const collisionMap = createCollisionMap(
    buildNeatensteinMap(state.seed),
    NEATENSTEIN_MAP_SIZE,
  );

  for (let step = 0; step < maxSteps; step += 1) {
    state = updateEpisode(state, timestepMs, collisionMap);
  }

  return endEpisode(state);
}
