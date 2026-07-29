/**
 * Deterministic game-state container for the Neatenstein host-side simulation.
 *
 * This module owns the canonical reset function and the pure state transitions
 * for player damage, ammo consumption, and dash activation. All transitions
 * return a new state object so callers can replay history from a seed without
 * accidental mutation.
 *
 * @module
 */

import seedrandom from 'seedrandom';
import {
  NEATENSTEIN_DASH_COOLDOWN_MS,
  NEATENSTEIN_DASH_INVULNERABILITY_MS,
  NEATENSTEIN_EPISODE_DEFAULT_DURATION_MS,
  NEATENSTEIN_PLAYER_MAX_AMMO,
  NEATENSTEIN_PLAYER_MAX_HEALTH,
  NEATENSTEIN_SPAWN_CENTER_X,
  NEATENSTEIN_SPAWN_CENTER_Y,
} from './constants';
import { NEATENSTEIN_DEFAULT_SEED } from '../../constants';
import { createInitialGunState } from '../../renderer/gun';
import type {
  CreateGameStateOptions,
  EnemyState,
  GameState,
  PlayerState,
} from './types';

/**
 * Reconstruct a seeded PRNG from a deterministic game seed.
 *
 * Callers use this to resume replay after cloning or transferring a plain
 * {@link GameState}. The seed is stored on the state, so the PRNG can always
 * be recreated on demand.
 *
 * @param seed - Deterministic seed originally passed to {@link createGameState}.
 * @returns A seeded PRNG matching the one used at episode start.
 *
 * @example
 * ```ts
 * const rng = createGameRng(state.seed);
 * const nextValue = rng();
 * ```
 */
export function createGameRng(seed: number): () => number {
  return seedrandom(String(seed));
}

/**
 * Build the deterministic initial state for a Neatenstein episode.
 *
 * The same `seed` always produces the same canonical snapshot. The returned
 * object is safe to clone, transfer, and mutate through the pure helpers in
 * this module; deterministic replay reconstructs the PRNG from the stored
 * seed via {@link createGameRng}.
 *
 * @param options - Optional reset configuration; `seed` defaults to
 *   {@link NEATENSTEIN_DEFAULT_SEED}.
 * @returns A fresh canonical {@link GameState} for the requested seed.
 *
 * @example
 * ```ts
 * const state = createGameState({ seed: 7 });
 * expect(state.player.health).toBe(state.player.maxHealth);
 * ```
 */
export function createGameState(
  options: CreateGameStateOptions = {},
): GameState {
  const seed = options.seed ?? NEATENSTEIN_DEFAULT_SEED;
  const rng = createGameRng(seed);
  const spawnPosition = {
    x: NEATENSTEIN_SPAWN_CENTER_X,
    y: NEATENSTEIN_SPAWN_CENTER_Y,
  };
  const player: PlayerState = {
    position: { ...spawnPosition },
    previousPosition: { ...spawnPosition },
    angleRad: rng() * 2 * Math.PI,
    health: NEATENSTEIN_PLAYER_MAX_HEALTH,
    maxHealth: NEATENSTEIN_PLAYER_MAX_HEALTH,
    ammo: NEATENSTEIN_PLAYER_MAX_AMMO,
    maxAmmo: NEATENSTEIN_PLAYER_MAX_AMMO,
    dashTimeRemainingMs: 0,
    dashCooldownMs: 0,
    contactIFrameMs: 0,
  };

  const enemies: EnemyState[] = [];

  return {
    seed,
    simTimeMs: 0,
    episodeTimeMs: 0,
    episodeDurationMs: NEATENSTEIN_EPISODE_DEFAULT_DURATION_MS,
    player,
    enemies,
    impacts: [],
    gun: createInitialGunState(),
    bolts: [],
    lightEnabled: true,
    kills: 0,
    spawnCount: 0,
    generation: 1,
  };
}

/**
 * Check whether the player is currently invulnerable to damage.
 *
 * Invulnerability is granted by a recent dash or by recent enemy contact
 * damage and expires once the corresponding counters reach zero.
 *
 * @param state - Snapshot to inspect.
 * @returns `true` while the player is in an invulnerability window.
 */
export function isInvulnerable(state: GameState): boolean {
  return (
    state.player.dashTimeRemainingMs > 0 ||
    (state.player.contactIFrameMs ?? 0) > 0
  );
}

/**
 * Check whether the player can initiate a new dash.
 *
 * A dash is available only when {@link PlayerState.dashCooldownMs} has reached
 * zero, which happens after the configured cooldown duration.
 *
 * @param state - Snapshot to inspect.
 * @returns `true` if a new dash can be started.
 */
export function canDash(state: GameState): boolean {
  return state.player.dashCooldownMs <= 0;
}

/**
 * Apply damage to the player, clamping health at zero.
 *
 * Damage is ignored while the player is invulnerable (dash i-frames or contact i-frames).
 *
 * @param state - Snapshot before damage.
 * @param amount - Damage amount; clamped to zero if negative.
 * @returns New snapshot with reduced (and clamped) player health, or an
 * immutable clone of the snapshot (with a new player object) if the player is
 * currently invulnerable.
 */
export function applyDamage(state: GameState, amount: number): GameState {
  if (isInvulnerable(state)) {
    return { ...state, player: { ...state.player } };
  }
  const clamped = Math.max(0, amount);
  return {
    ...state,
    player: {
      ...state.player,
      health: Math.max(0, state.player.health - clamped),
    },
  };
}

/**
 * Consume a single unit of ammo.
 *
 * @param state - Snapshot before consumption.
 * @returns New snapshot with ammo decremented by one, clamped at zero.
 */
export function consumeAmmo(state: GameState): GameState {
  return {
    ...state,
    player: {
      ...state.player,
      ammo: Math.max(0, state.player.ammo - 1),
    },
  };
}

/**
 * Activate a dash, granting invulnerability and starting the cooldown.
 *
 * The dash only takes effect when {@link canDash} returns `true`. While on
 * cooldown, an immutable clone of the snapshot (with a new player object) is
 * returned so the observable cooldown prevents indefinite chaining.
 *
 * @param state - Snapshot before the dash.
 * @returns New snapshot with {@link PlayerState.dashTimeRemainingMs} set to the
 * configured invulnerability window and {@link PlayerState.dashCooldownMs} set
 * to the configured cooldown duration, or an immutable clone of the
 * snapshot (with a new player object) while on cooldown.
 */
export function applyDash(state: GameState): GameState {
  if (!canDash(state)) {
    return { ...state, player: { ...state.player } };
  }
  return {
    ...state,
    player: {
      ...state.player,
      dashTimeRemainingMs: NEATENSTEIN_DASH_INVULNERABILITY_MS,
      dashCooldownMs: NEATENSTEIN_DASH_COOLDOWN_MS,
    },
  };
}
