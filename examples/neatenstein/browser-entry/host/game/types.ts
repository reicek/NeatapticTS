/**
 * Shared type definitions for the Neatenstein host-side game simulation.
 *
 * These types are intentionally plain objects (not classes) so the state is
 * safe to clone, transfer, and replay deterministically. The deterministic
 * seed is stored on {@link GameState}; callers reconstruct the PRNG from it
 * when replay is required. All coordinates are in world units and all angles
 * are in radians.
 *
 * @module
 */

/** Two-dimensional point or vector in world space. */
export interface Vector2 {
  /** X component in world units. */
  x: number;
  /** Y component in world units. */
  y: number;
}

/** Mutable-style snapshot of the player character. */
export interface PlayerState {
  /** Current world position. */
  position: Vector2;
  /** Current look angle in radians, 0 = +X axis. */
  angleRad: number;
  /** Current hit points. */
  health: number;
  /** Maximum hit points the player can have. */
  maxHealth: number;
  /** Current ammo count. */
  ammo: number;
  /** Maximum ammo the player can carry. */
  maxAmmo: number;
  /** Milliseconds of invulnerability remaining from the most recent dash. */
  dashTimeRemainingMs: number;
  /** Milliseconds until another dash can be initiated. */
  dashCooldownMs: number;
}

/** Minimal enemy state used by the spawn/wave logic. */
export interface EnemyState {
  /** Current world position. */
  position: Vector2;
  /** Current hit points. */
  health: number;
}

/** Options accepted by {@link createGameState}. */
export interface CreateGameStateOptions {
  /** Deterministic seed used to initialize RNG-driven state. */
  seed?: number;
}

/** Complete deterministic snapshot of one Neatenstein game instance. */
export interface GameState {
  /** Seed used to create this snapshot; replay uses the same seed. */
  seed: number;
  /** Accumulated simulation time in milliseconds. */
  simTimeMs: number;
  /** Accumulated episode (real-world) time in milliseconds. */
  episodeTimeMs: number;
  /** Player character state. */
  player: PlayerState;
  /** Active enemies in the world. */
  enemies: EnemyState[];
  /** Total confirmed kills for scoring and evolution pressure. */
  kills: number;
  /** Current evolutionary generation (1 = first human-played generation). */
  generation: number;
}
