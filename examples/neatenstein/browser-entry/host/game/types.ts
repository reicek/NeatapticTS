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
  /**
   * Position before the most recent movement step.
   *
   * Used by wall-slide collision to attempt horizontal-only and vertical-only
   * movement before fully reverting a blocked move.
   */
  previousPosition?: Vector2;
  /** Milliseconds of invulnerability remaining from recent enemy contact. */
  contactIFrameMs?: number;
}

/** Minimal enemy state used by the spawn/wave logic. */
export interface EnemyState {
  /** Current world position. */
  position: Vector2;
  /** Current hit points. */
  health: number;
}

/** One frame-visible neon beam tracer produced by {@link fireNeonBeam}. */
export interface TracerState {
  /** Beam origin in world units (player muzzle position). */
  origin: Vector2;
  /** Normalized beam direction. */
  direction: Vector2;
  /** World-space endpoint of the tracer (wall or enemy hit). */
  hit: Vector2;
  /** Distance from origin to hit in world units. */
  distance: number;
  /** Kind of target the beam terminated on. */
  hitType: 'wall' | 'enemy';
  /** Milliseconds the tracer remains visible. */
  durationMs: number;
  /** CSS color string used by the renderer. */
  color: string;
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
  /** Active neon beam tracers visible this frame. */
  tracers: TracerState[];
  /** Total confirmed kills for scoring and evolution pressure. */
  kills: number;
  /**
   * Monotonic counter of enemies spawned since episode start.
   *
   * Used to derive a unique deterministic RNG seed for every spawn event so
   * different game histories (different kill counts, active rosters, etc.)
   * cannot accidentally collide and produce identical spawn positions.
   */
  spawnCount: number;
  /** Current evolutionary generation (1 = first human-played generation). */
  generation: number;
  /**
   * Target episode duration in milliseconds.
   *
   * When set, {@link isEpisodeComplete} uses this value as the time-limit
   * threshold. If omitted, the episode falls back to the canonical default
   * duration defined in the constants module.
   */
  episodeDurationMs?: number;
}
