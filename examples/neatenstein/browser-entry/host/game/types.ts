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

/** Mutable-style snapshot of the on-screen weapon overlay state. */
export interface GunState {
  /** Current vertical screen-space recoil offset applied to the gun overlay. */
  recoilOffset: number;
}

/** One traveling plasma projectile owned by the host simulation. */
export interface BoltState {
  /** Current world-space position. */
  position: Vector2;
  /** Normalized travel direction. */
  direction: Vector2;
  /** Movement speed in world cells per second. */
  speedCellsPerSecond: number;
  /** `true` while the bolt is still moving; `false` after expiry or deactivation. */
  active: boolean;
  /** Simulation time at which the bolt was created, in milliseconds. */
  createdAtMs: number;
  /** Optional spawn origin used for screen-space interpolation. */
  origin?: Vector2;
  /** Optional distance in cells the bolt should travel from origin. */
  targetDistance?: number;
}

/** Persistent neon impact marker left on a wall by a plasma bolt hit. */
export interface ImpactSpot {
  /** Raycast metadata that identifies the exact wall face that was hit. */
  wallHit: {
    /** Grid X coordinate of the hit wall cell. */
    mapX: number;
    /** Grid Y coordinate of the hit wall cell. */
    mapY: number;
    /** Wall side that was hit: `0` = X-side, `1` = Y-side. */
    side: 0 | 1;
    /** Fractional coordinate along the hit wall face, in [0, 1). */
    wallX: number;
  };
  /** Exact world-space position of the impact on the wall face. */
  position: Vector2;
  /** Simulation time at which the spot was created, in milliseconds. */
  createdAtMs: number;
  /** Milliseconds the spot remains visible before expiring. */
  lifetimeMs: number;
  /** Perpendicular distance from the camera to the wall hit when created. */
  perpWallDist: number;
  /**
   * Expected travel time in milliseconds for the spawning bolt to reach the
   * wall. The impact spot is rendered only once the bolt has arrived, i.e.
   * when `simTimeMs - createdAtMs >= boltTravelTimeMs`.
   */
  boltTravelTimeMs: number;
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
  /** Persistent neon impact spots left on walls by bolt hits. */
  impacts: ImpactSpot[];
  /** Current weapon overlay state, including screen-space recoil. */
  gun?: GunState;
  /** Active traveling plasma bolts in the world. */
  bolts?: BoltState[];
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
