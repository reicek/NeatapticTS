/* istanbul ignore file */
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
  /** Maximum hit points (set at spawn time). */
  maxHealth?: number;
  /** Whether the enemy is still active (not fully de-rezzed). */
  active?: boolean;
  /** Position synced from the enemy AI controller for hero collision. */
  controllerPosition?: Vector2;
  /** Remaining hit-stun time in milliseconds (0 when not stunned). */
  stunTimerMs?: number;
}

/** Mutable-style snapshot of the on-screen weapon overlay state. */
export interface GunState {
  /** Current vertical screen-space recoil offset applied to the gun overlay. */
  recoilOffset: number;
  /** True while the gun is actively firing (drives muzzle-flash burst). */
  firing: boolean;
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
  /** Collision radius of the bolt in world cells. */
  radius?: number;
  /** Index of the first enemy this bolt collided with, if any. */
  hitEnemyIndex?: number;
}

/**
 * One traveling enemy plasma projectile owned by the host simulation.
 *
 * Enemy bolts are spawned from {@link HitscanEvent} data produced by the
 * enemy controller. Their origin and direction come directly from the
 * enemy's computed hitscan ray — no additional RNG is used, keeping the
 * simulation fully deterministic.
 */
export interface EnemyBoltState {
  /** Current world-space position. */
  position: Vector2;
  /** Normalized travel direction (from enemy toward player at fire time). */
  direction: Vector2;
  /** Movement speed in world cells per second. */
  speedCellsPerSecond: number;
  /** `true` while the bolt is still moving; `false` after expiry or hit. */
  active: boolean;
  /** Simulation time at which the bolt was created, in milliseconds. */
  createdAtMs: number;
  /** Spawn origin (enemy position at fire time) for screen-space interpolation. */
  origin?: Vector2;
  /** Distance in cells the bolt should travel from origin before expiring. */
  targetDistance?: number;
  /** Damage applied to the player on hit (10 = 10% of 100 maxHealth). */
  damage: number;
  /** `true` when this bolt has already struck the player. */
  hitPlayer?: boolean;
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

/**
 * Persistent neon impact marker left on an enemy by a plasma bolt hit.
 *
 * Unlike {@link ImpactSpot} (wall impacts), enemy impacts sit on the floor
 * plane at the enemy's world position and do not carry wall-face metadata or
 * a perpendicular wall distance. They are projected to screen space using the
 * same floor projection as bolts and sprites.
 */
export interface EnemyImpactSpot {
  /** World-space position of the enemy at the moment of impact. */
  position: Vector2;
  /** Simulation time at which the spot was created, in milliseconds. */
  createdAtMs: number;
  /** Milliseconds the spot remains visible before expiring. */
  lifetimeMs: number;
  /**
   * Expected travel time in milliseconds for the spawning bolt to reach the
   * enemy. The impact spot is rendered only once the bolt has arrived, i.e.
   * when `simTimeMs - createdAtMs >= boltTravelTimeMs`. Set to 0 for the
   * traveling-bolt path where the bolt has already arrived at the enemy.
   */
  boltTravelTimeMs: number;
}

/**
 * One ammo pickup dropped by a dying enemy.
 *
 * Ammo pickups are shiny squares that rest on the floor plane at the enemy's
 * death position. The player collects them by proximity, restoring ammo via
 * {@link restoreAmmo}. Pickups expire after their lifetime elapses.
 */
export interface AmmoPickupState {
  /** World-space position where the enemy died. */
  position: Vector2;
  /** Amount of ammo restored when collected. */
  amount: number;
  /** `true` while the pickup is active (not yet collected or expired). */
  active: boolean;
  /** Simulation time at which the pickup was created, in milliseconds. */
  createdAtMs: number;
  /** Milliseconds the pickup remains before expiring. Defaults to {@link NEATENSTEIN_AMMO_PICKUP_LIFETIME_MS}. */
  lifetimeMs?: number;
}

/** Options accepted by {@link createGameState}. */
export interface CreateGameStateOptions {
  /** Deterministic seed used to initialize RNG-driven state. */
  seed?: number;
}

/**
 * Per-episode combat telemetry accumulated during a Neatenstein fitness episode.
 *
 * The fitness harness reads these counters after the episode completes to
 * derive combat-quality signals such as accuracy and damage efficiency. All
 * counters start at zero and are incremented by the combat functions
 * ({@link fireBolt} increments `shotsFired`; {@link applyEnemyDamage} increments
 * `damageDealt` and `shotsHit`). The `aimMissRate` field is recomputed from
 * the raw counters on every update.
 *
 * The shot outcome taxonomy fields (`shotsWallHit`, `shotsRangeExpired`,
 * `shotsBlindFire`, `shotsNearMiss`) classify every non-hitting shot into
 * exactly one category, enabling the fitness function to penalize wasteful
 * shooting patterns differently. A shot that hits an enemy increments only
 * `shotsHit` (via {@link applyEnemyDamage}) and does not increment any
 * taxonomy counter.
 *
 * @property damageDealt - Cumulative damage applied to enemies.
 * @property shotsFired - Total bolts the player fired during the episode.
 * @property shotsHit - Total bolts that struck an enemy.
 * @property aimMissRate - Fraction of shots that missed: `(shotsFired - shotsHit) / shotsFired`, or `0` when no shots were fired.
 * @property shotsWallHit - Shots that terminated on a wall with no enemy near the bolt path.
 * @property shotsRangeExpired - Shots that expired at max range with no enemy near the bolt path.
 * @property shotsBlindFire - Shots fired when no active enemy exists in the world.
 * @property shotsNearMiss - Shots that passed near an enemy but did not hit it.
 */
export interface EpisodeTelemetry {
  /** Cumulative damage applied to enemies during the episode. */
  damageDealt: number;
  /** Total number of bolts fired by the player. */
  shotsFired: number;
  /** Total number of bolts that struck an enemy. */
  shotsHit: number;
  /**
   * Fraction of shots that missed: `(shotsFired - shotsHit) / shotsFired`.
   * Returns `0` when no shots have been fired.
   */
  aimMissRate: number;
  /**
   * Shots that terminated on a wall with no enemy near the bolt path.
   *
   * Incremented when the bolt hits a wall and no active enemy was within the
   * near-miss threshold of the bolt's travel path.
   */
  shotsWallHit: number;
  /**
   * Shots that expired at maximum range with no enemy near the bolt path.
   *
   * Incremented when the bolt reaches its max travel distance without hitting
   * a wall or enemy, and no active enemy was within the near-miss threshold.
   */
  shotsRangeExpired: number;
  /**
   * Shots fired when no active enemy exists in the world.
   *
   * Incremented when the player fires but there are no living enemies to
   * shoot at, indicating completely blind fire.
   */
  shotsBlindFire: number;
  /**
   * Shots that passed near an enemy but did not hit it.
   *
   * Incremented when an active enemy was within the near-miss threshold of
   * the bolt's travel path but outside the hit radius, indicating the player
   * was aiming at an enemy but missed.
   */
  shotsNearMiss: number;
  /**
   * Number of ammo pickups collected during the episode (P2S1).
   *
   * Counts discrete pickup collection events, not total ammo units restored.
   * Used by the fitness composite to apply a small opportunistic bonus
   * without letting ammo collection dominate combat rewards.
   */
  ammoPickupsCollected?: number;
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
  /**
   * Persistent neon impact marks left on enemies by bolt hits.
   *
   * Optional for backward compatibility with existing state factories that do
   * not initialize it; callers should use `enemyImpacts ?? []` when reading.
   */
  enemyImpacts?: EnemyImpactSpot[];
  /** Current weapon overlay state, including screen-space recoil. */
  gun?: GunState;
  /** Active traveling plasma bolts in the world. */
  bolts?: BoltState[];
  /** Active traveling enemy plasma bolts in the world. */
  enemyBolts?: EnemyBoltState[];
  /** Active ammo pickups dropped by dying enemies. */
  ammoPickups?: AmmoPickupState[];
  /** Total confirmed kills for scoring and evolution pressure. */
  kills: number;
  /**
   * Monotonic counter of hero deaths (health reaching zero).
   *
   * Incremented each time the player respawns after health depletion. The
   * interactive game respawns the hero at the map center with full health
   * and ammo; the evolution harness may use this counter for fitness scoring.
   */
  deaths?: number;
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
  /**
   * Per-episode combat telemetry accumulated by `fireBolt` and
   * `applyEnemyDamage`.
   *
   * Optional for backward compatibility with existing state factories that do
   * not initialize it; combat functions initialize it to zero-valued defaults
   * when absent. The fitness harness reads these counters after the episode
   * completes.
   */
  telemetry?: EpisodeTelemetry;
  /**
   * Per-tick flag indicating whether the most recent shot hit an enemy.
   *
   * Reset to `false` at the start of each game tick and set to `true` when
   * a plasma bolt strikes an active enemy during the bolt-enemy collision
   * pass. The sensor system reads this as input [14] for the main-agent
   * NEAT network.
   *
   * Optional for backward compatibility with existing state factories.
   */
  lastShotHit?: boolean;
}
