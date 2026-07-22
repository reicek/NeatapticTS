/**
 * Gameplay constants for the Neatenstein host-side simulation.
 *
 * These values are locked by the Phase 2 design consensus and are the
 * authoritative source for tick cadence, player limits, enemy concurrency,
 * dash timing, and episode bounds. They are kept in a single file so that
 * red-phase contract tests can assert their exact values before the rest of
 * the game logic consumes them.
 *
 * @module
 */

/** Fixed simulation timestep in milliseconds (≈60 Hz). */
export const NEATENSTEIN_FIXED_TIMESTEP_MS = 16;

/** Maximum player health at the start of an episode. */
export const NEATENSTEIN_PLAYER_MAX_HEALTH = 100;

/** Maximum ammo the player can carry at the start of an episode. */
export const NEATENSTEIN_PLAYER_MAX_AMMO = 30;

/** Maximum number of enemies that can be active at the same time. */
export const NEATENSTEIN_ENEMY_MAX_CONCURRENT = 8;

/** Milliseconds of invulnerability granted by a single dash. */
export const NEATENSTEIN_DASH_INVULNERABILITY_MS = 200;

/**
 * Minimum milliseconds between consecutive dashes.
 *
 * Must remain strictly greater than {@link NEATENSTEIN_DASH_INVULNERABILITY_MS}
 * so the cooldown is observable after the i-frame window ends.
 */
export const NEATENSTEIN_DASH_COOLDOWN_MS = 500;

/** Minimum duration of a default episode in milliseconds. */
export const NEATENSTEIN_EPISODE_MIN_DURATION_MS = 15_000;

/** Maximum duration of a default episode in milliseconds. */
export const NEATENSTEIN_EPISODE_MAX_DURATION_MS = 25_000;

/**
 * Minimum number of evolutionary generations the harness must support per
 * minute of wall-clock time. Used to sanity-check episode length plus
 * evaluation overhead.
 */
export const NEATENSTEIN_MIN_GENERATIONS_PER_MINUTE = 2;
