/**
 * @module enemy-navigation.constants
 *
 * Extracted constants for the enemy navigation module.
 *
 * Owns the BFS sentinel values, vision/sensor tuning constants, and the
 * named sensor-index constants used by `extractSensors`. Re-exports
 * shared constants from `enemy-controller.constants` for convenience.
 */

// Re-export shared constants so navigation files can import from one place.
export {
  PREVIOUS_STEP_DISTANCE_SENTINEL,
  CELL_CENTER_OFFSET,
  NUM_DIRECTIONS,
} from './enemy-controller.constants';

/**
 * Sentinel value stored in the distance buffer for wall cells.
 *
 * Used by `buildEnemyDistanceMap` to mark cells that are solid walls and
 * therefore impassable for BFS pathfinding.
 */
export const WALL_VALUE = -2;

/**
 * Sentinel value stored in the distance buffer for unreachable cells.
 *
 * Cells that cannot be reached from the goal via BFS expansion retain this
 * value, indicating no path exists from the player to that cell.
 */
export const UNREACHABLE_VALUE = -1;

/**
 * Maximum vision range in grid cells for enemy detection.
 *
 * Enemies beyond this Euclidean distance from the player are invisible —
 * their sensor values are zeroed. Set to 15 (half the bolt max range of 30)
 * to force active exploration without making the hero omniscient.
 *
 * @see AC-P3S1b-003
 */
export const VISION_RANGE_CELLS = 15;

/**
 * Half-angle of the firing arc in radians (30°).
 *
 * An enemy is considered "in the firing arc" when the absolute relative
 * bearing is within this angle of the player's facing direction. Matches
 * the fallback AI's `NEATENSTEIN_FALLBACK_FIRE_ARC`.
 */
export const FIRING_ARC_HALF_ANGLE = Math.PI / 6;

/**
 * Scalar step per compass direction: `0` = N, `0.25` = E, `0.5` = S,
 * `0.75` = W.
 *
 * Multiplied by the best cardinal direction index to produce the compass
 * scalar in the NEAT vision vector.
 */
export const COMPASS_STEP = 0.25;

/**
 * Absolute clip applied to the step-delta when computing the progress signal.
 *
 * The raw `previousDistance − currentDistance` value is clamped to
 * `[−PROGRESS_CLIP, +PROGRESS_CLIP]` before scaling.
 */
export const PROGRESS_CLIP = 2;

/**
 * Scale applied to the clipped progress delta before adding to the neutral
 * baseline.
 *
 * The final progress value is `PROGRESS_NEUTRAL + clipped / PROGRESS_SCALE`.
 */
export const PROGRESS_SCALE = 4;

/**
 * Neutral progress value returned when progress cannot be computed.
 *
 * Represents the midpoint `[0, 1]` — no information about whether the enemy
 * is getting closer or farther from the player.
 */
export const PROGRESS_NEUTRAL = 0.5;

// ── Sensor indices ──────────────────────────────────────────────────────

/** Sensor index for the player health ratio. */
export const SENSOR_INDEX_PLAYER_HEALTH = 0;

/** Sensor index for the player ammo ratio. */
export const SENSOR_INDEX_PLAYER_AMMO = 1;

/** Sensor index for the player look angle. */
export const SENSOR_INDEX_PLAYER_LOOK_ANGLE = 2;

/** Sensor index for the player position X. */
export const SENSOR_INDEX_PLAYER_POS_X = 3;

/** Sensor index for the player position Y. */
export const SENSOR_INDEX_PLAYER_POS_Y = 4;

/** Sensor index for the nearest visible enemy bearing. */
export const SENSOR_INDEX_ENEMY_BEARING = 5;

/** Sensor index for the nearest visible enemy distance. */
export const SENSOR_INDEX_ENEMY_DISTANCE = 6;

/** Sensor index for the nearest visible enemy health ratio. */
export const SENSOR_INDEX_ENEMY_HEALTH = 7;

/** Sensor index for the wall raycast distance North. */
export const SENSOR_INDEX_WALL_NORTH = 8;

/** Sensor index for the wall raycast distance East. */
export const SENSOR_INDEX_WALL_EAST = 9;

/** Sensor index for the wall raycast distance South. */
export const SENSOR_INDEX_WALL_SOUTH = 10;

/** Sensor index for the wall raycast distance West. */
export const SENSOR_INDEX_WALL_WEST = 11;

/** Sensor index for the enemy-visible binary flag. */
export const SENSOR_INDEX_ENEMY_VISIBLE = 12;

/** Sensor index for the enemy-in-firing-arc binary flag. */
export const SENSOR_INDEX_ENEMY_IN_FIRING_ARC = 13;

/** Sensor index for the last-shot-hit binary flag. */
export const SENSOR_INDEX_LAST_SHOT_HIT = 14;

/** Sensor index for the low-ammo gate binary flag. */
export const SENSOR_INDEX_LOW_AMMO_GATE = 21;
