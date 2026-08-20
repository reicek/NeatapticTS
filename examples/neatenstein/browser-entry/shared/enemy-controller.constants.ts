/**
 * @module enemy-controller.constants
 *
 * Extracted constants for the enemy controller module.
 *
 * Single source of truth for the shared `DIRECTIONS` cardinal-direction
 * array, the `PREVIOUS_STEP_DISTANCE_SENTINEL` and `CELL_CENTER_OFFSET`
 * values used across multiple controller util files, eight-direction
 * compass labels, and all gameplay-tuned `ENEMY_CONTROLLER_*` constants.
 */

import { NEATENSTEIN_ENEMY_COLLISION_RADIUS_CELLS } from '../host/game/constants';

/**
 * Sentinel value indicating no previous-step distance is available.
 *
 * Used in `previousStepDistance` when an enemy is on its first tick or has
 * just respawned. Consolidated from the inline `-1` literal that appeared
 * in five or more controller files.
 */
export const PREVIOUS_STEP_DISTANCE_SENTINEL = -1;

/**
 * Half-cell offset used to snap coordinates to a cell's center.
 *
 * Applied during pre-move centering, post-move corridor centering, and
 * collision retry logic throughout the movement executor.
 */
export const CELL_CENTER_OFFSET = 0.5;

/**
 * Four cardinal direction vectors `[dx, dy]` in N, E, S, W order.
 *
 * The order matters for tie-breaking: when two neighbours have the same
 * BFS distance, the first one in this array wins (strict `<` comparison),
 * so east is preferred over south on equal distances. Single source of
 * truth — `enemy-navigation.ts` and `enemy-controller.move.utils.ts`
 * both re-export or import from here.
 */
export const DIRECTIONS: readonly (readonly [number, number])[] = [
  [0, -1], // N
  [1, 0], // E
  [0, 1], // S
  [-1, 0], // W
];

/**
 * Number of equally spaced yaw directions in the sprite sheet.
 *
 * Used by the sprite system, snapshot renderer, and reference snapshot
 * generator to determine how many compass views are rendered.
 */
export const NUM_DIRECTIONS = 8;

/**
 * Compass label for the north direction, corresponding to sprite yaw index 0.
 */
export const DIR_N = 'N';

/**
 * Compass label for the north-west direction, corresponding to sprite yaw
 * index 1.
 */
export const DIR_NW = 'NW';

/**
 * Compass label for the west direction, corresponding to sprite yaw index 2.
 */
export const DIR_W = 'W';

/**
 * Compass label for the south-west direction, corresponding to sprite yaw
 * index 3.
 */
export const DIR_SW = 'SW';

/**
 * Compass label for the south direction, corresponding to sprite yaw index 4.
 */
export const DIR_S = 'S';

/**
 * Compass label for the south-east direction, corresponding to sprite yaw
 * index 5.
 */
export const DIR_SE = 'SE';

/**
 * Compass label for the east direction, corresponding to sprite yaw index 6.
 */
export const DIR_E = 'E';

/**
 * Compass label for the north-east direction, corresponding to sprite yaw
 * index 7.
 */
export const DIR_NE = 'NE';

/**
 * Number of sim ticks the muzzle-flash shoot blink lasts on the upper body.
 *
 * When an enemy fires, the upper body shows the shoot frame for this many
 * ticks before reverting to the walk frame. Must be in the 3–5 range per
 * AC-10f-005.
 */
export const ENEMY_CONTROLLER_SHOOT_BLINK_TICKS = 4;

/**
 * Enemy movement speed in world cells per second, tuned so evolved enemies
 * close distance quickly without overshooting the player at short range.
 */
export const ENEMY_CONTROLLER_SPEED_CELLS_PER_SECOND = 2.5;

/**
 * Collision radius in world cells.
 *
 * Enemies have a 192×192 block footprint and each floor cell spans 252
 * blocks, giving a radius of 96 / 252 ≈ 0.381 cells.
 */
export const ENEMY_CONTROLLER_RADIUS_CELLS =
  NEATENSTEIN_ENEMY_COLLISION_RADIUS_CELLS;

/**
 * Wall collision radius for BFS navigation movement, in world cells.
 *
 * Set to a quarter cell (0.25) so the center of an agent always stays at
 * least 0.25 cells away from any wall edge.
 */
export const ENEMY_CONTROLLER_WALL_COLLISION_RADIUS_CELLS = 0.25;

/** Maximum cell distance at which an enemy will attempt to fire. */
export const ENEMY_CONTROLLER_FIRE_RANGE_CELLS = 8;

/**
 * Minimum cell distance the enemy tries to maintain from the player.
 *
 * Enemies chase the player but stop at this distance to prevent overlapping
 * the player's position.
 */
export const ENEMY_CONTROLLER_STOP_DISTANCE_CELLS = 1.5;

/**
 * Radius within which enemies switch from BFS approach to flanking
 * (circling toward an assigned slot angle around the player).
 */
export const ENEMY_CONTROLLER_FLANKING_RADIUS_CELLS = 3.5;

/**
 * Cooldown between hitscan shots in milliseconds, preventing enemies from
 * firing continuously and giving the player a predictable rhythm.
 */
export const ENEMY_CONTROLLER_FIRE_COOLDOWN_MS = 1000;

/** Hit points removed from the player by a single enemy hitscan shot. */
export const ENEMY_CONTROLLER_HITSCAN_DAMAGE = 10;

/**
 * Starting ammunition for each freshly tracked enemy, depleted by hitscan
 * shots and used to trigger the de-rez death animation.
 */
export const ENEMY_CONTROLLER_STARTING_AMMO = 3;

/**
 * Duration of the death de-rez animation in milliseconds, kept identical to
 * the sprite-side timing so visual and AI states stay synchronized.
 *
 * The 700 ms window delivers a fast Tron-style pixel scatter rather than the
 * original 4-second fade, keeping combat feedback snappy.
 */
export const ENEMY_CONTROLLER_DE_REZ_DURATION_MS = 700;
