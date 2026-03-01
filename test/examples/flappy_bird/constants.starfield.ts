import { FLAPPY_WORLD_HEIGHT_PX } from './constants';

/**
 * Parallax starfield rendering constants.
 *
 * These values tune tile sizing, cyan star tint, and per-layer scroll ratios
 * for the Radiant-style background drawn behind the gameplay world.
 */

/** Width of one repeated starfield tile (pixels). */
export const FLAPPY_STARFIELD_TILE_WIDTH_PX = 512;

/** Height of one repeated starfield tile (pixels). */
export const FLAPPY_STARFIELD_TILE_HEIGHT_PX = FLAPPY_WORLD_HEIGHT_PX;

/** Fill color used for cyan stars and glow dots in the generated tile. */
export const FLAPPY_STARFIELD_CYAN_FILL_STYLE = 'rgba(95, 255, 255, 1)';

/** Horizontal scroll ratio for the farthest parallax layer. */
export const FLAPPY_STARFIELD_FAR_SCROLL_RATIO = 0.08;

/** Horizontal scroll ratio for the middle parallax layer. */
export const FLAPPY_STARFIELD_MID_SCROLL_RATIO = 0.12;

/** Horizontal scroll ratio for the nearest parallax layer. */
export const FLAPPY_STARFIELD_NEAR_SCROLL_RATIO = 0.18;
