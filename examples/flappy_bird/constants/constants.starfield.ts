import { FLAPPY_WORLD_HEIGHT_PX } from './constants.world';

/**
 * Parallax starfield rendering constants.
 *
 * These values tune tile sizing, cyan star tint, and per-layer scroll ratios
 * for the Radiant-style background drawn behind the gameplay world.
 */

/** Width of one repeated starfield tile (pixels). */
export const FLAPPY_STARFIELD_TILE_WIDTH_PX = 512;

/** Minimum positive dimension allowed for generated starfield canvases. */
export const FLAPPY_STARFIELD_MIN_DIMENSION_PX = 1;

/** Zero pixel origin used for starfield canvas clears and draws. */
export const FLAPPY_STARFIELD_ORIGIN_PX = 0;

/** Canvas rendering context identifier used for starfield tile generation. */
export const FLAPPY_STARFIELD_CANVAS_CONTEXT_ID = '2d';

/** Composite mode used for normal starfield canvas drawing passes. */
export const FLAPPY_STARFIELD_COMPOSITE_SOURCE_OVER = 'source-over';

/** Transparent shadow reset value used after glow drawing. */
export const FLAPPY_STARFIELD_TRANSPARENT_SHADOW_COLOR = 'transparent';

/** Default fully opaque alpha restored after tile generation. */
export const FLAPPY_STARFIELD_FULL_ALPHA = 1;

/** Blur reset value used after the star glow pass. */
export const FLAPPY_STARFIELD_NO_BLUR_PX = 0;

/** Inclusive range offset used when converting size bounds into random spans. */
export const FLAPPY_STARFIELD_INCLUSIVE_RANGE_OFFSET = 1;

/** Divisor that normalizes unsigned 32-bit RNG state into the unit interval. */
export const FLAPPY_STARFIELD_UNSIGNED_NORMALIZATION_DIVISOR = 0x1_0000_0000;

/** Primary left-shift used by the xorshift32 random transition. */
export const FLAPPY_STARFIELD_XORSHIFT_LEFT_SHIFT_PRIMARY = 13;

/** Middle right-shift used by the xorshift32 random transition. */
export const FLAPPY_STARFIELD_XORSHIFT_RIGHT_SHIFT = 17;

/** Final left-shift used by the xorshift32 random transition. */
export const FLAPPY_STARFIELD_XORSHIFT_LEFT_SHIFT_FINAL = 5;

/** Height of one repeated starfield tile (pixels). */
export const FLAPPY_STARFIELD_TILE_HEIGHT_PX = FLAPPY_WORLD_HEIGHT_PX;

/** Fill color used for cyan stars and glow dots in the generated tile. */
export const FLAPPY_STARFIELD_CYAN_FILL_STYLE = 'rgba(95, 255, 255, 1)';

/** Horizontal scroll ratio for the farthest parallax layer. */
export const FLAPPY_STARFIELD_FAR_SCROLL_RATIO = 0.03;

/** Horizontal scroll ratio for the middle parallax layer. */
export const FLAPPY_STARFIELD_MID_SCROLL_RATIO = 0.04;

/** Horizontal scroll ratio for the nearest parallax layer. */
export const FLAPPY_STARFIELD_NEAR_SCROLL_RATIO = 0.05;

/**
 * Declarative layer specs for the default starfield parallax bands.
 *
 * Keeping the layer recipe in data form makes it easier to swap the lower
 * segment to a different parallax family without changing the tile builder.
 */
export const FLAPPY_STARFIELD_LAYER_SPECS = [
  {
    seed: 1_337,
    scrollRatio: FLAPPY_STARFIELD_FAR_SCROLL_RATIO,
    starCount: 35,
    minSizePx: 1,
    maxSizePx: 2,
    minAlpha: 0.08,
    maxAlpha: 0.22,
    blurPx: 4,
  },
  {
    seed: 2_777,
    scrollRatio: FLAPPY_STARFIELD_MID_SCROLL_RATIO,
    starCount: 28,
    minSizePx: 1,
    maxSizePx: 3,
    minAlpha: 0.1,
    maxAlpha: 0.28,
    blurPx: 6,
  },
  {
    seed: 4_242,
    scrollRatio: FLAPPY_STARFIELD_NEAR_SCROLL_RATIO,
    starCount: 23,
    minSizePx: 2,
    maxSizePx: 4,
    minAlpha: 0.12,
    maxAlpha: 0.34,
    blurPx: 8,
  },
] as const;
