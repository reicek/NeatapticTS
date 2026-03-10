/**
 * Bird, trail, and champion-highlight rendering constants.
 *
 * This module centralizes all per-bird visual treatments so appearance tuning
 * stays decoupled from simulation and UI layout logic.
 */

/** Maximum number of trail points retained per bird trail polyline. */
export const FLAPPY_TRAIL_MAX_POINTS = 40;

/** Maximum number of trail points retained for the champion-only short trail. */
export const FLAPPY_CHAMPION_TRAIL_MAX_POINTS = 12;

/** Stroke width used for per-bird trail line rendering. */
export const FLAPPY_TRAIL_LINE_WIDTH_PX = 1.5;

/** Distance from world edge over which trails fade to transparent (pixels). */
export const FLAPPY_TRAIL_EDGE_FADE_DISTANCE_PX = 48;

/** Opacity used for non-champion bird fills and trails. */
export const FLAPPY_NON_CHAMPION_OPACITY = 0.1;

/** Ratio of bird side length used for top-left shine square. */
export const FLAPPY_BIRD_SHINE_SIZE_RATIO = 0.42;

/** Ratio of bird side length used to inset shine from top-left corner. */
export const FLAPPY_BIRD_SHINE_INSET_RATIO = 0.2;

/** Fill style used for the bird shine highlight. */
export const FLAPPY_BIRD_SHINE_FILL_STYLE = 'rgba(255, 255, 255, 0.7)';

/** Fill style used for the champion bird shine highlight (xterm 196 red). */
export const FLAPPY_BIRD_CHAMPION_SHINE_FILL_STYLE = 'rgba(255, 0, 0, 0.7)';

/** Neon glow color used for the condensed white highlight shine. */
export const FLAPPY_BIRD_WHITE_SHINE_GLOW_COLOR = '#ffffff';

/** Neon glow color used for the champion shine highlight (xterm 196 red). */
export const FLAPPY_BIRD_CHAMPION_SHINE_GLOW_COLOR = '#ff0000';

/** Neon blur radius used for the condensed white highlight shine. */
export const FLAPPY_BIRD_WHITE_SHINE_GLOW_BLUR_PX = 2;

/** Base neon blur radius used for bird body glow. */
export const FLAPPY_BIRD_BODY_GLOW_BLUR_PX = 10;

/** Glow blur radius used for simplified non-champion bird bodies. */
export const FLAPPY_NON_CHAMPION_BODY_GLOW_BLUR_PX = 0;

/**
 * Opacity used for the extra Radiant-style aura around each bird.
 *
 * This is intentionally subtle: it should read as a soft bloom that lifts the
 * bird off the background, without turning the bird into a big glowing blob.
 */
export const FLAPPY_BIRD_AURA_ALPHA = 0.16;

/** Pixel expansion used for the Radiant-style bird aura plate. */
export const FLAPPY_BIRD_AURA_EXPAND_PX = 3;

/** Blur multiplier used for the Radiant-style bird aura plate. */
export const FLAPPY_BIRD_AURA_BLUR_MULTIPLIER = 2.6;

/** Additional blur radius applied to the champion red body glow. */
export const FLAPPY_BIRD_CHAMPION_EXTRA_GLOW_BLUR_PX = 7;

/** Opacity used for the expanded champion red glow plate. */
export const FLAPPY_BIRD_CHAMPION_RED_GLOW_ALPHA = 0.62;

/** Pixel expansion used for the champion red glow plate. */
export const FLAPPY_BIRD_CHAMPION_RED_GLOW_EXPAND_PX = 4;

/** Minimum horizontal segment length used by stepped trail rendering. */
export const FLAPPY_TRAIL_MIN_HORIZONTAL_SEGMENT_PX = 2;

/** Minimum vertical segment length used by stepped trail rendering. */
export const FLAPPY_TRAIL_MIN_VERTICAL_SEGMENT_PX = 2;

/** Additional radius applied when drawing champion leader ring. */
export const FLAPPY_LEADER_RING_RADIUS_OFFSET_PX = 2;

/** Stroke width used when drawing champion leader ring. */
export const FLAPPY_LEADER_RING_LINE_WIDTH_PX = 2;

/** Neon blur radius used when drawing the champion outline stroke. */
export const FLAPPY_LEADER_RING_GLOW_BLUR_PX = 10;
