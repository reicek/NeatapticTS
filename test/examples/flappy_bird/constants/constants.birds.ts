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

/**
 * Additional blur radius applied to the champion red body glow.
 *
 * This keeps the champion body bloom aligned with the same neon blur
 * intensity used by the pipe outline glow, so the leader reads with
 * comparable visual weight.
 */
export const FLAPPY_BIRD_CHAMPION_EXTRA_GLOW_BLUR_PX =
	Math.max(24, Math.round(FLAPPY_BIRD_BODY_GLOW_BLUR_PX * 3.2)) -
	FLAPPY_BIRD_BODY_GLOW_BLUR_PX;

/** Minimum horizontal segment length used by stepped trail rendering. */
export const FLAPPY_TRAIL_MIN_HORIZONTAL_SEGMENT_PX = 2;

/** Minimum vertical segment length used by stepped trail rendering. */
export const FLAPPY_TRAIL_MIN_VERTICAL_SEGMENT_PX = 2;

