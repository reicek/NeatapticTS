import { FLAPPY_NEON_PALETTE } from '../../../../constants/constants';
import type { PlaybackBackgroundGroundGridStyle } from './playback.background.ground-grid.types';

/**
 * Number of horizontal depth bands used by the neon ground grid.
 *
 * More bands increase the sense of depth, but they also thicken the lower band
 * visually and add more line work to each frame.
 */
export const FLAPPY_GROUND_GRID_HORIZONTAL_LINE_COUNT = 16;

/**
 * Target spacing between perspective-ray anchors at the bottom edge (pixels).
 *
 * This controls how wide the ground lanes feel near the viewer before they
 * converge toward the horizon.
 */
export const FLAPPY_GROUND_GRID_TARGET_VERTICAL_LINE_SPACING_PX = 320;

/**
 * Minimum visible perspective-ray count used on very narrow viewports.
 *
 * Even on a small canvas, the grid still needs at least a few rays to read as
 * perspective instead of a flat block of horizontal stripes.
 */
export const FLAPPY_GROUND_GRID_MIN_VERTICAL_LINE_COUNT = 2;

/**
 * Extra off-screen perspective rays drawn for seamless wrap.
 *
 * Overflow rays prevent the parallax cycle from exposing empty gaps when the
 * wrapped scroll offset lands near a lane boundary.
 */
export const FLAPPY_GROUND_GRID_VERTICAL_OVERFLOW_COUNT = 1;

/**
 * Target screen-space height for one vertical-ray style segment (pixels).
 *
 * Segmenting the rays lets the renderer vary alpha, thickness, and blur by
 * depth rather than drawing each ray with one flat style.
 */
export const FLAPPY_GROUND_GRID_TARGET_VERTICAL_SEGMENT_HEIGHT_PX = 24;

/**
 * Approximate playback frame duration used for deterministic pulse timing.
 *
 * The pulse system is designed to feel stable at ordinary browser animation
 * cadence without requiring access to wall-clock time in every helper.
 */
export const FLAPPY_GROUND_GRID_APPROX_FRAME_DURATION_MS = 1000 / 60;

/**
 * Interval between visible pulse events (milliseconds).
 *
 * A relatively slow cadence keeps the pulses as occasional accent lights rather
 * than a constant distraction under the birds.
 */
export const FLAPPY_GROUND_GRID_PULSE_INTERVAL_MS = 6000;

/**
 * Lifetime of one pulse as it travels across its chosen line (milliseconds).
 *
 * The lifetime is slightly shorter than the full interval so one pulse fades
 * out before the next slot becomes active.
 */
export const FLAPPY_GROUND_GRID_PULSE_LIFETIME_MS = 5900;

/** Peak opacity used by visible pulse squares. */
export const FLAPPY_GROUND_GRID_PULSE_ALPHA = 0.99;

/** Smallest visible pulse square size (pixels). */
export const FLAPPY_GROUND_GRID_PULSE_MIN_SIZE_PX = 1;

/** Largest visible pulse square size (pixels). */
export const FLAPPY_GROUND_GRID_PULSE_MAX_SIZE_PX = 6;

/**
 * Earliest progress ratio allowed for vertical pulse travel.
 *
 * Vertical pulses start a little away from the horizon so they are visible as
 * distinct squares instead of immediately disappearing into compressed depth.
 */
export const FLAPPY_GROUND_GRID_VERTICAL_PULSE_START_RATIO = 0.14;

/**
 * Latest progress ratio allowed for vertical pulse travel.
 *
 * Ending early keeps the pulse out of the extreme foreground, where its square
 * would become too large and visually heavy.
 */
export const FLAPPY_GROUND_GRID_VERTICAL_PULSE_END_RATIO = 0.62;

/** Minimum horizontal line thickness eligible for pulse travel. */
export const FLAPPY_GROUND_GRID_PULSE_MIN_ELIGIBLE_THICKNESS_PX = 1.5;

/**
 * Earliest eligible slice of horizontal lines used for visible pulse picks.
 *
 * This biases horizontal pulses toward the more legible near-midground instead
 * of the compressed lines nearest the horizon.
 */
export const FLAPPY_GROUND_GRID_PULSE_PREFERRED_HORIZONTAL_START_RATIO = 0.4;

/** Horizontal inset that keeps vertical pulse picks away from clipped edges. */
export const FLAPPY_GROUND_GRID_PULSE_VISIBLE_VIEWPORT_INSET_PX = 24;

/**
 * Normalization divisor used for deterministic pulse hash generation.
 *
 * The pulse selection helpers convert unsigned integer hashes into stable
 * floating-point picks in the unit interval.
 */
export const FLAPPY_GROUND_GRID_UNSIGNED_NORMALIZATION_DIVISOR = 0x1_0000_0000;

/**
 * Non-linear exponent used to compress depth lines toward the horizon.
 *
 * This is the main perspective stylization control: higher values bunch more of
 * the depth bands near the horizon and leave broader spacing near the viewer.
 */
export const FLAPPY_GROUND_GRID_DEPTH_CURVE_EXPONENT = 2.35;

/**
 * Near-edge horizontal line offset used for the lower-pipe floor illusion.
 *
 * `1` targets the first usable grid band above the bottom edge rather than the
 * terminal line that coincides with the lower-band boundary itself.
 */
export const FLAPPY_GROUND_GRID_PIPE_CONNECTION_LINE_OFFSET_FROM_BOTTOM = 1;

/**
 * Scroll ratio applied to the moving vertical perspective rays.
 *
 * Keeping the rays slower than gameplay motion makes the grid feel like a deep
 * environmental layer rather than a surface glued to the pipes.
 */
export const FLAPPY_GROUND_GRID_SCROLL_RATIO = 0.16;

/**
 * Decimal precision used when quantizing the wrapped vertical-ray offset.
 *
 * Quantization stabilizes cache reuse by preventing tiny floating-point drift
 * from generating effectively identical geometry variants.
 */
export const FLAPPY_GROUND_GRID_SCROLL_OFFSET_QUANTIZATION_DECIMALS = 3;

/** Minimum alpha used by the farthest horizontal depth lines. */
export const FLAPPY_GROUND_GRID_MIN_ALPHA = 0.12;

/** Maximum alpha used by the nearest depth and perspective lines. */
export const FLAPPY_GROUND_GRID_MAX_ALPHA = 0.58;

/** Blur radius used by the farthest depth lines near the horizon. */
export const FLAPPY_GROUND_GRID_MAX_BLUR_PX = 6;

/** Blur radius used by the nearest depth lines at the bottom edge. */
export const FLAPPY_GROUND_GRID_MIN_BLUR_PX = 0;

/** Minimum stroke width used by far depth lines. */
export const FLAPPY_GROUND_GRID_MIN_THICKNESS_PX = 1;

/** Maximum stroke width used by near depth lines. */
export const FLAPPY_GROUND_GRID_MAX_THICKNESS_PX = 3;

/**
 * Height ratio reserved for the subtle lower-band neon fog wash.
 *
 * The fog sits in the lower portion of the band so it enriches the foreground
 * without muting the crisp horizon seam.
 */
export const FLAPPY_GROUND_GRID_FOG_HEIGHT_RATIO = 0.45;

/**
 * Peak opacity used by the lower-band neon fog wash.
 *
 * The fog should tint the band, not obscure the line geometry, so the opacity
 * is intentionally modest.
 */
export const FLAPPY_GROUND_GRID_FOG_ALPHA = 0.24;

/**
 * Frozen neon style bundle reused by the playback ground-grid renderer.
 *
 * The values stay theme-owned but are materialized once so the renderer does
 * not allocate a new style object during every frame.
 */
export const FLAPPY_BACKGROUND_GROUND_GRID_STYLE: PlaybackBackgroundGroundGridStyle =
  Object.freeze({
    lineColor: FLAPPY_NEON_PALETTE.groundGridLine,
    fogColor: FLAPPY_NEON_PALETTE.groundGridFog,
    pulseFillColor: FLAPPY_NEON_PALETTE.groundGridPulseFill,
  });
