/** Number of horizontal depth bands used by the neon ground grid. */
export const FLAPPY_GROUND_GRID_HORIZONTAL_LINE_COUNT = 16;

/** Target spacing between perspective-ray anchors at the bottom edge (pixels). */
export const FLAPPY_GROUND_GRID_TARGET_VERTICAL_LINE_SPACING_PX = 320;

/** Minimum visible perspective-ray count used on very narrow viewports. */
export const FLAPPY_GROUND_GRID_MIN_VERTICAL_LINE_COUNT = 2;

/** Extra off-screen perspective rays drawn for seamless wrap. */
export const FLAPPY_GROUND_GRID_VERTICAL_OVERFLOW_COUNT = 1;

/** Target screen-space height for one vertical-ray style segment (pixels). */
export const FLAPPY_GROUND_GRID_TARGET_VERTICAL_SEGMENT_HEIGHT_PX = 24;

/** Approximate playback frame duration used for deterministic pulse timing. */
export const FLAPPY_GROUND_GRID_APPROX_FRAME_DURATION_MS = 1000 / 60;

/** Interval between visible pulse events (milliseconds). */
export const FLAPPY_GROUND_GRID_PULSE_INTERVAL_MS = 3000;

/** Lifetime of one pulse as it travels across its chosen line (milliseconds). */
export const FLAPPY_GROUND_GRID_PULSE_LIFETIME_MS = 1100;

/** Glow blur applied behind each visible pulse square (pixels). */
export const FLAPPY_GROUND_GRID_PULSE_GLOW_BLUR_PX = 8;

/** Peak opacity used by visible pulse squares. */
export const FLAPPY_GROUND_GRID_PULSE_ALPHA = 0.92;

/** Smallest visible pulse square size (pixels). */
export const FLAPPY_GROUND_GRID_PULSE_MIN_SIZE_PX = 1;

/** Largest visible pulse square size (pixels). */
export const FLAPPY_GROUND_GRID_PULSE_MAX_SIZE_PX = 6;

/** Earliest progress ratio allowed for vertical pulse travel. */
export const FLAPPY_GROUND_GRID_VERTICAL_PULSE_START_RATIO = 0.14;

/** Latest progress ratio allowed for vertical pulse travel. */
export const FLAPPY_GROUND_GRID_VERTICAL_PULSE_END_RATIO = 0.62;

/** Minimum horizontal line thickness eligible for pulse travel. */
export const FLAPPY_GROUND_GRID_PULSE_MIN_ELIGIBLE_THICKNESS_PX = 1.5;

/** Earliest eligible slice of horizontal lines used for visible pulse picks. */
export const FLAPPY_GROUND_GRID_PULSE_PREFERRED_HORIZONTAL_START_RATIO = 0.4;

/** Horizontal inset that keeps vertical pulse picks away from clipped edges. */
export const FLAPPY_GROUND_GRID_PULSE_VISIBLE_VIEWPORT_INSET_PX = 24;

/** Size multiplier used for the soft glow halo behind one pulse square. */
export const FLAPPY_GROUND_GRID_PULSE_GLOW_SIZE_MULTIPLIER = 2.4;

/** Alpha ratio used for the soft glow halo behind one pulse square. */
export const FLAPPY_GROUND_GRID_PULSE_GLOW_ALPHA_RATIO = 0.45;

/** Normalization divisor used for deterministic pulse hash generation. */
export const FLAPPY_GROUND_GRID_UNSIGNED_NORMALIZATION_DIVISOR = 0x1_0000_0000;

/** Non-linear exponent used to compress depth lines toward the horizon. */
export const FLAPPY_GROUND_GRID_DEPTH_CURVE_EXPONENT = 2.35;

/** Scroll ratio applied to the moving vertical perspective rays. */
export const FLAPPY_GROUND_GRID_SCROLL_RATIO = 0.16;

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

/** Height ratio reserved for the subtle lower-band neon fog wash. */
export const FLAPPY_GROUND_GRID_FOG_HEIGHT_RATIO = 0.45;

/** Peak opacity used by the lower-band neon fog wash. */
export const FLAPPY_GROUND_GRID_FOG_ALPHA = 0.24;
