/** Number of horizontal depth bands used by the neon ground grid. */
export const FLAPPY_GROUND_GRID_HORIZONTAL_LINE_COUNT = 16;

/** Target spacing between perspective-ray anchors at the bottom edge (pixels). */
export const FLAPPY_GROUND_GRID_TARGET_VERTICAL_LINE_SPACING_PX = 80;

/** Minimum visible perspective-ray count used on very narrow viewports. */
export const FLAPPY_GROUND_GRID_MIN_VERTICAL_LINE_COUNT = 8;

/** Extra off-screen perspective rays drawn for seamless wrap. */
export const FLAPPY_GROUND_GRID_VERTICAL_OVERFLOW_COUNT = 3;

/** Target screen-space height for one vertical-ray style segment (pixels). */
export const FLAPPY_GROUND_GRID_TARGET_VERTICAL_SEGMENT_HEIGHT_PX = 24;

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