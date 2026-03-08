/**
 * Background layout ratio reserved for the starfield sky band.
 *
 * The top band intentionally occupies most of the scene so the future ground
 * layer can take over the lower strip without competing with the stars.
 */
export const FLAPPY_BACKGROUND_SKY_HEIGHT_RATIO = 2 / 3;

/** Minimum safe viewport dimension used by background render math (pixels). */
export const FLAPPY_BACKGROUND_MIN_VIEWPORT_DIMENSION_PX = 1;

/** Index offset used to draw one extra tile before the visible left edge. */
export const FLAPPY_BACKGROUND_TILE_ROW_START_INDEX = -1;

/** Extra tile count rendered past the visible right edge for seamless wrap. */
export const FLAPPY_BACKGROUND_TILE_ROW_BUFFER_COUNT = 1;

/** Half-thickness multiplier used to center the horizon line on the split. */
export const FLAPPY_BACKGROUND_HORIZON_HALF_THICKNESS_MULTIPLIER = 0.5;

/** Divisor used to detect odd stroke widths for pixel snapping. */
export const FLAPPY_BACKGROUND_ODD_STROKE_DIVISOR = 2;

/** Pixel offset used to align odd-width strokes to the device pixel grid. */
export const FLAPPY_BACKGROUND_ODD_STROKE_ALIGNMENT_OFFSET_PX = 0.5;

/** Composite mode used for standard opaque drawing passes. */
export const FLAPPY_BACKGROUND_COMPOSITE_SOURCE_OVER = 'source-over';

/** Composite mode used when stacking glow-heavy starfield layers. */
export const FLAPPY_BACKGROUND_COMPOSITE_LIGHTER = 'lighter';

/** Transparent shadow color used to reset canvas glow state. */
export const FLAPPY_BACKGROUND_TRANSPARENT_SHADOW_COLOR = 'transparent';

/** Thickness of the neon horizon divider line (pixels). */
export const FLAPPY_BACKGROUND_HORIZON_LINE_THICKNESS_PX = 4;

/** Soft glow opacity applied during the horizon glow pass. */
export const FLAPPY_BACKGROUND_HORIZON_GLOW_ALPHA = 0.92;

/** Blur radius used to bloom the horizon divider glow (pixels). */
export const FLAPPY_BACKGROUND_HORIZON_GLOW_BLUR_PX = 14;
