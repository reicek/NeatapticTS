import { FLAPPY_NEON_PALETTE } from '../../../constants/constants';
import type { PlaybackHorizonStyle } from './playback.background.types';

/**
 * Background layout ratio reserved for the starfield sky band.
 *
 * The top band intentionally occupies most of the scene so the future ground
 * layer can take over the lower strip without competing with the stars.
 */
export const FLAPPY_BACKGROUND_SKY_HEIGHT_RATIO = 2 / 3;

/**
 * Minimum safe viewport dimension used by background render math (pixels).
 *
 * Canvas helpers in this module assume positive dimensions. Clamping tiny or
 * temporarily zero-sized layouts to this floor prevents resize races from
 * producing invalid cache keys or negative geometry.
 */
export const FLAPPY_BACKGROUND_MIN_VIEWPORT_DIMENSION_PX = 1;

/**
 * Index offset used to draw one extra tile before the visible left edge.
 *
 * Starting one tile early hides wrap seams when the parallax offset lands near
 * a tile boundary and the camera reveals a sliver of content just off-screen.
 */
export const FLAPPY_BACKGROUND_TILE_ROW_START_INDEX = -1;

/**
 * Extra tile count rendered past the visible right edge for seamless wrap.
 *
 * The starfield is drawn as repeated cached strips. One buffered strip beyond
 * the viewport prevents empty columns from appearing while the parallax offset
 * advances between frames.
 */
export const FLAPPY_BACKGROUND_TILE_ROW_BUFFER_COUNT = 1;

/**
 * Half-thickness multiplier used to center the horizon line on the split.
 *
 * The layout computes the horizon around the sky/lower-band seam, so this
 * multiplier converts stroke thickness into the offset needed to center the
 * divider on that seam rather than placing it fully below it.
 */
export const FLAPPY_BACKGROUND_HORIZON_HALF_THICKNESS_MULTIPLIER = 0.5;

/**
 * Divisor used to detect odd stroke widths for pixel snapping.
 *
 * Canvas 2D strokes look soft when odd-width lines are left on whole pixels.
 * This constant supports the classic half-pixel alignment check used to keep
 * the horizon divider visually crisp.
 */
export const FLAPPY_BACKGROUND_ODD_STROKE_DIVISOR = 2;

/**
 * Pixel offset used to align odd-width strokes to the device pixel grid.
 *
 * Offsetting odd-width lines by half a pixel is a standard raster technique
 * for reducing blur in canvas line rendering.
 */
export const FLAPPY_BACKGROUND_ODD_STROKE_ALIGNMENT_OFFSET_PX = 0.5;

/**
 * Composite mode used for standard opaque drawing passes.
 *
 * Most background passes should replace pixels normally so the scene remains
 * predictable before selective glow passes are added on top.
 */
export const FLAPPY_BACKGROUND_COMPOSITE_SOURCE_OVER = 'source-over';

/**
 * Composite mode used when stacking glow-heavy starfield layers.
 *
 * The background currently resets to ordinary compositing for main passes, but
 * this constant documents the additive blend mode used when glow layers need to
 * visually accumulate rather than overwrite one another.
 */
export const FLAPPY_BACKGROUND_COMPOSITE_LIGHTER = 'lighter';

/**
 * Transparent shadow color used to reset canvas glow state.
 *
 * Canvas shadow state is sticky, so explicit transparent resets prevent one
 * glow-heavy pass from leaking blur into later solid fills or line work.
 */
export const FLAPPY_BACKGROUND_TRANSPARENT_SHADOW_COLOR = 'transparent';

/**
 * Thickness of the neon horizon divider line (pixels).
 *
 * A slightly heavier stroke helps the divider remain legible against both the
 * starfield and the bright grid below it.
 */
export const FLAPPY_BACKGROUND_HORIZON_LINE_THICKNESS_PX = 4;

/**
 * Soft glow opacity applied during the horizon glow pass.
 *
 * The glow is intentionally strong enough to read as neon, but still shy of a
 * full opaque bloom so the crisp core line remains visible.
 */
export const FLAPPY_BACKGROUND_HORIZON_GLOW_ALPHA = 0.92;

/**
 * Blur radius used to bloom the horizon divider glow (pixels).
 *
 * This is the main control for how far the horizon's light appears to bleed
 * into the neighboring sky and ground bands.
 */
export const FLAPPY_BACKGROUND_HORIZON_GLOW_BLUR_PX = 14;

/**
 * Frozen neon paint bundle reused by the playback horizon renderer.
 *
 * Keeping this style object in the constants module prevents repeated
 * allocation during every background frame while still keeping the palette
 * centrally theme-owned.
 */
export const FLAPPY_BACKGROUND_HORIZON_STYLE: PlaybackHorizonStyle =
  Object.freeze({
    lineColor: FLAPPY_NEON_PALETTE.horizonLine,
    glowColor: FLAPPY_NEON_PALETTE.horizonGlow,
    glowAlpha: FLAPPY_BACKGROUND_HORIZON_GLOW_ALPHA,
    glowBlurPx: FLAPPY_BACKGROUND_HORIZON_GLOW_BLUR_PX,
    lineThicknessPx: FLAPPY_BACKGROUND_HORIZON_LINE_THICKNESS_PX,
  });
