/**
 * Floor rendering constants extracted from the floor renderer modules.
 *
 * Centralises alpha/width/blur/projection values so they are defined once and
 * reused across {@link module:./floor}, {@link module:./floor.band.utils},
 * {@link module:./floor.projection.utils}, and
 * {@link module:./floor.shade.utils}.
 *
 * @module
 */

import { NEATENSTEIN_CANVAS_2D_CONTEXT } from '../constants';

/** Re-export the shared Canvas 2D context identifier for floor consumers. */
export { NEATENSTEIN_CANVAS_2D_CONTEXT };

/**
 * Test fallback canvas width used when the render context has no backing
 * `canvas`, such as lightweight mock contexts in unit tests.
 */
export const NEATENSTEIN_FLOOR_DEFAULT_WIDTH = 320;

/**
 * Test fallback canvas height used when the render context has no backing
 * `canvas`, such as lightweight mock contexts in unit tests.
 */
export const NEATENSTEIN_FLOOR_DEFAULT_HEIGHT = 240;

/**
 * Fraction of canvas height where the horizon line sits.
 *
 * The current renderer assumes a level camera, so the horizon is horizontal.
 * Everything below the horizon is floor; everything above it is ceiling.
 */
export const NEATENSTEIN_FLOOR_HORIZON_RATIO = 0.5;

/**
 * Vertical field of view in radians.
 *
 * The renderer treats this as the vertical FOV and derives focal length from
 * canvas height. The camera plane is scaled by the viewport aspect ratio so
 * horizontal FOV widens on wider screens.
 */
export const NEATENSTEIN_FLOOR_FOV_RADIANS = Math.PI / 3;

/**
 * Camera height above the floor in world units.
 *
 * This value controls how strongly floor and ceiling grid points project away
 * from the horizon.
 */
export const NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD = 0.5;

/**
 * Ratio of canvas height used as the camera height for separate screen-space
 * forced-perspective helpers, such as pulses or tracers.
 */
export const NEATENSTEIN_FLOOR_CAMERA_HEIGHT_SCREEN_RATIO = 0.5;

/**
 * Minimum positive camera-space depth required for projection.
 *
 * Points at or behind the camera plane are culled. The epsilon also avoids
 * extreme projected coordinates for samples that are nearly on the camera
 * plane.
 */
export const NEATENSTEIN_FLOOR_NEAR_PLANE_EPSILON = 0.001;

/**
 * Number of alpha bands used to batch floor/ceiling strokes.
 *
 * More bands produce smoother depth fading but require more canvas stroke
 * calls. This value keeps the effect visually graded while preserving batching.
 */
export const NEATENSTEIN_FLOOR_ALPHA_BANDS = 4;

/**
 * Number of samples per projected world grid line.
 *
 * Each integer grid line is sampled at `SAMPLES + 1` points and consecutive
 * visible samples are connected into screen-space line segments.
 */
export const NEATENSTEIN_FLOOR_LINE_SAMPLES = 80;

/** Minimum line opacity near the horizon. */
export const NEATENSTEIN_FLOOR_MIN_ALPHA = 0.05;

/** Maximum line opacity near the camera. */
export const NEATENSTEIN_FLOOR_MAX_ALPHA = 0.58;

/**
 * Number of fractional digits retained for cached alpha stroke styles.
 *
 * The renderer uses banded alpha values, so this quantization preserves visual
 * stability while preventing unbounded cache growth from tiny float
 * differences.
 */
export const NEATENSTEIN_FLOOR_ALPHA_CACHE_PRECISION = 4;

/**
 * Glow strategy used for the floor and ceiling grid lines.
 *
 * - `'double-stroke'` draws a wide low-alpha halo followed by a narrow core.
 * - `'shadow-blur'` uses Canvas shadow blur for a softer but less predictable
 *   compositor-dependent glow cost.
 */
export const NEATENSTEIN_FLOOR_GLOW_METHOD: 'double-stroke' | 'shadow-blur' =
  'double-stroke';

/** Width in pixels of the bright core grid line. */
export const NEATENSTEIN_FLOOR_LINE_WIDTH_PX = 2;

/** Width in pixels of the halo stroke used by the double-stroke glow mode. */
export const NEATENSTEIN_FLOOR_GLOW_WIDTH_PX = 4;

/** Alpha multiplier applied to the halo pass in double-stroke glow mode. */
export const NEATENSTEIN_FLOOR_GLOW_ALPHA_MULTIPLIER = 0.8;

/** Shadow blur radius used by the shadow-blur glow mode. */
export const NEATENSTEIN_FLOOR_SHADOW_BLUR_PX = 4;
