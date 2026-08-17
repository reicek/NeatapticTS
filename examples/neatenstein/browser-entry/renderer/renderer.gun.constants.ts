/**
 * Gun overlay constants extracted from {@link module:./gun}.
 *
 * Centralises the gun body aspect ratio and re-exports the shared opaque
 * alpha constant from `browser-entry/constants.ts`.
 *
 * @module
 */

/**
 * Fully opaque alpha value for non-transparent gun overlay pixels.
 *
 * Re-exports the shared `RGBA_OPAQUE_ALPHA` from `browser-entry/constants.ts`
 * as the single source of truth.
 */
export { RGBA_OPAQUE_ALPHA } from '../constants';

/**
 * Gun body aspect ratio (width / height).
 *
 * Derived from the wide Wolfenstein-style chaingun reference silhouette so
 * the weapon reads as a horizontally elongated rotary cannon regardless of
 * viewport width.
 */
export const GUN_BODY_ASPECT_RATIO = 1.6;
