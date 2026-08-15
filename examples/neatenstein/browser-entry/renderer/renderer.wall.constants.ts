/**
 * Wall rendering constants extracted from {@link module:./walls}.
 *
 * Re-exports the shared RGBA channel count and opaque alpha constant from
 * `browser-entry/constants.ts` as the single source of truth.
 *
 * @module
 */

/**
 * Number of RGBA channels per framebuffer pixel.
 *
 * Re-exports the shared `RGBA_CHANNELS` from `browser-entry/constants.ts`.
 */
export { RGBA_CHANNELS } from '../constants';

/**
 * Fully opaque alpha value written for wall pixels.
 *
 * Re-exports the shared `RGBA_OPAQUE_ALPHA` from `browser-entry/constants.ts`.
 */
export { RGBA_OPAQUE_ALPHA } from '../constants';