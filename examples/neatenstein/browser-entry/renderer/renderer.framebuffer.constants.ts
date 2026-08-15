/**
 * Framebuffer constants extracted from {@link module:./framebuffer}.
 *
 * Centralises the channel count, view-distance limits, and background color
 * so they can be shared across the wall, floor, and sprite renderer passes
 * without duplication.
 *
 * @module
 */

/**
 * Number of color channels stored per pixel in the CPU framebuffer.
 *
 * The framebuffer layout is always RGBA (red, green, blue, alpha). This
 * re-exports the shared `RGBA_CHANNELS` from `browser-entry/constants.ts` as
 * the single source of truth for the channel count.
 */
export { RGBA_CHANNELS } from '../constants';

/**
 * Backward-compatibility alias for the framebuffer channel count.
 *
 * Kept so existing test imports continue to resolve. Prefer
 * {@link RGBA_CHANNELS} in new code.
 */
export const NEATENSTEIN_FRAMEBUFFER_CHANNELS: number = 4;

/**
 * Maximum view distance for the Neatenstein neon renderer.
 *
 * Walls at or beyond this distance are fully absorbed into the background
 * color by the distance-fog pass. Perimeter walls at the spawn distance
 * (~60 map cells) therefore remain visible but faded instead of being
 * hard-clipped to invisibility.
 */
export const NEATENSTEIN_MAX_VIEW_DIST = 140;

/**
 * Hard render-distance cap shared by the raycaster, sprite projector, and wall
 * renderer.
 *
 * Rays and sprites beyond this distance are treated as empty/no-hit. This cap
 * prevents the DDA from walking indefinitely on open sight lines and keeps
 * distant geometry from being projected or drawn.
 */
export const NEATENSTEIN_RENDER_DISTANCE_CAP = 30;

/**
 * Background RGB used by the distance-fog pass.
 *
 * This is the dark neon void color `#060b14` from the Phase 1 design
 * consensus. Far wall columns are linearly interpolated toward this value.
 */
export const NEATENSTEIN_BACKGROUND_RGB = {
  r: 6,
  g: 11,
  b: 20,
} as const;