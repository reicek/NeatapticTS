/**
 * Framebuffer utilities for the Neatenstein neon CPU renderer.
 *
 * The CPU tier renders into a square RGBA framebuffer backed by a
 * `Uint8ClampedArray`. Helpers here own the view-distance constant and the
 * background color used by the distance-fog pass so that wall and floor
 * renderers share a single source of truth.
 *
 * @module
 */

/**
 * Maximum view distance for the Neatenstein neon renderer.
 *
 * Walls at or beyond this distance are fully absorbed into the background
 * color by the distance-fog pass.
 */
export const NEATENSTEIN_MAX_VIEW_DIST = 20;

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

/**
 * Resolved square dimensions of a flat RGBA framebuffer.
 */
export interface NeatensteinFramebufferSize {
  /** Width in pixels. */
  width: number;
  /** Height in pixels (same as width for the square CPU framebuffer). */
  height: number;
}

/**
 * Resolve the square width/height of a flat RGBA framebuffer.
 *
 * The framebuffer is square, so `width === height`. The size is derived from
 * the total byte length: `length / 4` pixels, then `sqrt(pixels)` for the side.
 * Non-square or malformed inputs are handled by integer truncation so the
 * renderer can still proceed defensively.
 *
 * @param framebuffer - Flat RGBA pixel buffer.
 * @returns The inferred `{ width, height }` dimensions.
 *
 * @example
 * ```ts
 * const framebuffer = new Uint8ClampedArray(8 * 8 * 4);
 * const { width, height } = resolveNeatensteinFramebufferSize(framebuffer);
 * console.log(width, height); // 8, 8
 * ```
 */
export function resolveNeatensteinFramebufferSize(
  framebuffer: Uint8ClampedArray,
): NeatensteinFramebufferSize {
  const width = Math.floor(Math.sqrt(framebuffer.length / 4));
  return { width, height: width };
}

/**
 * Clamp an integer to the inclusive `[min, max]` range.
 *
 * Values are truncated toward zero before clamping, so floating-point bounds
 * such as `drawEnd` from the raycaster are safe to pass directly.
 */
function clampInt(value: number, min: number, max: number): number {
  const truncated = Math.trunc(value);
  if (truncated < min) return min;
  if (truncated > max) return max;
  return truncated;
}

export { clampInt };
