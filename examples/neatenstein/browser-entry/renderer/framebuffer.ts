/**
 * Framebuffer utilities for the Neatenstein neon CPU renderer.
 *
 * The CPU tier renders into a flat RGBA framebuffer backed by a
 * `Uint8ClampedArray`. Helpers here define shared renderer constants and
 * provide safe dimension and integer-clamping utilities used by the wall,
 * floor, and distance-fog passes.
 *
 * A framebuffer stores four bytes per pixel in RGBA order:
 *
 * ```ts
 * const offset = (y * width + x) * 4;
 * framebuffer[offset + 0] = red;
 * framebuffer[offset + 1] = green;
 * framebuffer[offset + 2] = blue;
 * framebuffer[offset + 3] = alpha;
 * ```
 *
 * @module
 */

/**
 * Number of color channels stored per pixel in the CPU framebuffer.
 *
 * The framebuffer layout is always RGBA.
 */
export const NEATENSTEIN_FRAMEBUFFER_CHANNELS = 4;

/**
 * Maximum view distance for the Neatenstein neon renderer.
 *
 * Walls at or beyond this distance are fully absorbed into the background
 * color by the distance-fog pass.
 */
export const NEATENSTEIN_MAX_VIEW_DIST = 30;

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
 * Resolved dimensions of a flat RGBA framebuffer.
 */
export interface NeatensteinFramebufferSize {
  /** Width in pixels. */
  width: number;
  /** Height in pixels. */
  height: number;
}

/**
 * Return whether dimensions describe a drawable framebuffer area.
 *
 * @param width - Candidate framebuffer width in pixels.
 * @param height - Candidate framebuffer height in pixels.
 * @returns Whether both dimensions are positive finite integers.
 */
export function isValidNeatensteinFramebufferSize(
  width: number,
  height: number,
): boolean {
  return (
    Number.isInteger(width) &&
    Number.isInteger(height) &&
    width > 0 &&
    height > 0
  );
}

/**
 * Return whether a framebuffer has exactly enough bytes for the given size.
 *
 * @param framebuffer - Flat RGBA pixel buffer.
 * @param width - Framebuffer width in pixels.
 * @param height - Framebuffer height in pixels.
 * @returns Whether `framebuffer.length === width * height * 4`.
 */
export function hasExactNeatensteinFramebufferByteLength(
  framebuffer: Uint8ClampedArray,
  width: number,
  height: number,
): boolean {
  if (!isValidNeatensteinFramebufferSize(width, height)) {
    return false;
  }

  return (
    framebuffer.length === width * height * NEATENSTEIN_FRAMEBUFFER_CHANNELS
  );
}

/**
 * Clamp a numeric value to an integer in the inclusive `[min, max]` range.
 *
 * The value is truncated toward zero before clamping, so floating-point
 * raycaster bounds such as `drawStart` and `drawEnd` can be passed directly.
 *
 * Non-finite values are handled defensively:
 *
 * - `NaN` returns `min`
 * - `-Infinity` returns `min`
 * - `Infinity` returns `max`
 *
 * Bounds are normalized with `Math.trunc`. If `min > max`, the bounds are
 * swapped so the function remains total.
 *
 * @param value - Value to truncate and clamp.
 * @param min - Inclusive lower bound.
 * @param max - Inclusive upper bound.
 * @returns Integer clamped into the normalized range.
 *
 * @example
 * ```ts
 * clampInt(12.9, 0, 10); // 10
 * clampInt(-2.2, 0, 10); // 0
 * clampInt(4.8, 0, 10); // 4
 * ```
 */
function clampInt(value: number, min: number, max: number): number {
  const rawMin = Number.isFinite(min) ? Math.trunc(min) : 0;
  const rawMax = Number.isFinite(max) ? Math.trunc(max) : rawMin;

  // Normalize reversed bounds so callers do not need to pre-sort them.
  const lower = Math.min(rawMin, rawMax);
  const upper = Math.max(rawMin, rawMax);

  if (Number.isNaN(value) || value === Number.NEGATIVE_INFINITY) {
    return lower;
  }

  if (value === Number.POSITIVE_INFINITY) {
    return upper;
  }

  const truncated = Math.trunc(value);

  if (truncated < lower) {
    return lower;
  }

  if (truncated > upper) {
    return upper;
  }

  return truncated;
}

export { clampInt };
