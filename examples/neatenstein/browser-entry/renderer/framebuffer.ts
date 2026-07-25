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
 * Resolve the dimensions of a flat RGBA framebuffer.
 *
 * Prefer passing explicit `width` and `height` from the canvas or render target.
 * A flat buffer length alone is ambiguous for non-square framebuffers, so
 * explicit dimensions are required to correctly represent targets such as
 * `640x480`.
 *
 * Backwards compatibility:
 *
 * - If `width` and `height` are supplied and valid, they are returned.
 * - If explicit dimensions are omitted, this function falls back to legacy
 *   square inference using `sqrt(framebuffer.length / 4)`.
 * - If the buffer length does not represent a perfect square pixel count, the
 *   inferred side is truncated and may not cover the entire buffer.
 *
 * @param framebuffer - Flat RGBA pixel buffer.
 * @param width - Optional explicit framebuffer width in pixels.
 * @param height - Optional explicit framebuffer height in pixels.
 * @returns Resolved framebuffer dimensions.
 *
 * @example
 * ```ts
 * const framebuffer = new Uint8ClampedArray(640 * 480 * 4);
 * const size = resolveNeatensteinFramebufferSize(framebuffer, 640, 480);
 *
 * console.log(size.width, size.height);
 * // 640, 480
 * ```
 *
 * @example
 * ```ts
 * const squareFramebuffer = new Uint8ClampedArray(8 * 8 * 4);
 * const size = resolveNeatensteinFramebufferSize(squareFramebuffer);
 *
 * console.log(size.width, size.height);
 * // 8, 8
 * ```
 */
export function resolveNeatensteinFramebufferSize(
  framebuffer: Uint8ClampedArray,
  width?: number,
  height?: number,
): NeatensteinFramebufferSize {
  // Explicit dimensions are the only unambiguous representation for
  // non-square framebuffers. Use them whenever the caller provides valid values.
  if (
    width !== undefined &&
    height !== undefined &&
    isValidNeatensteinFramebufferSize(width, height)
  ) {
    return { width, height };
  }

  // Legacy fallback: infer a square side from the pixel count. This preserves
  // old callers but cannot represent rectangular render targets.
  const pixelCount = Math.floor(
    framebuffer.length / NEATENSTEIN_FRAMEBUFFER_CHANNELS,
  );
  const side = Math.floor(Math.sqrt(pixelCount));

  return { width: side, height: side };
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
