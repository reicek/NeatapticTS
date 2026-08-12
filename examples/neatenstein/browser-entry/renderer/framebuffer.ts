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

/**
 * Resolve a distance-based fog factor shared by the floor, ceiling, and sprite
 * renderers.
 *
 * Returns `0` (no fog) at distance `0` and `1` (fully fogged) at
 * {@link NEATENSTEIN_RENDER_DISTANCE_CAP}. Non-finite distances are treated as
 * fully fogged so malformed projections fade into the background instead of
 * producing invalid color channels.
 *
 * @param distance - World-space distance from the camera.
 * @returns Fog interpolation factor in `[0, 1]`.
 */
export function resolveNeatensteinFogFactor(distance: number): number {
  if (!Number.isFinite(distance)) {
    return 1;
  }

  return distance >= NEATENSTEIN_RENDER_DISTANCE_CAP ? 1 : 0;
}

/**
 * Blend a base RGB color toward the background fog color.
 *
 * @param base - Base RGB color to interpolate from.
 * @param fogFactor - Fog interpolation factor in `[0, 1]` where `0` is the base
 *   color and `1` is fully fogged.
 * @returns Fogged RGB color.
 */
export function resolveNeatensteinFoggedColor(
  base: { r: number; g: number; b: number },
  fogFactor: number,
): { r: number; g: number; b: number } {
  const { r: bgR, g: bgG, b: bgB } = NEATENSTEIN_BACKGROUND_RGB;
  const invFog = 1 - fogFactor;

  return {
    r: Math.round(base.r * invFog + bgR * fogFactor),
    g: Math.round(base.g * invFog + bgG * fogFactor),
    b: Math.round(base.b * invFog + bgB * fogFactor),
  };
}

export { clampInt };
