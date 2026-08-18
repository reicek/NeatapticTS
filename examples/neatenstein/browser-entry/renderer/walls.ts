/**
 * Neon wall column renderer for the Neatenstein CPU-tier raycaster.
 *
 * This module implements the CPU ImageData path for wall columns. Columns are
 * written directly into a shared `Uint8ClampedArray` framebuffer, avoiding
 * per-column `fillRect`, `globalAlpha` mutations, and other canvas state churn.
 *
 * For optimal performance, callers should use {@link writeNeonWallColumn} for
 * each wall stripe and then flush the finished framebuffer once with
 * `ctx.putImageData(...)`.
 *
 * @module
 */

import {
  clampInt,
  NEATENSTEIN_BACKGROUND_RGB,
  resolveNeatensteinFogFactor,
} from './framebuffer';
import { RGBA_CHANNELS, RGBA_OPAQUE_ALPHA } from './renderer.wall.constants';
import type { ParsedRgb } from './renderer.wall.types';

// Re-export constants and types for external consumers.
export { RGBA_CHANNELS, RGBA_OPAQUE_ALPHA } from './renderer.wall.constants';
export type {
  NeatensteinWallRenderContext,
  ParsedRgb,
} from './renderer.wall.types';

/**
 * Cache of parsed wall colors.
 *
 * Wall colors are reused across many columns and frames, so caching avoids
 * repeated string parsing in the hot path.
 */
const WALL_COLOR_CACHE = new Map<string, ParsedRgb>();

/**
 * Return whether a number is a positive integer dimension.
 *
 * @param value - Candidate dimension.
 * @returns Whether the value is usable as a framebuffer dimension.
 */
function isPositiveIntegerDimension(value: number): boolean {
  return Number.isInteger(value) && value > 0;
}

/**
 * Parse a strict `#rrggbb` hex color string into RGB channels.
 *
 * @param hex - Color string in `#rrggbb` format.
 * @returns Parsed RGB triplet.
 * @throws {Error} When the string is not a valid `#rrggbb` color.
 */
function parseHexColor(hex: string): ParsedRgb {
  const cached = WALL_COLOR_CACHE.get(hex);
  if (cached !== undefined) {
    return cached;
  }

  const match = /^#([0-9a-fA-F]{6})$/.exec(hex);
  if (match === null) {
    throw new Error(`Expected #rrggbb hex color, got "${hex}"`);
  }

  const digits = match[1];
  const rgb = {
    r: Number.parseInt(digits.slice(0, 2), 16),
    g: Number.parseInt(digits.slice(2, 4), 16),
    b: Number.parseInt(digits.slice(4, 6), 16),
  };

  WALL_COLOR_CACHE.set(hex, rgb);
  return rgb;
}

/**
 * Resolve the wall fog factor for a perpendicular wall distance.
 *
 * Delegates to the shared smoothstep fog function so walls, floor, ceiling,
 * and sprites all use the same distance-based fog curve.
 *
 * @param perpWallDist - Perpendicular wall distance.
 * @returns Fog interpolation factor where `0` is near and `1` is fully fogged.
 */
function resolveWallFogFactor(perpWallDist: number): number {
  return resolveNeatensteinFogFactor(perpWallDist);
}

/**
 * Blend a base wall color toward the background color using distance fog.
 *
 * @param base - Base neon wall color.
 * @param fogT - Fog interpolation factor in `[0, 1]`.
 * @returns Fogged RGB color.
 */
function resolveFoggedWallColor(base: ParsedRgb, fogT: number): ParsedRgb {
  const { r: bgR, g: bgG, b: bgB } = NEATENSTEIN_BACKGROUND_RGB;
  const invFog = 1 - fogT;

  return {
    r: Math.round(base.r * invFog + bgR * fogT),
    g: Math.round(base.g * invFog + bgG * fogT),
    b: Math.round(base.b * invFog + bgB * fogT),
  };
}

/**
 * Compute the wall texture coordinate for a DDA wall hit.
 *
 * Returns the fractional part of the wall intersection point along the wall
 * surface, in `[0, 1)`. For an X-side hit (`side === 0`) the wall coordinate is
 * `posY + perpWallDist * dirY`; for a Y-side hit it is
 * `posX + perpWallDist * dirX`. This function computes the X-side variant
 * directly; callers pass the appropriate fixed coordinate and ray-perpendicular
 * component.
 *
 * @param fixedCoord - Camera coordinate perpendicular to the ray direction on
 *   the hit side (posY for X-side, posX for Y-side).
 * @param perpWallDist - Perpendicular wall distance from the DDA hit.
 * @param rayPerp - Ray direction component perpendicular to the hit side
 *   (dirY for X-side, dirX for Y-side).
 * @returns Texture coordinate in `[0, 1)` via `fract(wallX)`.
 */
export function computeWallTexcoord(
  fixedCoord: number,
  perpWallDist: number,
  rayPerp: number,
): number {
  const wallX = fixedCoord + perpWallDist * rayPerp;
  return wallX - Math.floor(wallX);
}

/**
 * Write a single neon wall column into the CPU ImageData framebuffer.
 *
 * This function performs no canvas flush. It is the preferred hot-path helper
 * for renderers that draw many columns and call `putImageData` once after the
 * framebuffer is complete.
 *
 * Distance fog linearly interpolates the wall color toward
 * {@link NEATENSTEIN_BACKGROUND_RGB} as `perpWallDist` approaches
 * {@link NEATENSTEIN_RENDER_DISTANCE_CAP}. Non-finite distances are treated as
 * fully fogged.
 *
 * When a `texcoord` in `[0, 1)` is provided (computed via
 * {@link computeWallTexcoord}), the column is shaded with a subtle vertical
 * stripe pattern derived from the texcoord. This breaks up the flat colour
 * without altering the neon hue, preserving the Invariant §3 aesthetic. When
 * no texcoord is supplied (or the value is outside `[0, 1)`), the column is
 * flat-shaded as before.
 *
 * @param framebuffer - Flat RGBA framebuffer.
 * @param framebufferWidth - Framebuffer width in pixels.
 * @param framebufferHeight - Framebuffer height in pixels.
 * @param column - Horizontal column index to write.
 * @param drawStart - Top row of the wall stripe, inclusive.
 * @param drawEnd - Bottom row of the wall stripe, exclusive.
 * @param hexColor - Wall color as `#rrggbb`.
 * @param perpWallDist - Perpendicular wall distance.
 * @param texcoord - Optional wall texture coordinate in `[0, 1)` from
 *   {@link computeWallTexcoord}. Drives vertical stripe shading.
 * @throws {Error} When `hexColor` is not a valid `#rrggbb` string.
 *
 * @example
 * ```ts
 * writeNeonWallColumn(framebuffer, 640, 480, 12, 120, 340, '#00bfff', 4.5, 0.37);
 * ```
 */
export function writeNeonWallColumn(
  framebuffer: Uint8ClampedArray,
  framebufferWidth: number,
  framebufferHeight: number,
  column: number,
  drawStart: number,
  drawEnd: number,
  hexColor: string,
  perpWallDist: number,
  texcoord: number = -1,
): void {
  if (
    !isPositiveIntegerDimension(framebufferWidth) ||
    !isPositiveIntegerDimension(framebufferHeight)
  ) {
    return;
  }

  // Do not clamp invalid columns to the edge; skip them instead.
  if (!Number.isFinite(column)) {
    return;
  }

  const x = Math.trunc(column);
  if (x < 0 || x >= framebufferWidth) {
    return;
  }

  const baseColor = parseHexColor(hexColor);
  const fogT = resolveWallFogFactor(perpWallDist);
  const foggedColor = resolveFoggedWallColor(baseColor, fogT);

  // Compute texcoord-driven brightness modulation. A sinusoidal pattern
  // creates subtle vertical striping along the wall surface, breaking up
  // the flat colour while preserving the neon hue (Invariant §3).
  const hasTexcoord = texcoord >= 0 && texcoord < 1;
  const brightness = hasTexcoord
    ? 1.0 - 0.10 * (1.0 - Math.cos(texcoord * Math.PI * 4))
    : 1.0;

  const finalColor = hasTexcoord
    ? {
        r: Math.round(foggedColor.r * brightness),
        g: Math.round(foggedColor.g * brightness),
        b: Math.round(foggedColor.b * brightness),
      }
    : foggedColor;

  const clampedStart = clampInt(drawStart, 0, framebufferHeight);
  const clampedEnd = clampInt(drawEnd, 0, framebufferHeight);

  if (clampedStart >= clampedEnd) {
    return;
  }

  for (let row = clampedStart; row < clampedEnd; row += 1) {
    const offset = (row * framebufferWidth + x) * RGBA_CHANNELS;

    // Defensive guard for mismatched framebuffer dimensions.
    if (offset + 3 >= framebuffer.length) {
      break;
    }

    framebuffer[offset] = finalColor.r;
    framebuffer[offset + 1] = finalColor.g;
    framebuffer[offset + 2] = finalColor.b;
    framebuffer[offset + 3] = RGBA_OPAQUE_ALPHA;
  }
}
