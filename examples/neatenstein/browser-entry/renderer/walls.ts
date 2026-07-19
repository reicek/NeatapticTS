/**
 * Neon wall column renderer for the Neatenstein CPU-tier raycaster.
 *
 * This module implements the CPU ImageData path: each wall column is written
 * directly into a shared `Uint8ClampedArray` framebuffer and flushed with a
 * single `putImageData` call. It deliberately avoids per-column `fillRect`,
 * `globalAlpha` mutations, and multiple flushes.
 *
 * @module
 */

import {
  clampInt,
  NEATENSTEIN_BACKGROUND_RGB,
  NEATENSTEIN_MAX_VIEW_DIST,
  resolveNeatensteinFramebufferSize,
} from './framebuffer';

/**
 * Minimal canvas-like context consumed by the CPU wall renderer.
 *
 * Only `putImageData` is required. The interface is intentionally narrow so
 * the renderer can be unit-tested with a lightweight mock and still accept a
 * real `CanvasRenderingContext2D` at runtime through structural typing.
 */
export interface NeatensteinWallRenderContext {
  /**
   * Flush an ImageData-like payload to the canvas.
   *
   * @param imageData - Object with `data`, `width`, and `height`.
   * @param dx - Destination X coordinate.
   * @param dy - Destination Y coordinate.
   */
  putImageData(
    imageData: { data: Uint8ClampedArray; width: number; height: number },
    dx: number,
    dy: number,
  ): void;
}

/**
 * Parsed RGB triplet from a `#rrggbb` hex color string.
 */
interface ParsedRgb {
  r: number;
  g: number;
  b: number;
}

/**
 * Parse a `#rrggbb` hex color string into an RGB triplet.
 *
 * @param hex - Color string in `#rrggbb` format.
 * @returns Parsed `{ r, g, b }` values in [0, 255].
 * @throws Error when the string is not a valid `#rrggbb` color.
 */
function parseHexColor(hex: string): ParsedRgb {
  if (hex.length !== 7 || hex[0] !== '#') {
    throw new Error(`Expected #rrggbb hex color, got "${hex}"`);
  }

  const r = Number.parseInt(hex.slice(1, 3), 16);
  const g = Number.parseInt(hex.slice(3, 5), 16);
  const b = Number.parseInt(hex.slice(5, 7), 16);

  if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) {
    throw new Error(`Invalid hex color components in "${hex}"`);
  }

  return { r, g, b };
}

/**
 * Render a single neon wall column into the CPU ImageData framebuffer.
 *
 * This is the CPU-tier render path. It writes the fogged wall color directly
 * into the supplied {@link framebuffer} for the requested column and flushes the
 * whole framebuffer with exactly one `putImageData` call. It never uses
 * `ctx.fillRect` or mutates `ctx.globalAlpha`.
 *
 * Distance fog linearly interpolates the wall color toward the background
 * color `{ r: 6, g: 11, b: 20 }` as `perpWallDist` approaches
 * {@link NEATENSTEIN_MAX_VIEW_DIST}. At `perpWallDist = 0` the wall keeps its
 * full base color; at or beyond the max view distance it becomes the
 * background color.
 *
 * @param framebuffer - Flat RGBA framebuffer (Uint8ClampedArray, square).
 * @param column - Horizontal column index to write.
 * @param drawStart - Top row of the wall stripe (inclusive, clamped to canvas).
 * @param drawEnd - Bottom row of the wall stripe (exclusive, clamped to canvas).
 * @param hexColor - Wall color as `#rrggbb`.
 * @param perpWallDist - Perpendicular wall distance; 0 = near, larger = farther.
 * @param ctx - Canvas-like context with `putImageData`.
 * @returns void
 * @throws {Error} when hexColor is not a valid `#rrggbb` string.
 *
 * @example
 * ```ts
 * const framebuffer = new Uint8ClampedArray(8 * 8 * 4);
 * renderNeonWallColumn(framebuffer, 0, 2, 5, '#00bfff', 1, ctx);
 * ```
 */
export function renderNeonWallColumn(
  framebuffer: Uint8ClampedArray,
  column: number,
  drawStart: number,
  drawEnd: number,
  hexColor: string,
  perpWallDist: number,
  ctx: NeatensteinWallRenderContext,
): void {
  const { r: baseR, g: baseG, b: baseB } = parseHexColor(hexColor);
  const { r: bgR, g: bgG, b: bgB } = NEATENSTEIN_BACKGROUND_RGB;

  const fogT = Math.min(
    Math.max(perpWallDist / NEATENSTEIN_MAX_VIEW_DIST, 0),
    1,
  );
  const invFog = 1 - fogT;

  const finalR = Math.round(baseR * invFog + bgR * fogT);
  const finalG = Math.round(baseG * invFog + bgG * fogT);
  const finalB = Math.round(baseB * invFog + bgB * fogT);

  const { width, height } = resolveNeatensteinFramebufferSize(framebuffer);
  const clampedColumn = clampInt(column, 0, width - 1);
  const clampedStart = clampInt(drawStart, 0, height);
  const clampedEnd = clampInt(drawEnd, 0, height);

  if (clampedStart < clampedEnd) {
    for (let row = clampedStart; row < clampedEnd; row++) {
      const offset = (row * width + clampedColumn) * 4;
      framebuffer[offset] = finalR;
      framebuffer[offset + 1] = finalG;
      framebuffer[offset + 2] = finalB;
      framebuffer[offset + 3] = 255;
    }
  }

  ctx.putImageData({ data: framebuffer, width, height }, 0, 0);
}
