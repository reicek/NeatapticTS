/**
 * Pure color utility executors extracted from the display worker.
 *
 * These helpers format RGB triples, resolve enemy team colors, and apply
 * distance fog to wall colors. They are pure functions with no dependency on
 * module-level mutable state.
 *
 * @module
 */

import {
  NEATENSTEIN_BACKGROUND_RGB,
  NEATENSTEIN_RENDER_DISTANCE_CAP,
} from '../renderer/framebuffer';
import {
  GOLDEN_ANGLE_DEG,
  COLOR_HSL_HUE,
  COLOR_HSL_SAT,
  COLOR_HSL_LIGHT,
  COLOR_HSL_ALPHA,
} from './display.worker.constants';

/**
 * Format an RGB triple as a CSS `rgb(...)` string.
 *
 * @param color - RGB color object.
 * @returns CSS color string.
 */
export function formatRgb(color: { r: number; g: number; b: number }): string {
  return `rgb(${Math.round(color.r)}, ${Math.round(color.g)}, ${Math.round(color.b)})`;
}

/**
 * Resolve a deterministic enemy team color `[r, g, b]` from the enemy type
 * index by rotating the HSL hue wheel with the golden angle. Palette indices
 * 5/6/7 in the robot sprite atlas are swapped with this color at render time,
 * giving each enemy a stable neon tint while preserving the sprite alpha.
 *
 * @param typeIndex - Enemy type index (the source {@link GameState.enemies}
 *   array position). Negative indices fall back to index 0.
 * @returns RGB triple in the `[0, 255]` range.
 */
export function resolveEnemyTeamColor(
  typeIndex: number,
): readonly [number, number, number] {
  const seed = typeIndex >= 0 ? typeIndex : 0;
  const hue = (seed * GOLDEN_ANGLE_DEG) % COLOR_HSL_HUE;
  const saturation = COLOR_HSL_SAT;
  const lightness = COLOR_HSL_LIGHT;
  const chroma = (1 - Math.abs(2 * lightness - 1)) * saturation;
  const huePrime = hue / COLOR_HSL_ALPHA;
  const intermediate = chroma * (1 - Math.abs((huePrime % 2) - 1));
  let r = 0;
  let g = 0;
  let b = 0;
  if (huePrime < 1) {
    r = chroma;
    g = intermediate;
  } else if (huePrime < 2) {
    r = intermediate;
    g = chroma;
  } else if (huePrime < 3) {
    g = chroma;
    b = intermediate;
  } else if (huePrime < 4) {
    g = intermediate;
    b = chroma;
  } else if (huePrime < 5) {
    r = intermediate;
    b = chroma;
  } else {
    r = chroma;
    b = intermediate;
  }
  const match = lightness - chroma / 2;
  return [
    Math.round((r + match) * 255),
    Math.round((g + match) * 255),
    Math.round((b + match) * 255),
  ] as const;
}

/**
 * Clamp a value to a `[min, max]` range.
 *
 * @param value - Value to clamp.
 * @param min - Inclusive minimum.
 * @param max - Inclusive maximum.
 * @returns The clamped value.
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Resolve a safe distance-fog interpolation factor.
 *
 * @param perpWallDist - Perpendicular wall distance.
 * @returns Fog factor in `[0, 1]`.
 */
export function resolveWallFogFactor(perpWallDist: number): number {
  return perpWallDist >= NEATENSTEIN_RENDER_DISTANCE_CAP ? 1 : 0;
}

/**
 * Apply distance fog to a neon wall color and return a CSS color string.
 *
 * @param wallColor - Raw wall RGB.
 * @param perpWallDist - Perpendicular distance from camera to wall.
 * @returns CSS `rgb(...)` string for `context.fillStyle`.
 */
export function applyWallFog(
  wallColor: { r: number; g: number; b: number },
  perpWallDist: number,
): string {
  const fogFactor = resolveWallFogFactor(perpWallDist);
  const bg = NEATENSTEIN_BACKGROUND_RGB;

  return formatRgb({
    r: wallColor.r + (bg.r - wallColor.r) * fogFactor,
    g: wallColor.g + (bg.g - wallColor.g) * fogFactor,
    b: wallColor.b + (bg.b - wallColor.b) * fogFactor,
  });
}
