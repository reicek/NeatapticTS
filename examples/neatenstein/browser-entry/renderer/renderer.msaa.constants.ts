/**
 * MSAA (multi-sample anti-aliasing) constants for the Neatenstein renderer
 * (C1.6).
 *
 * 2× MSAA resolve cleans wall-sprite seams cheaply by averaging two
 * sub-sample colors per resolved column. The fog blend helper reuses the
 * shared smoothstep fog factor so MSAA-resolved colors stay consistent with
 * the floor and wall fog curves (Invariant §7).
 *
 * @module
 */

import { resolveNeatensteinFogFactor } from './framebuffer';

/**
 * Number of sub-samples per resolved column for MSAA.
 *
 * 2× MSAA is the cheapest meaningful anti-aliasing pass: two sub-samples are
 * averaged to produce the resolved color, removing single-pixel seams between
 * walls and sprites.
 */
export const NEATENSTEIN_MSAA_SAMPLE_COUNT = 2 as const;

/**
 * RGB color tuple used by MSAA resolve helpers.
 */
interface NeatensteinRgb {
  r: number;
  g: number;
  b: number;
}

/**
 * Resolve one MSAA column by averaging two sub-sample colors (C1.6).
 *
 * Each channel is averaged and clamped into `[0, 255]` so the result is always
 * a valid 8-bit color value.
 *
 * @param sampleA - First sub-sample color.
 * @param sampleB - Second sub-sample color.
 * @returns Averaged and clamped RGB color.
 *
 * @example
 * ```ts
 * const resolved = resolveNeatensteinMsaaResolvedColumn(
 *   { r: 0, g: 183, b: 255 },
 *   { r: 10, g: 142, b: 160 },
 * );
 * // { r: 5, g: 162.5, b: 207.5 }
 * ```
 */
export function resolveNeatensteinMsaaResolvedColumn(
  sampleA: NeatensteinRgb,
  sampleB: NeatensteinRgb,
): NeatensteinRgb {
  const avgR = (sampleA.r + sampleB.r) / 2;
  const avgG = (sampleA.g + sampleB.g) / 2;
  const avgB = (sampleA.b + sampleB.b) / 2;

  return {
    r: Math.max(0, Math.min(255, avgR)),
    g: Math.max(0, Math.min(255, avgG)),
    b: Math.max(0, Math.min(255, avgB)),
  };
}

/**
 * Blend an MSAA-resolved color toward the fog background using the shared
 * smoothstep fog factor (C1.6, Invariant §7).
 *
 * The fog factor is derived from the same `resolveNeatensteinFogFactor` used
 * by walls and floor, ensuring consistent depth-fading across all rendering
 * paths.
 *
 * @param resolved - MSAA-resolved RGB color.
 * @param distance - World-space distance from the camera.
 * @param background - Background RGB to blend toward.
 * @returns Fog-blended RGB color.
 */
export function resolveNeatensteinMsaaFogBlend(
  resolved: NeatensteinRgb,
  distance: number,
  background: NeatensteinRgb,
): NeatensteinRgb {
  const fogFactor = resolveNeatensteinFogFactor(distance);
  const invFog = 1 - fogFactor;

  return {
    r: resolved.r * invFog + background.r * fogFactor,
    g: resolved.g * invFog + background.g * fogFactor,
    b: resolved.b * invFog + background.b * fogFactor,
  };
}