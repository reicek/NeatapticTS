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

import {
  NEATENSTEIN_BACKGROUND_RGB,
  NEATENSTEIN_FOG_START_DISTANCE,
  NEATENSTEIN_RENDER_DISTANCE_CAP,
  RGBA_CHANNELS,
} from './renderer.framebuffer.constants';
import { clampInt } from '../shared/math-guards.utils';

// Re-export constants and types for external consumers.
export {
  NEATENSTEIN_BACKGROUND_RGB,
  NEATENSTEIN_FOG_START_DISTANCE,
  NEATENSTEIN_FRAMEBUFFER_CHANNELS,
  NEATENSTEIN_MAX_VIEW_DIST,
  NEATENSTEIN_RENDER_DISTANCE_CAP,
  RGBA_CHANNELS,
} from './renderer.framebuffer.constants';
export type { NeatensteinFramebufferSize } from './renderer.framebuffer.types';

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

  return framebuffer.length === width * height * RGBA_CHANNELS;
}

/**
 * Resolve a distance-based fog factor shared by the floor, ceiling, and sprite
 * renderers.
 *
 * Returns `0` (no fog) at distances up to
 * {@link NEATENSTEIN_FOG_START_DISTANCE}, then smoothly ramps to `1` (fully
 * fogged) at {@link NEATENSTEIN_RENDER_DISTANCE_CAP} using a smoothstep curve.
 * Non-finite distances are treated as fully fogged so malformed projections
 * fade into the background instead of producing invalid color channels.
 *
 * @param distance - World-space distance from the camera.
 * @returns Fog interpolation factor in `[0, 1]`.
 */
export function resolveNeatensteinFogFactor(distance: number): number {
  if (!Number.isFinite(distance)) {
    return 1;
  }

  if (distance >= NEATENSTEIN_RENDER_DISTANCE_CAP) {
    return 1;
  }

  if (distance <= NEATENSTEIN_FOG_START_DISTANCE) {
    return 0;
  }

  const range =
    NEATENSTEIN_RENDER_DISTANCE_CAP - NEATENSTEIN_FOG_START_DISTANCE;
  const t = (distance - NEATENSTEIN_FOG_START_DISTANCE) / range;

  // Smoothstep: t * t * (3 - 2 * t) for a C1-continuous transition.
  return t * t * (3 - 2 * t);
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
