/**
 * Shading and lighting executors extracted from the floor renderer.
 *
 * Contains pure leaf functions that compute alpha from depth, resolve cached
 * RGBA stroke styles, and parse hex colors into RGB tuples.
 *
 * @module
 */

import { FLAPPY_NEON_PALETTE } from '../../../flappy_bird/constants/constants.palette';
import {
  NEATENSTEIN_BACKGROUND_RGB,
  NEATENSTEIN_RENDER_DISTANCE_CAP,
  resolveNeatensteinFogFactor,
} from './framebuffer';
import { strokeNeatensteinFloorBand } from './floor.band.utils';
import type {
  NeatensteinFloorRenderContext,
  NeatensteinFloorSegmentBuffer,
} from './renderer.floor.types';
import {
  NEATENSTEIN_FLOOR_ALPHA_BANDS,
  NEATENSTEIN_FLOOR_MIN_ALPHA,
  NEATENSTEIN_FLOOR_MAX_ALPHA,
  NEATENSTEIN_FLOOR_ALPHA_CACHE_PRECISION,
  NEATENSTEIN_FLOOR_GLOW_METHOD,
  NEATENSTEIN_FLOOR_LINE_WIDTH_PX,
  NEATENSTEIN_FLOOR_GLOW_WIDTH_PX,
  NEATENSTEIN_FLOOR_GLOW_ALPHA_MULTIPLIER,
  NEATENSTEIN_FLOOR_SHADOW_BLUR_PX,
} from './renderer.floor.constants';
import { clamp } from '../shared/math-guards.utils';

// Re-export previously-public symbols that moved to dedicated files.
export {
  NEATENSTEIN_FLOOR_MIN_ALPHA,
  NEATENSTEIN_FLOOR_MAX_ALPHA,
} from './renderer.floor.constants';

/** Fallback stroke style used when the palette color cannot be parsed as hex. */
const NEATENSTEIN_FLOOR_FALLBACK_STROKE_STYLE =
  FLAPPY_NEON_PALETTE.groundGridLine;

/** Parsed RGB components of the shared neon floor line color. */
const FLOOR_BASE_RGB = parseNeatensteinFloorHexColor(
  FLAPPY_NEON_PALETTE.groundGridLine,
);

/** Shared neon glow color from the palette. */
const NEATENSTEIN_FLOOR_SHADOW_COLOR = FLAPPY_NEON_PALETTE.groundGridGlow;

/**
 * Cache of computed RGBA stroke styles keyed by quantized alpha and RGB color.
 *
 * The RGB color varies per depth band because fog blending shifts the base
 * color toward the background at greater distances, so both RGB and alpha are
 * part of the cache key.
 */
const NEATENSTEIN_FLOOR_STROKE_STYLE_CACHE = new Map<string, string>();

/**
 * Resolve floor grid color with smoothstep fog applied to RGB channels (C1.3).
 *
 * Blends the base grid color toward the background void color using the shared
 * smoothstep fog factor ({@link resolveNeatensteinFogFactor}). At
 * `FOG_START_DISTANCE` the fog factor is 0 (base color unchanged); at
 * `RENDER_DISTANCE_CAP` the fog factor is 1 (fully background color).
 *
 * @param r - Base color red channel [0, 255].
 * @param g - Base color green channel [0, 255].
 * @param b - Base color blue channel [0, 255].
 * @param distance - World-space distance from the camera.
 * @returns Fogged RGB color `{ r, g, b }`.
 */
export function resolveNeatensteinFloorFoggedColor(
  r: number,
  g: number,
  b: number,
  distance: number,
): { r: number; g: number; b: number } {
  const fogFactor = resolveNeatensteinFogFactor(distance);
  const invFog = 1 - fogFactor;

  return {
    r: r * invFog + NEATENSTEIN_BACKGROUND_RGB.r * fogFactor,
    g: g * invFog + NEATENSTEIN_BACKGROUND_RGB.g * fogFactor,
    b: b * invFog + NEATENSTEIN_BACKGROUND_RGB.b * fogFactor,
  };
}

/**
 * Resolve neon alpha for one depth band.
 *
 * @param depthRatio - Normalized depth where `0` is far and `1` is near.
 * @returns Opacity for the rendered grid band.
 */
export function resolveNeatensteinFloorAlpha(depthRatio: number): number {
  const clampedRatio = clamp(
    Number.isFinite(depthRatio) ? depthRatio : 0,
    0,
    1,
  );

  return (
    NEATENSTEIN_FLOOR_MIN_ALPHA +
    (NEATENSTEIN_FLOOR_MAX_ALPHA - NEATENSTEIN_FLOOR_MIN_ALPHA) * clampedRatio
  );
}

/**
 * Resolve floor grid alpha from world-space distance using the unified
 * smoothstep fog factor.
 *
 * Replaces the depth-band alpha system with a single continuous fog
 * derivation: `alpha = MAX_ALPHA * (1 - fogFactor)`. At
 * `FOG_START_DISTANCE` the fog factor is 0 (full alpha); at
 * `RENDER_DISTANCE_CAP` the fog factor is 1 (alpha ≈ 0).
 *
 * @param distance - World-space distance from the camera.
 * @returns Alpha value in `[0, MAX_ALPHA]`.
 */
export function resolveNeatensteinFloorAlphaFromDistance(
  distance: number,
): number {
  const fogFactor = resolveNeatensteinFogFactor(distance);
  return NEATENSTEIN_FLOOR_MAX_ALPHA * (1 - fogFactor);
}

/**
 * Resolve an alpha-aware RGBA stroke style for the neon grid.
 *
 * If the configured palette color cannot be parsed as RGB, this falls back to
 * the raw palette stroke color.
 *
 * @param baseRgb - Parsed RGB tuple from the shared palette, or `null`.
 * @param alpha - Desired opacity.
 * @returns Canvas stroke style string.
 */
export function resolveNeatensteinFloorStrokeStyle(
  baseRgb: { r: number; g: number; b: number } | null,
  alpha: number,
): string {
  /* istanbul ignore next -- FLOOR_BASE_RGB is always non-null with the valid palette color */
  if (baseRgb === null) {
    return NEATENSTEIN_FLOOR_FALLBACK_STROKE_STYLE;
  }

  // alpha always comes from resolveNeatensteinFloorAlpha (clamped to [0,1]),
  // so the non-finite fallback branch is unreachable dead code.
  /* istanbul ignore next -- alpha is always finite from resolveNeatensteinFloorAlpha */
  const clampedAlpha = clamp(Number.isFinite(alpha) ? alpha : 1, 0, 1);
  const alphaKey = clampedAlpha.toFixed(
    NEATENSTEIN_FLOOR_ALPHA_CACHE_PRECISION,
  );

  const cacheKey = `${baseRgb.r},${baseRgb.g},${baseRgb.b},${alphaKey}`;

  const cached = NEATENSTEIN_FLOOR_STROKE_STYLE_CACHE.get(cacheKey);
  if (cached !== undefined) {
    return cached;
  }

  const style = `rgba(${baseRgb.r}, ${baseRgb.g}, ${baseRgb.b}, ${alphaKey})`;
  NEATENSTEIN_FLOOR_STROKE_STYLE_CACHE.set(cacheKey, style);

  return style;
}

/**
 * Parse a six-digit hex color into RGB components.
 *
 * Accepts both `#rrggbb` and `rrggbb`.
 *
 * @param baseColor - Hex color string.
 * @returns Parsed RGB tuple, or `null` if parsing fails.
 */
export function parseNeatensteinFloorHexColor(
  baseColor: string,
): { r: number; g: number; b: number } | null {
  const match = /^#?([0-9a-fA-F]{6})$/.exec(baseColor);

  /* istanbul ignore next -- palette color is always valid 6-digit hex */
  if (match === null) {
    return null;
  }

  const hexDigits = match[1];
  const red = Number.parseInt(hexDigits.slice(0, 2), 16);
  const green = Number.parseInt(hexDigits.slice(2, 4), 16);
  const blue = Number.parseInt(hexDigits.slice(4, 6), 16);

  /* istanbul ignore next -- parseInt on valid hex digits always returns finite values */
  if (
    !Number.isFinite(red) ||
    !Number.isFinite(green) ||
    !Number.isFinite(blue)
  ) {
    return null;
  }

  return { r: red, g: green, b: blue };
}

/**
 * Stroke every non-empty depth band using the configured glow method.
 *
 * @param ctx - Canvas-like rendering context.
 * @param bands - Depth-banded flat segment buffers.
 */
export function strokeNeatensteinGridBands(
  ctx: NeatensteinFloorRenderContext,
  bands: NeatensteinFloorSegmentBuffer[],
): void {
  ctx.save();
  ctx.shadowColor = NEATENSTEIN_FLOOR_SHADOW_COLOR;

  for (let bandIndex = 0; bandIndex < bands.length; bandIndex += 1) {
    const segments = bands[bandIndex];

    if (segments.length === 0) {
      continue;
    }

    const bandRatio = (bandIndex + 0.5) / NEATENSTEIN_FLOOR_ALPHA_BANDS;
    const coreAlpha = resolveNeatensteinFloorAlpha(bandRatio);

    // C1.3: Apply smoothstep fog to the grid color per band. Each band's
    // representative distance is derived from its depth ratio (0 = far,
    // 1 = near), so distant bands fade toward the background void.
    const representativeDistance =
      NEATENSTEIN_RENDER_DISTANCE_CAP * (1 - bandRatio);

    let foggedRgb: { r: number; g: number; b: number };
    /* istanbul ignore else -- FLOOR_BASE_RGB is always non-null with the valid palette color */
    if (FLOOR_BASE_RGB !== null) {
      foggedRgb = resolveNeatensteinFloorFoggedColor(
        FLOOR_BASE_RGB.r,
        FLOOR_BASE_RGB.g,
        FLOOR_BASE_RGB.b,
        representativeDistance,
      );
    } else {
      foggedRgb = NEATENSTEIN_BACKGROUND_RGB;
    }

    /* istanbul ignore else -- constant-controlled alternative glow path */
    if (NEATENSTEIN_FLOOR_GLOW_METHOD === 'double-stroke') {
      // Pass 1: wide, dim halo. This creates a predictable neon glow without
      // relying on compositor-specific shadow blur performance.
      ctx.lineWidth = NEATENSTEIN_FLOOR_GLOW_WIDTH_PX;
      ctx.shadowBlur = 0;
      ctx.strokeStyle = resolveNeatensteinFloorStrokeStyle(
        foggedRgb,
        coreAlpha * NEATENSTEIN_FLOOR_GLOW_ALPHA_MULTIPLIER,
      );
      strokeNeatensteinFloorBand(ctx, segments);

      // Pass 2: narrow, bright core line.
      ctx.lineWidth = NEATENSTEIN_FLOOR_LINE_WIDTH_PX;
      ctx.shadowBlur = 0;
      ctx.strokeStyle = resolveNeatensteinFloorStrokeStyle(
        foggedRgb,
        coreAlpha,
      );
      strokeNeatensteinFloorBand(ctx, segments);
    } else {
      // Alternative softer glow path. Kept behind a constant so experiments can
      // switch glow style without changing projection or batching logic.
      ctx.lineWidth = NEATENSTEIN_FLOOR_LINE_WIDTH_PX;
      ctx.shadowBlur = NEATENSTEIN_FLOOR_SHADOW_BLUR_PX;
      ctx.strokeStyle = resolveNeatensteinFloorStrokeStyle(
        foggedRgb,
        coreAlpha,
      );
      strokeNeatensteinFloorBand(ctx, segments);
    }
  }

  ctx.restore();
}