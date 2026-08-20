/**
 * World-fixed floor and ceiling grids for the Neatenstein neon raycasting demo.
 *
 * This module renders procedural neon grid lines by projecting integer
 * world-space X/Y grid lines into screen space. Each world grid line is sampled
 * at fixed intervals, transformed into camera space, perspective-projected,
 * grouped into depth-based alpha bands, and stroked in batches.
 *
 * The implementation intentionally avoids per-pixel floor casting. Instead, it
 * draws continuous projected line segments, which is much cheaper for the 2D
 * Canvas renderer while still producing a convincing perspective grid.
 *
 * The floor and ceiling share the same world-space geometry. The ceiling is
 * rendered as a vertical mirror of the floor across the horizon line.
 *
 * @module
 */

import {
  NEATENSTEIN_RENDER_DISTANCE_CAP,
  NEATENSTEIN_BACKGROUND_RGB,
  RGBA_CHANNELS,
} from './framebuffer';
import {
  appendNeatensteinGridLine,
  createNeatensteinFloorSegmentBands,
} from './floor.band.utils';
import {
  strokeNeatensteinGridBands,
  resolveNeatensteinFloorAlphaFromDistance,
  parseNeatensteinFloorHexColor,
} from './floor.shade.utils';
import {
  resolveContextCanvasDimension,
  sanitizeNeatensteinFloorCamera,
} from './floor.projection.utils';
import { isPositiveFiniteDimension } from '../shared/math-guards.utils';
import {
  NEATENSTEIN_FLOOR_DEFAULT_WIDTH,
  NEATENSTEIN_FLOOR_DEFAULT_HEIGHT,
  NEATENSTEIN_FLOOR_HORIZON_RATIO,
  NEATENSTEIN_FLOOR_FOV_RADIANS,
  NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
  NEATENSTEIN_FLOOR_GLOW_WIDTH_PX,
  NEATENSTEIN_FLOOR_LINE_WIDTH_PX,
  NEATENSTEIN_FLOOR_GLOW_ALPHA_MULTIPLIER,
} from './renderer.floor.constants';
import type {
  NeatensteinFloorCamera,
  NeatensteinFloorRenderContext,
  NeatensteinGridProjectionContext,
} from './renderer.floor.types';
import { FLAPPY_NEON_PALETTE } from '../../../flappy_bird/constants/constants.palette';

// Re-export previously-public symbols that moved to dedicated files.
export {
  NEATENSTEIN_FLOOR_DEFAULT_WIDTH,
  NEATENSTEIN_FLOOR_DEFAULT_HEIGHT,
  NEATENSTEIN_FLOOR_HORIZON_RATIO,
  NEATENSTEIN_FLOOR_FOV_RADIANS,
  NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
  NEATENSTEIN_FLOOR_CAMERA_HEIGHT_SCREEN_RATIO,
  NEATENSTEIN_FLOOR_MIN_ALPHA,
  NEATENSTEIN_FLOOR_MAX_ALPHA,
} from './renderer.floor.constants';

/**
 * Unconditional flag indicating the per-pixel floor caster is the default
 * rendering path across all tiers (C1.1).
 *
 * When `true`, the per-pixel caster ({@link castNeatensteinFloorPerPixel}) is
 * the unconditional default. The Canvas 2D line-projection path remains
 * available as a fallback but is no longer the primary path.
 */
export const NEATENSTEIN_PER_PIXEL_FLOOR_UNCONDITIONAL = true as const;

/**
 * World-space spacing between floor grid lines in world units (C1.1, Invariant §4).
 *
 * The procedural floor grid renders lines at integer world coordinates, so the
 * grid pitch is exactly 1 world unit = 1 map cell. This MUST match the DDA cell
 * size ({@link NEATENSTEIN_DDA_CELL_SIZE_WORLD}) so walls align to floor lines.
 */
export const NEATENSTEIN_FLOOR_GRID_SPACING_WORLD = 1.0 as const;

/**
 * Predicate that returns whether the per-pixel floor caster is active (C1.1).
 *
 * After C1.1 the per-pixel caster is unconditional, so this always returns
 * `true` regardless of the tier or quality argument.
 *
 * @param _tier - Rendering tier label (ignored — per-pixel is unconditional).
 * @returns Always `true`.
 */
// eslint-disable-next-line @typescript-eslint/no-unused-vars
export function isNeatensteinPerPixelFloorActive(_tier?: string): boolean {
  return NEATENSTEIN_PER_PIXEL_FLOOR_UNCONDITIONAL;
}
export type {
  NeatensteinFloorCamera,
  NeatensteinFloorRenderContext,
} from './renderer.floor.types';
export { resolveNeatensteinFloorAlpha } from './floor.shade.utils';
export {
  projectNeatensteinFloorPoint,
  projectNeatensteinCeilingPoint,
} from './floor.projection.utils';

/**
 * Number of world cells in each direction around the camera considered for
 * projected grid lines.
 *
 * This fixed range bounds per-frame work independently of map size.
 */
const NEATENSTEIN_FLOOR_VISIBLE_CELL_RANGE = NEATENSTEIN_RENDER_DISTANCE_CAP;

/**
 * Pooled depth-banded segment buffers — created once, cleared (`.length = 0`)
 * each frame. Safe for flat number arrays.
 */
const pooledBands = createNeatensteinFloorSegmentBands();

/**
 * Pooled projection context — mutated in place each frame instead of
 * allocating a fresh object.
 */
const pooledProjection: NeatensteinGridProjectionContext = {
  width: 0,
  height: 0,
  cameraX: 0,
  cameraY: 0,
  cosYaw: 1,
  sinYaw: 0,
  focalLength: 0,
  halfWidth: 0,
  horizonY: 0,
  cameraHeight: NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
};

/**
 * Draw the world-fixed neon floor grid below the horizon.
 *
 * The grid is a procedural texture made from projected world-space integer X/Y
 * lines. The renderer samples nearby world lines, projects visible samples,
 * batches the resulting screen-space segments by depth, and draws each band
 * with a glow pass plus a core pass.
 *
 * @param ctx - Canvas-like context with path, stroke, and state methods.
 * @param canvasWidth - Canvas width in backing-store pixels.
 * @param canvasHeight - Canvas height in backing-store pixels.
 * @param camera - Current camera state.
 *
 * @example
 * ```ts
 * drawNeatensteinFloor(ctx, 640, 480, { yaw: 0, x: 12.5, y: 12.5 });
 * ```
 */
export function drawNeatensteinFloor(
  ctx: NeatensteinFloorRenderContext,
  canvasWidth: number,
  canvasHeight: number,
  camera: NeatensteinFloorCamera,
): void {
  drawNeatensteinGrid(ctx, canvasWidth, canvasHeight, camera, false);
}

/**
 * Draw the world-fixed neon ceiling grid above the horizon.
 *
 * The ceiling is a vertical mirror of the floor projection across the horizon.
 * It uses the same world-space grid, color palette, batching, and depth-band
 * alpha falloff.
 *
 * @param ctx - Canvas-like context with path, stroke, and state methods.
 * @param canvasWidth - Canvas width in backing-store pixels.
 * @param canvasHeight - Canvas height in backing-store pixels.
 * @param camera - Current camera state.
 *
 * @example
 * ```ts
 * drawNeatensteinCeiling(ctx, 640, 480, { yaw: 0, x: 12.5, y: 12.5 });
 * ```
 */
export function drawNeatensteinCeiling(
  ctx: NeatensteinFloorRenderContext,
  canvasWidth: number,
  canvasHeight: number,
  camera: NeatensteinFloorCamera,
): void {
  drawNeatensteinGrid(ctx, canvasWidth, canvasHeight, camera, true);
}

/**
 * Backwards-compatible wrapper that reads canvas dimensions from `ctx.canvas`.
 *
 * If no backing canvas is present, test fallback dimensions are used so
 * lightweight mock contexts can still exercise the rendering path.
 *
 * @param ctx - Canvas-like context with an optional backing canvas.
 * @param camera - Current camera state.
 *
 * @example
 * ```ts
 * renderNeatensteinFloor(ctx, { yaw: 0, x: 12.5, y: 12.5 });
 * ```
 */
export function renderNeatensteinFloor(
  ctx: NeatensteinFloorRenderContext,
  camera: NeatensteinFloorCamera,
): void {
  const width = resolveContextCanvasDimension(
    ctx.canvas?.width,
    NEATENSTEIN_FLOOR_DEFAULT_WIDTH,
  );
  const height = resolveContextCanvasDimension(
    ctx.canvas?.height,
    NEATENSTEIN_FLOOR_DEFAULT_HEIGHT,
  );

  drawNeatensteinFloor(ctx, width, height, camera);
}

/**
 * Backwards-compatible wrapper that reads canvas dimensions from `ctx.canvas`.
 *
 * If no backing canvas is present, test fallback dimensions are used so
 * lightweight mock contexts can still exercise the rendering path.
 *
 * @param ctx - Canvas-like context with an optional backing canvas.
 * @param camera - Current camera state.
 *
 * @example
 * ```ts
 * renderNeatensteinCeiling(ctx, { yaw: 0, x: 12.5, y: 12.5 });
 * ```
 */
export function renderNeatensteinCeiling(
  ctx: NeatensteinFloorRenderContext,
  camera: NeatensteinFloorCamera,
): void {
  const width = resolveContextCanvasDimension(
    ctx.canvas?.width,
    NEATENSTEIN_FLOOR_DEFAULT_WIDTH,
  );
  const height = resolveContextCanvasDimension(
    ctx.canvas?.height,
    NEATENSTEIN_FLOOR_DEFAULT_HEIGHT,
  );

  drawNeatensteinCeiling(ctx, width, height, camera);
}

/**
 * Shared implementation for drawing either the floor or the ceiling grid.
 *
 * @param ctx - Canvas-like context.
 * @param canvasWidth - Canvas width in backing-store pixels.
 * @param canvasHeight - Canvas height in backing-store pixels.
 * @param camera - Raw camera state.
 * @param forCeiling - Whether to mirror projection above the horizon.
 */
function drawNeatensteinGrid(
  ctx: NeatensteinFloorRenderContext,
  canvasWidth: number,
  canvasHeight: number,
  camera: NeatensteinFloorCamera,
  forCeiling: boolean,
): void {
  // Explicit draw calls with invalid dimensions are ignored. Wrappers provide
  // fallbacks before reaching this function.
  if (
    !isPositiveFiniteDimension(canvasWidth) ||
    !isPositiveFiniteDimension(canvasHeight)
  ) {
    return;
  }

  const safeCamera = sanitizeNeatensteinFloorCamera(camera);
  const horizonY = canvasHeight * NEATENSTEIN_FLOOR_HORIZON_RATIO;
  const halfWidth = canvasWidth / 2;
  const focalLength =
    canvasHeight / 2 / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);

  // C1.1: Gate per-pixel vs band-based floor. When the per-pixel caster is
  // active (unconditional after C1.1), this Canvas 2D band-based grid is the
  // fallback path — the per-pixel caster handles the framebuffer tier via
  // seedFramebufferProcedurally. When disabled, skip the grid entirely.
  if (!isNeatensteinPerPixelFloorActive()) {
    return;
  }

  // If any projection constant somehow becomes invalid, skip the frame rather
  // than writing invalid coordinates into the canvas path.
  if (
    !Number.isFinite(horizonY) ||
    !Number.isFinite(focalLength) ||
    focalLength <= 0
  ) {
    return;
  }

  const projection = pooledProjection;
  projection.width = canvasWidth;
  projection.height = canvasHeight;
  projection.cameraX = safeCamera.x;
  projection.cameraY = safeCamera.y;
  projection.cosYaw = Math.cos(safeCamera.yaw);
  projection.sinYaw = Math.sin(safeCamera.yaw);
  projection.focalLength = focalLength;
  projection.halfWidth = halfWidth;
  projection.horizonY = horizonY;
  projection.cameraHeight = NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD;

  // Clear pooled band buffers in place (flat number arrays — safe to truncate).
  for (let bi = 0; bi < pooledBands.length; bi += 1) {
    pooledBands[bi].length = 0;
  }
  const bands = pooledBands;
  const range = NEATENSTEIN_FLOOR_VISIBLE_CELL_RANGE;
  const minX = Math.floor(safeCamera.x - range);
  const maxX = Math.ceil(safeCamera.x + range);
  const minY = Math.floor(safeCamera.y - range);
  const maxY = Math.ceil(safeCamera.y + range);

  // Project constant-X world grid lines.
  for (let x = minX; x <= maxX; x += 1) {
    appendNeatensteinGridLine(
      bands,
      projection,
      x,
      true,
      minY,
      maxY,
      forCeiling,
    );
  }

  // Project constant-Y world grid lines.
  for (let y = minY; y <= maxY; y += 1) {
    appendNeatensteinGridLine(
      bands,
      projection,
      y,
      false,
      minX,
      maxX,
      forCeiling,
    );
  }

  strokeNeatensteinGridBands(ctx, bands);
}

/**
 * Test-only export of {@link strokeNeatensteinGridBands} for direct coverage
 * of the empty-band `continue` branch.
 */
export const __testOnlyStrokeNeatensteinGridBands = strokeNeatensteinGridBands;

/**
 * Per-pixel floor and ceiling caster for the Neatenstein neon renderer.
 *
 * Unlike the line-projection path ({@link drawNeatensteinFloor}), this
 * function writes each pixel directly into the framebuffer by computing
 * world coordinates via ray-direction interpolation and detecting grid lines
 * using `fract(worldCoord)`. This is the CPU equivalent of the GPU
 * floor-caster shader.
 *
 * The framebuffer is first filled with the background color so that ceiling
 * pixels and far-distance areas are non-zero. Then, for each row below (floor)
 * and above (ceiling) the horizon, world coordinates are computed per column,
 * and grid-line pixels are blended with the neon grid color using the unified
 * smoothstep fog factor alpha (B3.2).
 *
 * Halo glow is replicated using `NEATENSTEIN_FLOOR_GLOW_WIDTH_PX` (6px halo)
 * plus a `NEATENSTEIN_FLOOR_LINE_WIDTH_PX` (2px core), matching the
 * double-stroke glow method of the line-projection path.
 *
 * @param framebuffer - Flat RGBA framebuffer.
 * @param width - Framebuffer width in pixels.
 * @param height - Framebuffer height in pixels.
 * @param projection - Camera projection context (without width/height, which
 *   are passed as separate parameters).
 */
export function castNeatensteinFloorPerPixel(
  framebuffer: Uint8ClampedArray,
  width: number,
  height: number,
  projection: Omit<NeatensteinGridProjectionContext, 'width' | 'height'>,
): void {
  if (!isPositiveFiniteDimension(width) || !isPositiveFiniteDimension(height)) {
    return;
  }

  const totalPixels = width * height;
  if (framebuffer.length < totalPixels * RGBA_CHANNELS) {
    return;
  }

  const {
    cameraX,
    cameraY,
    cosYaw,
    sinYaw,
    focalLength,
    halfWidth,
    horizonY,
    cameraHeight,
  } = projection;

  if (
    !Number.isFinite(focalLength) ||
    focalLength <= 0 ||
    !Number.isFinite(horizonY)
  ) {
    return;
  }

  // Derive plane scale from shared constants: planeScale * focalLength = halfWidth
  const planeScale = halfWidth / focalLength;

  // Parse the shared neon grid line color once.
  const gridColor = parseNeatensteinFloorHexColor(
    FLAPPY_NEON_PALETTE.groundGridLine,
  );
  const gridR = gridColor !== null ? gridColor.r : NEATENSTEIN_BACKGROUND_RGB.r;
  const gridG = gridColor !== null ? gridColor.g : NEATENSTEIN_BACKGROUND_RGB.g;
  const gridB = gridColor !== null ? gridColor.b : NEATENSTEIN_BACKGROUND_RGB.b;

  const bgR = NEATENSTEIN_BACKGROUND_RGB.r;
  const bgG = NEATENSTEIN_BACKGROUND_RGB.g;
  const bgB = NEATENSTEIN_BACKGROUND_RGB.b;

  // Fill the entire framebuffer with the background color so that ceiling
  // and far-distance pixels are non-zero.
  for (let i = 0; i < totalPixels; i += 1) {
    const o = i * RGBA_CHANNELS;
    framebuffer[o] = bgR;
    framebuffer[o + 1] = bgG;
    framebuffer[o + 2] = bgB;
    framebuffer[o + 3] = 255;
  }

  // Render floor (below horizon) and ceiling (above horizon) per-pixel.
  for (let y = 0; y < height; y += 1) {
    const isFloor = y > horizonY;
    const isCeiling = y < horizonY;
    if (!isFloor && !isCeiling) {
      continue;
    }

    // Vertical distance from the horizon in pixels (always > 0 here).
    const verticalPx = isFloor ? y - horizonY : horizonY - y;
    if (verticalPx <= 0) {
      continue;
    }

    // rowDistance from shared constants:
    // rowDistance = cameraHeight * focalLength / verticalPx
    const rowDistance = (cameraHeight * focalLength) / verticalPx;
    if (
      !Number.isFinite(rowDistance) ||
      rowDistance > NEATENSTEIN_RENDER_DISTANCE_CAP
    ) {
      continue;
    }

    // Alpha from the unified smoothstep fog factor (B3.2).
    const alpha = resolveNeatensteinFloorAlphaFromDistance(rowDistance);
    if (alpha <= 0) {
      continue;
    }

    // World-space size of one screen pixel at this depth.
    const pixelWorldSize = rowDistance / focalLength;

    // Screen-space distance thresholds (perspective-corrected: shrink
    // with distance so lines converge naturally toward the horizon).
    const coreThresh = NEATENSTEIN_FLOOR_LINE_WIDTH_PX / rowDistance;
    const glowThresh = NEATENSTEIN_FLOOR_GLOW_WIDTH_PX / rowDistance;
    const coreThresh2 = coreThresh * coreThresh;
    const glowThresh2 = glowThresh * glowThresh;

    // Per-row anisotropic projection gradient terms.  Grid lines on
    // different axes project through different gradients (row-direction
    // vs column-direction), so we compute screen-space distance per axis
    // to ensure horizontal and vertical lines have identical thickness
    // at their intersections.
    const cosYawAbs = Math.abs(cosYaw);
    const sinYawAbs = Math.abs(sinYaw);
    const rowOverVert = rowDistance / verticalPx;
    const rowOverVertSq = rowOverVert * rowOverVert;
    const sinTerm2 = pixelWorldSize * sinYawAbs * (pixelWorldSize * sinYawAbs);
    const cosTerm2 = pixelWorldSize * cosYawAbs * (pixelWorldSize * cosYawAbs);

    const haloAlpha = alpha * NEATENSTEIN_FLOOR_GLOW_ALPHA_MULTIPLIER;

    for (let x = 0; x < width; x += 1) {
      // Screen-space offset in [-1, 1].
      const screenOffset = halfWidth > 0 ? (x - halfWidth) / halfWidth : 0;

      // Ray direction: dir + plane * offset
      const rayDirX = cosYaw + -sinYaw * planeScale * screenOffset;
      const rayDirY = sinYaw + cosYaw * planeScale * screenOffset;

      // World coordinates via ray-direction interpolation.
      const worldX = cameraX + rowDistance * rayDirX;
      const worldY = cameraY + rowDistance * rayDirY;

      // Procedural integer grid detection via fract(worldCoord).
      const fx = worldX - Math.floor(worldX);
      const fy = worldY - Math.floor(worldY);

      // World-space distance to nearest grid line on each axis.
      const distX = Math.min(fx, 1 - fx);
      const distY = Math.min(fy, 1 - fy);

      // Convert world-space distance to screen-space distance per axis.
      // Each axis has a row-direction gradient (varies per column via
      // rayDir) and a column-direction gradient (per-row constant).  The
      // combined gradient magnitude gives the true screen-space distance.
      const gradX2 = rowOverVertSq * rayDirX * rayDirX + sinTerm2;
      const gradY2 = rowOverVertSq * rayDirY * rayDirY + cosTerm2;
      const screenDistX2 = (distX * distX) / (gradX2 + 1e-12);
      const screenDistY2 = (distY * distY) / (gradY2 + 1e-12);
      const minScreen2 = Math.min(screenDistX2, screenDistY2);

      if (minScreen2 > glowThresh2) {
        continue;
      }

      const o = (y * width + x) * RGBA_CHANNELS;

      // Smooth blend in screen-space: full alpha in core, smoothstep
      // fade in halo.  Both axes share the same screen-space threshold
      // so horizontal and vertical lines have identical thickness.
      const minScreen = Math.sqrt(minScreen2);
      let blendAlpha: number;
      if (minScreen <= coreThresh) {
        blendAlpha = alpha;
      } else {
        const haloT = (minScreen - coreThresh) / (glowThresh - coreThresh);
        const smooth = haloT * haloT * (3 - 2 * haloT);
        blendAlpha = haloAlpha * (1 - smooth);
      }
      const invAlpha = 1 - blendAlpha;

      framebuffer[o] = Math.round(gridR * blendAlpha + bgR * invAlpha);
      framebuffer[o + 1] = Math.round(gridG * blendAlpha + bgG * invAlpha);
      framebuffer[o + 2] = Math.round(gridB * blendAlpha + bgB * invAlpha);
      framebuffer[o + 3] = 255;
    }
  }
}
