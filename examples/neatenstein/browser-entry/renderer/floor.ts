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

import { NEATENSTEIN_RENDER_DISTANCE_CAP } from './framebuffer';
import {
  appendNeatensteinGridLine,
  createNeatensteinFloorSegmentBands,
} from './floor.band.utils';
import { strokeNeatensteinGridBands } from './floor.shade.utils';
import {
  isPositiveFiniteDimension,
  resolveContextCanvasDimension,
  sanitizeNeatensteinFloorCamera,
} from './floor.projection.utils';
import {
  NEATENSTEIN_FLOOR_DEFAULT_WIDTH,
  NEATENSTEIN_FLOOR_DEFAULT_HEIGHT,
  NEATENSTEIN_FLOOR_HORIZON_RATIO,
  NEATENSTEIN_FLOOR_FOV_RADIANS,
  NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
} from './renderer.floor.constants';
import type {
  NeatensteinFloorCamera,
  NeatensteinFloorRenderContext,
  NeatensteinGridProjectionContext,
} from './renderer.floor.types';

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

  // If any projection constant somehow becomes invalid, skip the frame rather
  // than writing invalid coordinates into the canvas path.
  if (
    !Number.isFinite(horizonY) ||
    !Number.isFinite(focalLength) ||
    focalLength <= 0
  ) {
    return;
  }

  const projection: NeatensteinGridProjectionContext = {
    width: canvasWidth,
    height: canvasHeight,
    cameraX: safeCamera.x,
    cameraY: safeCamera.y,
    cosYaw: Math.cos(safeCamera.yaw),
    sinYaw: Math.sin(safeCamera.yaw),
    focalLength,
    halfWidth,
    horizonY,
    cameraHeight: NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
  };

  const bands = createNeatensteinFloorSegmentBands();
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
