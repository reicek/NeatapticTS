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

import { FLAPPY_NEON_PALETTE } from '../../../flappy_bird/constants/constants.palette';

/**
 * Test fallback canvas width used when the render context has no backing
 * `canvas`, such as lightweight mock contexts in unit tests.
 *
 * Production callers should normally render through a real canvas context or
 * pass explicit dimensions to {@link drawNeatensteinFloor} /
 * {@link drawNeatensteinCeiling}.
 */
export const NEATENSTEIN_FLOOR_DEFAULT_WIDTH = 320;

/**
 * Test fallback canvas height used when the render context has no backing
 * `canvas`, such as lightweight mock contexts in unit tests.
 *
 * Production callers should normally render through a real canvas context or
 * pass explicit dimensions to {@link drawNeatensteinFloor} /
 * {@link drawNeatensteinCeiling}.
 */
export const NEATENSTEIN_FLOOR_DEFAULT_HEIGHT = 240;

/**
 * Fraction of canvas height where the horizon line sits.
 *
 * The current renderer assumes a level camera, so the horizon is horizontal.
 * Everything below the horizon is floor; everything above it is ceiling.
 */
export const NEATENSTEIN_FLOOR_HORIZON_RATIO = 0.5;

/**
 * Horizontal field of view in radians.
 *
 * This should match the wall raycaster's horizontal FOV so floor/ceiling
 * perspective aligns with projected wall columns.
 */
export const NEATENSTEIN_FLOOR_FOV_RADIANS = Math.PI / 3;

/**
 * Camera height above the floor in world units.
 *
 * This value controls how strongly floor and ceiling grid points project away
 * from the horizon.
 */
export const NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD = 0.5;

/**
 * Ratio of canvas height used as the camera height for separate screen-space
 * forced-perspective helpers, such as pulses or tracers.
 *
 * This module's world-space projection uses
 * {@link NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD}; this exported ratio is kept
 * for companion effects that work directly in screen space.
 */
export const NEATENSTEIN_FLOOR_CAMERA_HEIGHT_SCREEN_RATIO = 0.5;

/** Minimum line opacity near the horizon. */
export const NEATENSTEIN_FLOOR_MIN_ALPHA = 0.12;

/** Maximum line opacity near the camera. */
export const NEATENSTEIN_FLOOR_MAX_ALPHA = 0.58;

/**
 * Glow strategy used for the floor and ceiling grid lines.
 *
 * - `'double-stroke'` draws a wide low-alpha halo followed by a narrow core.
 * - `'shadow-blur'` uses Canvas shadow blur for a softer but less predictable
 *   compositor-dependent glow cost.
 */
const NEATENSTEIN_FLOOR_GLOW_METHOD: 'double-stroke' | 'shadow-blur' =
  'double-stroke';

/** Width in pixels of the bright core grid line. */
const NEATENSTEIN_FLOOR_LINE_WIDTH_PX = 1;

/** Width in pixels of the halo stroke used by the double-stroke glow mode. */
const NEATENSTEIN_FLOOR_GLOW_WIDTH_PX = 3;

/** Alpha multiplier applied to the halo pass in double-stroke glow mode. */
const NEATENSTEIN_FLOOR_GLOW_ALPHA_MULTIPLIER = 0.35;

/** Shadow blur radius used by the shadow-blur glow mode. */
const NEATENSTEIN_FLOOR_SHADOW_BLUR_PX = 4;

/**
 * Number of alpha bands used to batch floor/ceiling strokes.
 *
 * More bands produce smoother depth fading but require more canvas stroke
 * calls. This value keeps the effect visually graded while preserving batching.
 */
const NEATENSTEIN_FLOOR_ALPHA_BANDS = 4;

/**
 * Number of world cells in each direction around the camera considered for
 * projected grid lines.
 *
 * This fixed range bounds per-frame work independently of map size.
 */
const NEATENSTEIN_FLOOR_VISIBLE_CELL_RANGE = 20;

/**
 * Number of samples per projected world grid line.
 *
 * Each integer grid line is sampled at `SAMPLES + 1` points and consecutive
 * visible samples are connected into screen-space line segments.
 */
const NEATENSTEIN_FLOOR_LINE_SAMPLES = 80;

/**
 * Minimum positive camera-space depth required for projection.
 *
 * Points at or behind the camera plane are culled. The epsilon also avoids
 * extreme projected coordinates for samples that are nearly on the camera
 * plane.
 */
const NEATENSTEIN_FLOOR_NEAR_PLANE_EPSILON = 0.001;

/**
 * Number of fractional digits retained for cached alpha stroke styles.
 *
 * The renderer uses banded alpha values, so this quantization preserves visual
 * stability while preventing unbounded cache growth from tiny float differences.
 */
const NEATENSTEIN_FLOOR_ALPHA_CACHE_PRECISION = 4;

/** Parsed RGB components of the shared neon floor line color. */
const FLOOR_BASE_RGB = parseNeatensteinFloorHexColor(
  FLAPPY_NEON_PALETTE.groundGridLine,
);

/** Shared neon glow color from the palette. */
const NEATENSTEIN_FLOOR_SHADOW_COLOR = FLAPPY_NEON_PALETTE.groundGridGlow;

/** Fallback stroke style used when the palette color cannot be parsed as hex. */
const NEATENSTEIN_FLOOR_FALLBACK_STROKE_STYLE =
  FLAPPY_NEON_PALETTE.groundGridLine;

/**
 * Cache of computed RGBA stroke styles keyed by quantized alpha.
 *
 * The RGB color is fixed at module load, so alpha is the only variable part of
 * the stroke style.
 */
const NEATENSTEIN_FLOOR_STROKE_STYLE_CACHE = new Map<string, string>();

/**
 * Flat screen-space segment buffer.
 *
 * Values are stored in groups of four:
 *
 * ```ts
 * [x1, y1, x2, y2, x1, y1, x2, y2, ...]
 * ```
 *
 * This avoids allocating one tuple/object per projected line segment.
 */
type NeatensteinFloorSegmentBuffer = number[];

/** Camera state consumed by the floor and ceiling renderer. */
export interface NeatensteinFloorCamera {
  /** Camera yaw in radians. `0` looks down the world +X axis. */
  yaw: number;
  /** Camera world X position. */
  x: number;
  /** Camera world Y position. */
  y: number;
}

/**
 * Minimal canvas-like rendering context consumed by the grid renderer.
 *
 * The interface is intentionally narrow so tests can provide small mocks
 * instead of implementing the full `CanvasRenderingContext2D` API.
 */
export interface NeatensteinFloorRenderContext {
  /** Start a new path. */
  beginPath(): void;
  /** Move the path cursor to `(x, y)`. */
  moveTo(x: number, y: number): void;
  /** Add a line segment to `(x, y)`. */
  lineTo(x: number, y: number): void;
  /** Stroke the current path. */
  stroke(): void;
  /** Snapshot current context state. */
  save(): void;
  /** Restore the most recently saved context state. */
  restore(): void;
  /** Optional backing canvas dimensions. */
  canvas?: { width: number; height: number };
  /** Optional stroke color. */
  strokeStyle?: string | CanvasGradient | CanvasPattern;
  /** Optional line width in pixels. */
  lineWidth?: number;
  /** Optional glow blur radius in pixels. */
  shadowBlur?: number;
  /** Optional glow color. */
  shadowColor?: string | CanvasGradient | CanvasPattern;
}

/**
 * Sanitized camera values used by the projection hot path.
 */
interface SafeNeatensteinFloorCamera {
  /** Finite camera yaw in radians. */
  yaw: number;
  /** Finite camera world X coordinate. */
  x: number;
  /** Finite camera world Y coordinate. */
  y: number;
}

/**
 * Shared projection constants for a single grid draw call.
 */
interface NeatensteinGridProjectionContext {
  /** Canvas width in backing-store pixels. */
  width: number;
  /** Canvas height in backing-store pixels. */
  height: number;
  /** Camera world X coordinate. */
  cameraX: number;
  /** Camera world Y coordinate. */
  cameraY: number;
  /** Cosine of camera yaw. */
  cosYaw: number;
  /** Sine of camera yaw. */
  sinYaw: number;
  /** Perspective focal length in pixels. */
  focalLength: number;
  /** Half canvas width in pixels. */
  halfWidth: number;
  /** Horizon Y coordinate in pixels. */
  horizonY: number;
  /** Camera height above the floor in world units. */
  cameraHeight: number;
}

/**
 * Projected screen-space point used while building line segments.
 */
interface ProjectedNeatensteinGridPoint {
  /** Screen-space X coordinate. */
  x: number;
  /** Screen-space Y coordinate. */
  y: number;
  /** Normalized depth ratio, where `0` is far and `1` is near. */
  depthRatio: number;
  /** Positive camera-space forward distance. */
  distance: number;
}

/**
 * Clamp a number into the inclusive range `[min, max]`.
 *
 * @param value - Value to clamp.
 * @param min - Lower bound.
 * @param max - Upper bound.
 * @returns Clamped value.
 */
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Return whether a render dimension is finite and drawable.
 *
 * @param value - Candidate width or height.
 * @returns Whether the value is a positive finite number.
 */
function isPositiveFiniteDimension(value: number): boolean {
  return Number.isFinite(value) && value > 0;
}

/**
 * Sanitize camera values for renderer use.
 *
 * Non-finite camera values are treated as `0` so a malformed frame cannot
 * poison the canvas path with `NaN` coordinates.
 *
 * @param camera - Raw camera state.
 * @returns Finite camera state.
 */
function sanitizeNeatensteinFloorCamera(
  camera: NeatensteinFloorCamera,
): SafeNeatensteinFloorCamera {
  return {
    x: Number.isFinite(camera.x) ? camera.x : 0,
    y: Number.isFinite(camera.y) ? camera.y : 0,
    yaw: Number.isFinite(camera.yaw) ? camera.yaw : 0,
  };
}

/**
 * Resolve a canvas dimension from a context, using a test fallback if missing.
 *
 * @param value - Optional canvas dimension.
 * @param fallback - Fallback dimension for mock contexts.
 * @returns Positive finite dimension.
 */
function resolveContextCanvasDimension(
  value: number | undefined,
  fallback: number,
): number {
  return isPositiveFiniteDimension(value ?? Number.NaN) ? value! : fallback;
}

/**
 * Map a normalized depth ratio to an alpha-band index.
 *
 * @param depthRatio - Normalized depth ratio where `0` is far and `1` is near.
 * @returns Band index in the range `0..NEATENSTEIN_FLOOR_ALPHA_BANDS - 1`.
 */
function bandIndexForDepthRatio(depthRatio: number): number {
  const clampedRatio = clamp(depthRatio, 0, 1);

  return Math.min(
    NEATENSTEIN_FLOOR_ALPHA_BANDS - 1,
    Math.floor(clampedRatio * NEATENSTEIN_FLOOR_ALPHA_BANDS),
  );
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
  const focalLength = halfWidth / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);

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
 * Create one flat segment buffer per alpha band.
 *
 * @returns Empty depth-banded segment buffers.
 */
function createNeatensteinFloorSegmentBands(): NeatensteinFloorSegmentBuffer[] {
  return Array.from(
    { length: NEATENSTEIN_FLOOR_ALPHA_BANDS },
    () => [] as NeatensteinFloorSegmentBuffer,
  );
}

/**
 * Append one sampled world-space grid line to the depth-banded segment buffers.
 *
 * The line is sampled at evenly spaced points. Consecutive visible projected
 * samples become one screen-space segment. If a sample is behind the camera,
 * the visible run is broken so the next valid sample starts a new segment.
 *
 * @param bands - Depth-banded flat segment buffers.
 * @param projection - Shared projection constants for this frame.
 * @param fixedCoord - Integer world coordinate for the fixed axis.
 * @param isXLine - `true` for constant-X lines, `false` for constant-Y lines.
 * @param startCoord - Start value for the varying axis.
 * @param endCoord - End value for the varying axis.
 * @param forCeiling - Whether to mirror the projection above the horizon.
 */
function appendNeatensteinGridLine(
  bands: NeatensteinFloorSegmentBuffer[],
  projection: NeatensteinGridProjectionContext,
  fixedCoord: number,
  isXLine: boolean,
  startCoord: number,
  endCoord: number,
  forCeiling: boolean,
): void {
  let previous: ProjectedNeatensteinGridPoint | null = null;

  for (let i = 0; i <= NEATENSTEIN_FLOOR_LINE_SAMPLES; i += 1) {
    const t = i / NEATENSTEIN_FLOOR_LINE_SAMPLES;
    const coord = startCoord + (endCoord - startCoord) * t;

    // Constant-X lines vary Y; constant-Y lines vary X.
    const worldX = isXLine ? fixedCoord : coord;
    const worldY = isXLine ? coord : fixedCoord;

    const projected = projectNeatensteinGridPoint(
      worldX,
      worldY,
      projection,
      forCeiling,
    );

    if (projected === null) {
      previous = null;
      continue;
    }

    if (previous !== null) {
      const averageDepthRatio =
        (previous.depthRatio + projected.depthRatio) / 2;
      const band = bandIndexForDepthRatio(averageDepthRatio);

      // Store as a flat tuple to avoid allocating one array/object per segment.
      bands[band].push(previous.x, previous.y, projected.x, projected.y);
    }

    previous = projected;
  }
}

/**
 * Stroke every non-empty depth band using the configured glow method.
 *
 * @param ctx - Canvas-like rendering context.
 * @param bands - Depth-banded flat segment buffers.
 */
function strokeNeatensteinGridBands(
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

    if (NEATENSTEIN_FLOOR_GLOW_METHOD === 'double-stroke') {
      // Pass 1: wide, dim halo. This creates a predictable neon glow without
      // relying on compositor-specific shadow blur performance.
      ctx.lineWidth = NEATENSTEIN_FLOOR_GLOW_WIDTH_PX;
      ctx.shadowBlur = 0;
      ctx.strokeStyle = resolveNeatensteinFloorStrokeStyle(
        FLOOR_BASE_RGB,
        coreAlpha * NEATENSTEIN_FLOOR_GLOW_ALPHA_MULTIPLIER,
      );
      strokeNeatensteinFloorBand(ctx, segments);

      // Pass 2: narrow, bright core line.
      ctx.lineWidth = NEATENSTEIN_FLOOR_LINE_WIDTH_PX;
      ctx.shadowBlur = 0;
      ctx.strokeStyle = resolveNeatensteinFloorStrokeStyle(
        FLOOR_BASE_RGB,
        coreAlpha,
      );
      strokeNeatensteinFloorBand(ctx, segments);
    } else {
      // Alternative softer glow path. Kept behind a constant so experiments can
      // switch glow style without changing projection or batching logic.
      ctx.lineWidth = NEATENSTEIN_FLOOR_LINE_WIDTH_PX;
      ctx.shadowBlur = NEATENSTEIN_FLOOR_SHADOW_BLUR_PX;
      ctx.strokeStyle = resolveNeatensteinFloorStrokeStyle(
        FLOOR_BASE_RGB,
        coreAlpha,
      );
      strokeNeatensteinFloorBand(ctx, segments);
    }
  }

  ctx.restore();
}

/**
 * Stroke one flat segment buffer.
 *
 * Values are read in groups of four: `x1, y1, x2, y2`.
 *
 * @param ctx - Canvas-like rendering context.
 * @param segments - Flat screen-space segment buffer.
 */
function strokeNeatensteinFloorBand(
  ctx: NeatensteinFloorRenderContext,
  segments: NeatensteinFloorSegmentBuffer,
): void {
  ctx.beginPath();

  for (let i = 0; i < segments.length; i += 4) {
    ctx.moveTo(segments[i], segments[i + 1]);
    ctx.lineTo(segments[i + 2], segments[i + 3]);
  }

  ctx.stroke();
}

/**
 * Project a world-space floor point to screen space.
 *
 * The point is translated into camera-relative coordinates, rotated into
 * camera space, and projected using the shared horizontal FOV. Points behind
 * or too close to the camera plane are rejected.
 *
 * @param worldX - World X coordinate.
 * @param worldY - World Y coordinate.
 * @param cameraX - Camera world X.
 * @param cameraY - Camera world Y.
 * @param cosYaw - Cosine of camera yaw.
 * @param sinYaw - Sine of camera yaw.
 * @param focalLength - Perspective focal length in pixels.
 * @param halfWidth - Half canvas width in pixels.
 * @param horizonY - Horizon Y coordinate in pixels.
 * @param height - Canvas height in pixels.
 * @param cameraHeight - Camera height above floor in world units.
 * @returns Projected screen point, or `null` if the point is not drawable.
 */
export function projectNeatensteinFloorPoint(
  worldX: number,
  worldY: number,
  cameraX: number,
  cameraY: number,
  cosYaw: number,
  sinYaw: number,
  focalLength: number,
  halfWidth: number,
  horizonY: number,
  height: number,
  cameraHeight: number,
): { x: number; y: number; depthRatio: number; distance: number } | null {
  return projectNeatensteinGridPoint(
    worldX,
    worldY,
    {
      width: halfWidth * 2,
      height,
      cameraX,
      cameraY,
      cosYaw,
      sinYaw,
      focalLength,
      halfWidth,
      horizonY,
      cameraHeight,
    },
    false,
  );
}

/**
 * Project a world-space ceiling point to screen space.
 *
 * The ceiling uses the same camera-space transform as the floor, but its
 * vertical projection is mirrored above the horizon.
 *
 * @param worldX - World X coordinate.
 * @param worldY - World Y coordinate.
 * @param cameraX - Camera world X.
 * @param cameraY - Camera world Y.
 * @param cosYaw - Cosine of camera yaw.
 * @param sinYaw - Sine of camera yaw.
 * @param focalLength - Perspective focal length in pixels.
 * @param halfWidth - Half canvas width in pixels.
 * @param horizonY - Horizon Y coordinate in pixels.
 * @param height - Canvas height in pixels.
 * @param cameraHeight - Camera height above floor in world units.
 * @returns Projected screen point, or `null` if the point is not drawable.
 */
export function projectNeatensteinCeilingPoint(
  worldX: number,
  worldY: number,
  cameraX: number,
  cameraY: number,
  cosYaw: number,
  sinYaw: number,
  focalLength: number,
  halfWidth: number,
  horizonY: number,
  height: number,
  cameraHeight: number,
): { x: number; y: number; depthRatio: number; distance: number } | null {
  return projectNeatensteinGridPoint(
    worldX,
    worldY,
    {
      width: halfWidth * 2,
      height,
      cameraX,
      cameraY,
      cosYaw,
      sinYaw,
      focalLength,
      halfWidth,
      horizonY,
      cameraHeight,
    },
    true,
  );
}

/**
 * Shared floor/ceiling world-to-screen projection.
 *
 * @param worldX - World X coordinate.
 * @param worldY - World Y coordinate.
 * @param projection - Shared projection constants.
 * @param forCeiling - Whether to mirror vertically above the horizon.
 * @returns Projected point, or `null` if culled.
 */
function projectNeatensteinGridPoint(
  worldX: number,
  worldY: number,
  projection: NeatensteinGridProjectionContext,
  forCeiling: boolean,
): ProjectedNeatensteinGridPoint | null {
  const dx = worldX - projection.cameraX;
  const dy = worldY - projection.cameraY;

  // Camera-space Y is forward depth. Camera yaw of 0 looks down world +X.
  const camSpaceY = dx * projection.cosYaw + dy * projection.sinYaw;

  if (camSpaceY <= NEATENSTEIN_FLOOR_NEAR_PLANE_EPSILON) {
    return null;
  }

  // Camera-space X is horizontal right/left displacement.
  const camSpaceX = -dx * projection.sinYaw + dy * projection.cosYaw;
  const verticalOffset =
    (projection.cameraHeight / camSpaceY) * projection.focalLength;

  const screenX =
    projection.halfWidth + (camSpaceX / camSpaceY) * projection.focalLength;

  const screenY = forCeiling
    ? projection.horizonY - verticalOffset
    : projection.horizonY + verticalOffset;

  if (!Number.isFinite(screenX) || !Number.isFinite(screenY)) {
    return null;
  }

  const depthDenominator = forCeiling
    ? projection.horizonY
    : projection.height - projection.horizonY;

  if (!Number.isFinite(depthDenominator) || depthDenominator <= 0) {
    return null;
  }

  const rawDepthRatio = forCeiling
    ? (projection.horizonY - screenY) / depthDenominator
    : (screenY - projection.horizonY) / depthDenominator;

  return {
    x: screenX,
    y: screenY,
    depthRatio: clamp(rawDepthRatio, 0, 1),
    distance: camSpaceY,
  };
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
function resolveNeatensteinFloorStrokeStyle(
  baseRgb: { r: number; g: number; b: number } | null,
  alpha: number,
): string {
  if (baseRgb === null) {
    return NEATENSTEIN_FLOOR_FALLBACK_STROKE_STYLE;
  }

  const clampedAlpha = clamp(Number.isFinite(alpha) ? alpha : 1, 0, 1);
  const alphaKey = clampedAlpha.toFixed(
    NEATENSTEIN_FLOOR_ALPHA_CACHE_PRECISION,
  );

  const cached = NEATENSTEIN_FLOOR_STROKE_STYLE_CACHE.get(alphaKey);
  if (cached !== undefined) {
    return cached;
  }

  const style = `rgba(${baseRgb.r}, ${baseRgb.g}, ${baseRgb.b}, ${alphaKey})`;
  NEATENSTEIN_FLOOR_STROKE_STYLE_CACHE.set(alphaKey, style);

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
function parseNeatensteinFloorHexColor(
  baseColor: string,
): { r: number; g: number; b: number } | null {
  const match = /^#?([0-9a-fA-F]{6})$/.exec(baseColor);

  if (match === null) {
    return null;
  }

  const hexDigits = match[1];
  const red = Number.parseInt(hexDigits.slice(0, 2), 16);
  const green = Number.parseInt(hexDigits.slice(2, 4), 16);
  const blue = Number.parseInt(hexDigits.slice(4, 6), 16);

  if (
    !Number.isFinite(red) ||
    !Number.isFinite(green) ||
    !Number.isFinite(blue)
  ) {
    return null;
  }

  return { r: red, g: green, b: blue };
}
