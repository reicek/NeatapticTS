/**
 * World-fixed floor grid for the Neatenstein neon raycasting demo.
 *
 * This module renders the floor by projecting world-space grid lines into
 * screen space. For each integer X and Y grid line near the camera, a set of
 * sample points along the line is transformed from world coordinates into
 * camera-space coordinates using the camera yaw, then perspective-projected
 * onto the canvas. The resulting screen points are connected into one
 * batched path and stroked once per depth band. A subtle neon glow is added
 * with a second low-alpha, wide-line halo pass followed by the normal core
 * line. This produces continuous perspective lines and avoids the heavy
 * per-row, per-pixel work of the previous row-casting approach.
 *
 * @module
 */

import { FLAPPY_NEON_PALETTE } from '../../../flappy_bird/constants/constants.palette';

/**
 * Test fallback canvas width used when the render context has no backing
 * `canvas` (e.g., lightweight mock contexts in unit tests).
 *
 * This is **not** a production default; real rendering should pass explicit
 * dimensions to {@link drawNeatensteinFloor} or read `ctx.canvas` via the
 * {@link renderNeatensteinFloor} wrapper.
 */
export const NEATENSTEIN_FLOOR_DEFAULT_WIDTH = 320;

/**
 * Test fallback canvas height used when the render context has no backing
 * `canvas` (e.g., lightweight mock contexts in unit tests).
 *
 * This is **not** a production default; real rendering should pass explicit
 * dimensions to {@link drawNeatensteinFloor} or read `ctx.canvas` via the
 * {@link renderNeatensteinFloor} wrapper.
 */
export const NEATENSTEIN_FLOOR_DEFAULT_HEIGHT = 240;

/**
 * Fraction of canvas height where the horizon line sits.
 *
 * The camera always looks horizontally, so the horizon is a horizontal line
 * at this fraction of the canvas height. Everything below it is floor.
 */
export const NEATENSTEIN_FLOOR_HORIZON_RATIO = 0.5;

/**
 * Horizontal field of view in radians.
 *
 * Shared with the wall raycaster so the floor projection matches the wall
 * column projection.
 */
export const NEATENSTEIN_FLOOR_FOV_RADIANS = Math.PI / 3;

/**
 * Camera height above the floor in world units.
 *
 * The floor caster uses this directly as the world-space eye height when
 * projecting world points to screen space. This is intentionally separate
 * from the screen-space ratio used by pulse/tracer helpers.
 */
export const NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD = 0.5;

/**
 * Ratio of canvas height used as the camera height for screen-space
 * forced-perspective helpers (floor pulses, tracers).
 *
 * This is intentionally separate from
 * {@link NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD} because those helpers work in
 * pixel-space, not world-space.
 */
export const NEATENSTEIN_FLOOR_CAMERA_HEIGHT_SCREEN_RATIO = 0.5;

/**
 * Minimum line opacity, applied near the horizon (far depth rows).
 */
export const NEATENSTEIN_FLOOR_MIN_ALPHA = 0.12;

/**
 * Maximum line opacity, applied near the camera (near depth rows).
 */
export const NEATENSTEIN_FLOOR_MAX_ALPHA = 0.58;

/**
 * Glow method used for the floor grid lines.
 *
 * - `'double-stroke'`: draws a wide, low-alpha halo followed by the core line.
 *   This avoids the variable GPU cost of `shadowBlur` and keeps the glow cost
 *   deterministic (two simple strokes per band).
 * - `'shadow-blur'`: uses a single batched stroke with a small `shadowBlur`.
 *   This can look softer but its GPU cost depends on the browser compositor.
 */
const NEATENSTEIN_FLOOR_GLOW_METHOD: 'double-stroke' | 'shadow-blur' =
  'double-stroke';

/**
 * Width in pixels of the projected grid lines.
 */
const NEATENSTEIN_FLOOR_LINE_WIDTH_PX = 1;

/**
 * Width in pixels of the glow halo pass for the double-stroke method.
 */
const NEATENSTEIN_FLOOR_GLOW_WIDTH_PX = 3;

/**
 * Alpha multiplier applied to the glow halo pass relative to the band alpha.
 */
const NEATENSTEIN_FLOOR_GLOW_ALPHA_MULTIPLIER = 0.35;

/**
 * Shadow blur radius in pixels for the shadow-blur glow method.
 */
const NEATENSTEIN_FLOOR_SHADOW_BLUR_PX = 4;

/**
 * Number of depth bands used for batched alpha-aware floor strokes.
 *
 * Segments are grouped into this many bands so that farther lines fade while
 * near lines remain bright. Each band is stroked as its own batched path,
 * preserving batching within a band.
 */
const NEATENSTEIN_FLOOR_ALPHA_BANDS = 4;

/**
 * Number of world cells in each direction around the camera that are
 * considered for the visible floor grid.
 *
 * A fixed range keeps the per-frame workload bounded. The camera is very
 * close to the floor (0.5 world units), so the visible ground is only a few
 * world units away; a range of 20 comfortably covers the field of view.
 */
const NEATENSTEIN_FLOOR_VISIBLE_CELL_RANGE = 20;

/**
 * Number of line segments each grid line is broken into before projection.
 *
 * Each grid line is sampled at `SAMPLES + 1` evenly spaced world points and
 * the visible samples are connected with canvas line segments. A count of 80
 * over a 40 world-unit span gives roughly half-unit sampling, which is dense
 * enough to produce smooth perspective curves while staying far cheaper than
 * per-pixel sampling.
 */
const NEATENSTEIN_FLOOR_LINE_SAMPLES = 80;

/**
 * Minimum camera-space depth a sample point must have to be drawn.
 *
 * Points at or behind the camera plane (`camSpaceY <= 0`) are culled; a small
 * epsilon avoids extreme screen-space coordinates for points that graze the
 * camera plane.
 */
const NEATENSTEIN_FLOOR_NEAR_PLANE_EPSILON = 0.001;

/**
 * Parsed RGB components of the shared floor grid line color.
 *
 * Parsing once at module load avoids repeating `replace`/`parseInt` work in
 * the per-frame hot path.
 */
const FLOOR_BASE_RGB = parseNeatensteinFloorHexColor(
  FLAPPY_NEON_PALETTE.groundGridLine,
);

/**
 * Precomputed shadow color used for the single floor grid stroke.
 *
 * The value is a constant palette string, captured once at module load to avoid
 * repeated property lookups in the per-frame hot path. The shadow blur itself
 * is disabled for performance because the batched path is stroked once.
 */
const NEATENSTEIN_FLOOR_SHADOW_COLOR = FLAPPY_NEON_PALETTE.groundGridGlow;

/**
 * Precomputed fallback stroke style used when the base color cannot be parsed.
 */
const NEATENSTEIN_FLOOR_FALLBACK_STROKE_STYLE =
  FLAPPY_NEON_PALETTE.groundGridLine;

/**
 * Cache of computed RGBA stroke styles keyed by alpha.
 *
 * Reusing the same rgba string for repeated alpha values avoids per-frame
 * string allocation churn while still allowing arbitrary depth-aware alpha
 * values.
 */
const NEATENSTEIN_FLOOR_STROKE_STYLE_CACHE = new Map<number, string>();

/**
 * One depth-banded floor line segment, expressed as two screen-space points.
 */
type NeatensteinFloorSegment = readonly [number, number, number, number];

/**
 * Maps a normalized depth ratio to the index of the alpha band it belongs to.
 *
 * @param depthRatio - 0 (far, horizon) to 1 (near, bottom of screen).
 * @returns Band index in the range 0..`NEATENSTEIN_FLOOR_ALPHA_BANDS - 1`.
 */
function bandIndexForDepthRatio(depthRatio: number): number {
  return Math.min(
    NEATENSTEIN_FLOOR_ALPHA_BANDS - 1,
    Math.max(0, Math.floor(depthRatio * NEATENSTEIN_FLOOR_ALPHA_BANDS)),
  );
}

/**
 * Camera state consumed by the floor renderer.
 */
export interface NeatensteinFloorCamera {
  /** Camera yaw in radians. 0 looks down the world +X axis. */
  yaw: number;
  /** Camera world X position. */
  x: number;
  /** Camera world Y position. */
  y: number;
}

/**
 * Minimal canvas-like rendering context consumed by the floor renderer.
 *
 * The interface is intentionally narrow so lightweight mocks can be used in
 * unit tests without implementing the full CanvasRenderingContext2D API.
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
 * Resolves neon alpha for one row based on its normalized depth.
 *
 * @param depthRatio - Normalized 0..1 depth where 0 is far (at the horizon)
 *   and 1 is near (at the bottom edge).
 * @returns Opacity for the rendered row.
 */
export function resolveNeatensteinFloorAlpha(depthRatio: number): number {
  const clampedRatio = Math.max(0, Math.min(1, depthRatio));
  return (
    NEATENSTEIN_FLOOR_MIN_ALPHA +
    (NEATENSTEIN_FLOOR_MAX_ALPHA - NEATENSTEIN_FLOOR_MIN_ALPHA) * clampedRatio
  );
}

/**
 * Draw the world-fixed neon floor grid below the horizon.
 *
 * The grid is a procedural texture with thin lines at every integer world X
 * and Y. Instead of sampling every screen pixel, the renderer projects
 * world-space grid lines into screen space: for each integer line within a
 * bounded range around the camera, sample points are transformed into camera
 * space, perspective-projected, and connected into a single batched path per
 * depth band. A low-alpha, wide-line halo pass is drawn first to create a
 * subtle neon glow, followed by the normal core line pass. Both passes reuse
 * the same batched segment list so the glow cost stays predictable.
 *
 * @param ctx - Canvas-like context with path, stroke, and state methods.
 * @param canvasWidth - Canvas width in CSS pixels.
 * @param canvasHeight - Canvas height in CSS pixels.
 * @param camera - Current camera look state.
 *
 * @example
 * ```ts
 * drawNeatensteinFloor(ctx, 640, 360, { yaw: 0, x: 12.5, y: 12.5 });
 * ```
 */
export function drawNeatensteinFloor(
  ctx: NeatensteinFloorRenderContext,
  canvasWidth: number,
  canvasHeight: number,
  camera: NeatensteinFloorCamera,
): void {
  if (canvasWidth <= 0 || canvasHeight <= 0) {
    return;
  }

  const width = canvasWidth;
  const height = canvasHeight;
  const safeX = Number.isFinite(camera.x) ? camera.x : 0;
  const safeY = Number.isFinite(camera.y) ? camera.y : 0;
  const safeYaw = Number.isFinite(camera.yaw) ? camera.yaw : 0;

  const horizonY = height * NEATENSTEIN_FLOOR_HORIZON_RATIO;
  const halfWidth = width / 2;
  const focalLength = halfWidth / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);

  const cosYaw = Math.cos(safeYaw);
  const sinYaw = Math.sin(safeYaw);

  const cameraHeightWorld = NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD;

  const range = NEATENSTEIN_FLOOR_VISIBLE_CELL_RANGE;
  const minX = Math.floor(safeX - range);
  const maxX = Math.ceil(safeX + range);
  const minY = Math.floor(safeY - range);
  const maxY = Math.ceil(safeY + range);

  const bands: NeatensteinFloorSegment[][] = Array.from(
    { length: NEATENSTEIN_FLOOR_ALPHA_BANDS },
    () => [],
  );

  for (let x = minX; x <= maxX; x += 1) {
    appendNeatensteinFloorLine(
      bands,
      x,
      true,
      minY,
      maxY,
      safeX,
      safeY,
      cosYaw,
      sinYaw,
      focalLength,
      halfWidth,
      horizonY,
      height,
      cameraHeightWorld,
    );
  }

  for (let y = minY; y <= maxY; y += 1) {
    appendNeatensteinFloorLine(
      bands,
      y,
      false,
      minX,
      maxX,
      safeX,
      safeY,
      cosYaw,
      sinYaw,
      focalLength,
      halfWidth,
      horizonY,
      height,
      cameraHeightWorld,
    );
  }

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
      // Pass 1: wide, low-alpha halo for the neon glow.
      ctx.lineWidth = NEATENSTEIN_FLOOR_GLOW_WIDTH_PX;
      ctx.shadowBlur = 0;
      ctx.strokeStyle = resolveNeatensteinFloorStrokeStyle(
        FLOOR_BASE_RGB,
        coreAlpha * NEATENSTEIN_FLOOR_GLOW_ALPHA_MULTIPLIER,
      );
      strokeNeatensteinFloorBand(ctx, segments);

      // Pass 2: narrow, full-alpha core line.
      ctx.lineWidth = NEATENSTEIN_FLOOR_LINE_WIDTH_PX;
      ctx.strokeStyle = resolveNeatensteinFloorStrokeStyle(
        FLOOR_BASE_RGB,
        coreAlpha,
      );
      strokeNeatensteinFloorBand(ctx, segments);
    } else {
      // Single shadow-blur pass for a softer glow.
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
 * Stroke one batched band of floor segments.
 *
 * Reusing the same segment list for the glow halo and core line passes keeps
 * the path geometry identical so the glow sits directly behind the core line.
 *
 * @param ctx - Canvas-like context.
 * @param segments - Screen-space line segments for this band.
 */
function strokeNeatensteinFloorBand(
  ctx: NeatensteinFloorRenderContext,
  segments: NeatensteinFloorSegment[],
): void {
  ctx.beginPath();
  for (const [x1, y1, x2, y2] of segments) {
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
  }
  ctx.stroke();
}

/**
 * Backwards-compatible wrapper that reads canvas dimensions from `ctx.canvas`.
 *
 * @param ctx - Canvas-like context with an optional backing `canvas`.
 * @param camera - Current camera look state.
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
  const rawWidth = ctx.canvas?.width ?? 0;
  const rawHeight = ctx.canvas?.height ?? 0;
  drawNeatensteinFloor(ctx, rawWidth, rawHeight, camera);
}

/**
 * Append one world-space grid line to the depth-banded floor path.
 *
 * The line is sampled at evenly spaced points along its fixed coordinate.
 * Each sample is projected to screen space; consecutive visible samples are
 * connected into a segment and placed in the alpha band matching their
 * average depth. Invisible or behind-camera samples break the run so the next
 * visible sample starts a new segment.
 *
 * @param bands - Depth-banded segment buffers receiving the projected line
 *   segments.
 * @param fixedCoord - The integer world coordinate that defines this line
 *   (X for X-lines, Y for Y-lines).
 * @param isXLine - `true` for lines of constant X (vary Y), `false` for lines
 *   of constant Y (vary X).
 * @param startCoord - Start of the varying coordinate range.
 * @param endCoord - End of the varying coordinate range.
 * @param cameraX - Safe camera world X.
 * @param cameraY - Safe camera world Y.
 * @param cosYaw - Cosine of the safe camera yaw.
 * @param sinYaw - Sine of the safe camera yaw.
 * @param focalLength - Perspective focal length in pixels.
 * @param halfWidth - Half the canvas width in pixels.
 * @param horizonY - Horizon line Y coordinate in pixels.
 * @param height - Canvas height in pixels.
 * @param cameraHeight - Camera height above the floor in world units.
 */
function appendNeatensteinFloorLine(
  bands: NeatensteinFloorSegment[][],
  fixedCoord: number,
  isXLine: boolean,
  startCoord: number,
  endCoord: number,
  cameraX: number,
  cameraY: number,
  cosYaw: number,
  sinYaw: number,
  focalLength: number,
  halfWidth: number,
  horizonY: number,
  height: number,
  cameraHeight: number,
): void {
  let previous: { x: number; y: number; depthRatio: number } | null = null;

  for (let i = 0; i <= NEATENSTEIN_FLOOR_LINE_SAMPLES; i += 1) {
    const t = i / NEATENSTEIN_FLOOR_LINE_SAMPLES;
    const coord = startCoord + (endCoord - startCoord) * t;
    const worldX = isXLine ? fixedCoord : coord;
    const worldY = isXLine ? coord : fixedCoord;

    const projected = projectNeatensteinFloorPoint(
      worldX,
      worldY,
      cameraX,
      cameraY,
      cosYaw,
      sinYaw,
      focalLength,
      halfWidth,
      horizonY,
      height,
      cameraHeight,
    );

    if (projected === null) {
      previous = null;
      continue;
    }

    if (previous !== null) {
      const avgRatio = (previous.depthRatio + projected.depthRatio) / 2;
      const band = bandIndexForDepthRatio(avgRatio);
      bands[band].push([previous.x, previous.y, projected.x, projected.y]);
    }

    previous = projected;
  }
}

/**
 * Project a world-space floor point to screen space.
 *
 * The point is first translated by the camera position and rotated into
 * camera space using the camera yaw. The camera-space X axis points to the
 * right of the view and the Y axis points forward. Points at or behind the
 * camera (`camSpaceY <= 0`) are rejected.
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
 * @returns Screen coordinates, normalized depth ratio, and perpendicular
 *   camera-space depth, or `null` if the point is behind the camera.
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
  const dx = worldX - cameraX;
  const dy = worldY - cameraY;

  const camSpaceY = dx * cosYaw + dy * sinYaw;
  if (camSpaceY <= NEATENSTEIN_FLOOR_NEAR_PLANE_EPSILON) {
    return null;
  }

  const camSpaceX = -dx * sinYaw + dy * cosYaw;
  const screenY = horizonY + (cameraHeight / camSpaceY) * focalLength;
  const depthRatio = Math.max(
    0,
    Math.min(1, (screenY - horizonY) / (height - horizonY)),
  );

  return {
    x: halfWidth + (camSpaceX / camSpaceY) * focalLength,
    y: screenY,
    depthRatio,
    distance: camSpaceY,
  };
}

/**
 * Mix a base neon color with a depth-aware alpha.
 *
 * The base color is assumed to be a 6-digit hex string (e.g., `#0a8ea0`) as
 * supplied by the shared Flappy palette. If parsing fails, the original color
 * is returned unchanged.
 *
 * @param baseRgb - Parsed RGB tuple from the shared palette, or `null`.
 * @param alpha - Opacity in the range 0..1.
 * @returns RGBA color string ready for `ctx.strokeStyle`.
 */
function resolveNeatensteinFloorStrokeStyle(
  baseRgb: { r: number; g: number; b: number } | null,
  alpha: number,
): string {
  if (baseRgb === null) {
    return NEATENSTEIN_FLOOR_FALLBACK_STROKE_STYLE;
  }

  const cached = NEATENSTEIN_FLOOR_STROKE_STYLE_CACHE.get(alpha);
  if (cached !== undefined) {
    return cached;
  }

  const style = `rgba(${baseRgb.r}, ${baseRgb.g}, ${baseRgb.b}, ${alpha})`;
  NEATENSTEIN_FLOOR_STROKE_STYLE_CACHE.set(alpha, style);
  return style;
}

/**
 * Parse a 6-digit hex color into RGB components.
 *
 * @param baseColor - Hex color string.
 * @returns Parsed RGB tuple, or `null` if parsing fails.
 */
function parseNeatensteinFloorHexColor(
  baseColor: string,
): { r: number; g: number; b: number } | null {
  const hexDigits = baseColor.replace('#', '');
  if (hexDigits.length !== 6) {
    return null;
  }

  const red = parseInt(hexDigits.slice(0, 2), 16);
  const green = parseInt(hexDigits.slice(2, 4), 16);
  const blue = parseInt(hexDigits.slice(4, 6), 16);

  if (!Number.isFinite(red + green + blue)) {
    return null;
  }

  return { r: red, g: green, b: blue };
}
