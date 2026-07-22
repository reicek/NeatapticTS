/**
 * Synthwave floor renderer for the Neatenstein neon raycasting demo.
 *
 * This module renders the floor as a true 3D perspective grid using floor-casting
 * math. Horizontal depth rows are drawn at fixed world-space intervals
 * (transverse to the camera view), and longitudinal world-axis grid lines are
 * sampled at each depth row and drawn as polylines. The result is a grid of
 * perspective squares rather than a fan of rays converging to a single
 * vanishing point.
 *
 * @module
 */

import { FLAPPY_NEON_PALETTE } from '../../../flappy_bird/constants/constants.palette';

/**
 * Test fallback canvas width used when the render context has no backing
 * `canvas` (e.g., lightweight mock contexts in unit tests).
 *
 * This is **not** a production default; real rendering must read
 * `ctx.canvas.width` at runtime.
 */
export const NEATENSTEIN_FLOOR_DEFAULT_WIDTH = 320;

/**
 * Test fallback canvas height used when the render context has no backing
 * `canvas` (e.g., lightweight mock contexts in unit tests).
 *
 * This is **not** a production default; real rendering must read
 * `ctx.canvas.height` at runtime.
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
 * Number of horizontal depth rows drawn below the horizon.
 *
 * Each row corresponds to a fixed world-space distance from the camera.
 */
export const NEATENSTEIN_FLOOR_ROW_COUNT = 12;

/**
 * Horizontal field of view in radians.
 *
 * This controls how wide the visible floor span is at each depth row.
 */
export const NEATENSTEIN_FLOOR_FOV_RADIANS = Math.PI / 3;

/**
 * World-space size of one grid cell.
 *
 * The floor grid is aligned with the world X/Y axes. Depth rows and lateral grid
 * lines are spaced by this distance so the floor reads as a grid of squares.
 */
export const NEATENSTEIN_FLOOR_CELL_SIZE_WORLD = 1;

/**
 * Camera height above the floor, expressed as a fraction of the canvas height.
 *
 * This scale factor maps world-space depth to screen-space y offset in the
 * floor-casting formula `screenY = horizonY + cameraHeight / depth`.
 */
export const NEATENSTEIN_FLOOR_CAMERA_HEIGHT_RATIO = 0.5;

/**
 * Minimum line opacity, applied near the horizon (far depth rows).
 */
export const NEATENSTEIN_FLOOR_MIN_ALPHA = 0.12;

/**
 * Maximum line opacity, applied near the camera (near depth rows).
 */
export const NEATENSTEIN_FLOOR_MAX_ALPHA = 0.58;

/**
 * Minimum glow blur radius in pixels, applied near the horizon.
 */
const NEATENSTEIN_FLOOR_MIN_BLUR_PX = 0;

/**
 * Maximum glow blur radius in pixels, applied near the camera.
 */
const NEATENSTEIN_FLOOR_MAX_BLUR_PX = 8;

/**
 * Minimum stroke thickness in pixels, applied near the horizon.
 */
const NEATENSTEIN_FLOOR_MIN_THICKNESS_PX = 1;

/**
 * Maximum stroke thickness in pixels, applied near the camera.
 */
const NEATENSTEIN_FLOOR_MAX_THICKNESS_PX = 2.5;

/**
 * Small tolerance for treating a camera-plane component as zero.
 */
const NEATENSTEIN_FLOOR_PARALLEL_EPSILON = 1e-6;

/**
 * Pixel tolerance for skipping a parallel grid line that would overlap an
 * existing horizontal depth row stroke.
 */
const NEATENSTEIN_FLOOR_DUPLICATE_STROKE_THRESHOLD = 0.5;

/**
 * World axes iterated when projecting the floor grid.
 */
const NEATENSTEIN_FLOOR_WORLD_AXES = ['x', 'y'] as const;

/**
 * Off-screen culling margin expressed as a multiple of the canvas dimension.
 *
 * A value of 1 means one full screen width/height beyond each edge. The same
 * ratio is applied horizontally and vertically so the culling window stays
 * symmetric around the visible screen.
 */
const NEATENSTEIN_FLOOR_CULL_MARGIN_RATIO = 1;

/**
 * Maximum number of world-axis grid lines the renderer can project per frame.
 *
 * This bounds the reusable point and polyline pools. The value is derived from
 * the worst-case visible span at the farthest depth row with the current FOV.
 */
const NEATENSTEIN_FLOOR_MAX_GRID_LINES = 128;

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
 * Precomputed shadow color used for every floor grid stroke.
 *
 * The value is a constant palette string, captured once at module load to avoid
 * repeated property lookups in the per-frame hot path.
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
 * Reusable pool of projected depth rows.
 *
 * The same objects are refilled every frame so the render hot path does not
 * allocate new row containers.
 */
const FLOOR_ROWS_POOL: NeatensteinFloorRow[] = Array.from(
  { length: NEATENSTEIN_FLOOR_ROW_COUNT },
  () => ({ depth: 0, depthRatio: 0, screenY: 0 }),
);

/**
 * Reusable pool of projected screen-space points.
 *
 * Each point is allocated from this ring and reused on the next frame.
 */
const FLOOR_POINT_POOL: NeatensteinFloorPoint[] = Array.from(
  {
    length: NEATENSTEIN_FLOOR_ROW_COUNT * NEATENSTEIN_FLOOR_MAX_GRID_LINES,
  },
  () => ({ x: 0, y: 0, depthRatio: 0 }),
);

/** Cursor into {@link FLOOR_POINT_POOL}. Reset at the start of each frame. */
let floorPointCursor = 0;

/**
 * Reusable pool of projected polyline descriptors.
 *
 * Each descriptor references a contiguous range in {@link FLOOR_POINT_POOL}.
 */
const FLOOR_POLYLINE_POOL: NeatensteinFloorGridPolyline[] = Array.from(
  { length: NEATENSTEIN_FLOOR_MAX_GRID_LINES },
  () => ({ start: 0, count: 0, depthRatio: 0 }),
);

/** Cursor into {@link FLOOR_POLYLINE_POOL}. Reset at the start of each frame. */
let floorPolylineCursor = 0;

/**
 * Reusable result array for projected polylines.
 *
 * Cleared and refilled every frame to avoid per-frame array allocation.
 */
const FLOOR_POLYLINE_RESULT: NeatensteinFloorGridPolyline[] = [];

/**
 * Per-frame style buckets.
 *
 * Each bucket corresponds to one depth row and collects all grid polylines whose
 * average depth ratio rounds to that row. This lets the renderer stroke every
 * polyline that shares a style in a single path.
 */
const FLOOR_BUCKETS: NeatensteinFloorGridPolyline[][] = Array.from(
  { length: NEATENSTEIN_FLOOR_ROW_COUNT },
  () => [],
);

/**
 * Reusable render-dimension state.
 *
 * Filled once per frame so the render hot path never allocates a fresh
 * `{ width, height }` object.
 */
const FLOOR_DIMENSIONS = { width: 0, height: 0 };

/**
 * Reusable finite-guarded camera state.
 *
 * The same object is refilled every frame to avoid allocating a fresh camera
 * container in the render hot path.
 */
const FLOOR_SAFE_CAMERA: NeatensteinFloorCamera = { x: 0, y: 0, yaw: 0 };

/**
 * Reusable camera-plane state.
 *
 * Forward direction and camera-plane vectors are overwritten each frame so no
 * new object is allocated in the render hot path.
 */
const FLOOR_CAMERA_PLANE: NeatensteinFloorCameraPlane = {
  dirX: 0,
  dirY: 0,
  planeX: 0,
  planeY: 0,
};

/**
 * Reusable parameter object passed to {@link resolveNeatensteinFloorGridPolylines}.
 *
 * The same object is repopulated each frame to avoid per-frame allocation of
 * the options container.
 */
const FLOOR_GRID_PARAMS = {
  width: 0,
  height: 0,
  horizonY: 0,
  rows: FLOOR_ROWS_POOL,
  cameraPlane: FLOOR_CAMERA_PLANE,
  camera: FLOOR_SAFE_CAMERA,
  cameraHeight: 0,
  cellSize: NEATENSTEIN_FLOOR_CELL_SIZE_WORLD,
};

/**
 * Reusable parameter object passed to {@link resolveNeatensteinFloorParallelGridLine}.
 *
 * The same object is repopulated each grid line to avoid per-iteration
 * allocation of the options container.
 */
const FLOOR_PARALLEL_PARAMS = {
  width: 0,
  height: 0,
  horizonY: 0,
  worldCoordinate: 0,
  dirComponent: 0,
  cameraComponent: 0,
  cameraHeight: 0,
  rows: FLOOR_ROWS_POOL,
};

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
  strokeStyle?: string;
  /** Optional line width in pixels. */
  lineWidth?: number;
  /** Optional glow blur radius in pixels. */
  shadowBlur?: number;
  /** Optional glow color. */
  shadowColor?: string;
}

/**
 * A single projected depth row on the screen.
 */
interface NeatensteinFloorRow {
  /** World-space distance from the camera to this row. */
  depth: number;
  /** Normalized 0..1 depth where 0 is far and 1 is near. */
  depthRatio: number;
  /** Vertical screen coordinate of this row in pixels. */
  screenY: number;
}

/**
 * Camera direction and plane vectors used for floor-casting.
 */
interface NeatensteinFloorCameraPlane {
  /** Forward direction X component. */
  dirX: number;
  /** Forward direction Y component. */
  dirY: number;
  /** Camera-plane X component (perpendicular to direction, scaled by FOV). */
  planeX: number;
  /** Camera-plane Y component (perpendicular to direction, scaled by FOV). */
  planeY: number;
}

/**
 * 2D point in screen space.
 */
interface NeatensteinFloorPoint {
  /** Screen x coordinate in pixels. */
  x: number;
  /** Screen y coordinate in pixels. */
  y: number;
  /** Normalized depth ratio of the row this point was projected from. */
  depthRatio: number;
}

/**
 * Projected polyline for one world grid line.
 *
 * The actual point data lives in {@link FLOOR_POINT_POOL}; this descriptor
 * stores the contiguous range to draw.
 */
interface NeatensteinFloorGridPolyline {
  /** Index of the first point in {@link FLOOR_POINT_POOL}. */
  start: number;
  /** Number of points in the polyline. */
  count: number;
  /** Normalized depth used for styling (0 far, 1 near). */
  depthRatio: number;
}

/**
 * Resolves neon alpha for one line based on its normalized depth.
 *
 * @param depthRatio - Normalized 0..1 depth where 0 is far and 1 is near.
 * @returns Opacity for the rendered line.
 */
export function resolveNeatensteinFloorAlpha(depthRatio: number): number {
  const clampedRatio = Math.max(0, Math.min(1, depthRatio));
  return (
    NEATENSTEIN_FLOOR_MIN_ALPHA +
    (NEATENSTEIN_FLOOR_MAX_ALPHA - NEATENSTEIN_FLOOR_MIN_ALPHA) * clampedRatio
  );
}

/**
 * Resolves glow blur for one line based on its normalized depth.
 *
 * @param depthRatio - Normalized 0..1 depth where 0 is far and 1 is near.
 * @returns Blur radius in pixels.
 */
function resolveNeatensteinFloorBlur(depthRatio: number): number {
  const clampedRatio = Math.max(0, Math.min(1, depthRatio));
  return (
    NEATENSTEIN_FLOOR_MIN_BLUR_PX +
    (NEATENSTEIN_FLOOR_MAX_BLUR_PX - NEATENSTEIN_FLOOR_MIN_BLUR_PX) *
      clampedRatio
  );
}

/**
 * Resolves stroke thickness for one line based on its normalized depth.
 *
 * @param depthRatio - Normalized 0..1 depth where 0 is far and 1 is near.
 * @returns Stroke width in pixels.
 */
function resolveNeatensteinFloorThickness(depthRatio: number): number {
  const clampedRatio = Math.max(0, Math.min(1, depthRatio));
  return (
    NEATENSTEIN_FLOOR_MIN_THICKNESS_PX +
    (NEATENSTEIN_FLOOR_MAX_THICKNESS_PX - NEATENSTEIN_FLOOR_MIN_THICKNESS_PX) *
      clampedRatio
  );
}

/**
 * Render the neon synthwave floor grid below the horizon.
 *
 * The floor is drawn as a true perspective grid of squares using floor-casting
 * math. Horizontal depth rows run transverse to the view direction, and
 * longitudinal world-axis grid lines are sampled at each depth row and drawn
 * as polylines. Camera yaw rotates the grid; all context mutations are wrapped
 * in `ctx.save()` / `ctx.restore()`.
 *
 * @param ctx - Canvas-like context with path, stroke, and state methods.
 * @param camera - Current camera look state.
 * @returns void
 *
 * @example
 * ```ts
 * renderNeatensteinFloor(ctx, { yaw: 0, x: 0, y: 0 });
 * ```
 */
export function renderNeatensteinFloor(
  ctx: NeatensteinFloorRenderContext,
  camera: NeatensteinFloorCamera,
): void {
  resetNeatensteinFloorFrameState();

  resolveNeatensteinFloorDimensions(ctx);
  const width = FLOOR_DIMENSIONS.width;
  const height = FLOOR_DIMENSIONS.height;
  const horizonY = height * NEATENSTEIN_FLOOR_HORIZON_RATIO;
  const safeYaw = normalizeNeatensteinFloorYaw(camera.yaw);

  FLOOR_SAFE_CAMERA.yaw = safeYaw;
  FLOOR_SAFE_CAMERA.x = Number.isFinite(camera.x) ? camera.x : 0;
  FLOOR_SAFE_CAMERA.y = Number.isFinite(camera.y) ? camera.y : 0;

  resolveNeatensteinFloorCameraPlane(safeYaw, NEATENSTEIN_FLOOR_FOV_RADIANS);
  const cameraHeight = height * NEATENSTEIN_FLOOR_CAMERA_HEIGHT_RATIO;
  const rows = resolveNeatensteinFloorDepthRows(horizonY, cameraHeight);

  ctx.save();

  // Step 1: horizontal depth rows below the horizon (transverse to view).
  for (const row of rows) {
    applyNeatensteinFloorStyle(ctx, row.depthRatio);
    ctx.beginPath();
    ctx.moveTo(0, row.screenY);
    ctx.lineTo(width, row.screenY);
    ctx.stroke();
  }

  // Step 2: longitudinal world-axis grid lines drawn as polylines. Both
  // world X and world Y grid lines are projected at each depth row so the grid
  // stays complete regardless of camera yaw.
  FLOOR_GRID_PARAMS.width = width;
  FLOOR_GRID_PARAMS.height = height;
  FLOOR_GRID_PARAMS.horizonY = horizonY;
  FLOOR_GRID_PARAMS.cameraHeight = cameraHeight;
  const gridPolylines = resolveNeatensteinFloorGridPolylines(FLOOR_GRID_PARAMS);

  // Step 3: assign polylines to style buckets so all lines sharing a rounded
  // depth ratio can be stroked in a single path.
  for (const polyline of gridPolylines) {
    const bucketIndex = resolveNeatensteinFloorStyleBucketIndex(
      polyline.depthRatio,
    );
    FLOOR_BUCKETS[bucketIndex].push(polyline);
  }

  for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
    const bucket = FLOOR_BUCKETS[rowIndex];
    if (bucket.length === 0) {
      continue;
    }

    applyNeatensteinFloorStyle(ctx, rows[rowIndex].depthRatio);
    ctx.beginPath();
    for (const polyline of bucket) {
      if (polyline.count < 2) {
        continue;
      }
      const firstPoint = FLOOR_POINT_POOL[polyline.start];
      ctx.moveTo(firstPoint.x, firstPoint.y);
      for (
        let pointIndex = polyline.start + 1;
        pointIndex < polyline.start + polyline.count;
        pointIndex += 1
      ) {
        const point = FLOOR_POINT_POOL[pointIndex];
        ctx.lineTo(point.x, point.y);
      }
    }
    ctx.stroke();
  }

  ctx.restore();
}

/**
 * Resolve the active render dimensions from the context, falling back to the
 * test constants only when the backing canvas is missing or reports zero size.
 *
 * Mutates {@link FLOOR_DIMENSIONS} and returns the same object every frame so
 * the render hot path never allocates a fresh dimension container.
 *
 * @param ctx - Canvas-like render context.
 * @returns The reusable dimension state object.
 */
function resolveNeatensteinFloorDimensions(
  ctx: NeatensteinFloorRenderContext,
): { width: number; height: number } {
  const rawWidth = ctx.canvas?.width ?? 0;
  const rawHeight = ctx.canvas?.height ?? 0;
  FLOOR_DIMENSIONS.width =
    rawWidth > 0 ? rawWidth : NEATENSTEIN_FLOOR_DEFAULT_WIDTH;
  FLOOR_DIMENSIONS.height =
    rawHeight > 0 ? rawHeight : NEATENSTEIN_FLOOR_DEFAULT_HEIGHT;
  return FLOOR_DIMENSIONS;
}

/**
 * Wrap an external yaw value into the expected finite [-π, π] range.
 *
 * NaN or Infinity is treated as 0 so that bad input cannot poison the render
 * path with non-finite geometry.
 *
 * @param yaw - Raw horizontal look angle in radians.
 * @returns Finite yaw wrapped to [-π, π].
 */
function normalizeNeatensteinFloorYaw(yaw: number): number {
  if (!Number.isFinite(yaw)) {
    return 0;
  }
  const twoPi = 2 * Math.PI;
  let normalized = ((yaw % twoPi) + twoPi) % twoPi;
  if (normalized > Math.PI) {
    normalized -= twoPi;
  }
  return normalized;
}

/**
 * Build the camera direction and camera-plane vectors for floor-casting.
 *
 * Mutates {@link FLOOR_CAMERA_PLANE} and returns the same object every frame so
 * the render hot path never allocates a fresh camera-plane container.
 *
 * @param yaw - Normalized horizontal look angle in radians.
 * @param fovRadians - Full horizontal field of view in radians.
 * @returns The reusable camera-plane state object.
 */
function resolveNeatensteinFloorCameraPlane(
  yaw: number,
  fovRadians: number,
): NeatensteinFloorCameraPlane {
  const dirX = Math.cos(yaw);
  const dirY = Math.sin(yaw);
  const planeScale = Math.tan(fovRadians / 2);
  FLOOR_CAMERA_PLANE.dirX = dirX;
  FLOOR_CAMERA_PLANE.dirY = dirY;
  FLOOR_CAMERA_PLANE.planeX = -dirY * planeScale;
  FLOOR_CAMERA_PLANE.planeY = dirX * planeScale;
  return FLOOR_CAMERA_PLANE;
}

/**
 * Build the list of horizontal depth rows from the horizon down to the bottom
 * of the canvas.
 *
 * Rows are returned far-to-near so the depth ratio increases as the screen
 * coordinate approaches the bottom of the canvas.
 *
 * @param horizonY - Horizon y-coordinate in pixels.
 * @param cameraHeight - Camera height above the floor in pixels.
 * @returns Array of projected depth rows.
 */
function resolveNeatensteinFloorDepthRows(
  horizonY: number,
  cameraHeight: number,
): NeatensteinFloorRow[] {
  for (
    let poolIndex = 0, rowIndex = NEATENSTEIN_FLOOR_ROW_COUNT;
    rowIndex >= 1;
    poolIndex += 1, rowIndex -= 1
  ) {
    const row = FLOOR_ROWS_POOL[poolIndex];
    row.depth = rowIndex * NEATENSTEIN_FLOOR_CELL_SIZE_WORLD;
    row.screenY = horizonY + cameraHeight / row.depth;
    row.depthRatio =
      (NEATENSTEIN_FLOOR_ROW_COUNT - rowIndex) /
      (NEATENSTEIN_FLOOR_ROW_COUNT - 1);
  }
  return FLOOR_ROWS_POOL;
}

/**
 * Build all projected world-axis grid polylines for the current view.
 *
 * For each world X and world Y grid line index that could be visible, this
 * function samples the line at every depth row and produces a screen-space
 * polyline. When the camera looks almost exactly along one axis, the grid
 * lines for that axis become parallel to the screen plane; in that case they
 * are drawn as a single horizontal segment at the computed screen y, provided
 * the segment does not overlap a depth row that is already being drawn.
 *
 * @param input - Projection parameters.
 * @returns Array of polylines ready to stroke.
 */
function resolveNeatensteinFloorGridPolylines(input: {
  width: number;
  height: number;
  horizonY: number;
  rows: NeatensteinFloorRow[];
  cameraPlane: NeatensteinFloorCameraPlane;
  camera: NeatensteinFloorCamera;
  cameraHeight: number;
  cellSize: number;
}): NeatensteinFloorGridPolyline[] {
  const {
    width,
    height,
    horizonY,
    rows,
    cameraPlane,
    camera,
    cameraHeight,
    cellSize,
  } = input;

  FLOOR_POLYLINE_RESULT.length = 0;

  if (rows.length === 0) {
    return FLOOR_POLYLINE_RESULT;
  }

  const farDepth = rows[0].depth;
  const marginX = width * NEATENSTEIN_FLOOR_CULL_MARGIN_RATIO;
  const marginY = height * NEATENSTEIN_FLOOR_CULL_MARGIN_RATIO;
  const minX = -marginX;
  const maxX = width + marginX;
  const minY = -marginY;
  const maxY = height + marginY;

  for (const axis of NEATENSTEIN_FLOOR_WORLD_AXES) {
    const dirComponent = axis === 'x' ? cameraPlane.dirX : cameraPlane.dirY;
    const planeComponent =
      axis === 'x' ? cameraPlane.planeX : cameraPlane.planeY;
    const cameraComponent = axis === 'x' ? camera.x : camera.y;

    const worldSpanAtFar =
      farDepth * (Math.abs(dirComponent) + Math.abs(planeComponent));
    const minIndex =
      Math.floor((cameraComponent - worldSpanAtFar) / cellSize) - 1;
    const maxIndex =
      Math.ceil((cameraComponent + worldSpanAtFar) / cellSize) + 1;

    for (let index = minIndex; index <= maxIndex; index += 1) {
      const worldCoordinate = index * cellSize;

      if (Math.abs(planeComponent) < NEATENSTEIN_FLOOR_PARALLEL_EPSILON) {
        FLOOR_PARALLEL_PARAMS.width = width;
        FLOOR_PARALLEL_PARAMS.height = height;
        FLOOR_PARALLEL_PARAMS.horizonY = horizonY;
        FLOOR_PARALLEL_PARAMS.worldCoordinate = worldCoordinate;
        FLOOR_PARALLEL_PARAMS.dirComponent = dirComponent;
        FLOOR_PARALLEL_PARAMS.cameraComponent = cameraComponent;
        FLOOR_PARALLEL_PARAMS.cameraHeight = cameraHeight;
        const parallelLine = resolveNeatensteinFloorParallelGridLine(
          FLOOR_PARALLEL_PARAMS,
        );
        if (parallelLine !== null) {
          FLOOR_POLYLINE_RESULT.push(parallelLine);
        }
        continue;
      }

      const start = floorPointCursor;
      let projectedCount = 0;

      for (const row of rows) {
        const rayOffset =
          worldCoordinate - cameraComponent - row.depth * dirComponent;
        const screenX =
          (width / 2) * (1 + rayOffset / (row.depth * planeComponent));
        if (Number.isFinite(screenX)) {
          const point = allocateNeatensteinFloorPoint();
          point.x = screenX;
          point.y = row.screenY;
          point.depthRatio = row.depthRatio;
          projectedCount += 1;
        }
      }

      let visibleCount = 0;
      let visibleDepthRatioSum = 0;
      for (
        let readIndex = start;
        readIndex < start + projectedCount;
        readIndex += 1
      ) {
        const point = FLOOR_POINT_POOL[readIndex];
        if (
          point.x >= minX &&
          point.x <= maxX &&
          point.y >= minY &&
          point.y <= maxY
        ) {
          const writeIndex = start + visibleCount;
          if (writeIndex !== readIndex) {
            FLOOR_POINT_POOL[writeIndex] = point;
          }
          visibleDepthRatioSum += point.depthRatio;
          visibleCount += 1;
        }
      }

      if (visibleCount < 2) {
        floorPointCursor = start;
        continue;
      }

      const polyline = allocateNeatensteinFloorPolyline();
      polyline.start = start;
      polyline.count = visibleCount;
      polyline.depthRatio = visibleDepthRatioSum / visibleCount;
      FLOOR_POLYLINE_RESULT.push(polyline);
    }
  }

  return FLOOR_POLYLINE_RESULT;
}

/**
 * Project one grid line that is nearly parallel to the screen plane.
 *
 * When the camera plane component for an axis is close to zero, the standard
 * per-row screen-x formula divides by a near-zero value. Instead, the grid
 * line intersects the camera forward line at a single world-space depth,
 * which maps to one horizontal screen segment. Segments that coincide with
 * an existing depth row are skipped to avoid drawing the same transverse stroke
 * twice when the camera is axis-aligned.
 *
 * @param input - Parallel-line projection parameters.
 * @returns A two-point polyline or `null` if off-screen, behind the camera, or
 *   overlapping a depth row.
 */
function resolveNeatensteinFloorParallelGridLine(input: {
  width: number;
  height: number;
  horizonY: number;
  worldCoordinate: number;
  dirComponent: number;
  cameraComponent: number;
  cameraHeight: number;
  rows: NeatensteinFloorRow[];
}): NeatensteinFloorGridPolyline | null {
  const {
    width,
    height,
    horizonY,
    worldCoordinate,
    dirComponent,
    cameraComponent,
    cameraHeight,
    rows,
  } = input;

  if (Math.abs(dirComponent) < NEATENSTEIN_FLOOR_PARALLEL_EPSILON) {
    return null;
  }

  const depth = (worldCoordinate - cameraComponent) / dirComponent;
  if (depth <= 0 || !Number.isFinite(depth)) {
    return null;
  }

  const screenY = horizonY + cameraHeight / depth;
  if (screenY <= horizonY || screenY > height) {
    return null;
  }

  const overlapsDepthRow = rows.some(
    (row) =>
      Math.abs(row.screenY - screenY) <
      NEATENSTEIN_FLOOR_DUPLICATE_STROKE_THRESHOLD,
  );
  if (overlapsDepthRow) {
    return null;
  }

  const depthRatio = (screenY - horizonY) / (height - horizonY);
  const start = floorPointCursor;
  const left = allocateNeatensteinFloorPoint();
  left.x = 0;
  left.y = screenY;
  left.depthRatio = depthRatio;
  const right = allocateNeatensteinFloorPoint();
  right.x = width;
  right.y = screenY;
  right.depthRatio = depthRatio;

  const polyline = allocateNeatensteinFloorPolyline();
  polyline.start = start;
  polyline.count = 2;
  polyline.depthRatio = depthRatio;
  return polyline;
}

/**
 * Apply a fixed neon line style with depth-aware alpha, blur, and thickness.
 *
 * @param ctx - Render context whose stroke properties will be mutated.
 * @param depthRatio - Normalized 0..1 depth where 0 is far and 1 is near.
 */
function applyNeatensteinFloorStyle(
  ctx: NeatensteinFloorRenderContext,
  depthRatio: number,
): void {
  ctx.strokeStyle = resolveNeatensteinFloorStrokeStyle(
    FLOOR_BASE_RGB,
    resolveNeatensteinFloorAlpha(depthRatio),
  );
  ctx.lineWidth = resolveNeatensteinFloorThickness(depthRatio);
  ctx.shadowBlur = resolveNeatensteinFloorBlur(depthRatio);
  ctx.shadowColor = NEATENSTEIN_FLOOR_SHADOW_COLOR;
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

/**
 * Reset all reusable frame state at the start of a render.
 */
function resetNeatensteinFloorFrameState(): void {
  floorPointCursor = 0;
  floorPolylineCursor = 0;
  FLOOR_POLYLINE_RESULT.length = 0;
  for (const bucket of FLOOR_BUCKETS) {
    bucket.length = 0;
  }
}

/**
 * Allocate a reusable point from {@link FLOOR_POINT_POOL}.
 *
 * @returns A point object whose fields will be overwritten by the caller.
 */
function allocateNeatensteinFloorPoint(): NeatensteinFloorPoint {
  const point = FLOOR_POINT_POOL[floorPointCursor];
  floorPointCursor += 1;
  return point;
}

/**
 * Allocate a reusable polyline descriptor from {@link FLOOR_POLYLINE_POOL}.
 *
 * @returns A polyline descriptor whose fields will be overwritten by the caller.
 */
function allocateNeatensteinFloorPolyline(): NeatensteinFloorGridPolyline {
  const polyline = FLOOR_POLYLINE_POOL[floorPolylineCursor];
  floorPolylineCursor += 1;
  return polyline;
}

/**
 * Map a depth ratio to the nearest style bucket (one per depth row).
 *
 * @param depthRatio - Normalized 0..1 depth where 0 is far and 1 is near.
 * @returns Bucket index in [0, NEATENSTEIN_FLOOR_ROW_COUNT - 1].
 */
function resolveNeatensteinFloorStyleBucketIndex(depthRatio: number): number {
  const clampedRatio = Math.max(0, Math.min(1, depthRatio));
  return Math.round(clampedRatio * (NEATENSTEIN_FLOOR_ROW_COUNT - 1));
}
