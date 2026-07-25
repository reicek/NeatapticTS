/**
 * Display worker entrypoint for the Neatenstein neon raycasting demo.
 *
 * This worker receives an `init` message that selects a renderer tier, then
 * either:
 *
 * - renders directly to an {@link OffscreenCanvas} for the `worker` tier, or
 * - builds a packed {@link NeatensteinRenderFrame} and posts it back to the
 *   host for the `cpu` and `gpu` tiers.
 *
 * The worker also defensively constrains render dimensions to the Neatenstein
 * maximum output bounds. This keeps direct OffscreenCanvas rendering aligned
 * with the same maximum resolution policy used by the host canvas backing
 * store.
 *
 * @module
 */

/// <reference lib="webworker" />

import {
  NEATENSTEIN_CPU_COLUMN_COUNT,
  NEATENSTEIN_DEFAULT_SEED,
  NEATENSTEIN_FIXED_TIMESTEP_MS,
  NEATENSTEIN_GPU_COLUMN_COUNT,
  NEATENSTEIN_IMPACT_SPOT_COLOR,
  NEATENSTEIN_IMPACT_SPOT_GLOW_BLUR_PX,
  NEATENSTEIN_IMPACT_SPOT_GLOW_COLOR,
  NEATENSTEIN_IMPACT_SPOT_LIFETIME_MS,
  NEATENSTEIN_IMPACT_SPOT_RADIUS_PX,
  NEATENSTEIN_INPUT_MESSAGE_TYPE,
  NEATENSTEIN_MAP_SIZE,
  NEATENSTEIN_PULSE_ALPHA_MAX,
  NEATENSTEIN_PULSE_ALPHA_MIN,
  NEATENSTEIN_PULSE_GLOW_BLUR_RADIUS,
  NEATENSTEIN_PULSE_MAX_CONCURRENT,
  NEATENSTEIN_PULSE_SCREEN_DOT_RADIUS_PX,
  NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
  NEATENSTEIN_TRACER_COLOR,
  NEATENSTEIN_TRACER_GLOW_BLUR_RADIUS,
  NEATENSTEIN_TRACER_GLOW_COLOR,
  NEATENSTEIN_TRACER_LINE_WIDTH,
  NEATENSTEIN_TRACER_NEAR_CLIP_EPSILON,
  NEATENSTEIN_WORKER_COLUMN_COUNT,
} from '../constants';
import {
  buildNeatensteinRenderFrame,
  resolveNeatensteinRenderFrameTransferList,
  type NeatensteinRenderState,
} from '../renderer/frame';
import {
  drawNeatensteinCeiling,
  drawNeatensteinFloor,
  NEATENSTEIN_FLOOR_CAMERA_HEIGHT_SCREEN_RATIO,
  NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
  NEATENSTEIN_FLOOR_FOV_RADIANS,
  NEATENSTEIN_FLOOR_HORIZON_RATIO,
  projectNeatensteinCeilingPoint,
  projectNeatensteinFloorPoint,
  type NeatensteinFloorCamera,
} from '../renderer/floor';
import {
  depthTestPulse,
  emitNeatensteinAmbientPulse,
  NEATENSTEIN_PULSE_AMBIENT_LIFETIME_TICKS,
  NEATENSTEIN_PULSE_LAYER_CEILING,
  NEATENSTEIN_PULSE_LAYER_FLOOR,
  updateNeatensteinPulses,
  type NeatensteinPulse,
} from '../renderer/pulse';
import {
  buildNeatensteinMap,
  createCollisionMap,
  type CollisionMap,
} from '../renderer/map';
import { castRayDDAFromFlatMap } from '../renderer/raycast';
import {
  createGameState,
  gameTick,
  type GameTickInputSnapshot,
} from '../host/game/tick';
import type {
  GameState,
  ImpactSpot,
  TracerState,
  Vector2,
} from '../host/game/types';
import {
  NEATENSTEIN_BACKGROUND_RGB,
  NEATENSTEIN_MAX_VIEW_DIST,
} from '../renderer/framebuffer';

type DisplayTier = 'worker' | 'cpu' | 'gpu';

let currentTier: DisplayTier | null = null;
let workerCanvas: OffscreenCanvas | null = null;
let workerContext: OffscreenCanvasRenderingContext2D | null = null;
let latestState: NeatensteinRenderState | null = null;

/** Canonical flat deterministic map used by the worker raycaster. */
let wallMap: Uint8Array | null = null;

/** Reusable worker-tier z-buffer, resized only when the render width changes. */
let workerZBuffer: Float32Array | null = null;

/** Deterministic game state maintained and advanced by the worker. */
let gameState: GameState | null = null;

/** Collision map built from the init seed; reused for every movement tick. */
let collisionMap: CollisionMap | null = null;

/** Active ambient floor and ceiling pulses tracked across worker frames. */
let activePulses: NeatensteinPulse[] = [];

/**
 * Input snapshot captured from the most recent `input` message.
 *
 * Movement/look use the latest value, while one-shot actions such as fire and
 * dash are latched until the next simulation tick consumes them.
 */
let pendingTickInput: GameTickInputSnapshot | null = null;

/**
 * Worker-side maximum canvas width.
 *
 * The source constant is named for GPU raycasting columns, but in this worker
 * context it represents the maximum horizontal render resolution.
 */
const NEATENSTEIN_WORKER_MAX_CANVAS_WIDTH = NEATENSTEIN_GPU_COLUMN_COUNT * 2;

/**
 * Worker-side maximum canvas height.
 *
 * The source constant is named for worker columns in the wider renderer
 * terminology, but in this worker context it is used as the maximum vertical
 * render resolution.
 */
const NEATENSTEIN_WORKER_MAX_CANVAS_HEIGHT =
  NEATENSTEIN_WORKER_COLUMN_COUNT * 2;

/**
 * Background clear color used before each worker-tier frame.
 *
 * Derived from {@link NEATENSTEIN_BACKGROUND_RGB} so the worker tier matches
 * the CPU/GPU fog background exactly.
 */
const NEATENSTEIN_WORKER_CLEAR_COLOR = formatRgb(NEATENSTEIN_BACKGROUND_RGB);

/** RGB of the neon wall color for X-axis-side hits. */
const NEATENSTEIN_WALL_X_SIDE_RGB = { r: 0, g: 191, b: 255 } as const;

/** RGB of the neon wall color for Y-axis-side hits. */
const NEATENSTEIN_WALL_Y_SIDE_RGB = { r: 0, g: 80, b: 180 } as const;

/** CSS color string for the yellow ambient pulse dot. */
const NEATENSTEIN_PULSE_COLOR = '#fff14a';

/** CSS shadow color string for the yellow ambient pulse glow. */
const NEATENSTEIN_PULSE_GLOW_COLOR = 'rgba(255, 241, 74, 0.42)';

/**
 * Integer render size used by the worker after applying max-resolution bounds.
 */
interface ConstrainedRenderSize {
  /** Constrained backing-store width in pixels. */
  width: number;
  /** Constrained backing-store height in pixels. */
  height: number;
}

/**
 * Format an RGB triple as a CSS `rgb(...)` string.
 *
 * @param color - Object with r, g, b components.
 * @returns A CSS `rgb(...)` color string.
 */
function formatRgb(color: { r: number; g: number; b: number }): string {
  return `rgb(${Math.round(color.r)}, ${Math.round(color.g)}, ${Math.round(color.b)})`;
}

/**
 * Clamp a value to a `[min, max]` range.
 *
 * @param value - Value to clamp.
 * @param min - Inclusive minimum.
 * @param max - Inclusive maximum.
 * @returns The clamped value.
 */
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Return whether a value is usable as a positive render dimension.
 *
 * @param value - Candidate dimension.
 * @returns Whether the dimension is positive and finite.
 */
function isPositiveFiniteDimension(value: number): boolean {
  return Number.isFinite(value) && value > 0;
}

/**
 * Resolve a constrained render size from the latest host-provided dimensions.
 *
 * The host is expected to constrain the visible canvas backing store before
 * transfer, but the worker repeats the operation defensively because it is the
 * final owner of the direct OffscreenCanvas render target.
 *
 * The result:
 *
 * - preserves the source aspect ratio
 * - uses the largest size inside the worker max bounds
 * - may upscale or downscale relative to the incoming size
 * - always returns positive integer dimensions
 *
 * @param sourceWidth - Incoming render width from the host state.
 * @param sourceHeight - Incoming render height from the host state.
 * @returns Constrained integer render size, or `null` for invalid input.
 */
function resolveConstrainedRenderSize(
  sourceWidth: number,
  sourceHeight: number,
): ConstrainedRenderSize | null {
  if (
    !isPositiveFiniteDimension(sourceWidth) ||
    !isPositiveFiniteDimension(sourceHeight)
  ) {
    return null;
  }

  // Use the smaller scale so both dimensions fit within the maximum bounds.
  // Do not cap at 1: the CSS/host size provides aspect ratio, while these
  // bounds define the intended maximum render resolution.
  const scale = Math.min(
    NEATENSTEIN_WORKER_MAX_CANVAS_WIDTH / sourceWidth,
    NEATENSTEIN_WORKER_MAX_CANVAS_HEIGHT / sourceHeight,
  );

  return {
    width: Math.max(1, Math.floor(sourceWidth * scale)),
    height: Math.max(1, Math.floor(sourceHeight * scale)),
  };
}

/**
 * Synchronize the transferred OffscreenCanvas with the constrained render size.
 *
 * This is critical for direct worker rendering. Projection math, floor/ceiling
 * drawing, wall stripes, z-buffer columns, and the actual canvas backing store
 * must agree on the same dimensions.
 *
 * @param canvas - Transferred worker-owned canvas.
 * @param size - Constrained render size.
 */
function syncWorkerCanvasSize(
  canvas: OffscreenCanvas,
  size: ConstrainedRenderSize,
): void {
  if (canvas.width !== size.width) {
    canvas.width = size.width;
  }

  if (canvas.height !== size.height) {
    canvas.height = size.height;
  }
}

/**
 * Resolve the direct worker-tier raycast column count.
 *
 * The worker tier renders directly into the OffscreenCanvas, so its horizontal
 * ray density should match the constrained backing-store width. This prevents a
 * lower-resolution column set from being stretched across a wider canvas.
 *
 * @param canvasWidth - Constrained canvas backing-store width.
 * @returns Number of direct worker raycast columns.
 */
function resolveWorkerCanvasColumnCount(canvasWidth: number): number {
  if (!isPositiveFiniteDimension(canvasWidth)) {
    return 0;
  }

  return Math.min(
    NEATENSTEIN_WORKER_MAX_CANVAS_WIDTH,
    Math.max(1, Math.floor(canvasWidth)),
  );
}

/**
 * Map a packed-frame tier to the column count used in typed frame payloads.
 *
 * Direct worker rendering does not use this helper; it raycasts at the
 * constrained canvas width instead.
 *
 * @param tier - Active renderer tier.
 * @returns Packed-frame column count.
 */
function resolvePackedColumnCount(tier: DisplayTier): number {
  switch (tier) {
    case 'gpu':
      return NEATENSTEIN_GPU_COLUMN_COUNT;
    case 'cpu':
    case 'worker':
    default:
      return NEATENSTEIN_CPU_COLUMN_COUNT;
  }
}

/**
 * Resolve and clear the reusable worker z-buffer.
 *
 * @param columnCount - Required number of z-buffer columns.
 * @returns A cleared z-buffer.
 */
function resolveWorkerZBuffer(columnCount: number): Float32Array {
  if (!workerZBuffer || workerZBuffer.length !== columnCount) {
    workerZBuffer = new Float32Array(columnCount);
  }

  workerZBuffer.fill(Number.POSITIVE_INFINITY);
  return workerZBuffer;
}

/**
 * Resolve a safe distance-fog interpolation factor.
 *
 * @param perpWallDist - Perpendicular wall distance.
 * @returns Fog factor in `[0, 1]`.
 */
function resolveWallFogFactor(perpWallDist: number): number {
  if (!Number.isFinite(perpWallDist)) {
    return 1;
  }

  return clamp(perpWallDist / NEATENSTEIN_MAX_VIEW_DIST, 0, 1);
}

/**
 * Blend a neon wall color toward the background based on wall distance.
 *
 * @param wallColor - Raw wall RGB.
 * @param perpWallDist - Perpendicular distance from camera to wall.
 * @returns CSS color string for the fogged wall stripe.
 */
function applyWallFog(
  wallColor: { r: number; g: number; b: number },
  perpWallDist: number,
): string {
  const fogFactor = resolveWallFogFactor(perpWallDist);
  const backgroundColor = NEATENSTEIN_BACKGROUND_RGB;

  return formatRgb({
    r: wallColor.r + (backgroundColor.r - wallColor.r) * fogFactor,
    g: wallColor.g + (backgroundColor.g - wallColor.g) * fogFactor,
    b: wallColor.b + (backgroundColor.b - wallColor.b) * fogFactor,
  });
}

/** Return type of the shared raycaster used to build every column. */
type RaycastHit = ReturnType<typeof castRayDDAFromFlatMap>;

/**
 * Cast a single camera ray for the given column index.
 *
 * @param column - Column index in `[0, columnCount)`.
 * @param columnCount - Total number of raycast columns.
 * @param cameraPositionX - Camera X position.
 * @param cameraPositionY - Camera Y position.
 * @param cameraDirectionX - Camera direction X.
 * @param cameraDirectionY - Camera direction Y.
 * @param cameraPlaneX - Camera plane X.
 * @param cameraPlaneY - Camera plane Y.
 * @returns DDA raycast hit or miss sentinel.
 */
function castColumnRay(
  column: number,
  columnCount: number,
  cameraPositionX: number,
  cameraPositionY: number,
  cameraDirectionX: number,
  cameraDirectionY: number,
  cameraPlaneX: number,
  cameraPlaneY: number,
): RaycastHit {
  if (!wallMap) {
    return {
      perpWallDist: Number.POSITIVE_INFINITY,
      side: 0,
      mapX: -1,
      mapY: -1,
    };
  }

  // Map the column index to a -1..+1 offset on the camera plane.
  const cameraPlaneOffset = (2 * column) / columnCount - 1;

  // Combine camera forward direction with the camera plane offset.
  const rayDirectionX = cameraDirectionX + cameraPlaneX * cameraPlaneOffset;
  const rayDirectionY = cameraDirectionY + cameraPlaneY * cameraPlaneOffset;

  return castRayDDAFromFlatMap(
    wallMap,
    NEATENSTEIN_MAP_SIZE,
    cameraPositionX,
    cameraPositionY,
    rayDirectionX,
    rayDirectionY,
  );
}

/**
 * Build a packed frame from the latest state or render directly to the worker
 * OffscreenCanvas.
 *
 * Worker-tier painter order:
 *
 * clear → floor → ceiling → walls → pulses → tracers → impact spots
 */
function buildAndPostFrame(): void {
  if (!latestState || !currentTier || !wallMap || !gameState) {
    return;
  }

  const constrainedSize = resolveConstrainedRenderSize(
    latestState.canvasWidth,
    latestState.canvasHeight,
  );

  if (constrainedSize === null) {
    return;
  }

  const canvasWidth = constrainedSize.width;
  const canvasHeight = constrainedSize.height;

  const cameraPositionX = gameState.player.position.x;
  const cameraPositionY = gameState.player.position.y;
  const cameraYaw = gameState.player.angleRad;

  // Derive camera direction and projection plane from the player yaw.
  const cameraDirectionX = Math.cos(cameraYaw);
  const cameraDirectionY = Math.sin(cameraYaw);
  const planeScale = Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
  const cameraPlaneX = -cameraDirectionY * planeScale;
  const cameraPlaneY = cameraDirectionX * planeScale;

  if (currentTier === 'worker') {
    if (!workerCanvas) {
      return;
    }

    // The worker owns the transferred canvas, so it must enforce the final
    // constrained backing-store dimensions before any drawing happens.
    syncWorkerCanvasSize(workerCanvas, constrainedSize);

    if (!workerContext) {
      workerContext = workerCanvas.getContext('2d');
    }

    const context = workerContext;
    if (!context) {
      return;
    }

    const columnCount = resolveWorkerCanvasColumnCount(canvasWidth);
    if (columnCount <= 0) {
      return;
    }

    context.fillStyle = NEATENSTEIN_WORKER_CLEAR_COLOR;
    context.fillRect(0, 0, canvasWidth, canvasHeight);

    // Draw with explicit dimensions so floor/ceiling cannot accidentally read
    // stale context canvas dimensions.
    drawNeatensteinFloor(context, canvasWidth, canvasHeight, {
      x: cameraPositionX,
      y: cameraPositionY,
      yaw: cameraYaw,
    });

    drawNeatensteinCeiling(context, canvasWidth, canvasHeight, {
      x: cameraPositionX,
      y: cameraPositionY,
      yaw: cameraYaw,
    });

    const stripeWidth = canvasWidth / columnCount;
    const wallFocalLength =
      canvasWidth / 2 / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
    const zBuffer = resolveWorkerZBuffer(columnCount);

    for (let column = 0; column < columnCount; column += 1) {
      const hit = castColumnRay(
        column,
        columnCount,
        cameraPositionX,
        cameraPositionY,
        cameraDirectionX,
        cameraDirectionY,
        cameraPlaneX,
        cameraPlaneY,
      );

      zBuffer[column] = hit.perpWallDist;

      const lineHeight = wallFocalLength / hit.perpWallDist;
      const drawStart = clamp((canvasHeight - lineHeight) / 2, 0, canvasHeight);
      const drawEnd = clamp((canvasHeight + lineHeight) / 2, 0, canvasHeight);

      context.fillStyle = applyWallFog(
        hit.side === 0
          ? NEATENSTEIN_WALL_X_SIDE_RGB
          : NEATENSTEIN_WALL_Y_SIDE_RGB,
        hit.perpWallDist,
      );

      // Integer stripe bounds prevent subpixel gaps when the column count does
      // not divide the canvas width evenly.
      const xStart = Math.floor(column * stripeWidth);
      const xEnd = Math.floor((column + 1) * stripeWidth);
      const stripePixelWidth = Math.max(0, xEnd - xStart);

      if (stripePixelWidth > 0 && drawStart < drawEnd) {
        context.fillRect(
          xStart,
          drawStart,
          stripePixelWidth,
          drawEnd - drawStart,
        );
      }
    }

    // Update and emit ambient pulses after the wall z-buffer exists.
    activePulses = updateNeatensteinPulses(activePulses, latestState.simTick);

    const newFloorPulse = emitNeatensteinAmbientPulse(
      latestState.simTick,
      gameState.seed,
      NEATENSTEIN_PULSE_LAYER_FLOOR,
    );

    if (
      newFloorPulse &&
      activePulses.length < NEATENSTEIN_PULSE_MAX_CONCURRENT
    ) {
      activePulses.push(newFloorPulse);
    }

    const newCeilingPulse = emitNeatensteinAmbientPulse(
      latestState.simTick,
      gameState.seed,
      NEATENSTEIN_PULSE_LAYER_CEILING,
    );

    if (
      newCeilingPulse &&
      activePulses.length < NEATENSTEIN_PULSE_MAX_CONCURRENT
    ) {
      activePulses.push(newCeilingPulse);
    }

    drawNeatensteinPulses(
      context,
      activePulses,
      zBuffer,
      { x: cameraPositionX, y: cameraPositionY, yaw: cameraYaw },
      canvasWidth,
      canvasHeight,
    );

    drawNeatensteinTracers(
      context,
      gameState.tracers,
      { x: cameraPositionX, y: cameraPositionY, yaw: cameraYaw },
      canvasWidth,
      canvasHeight,
    );

    drawImpactSpots(
      context,
      gameState.impacts,
      zBuffer,
      { x: cameraPositionX, y: cameraPositionY, yaw: cameraYaw },
      canvasWidth,
      canvasHeight,
    );

    return;
  }

  // CPU/GPU tiers ship packed frame data back to the host.
  const columnCount = resolvePackedColumnCount(currentTier);
  const renderState = {
    ...latestState,
    canvasWidth,
    canvasHeight,
  };

  const frame = buildNeatensteinRenderFrame(renderState, columnCount);

  for (let column = 0; column < columnCount; column += 1) {
    const hit = castColumnRay(
      column,
      columnCount,
      cameraPositionX,
      cameraPositionY,
      cameraDirectionX,
      cameraDirectionY,
      cameraPlaneX,
      cameraPlaneY,
    );

    frame.wallDistances[column] = hit.perpWallDist;
    frame.wallSides[column] = hit.side;
    frame.zBuffer[column] = hit.perpWallDist;
  }

  self.postMessage(
    { type: 'frame', frame },
    resolveNeatensteinRenderFrameTransferList(frame),
  );
}

/**
 * Draw active ambient floor and ceiling pulses that pass the z-buffer test.
 *
 * @param context - Worker-tier 2D canvas context.
 * @param pulses - Active pulses.
 * @param zBuffer - Per-column depth buffer from the wall pass.
 * @param camera - Current camera state.
 * @param canvasWidth - Canvas width in pixels.
 * @param canvasHeight - Canvas height in pixels.
 */
function drawNeatensteinPulses(
  context: OffscreenCanvasRenderingContext2D,
  pulses: readonly NeatensteinPulse[],
  zBuffer: Float32Array,
  camera: NeatensteinFloorCamera,
  canvasWidth: number,
  canvasHeight: number,
): void {
  const safeX = Number.isFinite(camera.x) ? camera.x : 0;
  const safeY = Number.isFinite(camera.y) ? camera.y : 0;
  const safeYaw = Number.isFinite(camera.yaw) ? camera.yaw : 0;

  const horizonY = canvasHeight * NEATENSTEIN_FLOOR_HORIZON_RATIO;
  const halfWidth = canvasWidth / 2;
  const focalLength = halfWidth / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
  const cosYaw = Math.cos(safeYaw);
  const sinYaw = Math.sin(safeYaw);

  for (const pulse of pulses) {
    if (!pulse.active) {
      continue;
    }

    const projected =
      pulse.layer === NEATENSTEIN_PULSE_LAYER_CEILING
        ? projectNeatensteinCeilingPoint(
            pulse.worldX,
            pulse.worldY,
            safeX,
            safeY,
            cosYaw,
            sinYaw,
            focalLength,
            halfWidth,
            horizonY,
            canvasHeight,
            NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
          )
        : projectNeatensteinFloorPoint(
            pulse.worldX,
            pulse.worldY,
            safeX,
            safeY,
            cosYaw,
            sinYaw,
            focalLength,
            halfWidth,
            horizonY,
            canvasHeight,
            NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
          );

    if (projected === null) {
      continue;
    }

    const depthTestPulseInput = {
      screenColumn: (projected.x / canvasWidth) * zBuffer.length,
      distance: projected.distance,
    };

    if (!depthTestPulse(depthTestPulseInput, zBuffer)) {
      continue;
    }

    const alpha = clamp(
      pulse.lifetimeTicks / NEATENSTEIN_PULSE_AMBIENT_LIFETIME_TICKS,
      NEATENSTEIN_PULSE_ALPHA_MIN,
      NEATENSTEIN_PULSE_ALPHA_MAX,
    );

    context.shadowColor = NEATENSTEIN_PULSE_GLOW_COLOR;
    context.shadowBlur = NEATENSTEIN_PULSE_GLOW_BLUR_RADIUS;
    context.fillStyle = NEATENSTEIN_PULSE_COLOR;
    context.globalAlpha = alpha;
    context.beginPath();
    context.arc(
      projected.x,
      projected.y,
      NEATENSTEIN_PULSE_SCREEN_DOT_RADIUS_PX,
      0,
      Math.PI * 2,
    );
    context.fill();
  }

  context.globalAlpha = 1;
  context.shadowBlur = 0;
}

/** Camera-relative transform of a world point for tracer projection. */
interface TracerCamera {
  /** Camera world X position. */
  x: number;
  /** Camera world Y position. */
  y: number;
  /** Camera yaw in radians. */
  yaw: number;
}

/**
 * Project a world-space point into screen coordinates.
 *
 * @param point - World-space point.
 * @param camera - Camera position and yaw.
 * @param canvasWidth - Canvas width.
 * @param canvasHeight - Canvas height.
 * @returns Screen coordinates, or `null` if behind the camera.
 */
function projectTracerPoint(
  point: Vector2,
  camera: TracerCamera,
  canvasWidth: number,
  canvasHeight: number,
): { x: number; y: number } | null {
  const dx = point.x - camera.x;
  const dy = point.y - camera.y;

  const cos = Math.cos(camera.yaw);
  const sin = Math.sin(camera.yaw);
  const depth = dx * cos + dy * sin;

  if (depth <= NEATENSTEIN_TRACER_NEAR_CLIP_EPSILON) {
    return null;
  }

  const lateral = -dx * sin + dy * cos;
  const horizonY = canvasHeight * NEATENSTEIN_FLOOR_HORIZON_RATIO;
  const cameraHeight =
    canvasHeight * NEATENSTEIN_FLOOR_CAMERA_HEIGHT_SCREEN_RATIO;
  const screenY = horizonY + cameraHeight / depth;
  const planeScale = Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
  const screenX =
    canvasWidth / 2 + (lateral / (depth * planeScale)) * (canvasWidth / 2);

  if (!Number.isFinite(screenX) || !Number.isFinite(screenY)) {
    return null;
  }

  return { x: screenX, y: screenY };
}

/**
 * Draw active neon beam tracers as glowing perspective lines.
 *
 * @param context - Worker-tier 2D canvas context.
 * @param tracers - Active tracer list.
 * @param camera - Camera position and yaw.
 * @param canvasWidth - Canvas width.
 * @param canvasHeight - Canvas height.
 */
function drawNeatensteinTracers(
  context: OffscreenCanvasRenderingContext2D,
  tracers: readonly TracerState[],
  camera: TracerCamera,
  canvasWidth: number,
  canvasHeight: number,
): void {
  if (tracers.length === 0) {
    return;
  }

  const savedComposite = context.globalCompositeOperation;
  context.globalCompositeOperation = 'lighter';

  for (const tracer of tracers) {
    const originScreen = projectTracerPoint(
      tracer.origin,
      camera,
      canvasWidth,
      canvasHeight,
    );
    const hitScreen = projectTracerPoint(
      tracer.hit,
      camera,
      canvasWidth,
      canvasHeight,
    );

    if (!originScreen || !hitScreen) {
      continue;
    }

    context.shadowColor = NEATENSTEIN_TRACER_GLOW_COLOR;
    context.shadowBlur = NEATENSTEIN_TRACER_GLOW_BLUR_RADIUS;
    context.strokeStyle = NEATENSTEIN_TRACER_COLOR;
    context.lineWidth = NEATENSTEIN_TRACER_LINE_WIDTH;
    context.beginPath();
    context.moveTo(originScreen.x, originScreen.y);
    context.lineTo(hitScreen.x, hitScreen.y);
    context.stroke();
  }

  context.shadowBlur = 0;
  context.globalCompositeOperation = savedComposite;
}

/**
 * Draw active wall-impact neon spots in the worker tier.
 *
 * @param context - Worker-tier 2D canvas context.
 * @param impacts - Active wall-impact list.
 * @param zBuffer - Per-column depth buffer.
 * @param camera - Camera position and yaw.
 * @param canvasWidth - Canvas width.
 * @param canvasHeight - Canvas height.
 */
function drawImpactSpots(
  context: OffscreenCanvasRenderingContext2D,
  impacts: readonly ImpactSpot[],
  zBuffer: Float32Array,
  camera: TracerCamera,
  canvasWidth: number,
  canvasHeight: number,
): void {
  if (impacts.length === 0) {
    return;
  }

  const savedComposite = context.globalCompositeOperation;
  context.globalCompositeOperation = 'lighter';

  const dirX = Math.cos(camera.yaw);
  const dirY = Math.sin(camera.yaw);
  const planeScale = Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);

  for (const impact of impacts) {
    const relX = impact.position.x - camera.x;
    const relY = impact.position.y - camera.y;
    const perpDist = relX * dirX + relY * dirY;

    if (!Number.isFinite(perpDist) || perpDist <= 0) {
      continue;
    }

    const lateral = -relX * dirY + relY * dirX;
    const screenX =
      canvasWidth / 2 + (lateral / (perpDist * planeScale)) * (canvasWidth / 2);

    if (!Number.isFinite(screenX)) {
      continue;
    }

    const screenColumn = (screenX / canvasWidth) * zBuffer.length;

    if (!depthTestPulse({ screenColumn, distance: perpDist }, zBuffer)) {
      continue;
    }

    const alpha = clamp(
      impact.lifetimeMs / NEATENSTEIN_IMPACT_SPOT_LIFETIME_MS,
      0,
      1,
    );

    const radius = Math.max(1, NEATENSTEIN_IMPACT_SPOT_RADIUS_PX / perpDist);

    context.shadowColor = NEATENSTEIN_IMPACT_SPOT_GLOW_COLOR;
    context.shadowBlur = NEATENSTEIN_IMPACT_SPOT_GLOW_BLUR_PX;
    context.fillStyle = NEATENSTEIN_IMPACT_SPOT_COLOR;
    context.globalAlpha = alpha;
    context.beginPath();
    context.arc(screenX, canvasHeight / 2, radius, 0, Math.PI * 2);
    context.fill();
  }

  context.globalAlpha = 1;
  context.shadowBlur = 0;
  context.globalCompositeOperation = savedComposite;
}

/**
 * Merge a new input snapshot into the pending snapshot.
 *
 * Movement and look use the latest values. One-shot actions are latched so a
 * short fire/dash press cannot be overwritten before the next `simState` tick.
 *
 * @param previous - Existing pending input, if any.
 * @param next - Newly received input.
 * @returns Merged pending input.
 */
function mergePendingTickInput(
  previous: GameTickInputSnapshot | null,
  next: GameTickInputSnapshot,
): GameTickInputSnapshot {
  if (previous === null) {
    return next;
  }

  return {
    move: next.move,
    lookDelta: next.lookDelta,
    fire: previous.fire || next.fire,
    dash: previous.dash || next.dash,
  };
}

/**
 * Normalize a host input message into the snapshot shape expected by game tick.
 *
 * @param raw - Value attached to the input message by the host.
 * @returns Normalized tick input snapshot.
 */
function inputMessageToTickInput(raw: unknown): GameTickInputSnapshot {
  if (!raw || typeof raw !== 'object') {
    return { move: { x: 0, y: 0 }, lookDelta: 0, fire: false, dash: false };
  }

  const input = raw as Record<string, unknown>;

  const yawDelta =
    typeof input.yawDelta === 'number' && Number.isFinite(input.yawDelta)
      ? input.yawDelta
      : 0;

  let lookDelta = yawDelta;
  if (input.look && typeof input.look === 'object') {
    const lookRecord = input.look as Record<string, unknown>;
    const lookYaw = lookRecord.yawDelta;

    if (typeof lookYaw === 'number' && Number.isFinite(lookYaw)) {
      lookDelta = lookYaw;
    }
  }

  let moveX = 0;
  let moveY = 0;

  if (input.movement && typeof input.movement === 'object') {
    const movement = input.movement as Record<string, unknown>;

    if (movement.forward === true) moveY += 1;
    if (movement.backward === true) moveY -= 1;
    if (movement.left === true) moveX -= 1;
    if (movement.right === true) moveX += 1;
  } else if (input.move && typeof input.move === 'object') {
    const move = input.move as Record<string, unknown>;

    if (typeof move.x === 'number' && Number.isFinite(move.x)) {
      moveX = move.x;
    }

    if (typeof move.y === 'number' && Number.isFinite(move.y)) {
      moveY = move.y;
    }
  }

  return {
    move: { x: moveX, y: moveY },
    lookDelta,
    fire: input.fire === true,
    dash: input.dash === true,
  };
}

self.onmessage = (event: MessageEvent) => {
  const data = event.data;

  if (!data || typeof data !== 'object') {
    return;
  }

  if (data.type === 'init') {
    const tier =
      data.tier === 'worker' || data.tier === 'cpu' || data.tier === 'gpu'
        ? (data.tier as DisplayTier)
        : null;

    if (tier === null) {
      return;
    }

    currentTier = tier;
    latestState = null;
    activePulses = [];
    pendingTickInput = null;
    workerZBuffer = null;
    workerContext = null;

    if (data.canvas) {
      workerCanvas = data.canvas as OffscreenCanvas;
    } else {
      workerCanvas = null;
    }

    const seed =
      typeof data.mapSeed === 'number' && Number.isFinite(data.mapSeed)
        ? data.mapSeed
        : NEATENSTEIN_DEFAULT_SEED;

    // Build the deterministic map once, then share it between raycasting and
    // collision systems.
    wallMap = buildNeatensteinMap(seed);
    collisionMap = createCollisionMap(wallMap, NEATENSTEIN_MAP_SIZE);
    gameState = createGameState({ seed });

    self.postMessage({
      type: 'initialized',
      tier,
      version: data.version ?? NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
    });

    return;
  }

  if (data.type === 'simState') {
    latestState = data.state as NeatensteinRenderState;

    if (gameState && collisionMap) {
      const tickInput = pendingTickInput ?? {
        move: { x: 0, y: 0 },
        lookDelta: 0,
        fire: false,
        dash: false,
      };

      gameState = gameTick(
        gameState,
        tickInput,
        collisionMap,
        NEATENSTEIN_FIXED_TIMESTEP_MS,
      );

      pendingTickInput = null;
    }

    buildAndPostFrame();
    return;
  }

  if (data.type === NEATENSTEIN_INPUT_MESSAGE_TYPE) {
    const nextInput = inputMessageToTickInput(data.input);
    pendingTickInput = mergePendingTickInput(pendingTickInput, nextInput);
  }
};
