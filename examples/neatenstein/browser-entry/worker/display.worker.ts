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
 * All tiers are driven by `simState` messages from the host. The worker tier
 * rasterizes to the transferred {@link OffscreenCanvas}; the CPU/GPU tiers ship
 * packed frames back for host-side blitting.
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
  NEATENSTEIN_FLOOR_CAMERA_HEIGHT_SCREEN_RATIO,
  NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
  NEATENSTEIN_FLOOR_FOV_RADIANS,
  NEATENSTEIN_FLOOR_HORIZON_RATIO,
  projectNeatensteinFloorPoint,
  renderNeatensteinFloor,
  type NeatensteinFloorCamera,
} from '../renderer/floor';
import {
  depthTestPulse,
  emitNeatensteinAmbientPulse,
  NEATENSTEIN_PULSE_AMBIENT_LIFETIME_TICKS,
  updateNeatensteinPulses,
  type NeatensteinPulse,
} from '../renderer/pulse';
import {
  buildNeatensteinMap,
  createCollisionMap,
  type CollisionMap,
} from '../renderer/map';
import { castRayDDA } from '../renderer/raycast';
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
let latestState: NeatensteinRenderState | null = null;
let wallGrid: number[][] | null = null;

/** Deterministic game state maintained and advanced by the worker. */
let gameState: GameState | null = null;

/** Collision map built from the init seed; reused for every movement tick. */
let collisionMap: CollisionMap | null = null;

/** Active ambient floor pulses tracked across frames in the worker tier. */
let activePulses: NeatensteinPulse[] = [];

/**
 * Input snapshot captured from the most recent `input` message. Consumed by
 * the next `simState` tick so look, fire, dash, and movement are applied once
 * per rendered frame.
 */
let pendingTickInput: GameTickInputSnapshot | null = null;

/**
 * Background clear color used before each worker-tier frame.
 *
 * Derived from {@link NEATENSTEIN_BACKGROUND_RGB} so the worker tier matches
 * the CPU/GPU fog background exactly.
 */
const NEATENSTEIN_WORKER_CLEAR_COLOR = formatRgb(NEATENSTEIN_BACKGROUND_RGB);

/** RGB of the neon wall color for X-axis-side hits (east/west walls). */
const NEATENSTEIN_WALL_X_SIDE_RGB = { r: 0, g: 191, b: 255 } as const;

/** RGB of the neon wall color for Y-axis-side hits (north/south walls). */
const NEATENSTEIN_WALL_Y_SIDE_RGB = { r: 0, g: 80, b: 180 } as const;

/**
 * Format an RGB triple as a CSS `rgb(...)` string.
 *
 * @param color - Object with integer/float r, g, b components.
 * @returns A CSS color string.
 */
function formatRgb(color: { r: number; g: number; b: number }): string {
  return `rgb(${Math.round(color.r)}, ${Math.round(color.g)}, ${Math.round(color.b)})`;
}

/**
 * Blend a neon wall color toward the background based on perpendicular
 * wall distance.
 *
 * Distant walls fade into the neon void so the worker tier matches the CPU/GPU
 * distance-fog pass.
 *
 * @param wallColor - RGB of the raw wall color.
 * @param perpWallDist - Perpendicular distance from the camera to the wall hit.
 * @returns A CSS color string for the fogged wall stripe.
 */
function applyWallFog(
  wallColor: { r: number; g: number; b: number },
  perpWallDist: number,
): string {
  const fogFactor = Math.min(perpWallDist / NEATENSTEIN_MAX_VIEW_DIST, 1);
  const backgroundColor = NEATENSTEIN_BACKGROUND_RGB;
  return formatRgb({
    r: wallColor.r + (backgroundColor.r - wallColor.r) * fogFactor,
    g: wallColor.g + (backgroundColor.g - wallColor.g) * fogFactor,
    b: wallColor.b + (backgroundColor.b - wallColor.b) * fogFactor,
  });
}

/**
 * Map a tier name to the column count used for packed frames.
 *
 * @param tier - The active renderer tier.
 * @returns The column count for that tier.
 */
function resolveColumnCount(tier: DisplayTier): number {
  switch (tier) {
    case 'gpu':
      return NEATENSTEIN_GPU_COLUMN_COUNT;
    case 'worker':
      return NEATENSTEIN_WORKER_COLUMN_COUNT;
    case 'cpu':
    default:
      return NEATENSTEIN_CPU_COLUMN_COUNT;
  }
}

/**
 * Build a deterministic 2D wall grid from a seed.
 *
 * The flat {@link Uint8Array} returned by {@link buildNeatensteinMap} is
 * expanded into the X-major grid layout expected by {@link castRayDDA}.
 *
 * @param seed - Deterministic map seed.
 * @returns A 2D array where each inner array represents one X column.
 */
function buildWallGrid(seed: number): number[][] {
  const flat = buildNeatensteinMap(seed);
  const side = NEATENSTEIN_MAP_SIZE;
  const grid: number[][] = [];
  for (let x = 0; x < side; x++) {
    grid[x] = [];
    for (let y = 0; y < side; y++) {
      grid[x][y] = flat[y * side + x];
    }
  }
  return grid;
}

/**
 * Clamp a value to a [min, max] range.
 *
 * @param value - Value to clamp.
 * @param min - Inclusive minimum.
 * @param max - Inclusive maximum.
 * @returns The clamped value.
 */
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/** Return type of the shared raycaster used to build every column. */
type RaycastHit = ReturnType<typeof castRayDDA>;

/**
 * Cast a single camera ray for the given column index.
 *
 * @param column - Column index in [0, columnCount).
 * @param columnCount - Total number of columns being rendered.
 * @param cameraPositionX - Camera X position on the map.
 * @param cameraPositionY - Camera Y position on the map.
 * @param cameraDirectionX - X component of the camera direction vector.
 * @param cameraDirectionY - Y component of the camera direction vector.
 * @param cameraPlaneX - X component of the camera plane vector.
 * @param cameraPlaneY - Y component of the camera plane vector.
 * @returns The raycast hit for this column.
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
  const cameraPlaneOffset = (2 * column) / columnCount - 1;
  const rayDirectionX = cameraDirectionX + cameraPlaneX * cameraPlaneOffset;
  const rayDirectionY = cameraDirectionY + cameraPlaneY * cameraPlaneOffset;

  return castRayDDA(
    wallGrid!,
    NEATENSTEIN_MAP_SIZE,
    NEATENSTEIN_MAP_SIZE,
    cameraPositionX,
    cameraPositionY,
    rayDirectionX,
    rayDirectionY,
  );
}

/**
 * Build a packed frame from the latest state and post it back to the host.
 *
 * For the `worker` tier this renders the floor and walls directly to the
 * transferred {@link OffscreenCanvas}. For the `cpu` and `gpu` tiers it fills
 * the packed frame's typed arrays with raycast results so the host can blit
 * the walls later.
 */
function buildAndPostFrame(): void {
  if (!latestState || !currentTier || !wallGrid || !gameState) {
    return;
  }

  const columnCount = resolveColumnCount(currentTier);
  const canvasWidth = latestState.canvasWidth;
  const canvasHeight = latestState.canvasHeight;
  const cameraPositionX = gameState.player.position.x;
  const cameraPositionY = gameState.player.position.y;
  const cameraYaw = gameState.player.angleRad;

  const cameraDirectionX = Math.cos(cameraYaw);
  const cameraDirectionY = Math.sin(cameraYaw);
  const planeScale = Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
  const cameraPlaneX = -cameraDirectionY * planeScale;
  const cameraPlaneY = cameraDirectionX * planeScale;

  if (currentTier === 'worker') {
    const context = workerCanvas?.getContext('2d');
    if (!context) {
      return;
    }

    context.fillStyle = NEATENSTEIN_WORKER_CLEAR_COLOR;
    context.fillRect(0, 0, canvasWidth, canvasHeight);
    renderNeatensteinFloor(context, {
      x: cameraPositionX,
      y: cameraPositionY,
      yaw: cameraYaw,
    });

    const stripeWidth = canvasWidth / columnCount;
    const wallFocalLength =
      canvasWidth / 2 / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
    const zBuffer = new Float32Array(columnCount);
    for (let column = 0; column < columnCount; column++) {
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
      context.fillRect(
        column * stripeWidth,
        drawStart,
        stripeWidth,
        drawEnd - drawStart,
      );
    }

    activePulses = updateNeatensteinPulses(activePulses, latestState.simTick);
    const newPulse = emitNeatensteinAmbientPulse(
      latestState.simTick,
      gameState.seed,
    );
    if (newPulse && activePulses.length < NEATENSTEIN_PULSE_MAX_CONCURRENT) {
      activePulses.push(newPulse);
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

  const frame = buildNeatensteinRenderFrame(latestState, columnCount);
  for (let column = 0; column < columnCount; column++) {
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
 * CSS color string for the yellow ambient floor pulse.
 */
const NEATENSTEIN_PULSE_COLOR = '#fff14a';

/**
 * Glow color string for the yellow ambient floor pulse.
 */
const NEATENSTEIN_PULSE_GLOW_COLOR = 'rgba(255, 241, 74, 0.42)';

/**
 * Draw the active ambient floor pulses that pass the z-buffer depth test.
 *
 * Each pulse is projected from world space to screen space using the same
 * floor projection as the grid lines, then rendered as a small yellow dot.
 * Pulses behind walls or behind the camera are discarded.
 *
 * @param context - Worker-tier 2D canvas context.
 * @param pulses - Active pulse list for this frame.
 * @param zBuffer - Per-column depth buffer from the wall pass.
 * @param camera - Current camera look state for world-to-screen projection.
 * @param canvasWidth - Width of the canvas in pixels.
 * @param canvasHeight - Height of the canvas in pixels.
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

    const projected = projectNeatensteinFloorPoint(
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

/**
 * Camera-relative transform of a world point for tracer projection.
 */
interface TracerCamera {
  /** Camera world X position. */
  x: number;
  /** Camera world Y position. */
  y: number;
  /** Camera yaw in radians. */
  yaw: number;
}

/**
 * Project a world-space point into screen coordinates using the same
 * forced-perspective mapping used by the floor reticule.
 *
 * @param point - World-space point to project.
 * @param camera - Camera position and yaw.
 * @param canvasWidth - Canvas width in pixels.
 * @param canvasHeight - Canvas height in pixels.
 * @returns Screen coordinates, or `null` when the point is behind the camera.
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
  return { x: screenX, y: screenY };
}

/**
 * Draw active neon beam tracers as glowing perspective lines.
 *
 * Tracers that are fully or partially behind the camera are skipped so they
 * do not project to nonsensical screen coordinates.
 *
 * @param context - Worker-tier 2D canvas context.
 * @param tracers - Active tracer list from the game state.
 * @param camera - Camera position and yaw.
 * @param canvasWidth - Canvas width in pixels.
 * @param canvasHeight - Canvas height in pixels.
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
 * Each spot is projected from its world-space wall hit to the center of the
 * wall stripe at the matching screen column using the current camera position,
 * then faded out as its lifetime expires. The spot radius and screen position
 * are recomputed every frame from the current perpendicular distance so the
 * marker scales dynamically as the player moves closer or farther away. Spots
 * that fall behind another wall according to the z-buffer are skipped so the
 * marker only appears on the visible wall face.
 *
 * @param context - Worker-tier 2D canvas context.
 * @param impacts - Active wall-impact list from the game state.
 * @param zBuffer - Per-column depth buffer from the wall pass.
 * @param camera - Camera position and yaw.
 * @param canvasWidth - Canvas width in pixels.
 * @param canvasHeight - Canvas height in pixels.
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
    if (perpDist <= 0) {
      continue;
    }

    const lateral = -relX * dirY + relY * dirX;
    const screenX =
      canvasWidth / 2 + (lateral / (perpDist * planeScale)) * (canvasWidth / 2);
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
    if (data.canvas) {
      workerCanvas = data.canvas as OffscreenCanvas;
    }

    const seed =
      typeof data.mapSeed === 'number' && Number.isFinite(data.mapSeed)
        ? data.mapSeed
        : NEATENSTEIN_DEFAULT_SEED;
    wallGrid = buildWallGrid(seed);
    activePulses = [];
    collisionMap = createCollisionMap(
      buildNeatensteinMap(seed),
      NEATENSTEIN_MAP_SIZE,
    );
    gameState = createGameState({ seed });

    self.postMessage({
      type: 'initialized',
      tier,
      version: data.version ?? NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
    });
  } else if (data.type === 'simState') {
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
  } else if (data.type === NEATENSTEIN_INPUT_MESSAGE_TYPE) {
    pendingTickInput = inputMessageToTickInput(data.input);
  }
};

/**
 * Normalize a host input message into the snapshot shape expected by the game
 * tick.
 *
 * The bridge forwards either legacy `{ yawDelta }` messages from older tests or
 * full {@link InputSnapshot} objects from the host input router. Missing fields
 * default to zero/no-action so a partial message never crashes the tick.
 *
 * @param raw - Value attached to the input message by the host.
 * @returns A normalized tick input snapshot.
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
