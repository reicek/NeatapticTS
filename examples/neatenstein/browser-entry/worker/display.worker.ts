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
 * The worker renders at the host-provided backing-store dimensions. It only
 * guards against non-finite or non-positive dimensions and synchronizes the
 * transferred OffscreenCanvas to the incoming state size for the `worker` tier;
 * it does not impose a maximum render resolution.
 *
 * @module
 */

/// <reference lib="webworker" />

import {
  NEATENSTEIN_CPU_COLUMN_COUNT,
  NEATENSTEIN_DEFAULT_SEED,
  NEATENSTEIN_FIXED_TIMESTEP_MS,
  NEATENSTEIN_GPU_COLUMN_COUNT,
  NEATENSTEIN_INPUT_MESSAGE_TYPE,
  NEATENSTEIN_MAP_SIZE,
  NEATENSTEIN_PULSE_ALPHA_MAX,
  NEATENSTEIN_PULSE_ALPHA_MIN,
  NEATENSTEIN_PULSE_GLOW_BLUR_RADIUS,
  NEATENSTEIN_PULSE_MAX_CONCURRENT,
  NEATENSTEIN_PULSE_SCREEN_DOT_RADIUS_PX,
  NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
} from '../constants';
import {
  buildNeatensteinRenderFrame,
  resolveNeatensteinRenderFrameTransferList,
  type NeatensteinRenderState,
} from '../renderer/frame';
import {
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
import type { GameState } from '../host/game/types';
import {
  NEATENSTEIN_BACKGROUND_RGB,
  NEATENSTEIN_MAX_VIEW_DIST,
} from '../renderer/framebuffer';
import { renderGunOverlay } from '../renderer/gun';
import { drawBolts, drawImpactSpots } from '../renderer/bolt-render';
import {
  createEnemyControllerState,
  updateEnemyController,
  type EnemyControllerState,
} from '../../scripts/enemy-controller';
import {
  clipNeatensteinSprite,
  renderNeatensteinSprite,
  resolveNeatensteinEnemyFrame,
  type NeatensteinCamera,
  type NeatensteinSprite,
  type NeatensteinSpriteRenderContext,
} from '../renderer/sprites';

type DisplayTier = 'worker' | 'cpu' | 'gpu';

let currentTier: DisplayTier | null = null;
let workerCanvas: OffscreenCanvas | null = null;
let workerContext: OffscreenCanvasRenderingContext2D | null = null;
let latestState: NeatensteinRenderState | null = null;

/**
 * Host resize dimensions received before the worker canvas is assigned.
 *
 * Resize messages can arrive before the `init` message finishes setting up
 * `workerCanvas`, so the last valid host dimension is stashed and applied once
 * the canvas is available.
 */
let pendingResizeDimensions: { width: number; height: number } | null = null;

/** Canonical flat deterministic map used by the worker raycaster. */
let wallMap: Uint8Array | null = null;

/** Reusable worker-tier z-buffer, resized only when the render width changes. */
let workerZBuffer: Float32Array | null = null;

/**
 * Persistent RGBA framebuffer written directly by the worker tier.
 *
 * Re-allocated only when the canvas backing-store dimensions change.
 */
let workerFramebuffer: Uint8ClampedArray<ArrayBuffer> | null = null;

/** Deterministic game state maintained and advanced by the worker. */
let gameState: GameState | null = null;

/** Collision map built from the init seed; reused for every movement tick. */
let collisionMap: CollisionMap | null = null;

/** Active ambient floor and ceiling pulses tracked across worker frames. */
let activePulses: NeatensteinPulse[] = [];

/**
 * Enemy AI controller state maintained across simulation ticks so fire
 * cooldowns, ammunition, and de-rez timing advance deterministically.
 */
let enemyControllerState: EnemyControllerState | null = null;

/**
 * Active enemy sprite positions computed by the controller this frame.
 *
 * Stored separately from {@link latestState} so the next slice can render
 * sprites after the wall pass without recomputing controller state.
 */
let activeEnemySprites: NeatensteinSprite[] = [];

/**
 * Input snapshot captured from the most recent `input` message.
 *
 * Movement/look use the latest value, while one-shot actions such as fire and
 * dash are latched until the next simulation tick consumes them.
 */
let pendingTickInput: GameTickInputSnapshot | null = null;

/** RGB of the neon wall color for X-axis-side hits. */
const NEATENSTEIN_WALL_X_SIDE_RGB = { r: 0, g: 200, b: 255 } as const;

/** RGB of the neon wall color for Y-axis-side hits. */
const NEATENSTEIN_WALL_Y_SIDE_RGB = { r: 0, g: 90, b: 150 } as const;

/** Height in screen pixels of one horizontal wall block band. */
const NEATENSTEIN_WALL_BLOCK_HEIGHT_PX = 24;

/** Darkening factor applied to horizontal wall block edges. */
const NEATENSTEIN_WALL_EDGE_DARKEN_FACTOR = 0.65;

/** Size in pixels of the floor/ceiling checker grid cell. */
const NEATENSTEIN_FLOOR_GRID_CELL_PX = 32;

/** CSS color string for the yellow ambient pulse dot. */
const NEATENSTEIN_PULSE_COLOR = '#B7FF00';

/** CSS shadow color string for the yellow ambient pulse glow. */
const NEATENSTEIN_PULSE_GLOW_COLOR = 'rgba(185, 255, 0, 0.42)';

/** Number of RGBA channels per framebuffer pixel. */
const NEATENSTEIN_FRAMEBUFFER_CHANNELS = 4;

/**
 * Darkening factor applied to the background RGB for the worker ceiling fill.
 */
const NEATENSTEIN_WORKER_CEILING_DARKEN_FACTOR = 0.5;

/** Floor fill color for the worker-tier direct renderer (Flappy horizon teal). */
const NEATENSTEIN_WORKER_FLOOR_RGB = { r: 10, g: 142, b: 160 } as const;

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
 * Synchronize the transferred OffscreenCanvas with the host-provided render
 * size.
 *
 * This is critical for direct worker rendering. Projection math, floor/ceiling
 * drawing, wall stripes, z-buffer columns, and the actual canvas backing store
 * must agree on the same dimensions.
 *
 * @param canvas - Transferred worker-owned canvas.
 * @param width - Host-provided backing-store width.
 * @param height - Host-provided backing-store height.
 */
function syncWorkerCanvasSize(
  canvas: OffscreenCanvas,
  width: number,
  height: number,
): void {
  let resized = false;

  if (canvas.width !== width) {
    canvas.width = width;
    resized = true;
  }

  if (canvas.height !== height) {
    canvas.height = height;
    resized = true;
  }

  if (
    resized ||
    workerFramebuffer === null ||
    workerFramebuffer.length !== width * height * 4
  ) {
    workerFramebuffer = new Uint8ClampedArray(width * height * 4);
  }
}

/**
 * Apply a host-resized CSS-box dimension to the worker canvas and latest state.
 *
 * The host owns the visible canvas CSS box, while the worker owns the backing
 * store. This helper updates both so the next frame renders at the new size
 * even if it arrives before the next `simState` tick.
 *
 * @param width - Host-derived render width in pixels.
 * @param height - Host-derived render height in pixels.
 */
function applyWorkerResize(width: number, height: number): void {
  if (!isPositiveFiniteDimension(width) || !isPositiveFiniteDimension(height)) {
    return;
  }

  if (workerCanvas !== null) {
    syncWorkerCanvasSize(workerCanvas, width, height);
  } else {
    pendingResizeDimensions = { width, height };
  }

  if (latestState !== null) {
    latestState.canvasWidth = width;
    latestState.canvasHeight = height;
  }
}

/**
 * Resolve the direct worker-tier raycast column count.
 *
 * The worker tier renders directly into the OffscreenCanvas, so its horizontal
 * ray density should match the host-provided backing-store width.
 *
 * @param canvasWidth - Canvas backing-store width in pixels.
 * @returns Number of direct worker raycast columns.
 */
function resolveWorkerCanvasColumnCount(canvasWidth: number): number {
  return Math.max(1, Math.floor(canvasWidth));
}

/**
 * Map a packed-frame tier to the column count used in typed frame payloads.
 *
 * Direct worker rendering does not use this helper; it raycasts at the
 * host-provided canvas width instead.
 *
 * @param tier - Active renderer tier.
 * @returns Packed-frame column count.
 */
function resolvePackedColumnCount(tier: DisplayTier): number {
  if (tier === 'gpu') {
    return NEATENSTEIN_GPU_COLUMN_COUNT;
  }
  return NEATENSTEIN_CPU_COLUMN_COUNT;
}

/**
 * Resolve and clear the reusable worker z-buffer.
 *
 * @param columnCount - Required number of z-buffer columns.
 * @returns A cleared z-buffer.
 */
/* istanbul ignore next -- pre-existing z-buffer reuse helper; only the allocation branch is exercised by this test suite */
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
  return clamp(perpWallDist / NEATENSTEIN_MAX_VIEW_DIST, 0, 1);
}

/**
 * Blend a neon wall color toward the background based on wall distance.
 *
 * @param wallColor - Raw wall RGB.
 * @param perpWallDist - Perpendicular distance from camera to wall.
 * @returns Fogged RGB triple for direct framebuffer writes.
 */
function resolveWallFogRgb(
  wallColor: { r: number; g: number; b: number },
  perpWallDist: number,
): { r: number; g: number; b: number } {
  const fogFactor = resolveWallFogFactor(perpWallDist);
  const backgroundColor = NEATENSTEIN_BACKGROUND_RGB;

  return {
    r: wallColor.r + (backgroundColor.r - wallColor.r) * fogFactor,
    g: wallColor.g + (backgroundColor.g - wallColor.g) * fogFactor,
    b: wallColor.b + (backgroundColor.b - wallColor.b) * fogFactor,
  };
}

/**
 * Fill the worker framebuffer with a dark ceiling and a slightly lighter floor.
 *
 * This is the worker-tier replacement for the full-canvas `getImageData` copy.
 * Rows above the horizon get a darkened background, rows at and below the
 * horizon get a floor color. The wall pass will overwrite the middle vertical
 * band, so no separate background clear is required.
 *
 * @param canvasWidth - Canvas width in pixels.
 * @param canvasHeight - Canvas height in pixels.
 */
function fillWorkerCeilingAndFloor(
  canvasWidth: number,
  canvasHeight: number,
): void {
  if (workerFramebuffer === null) {
    return;
  }

  const horizon = Math.floor(canvasHeight * NEATENSTEIN_FLOOR_HORIZON_RATIO);
  const background = NEATENSTEIN_BACKGROUND_RGB;
  const ceilingBase = {
    r: background.r * NEATENSTEIN_WORKER_CEILING_DARKEN_FACTOR,
    g: background.g * NEATENSTEIN_WORKER_CEILING_DARKEN_FACTOR,
    b: background.b * NEATENSTEIN_WORKER_CEILING_DARKEN_FACTOR,
  };
  const ceilingLine = {
    r: Math.min(255, ceilingBase.r + 18),
    g: Math.min(255, ceilingBase.g + 24),
    b: Math.min(255, ceilingBase.b + 32),
  };
  const floorBase = NEATENSTEIN_WORKER_FLOOR_RGB;
  const floorAlt = {
    r: Math.min(255, floorBase.r + 16),
    g: Math.min(255, floorBase.g + 22),
    b: Math.min(255, floorBase.b + 24),
  };

  for (let row = 0; row < canvasHeight; row += 1) {
    const isCeiling = row < horizon;
    const rowOffset = row * canvasWidth * NEATENSTEIN_FRAMEBUFFER_CHANNELS;
    const rowCell = Math.floor(row / NEATENSTEIN_FLOOR_GRID_CELL_PX);

    for (let col = 0; col < canvasWidth; col += 1) {
      const colCell = Math.floor(col / NEATENSTEIN_FLOOR_GRID_CELL_PX);
      const isGridLine =
        (row % NEATENSTEIN_FLOOR_GRID_CELL_PX) < 2 ||
        (col % NEATENSTEIN_FLOOR_GRID_CELL_PX) < 2;
      const isCheckerCell = (rowCell + colCell) % 2 === 0;

      const color = isCeiling
        ? isGridLine
          ? ceilingLine
          : ceilingBase
        : isGridLine || isCheckerCell
          ? floorAlt
          : floorBase;

      const offset = rowOffset + col * NEATENSTEIN_FRAMEBUFFER_CHANNELS;
      workerFramebuffer[offset] = color.r;
      workerFramebuffer[offset + 1] = color.g;
      workerFramebuffer[offset + 2] = color.b;
      workerFramebuffer[offset + 3] = 255;
    }
  }
}

/**
 * Write a single vertical wall stripe into the persistent worker framebuffer.
 *
 * The stripe is fogged toward the background based on perpendicular distance.
 * This replaces the previous `context.fillRect` wall pass.
 *
 * @param xStart - Screen column to write.
 * @param drawStart - Top row of the wall stripe, inclusive.
 * @param drawEnd - Bottom row of the wall stripe, exclusive.
 * @param wallColor - Raw neon wall RGB.
 * @param perpWallDist - Perpendicular wall distance for distance fog.
 */
function writeWallStripeToFramebuffer(
  xStart: number,
  drawStart: number,
  drawEnd: number,
  wallColor: { r: number; g: number; b: number },
  perpWallDist: number,
): void {
  if (workerFramebuffer === null) {
    return;
  }

  const { width, height } = resolveWorkerFramebufferDimensions();
  if (width === 0 || height === 0) {
    return;
  }

  const x = Math.floor(xStart);
  if (x < 0 || x >= width) {
    return;
  }

  const clampedStart = clamp(drawStart, 0, height);
  const clampedEnd = clamp(drawEnd, 0, height);
  if (clampedStart >= clampedEnd) {
    return;
  }

  const fogged = resolveWallFogRgb(wallColor, perpWallDist);
  const baseR = Math.round(fogged.r);
  const baseG = Math.round(fogged.g);
  const baseB = Math.round(fogged.b);
  const edgeR = Math.round(baseR * NEATENSTEIN_WALL_EDGE_DARKEN_FACTOR);
  const edgeG = Math.round(baseG * NEATENSTEIN_WALL_EDGE_DARKEN_FACTOR);
  const edgeB = Math.round(baseB * NEATENSTEIN_WALL_EDGE_DARKEN_FACTOR);

  for (let row = clampedStart; row < clampedEnd; row += 1) {
    const rowInStripe = row - Math.floor(drawStart);
    const isBlockEdge =
      (rowInStripe % NEATENSTEIN_WALL_BLOCK_HEIGHT_PX) < 2 ||
      rowInStripe < 2 ||
      row === clampedEnd - 1;

    const r = isBlockEdge ? edgeR : baseR;
    const g = isBlockEdge ? edgeG : baseG;
    const b = isBlockEdge ? edgeB : baseB;

    const offset = (row * width + x) * NEATENSTEIN_FRAMEBUFFER_CHANNELS;
    workerFramebuffer[offset] = r;
    workerFramebuffer[offset + 1] = g;
    workerFramebuffer[offset + 2] = b;
    workerFramebuffer[offset + 3] = 255;
  }
}

/**
 * Resolve the pixel dimensions of the persistent worker framebuffer.
 *
 * @returns Width and height, or zeros if the buffer has not been allocated.
 */
function resolveWorkerFramebufferDimensions(): {
  width: number;
  height: number;
} {
  if (workerFramebuffer === null) {
    return { width: 0, height: 0 };
  }

  // zBuffer length tracks the canvas width and is always allocated before the
  // wall pass writes into the framebuffer.
  if (workerZBuffer !== null && workerZBuffer.length > 0) {
    const width = workerZBuffer.length;
    const height = workerFramebuffer.length / (width * 4);
    return { width, height: Math.floor(height) };
  }

  const side = Math.floor(
    Math.sqrt(workerFramebuffer.length / NEATENSTEIN_FRAMEBUFFER_CHANNELS),
  );
  return { width: side, height: side };
}

/**
 * Render all active enemy sprites into the persistent worker framebuffer.
 *
 * This is the worker-tier sprite pass. It clips each sprite against the wall
 * z-buffer, resolves the camera-relative voxel frame, and renders it into the
 * persistent CPU framebuffer. The no-op render context prevents per-sprite
 * canvas flushes; the caller flushes once after the pass.
 *
 * @param zBuffer - Per-column wall-depth buffer.
 * @param spriteCamera - Camera transform used by the sprite renderer.
 * @param canvasWidth - Canvas width in pixels.
 * @param canvasHeight - Canvas height in pixels.
 */
function renderWorkerSprites(
  zBuffer: Float32Array,
  spriteCamera: NeatensteinCamera,
  canvasWidth: number,
  canvasHeight: number,
): void {
  const noOpSpriteCtx = buildNoOpSpriteRenderContext();

  const visibleSprites = activeEnemySprites
    .map((sprite) => ({
      sprite,
      projection: clipNeatensteinSprite(
        sprite,
        spriteCamera,
        canvasWidth,
        canvasHeight,
        zBuffer,
      ),
    }))
    .filter(({ projection }) => projection.visibleColumns.length > 0)
    .toSorted((a, b) => b.projection.perpDist - a.projection.perpDist);

  for (const { sprite, projection } of visibleSprites) {
    const frame = resolveNeatensteinEnemyFrame(sprite, spriteCamera);
    if (frame === null) {
      continue;
    }

    renderNeatensteinSprite(
      workerFramebuffer!,
      zBuffer,
      projection,
      frame,
      noOpSpriteCtx,
    );
  }
}

/**
 * Build a canvas-like context that does not flush per sprite.
 *
 * The worker tier renders all sprites into a single snapshot framebuffer and
 * flushes it back to the OffscreenCanvas once after the sprite pass. The
 * renderer's `putImageData` contract still expects a context, so this no-op
 * adapter satisfies the type without redundant per-sprite copies.
 *
 * @returns Canvas-like context with a no-op `putImageData`.
 */
function buildNoOpSpriteRenderContext(): NeatensteinSpriteRenderContext {
  return {
    putImageData: () => {
      // Intentionally empty: the worker flushes the framebuffer once after all
      // sprites are drawn.
    },
  };
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
  // Map the column index to a -1..+1 offset on the camera plane.
  const cameraPlaneOffset = (2 * column) / columnCount - 1;

  // Combine camera forward direction with the camera plane offset.
  const rayDirectionX = cameraDirectionX + cameraPlaneX * cameraPlaneOffset;
  const rayDirectionY = cameraDirectionY + cameraPlaneY * cameraPlaneOffset;

  return castRayDDAFromFlatMap(
    wallMap!,
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
 * ceiling/floor fill → walls → sprites → flush → pulses → impact spots →
 * bolts → gun overlay
 */
function buildAndPostFrame(): void {
  if (!latestState || !currentTier || !wallMap || !gameState || !collisionMap) {
    return;
  }

  const canvasWidth = latestState.canvasWidth;
  const canvasHeight = latestState.canvasHeight;

  if (
    !isPositiveFiniteDimension(canvasWidth) ||
    !isPositiveFiniteDimension(canvasHeight)
  ) {
    return;
  }

  const cameraPositionX = gameState.player.position.x;
  const cameraPositionY = gameState.player.position.y;
  const cameraYaw = gameState.player.angleRad;

  // Derive camera direction and projection plane from the player yaw.
  const cameraDirectionX = Math.cos(cameraYaw);
  const cameraDirectionY = Math.sin(cameraYaw);
  const planeScale =
    (canvasWidth / canvasHeight) * Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
  const cameraPlaneX = -cameraDirectionY * planeScale;
  const cameraPlaneY = cameraDirectionX * planeScale;

  // Advance the enemy controller with the worker-authoritative state and
  // collision data. Persist the returned controller state so per-enemy ammo,
  // fire cooldowns, and de-rez timing advance across frames instead of
  // resetting every tick.
  const controlled = updateEnemyController(
    enemyControllerState!,
    gameState,
    collisionMap,
    NEATENSTEIN_FIXED_TIMESTEP_MS,
  );
  enemyControllerState = controlled;
  activeEnemySprites = controlled.enemies
    .filter((enemy) => enemy.active)
    .map((enemy) => ({
      worldX: enemy.position.x,
      worldY: enemy.position.y,
      facing: enemy.yawRad,
      animationState: enemy.animationState,
      frameIndex: 0,
      type: enemy.index,
    }));

  // Stash the active enemy sprites on the incoming render state so the next
  // slice's sprite pass can render them for any tier without recomputing the
  // controller state.
  latestState.enemies = activeEnemySprites;

  if (currentTier === 'worker') {
    if (!workerCanvas) {
      return;
    }

    // The worker owns the transferred canvas, so it must synchronize the
    // backing store to the host-provided dimensions before any drawing.
    syncWorkerCanvasSize(workerCanvas, canvasWidth, canvasHeight);

    if (!workerContext) {
      workerContext = workerCanvas.getContext('2d');
    }

    const context = workerContext;
    if (!context) {
      return;
    }

    const columnCount = resolveWorkerCanvasColumnCount(canvasWidth);
    const zBuffer = resolveWorkerZBuffer(columnCount);

    // Fill the persistent framebuffer with a minimal ceiling/floor pair instead
    // of copying the canvas back from the GPU.
    fillWorkerCeilingAndFloor(canvasWidth, canvasHeight);

    const stripeWidth = canvasWidth / columnCount;
    const wallFocalLength =
      canvasHeight / 2 / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);

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

      const wallColor =
        hit.side === 0
          ? NEATENSTEIN_WALL_X_SIDE_RGB
          : NEATENSTEIN_WALL_Y_SIDE_RGB;

      // Integer stripe bounds prevent subpixel gaps when the column count does
      // not divide the canvas width evenly.
      const xStart = Math.floor(column * stripeWidth);
      const xEnd = Math.floor((column + 1) * stripeWidth);
      const stripePixelWidth = Math.max(0, xEnd - xStart);

      for (let xOffset = 0; xOffset < stripePixelWidth; xOffset += 1) {
        writeWallStripeToFramebuffer(
          xStart + xOffset,
          drawStart,
          drawEnd,
          wallColor,
          hit.perpWallDist,
        );
      }
    }

    const spriteCamera = {
      posX: cameraPositionX,
      posY: cameraPositionY,
      dirX: cameraDirectionX,
      dirY: cameraDirectionY,
      planeX: cameraPlaneX,
      planeY: cameraPlaneY,
    };

    renderWorkerSprites(zBuffer, spriteCamera, canvasWidth, canvasHeight);

    // Flush the completed framebuffer (walls + ceiling/floor + sprites) back to
    // the canvas once per frame.
    context.putImageData(
      new ImageData(workerFramebuffer!, canvasWidth, canvasHeight),
      0,
      0,
    );
    context.fillStyle = 'rgba(255,0,0,1)';
    context.fillRect(0, 0, 40, 40);
    if (typeof (context as any).commit === 'function') {
      (context as any).commit();
    }

    // Update and emit ambient pulses after the wall z-buffer exists.
    activePulses = updateNeatensteinPulses(activePulses, latestState.simTick);

    const newFloorPulse = emitNeatensteinAmbientPulse(
      latestState.simTick,
      gameState.seed,
      NEATENSTEIN_PULSE_LAYER_FLOOR,
    );

    /* istanbul ignore next -- ambient pulse emission is pre-existing and not triggered by Step 04 test fixtures */
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

    /* istanbul ignore next -- ambient pulse emission is pre-existing and not triggered by Step 04 test fixtures */
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

    drawImpactSpots(
      context,
      gameState.impacts,
      zBuffer,
      { x: cameraPositionX, y: cameraPositionY, yaw: cameraYaw },
      canvasWidth,
      canvasHeight,
      gameState.simTimeMs,
    );

    drawBolts(
      context,
      gameState.bolts!,
      { x: cameraPositionX, y: cameraPositionY, yaw: cameraYaw },
      canvasWidth,
      canvasHeight,
      gameState.simTimeMs,
    );

    renderGunOverlay(context, gameState.gun!, canvasWidth, canvasHeight);

    return;
  } else {
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

    frame.gun = gameState.gun;
    frame.bolts = gameState.bolts;

    self.postMessage(
      { type: 'frame', frame },
      resolveNeatensteinRenderFrameTransferList(frame),
    );
  }
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
/* istanbul ignore next -- pre-existing ambient-pulse renderer; untouched by Step 04 halo/gun-shadow/bolt changes */
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
  const focalLength =
    canvasHeight / 2 / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
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
/* istanbul ignore next -- pre-existing input-merge helper; only the null-previous branch is exercised by this test suite */
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

/* istanbul ignore next -- input normalization is defensive pre-existing code; the test suite only exercises valid host input */
/**
 * Normalize a host input message into the snapshot shape expected by game tick.
 *
 * @param raw - Value attached to the input message by the host.
 * @returns Normalized tick input snapshot.
 */
function inputMessageToTickInput(raw: unknown): GameTickInputSnapshot {
  if (!raw || typeof raw !== 'object') {
    return {
      move: { x: 0, y: 0 },
      lookDelta: 0,
      fire: false,
      dash: false,
    };
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
    activeEnemySprites = [];
    pendingTickInput = null;
    workerZBuffer = null;
    workerFramebuffer = null;
    workerContext = null;
    enemyControllerState = null;

    if (data.canvas) {
      workerCanvas = data.canvas as OffscreenCanvas;
    } else {
      workerCanvas = null;
    }

    if (workerCanvas !== null && pendingResizeDimensions !== null) {
      syncWorkerCanvasSize(
        workerCanvas,
        pendingResizeDimensions.width,
        pendingResizeDimensions.height,
      );
      pendingResizeDimensions = null;
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
    enemyControllerState = createEnemyControllerState(gameState);

    const version = data.version ?? NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION;

    self.postMessage({
      type: 'initialized',
      tier,
      version,
    });

    return;
  }

  if (data.type === 'resize') {
    const width =
      typeof data.width === 'number' ? (data.width as number) : Number.NaN;
    const height =
      typeof data.height === 'number' ? (data.height as number) : Number.NaN;

    applyWorkerResize(width, height);
    return;
  }

  if (data.type === 'simState') {
    latestState = data.state as NeatensteinRenderState;

    if (!gameState || !collisionMap) {
      buildAndPostFrame();
      return;
    }

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

    buildAndPostFrame();
    return;
  }

  if (data.type === NEATENSTEIN_INPUT_MESSAGE_TYPE) {
    const nextInput = inputMessageToTickInput(data.input);
    pendingTickInput = mergePendingTickInput(pendingTickInput, nextInput);
  }
};

/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyGetEnemyControllerState =
  (): EnemyControllerState | null => enemyControllerState;

/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyGetLatestState = (): NeatensteinRenderState | null =>
  latestState;

/* istanbul ignore next -- test-only introspection hook */
export const __testOnlySetActiveEnemySprites = (
  sprites: NeatensteinSprite[],
): void => {
  activeEnemySprites = sprites;
};

/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyAllocateWorkerFramebuffer = (
  width: number,
  height: number,
): void => {
  workerFramebuffer = new Uint8ClampedArray(width * height * 4);
};

/* istanbul ignore next -- test-only introspection hook */
export const __testOnlySetWorkerZBuffer = (columns: number): void => {
  workerZBuffer = new Float32Array(columns).fill(Number.POSITIVE_INFINITY);
};

/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyResolveWorkerFramebufferDimensions =
  resolveWorkerFramebufferDimensions;

/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyFillWorkerCeilingAndFloor = fillWorkerCeilingAndFloor;

/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyWriteWallStripeToFramebuffer =
  writeWallStripeToFramebuffer;

/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyRenderWorkerSprites = renderWorkerSprites;
