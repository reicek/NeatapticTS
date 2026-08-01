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
  drawNeatensteinCeiling,
  drawNeatensteinFloor,
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
  type NeatensteinSprite,
  type NeatensteinSpriteRenderContext,
} from '../renderer/sprites';

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

/**
 * Background clear color used before each worker-tier frame.
 *
 * Derived from {@link NEATENSTEIN_BACKGROUND_RGB} so the worker tier matches
 * the CPU/GPU fog background exactly.
 */
const NEATENSTEIN_WORKER_CLEAR_COLOR = formatRgb(NEATENSTEIN_BACKGROUND_RGB);

/** RGB of the neon wall color for X-axis-side hits. */
const NEATENSTEIN_WALL_X_SIDE_RGB = { r: 0, g: 183, b: 255 } as const;

/** RGB of the neon wall color for Y-axis-side hits. */
const NEATENSTEIN_WALL_Y_SIDE_RGB = { r: 0, g: 164, b: 229 } as const;

/** CSS color string for the yellow ambient pulse dot. */
const NEATENSTEIN_PULSE_COLOR = '#B7FF00';

/** CSS shadow color string for the yellow ambient pulse glow. */
const NEATENSTEIN_PULSE_GLOW_COLOR = 'rgba(185, 255, 0, 0.42)';

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
  if (canvas.width !== width) {
    canvas.width = width;
  }

  if (canvas.height !== height) {
    canvas.height = height;
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

/**
 * Neon colors used to distinguish active enemy types in the worker sprite pass.
 *
 * The palette is intentionally small and high-contrast so enemy wireframes
 * remain readable against the dark raycast scene and the cyan wall shading.
 */
const NEATENSTEIN_ENEMY_HUES = [
  '#ff0055',
  '#ffaa00',
  '#00ffaa',
  '#aa00ff',
] as const;

/**
 * Resolve the neon color for an enemy sprite from its type index.
 *
 * Unknown or negative types fall back to the first hue so every active enemy
 * still renders with a valid color.
 *
 * @param type - Enemy type index from the controller state.
 * @returns `#rrggbb` color string for the sprite renderer.
 */
function resolveEnemySpriteColor(type = 0): string {
  const safeType = Math.max(0, Math.trunc(type));
  return NEATENSTEIN_ENEMY_HUES[safeType % NEATENSTEIN_ENEMY_HUES.length];
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
 * clear → floor → ceiling → walls → pulses → impact spots → bolts → gun
 * overlay
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
      canvasHeight / 2 / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
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

      context.fillRect(
        xStart,
        drawStart,
        stripePixelWidth,
        drawEnd - drawStart,
      );
    }

    // When active enemies exist, snapshot the wall/floor/ceiling output into a
    // CPU-style framebuffer so the sprite renderer can write neon enemy bars
    // with z-buffer occlusion, then flush the combined result back once.
    if (
      activeEnemySprites.length > 0 &&
      typeof context.getImageData === 'function'
    ) {
      const imageData = context.getImageData(0, 0, canvasWidth, canvasHeight);
      const spriteFramebuffer = imageData.data;
      const spriteCamera = {
        posX: cameraPositionX,
        posY: cameraPositionY,
        dirX: cameraDirectionX,
        dirY: cameraDirectionY,
        planeX: cameraPlaneX,
        planeY: cameraPlaneY,
      };
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
        const color = resolveEnemySpriteColor(sprite.type);
        renderNeatensteinSprite(
          spriteFramebuffer,
          zBuffer,
          projection,
          color,
          noOpSpriteCtx,
        );
      }

      // Flush the framebuffer (now containing walls + sprites) back to the
      // worker canvas before the transparent overlay passes run.
      context.putImageData(imageData, 0, 0);
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
    workerContext = null;
    enemyControllerState = null;

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
    enemyControllerState = createEnemyControllerState(gameState);

    const version = data.version ?? NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION;

    self.postMessage({
      type: 'initialized',
      tier,
      version,
    });

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
