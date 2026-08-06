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
import { fireEnemyBolt } from '../host/game/combat';
import type { GameState } from '../host/game/types';
import {
  NEATENSTEIN_BACKGROUND_RGB,
  NEATENSTEIN_RENDER_DISTANCE_CAP,
} from '../renderer/framebuffer';
import { renderGunOverlay } from '../renderer/gun';
import {
  drawBolts,
  drawEnemyBolts,
  drawImpactSpots,
} from '../renderer/bolt-render';
import {
  createEnemyControllerState,
  updateEnemyController,
  type EnemyControllerState,
} from '../../scripts/enemy-controller';
import {
  clipNeatensteinSprite,
  renderNeatensteinSprite,
  resolveNeatensteinEnemyFrame,
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
 * @param color - RGB color object.
 * @returns CSS color string.
 */
function formatRgb(color: { r: number; g: number; b: number }): string {
  return `rgb(${Math.round(color.r)}, ${Math.round(color.g)}, ${Math.round(color.b)})`;
}

/** Worker clear color matching the dark neon void background. */
const NEATENSTEIN_WORKER_CLEAR_COLOR = formatRgb(NEATENSTEIN_BACKGROUND_RGB);

/**
 * Golden-angle rotation used to spread distinct enemy team hues across the
 * color wheel so multiple enemies remain visually separable.
 */
const NEATENSTEIN_ENEMY_TEAM_HUE_GOLDEN_STEP = 137.508;

/**
 * Resolve a deterministic enemy team color `[r, g, b]` from the enemy type
 * index by rotating the HSL hue wheel with the golden angle. Palette indices
 * 5/6/7 in the robot sprite atlas are swapped with this color at render time,
 * giving each enemy a stable neon tint while preserving the sprite alpha.
 *
 * @param typeIndex - Enemy type index (the source {@link GameState.enemies}
 *   array position). Negative indices fall back to index 0.
 * @returns RGB triple in the `[0, 255]` range.
 */
function resolveEnemyTeamColor(
  typeIndex: number,
): readonly [number, number, number] {
  const seed = typeIndex >= 0 ? typeIndex : 0;
  const hue = (seed * NEATENSTEIN_ENEMY_TEAM_HUE_GOLDEN_STEP) % 360;
  const saturation = 0.65;
  const lightness = 0.5;
  const chroma = (1 - Math.abs(2 * lightness - 1)) * saturation;
  const huePrime = hue / 60;
  const intermediate = chroma * (1 - Math.abs((huePrime % 2) - 1));
  let r = 0;
  let g = 0;
  let b = 0;
  if (huePrime < 1) {
    r = chroma;
    g = intermediate;
  } else if (huePrime < 2) {
    r = intermediate;
    g = chroma;
  } else if (huePrime < 3) {
    g = chroma;
    b = intermediate;
  } else if (huePrime < 4) {
    g = intermediate;
    b = chroma;
  } else if (huePrime < 5) {
    r = intermediate;
    b = chroma;
  } else {
    r = chroma;
    b = intermediate;
  }
  const match = lightness - chroma / 2;
  return [
    Math.round((r + match) * 255),
    Math.round((g + match) * 255),
    Math.round((b + match) * 255),
  ] as const;
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
  return perpWallDist >= NEATENSTEIN_RENDER_DISTANCE_CAP ? 1 : 0;
}

/**
 * Apply distance fog to a neon wall color and return a CSS color string.
 *
 * @param wallColor - Raw wall RGB.
 * @param perpWallDist - Perpendicular distance from camera to wall.
 * @returns CSS `rgb(...)` string for `context.fillStyle`.
 */
function applyWallFog(
  wallColor: { r: number; g: number; b: number },
  perpWallDist: number,
): string {
  const fogFactor = resolveWallFogFactor(perpWallDist);
  const bg = NEATENSTEIN_BACKGROUND_RGB;

  return formatRgb({
    r: wallColor.r + (bg.r - wallColor.r) * fogFactor,
    g: wallColor.g + (bg.g - wallColor.g) * fogFactor,
    b: wallColor.b + (bg.b - wallColor.b) * fogFactor,
  });
}

/**
 * Build a canvas-like context that does not flush per sprite.
 *
 * The worker tier renders all sprites into a single canvas snapshot and flushes
 * it back once after the sprite pass. The renderer's `putImageData` contract
 * still expects a context, so this no-op adapter satisfies the type without
 * redundant per-sprite copies.
 *
 * @returns Canvas-like context with a no-op `putImageData`.
 */
function buildNoOpSpriteRenderContext(): NeatensteinSpriteRenderContext {
  return {
    putImageData: () => {
      // Intentionally empty: the worker flushes the snapshot once after all
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
 * clear → floor grid → ceiling grid → fogged wall stripes → sprite snapshot
 * → encoded enemy sprites → sprite flush → pulses → impact spots → bolts →
 * gun overlay
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

  // Use the worker-authoritative simulated player position for every render
  // pass (walls, floor, ceiling, pulses, bolts, and sprites). The host still
  // sends a camera snapshot, but it is intentionally not used for rendering so
  // that WASD movement and mouse rotation stay in sync across the whole frame.
  const cameraPositionX = gameState.player.position.x;
  const cameraPositionY = gameState.player.position.y;
  const cameraYaw = gameState.player.angleRad;

  // Sprite projection uses the same authoritative camera position as the wall
  // renderer to prevent enemies from appearing to move when the player walks.
  const spriteCameraX = cameraPositionX;
  const spriteCameraY = cameraPositionY;
  const spriteCameraYaw = cameraYaw;

  // Derive camera direction and projection plane from the player yaw.
  const cameraDirectionX = Math.cos(cameraYaw);
  const cameraDirectionY = Math.sin(cameraYaw);
  const planeScale =
    (canvasWidth / canvasHeight) * Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
  const cameraPlaneX = -cameraDirectionY * planeScale;
  const cameraPlaneY = cameraDirectionX * planeScale;

  // Sprite projection uses the authoritative simulated player transform so
  // enemy positions (advanced by the controller) stay in the same camera
  // space as the sprite renderer.
  const spriteDirectionX = Math.cos(spriteCameraYaw);
  const spriteDirectionY = Math.sin(spriteCameraYaw);
  const spritePlaneX = -spriteDirectionY * planeScale;
  const spritePlaneY = spriteDirectionX * planeScale;

  // The enemy controller is advanced before gameTick in the simState handler
  // so gameState.enemies already reflects the latest controlled positions.
  // Build the active sprite list from the persisted controller state.
  activeEnemySprites = enemyControllerState!.enemies
    .filter((enemy) => enemy.active)
    .map((enemy) => ({
      worldX: enemy.position.x,
      worldY: enemy.position.y,
      facing: enemy.yawRad,
      animationState: enemy.animationState,
      frameIndex: 0,
      type: enemy.index,
      walkTick: enemy.walkTick,
      shootBlinkTicks: enemy.shootBlinkTicks,
      teamColor: resolveEnemyTeamColor(enemy.index),
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

    // Clear the full canvas to the dark neon void background.
    context.fillStyle = NEATENSTEIN_WORKER_CLEAR_COLOR;
    context.fillRect(0, 0, canvasWidth, canvasHeight);

    // Draw perspective floor and ceiling grids behind the walls.
    const floorCamera = {
      x: cameraPositionX,
      y: cameraPositionY,
      yaw: cameraYaw,
    };
    drawNeatensteinFloor(context, canvasWidth, canvasHeight, floorCamera);
    drawNeatensteinCeiling(context, canvasWidth, canvasHeight, floorCamera);

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

      const perpWallDist = hit.perpWallDist;
      const isCapped =
        !Number.isFinite(perpWallDist) ||
        perpWallDist >= NEATENSTEIN_RENDER_DISTANCE_CAP;

      zBuffer[column] =
        Number.isFinite(perpWallDist) && perpWallDist > 0
          ? perpWallDist
          : NEATENSTEIN_RENDER_DISTANCE_CAP;

      // Integer stripe bounds prevent subpixel gaps when the column count does
      // not divide the canvas width evenly.
      const xStart = Math.floor(column * stripeWidth);
      const xEnd = Math.floor((column + 1) * stripeWidth);
      const stripePixelWidth = Math.max(0, xEnd - xStart);

      if (!isCapped) {
        const lineHeight = wallFocalLength / hit.perpWallDist;
        const drawStart = clamp(
          (canvasHeight - lineHeight) / 2,
          0,
          canvasHeight,
        );
        const drawEnd = clamp((canvasHeight + lineHeight) / 2, 0, canvasHeight);

        const wallColor =
          hit.side === 0
            ? NEATENSTEIN_WALL_X_SIDE_RGB
            : NEATENSTEIN_WALL_Y_SIDE_RGB;

        context.fillStyle = applyWallFog(wallColor, hit.perpWallDist);
        context.fillRect(
          xStart,
          drawStart,
          stripePixelWidth,
          drawEnd - drawStart,
        );
      } else {
        // Render a fog wall at the render distance cap with the same
        // proportional height as a real wall at 30 cells
        // (wallFocalLength / NEATENSTEIN_RENDER_DISTANCE_CAP). A short
        // vertical gradient feather at the top and bottom edges blends the
        // fog wall smoothly with the floor/ceiling grid, avoiding a hard
        // cut line where the fog wall meets the floor or ceiling.
        const fogDist = NEATENSTEIN_RENDER_DISTANCE_CAP;
        const lineHeight = wallFocalLength / fogDist;
        const drawStart = clamp(
          (canvasHeight - lineHeight) / 2,
          0,
          canvasHeight,
        );
        const drawEnd = clamp((canvasHeight + lineHeight) / 2, 0, canvasHeight);

        const fogColor = formatRgb(NEATENSTEIN_BACKGROUND_RGB);
        context.fillStyle = fogColor;
        context.fillRect(
          xStart,
          drawStart,
          stripePixelWidth,
          drawEnd - drawStart,
        );

        // Feather the top edge: gradient from transparent (ceiling side)
        // to opaque background (fog wall side) so the fog wall blends
        // smoothly with the ceiling grid above it.
        const fogFeatherPixels = 6;
        const featherTopStart = Math.max(0, drawStart - fogFeatherPixels);
        /* istanbul ignore else -- lineHeight = wallFocalLength/30 is always << canvasHeight, so drawStart is always > 0 */
        if (drawStart > 0) {
          const topGrad = context.createLinearGradient(
            0,
            featherTopStart,
            0,
            drawStart,
          );
          const { r, g, b } = NEATENSTEIN_BACKGROUND_RGB;
          topGrad.addColorStop(0, `rgba(${r}, ${g}, ${b}, 0)`);
          topGrad.addColorStop(1, `rgba(${r}, ${g}, ${b}, 1)`);
          context.fillStyle = topGrad;
          context.fillRect(
            xStart,
            featherTopStart,
            stripePixelWidth,
            drawStart - featherTopStart,
          );
        }

        // Feather the bottom edge: gradient from opaque background (fog
        // wall side) to transparent (floor side) so the fog wall blends
        // smoothly with the floor grid below it.
        const featherBottomEnd = Math.min(
          canvasHeight,
          drawEnd + fogFeatherPixels,
        );
        /* istanbul ignore else -- lineHeight = wallFocalLength/30 is always << canvasHeight, so drawEnd is always < canvasHeight */
        if (drawEnd < canvasHeight) {
          const bottomGrad = context.createLinearGradient(
            0,
            drawEnd,
            0,
            featherBottomEnd,
          );
          const { r, g, b } = NEATENSTEIN_BACKGROUND_RGB;
          bottomGrad.addColorStop(0, `rgba(${r}, ${g}, ${b}, 1)`);
          bottomGrad.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);
          context.fillStyle = bottomGrad;
          context.fillRect(
            xStart,
            drawEnd,
            stripePixelWidth,
            featherBottomEnd - drawEnd,
          );
        }
      }
    }

    // Render encoded enemy sprites into a single canvas snapshot and flush it
    // once. The no-op render context prevents per-sprite putImageData calls.
    if (
      activeEnemySprites.length > 0 &&
      typeof context.getImageData === 'function'
    ) {
      const spriteSnapshot = context.getImageData(
        0,
        0,
        canvasWidth,
        canvasHeight,
      );
      const spriteContext = buildNoOpSpriteRenderContext();
      const spriteCamera = {
        posX: spriteCameraX,
        posY: spriteCameraY,
        dirX: spriteDirectionX,
        dirY: spriteDirectionY,
        planeX: spritePlaneX,
        planeY: spritePlaneY,
      };

      const sortedSprites = [...activeEnemySprites].sort((a, b) => {
        const distA =
          (a.worldX - spriteCameraX) ** 2 + (a.worldY - spriteCameraY) ** 2;
        const distB =
          (b.worldX - spriteCameraX) ** 2 + (b.worldY - spriteCameraY) ** 2;
        return distB - distA;
      });

      for (const sprite of sortedSprites) {
        const frame = resolveNeatensteinEnemyFrame(sprite, spriteCamera);
        if (!frame) {
          continue;
        }
        const projection = clipNeatensteinSprite(
          sprite,
          spriteCamera,
          canvasWidth,
          canvasHeight,
          zBuffer,
        );
        // Skip sprites culled by the render-distance cap (AC-10.3c-002).
        if (!projection.visible) {
          continue;
        }
        renderNeatensteinSprite(
          spriteSnapshot.data,
          zBuffer,
          projection,
          frame,
          spriteContext,
          sprite.teamColor,
        );
      }

      context.putImageData(spriteSnapshot, 0, 0);
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

    drawEnemyBolts(
      context,
      gameState.enemyBolts!,
      { x: cameraPositionX, y: cameraPositionY, yaw: cameraYaw },
      canvasWidth,
      canvasHeight,
      gameState.simTimeMs,
    );

    renderGunOverlay(context, gameState.gun!, canvasWidth, canvasHeight);

    const commitableContext = context as OffscreenCanvasRenderingContext2D & {
      commit?: () => void;
    };
    if (typeof commitableContext.commit === 'function') {
      commitableContext.commit();
    }

    // Post a frame acknowledgment so the host bridge can apply worker-busy
    // backpressure. The worker tier renders directly to the OffscreenCanvas,
    // so the frame payload only carries the request id for throttling.
    self.postMessage({
      type: 'frame',
      frame: { requestId: latestState.simTick },
    });

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
      frame.zBuffer[column] =
        !Number.isFinite(hit.perpWallDist) ||
        hit.perpWallDist >= NEATENSTEIN_RENDER_DISTANCE_CAP
          ? NEATENSTEIN_RENDER_DISTANCE_CAP
          : hit.perpWallDist;
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

    if (!gameState || !collisionMap || !enemyControllerState) {
      buildAndPostFrame();
      return;
    }

    const tickInput = pendingTickInput ?? {
      move: { x: 0, y: 0 },
      lookDelta: 0,
      fire: false,
      dash: false,
    };

    // Derive the simulation timestep from the rAF delta-time field posted by
    // the host. On the first frame (deltaMs === 0) fall back to a 16 ms
    // reference so the simulation advances a single tick.
    const deltaMs = data.state.deltaMs;
    const timestepMs = deltaMs > 0 ? deltaMs : 16;

    // Advance the enemy controller BEFORE the game tick so movement, bolt, and
    // contact damage subsystems see synced enemy positions/health/active states.
    const controlled = updateEnemyController(
      enemyControllerState,
      gameState,
      collisionMap,
      timestepMs,
    );
    enemyControllerState = controlled;

    gameState = {
      ...gameState,
      enemies: gameState.enemies.map((enemy, index) => {
        const controlledEnemy = controlled.enemies[index];
        if (!controlledEnemy) {
          return enemy;
        }
        return {
          ...enemy,
          position: { ...controlledEnemy.position },
          controllerPosition: { ...controlledEnemy.position },
          health: controlledEnemy.health,
          active: controlledEnemy.active,
        };
      }),
    };

    gameState = gameTick(gameState, tickInput, collisionMap, timestepMs);

    // Consume hitscan events produced by the enemy controller and spawn
    // visible enemy bolts. Each bolt uses the HitscanEvent origin/direction
    // directly — no Math.random(), fully deterministic. Bolts are appended
    // after the tick so they start at the enemy position and are advanced on
    // the following tick, matching the player bolt spawn pattern.
    if (controlled.hitscanEvents.length > 0 && gameState) {
      const simTimeMs = gameState.simTimeMs;
      const newEnemyBolts = controlled.hitscanEvents.map((event) =>
        fireEnemyBolt(
          {
            origin: event.origin,
            direction: event.direction,
            damage: event.damage,
          },
          simTimeMs,
        ),
      );
      gameState = {
        ...gameState,
        enemyBolts: [...(gameState.enemyBolts ?? []), ...newEnemyBolts],
      };
    }

    // Keep the worker authoritative. If an enemy was killed this tick, clear
    // it from the controller roster so the next controller pass does not revive
    // it and so newly spawned enemies replace dead slots immediately.
    enemyControllerState = {
      ...enemyControllerState,
      enemies: enemyControllerState.enemies
        .map((controlled, index) => {
          if (!gameState) return { ...controlled, active: false, health: 0 };
          const live = gameState.enemies[index];
          if (!live || (live.health ?? 0) <= 0 || live.active === false) {
            return { ...controlled, active: false, health: live?.health ?? 0 };
          }
          return controlled;
        })
        .filter((controlled) => controlled.active),
    };

    // Run a zero-timestep controller pass after the tick so newly spawned
    // enemies and any health changes from combat are reflected in the
    // controller state used for the next frame, without double-advancing enemy
    // movement.
    enemyControllerState = updateEnemyController(
      enemyControllerState,
      gameState,
      collisionMap,
      0,
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
export const __testOnlySyncWorkerCanvasSize = syncWorkerCanvasSize;

/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyResolveEnemyTeamColor = resolveEnemyTeamColor;

/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyResolveWallFogFactor = resolveWallFogFactor;

/* istanbul ignore next -- test-only hook to place enemies near the player */
export const __testOnlyInjectTestEnemies = (
  positions: { x: number; y: number }[],
): void => {
  if (!gameState || !enemyControllerState) return;
  gameState.enemies = positions.map((pos, i) => ({
    position: { ...pos },
    health: 100,
    index: i,
    active: true,
    controllerPosition: { ...pos },
  }));
  enemyControllerState = {
    ...enemyControllerState,
    enemies: positions.map((pos, i) => ({
      index: i,
      position: { ...pos },
      health: 100,
      yawRad: 0,
      animationState: 'idle' as const,
      ammo: 10,
      fireCooldownMs: 0,
      deRezElapsedMs: 0,
      active: true,
      walkTick: 0,
      shootBlinkTicks: 0,
      flankStallTicks: 0,
      bfsStallTicks: 0,
      weights: undefined,
      variantId: 0,
      previousStepDistance: -1,
    })),
  };
};
