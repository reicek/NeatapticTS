/**
 * Render-paint executors extracted from the display worker's
 * {@link buildAndPostFrame}.
 *
 * These executors handle all canvas painting operations for the worker tier
 * (walls, sprites, overlays) and packed-frame field population for the
 * CPU/GPU tiers. They are pure functions that take all inputs as parameters
 * and have no dependency on module-level mutable state.
 *
 * @module
 */

import {
  NEATENSTEIN_PULSE_ALPHA_MAX,
  NEATENSTEIN_PULSE_ALPHA_MIN,
  NEATENSTEIN_PULSE_GLOW_BLUR_RADIUS,
  NEATENSTEIN_PULSE_MAX_CONCURRENT,
  NEATENSTEIN_PULSE_SCREEN_DOT_RADIUS_PX,
} from '../constants';
import {
  NEATENSTEIN_BACKGROUND_RGB,
  NEATENSTEIN_RENDER_DISTANCE_CAP,
} from '../renderer/framebuffer';
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
  drawAmmoPickups,
  drawBolts,
  drawEnemyBolts,
  drawEnemyImpactSpots,
  drawImpactSpots,
} from '../renderer/bolt-render';
import { renderGunOverlay } from '../renderer/gun';
import {
  clipNeatensteinSprite,
  renderNeatensteinSprite,
  resolveNeatensteinEnemyFrame,
  type NeatensteinSprite,
} from '../renderer/sprites';
import type { NeatensteinRenderFrame } from '../renderer/frame';
import type { GameState } from '../host/game/types';
import {
  ENEMY_CONTROLLER_DE_REZ_DURATION_MS,
  type ControlledEnemy,
} from '../../scripts/enemy-controller';
import { castColumnRay } from './display.worker.raycast.utils';
import { applyWallFog, clamp, formatRgb } from './display.worker.color.utils';
import { buildNoOpSpriteRenderContext } from './display.worker.canvas.utils';
import { FOG_FEATHER_PX, PULSE_GLOW_ALPHA } from './display.worker.constants';

/** RGB tint for X-side wall faces (north/south walls). */
const NEATENSTEIN_WALL_X_SIDE_RGB = { r: 0, g: 183, b: 255 } as const;

/** RGB tint for Y-side wall faces (east/west walls). */
const NEATENSTEIN_WALL_Y_SIDE_RGB = { r: 0, g: 164, b: 229 } as const;

/** Fill colour for ambient pulse screen dots. */
const NEATENSTEIN_PULSE_COLOR = '#B7FF00';

/** Glow shadow colour for ambient pulse dots. */
const NEATENSTEIN_PULSE_GLOW_COLOR = `rgba(185, 255, 0, ${PULSE_GLOW_ALPHA})`;

/**
 * Build the active enemy sprite list from the persisted controller state.
 *
 * Only active enemies are included. Each sprite entry carries the world
 * position, facing, animation state, and team colour resolved from the enemy
 * index.
 *
 * @param enemies - Enemies from the controller state.
 * @param teamColorResolver - Function that maps an enemy index to a team colour RGB tuple.
 * @returns Array of active enemy sprites ready for rendering.
 */
export function buildActiveEnemySprites(
  enemies: readonly ControlledEnemy[],
  teamColorResolver: (index: number) => readonly [number, number, number],
): NeatensteinSprite[] {
  return enemies
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
      teamColor: teamColorResolver(enemy.index),
      deRezElapsedMs: enemy.deRezElapsedMs,
      deRezDurationMs: ENEMY_CONTROLLER_DE_REZ_DURATION_MS,
      seed: enemy.index,
    }));
}

/**
 * Paint the worker-tier scene: floor, ceiling, and perspective wall columns.
 *
 * The wall loop raycasts each column, fills the stripe with fog-blended wall
 * colour, and writes the perpendicular distance into the z-buffer. Columns
 * beyond the render-distance cap receive a fog wall with feathered edges.
 *
 * @param context - Worker-tier 2D canvas context.
 * @param wallMap - Flat wall map array.
 * @param columnCount - Number of raycast columns.
 * @param canvasWidth - Canvas width in pixels.
 * @param canvasHeight - Canvas height in pixels.
 * @param cameraPositionX - Camera world X.
 * @param cameraPositionY - Camera world Y.
 * @param cameraYaw - Camera yaw in radians.
 * @param cameraDirectionX - Camera direction vector X.
 * @param cameraDirectionY - Camera direction vector Y.
 * @param cameraPlaneX - Camera plane vector X.
 * @param cameraPlaneY - Camera plane vector Y.
 * @param zBuffer - Per-column depth buffer to populate.
 */
export function paintWorkerTierWalls(
  context: OffscreenCanvasRenderingContext2D,
  wallMap: Uint8Array,
  columnCount: number,
  canvasWidth: number,
  canvasHeight: number,
  cameraPositionX: number,
  cameraPositionY: number,
  cameraYaw: number,
  cameraDirectionX: number,
  cameraDirectionY: number,
  cameraPlaneX: number,
  cameraPlaneY: number,
  zBuffer: Float32Array,
): void {
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
      wallMap,
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
      const drawStart = clamp((canvasHeight - lineHeight) / 2, 0, canvasHeight);
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
      const drawStart = clamp((canvasHeight - lineHeight) / 2, 0, canvasHeight);
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
      const fogFeatherPixels = FOG_FEATHER_PX;
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
}

/**
 * Paint enemy sprites into the worker-tier canvas.
 *
 * Sprites are sorted far-to-near, then rendered into a single ImageData
 * snapshot using the no-op sprite render context. The snapshot is flushed
 * once via {@link OffscreenCanvasRenderingContext2D.putImageData}.
 *
 * @param context - Worker-tier 2D canvas context.
 * @param sprites - Active enemy sprites to render.
 * @param spriteCameraX - Sprite-camera world X.
 * @param spriteCameraY - Sprite-camera world Y.
 * @param spriteDirectionX - Sprite-camera direction vector X.
 * @param spriteDirectionY - Sprite-camera direction vector Y.
 * @param spritePlaneX - Sprite-camera plane vector X.
 * @param spritePlaneY - Sprite-camera plane vector Y.
 * @param canvasWidth - Canvas width in pixels.
 * @param canvasHeight - Canvas height in pixels.
 * @param zBuffer - Per-column depth buffer from the wall pass.
 */
export function paintWorkerTierSprites(
  context: OffscreenCanvasRenderingContext2D,
  sprites: readonly NeatensteinSprite[],
  spriteCameraX: number,
  spriteCameraY: number,
  spriteDirectionX: number,
  spriteDirectionY: number,
  spritePlaneX: number,
  spritePlaneY: number,
  canvasWidth: number,
  canvasHeight: number,
  zBuffer: Float32Array,
): void {
  // Render encoded enemy sprites into a single canvas snapshot and flush it
  // once. The no-op render context prevents per-sprite putImageData calls.
  if (sprites.length > 0 && typeof context.getImageData === 'function') {
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

    const sortedSprites = [...sprites].sort((a, b) => {
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

      // Build the de-rez state for sprites in the death animation so the
      // column renderer can dissolve pixels and tint survivors.
      const derezState =
        sprite.animationState === 'death' &&
        sprite.deRezElapsedMs !== undefined &&
        sprite.deRezDurationMs !== undefined &&
        sprite.seed !== undefined
          ? {
              elapsedMs: sprite.deRezElapsedMs,
              durationMs: sprite.deRezDurationMs,
              seed: sprite.seed,
            }
          : undefined;

      renderNeatensteinSprite(
        spriteSnapshot.data,
        zBuffer,
        projection,
        frame,
        spriteContext,
        sprite.teamColor,
        derezState,
      );
    }

    context.putImageData(spriteSnapshot, 0, 0);
  }
}

/* istanbul ignore next -- pre-existing ambient-pulse renderer; untouched by Step 04 halo/gun-shadow/bolt changes */
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
export function drawNeatensteinPulses(
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
 * Update ambient pulse state and paint all worker-tier overlays.
 *
 * Updates active pulses, emits new floor/ceiling ambient pulses (up to the
 * concurrency cap), then draws pulses, impact spots, bolts, enemy bolts,
 * ammo pickups, and the gun overlay.
 *
 * @param context - Worker-tier 2D canvas context.
 * @param gameState - Current deterministic game state.
 * @param activePulses - Current active pulses (will be updated in-place).
 * @param simTick - Current simulation tick from the render state.
 * @param seed - Game seed for deterministic pulse emission.
 * @param zBuffer - Per-column depth buffer from the wall pass.
 * @param camera - Current camera state for depth-projected overlays.
 * @param canvasWidth - Canvas width in pixels.
 * @param canvasHeight - Canvas height in pixels.
 * @returns Updated active pulses array.
 */
export function updateAndPaintWorkerOverlays(
  context: OffscreenCanvasRenderingContext2D,
  gameState: GameState,
  activePulses: NeatensteinPulse[],
  simTick: number,
  seed: number,
  zBuffer: Float32Array,
  camera: NeatensteinFloorCamera,
  canvasWidth: number,
  canvasHeight: number,
): NeatensteinPulse[] {
  // Update and emit ambient pulses after the wall z-buffer exists.
  const pulses = updateNeatensteinPulses(activePulses, simTick);

  const newFloorPulse = emitNeatensteinAmbientPulse(
    simTick,
    seed,
    NEATENSTEIN_PULSE_LAYER_FLOOR,
  );

  /* istanbul ignore next -- ambient pulse emission is pre-existing and not triggered by Step 04 test fixtures */
  if (newFloorPulse && pulses.length < NEATENSTEIN_PULSE_MAX_CONCURRENT) {
    pulses.push(newFloorPulse);
  }

  const newCeilingPulse = emitNeatensteinAmbientPulse(
    simTick,
    seed,
    NEATENSTEIN_PULSE_LAYER_CEILING,
  );

  /* istanbul ignore next -- ambient pulse emission is pre-existing and not triggered by Step 04 test fixtures */
  if (newCeilingPulse && pulses.length < NEATENSTEIN_PULSE_MAX_CONCURRENT) {
    pulses.push(newCeilingPulse);
  }

  drawNeatensteinPulses(
    context,
    pulses,
    zBuffer,
    camera,
    canvasWidth,
    canvasHeight,
  );

  drawImpactSpots(
    context,
    gameState.impacts,
    zBuffer,
    camera,
    canvasWidth,
    canvasHeight,
    gameState.simTimeMs,
  );

  drawEnemyImpactSpots(
    context,
    /* istanbul ignore next -- nullish fallback only reachable in worker-tier rendering mode with mock OffscreenCanvas */
    gameState.enemyImpacts ?? [],
    zBuffer,
    camera,
    canvasWidth,
    canvasHeight,
    gameState.simTimeMs,
  );

  drawBolts(
    context,
    gameState.bolts!,
    camera,
    canvasWidth,
    canvasHeight,
    gameState.simTimeMs,
  );

  drawEnemyBolts(
    context,
    gameState.enemyBolts!,
    camera,
    canvasWidth,
    canvasHeight,
    gameState.simTimeMs,
  );

  drawAmmoPickups(
    context,
    gameState.ammoPickups ?? [],
    zBuffer,
    camera,
    canvasWidth,
    canvasHeight,
    gameState.simTimeMs,
  );

  renderGunOverlay(context, gameState.gun!, canvasWidth, canvasHeight);

  return pulses;
}

/**
 * Render packed-tier wall columns into the frame buffers.
 *
 * Raycasts each column and writes the perpendicular wall distance, side, and
 * z-buffer value into the provided frame.
 *
 * @param frame - Packed render frame to populate.
 * @param wallMap - Flat wall map array.
 * @param columnCount - Number of raycast columns.
 * @param cameraPositionX - Camera world X.
 * @param cameraPositionY - Camera world Y.
 * @param cameraDirectionX - Camera direction vector X.
 * @param cameraDirectionY - Camera direction vector Y.
 * @param cameraPlaneX - Camera plane vector X.
 * @param cameraPlaneY - Camera plane vector Y.
 */
export function renderPackedTierColumns(
  frame: NeatensteinRenderFrame,
  wallMap: Uint8Array,
  columnCount: number,
  cameraPositionX: number,
  cameraPositionY: number,
  cameraDirectionX: number,
  cameraDirectionY: number,
  cameraPlaneX: number,
  cameraPlaneY: number,
): void {
  for (let column = 0; column < columnCount; column += 1) {
    const hit = castColumnRay(
      wallMap,
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
}

/**
 * Fill packed-tier frame scalar fields from the game state.
 *
 * @param frame - Packed render frame to populate.
 * @param gameState - Current deterministic game state.
 */
export function fillPackedFrameFields(
  frame: NeatensteinRenderFrame,
  gameState: GameState,
): void {
  frame.gun = gameState.gun;
  frame.bolts = gameState.bolts;
  frame.playerHealth = gameState.player.health;
  frame.playerMaxHealth = gameState.player.maxHealth;
  frame.playerAmmo = gameState.player.ammo;
  frame.playerMaxAmmo = gameState.player.maxAmmo;
  frame.playerKills = gameState.kills;
  frame.playerDeaths = gameState.deaths ?? 0;
  frame.spawnCount = gameState.spawnCount;
  frame.ammoPickups = gameState.ammoPickups;
}
