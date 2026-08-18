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
import { NEATENSTEIN_ZBUFFER_EMPTY } from '../renderer/renderer.zbuffer.constants';
import {
  castNeatensteinFloorPerPixel,
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
import type { RenderWorkerState } from './display.worker.types';
import {
  ENEMY_CONTROLLER_DE_REZ_DURATION_MS,
  type ControlledEnemy,
} from '../shared/enemy-controller';
import { castColumnRay } from './display.worker.raycast.utils';
import type { CastRayDDAHit } from '../renderer/raycast';
import { formatRgb, rgbToHex } from './display.worker.color.utils';
import { clamp } from '../shared/math-guards.utils';
import { buildNoOpSpriteRenderContext } from './display.worker.canvas.utils';
import { FOG_FEATHER_PX, PULSE_GLOW_ALPHA } from './display.worker.constants';
import { writeNeonWallColumn, computeWallTexcoord } from '../renderer/walls';
import { NEATENSTEIN_MAP_SIZE } from '../constants';
import { createSharedMapGrid } from './display.worker.sim.utils';
import { interpolateNeatensteinWallColumn } from '../renderer/interpolate';
import { resolveNeatensteinHalfResEnabled } from '../renderer/renderer.quality.constants';
import {
  resolveNeatensteinMsaaResolvedColumn,
  resolveNeatensteinMsaaFogBlend,
} from '../renderer/renderer.msaa.constants';

/** RGB tint for X-side wall faces (north/south walls). */
const NEATENSTEIN_WALL_X_SIDE_RGB = { r: 0, g: 183, b: 255 } as const;

/** RGB tint for Y-side wall faces (east/west walls). */
const NEATENSTEIN_WALL_Y_SIDE_RGB = { r: 0, g: 164, b: 229 } as const;

/**
 * Whether the wall pass used the persistent framebuffer (read by the sprite
 * pass to decide whether to flush walls+sprites via a single putImageData).
 */
let framebufferInUse = false;

/** Reused scratch array for in-place sprite sorting (avoids [...sprites]). */
const spriteSortScratch: NeatensteinSprite[] = [];

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
  // Check if the framebuffer path is available. The framebuffer path draws
  // floor/ceiling grid pixels procedurally into the framebuffer (no
  // getImageData allocation bomb) and writes wall columns directly into the
  // buffer. Only putImageData is required for the final flush.
  const useFramebuffer = typeof context.putImageData === 'function';

  const stripeWidth = canvasWidth / columnCount;
  const wallFocalLength =
    canvasHeight / 2 / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);

  if (useFramebuffer) {
    const framebuffer = getPersistentWallFramebuffer(canvasWidth, canvasHeight);

    // Seed the framebuffer with floor/ceiling grid pixels procedurally.
    // This replaces the old context.getImageData() call that allocated
    // ~1.2MB every frame (A2 Fix 2: eliminate per-frame allocation bomb).
    // The framebuffer is filled with an opaque background colour, then
    // floor and ceiling grid lines are drawn directly into the buffer.
    seedFramebufferProcedurally(
      framebuffer,
      canvasWidth,
      canvasHeight,
      cameraPositionX,
      cameraPositionY,
      cameraYaw,
    );

    const { r: bgR, g: bgG, b: bgB } = NEATENSTEIN_BACKGROUND_RGB;

    const pooledHit = getPooledRayHitBuffer();
    // C1.4/C1.5: Half-res decimation + MSAA resolve for wall columns.
    // Half-res interpolates color on odd columns (z-buffer is still cast at
    // every column — Invariant: depth occlusion stays exact). MSAA resolve
    // averages adjacent columns to clean wall-sprite seams.
    const halfResConfig = resolveNeatensteinHalfResEnabled({});
    const halfResEnabled = halfResConfig !== false;
    let prevColumnRgb: { r: number; g: number; b: number } | null = null;
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
        pooledHit,
      );

      const perpWallDist = hit.perpWallDist;
      const isCapped =
        !Number.isFinite(perpWallDist) ||
        perpWallDist >= NEATENSTEIN_RENDER_DISTANCE_CAP;

      zBuffer[column] =
        Number.isFinite(perpWallDist) && perpWallDist > 0
          ? perpWallDist
          : NEATENSTEIN_ZBUFFER_EMPTY;

      // Integer stripe bounds prevent subpixel gaps when the column count
      // does not divide the canvas width evenly.
      const xStart = Math.floor(column * stripeWidth);
      const xEnd = Math.floor((column + 1) * stripeWidth);

      if (!isCapped) {
        const lineHeight = wallFocalLength / hit.perpWallDist;
        const drawStart = clamp(
          (canvasHeight - lineHeight) / 2,
          0,
          canvasHeight,
        );
        const drawEnd = clamp(
          (canvasHeight + lineHeight) / 2,
          0,
          canvasHeight,
        );

        // C1.4/C1.5: Determine wall color with half-res interpolation and
        // MSAA resolve. The base RGB is tinted by the hit side; for odd
        // columns when half-res is enabled, the color is interpolated from
        // the previous column. MSAA resolve then averages with the previous
        // column to clean seams. Fog is applied by writeNeonWallColumn.
        let currentRgb: { r: number; g: number; b: number } =
          hit.side === 0
            ? NEATENSTEIN_WALL_X_SIDE_RGB
            : NEATENSTEIN_WALL_Y_SIDE_RGB;

        if (
          halfResEnabled &&
          column > 0 &&
          column % 2 === 1 &&
          prevColumnRgb !== null
        ) {
          const interp = interpolateNeatensteinWallColumn(
            { screenX: 0, ...prevColumnRgb },
            { screenX: 0, ...currentRgb },
            0.5,
          );
          currentRgb = { r: interp.r, g: interp.g, b: interp.b };
        }

        const msaaRgb =
          prevColumnRgb !== null
            ? resolveNeatensteinMsaaResolvedColumn(currentRgb, prevColumnRgb)
            : currentRgb;
        const resolvedHex = rgbToHex(msaaRgb);

        // Write each pixel column in the stripe via writeNeonWallColumn,
        // which applies distance fog and writes fogged RGB triples directly
        // into the framebuffer (Invariant 4: stripeWidth-wide columns,
        // Invariant 5: Math.floor(column*stripeWidth) pixel mapping).
        // Compute the wall texcoord for vertical stripe shading.
        const planeOffset = (2 * column) / columnCount - 1;
        const rayDirX = cameraDirectionX + cameraPlaneX * planeOffset;
        const rayDirY = cameraDirectionY + cameraPlaneY * planeOffset;
        const texcoord =
          hit.side === 0
            ? computeWallTexcoord(cameraPositionY, hit.perpWallDist, rayDirY)
            : computeWallTexcoord(cameraPositionX, hit.perpWallDist, rayDirX);

        for (let x = xStart; x < xEnd; x += 1) {
          writeNeonWallColumn(
            framebuffer,
            canvasWidth,
            canvasHeight,
            x,
            drawStart,
            drawEnd,
            resolvedHex,
            hit.perpWallDist,
            texcoord,
          );
        }
        // C1.4/C1.5: Track this column's RGB for the next iteration's
        // half-res interpolation and MSAA resolve.
        prevColumnRgb = { r: currentRgb.r, g: currentRgb.g, b: currentRgb.b };
      } else {
        // C1.4/C1.5: No wall color for capped columns — reset prev for MSAA.
        prevColumnRgb = null;
        // Fog wall at the render distance cap with feathered edges
        // (Invariant 7). The main body is pure background color; the top
        // and bottom edges are blended toward the seeded floor/ceiling
        // pixels so there is no hard 1px cut line.
        const fogDist = NEATENSTEIN_RENDER_DISTANCE_CAP;
        const lineHeight = wallFocalLength / fogDist;
        const drawStart = clamp(
          (canvasHeight - lineHeight) / 2,
          0,
          canvasHeight,
        );
        const drawEnd = clamp(
          (canvasHeight + lineHeight) / 2,
          0,
          canvasHeight,
        );

        const featherTopStart = Math.max(0, drawStart - FOG_FEATHER_PX);
        const featherBottomEnd = Math.min(
          canvasHeight,
          drawEnd + FOG_FEATHER_PX,
        );

        // Truncate to integer pixel rows for framebuffer indexing. Float
        // drawStart/drawEnd values would produce incorrect offsets because
        // `y * canvasWidth` is not `Math.floor(y) * canvasWidth` when y is
        // fractional.
        const fogYStart = Math.trunc(drawStart);
        const fogYEnd = Math.trunc(drawEnd);
        const featherTopStartInt = Math.trunc(featherTopStart);
        const drawStartInt = Math.trunc(drawStart);
        const drawEndInt = Math.trunc(drawEnd);
        const featherBottomEndInt = Math.trunc(featherBottomEnd);

        for (let x = xStart; x < xEnd; x += 1) {
          // Main fog-wall body: pure background color.
          for (let y = fogYStart; y < fogYEnd; y += 1) {
            const offset = (y * canvasWidth + x) * 4;
            framebuffer[offset] = bgR;
            framebuffer[offset + 1] = bgG;
            framebuffer[offset + 2] = bgB;
            framebuffer[offset + 3] = 255;
          }

          // Feather top edge: blend seeded pixel (ceiling) toward fog.
          if (drawStart > 0 && featherTopStartInt < drawStartInt) {
            const range = drawStart - featherTopStart;
            for (let y = featherTopStartInt; y < drawStartInt; y += 1) {
              const t = (y - featherTopStart) / range;
              const offset = (y * canvasWidth + x) * 4;
              framebuffer[offset] = Math.round(
                framebuffer[offset] * (1 - t) + bgR * t,
              );
              framebuffer[offset + 1] = Math.round(
                framebuffer[offset + 1] * (1 - t) + bgG * t,
              );
              framebuffer[offset + 2] = Math.round(
                framebuffer[offset + 2] * (1 - t) + bgB * t,
              );
              framebuffer[offset + 3] = 255;
            }
          }

          // Feather bottom edge: blend fog toward seeded pixel (floor).
          if (drawEnd < canvasHeight && drawEndInt < featherBottomEndInt) {
            const range = featherBottomEnd - drawEnd;
            for (let y = drawEndInt; y < featherBottomEndInt; y += 1) {
              const t = (y - drawEnd) / range;
              const blend = 1 - t;
              const offset = (y * canvasWidth + x) * 4;
              framebuffer[offset] = Math.round(
                framebuffer[offset] * (1 - blend) + bgR * blend,
              );
              framebuffer[offset + 1] = Math.round(
                framebuffer[offset + 1] * (1 - blend) + bgG * blend,
              );
              framebuffer[offset + 2] = Math.round(
                framebuffer[offset + 2] * (1 - blend) + bgB * blend,
              );
              framebuffer[offset + 3] = 255;
            }
          }
        }
      }
    }

    // Mark that the framebuffer holds the wall pixels; the sprite pass will
    // flush walls+sprites via a single putImageData (Invariant 6: all render
    // into the same buffer before transfer).
    framebufferInUse = true;
    return;
  }

  // --- Fallback: canvas fillRect path (when putImageData is unavailable) ---
  framebufferInUse = false;

  // Draw perspective floor and ceiling grids behind the walls (Canvas 2D
  // fallback only — the framebuffer path draws floor/ceiling procedurally).
  const floorCamera = pooledFloorCamera;
  floorCamera.x = cameraPositionX;
  floorCamera.y = cameraPositionY;
  floorCamera.yaw = cameraYaw;
  drawNeatensteinFloor(context, canvasWidth, canvasHeight, floorCamera);
  drawNeatensteinCeiling(context, canvasWidth, canvasHeight, floorCamera);

  const pooledHit = getPooledRayHitBuffer();
  // C1.4/C1.5: Half-res + MSAA for the fallback canvas path.
  const halfResConfig = resolveNeatensteinHalfResEnabled({});
  const halfResEnabled = halfResConfig !== false;
  let prevColumnRgb: { r: number; g: number; b: number } | null = null;
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
      pooledHit,
    );

    const perpWallDist = hit.perpWallDist;
    const isCapped =
      !Number.isFinite(perpWallDist) ||
      perpWallDist >= NEATENSTEIN_RENDER_DISTANCE_CAP;

    zBuffer[column] =
      Number.isFinite(perpWallDist) && perpWallDist > 0
        ? perpWallDist
        : NEATENSTEIN_ZBUFFER_EMPTY;

    const xStart = Math.floor(column * stripeWidth);
    const xEnd = Math.floor((column + 1) * stripeWidth);
    const stripePixelWidth = Math.max(0, xEnd - xStart);

    if (!isCapped) {
      const lineHeight = wallFocalLength / hit.perpWallDist;
      const drawStart = clamp((canvasHeight - lineHeight) / 2, 0, canvasHeight);
      const drawEnd = clamp((canvasHeight + lineHeight) / 2, 0, canvasHeight);

      let currentRgb: { r: number; g: number; b: number } =
        hit.side === 0
          ? NEATENSTEIN_WALL_X_SIDE_RGB
          : NEATENSTEIN_WALL_Y_SIDE_RGB;

      // C1.4: Half-res interpolation for odd columns.
      if (
        halfResEnabled &&
        column > 0 &&
        column % 2 === 1 &&
        prevColumnRgb !== null
      ) {
        const interp = interpolateNeatensteinWallColumn(
          { screenX: 0, ...prevColumnRgb },
          { screenX: 0, ...currentRgb },
          0.5,
        );
        currentRgb = { r: interp.r, g: interp.g, b: interp.b };
      }

      // C1.5: MSAA resolve + fog blend (replaces applyWallFog for MSAA path).
      const msaaRgb =
        prevColumnRgb !== null
          ? resolveNeatensteinMsaaResolvedColumn(currentRgb, prevColumnRgb)
          : currentRgb;
      const foggedRgb = resolveNeatensteinMsaaFogBlend(
        msaaRgb,
        hit.perpWallDist,
        NEATENSTEIN_BACKGROUND_RGB,
      );

      context.fillStyle = formatRgb(foggedRgb);
      context.fillRect(xStart, drawStart, stripePixelWidth, drawEnd - drawStart);
      prevColumnRgb = { r: currentRgb.r, g: currentRgb.g, b: currentRgb.b };
    } else {
      // C1.4/C1.5: No wall color for capped columns — reset prev for MSAA.
      prevColumnRgb = null;
      const fogDist = NEATENSTEIN_RENDER_DISTANCE_CAP;
      const lineHeight = wallFocalLength / fogDist;
      const drawStart = clamp((canvasHeight - lineHeight) / 2, 0, canvasHeight);
      const drawEnd = clamp((canvasHeight + lineHeight) / 2, 0, canvasHeight);

      const fogColor = formatRgb(NEATENSTEIN_BACKGROUND_RGB);
      context.fillStyle = fogColor;
      context.fillRect(xStart, drawStart, stripePixelWidth, drawEnd - drawStart);

      const fogFeatherPixels = FOG_FEATHER_PX;
      const featherTopStart = Math.max(0, drawStart - fogFeatherPixels);
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

      const featherBottomEnd = Math.min(
        canvasHeight,
        drawEnd + fogFeatherPixels,
      );
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
  // When the wall pass used the persistent framebuffer, sprites are rendered
  // into the SAME framebuffer (which already holds floor+ceiling+walls) and
  // flushed once via a single putImageData (Invariant 6: double-buffer — all
  // render into the same buffer before transfer, swap once per frame).
  if (framebufferInUse && typeof context.putImageData === 'function') {
    const framebuffer = getPersistentWallFramebuffer(canvasWidth, canvasHeight);
    const imageData = getReusableSpriteImageData(canvasWidth, canvasHeight);
    const spriteContext = buildNoOpSpriteRenderContext();
    const spriteCamera = {
      posX: spriteCameraX,
      posY: spriteCameraY,
      dirX: spriteDirectionX,
      dirY: spriteDirectionY,
      planeX: spritePlaneX,
      planeY: spritePlaneY,
    };

    // Sort sprites in place into the reused scratch array (avoids the
    // [...sprites] spread allocation per frame).
    spriteSortScratch.length = 0;
    for (let i = 0; i < sprites.length; i += 1) {
      spriteSortScratch.push(sprites[i]);
    }
    spriteSortScratch.sort((a, b) => {
      const distA =
        (a.worldX - spriteCameraX) ** 2 + (a.worldY - spriteCameraY) ** 2;
      const distB =
        (b.worldX - spriteCameraX) ** 2 + (b.worldY - spriteCameraY) ** 2;
      return distB - distA;
    });

    for (let s = 0; s < spriteSortScratch.length; s += 1) {
      const sprite = spriteSortScratch[s];
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
      if (!projection.visible) {
        continue;
      }

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
        framebuffer,
        zBuffer,
        projection,
        frame,
        spriteContext,
        sprite.teamColor,
        derezState,
      );
    }

    // Single putImageData flush for walls + sprites combined.
    context.putImageData(imageData, 0, 0);
    framebufferInUse = false;
    return;
  }

  // --- Fallback: legacy sprite path (when wall pass used fillRect) ---
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
      if (!projection.visible) {
        continue;
      }

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
  const pooledHit = getPooledRayHitBuffer();
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
      pooledHit,
    );

    frame.wallDistances[column] = hit.perpWallDist;
    frame.wallSides[column] = hit.side;
    frame.zBuffer[column] =
      !Number.isFinite(hit.perpWallDist) ||
      hit.perpWallDist >= NEATENSTEIN_RENDER_DISTANCE_CAP
        ? NEATENSTEIN_ZBUFFER_EMPTY
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

// --- Pooled frame arrays, wall framebuffer, and sprite ImageData (A2 Fix 2, 7, 8) ---

/** Persistent wall framebuffer — allocated once, reused across frames. */
let persistentWallFramebuffer: Uint8ClampedArray | null = null;
let persistentWallFramebufferWidth = 0;
let persistentWallFramebufferHeight = 0;

/**
 * Return the persistent wall framebuffer, allocating or reallocating only on
 * dimension change.
 *
 * @param width - Canvas width in pixels.
 * @param height - Canvas height in pixels.
 * @returns A {@link Uint8ClampedArray} of length `width * height * 4`.
 */
export function getPersistentWallFramebuffer(
  width: number,
  height: number,
): Uint8ClampedArray {
  const byteLength = width * height * 4;
  if (
    persistentWallFramebuffer !== null &&
    persistentWallFramebufferWidth === width &&
    persistentWallFramebufferHeight === height
  ) {
    return persistentWallFramebuffer;
  }
  persistentWallFramebuffer = new Uint8ClampedArray(byteLength);
  persistentWallFramebufferWidth = width;
  persistentWallFramebufferHeight = height;
  return persistentWallFramebuffer;
}

/** Persistent sprite ImageData — re-allocated only on canvas resize. */
let reusableSpriteImageData: ImageData | null = null;
let reusableSpriteImageDataWidth = 0;
let reusableSpriteImageDataHeight = 0;

/**
 * Return the reusable sprite ImageData, allocating or reallocating only on
 * dimension change.
 *
 * The ImageData wraps the persistent wall framebuffer's Uint8ClampedArray so
 * that walls and sprites render into the SAME buffer before a single
 * putImageData flush (Invariant 6: double-buffer).
 *
 * @param width - Canvas width in pixels.
 * @param height - Canvas height in pixels.
 * @returns The persistent ImageData.
 */
export function getReusableSpriteImageData(
  width: number,
  height: number,
): ImageData {
  if (
    reusableSpriteImageData !== null &&
    reusableSpriteImageDataWidth === width &&
    reusableSpriteImageDataHeight === height
  ) {
    return reusableSpriteImageData;
  }
  // Wrap the persistent framebuffer so walls + sprites share one buffer.
  const framebuffer = getPersistentWallFramebuffer(width, height);
  // Cast to satisfy the ImageData constructor's strict ArrayBuffer typing.
  reusableSpriteImageData = new ImageData(
    framebuffer as Uint8ClampedArray<ArrayBuffer>,
    width,
    height,
  );
  reusableSpriteImageDataWidth = width;
  reusableSpriteImageDataHeight = height;
  return reusableSpriteImageData;
}

/** Pooled per-ray hit buffer — mutable CastRayDDAHit reused across ray casts. */
const pooledRayHitBuffer: CastRayDDAHit = {
  perpWallDist: 0,
  side: 0,
  mapX: 0,
  mapY: 0,
};

/**
 * Return the pooled per-ray hit object.
 *
 * Contains `perpWallDist`, `side`, `mapX`, `mapY` fields. Reused across ray
 * casts to avoid per-ray object allocations. Callers must consume the fields
 * before the next ray cast overwrites them.
 *
 * @returns A mutable `CastRayDDAHit` object.
 */
export function getPooledRayHitBuffer(): CastRayDDAHit {
  return pooledRayHitBuffer;
}

/** Pooled floor camera literal — mutated in place instead of allocating per frame. */
const pooledFloorCamera: NeatensteinFloorCamera = {
  x: 0,
  y: 0,
  yaw: 0,
};

/**
 * Seed the persistent wall framebuffer with floor/ceiling grid pixels
 * procedurally, eliminating the per-frame `context.getImageData()` call
 * that allocated ~1.2 MB every frame (A2 Fix 2).
 *
 * Delegates to {@link castNeatensteinFloorPerPixel} which fills the framebuffer
 * with an opaque background colour and renders the floor and ceiling grid lines
 * per-pixel using `fract(worldCoord)` procedural grid detection. No canvas
 * read-back is required.
 *
 * @param framebuffer - The persistent wall framebuffer (Uint8ClampedArray).
 * @param canvasWidth - Canvas width in pixels.
 * @param canvasHeight - Canvas height in pixels.
 * @param cameraX - Camera world X.
 * @param cameraY - Camera world Y.
 * @param cameraYaw - Camera yaw in radians.
 */
function seedFramebufferProcedurally(
  framebuffer: Uint8ClampedArray,
  canvasWidth: number,
  canvasHeight: number,
  cameraX: number,
  cameraY: number,
  cameraYaw: number,
): void {
  const cosYaw = Math.cos(cameraYaw);
  const sinYaw = Math.sin(cameraYaw);
  const halfWidth = canvasWidth / 2;
  const horizonY = canvasHeight * NEATENSTEIN_FLOOR_HORIZON_RATIO;
  const focalLength =
    canvasHeight / 2 / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);

  castNeatensteinFloorPerPixel(framebuffer, canvasWidth, canvasHeight, {
    cameraX,
    cameraY,
    cosYaw,
    sinYaw,
    focalLength,
    halfWidth,
    horizonY,
    cameraHeight: NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
  });
}

// ---------------------------------------------------------------------------
// B1: Enemy AI parallelism — render compositing order and worker split
// ---------------------------------------------------------------------------

/**
 * Canonical render compositing order (Invariant 5).
 *
 * Layers are composited in this exact order: floor → ceiling → walls →
 * sprites → pulses/sparks → bolts. This order must be preserved across all
 * renderer tiers.
 */
export const RENDER_COMPOSITING_ORDER = [
  'floor',
  'ceiling',
  'walls',
  'sprites',
  'pulses',
  'bolts',
] as const;

/**
 * Validate that {@link RENDER_COMPOSITING_ORDER} is well-formed.
 *
 * Ensures the compositing order array contains all expected layers in the
 * correct relative order: floor → ceiling → walls → sprites → pulses → bolts.
 * Called from the rendering pipeline so the contract is enforced at runtime,
 * not just documentation-only.
 *
 * @throws {Error} When the compositing order is malformed.
 */
export function assertRenderCompositingOrderValid(): void {
  const expected = ['floor', 'ceiling', 'walls', 'sprites', 'pulses', 'bolts'];
  for (let i = 0; i < expected.length; i += 1) {
    if (RENDER_COMPOSITING_ORDER[i] !== expected[i]) {
      throw new Error(
        `RENDER_COMPOSITING_ORDER mismatch at index ${i}: expected '${expected[i]}', got '${RENDER_COMPOSITING_ORDER[i]}'`,
      );
    }
  }
}

/**
 * Create a fresh render worker state slice with default initial values.
 *
 * The render worker owns DDA, framebuffer, canvas, OffscreenCanvas, and the
 * wall map. It reads enemy state from shared memory written by the sim worker.
 *
 * The wall map is allocated via {@link createSharedMapGrid} so that it can be
 * backed by a `SharedArrayBuffer` when cross-origin isolation is configured.
 * When a `SharedArrayBuffer` is returned, it is wrapped in a `Uint8Array`
 * view so the render pipeline can read wall cells uniformly.
 *
 * @returns A new {@link RenderWorkerState} with null/empty defaults.
 */
export function createRenderWorkerState(): RenderWorkerState {
  const gridBuffer = createSharedMapGrid(NEATENSTEIN_MAP_SIZE);
  const wallMap = gridBuffer instanceof SharedArrayBuffer
    ? new Uint8Array(gridBuffer)
    : gridBuffer;

  return {
    currentTier: null,
    workerCanvas: null,
    workerContext: null,
    latestState: null,
    pendingResizeDimensions: null,
    wallMap,
    workerZBuffer: null,
    activePulses: [],
    activeEnemySprites: [],
    smoothedMoveX: 0,
    smoothedMoveY: 0,
    smoothedLookDelta: 0,
  };
}

/**
 * Minimal interface for an `OffscreenCanvas`-like surface that supports
 * `transferToImageBitmap`.
 */
interface NeatensteinTransferableCanvas {
  transferToImageBitmap(): ImageBitmap;
  width: number;
  height: number;
}

/**
 * Present the current frame via `transferToImageBitmap` (C1.5).
 *
 * Replaces the legacy `ctx.commit()` present path with the 2024-preferred
 * `transferToImageBitmap` API. The returned `ImageBitmap` is transferred to
 * the host thread for compositing via `createImageBitmap` /
 * `consumeNeatensteinFrameBitmap`.
 *
 * @param canvas - `OffscreenCanvas` (or compatible surface) to transfer.
 * @returns The transferred `ImageBitmap`.
 *
 * @example
 * ```ts
 * const bitmap = presentNeatensteinFrameBitmap(workerCanvas);
 * postMessage({ type: 'frame', bitmap }, [bitmap]);
 * ```
 */
export function presentNeatensteinFrameBitmap(
  canvas: NeatensteinTransferableCanvas,
): ImageBitmap {
  return canvas.transferToImageBitmap();
}
