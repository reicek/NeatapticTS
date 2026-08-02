/**
 * Pure gun overlay renderer for the Neatenstein neon raycasting demo.
 *
 * The weapon is drawn as a center-screen DOOM-style plasma cannon: a wide
 * Neon White body with a teal energy strip, anchored at the bottom center of
 * the viewport and kicked upward by the current screen-space recoil offset.
 * All rendering is pure side effects on the supplied 2D context.
 *
 * @module
 */

import {
  NEATENSTEIN_GUN_ACCENT_COLOR as GUN_ACCENT_COLOR,
  NEATENSTEIN_GUN_BODY_COLOR as GUN_BODY_COLOR,
} from '../constants';
import type { GunState } from '../host/game/types';
import { GUN_BARREL_VOXEL_GRID, projectGunSprite } from './gun-sprite';

/**
 * CSS color applied to the gun body overlay.
 *
 * A warm near-white ("Neon White") so the weapon reads as painted plastic or
 * ceramic against the dark raycast scene.
 */
export const NEATENSTEIN_GUN_BODY_COLOR = GUN_BODY_COLOR;

/**
 * CSS color applied to gun accent lines and highlights.
 *
 * A bright teal used for energy strips, sight dots, and the matching dynamic
 * light overlay.
 */
export const NEATENSTEIN_GUN_ACCENT_COLOR = GUN_ACCENT_COLOR;

/**
 * Fraction of viewport height occupied by the gun body.
 */
const GUN_BODY_HEIGHT_FRACTION = 0.22;

/**
 * Gun body aspect ratio (width / height).
 *
 * Derived from the chunky DOOM plasma-cannon reference silhouette so the
 * weapon stays square and readable regardless of viewport width.
 */
export const GUN_BODY_ASPECT_RATIO = 0.75;

/**
 * Create the canonical initial {@link GunState} for a fresh episode.
 *
 * The recoil offset starts at zero so the overlay rests in its idle position.
 *
 * @returns A fresh {@link GunState} with zero recoil.
 *
 * @example
 * ```ts
 * const gun = createInitialGunState();
 * expect(gun.recoilOffset).toBe(0);
 * ```
 */
export function createInitialGunState(): GunState {
  return { recoilOffset: 0 };
}

/**
 * Render the gun overlay on top of the raycast frame.
 *
 * The gun is anchored at the bottom center of the viewport. When
 * {@link GunState.recoilOffset} is non-zero, the entire overlay is translated
 * upward by that amount before drawing, producing a visible kick that settles
 * back to the idle position as the recoil decays.
 *
 * The sprite is drawn as a chunky DOOM-style plasma cannon: a tapered Neon
 * White body, a glowing central energy core, and teal accent bolts.
 *
 * @param ctx - 2D canvas context to draw into.
 * @param gun - Current weapon overlay state.
 * @param width - Viewport width in CSS pixels.
 * @param height - Viewport height in CSS pixels.
 *
 * @example
 * ```ts
 * const gun = createInitialGunState();
 * renderGunOverlay(ctx, gun, 640, 360);
 * ```
 */
export function renderGunOverlay(
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
  gun: GunState,
  width: number,
  height: number,
): void {
  const centerX = width / 2;
  const gunBottomY = height;
  const gunHeight = height * GUN_BODY_HEIGHT_FRACTION;
  const gunWidth = gunHeight * GUN_BODY_ASPECT_RATIO;
  const gunTop = gunBottomY - gunHeight;

  ctx.save();
  if (gun.recoilOffset !== 0) {
    ctx.translate(0, -gun.recoilOffset);
  }

  // Tapered main chassis with a left-to-right plastic shading gradient.
  const baseHalfWidth = gunWidth / 2;
  const topHalfWidth = gunWidth * 0.33;

  const chassisGradient = ctx.createLinearGradient(
    centerX - baseHalfWidth,
    gunTop,
    centerX + baseHalfWidth,
    gunTop,
  );
  chassisGradient.addColorStop(0, '#c8d4d8');
  chassisGradient.addColorStop(0.25, '#eefcfd');
  chassisGradient.addColorStop(0.5, '#ffffff');
  chassisGradient.addColorStop(0.75, '#eefcfd');
  chassisGradient.addColorStop(1, '#b8c4c8');

  ctx.fillStyle = chassisGradient;
  ctx.beginPath();
  ctx.moveTo(centerX - baseHalfWidth, gunBottomY);
  ctx.lineTo(centerX - topHalfWidth, gunTop);
  ctx.lineTo(centerX + topHalfWidth, gunTop);
  ctx.lineTo(centerX + baseHalfWidth, gunBottomY);
  ctx.closePath();
  ctx.fill();

  // Left and right 3D side planes so the cannon reads as an extruded voxel
  // body rather than a flat gradient shape.
  const sidePanelInset = baseHalfWidth * 0.55;
  const topSideInset = topHalfWidth * 0.45;

  // Darker left face.
  ctx.fillStyle = '#8fa0a5';
  ctx.beginPath();
  ctx.moveTo(centerX - baseHalfWidth, gunBottomY);
  ctx.lineTo(centerX - topHalfWidth, gunTop);
  ctx.lineTo(centerX - topHalfWidth + topSideInset, gunTop);
  ctx.lineTo(centerX - baseHalfWidth + sidePanelInset, gunBottomY);
  ctx.closePath();
  ctx.fill();

  // Lighter right face.
  ctx.fillStyle = '#d8eef2';
  ctx.beginPath();
  ctx.moveTo(centerX + baseHalfWidth, gunBottomY);
  ctx.lineTo(centerX + topHalfWidth, gunTop);
  ctx.lineTo(centerX + topHalfWidth - topSideInset, gunTop);
  ctx.lineTo(centerX + baseHalfWidth - sidePanelInset, gunBottomY);
  ctx.closePath();
  ctx.fill();

  // Accent outline around the chassis.
  ctx.lineWidth = Math.max(1, width * 0.002);
  ctx.strokeStyle = NEATENSTEIN_GUN_ACCENT_COLOR;
  ctx.stroke();

  // Central plasma core, tapered to match the gun body's perspective.
  const coreWidthTop = gunWidth * 0.16;
  const coreWidthBottom = gunWidth * 0.3;
  const coreHeight = gunHeight * 0.72;
  const coreTop = gunTop + gunHeight * 0.08;
  const coreBottom = coreTop + coreHeight;
  const coreLeftTop = centerX - coreWidthTop / 2;
  const coreRightTop = centerX + coreWidthTop / 2;
  const coreLeftBottom = centerX - coreWidthBottom / 2;
  const coreRightBottom = centerX + coreWidthBottom / 2;
  const coreGradient = ctx.createLinearGradient(
    coreLeftBottom,
    coreTop,
    coreRightBottom,
    coreTop,
  );
  coreGradient.addColorStop(0, '#008b9a');
  coreGradient.addColorStop(0.4, NEATENSTEIN_GUN_ACCENT_COLOR);
  coreGradient.addColorStop(0.6, '#80f8ff');
  coreGradient.addColorStop(1, '#006b7a');

  ctx.fillStyle = coreGradient;
  ctx.beginPath();
  ctx.moveTo(coreLeftBottom, coreBottom);
  ctx.lineTo(coreLeftTop, coreTop);
  ctx.lineTo(coreRightTop, coreTop);
  ctx.lineTo(coreRightBottom, coreBottom);
  ctx.closePath();
  ctx.fill();

  // Lower teal accent bolts on each side.
  ctx.fillStyle = NEATENSTEIN_GUN_ACCENT_COLOR;
  for (const side of [-1, 1] as const) {
    for (const ridgeY of [gunTop + gunHeight * 0.55 + height * 0.015]) {
      ctx.beginPath();
      ctx.arc(
        centerX + side * baseHalfWidth * 0.84,
        ridgeY,
        width * 0.004,
        0,
        Math.PI * 2,
      );
      ctx.fill();
    }
  }

  // Raised barrel band near the muzzle.
  ctx.fillStyle = '#c8d4d8';
  const bandY = gunTop + gunHeight * 0.18;
  const bandHeight = gunHeight * 0.05;
  const bandWidth = gunWidth * 0.92;
  ctx.fillRect(centerX - bandWidth / 2, bandY, bandWidth, bandHeight);

  // Dark side-vent slits on the lower chassis.
  ctx.fillStyle = '#4a5a5e';
  const ventWidth = gunWidth * 0.08;
  const ventHeight = gunHeight * 0.08;
  const ventY = gunTop + gunHeight * 0.62;
  for (const side of [-1, 1] as const) {
    ctx.fillRect(
      centerX + side * baseHalfWidth * 0.72 - ventWidth / 2,
      ventY,
      ventWidth,
      ventHeight,
    );
  }

  // Teal top sight post above the muzzle.
  ctx.fillStyle = NEATENSTEIN_GUN_ACCENT_COLOR;
  const sightWidth = gunWidth * 0.14;
  const sightHeight = gunHeight * 0.06;
  ctx.fillRect(
    centerX - sightWidth / 2,
    gunTop - sightHeight * 0.7,
    sightWidth,
    sightHeight,
  );

  // Project a small 3D voxel barrel above the main chassis.
  const barrelScale = gunHeight * 0.05;
  const barrelBaseY = gunTop - gunHeight * 0.02;
  const projectedVoxels = projectGunSprite({
    voxelGrid: GUN_BARREL_VOXEL_GRID,
    screenX: centerX,
    screenY: barrelBaseY,
    scale: barrelScale,
  });
  for (const voxel of projectedVoxels) {
    ctx.fillStyle = voxel.color;
    ctx.fillRect(
      voxel.screenX - voxel.size / 2,
      voxel.screenY - voxel.size / 2,
      voxel.size,
      voxel.size,
    );
  }

  ctx.restore();
}
