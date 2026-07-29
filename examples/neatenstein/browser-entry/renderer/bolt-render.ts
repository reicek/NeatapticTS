/**
 * Bolt and impact-spot rendering for the Neatenstein worker tier.
 *
 * This module contains the pure drawing helpers used by the display worker to
 * render traveling plasma bolts and the wall-impact spots they create. Keeping
 * them in a standalone renderer file lets both the worker entrypoint and unit
 * tests import them without evaluating the worker's top-level `self.onmessage`
 * handler.
 *
 * @module
 */

import {
  NEATENSTEIN_GUN_ACCENT_COLOR,
  NEATENSTEIN_IMPACT_SPOT_COLOR,
  NEATENSTEIN_IMPACT_SPOT_GLOW_BLUR_PX,
  NEATENSTEIN_IMPACT_SPOT_GLOW_COLOR,
  NEATENSTEIN_IMPACT_SPOT_LIFETIME_MS,
  NEATENSTEIN_IMPACT_SPOT_RADIUS_PX,
} from '../constants';
import {
  NEATENSTEIN_BOLT_MAX_RANGE_CELLS,
  NEATENSTEIN_BOLT_TRAVEL_DURATION_MS,
} from '../host/game/constants';
import type { BoltState, ImpactSpot } from '../host/game/types';
import {
  NEATENSTEIN_FLOOR_FOV_RADIANS,
  NEATENSTEIN_FLOOR_HORIZON_RATIO,
  projectNeatensteinFloorPoint,
  type NeatensteinFloorCamera,
} from './floor';
import { depthTestPulse } from './pulse';

/**
 * Clamp a numeric value to the inclusive range [min, max].
 *
 * @param value - Value to clamp.
 * @param min - Lower bound.
 * @param max - Upper bound.
 * @returns Clamped value.
 */
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Camera-height offset used when projecting plasma bolts to screen space.
 *
 * A value of zero places the bolt on the horizon, so airborne projectiles
 * read as center-screen shots from the DOOM-style gun overlay rather than
 * floor-plane decals.
 */
const BOLT_PROJECTED_CAMERA_HEIGHT_WORLD = 0;

/**
 * Screen-space ratio for the plasma-cannon muzzle anchor.
 *
 * Bolts are interpolated from a point just below the gun barrel tip so the
 * projectile visibly leaves the weapon and travels toward the projected
 * target.
 */
const BOLT_MUZZLE_SCREEN_X_RATIO = 0.5;
const BOLT_MUZZLE_SCREEN_Y_RATIO = 0.82;

/**
 * Screen-space plasma bolt radius at the muzzle.
 *
 * Twice the previous 4.5 px bolt radius so the shot reads as a chunky
 * plasma projectile as it leaves the gun.
 */
const NEATENSTEIN_BOLT_MUZZLE_SCREEN_RADIUS_PX = 9;

/**
 * Screen-space plasma bolt radius at maximum range.
 *
 * At 30 cells the bolt must shrink to a single screen pixel in diameter.
 */
const NEATENSTEIN_BOLT_MIN_SCREEN_RADIUS_PX = 0.5;

/**
 * Maximum opacity of a freshly spawned plasma bolt.
 */
const NEATENSTEIN_BOLT_MAX_SCREEN_ALPHA = 0.95;

/**
 * Fraction of the outer bolt radius occupied by the bright inner core.
 */
const NEATENSTEIN_BOLT_CORE_RADIUS_RATIO = 0.6;

/**
 * Draw active wall-impact neon spots in the worker tier.
 *
 * Each spot is rendered only after the plasma bolt that created it has reached
 * the wall (`travelRatio >= 1`). Until then the spot stays invisible so the
 * impact does not appear to happen before the projectile arrives.
 *
 * @param context - Worker-tier 2D canvas context.
 * @param impacts - Active wall-impact list.
 * @param zBuffer - Per-column depth buffer.
 * @param camera - Camera position and yaw.
 * @param canvasWidth - Canvas width.
 * @param canvasHeight - Canvas height.
 * @param simTimeMs - Current simulation time in milliseconds.
 */
export function drawImpactSpots(
  context: OffscreenCanvasRenderingContext2D,
  impacts: readonly ImpactSpot[],
  zBuffer: Float32Array,
  camera: NeatensteinFloorCamera,
  canvasWidth: number,
  canvasHeight: number,
  simTimeMs: number,
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
    const elapsedMs = simTimeMs - impact.createdAtMs;
    const travelTimeMs = impact.boltTravelTimeMs;
    const travelRatio =
      travelTimeMs > 0 ? clamp(elapsedMs / travelTimeMs, 0, 1) : 1;

    if (travelRatio < 1) {
      continue;
    }

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
 * Draw active plasma bolts as traveling glowing projectiles.
 *
 * Each bolt is interpolated on screen from the gun muzzle toward its projected
 * impact point. The visual travel duration is fixed so every bolt moves at the
 * same screen-space speed regardless of target distance.
 *
 * @param context - Worker-tier 2D canvas context.
 * @param bolts - Active bolt list.
 * @param camera - Camera position and yaw.
 * @param canvasWidth - Canvas width.
 * @param canvasHeight - Canvas height.
 * @param simTimeMs - Current simulation time in milliseconds.
 */
export function drawBolts(
  context: OffscreenCanvasRenderingContext2D,
  bolts: readonly BoltState[],
  camera: NeatensteinFloorCamera,
  canvasWidth: number,
  canvasHeight: number,
  simTimeMs: number,
): void {
  if (bolts.length === 0) {
    return;
  }

  const safeX = Number.isFinite(camera.x) ? camera.x : 0;
  const safeY = Number.isFinite(camera.y) ? camera.y : 0;
  const safeYaw = Number.isFinite(camera.yaw) ? camera.yaw : 0;

  const horizonY = canvasHeight * NEATENSTEIN_FLOOR_HORIZON_RATIO;
  const halfWidth = canvasWidth / 2;
  const focalLength = halfWidth / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
  const cosYaw = Math.cos(safeYaw);
  const sinYaw = Math.sin(safeYaw);

  const muzzleScreenX = canvasWidth * BOLT_MUZZLE_SCREEN_X_RATIO;
  const muzzleScreenY = canvasHeight * BOLT_MUZZLE_SCREEN_Y_RATIO;

  const savedComposite = context.globalCompositeOperation;
  context.globalCompositeOperation = 'lighter';

  for (const bolt of bolts) {
    if (!bolt.active) {
      continue;
    }

    const projectedCurrent = projectNeatensteinFloorPoint(
      bolt.position.x,
      bolt.position.y,
      safeX,
      safeY,
      cosYaw,
      sinYaw,
      focalLength,
      halfWidth,
      horizonY,
      canvasHeight,
      BOLT_PROJECTED_CAMERA_HEIGHT_WORLD,
    );

    if (projectedCurrent === null) {
      continue;
    }

    let targetX = projectedCurrent.x;
    let targetY = projectedCurrent.y;

    if (
      bolt.origin &&
      bolt.direction &&
      typeof bolt.targetDistance === 'number' &&
      bolt.targetDistance > 0
    ) {
      const targetWorldX =
        bolt.origin.x + bolt.direction.x * bolt.targetDistance;
      const targetWorldY =
        bolt.origin.y + bolt.direction.y * bolt.targetDistance;
      const projectedTarget = projectNeatensteinFloorPoint(
        targetWorldX,
        targetWorldY,
        safeX,
        safeY,
        cosYaw,
        sinYaw,
        focalLength,
        halfWidth,
        horizonY,
        canvasHeight,
        BOLT_PROJECTED_CAMERA_HEIGHT_WORLD,
      );

      if (projectedTarget !== null) {
        targetX = projectedTarget.x;
        targetY = projectedTarget.y;
      }
    }

    const safeSimTimeMs = Number.isFinite(simTimeMs) ? simTimeMs : 0;
    const safeCreatedAtMs = Number.isFinite(bolt.createdAtMs)
      ? bolt.createdAtMs
      : safeSimTimeMs;
    const elapsedMs = safeSimTimeMs - safeCreatedAtMs;
    const travelRatio = clamp(
      elapsedMs / NEATENSTEIN_BOLT_TRAVEL_DURATION_MS,
      0,
      1,
    );

    const screenX = muzzleScreenX + (targetX - muzzleScreenX) * travelRatio;
    const screenY = muzzleScreenY + (targetY - muzzleScreenY) * travelRatio;

    const distanceTraveled =
      bolt.origin &&
      Number.isFinite(bolt.origin.x) &&
      Number.isFinite(bolt.origin.y)
        ? Math.hypot(
            bolt.position.x - bolt.origin.x,
            bolt.position.y - bolt.origin.y,
          )
        : 0;

    const safeTargetDistance =
      typeof bolt.targetDistance === 'number' &&
      Number.isFinite(bolt.targetDistance) &&
      bolt.targetDistance > 0
        ? bolt.targetDistance
        : distanceTraveled;

    const fadeRatio = clamp(
      travelRatio * (safeTargetDistance / NEATENSTEIN_BOLT_MAX_RANGE_CELLS),
      0,
      1,
    );

    if (fadeRatio >= 1) {
      continue;
    }

    const boltRadius =
      NEATENSTEIN_BOLT_MUZZLE_SCREEN_RADIUS_PX * (1 - fadeRatio) +
      NEATENSTEIN_BOLT_MIN_SCREEN_RADIUS_PX * fadeRatio;
    const boltAlpha = NEATENSTEIN_BOLT_MAX_SCREEN_ALPHA * (1 - fadeRatio);
    const coreRadius = Math.max(
      0,
      boltRadius * NEATENSTEIN_BOLT_CORE_RADIUS_RATIO,
    );

    context.shadowColor = NEATENSTEIN_GUN_ACCENT_COLOR;
    context.shadowBlur = Math.max(2, boltRadius * 2);
    context.fillStyle = NEATENSTEIN_GUN_ACCENT_COLOR;
    context.globalAlpha = boltAlpha;
    context.beginPath();
    context.arc(screenX, screenY, Math.max(0, boltRadius), 0, Math.PI * 2);
    context.fill();

    context.shadowBlur = 0;
    context.fillStyle = '#ffffff';
    context.globalAlpha = boltAlpha;
    context.beginPath();
    context.arc(screenX, screenY, coreRadius, 0, Math.PI * 2);
    context.fill();
  }

  context.globalAlpha = 1;
  context.shadowBlur = 0;
  context.globalCompositeOperation = savedComposite;
}
