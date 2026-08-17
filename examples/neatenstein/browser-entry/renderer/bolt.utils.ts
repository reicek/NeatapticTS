/**
 * Pure bolt-rendering executors extracted from {@link module:./bolt-render}.
 *
 * This module owns the per-item render logic, projection-context resolvers,
 * and shared drawing helpers used by the five declarative draw orchestrators
 * in the facade module. Every exported function is a pure leaf executor that
 * accepts only scalar parameters and existing object references — no
 * per-iteration allocations are introduced.
 *
 * @module
 */

import {
  NEATENSTEIN_AMMO_PICKUP_COLOR,
  NEATENSTEIN_AMMO_PICKUP_GLOW_BLUR_PX,
  NEATENSTEIN_AMMO_PICKUP_GLOW_COLOR,
  NEATENSTEIN_AMMO_PICKUP_RADIUS_PX,
  NEATENSTEIN_ENEMY_IMPACT_BURST_DURATION_MS,
  NEATENSTEIN_ENEMY_IMPACT_BURST_RADIUS_PX,
  NEATENSTEIN_ENEMY_IMPACT_COLOR,
  NEATENSTEIN_ENEMY_IMPACT_GLOW_BLUR_PX,
  NEATENSTEIN_ENEMY_IMPACT_GLOW_COLOR,
  NEATENSTEIN_ENEMY_IMPACT_LIFETIME_MS,
  NEATENSTEIN_ENEMY_IMPACT_RADIUS_PX,
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
  NEATENSTEIN_ENEMY_BOLT_LIFETIME_MS,
  NEATENSTEIN_ENEMY_BOLT_MAX_RANGE_CELLS,
} from '../host/game/constants';
import type {
  AmmoPickupState,
  BoltState,
  EnemyBoltState,
  EnemyImpactSpot,
  ImpactSpot,
} from '../host/game/types';
import {
  NEATENSTEIN_FLOOR_FOV_RADIANS,
  NEATENSTEIN_FLOOR_HORIZON_RATIO,
} from './renderer.floor.constants';
import { projectNeatensteinFloorPoint } from './floor.projection.utils';
import type { NeatensteinFloorCamera } from './renderer.floor.types';
import {
  BOLT_MUZZLE_SCREEN_X_RATIO,
  BOLT_MUZZLE_SCREEN_Y_RATIO,
  BOLT_PROJECTED_CAMERA_HEIGHT_WORLD,
  COLOR_EMPTY_STRING,
  COLOR_WHITE_HEX,
  NEATENSTEIN_BOLT_CORE_RADIUS_RATIO,
  NEATENSTEIN_BOLT_EXPLOSION_FLASH_ALPHA_MULTIPLIER,
  NEATENSTEIN_BOLT_EXPLOSION_FLASH_COLOR,
  NEATENSTEIN_BOLT_EXPLOSION_FLASH_GLOW_COLOR,
  NEATENSTEIN_BOLT_EXPLOSION_FLASH_RADIUS_MULTIPLIER,
  NEATENSTEIN_BOLT_MAX_SCREEN_ALPHA,
  NEATENSTEIN_BOLT_MIN_SCREEN_RADIUS_PX,
  NEATENSTEIN_BOLT_MUZZLE_SCREEN_RADIUS_PX,
  NEATENSTEIN_ENEMY_BOLT_COLOR,
  NEATENSTEIN_ENEMY_BOLT_CORE_COLOR,
  NEATENSTEIN_ENEMY_BOLT_CORE_RADIUS_RATIO,
  NEATENSTEIN_ENEMY_BOLT_MAX_SCREEN_ALPHA,
  NEATENSTEIN_ENEMY_BOLT_MIN_SCREEN_RADIUS_PX,
  NEATENSTEIN_ENEMY_BOLT_MUZZLE_SCREEN_RADIUS_PX,
} from './renderer.bolt.constants';
import type {
  FloorProjectionContext,
  LateralProjectionContext,
  MuzzleScreenPosition,
} from './renderer.bolt.types';
import { depthTestPulse } from './pulse';

// Re-export constants and types for external consumers.
export type {
  FloorProjectionContext,
  LateralProjectionContext,
  MuzzleScreenPosition,
} from './renderer.bolt.types';

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Projection-context resolvers
// ---------------------------------------------------------------------------

/**
 * Resolve the lateral (floor-plane) projection context for a frame.
 *
 * Computes the camera direction vector, frustum plane scale, and caches the
 * canvas dimensions and camera position so per-item render functions can
 * project world points to screen space without redundant trig calls.
 *
 * @param camera - Camera position and yaw.
 * @param canvasWidth - Canvas width in pixels.
 * @param canvasHeight - Canvas height in pixels.
 * @returns Frozen lateral projection context for the frame.
 */
export function resolveLateralProjectionContext(
  camera: NeatensteinFloorCamera,
  canvasWidth: number,
  canvasHeight: number,
): LateralProjectionContext {
  return {
    dirX: Math.cos(camera.yaw),
    dirY: Math.sin(camera.yaw),
    planeScale:
      (canvasWidth / canvasHeight) *
      Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2),
    canvasWidth,
    canvasHeight,
    cameraX: camera.x,
    cameraY: camera.y,
  };
}

/**
 * Resolve the floor-perspective projection context for a frame.
 *
 * Computes the horizon line, focal length, yaw trig values, and caches the
 * canvas dimensions and camera position so per-item render functions can call
 * {@link projectNeatensteinFloorPoint} without redundant trig or division
 * per item.
 *
 * @param camera - Camera position and yaw.
 * @param canvasWidth - Canvas width in pixels.
 * @param canvasHeight - Canvas height in pixels.
 * @returns Frozen floor projection context for the frame.
 */
export function resolveFloorProjectionContext(
  camera: NeatensteinFloorCamera,
  canvasWidth: number,
  canvasHeight: number,
): FloorProjectionContext {
  return {
    horizonY: canvasHeight * NEATENSTEIN_FLOOR_HORIZON_RATIO,
    halfWidth: canvasWidth / 2,
    focalLength: canvasHeight / 2 / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2),
    cosYaw: Math.cos(camera.yaw),
    sinYaw: Math.sin(camera.yaw),
    canvasWidth,
    canvasHeight,
    cameraX: camera.x,
    cameraY: camera.y,
  };
}

/**
 * Resolve the player-bolt muzzle screen position for a frame.
 *
 * @param canvasWidth - Canvas width in pixels.
 * @param canvasHeight - Canvas height in pixels.
 * @returns Muzzle screen position.
 */
export function resolveMuzzleScreenPosition(
  canvasWidth: number,
  canvasHeight: number,
): MuzzleScreenPosition {
  return {
    x: canvasWidth * BOLT_MUZZLE_SCREEN_X_RATIO,
    y: canvasHeight * BOLT_MUZZLE_SCREEN_Y_RATIO,
  };
}

// ---------------------------------------------------------------------------
// Shared drawing helpers
// ---------------------------------------------------------------------------

/**
 * Draw a filled circle with optional neon glow shadow.
 *
 * When `glowBlur` is greater than zero, both `shadowColor` and `shadowBlur`
 * are set before the fill. When `glowBlur` is zero, only `shadowBlur` is
 * reset to zero — `shadowColor` is left untouched, matching the original
 * inline behavior for core draws that had no glow.
 *
 * The radius is clamped to `Math.max(0, radius)` internally so callers do
 * not need to pre-clamp.
 *
 * @param context - Worker-tier 2D canvas context.
 * @param x - Screen-space X coordinate of the circle center.
 * @param y - Screen-space Y coordinate of the circle center.
 * @param radius - Circle radius in pixels (clamped to ≥ 0).
 * @param fillStyle - CSS color string for the fill.
 * @param alpha - Global alpha (0–1).
 * @param glowColor - CSS color string for the glow shadow (ignored when
 *   `glowBlur` is 0).
 * @param glowBlur - Shadow blur radius in pixels (0 disables glow).
 */
export function drawGlowCircle(
  context: OffscreenCanvasRenderingContext2D,
  x: number,
  y: number,
  radius: number,
  fillStyle: string,
  alpha: number,
  glowColor: string,
  glowBlur: number,
): void {
  if (glowBlur > 0) {
    context.shadowColor = glowColor;
    context.shadowBlur = glowBlur;
  } else {
    context.shadowBlur = 0;
  }
  context.fillStyle = fillStyle;
  context.globalAlpha = alpha;
  context.beginPath();
  context.arc(x, y, Math.max(0, radius), 0, Math.PI * 2);
  context.fill();
}

/**
 * Restore a 2D canvas context to its default render state after additive
 * blending.
 *
 * Resets `globalAlpha` to 1, `shadowBlur` to 0, and `globalCompositeOperation`
 * to the saved value captured before the draw loop.
 *
 * @param context - Worker-tier 2D canvas context.
 * @param savedComposite - The `globalCompositeOperation` value to restore.
 */
export function restoreRenderContext(
  context: OffscreenCanvasRenderingContext2D,
  savedComposite: GlobalCompositeOperation,
): void {
  context.globalAlpha = 1;
  context.shadowBlur = 0;
  context.globalCompositeOperation = savedComposite;
}

// ---------------------------------------------------------------------------
// Per-item render executors
// ---------------------------------------------------------------------------

/**
 * Render a single wall-impact spot.
 *
 * The spot is rendered only after the plasma bolt that created it has reached
 * the wall (`travelRatio >= 1`). Uses the lateral (floor-plane) projection to
 * map the world-space impact position to screen space, applies a z-buffer
 * depth test, and draws a neon-glow circle whose alpha fades with remaining
 * lifetime.
 *
 * @param context - Worker-tier 2D canvas context.
 * @param impact - Wall-impact spot to render.
 * @param zBuffer - Per-column depth buffer.
 * @param latCtx - Lateral projection context for the current frame.
 * @param simTimeMs - Current simulation time in milliseconds.
 */
export function renderImpactSpot(
  context: OffscreenCanvasRenderingContext2D,
  impact: ImpactSpot,
  zBuffer: Float32Array,
  latCtx: LateralProjectionContext,
  simTimeMs: number,
): void {
  const elapsedMs = simTimeMs - impact.createdAtMs;
  const travelTimeMs = impact.boltTravelTimeMs;
  const travelRatio =
    travelTimeMs > 0 ? clamp(elapsedMs / travelTimeMs, 0, 1) : 1;

  if (travelRatio < 1) {
    return;
  }

  const relX = impact.position.x - latCtx.cameraX;
  const relY = impact.position.y - latCtx.cameraY;
  const perpDist = relX * latCtx.dirX + relY * latCtx.dirY;

  if (!Number.isFinite(perpDist) || perpDist <= 0) {
    return;
  }

  const lateral = -relX * latCtx.dirY + relY * latCtx.dirX;
  const screenX =
    latCtx.canvasWidth / 2 +
    (lateral / (perpDist * latCtx.planeScale)) * (latCtx.canvasWidth / 2);

  if (!Number.isFinite(screenX)) {
    return;
  }

  const screenColumn = (screenX / latCtx.canvasWidth) * zBuffer.length;

  if (!depthTestPulse({ screenColumn, distance: perpDist }, zBuffer)) {
    return;
  }

  const alpha = clamp(
    impact.lifetimeMs / NEATENSTEIN_IMPACT_SPOT_LIFETIME_MS,
    0,
    1,
  );

  const radius = Math.max(1, NEATENSTEIN_IMPACT_SPOT_RADIUS_PX / perpDist);

  drawGlowCircle(
    context,
    screenX,
    latCtx.canvasHeight / 2,
    radius,
    NEATENSTEIN_IMPACT_SPOT_COLOR,
    alpha,
    NEATENSTEIN_IMPACT_SPOT_GLOW_COLOR,
    NEATENSTEIN_IMPACT_SPOT_GLOW_BLUR_PX,
  );
}

/**
 * Render a single active ammo pickup.
 *
 * Uses the lateral (floor-plane) projection to map the pickup's world-space
 * position to screen space, applies a z-buffer depth test, and draws a
 * white-core neon-glow circle.
 *
 * @param context - Worker-tier 2D canvas context.
 * @param pickup - Ammo pickup to render.
 * @param zBuffer - Per-column depth buffer.
 * @param latCtx - Lateral projection context for the current frame.
 */
export function renderAmmoPickup(
  context: OffscreenCanvasRenderingContext2D,
  pickup: AmmoPickupState,
  zBuffer: Float32Array,
  latCtx: LateralProjectionContext,
): void {
  const relX = pickup.position.x - latCtx.cameraX;
  const relY = pickup.position.y - latCtx.cameraY;
  const perpDist = relX * latCtx.dirX + relY * latCtx.dirY;

  if (!Number.isFinite(perpDist) || perpDist <= 0) {
    return;
  }

  const lateral = -relX * latCtx.dirY + relY * latCtx.dirX;
  const screenX =
    latCtx.canvasWidth / 2 +
    (lateral / (perpDist * latCtx.planeScale)) * (latCtx.canvasWidth / 2);

  if (!Number.isFinite(screenX)) {
    return;
  }

  const screenColumn = (screenX / latCtx.canvasWidth) * zBuffer.length;

  if (!depthTestPulse({ screenColumn, distance: perpDist }, zBuffer)) {
    return;
  }

  const radius = Math.max(1, NEATENSTEIN_AMMO_PICKUP_RADIUS_PX / perpDist);

  drawGlowCircle(
    context,
    screenX,
    latCtx.canvasHeight / 2,
    radius,
    NEATENSTEIN_AMMO_PICKUP_COLOR,
    1,
    NEATENSTEIN_AMMO_PICKUP_GLOW_COLOR,
    NEATENSTEIN_AMMO_PICKUP_GLOW_BLUR_PX,
  );
}

/**
 * Render a single enemy-impact spot.
 *
 * The spot is rendered only after the plasma bolt that created it has reached
 * the enemy (`travelRatio >= 1`). Uses the floor-perspective projection to map
 * the world-space impact position to screen space, applies a z-buffer depth
 * test, and draws a persistent neon-glow mark followed by a brief expanding
 * burst effect during the first
 * {@link NEATENSTEIN_ENEMY_IMPACT_BURST_DURATION_MS} after arrival.
 *
 * @param context - Worker-tier 2D canvas context.
 * @param impact - Enemy-impact spot to render.
 * @param zBuffer - Per-column depth buffer.
 * @param floorCtx - Floor projection context for the current frame.
 * @param simTimeMs - Current simulation time in milliseconds.
 */
export function renderEnemyImpactSpot(
  context: OffscreenCanvasRenderingContext2D,
  impact: EnemyImpactSpot,
  zBuffer: Float32Array,
  floorCtx: FloorProjectionContext,
  simTimeMs: number,
): void {
  const elapsedMs = simTimeMs - impact.createdAtMs;
  const travelTimeMs = impact.boltTravelTimeMs;
  const travelRatio =
    travelTimeMs > 0 ? clamp(elapsedMs / travelTimeMs, 0, 1) : 1;

  if (travelRatio < 1) {
    return;
  }

  const projected = projectNeatensteinFloorPoint(
    impact.position.x,
    impact.position.y,
    floorCtx.cameraX,
    floorCtx.cameraY,
    floorCtx.cosYaw,
    floorCtx.sinYaw,
    floorCtx.focalLength,
    floorCtx.halfWidth,
    floorCtx.horizonY,
    floorCtx.canvasHeight,
    BOLT_PROJECTED_CAMERA_HEIGHT_WORLD,
  );

  if (projected === null) {
    return;
  }

  const relX = impact.position.x - floorCtx.cameraX;
  const relY = impact.position.y - floorCtx.cameraY;
  const perpDist = relX * floorCtx.cosYaw + relY * floorCtx.sinYaw;

  /* istanbul ignore next -- defensive guard: projectNeatensteinFloorPoint returns null before perpDist is checked */
  if (!Number.isFinite(perpDist) || perpDist <= 0) {
    return;
  }

  const screenColumn = (projected.x / floorCtx.canvasWidth) * zBuffer.length;

  if (!depthTestPulse({ screenColumn, distance: perpDist }, zBuffer)) {
    return;
  }

  const visibleElapsedMs = Math.max(0, elapsedMs - travelTimeMs);
  const alpha = clamp(
    impact.lifetimeMs / NEATENSTEIN_ENEMY_IMPACT_LIFETIME_MS,
    0,
    1,
  );

  const radius = Math.max(1, NEATENSTEIN_ENEMY_IMPACT_RADIUS_PX / perpDist);

  // Persistent mark with additive-blend neon glow.
  drawGlowCircle(
    context,
    projected.x,
    projected.y,
    radius,
    NEATENSTEIN_ENEMY_IMPACT_COLOR,
    alpha,
    NEATENSTEIN_ENEMY_IMPACT_GLOW_COLOR,
    NEATENSTEIN_ENEMY_IMPACT_GLOW_BLUR_PX,
  );

  // Brief expanding burst effect (~200ms after arrival).
  if (visibleElapsedMs < NEATENSTEIN_ENEMY_IMPACT_BURST_DURATION_MS) {
    const burstRatio = clamp(
      visibleElapsedMs / NEATENSTEIN_ENEMY_IMPACT_BURST_DURATION_MS,
      0,
      1,
    );
    const burstRadius = Math.max(
      1,
      (NEATENSTEIN_ENEMY_IMPACT_BURST_RADIUS_PX / perpDist) * burstRatio,
    );
    const burstAlpha = (1 - burstRatio) * 0.7;

    drawGlowCircle(
      context,
      projected.x,
      projected.y,
      burstRadius,
      NEATENSTEIN_ENEMY_IMPACT_COLOR,
      burstAlpha,
      NEATENSTEIN_ENEMY_IMPACT_GLOW_COLOR,
      Math.max(4, burstRadius * 2),
    );
  }
}

/**
 * Render a single player plasma bolt.
 *
 * The bolt is interpolated on screen from the gun muzzle toward its projected
 * impact point. The visual travel duration is fixed so every bolt moves at the
 * same screen-space speed regardless of target distance. When the bolt has
 * origin/direction/targetDistance metadata, the actual impact point is
 * projected for a more accurate travel target.
 *
 * @param context - Worker-tier 2D canvas context.
 * @param bolt - Player bolt to render.
 * @param floorCtx - Floor projection context for the current frame.
 * @param muzzle - Muzzle screen position for the current frame.
 * @param simTimeMs - Current simulation time in milliseconds.
 */
export function renderPlayerBolt(
  context: OffscreenCanvasRenderingContext2D,
  bolt: BoltState,
  floorCtx: FloorProjectionContext,
  muzzle: MuzzleScreenPosition,
  simTimeMs: number,
): void {
  const elapsedMs = simTimeMs - bolt.createdAtMs;
  if (elapsedMs < 0 || elapsedMs > NEATENSTEIN_BOLT_TRAVEL_DURATION_MS) {
    return;
  }

  const projectedCurrent = projectNeatensteinFloorPoint(
    bolt.position.x,
    bolt.position.y,
    floorCtx.cameraX,
    floorCtx.cameraY,
    floorCtx.cosYaw,
    floorCtx.sinYaw,
    floorCtx.focalLength,
    floorCtx.halfWidth,
    floorCtx.horizonY,
    floorCtx.canvasHeight,
    BOLT_PROJECTED_CAMERA_HEIGHT_WORLD,
  );

  if (projectedCurrent === null) {
    return;
  }

  let targetX = projectedCurrent.x;
  let targetY = projectedCurrent.y;

  if (
    bolt.origin &&
    bolt.direction &&
    typeof bolt.targetDistance === 'number' &&
    bolt.targetDistance > 0
  ) {
    // Prefer the actual impact point when the bolt hit an enemy, falling
    // back to the precomputed wall/termination distance otherwise.
    const impactDistance =
      typeof bolt.hitEnemyIndex === 'number' &&
      bolt.hitEnemyIndex >= 0 &&
      typeof bolt.radius === 'number' &&
      bolt.radius > 0
        ? Math.hypot(
            bolt.position.x - bolt.origin.x,
            bolt.position.y - bolt.origin.y,
          )
        : bolt.targetDistance;
    const targetWorldX = bolt.origin.x + bolt.direction.x * impactDistance;
    const targetWorldY = bolt.origin.y + bolt.direction.y * impactDistance;
    const projectedTarget = projectNeatensteinFloorPoint(
      targetWorldX,
      targetWorldY,
      floorCtx.cameraX,
      floorCtx.cameraY,
      floorCtx.cosYaw,
      floorCtx.sinYaw,
      floorCtx.focalLength,
      floorCtx.halfWidth,
      floorCtx.horizonY,
      floorCtx.canvasHeight,
      BOLT_PROJECTED_CAMERA_HEIGHT_WORLD,
    );

    if (projectedTarget !== null) {
      targetX = projectedTarget.x;
      targetY = projectedTarget.y;
    }
  }

  const travelRatio = clamp(
    elapsedMs / NEATENSTEIN_BOLT_TRAVEL_DURATION_MS,
    0,
    1,
  );

  const screenX = muzzle.x + (targetX - muzzle.x) * travelRatio;
  const screenY = muzzle.y + (targetY - muzzle.y) * travelRatio;

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
    return;
  }

  const boltRadius =
    NEATENSTEIN_BOLT_MUZZLE_SCREEN_RADIUS_PX * (1 - fadeRatio) +
    NEATENSTEIN_BOLT_MIN_SCREEN_RADIUS_PX * fadeRatio;
  const boltAlpha = NEATENSTEIN_BOLT_MAX_SCREEN_ALPHA * (1 - fadeRatio);
  const coreRadius = Math.max(
    0,
    boltRadius * NEATENSTEIN_BOLT_CORE_RADIUS_RATIO,
  );

  // Outer glow — teal plasma.
  drawGlowCircle(
    context,
    screenX,
    screenY,
    boltRadius,
    NEATENSTEIN_GUN_ACCENT_COLOR,
    boltAlpha,
    NEATENSTEIN_GUN_ACCENT_COLOR,
    Math.max(2, boltRadius * 2),
  );

  // Bright inner core.
  drawGlowCircle(
    context,
    screenX,
    screenY,
    coreRadius,
    COLOR_WHITE_HEX,
    boltAlpha,
    COLOR_EMPTY_STRING,
    0,
  );
}

/**
 * Render a single enemy plasma bolt.
 *
 * Enemy bolts are rendered with a red/orange color palette to distinguish them
 * visually from the player's teal plasma bolts. Each bolt is projected from
 * its world-space position to screen space, interpolated from its origin
 * toward its current position, and faded based on both lifetime and travel
 * distance. An explosion flash is drawn when the bolt has hit the player.
 *
 * @param context - Worker-tier 2D canvas context.
 * @param bolt - Enemy bolt to render.
 * @param floorCtx - Floor projection context for the current frame.
 * @param simTimeMs - Current simulation time in milliseconds.
 */
export function renderEnemyBolt(
  context: OffscreenCanvasRenderingContext2D,
  bolt: EnemyBoltState,
  floorCtx: FloorProjectionContext,
  simTimeMs: number,
): void {
  const elapsedMs = simTimeMs - bolt.createdAtMs;
  if (elapsedMs < 0 || elapsedMs > NEATENSTEIN_ENEMY_BOLT_LIFETIME_MS) {
    return;
  }

  const projectedCurrent = projectNeatensteinFloorPoint(
    bolt.position.x,
    bolt.position.y,
    floorCtx.cameraX,
    floorCtx.cameraY,
    floorCtx.cosYaw,
    floorCtx.sinYaw,
    floorCtx.focalLength,
    floorCtx.halfWidth,
    floorCtx.horizonY,
    floorCtx.canvasHeight,
    BOLT_PROJECTED_CAMERA_HEIGHT_WORLD,
  );

  if (projectedCurrent === null) {
    return;
  }

  let originX = projectedCurrent.x;
  let originY = projectedCurrent.y;

  if (bolt.origin) {
    const projectedOrigin = projectNeatensteinFloorPoint(
      bolt.origin.x,
      bolt.origin.y,
      floorCtx.cameraX,
      floorCtx.cameraY,
      floorCtx.cosYaw,
      floorCtx.sinYaw,
      floorCtx.focalLength,
      floorCtx.halfWidth,
      floorCtx.horizonY,
      floorCtx.canvasHeight,
      BOLT_PROJECTED_CAMERA_HEIGHT_WORLD,
    );

    if (projectedOrigin !== null) {
      originX = projectedOrigin.x;
      originY = projectedOrigin.y;
    }
  }

  const lifetimeRatio = clamp(
    elapsedMs / NEATENSTEIN_ENEMY_BOLT_LIFETIME_MS,
    0,
    1,
  );

  const distanceTraveled =
    bolt.origin &&
    Number.isFinite(bolt.origin.x) &&
    Number.isFinite(bolt.origin.y)
      ? Math.hypot(
          bolt.position.x - bolt.origin.x,
          bolt.position.y - bolt.origin.y,
        )
      : 0;

  const rangeRatio = clamp(
    distanceTraveled / NEATENSTEIN_ENEMY_BOLT_MAX_RANGE_CELLS,
    0,
    1,
  );

  const fadeRatio = Math.max(lifetimeRatio, rangeRatio);

  if (fadeRatio >= 1) {
    return;
  }

  // Interpolate from the enemy origin toward the current bolt position.
  const screenX = originX + (projectedCurrent.x - originX) * lifetimeRatio;
  const screenY = originY + (projectedCurrent.y - originY) * lifetimeRatio;

  const boltRadius =
    NEATENSTEIN_ENEMY_BOLT_MUZZLE_SCREEN_RADIUS_PX * (1 - fadeRatio) +
    NEATENSTEIN_ENEMY_BOLT_MIN_SCREEN_RADIUS_PX * fadeRatio;
  const boltAlpha = NEATENSTEIN_ENEMY_BOLT_MAX_SCREEN_ALPHA * (1 - fadeRatio);
  const coreRadius = Math.max(
    0,
    boltRadius * NEATENSTEIN_ENEMY_BOLT_CORE_RADIUS_RATIO,
  );

  // Outer glow — red-orange.
  drawGlowCircle(
    context,
    screenX,
    screenY,
    boltRadius,
    NEATENSTEIN_ENEMY_BOLT_COLOR,
    boltAlpha,
    NEATENSTEIN_ENEMY_BOLT_COLOR,
    Math.max(2, boltRadius * 2),
  );

  // Inner core — bright yellow-orange.
  drawGlowCircle(
    context,
    screenX,
    screenY,
    coreRadius,
    NEATENSTEIN_ENEMY_BOLT_CORE_COLOR,
    boltAlpha,
    COLOR_EMPTY_STRING,
    0,
  );

  // Explosion flash when the bolt has hit the player.
  if (bolt.hitPlayer) {
    const flashRadius =
      boltRadius * NEATENSTEIN_BOLT_EXPLOSION_FLASH_RADIUS_MULTIPLIER;
    drawGlowCircle(
      context,
      screenX,
      screenY,
      flashRadius,
      NEATENSTEIN_BOLT_EXPLOSION_FLASH_COLOR,
      boltAlpha * NEATENSTEIN_BOLT_EXPLOSION_FLASH_ALPHA_MULTIPLIER,
      NEATENSTEIN_BOLT_EXPLOSION_FLASH_GLOW_COLOR,
      Math.max(4, flashRadius * 2),
    );
  }
}
