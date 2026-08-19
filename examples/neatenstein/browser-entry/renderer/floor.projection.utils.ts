/**
 * Ray-to-floor projection executors extracted from the floor renderer.
 *
 * Contains pure leaf functions that sanitize camera state, resolve canvas
 * dimensions, and project world-space floor/ceiling points into screen space
 * using perspective division.
 *
 * @module
 */

import { NEATENSTEIN_RENDER_DISTANCE_CAP } from './framebuffer';
import type { NeatensteinFloorCamera } from './renderer.floor.types';
import { NEATENSTEIN_FLOOR_NEAR_PLANE_EPSILON } from './renderer.floor.constants';
import type {
  SafeNeatensteinFloorCamera,
  NeatensteinGridProjectionContext,
  ProjectedNeatensteinGridPoint,
} from './renderer.floor.types';
import { clamp } from '../shared/math-guards.utils';
import { isPositiveFiniteDimension } from '../shared/math-guards.utils';

// Re-export previously-public symbols that moved to dedicated files.
export type {
  SafeNeatensteinFloorCamera,
  NeatensteinGridProjectionContext,
  ProjectedNeatensteinGridPoint,
} from './renderer.floor.types';

/**
 * Sanitize camera values for renderer use.
 *
 * Non-finite camera values are treated as `0` so a malformed frame cannot
 * poison the canvas path with `NaN` coordinates.
 *
 * @param camera - Raw camera state.
 * @returns Finite camera state.
 */
export function sanitizeNeatensteinFloorCamera(
  camera: NeatensteinFloorCamera,
): SafeNeatensteinFloorCamera {
  return {
    x: Number.isFinite(camera.x) ? camera.x : 0,
    y: Number.isFinite(camera.y) ? camera.y : 0,
    yaw: Number.isFinite(camera.yaw) ? camera.yaw : 0,
  };
}

/**
 * Resolve a canvas dimension from a context, using a test fallback if missing.
 *
 * @param value - Optional canvas dimension.
 * @param fallback - Fallback dimension for mock contexts.
 * @returns Positive finite dimension.
 */
export function resolveContextCanvasDimension(
  value: number | undefined,
  fallback: number,
): number {
  return isPositiveFiniteDimension(value ?? Number.NaN) ? value! : fallback;
}

/**
 * Shared floor/ceiling world-to-screen projection.
 *
 * @param worldX - World X coordinate.
 * @param worldY - World Y coordinate.
 * @param projection - Shared projection constants.
 * @param forCeiling - Whether to mirror vertically above the horizon.
 * @returns Projected point, or `null` if culled.
 */
export function projectNeatensteinGridPoint(
  worldX: number,
  worldY: number,
  projection: NeatensteinGridProjectionContext,
  forCeiling: boolean,
): ProjectedNeatensteinGridPoint | null {
  const dx = worldX - projection.cameraX;
  const dy = worldY - projection.cameraY;

  // Camera-space Y is forward depth. Camera yaw of 0 looks down world +X.
  const camSpaceY = dx * projection.cosYaw + dy * projection.sinYaw;

  if (camSpaceY <= NEATENSTEIN_FLOOR_NEAR_PLANE_EPSILON) {
    return null;
  }

  // Cull grid points beyond the hard render distance cap. This matches the
  // wall DDA stop distance so the floor/ceiling grid terminates at the same
  // depth as the wall cap — no visible tunnel or gap between them.
  if (camSpaceY > NEATENSTEIN_RENDER_DISTANCE_CAP) {
    return null;
  }

  // Camera-space X is horizontal right/left displacement.
  const camSpaceX = -dx * projection.sinYaw + dy * projection.cosYaw;
  const verticalOffset =
    (projection.cameraHeight / camSpaceY) * projection.focalLength;

  const screenX =
    projection.halfWidth + (camSpaceX / camSpaceY) * projection.focalLength;

  const screenY = forCeiling
    ? projection.horizonY - verticalOffset
    : projection.horizonY + verticalOffset;

  if (!Number.isFinite(screenX) || !Number.isFinite(screenY)) {
    return null;
  }

  const depthDenominator = forCeiling
    ? projection.horizonY
    : projection.height - projection.horizonY;

  if (!Number.isFinite(depthDenominator) || depthDenominator <= 0) {
    return null;
  }

  const rawDepthRatio = forCeiling
    ? (projection.horizonY - screenY) / depthDenominator
    : (screenY - projection.horizonY) / depthDenominator;

  return {
    x: screenX,
    y: screenY,
    depthRatio: clamp(rawDepthRatio, 0, 1),
    distance: camSpaceY,
  };
}

/**
 * Project a world-space grid point into a caller-provided scratch slot.
 *
 * Identical math to {@link projectNeatensteinGridPoint} but mutates the
 * `scratch` object in place and returns the same reference, enabling a
 * ping-pong (two-slot) pattern in {@link appendNeatensteinGridLine} so that
 * consecutive projections don't overwrite each other.
 *
 * @param worldX - World X coordinate.
 * @param worldY - World Y coordinate.
 * @param projection - Shared projection constants.
 * @param forCeiling - Whether to mirror vertically above the horizon.
 * @param scratch - Caller-provided scratch object to write into.
 * @returns The same `scratch` reference with updated fields, or `null` if
 *   culled.
 */
export function projectNeatensteinGridPointInto(
  worldX: number,
  worldY: number,
  projection: NeatensteinGridProjectionContext,
  forCeiling: boolean,
  scratch: ProjectedNeatensteinGridPoint,
): ProjectedNeatensteinGridPoint | null {
  const dx = worldX - projection.cameraX;
  const dy = worldY - projection.cameraY;

  const camSpaceY = dx * projection.cosYaw + dy * projection.sinYaw;

  if (camSpaceY <= NEATENSTEIN_FLOOR_NEAR_PLANE_EPSILON) {
    return null;
  }

  if (camSpaceY > NEATENSTEIN_RENDER_DISTANCE_CAP) {
    return null;
  }

  const camSpaceX = -dx * projection.sinYaw + dy * projection.cosYaw;
  const verticalOffset =
    (projection.cameraHeight / camSpaceY) * projection.focalLength;

  const screenX =
    projection.halfWidth + (camSpaceX / camSpaceY) * projection.focalLength;

  const screenY = forCeiling
    ? projection.horizonY - verticalOffset
    : projection.horizonY + verticalOffset;

  if (!Number.isFinite(screenX) || !Number.isFinite(screenY)) {
    return null;
  }

  const depthDenominator = forCeiling
    ? projection.horizonY
    : projection.height - projection.horizonY;

  if (!Number.isFinite(depthDenominator) || depthDenominator <= 0) {
    return null;
  }

  const rawDepthRatio = forCeiling
    ? (projection.horizonY - screenY) / depthDenominator
    : (screenY - projection.horizonY) / depthDenominator;

  scratch.x = screenX;
  scratch.y = screenY;
  scratch.depthRatio = clamp(rawDepthRatio, 0, 1);
  scratch.distance = camSpaceY;
  return scratch;
}

/**
 * Project a world-space floor point to screen space.
 *
 * The point is translated into camera-relative coordinates, rotated into
 * camera space, and projected using the shared vertical FOV. Points behind
 * or too close to the camera plane are rejected.
 *
 * @param worldX - World X coordinate.
 * @param worldY - World Y coordinate.
 * @param cameraX - Camera world X.
 * @param cameraY - Camera world Y.
 * @param cosYaw - Cosine of camera yaw.
 * @param sinYaw - Sine of camera yaw.
 * @param focalLength - Perspective focal length in pixels.
 * @param halfWidth - Half canvas width in pixels.
 * @param horizonY - Horizon Y coordinate in pixels.
 * @param height - Canvas height in pixels.
 * @param cameraHeight - Camera height above floor in world units.
 * @returns Projected screen point, or `null` if the point is not drawable.
 */
export function projectNeatensteinFloorPoint(
  worldX: number,
  worldY: number,
  cameraX: number,
  cameraY: number,
  cosYaw: number,
  sinYaw: number,
  focalLength: number,
  halfWidth: number,
  horizonY: number,
  height: number,
  cameraHeight: number,
): { x: number; y: number; depthRatio: number; distance: number } | null {
  return projectNeatensteinGridPoint(
    worldX,
    worldY,
    {
      width: halfWidth * 2,
      height,
      cameraX,
      cameraY,
      cosYaw,
      sinYaw,
      focalLength,
      halfWidth,
      horizonY,
      cameraHeight,
    },
    false,
  );
}

/**
 * Project a world-space ceiling point to screen space.
 *
 * The ceiling uses the same camera-space transform as the floor, but its
 * vertical projection is mirrored above the horizon.
 *
 * @param worldX - World X coordinate.
 * @param worldY - World Y coordinate.
 * @param cameraX - Camera world X.
 * @param cameraY - Camera world Y.
 * @param cosYaw - Cosine of camera yaw.
 * @param sinYaw - Sine of camera yaw.
 * @param focalLength - Perspective focal length in pixels.
 * @param halfWidth - Half canvas width in pixels.
 * @param horizonY - Horizon Y coordinate in pixels.
 * @param height - Canvas height in pixels.
 * @param cameraHeight - Camera height above floor in world units.
 * @returns Projected screen point, or `null` if the point is not drawable.
 */
export function projectNeatensteinCeilingPoint(
  worldX: number,
  worldY: number,
  cameraX: number,
  cameraY: number,
  cosYaw: number,
  sinYaw: number,
  focalLength: number,
  halfWidth: number,
  horizonY: number,
  height: number,
  cameraHeight: number,
): { x: number; y: number; depthRatio: number; distance: number } | null {
  return projectNeatensteinGridPoint(
    worldX,
    worldY,
    {
      width: halfWidth * 2,
      height,
      cameraX,
      cameraY,
      cosYaw,
      sinYaw,
      focalLength,
      halfWidth,
      horizonY,
      cameraHeight,
    },
    true,
  );
}
