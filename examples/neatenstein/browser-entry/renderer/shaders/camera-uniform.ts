/**
 * Camera uniform struct factory for the Neatenstein shader pipeline.
 *
 * Produces a plain-object camera uniform that mirrors the shared
 * {@link NEATENSTEIN_FLOOR_*} projection constants so the GPU shader tier and
 * the CPU tier derive identical projection geometry.
 *
 * This module is a **forward-looking scaffold** for a future GPU (WebGL2/WebGPU)
 * tier. It is not imported by the current CPU-tier worker or renderer. The
 * uniform struct is exported so that alignment tests can verify the GPU-tier
 * projection constants match the CPU-tier constants.
 *
 * @module
 */

import {
  NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
  NEATENSTEIN_FLOOR_HORIZON_RATIO,
  NEATENSTEIN_FLOOR_FOV_RADIANS,
} from '../renderer.floor.constants';
import { NEATENSTEIN_RENDER_DISTANCE_CAP } from '../framebuffer';

/**
 * Parameters for {@link createNeatensteinCameraUniform}.
 */
export interface NeatensteinCameraUniformParams {
  /** Canvas width in backing-store pixels. */
  canvasWidth: number;
  /** Canvas height in backing-store pixels. */
  canvasHeight: number;
  /** Camera world X coordinate. */
  cameraX: number;
  /** Camera world Y coordinate. */
  cameraY: number;
  /** Camera yaw in radians. */
  cameraYaw: number;
}

/**
 * Camera uniform struct consumed by the wall-DDA and floor-caster shaders.
 */
export interface NeatensteinCameraUniform {
  /** Perspective focal length in pixels: `H/2 / tan(FOV/2)`. */
  focalLength: number;
  /** Camera plane scale: `(W/H) * tan(FOV/2)`. */
  planeScale: number;
  /** Unit forward direction `[cos(yaw), sin(yaw)]`. */
  cameraDirection: [number, number];
  /** Camera plane vector `[-sin(yaw)*planeScale, cos(yaw)*planeScale]`. */
  cameraPlane: [number, number];
  /** Camera height above the floor in world units. */
  cameraHeight: number;
  /** Horizon Y coordinate in pixels: `H * HORIZON_RATIO`. */
  horizon: number;
  /** Maximum perpendicular render distance in world units. */
  renderDistanceCap: number;
}

/**
 * Build a camera uniform struct from canvas dimensions and camera pose.
 *
 * The returned struct reuses the exact same shared constants
 * (`NEATENSTEIN_FLOOR_FOV_RADIANS`, `NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD`,
 * `NEATENSTEIN_FLOOR_HORIZON_RATIO`, `NEATENSTEIN_RENDER_DISTANCE_CAP`) as the
 * CPU-tier projection code, ensuring floor-wall alignment across tiers.
 *
 * @param params - Canvas dimensions and camera pose.
 * @returns Camera uniform struct.
 */
export function createNeatensteinCameraUniform(
  params: NeatensteinCameraUniformParams,
): NeatensteinCameraUniform {
  const { canvasWidth, canvasHeight, cameraYaw } = params;

  const focalLength =
    canvasHeight / 2 / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
  const planeScale =
    (canvasWidth / canvasHeight) * Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
  const horizon = canvasHeight * NEATENSTEIN_FLOOR_HORIZON_RATIO;

  const cosYaw = Math.cos(cameraYaw);
  const sinYaw = Math.sin(cameraYaw);

  return {
    focalLength,
    planeScale,
    cameraDirection: [cosYaw, sinYaw],
    cameraPlane: [-sinYaw * planeScale, cosYaw * planeScale],
    cameraHeight: NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
    horizon,
    renderDistanceCap: NEATENSTEIN_RENDER_DISTANCE_CAP,
  };
}
