/**
 * Type definitions for the bolt renderer extracted from {@link module:./bolt.utils}.
 *
 * @module
 */

/**
 * Lateral (floor-plane) projection context shared across per-item renders.
 *
 * Computed once per frame and passed to each `renderImpactSpot` /
 * `renderAmmoPickup` call so the hot loop avoids recomputing camera
 * direction and plane scale.
 */
export interface LateralProjectionContext {
  /** Camera forward-direction X component (cos(yaw)). */
  dirX: number;
  /** Camera forward-direction Y component (sin(yaw)). */
  dirY: number;
  /** Frustum half-width divided by focal length. */
  planeScale: number;
  /** Canvas width in pixels. */
  canvasWidth: number;
  /** Canvas height in pixels. */
  canvasHeight: number;
  /** Camera world-space X position. */
  cameraX: number;
  /** Camera world-space Y position. */
  cameraY: number;
}

/**
 * Floor-perspective projection context shared across per-item renders.
 *
 * Computed once per frame and passed to each `renderEnemyImpactSpot` /
 * `renderPlayerBolt` / `renderEnemyBolt` call so the hot loop avoids
 * recomputing the horizon, focal length, and yaw trig values.
 */
export interface FloorProjectionContext {
  /** Screen-space Y coordinate of the horizon line. */
  horizonY: number;
  /** Half the canvas width in pixels. */
  halfWidth: number;
  /** Focal length derived from the vertical field-of-view. */
  focalLength: number;
  /** Cosine of the camera yaw. */
  cosYaw: number;
  /** Sine of the camera yaw. */
  sinYaw: number;
  /** Canvas width in pixels. */
  canvasWidth: number;
  /** Canvas height in pixels. */
  canvasHeight: number;
  /** Camera world-space X position. */
  cameraX: number;
  /** Camera world-space Y position. */
  cameraY: number;
}

/**
 * Muzzle screen position for player bolt rendering.
 *
 * Computed once per frame from the canvas dimensions and the muzzle anchor
 * ratios.
 */
export interface MuzzleScreenPosition {
  /** Screen-space X coordinate of the muzzle anchor. */
  x: number;
  /** Screen-space Y coordinate of the muzzle anchor. */
  y: number;
}
