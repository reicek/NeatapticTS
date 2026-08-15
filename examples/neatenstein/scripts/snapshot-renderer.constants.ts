/**
 * @module snapshot-renderer.constants
 *
 * Constants for the 8-direction voxel snapshot renderer. Re-exports frame-size
 * and direction-count constants from their canonical sources and defines
 * rendering-specific lighting, projection, and edge-darkening parameters.
 */

// Re-export frame-size constant from the animator constants (DRY consolidation).
export { ENEMY_FRAME_SIZE_PX } from './enemy-animator.constants';

// Re-export direction count from the controller constants (DRY consolidation).
export { NUM_DIRECTIONS } from './enemy-controller.constants';

// Re-export opaque alpha from the shared browser-entry constants.
export { RGBA_OPAQUE_ALPHA } from '../browser-entry/constants';

/**
 * Angular step between adjacent yaw directions, in degrees.
 */
export const YAW_STEP_DEGREES = 45;

/**
 * Margin left around the projected silhouette, in pixels.
 */
export const PROJECTION_MARGIN = 4;

/**
 * Ambient light level applied to every voxel.
 */
export const AMBIENT_LIGHT = 0.35;

/**
 * Maximum diffuse contribution from the directional light.
 */
export const MAX_DIFFUSE = 0.55;

/**
 * Extra brightness added to emissive (neon/accent) voxels.
 */
export const EMISSIVE_BOOST = 0.55;

/**
 * Maximum edge-darkening penalty for isolated silhouette voxels.
 */
export const EDGE_DARKENING_MAX = 0.35;

/**
 * Per-face edge-darkening factor applied for each exposed voxel face.
 */
export const SNAPSHOT_EDGE_DARKENING_PER_FACE = 0.08;

/**
 * Light direction in camera space: from camera-left and slightly above.
 */
export const LIGHT_DIRECTION = Object.freeze({ x: -1, y: 0.3, z: 1 });

/**
 * Pre-computed norm of the light direction vector, used for diffuse
 * normalization.
 */
export const LIGHT_NORM = Math.hypot(
  LIGHT_DIRECTION.x,
  LIGHT_DIRECTION.y,
  LIGHT_DIRECTION.z,
);