/**
 * @module voxel-enemy.constants
 *
 * Constants for the voxel enemy descriptor: canonical palette colors, grid
 * dimensions, body-part names, and material-slot definitions. Extracted from
 * `voxel-enemy.ts` so other modules can reference these values without a
 * circular dependency on the builder implementation.
 */

/** Default accent color (Ares Red) used when no override is supplied. */
export const DEFAULT_ACCENT = '#DD2200';

/** Neon white/bright emission color. */
export const NEON_WHITE = '#FBFFFF';

/** Dark suit base color. */
export const DARK_SUIT = '#121418';

/** Darker shadow variant of the suit. */
export const DEEP_SHADOW = '#0A0B0E';

/** Damage flash color. */
export const DAMAGE_FLASH = '#FF3300';

/** Voxel grid width, derived from the approved 192×192 PNG silhouettes. */
export const GRID_WIDTH = 64;

/** Voxel grid height, matching the 192-px reference snapshot size. */
export const GRID_HEIGHT = 192;

/** Voxel grid depth, derived from the approved 192×192 PNG silhouettes. */
export const GRID_DEPTH = 64;

/** Horizontal center of the voxel grid. */
export const GRID_CENTER_X = GRID_WIDTH / 2;

/** Depth center of the voxel grid. */
export const GRID_CENTER_Z = GRID_DEPTH / 2;

/** Shared edge/neon/disk thickness in voxels (must stay within 2–4). */
export const THICKNESS = 3;

/** Body part names in the descriptor. */
export const PARTS = [
  'head',
  'torso',
  'arms',
  'legs',
  'cannon',
  'back disk',
] as const;

/** Material slot label for the swappable accent color. */
export const MATERIAL_ACCENT = 'accent' as const;

/** Material slot label for the neon white/bright emission color. */
export const MATERIAL_NEON = 'neon' as const;

/** Material slot label for the dark suit base color. */
export const MATERIAL_SUIT = 'suit' as const;

/** Material slot label for the darker shadow variant. */
export const MATERIAL_DARK = 'dark' as const;

/** Material slot label for the damage flash color. */
export const MATERIAL_DAMAGE = 'damage' as const;

// ---------------------------------------------------------------------------
// Voxel anatomy dimensions (promoted from inline literals in voxel-enemy.ts)
// ---------------------------------------------------------------------------

/** Leg width in voxels. */
export const LEG_WIDTH = 5;

/** Leg depth in voxels. */
export const LEG_DEPTH = 8;

/** Leg height in voxels (from the ground up to the hip joint). */
export const LEG_HEIGHT = 52;

/** Knee cap Y-min (bottom of the accent knee band). */
export const KNEE_Y_MIN = 18;

/** Knee cap Y-max (top of the accent knee band). */
export const KNEE_Y_MAX = 24;

/** Torso Y-min (bottom of the torso block). */
export const TORSO_Y_MIN = 54;

/** Torso Y-max (top of the torso block). */
export const TORSO_Y_MAX = 108;

/** Torso half-width in voxels (from grid center). */
export const TORSO_X_HALF = 14;

/** Torso half-depth in voxels (from grid center). */
export const TORSO_Z_HALF = 8;

/** Arm Y-min (top of the legs where arms begin). */
export const ARM_Y_MIN = 56;

/** Arm Y-max (shoulder height where arms end). */
export const ARM_Y_MAX = 92;

/** Arm outer half-width offset from grid center. */
export const ARM_X_HALF_OUTER = 22;

/** Arm inner half-width offset from grid center. */
export const ARM_X_HALF_INNER = 15;

/** Arm half-depth in voxels (from grid center). */
export const ARM_Z_HALF = 5;

/** Cannon Y-min (bottom of the chest cannon barrel). */
export const CANNON_Y_MIN = 70;

/** Cannon Y-max (top of the chest cannon barrel). */
export const CANNON_Y_MAX = 78;

/** Cannon X-offset min from grid center. */
export const CANNON_X_OFFSET_MIN = 16;

/** Cannon X-offset max from grid center. */
export const CANNON_X_OFFSET_MAX = 30;

/** Cannon Z-offset min from grid center. */
export const CANNON_Z_OFFSET_MIN = -2;

/** Cannon Z-offset max from grid center. */
export const CANNON_Z_OFFSET_MAX = 12;

/** Head Y-min (bottom of the helmet, above the torso). */
export const HEAD_Y_MIN = 112;

/** Head Y-max (top of the helmet). */
export const HEAD_Y_MAX = 142;

/** Head half-width in voxels (from grid center). */
export const HEAD_X_HALF = 10;

/** Head half-depth in voxels (from grid center). */
export const HEAD_Z_HALF = 8;

/** Back identity disk radius in voxels. */
export const DISK_RADIUS = 12;

/** Back identity disk center Y coordinate. */
export const DISK_CENTER_Y = 102;

/** Back identity disk center Z offset from grid center. */
export const DISK_CENTER_Z_OFFSET = -12;
