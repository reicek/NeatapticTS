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