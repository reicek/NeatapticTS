/**
 * @module generate-enemy-sprites.voxel.utils
 *
 * Voxel build executors for the enemy sprite generator.
 *
 * Produces deformed voxel grids for each animation state and frame, caching the
 * heavy base mesh per accent color to preserve determinism while avoiding
 * redundant reconstruction.
 */

import { buildVoxelEnemy, type Voxel, type VoxelGrid } from './voxel-enemy';
import {
  ENEMY_ANIMATION_FRAME_COUNTS,
  type EnemyAnimationState,
} from './enemy-animator';

/**
 * Cache the heavy base voxel mesh per accent color. This preserves full
 * determinism because each animation frame still receives an independent clone
 * of the cached base voxels; only the immutable dimensions/palette are reused.
 */
const baseVoxelCache = new Map<string, { grid: VoxelGrid; voxels: Voxel[] }>();

/**
 * Build a flat occupancy grid from a voxel list.
 *
 * The animated grid is clamped to its bounds before rendering, so every voxel
 * is guaranteed to be in-bounds.
 *
 * @param grid - Voxel grid dimensions.
 * @param voxels - Voxel list to mark in the occupancy grid.
 * @returns A `Uint8Array` of `width * height * depth` with 1 for occupied cells.
 */
export function buildOccupancyGrid(
  grid: VoxelGrid,
  voxels: readonly Voxel[],
): Uint8Array {
  const { width, height, depth } = grid;
  const occupancy = new Uint8Array(width * height * depth);
  for (const voxel of voxels) {
    occupancy[voxel.z * (height * width) + voxel.y * width + voxel.x] = 1;
  }
  return occupancy;
}

/**
 * Retrieve (or build and cache) the base voxel enemy for a given accent color.
 *
 * @param accentColor - Optional accent override; defaults to Ares Red.
 * @returns A cloned base voxel grid and voxel list.
 */
export function getBaseVoxelEnemy(accentColor: string | undefined): {
  grid: VoxelGrid;
  voxels: Voxel[];
} {
  const key = accentColor ?? 'default';
  let entry = baseVoxelCache.get(key);
  if (!entry) {
    const grid = buildVoxelEnemy(accentColor);
    entry = { grid, voxels: grid.voxels.map((voxel) => ({ ...voxel })) };
    baseVoxelCache.set(key, entry);
  }
  return {
    grid: entry.grid,
    voxels: entry.voxels.map((voxel) => ({ ...voxel })),
  };
}

/**
 * Build a voxel grid for a specific animation state and frame.
 *
 * Deformations are deterministic and based only on `(state, frameIndex)`:
 * - `idle`: subtle whole-body vertical bob and slight breathing.
 * - `move`: alternating leg stride and counter-swinging arms.
 * - `fire`: cannon recoil and a short-lived muzzle flash.
 * - `death`: collapse, darkening, and back-disk flicker.
 *
 * @param accentColor - Optional accent override; defaults to Ares Red.
 * @param state - Animation state.
 * @param frameIndex - Frame index within the state.
 * @returns A deformed voxel grid ready for rendering.
 */
export function buildAnimatedVoxelEnemy(
  accentColor: string | undefined,
  state: EnemyAnimationState,
  frameIndex: number,
): VoxelGrid {
  const base = getBaseVoxelEnemy(accentColor);
  const voxels: Voxel[] = base.voxels.map((voxel) => ({ ...voxel }));

  switch (state) {
    case 'idle': {
      const bob = Math.round(
        2 *
          Math.sin(
            (frameIndex / ENEMY_ANIMATION_FRAME_COUNTS.idle) * 2 * Math.PI,
          ),
      );
      for (const voxel of voxels) {
        voxel.y += bob;
      }
      break;
    }

    case 'move': {
      const stride = Math.sin(
        (frameIndex / ENEMY_ANIMATION_FRAME_COUNTS.move) * 2 * Math.PI,
      );
      for (const voxel of voxels) {
        if (voxel.part === 'legs') {
          voxel.y += Math.round(
            stride * (voxel.x < base.grid.width / 2 ? 2 : -2),
          );
        } else if (voxel.part === 'arms') {
          voxel.y += Math.round(-stride * 2);
        }
      }
      break;
    }

    case 'fire': {
      const recoil =
        1 - frameIndex / Math.max(1, ENEMY_ANIMATION_FRAME_COUNTS.fire - 1);
      for (const voxel of voxels) {
        if (voxel.part === 'cannon') {
          voxel.z -= Math.round(2 * recoil);
          voxel.y += Math.round(1 * recoil);
        }
      }

      if (frameIndex === 0) {
        const cannonVoxels = voxels.filter((voxel) => voxel.part === 'cannon');
        const maxZ = Math.max(...cannonVoxels.map((voxel) => voxel.z));
        const tipVoxels = cannonVoxels.filter((voxel) => voxel.z === maxZ);
        for (const tip of tipVoxels) {
          for (let dz = 1; dz <= 2; dz++) {
            voxels.push({
              ...tip,
              z: tip.z + dz,
              material: 'damage',
              emissive: true,
            });
          }
        }
      }
      break;
    }

    case 'death': {
      const progress =
        frameIndex / Math.max(1, ENEMY_ANIMATION_FRAME_COUNTS.death - 1);
      for (const voxel of voxels) {
        voxel.y = Math.round(voxel.y * (1 - progress * 0.5));
        voxel.y -= Math.round(progress * 20);
        voxel.r = Math.round(voxel.r * (1 - progress * 0.6));
        voxel.g = Math.round(voxel.g * (1 - progress * 0.6));
        voxel.b = Math.round(voxel.b * (1 - progress * 0.6));

        if (voxel.part === 'back disk' && frameIndex % 3 === 0) {
          voxel.emissive = !voxel.emissive;
        }
      }
      break;
    }
  }

  const clamped = voxels.filter(
    (voxel) =>
      voxel.x >= 0 &&
      voxel.x < base.grid.width &&
      voxel.y >= 0 &&
      voxel.y < base.grid.height &&
      voxel.z >= 0 &&
      voxel.z < base.grid.depth,
  );

  return { ...base.grid, voxels: clamped };
}
