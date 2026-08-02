/**
 * Dedicated voxel/3D gun-sprite projection for the Neatenstein gun overlay.
 *
 * Projects a small voxel grid into screen-space squares. The grid is indexed
 * as rows (depth) by columns (width). A positive cell value emits that many
 * vertical voxels, so a 2D pattern can be extruded into a chunky 3D barrel.
 *
 * @module
 */

/** One projected voxel square ready for the 2D canvas. */
export interface ProjectedGunVoxel {
  /** Screen-space X coordinate of the voxel center. */
  screenX: number;
  /** Screen-space Y coordinate of the voxel center. */
  screenY: number;
  /** Screen-space edge length of the voxel square. */
  size: number;
  /** CSS color to fill this voxel. */
  color: string;
}

/**
 * Default barrel-shaped voxel grid for the plasma cannon muzzle.
 *
 * A 5x5 logical grid: `1` marks an occupied voxel column. When projected it
 * produces a short 3D block above the gun body.
 */
export const GUN_BARREL_VOXEL_GRID: number[][] = [
  [0, 0, 1, 0, 0],
  [0, 1, 1, 1, 0],
  [0, 1, 1, 1, 0],
  [0, 1, 1, 1, 0],
  [0, 0, 1, 0, 0],
];

/** Color for the front-facing sides of projected voxels. */
const VOXEL_FACE_SHADE = '#eefcfd';

/** Color for the top cap of the uppermost voxel in a column. */
const VOXEL_TOP_SHADE = '#ffffff';

/**
 * Project a small voxel grid into screen-space fillable squares.
 *
 * Uses a simple dimetric projection: each voxel column is offset laterally by
 * its grid column, and each stacked voxel is offset upward to give depth. The
 * result is a non-empty list of squares for any non-empty grid.
 *
 * @param options - Projection inputs.
 * @param options.voxelGrid - Row-major occupancy grid; positive values emit that
 *   many stacked voxels.
 * @param options.screenX - Screen-space anchor X.
 * @param options.screenY - Screen-space anchor Y (bottom of the grid stack).
 * @param options.scale - Screen-space size of one voxel unit.
 *
 * @returns Array of projected voxel squares to draw on the 2D context.
 */
export function projectGunSprite(options: {
  voxelGrid: number[][];
  screenX: number;
  screenY: number;
  scale: number;
}): ProjectedGunVoxel[] {
  const { voxelGrid, screenX, screenY, scale } = options;
  const rows = voxelGrid.length;
  const cols = rows > 0 ? voxelGrid[0].length : 0;
  const midRow = (rows - 1) / 2;
  const midCol = (cols - 1) / 2;

  const projected: ProjectedGunVoxel[] = [];

  for (let r = 0; r < rows; r++) {
    const row = voxelGrid[r];
    for (let c = 0; c < cols; c++) {
      const height = row[c];
      if (height <= 0) {
        continue;
      }
      for (let v = 0; v < height; v++) {
        const dx = (c - midCol) * scale;
        const dy = (r - midRow) * scale * 0.5;
        const dz = v * scale;
        // Simple dimetric projection: lateral offset plus a little depth skew,
        // with vertical offset lifted by voxel height so taller columns rise up.
        const x = screenX + dx - dy * 0.4;
        const y = screenY - dy - dz * 0.6;
        const size = scale * 0.9;
        const color = v === height - 1 ? VOXEL_TOP_SHADE : VOXEL_FACE_SHADE;
        projected.push({ screenX: x, screenY: y, size, color });
      }
    }
  }

  return projected;
}
