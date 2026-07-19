/**
 * DDA ray casting for the Neatenstein neon renderer.
 *
 * Steps a ray through a 2D wall grid using the digital differential analyzer
 * (DDA) algorithm and returns the first occupied cell it reaches, together
 * with the perpendicular wall distance and the side of the wall that was hit.
 *
 * @module
 */

import { buildNeatensteinMap } from './map';

export { buildNeatensteinMap };

/** Result of a single DDA ray cast. */
export interface CastRayDDAHit {
  /** Perpendicular distance from the camera plane to the wall hit. */
  perpWallDist: number;
  /**
   * Wall side that was hit:
   * - `0` = X-side (east/west wall face)
   * - `1` = Y-side (north/south wall face)
   */
  side: 0 | 1;
  /** Grid X coordinate of the hit cell. */
  mapX: number;
  /** Grid Y coordinate of the hit cell. */
  mapY: number;
}

/** Direction component below this magnitude is treated as zero. */
const RAY_DIRECTION_EPSILON = 1e-9;

/** Safety cap on DDA iterations to avoid infinite loops on malformed grids. */
const MAX_DDA_STEPS = 256;

/**
 * Cast a single ray through a wall grid using DDA.
 *
 * The ray starts at `(posX, posY)` and travels in direction `(dirX, dirY)`
 * until it enters a cell whose value is not `0`. The return value includes
 * the perpendicular distance to the wall (suitable for column height
 * projection), which side of the wall was hit, and the grid coordinates of
 * the hit cell.
 *
 * @param grid - 2D wall grid where `0` is open floor and any non-zero value
 *   is a wall. The outer array is indexed by X and each inner array by Y.
 * @param gridWidth - Horizontal cell count of the grid.
 * @param gridHeight - Vertical cell count of the grid.
 * @param posX - Ray origin X coordinate in grid units.
 * @param posY - Ray origin Y coordinate in grid units.
 * @param dirX - Ray direction X component (does not need to be normalized).
 * @param dirY - Ray direction Y component (does not need to be normalized).
 * @returns The first wall hit encountered by the ray.
 *
 * @example
 * ```ts
 * const hit = castRayDDA(grid, 24, 24, 1.5, 1.5, 1, 0);
 * console.log(hit.mapX, hit.mapY, hit.side);
 * ```
 */
export function castRayDDA(
  grid: ReadonlyArray<ReadonlyArray<number>>,
  gridWidth: number,
  gridHeight: number,
  posX: number,
  posY: number,
  dirX: number,
  dirY: number,
): CastRayDDAHit {
  // Step 1: Locate the starting cell and decide the step direction per axis.
  let mapX = Math.floor(posX);
  let mapY = Math.floor(posY);

  const stepX = dirX >= 0 ? 1 : -1;
  const stepY = dirY >= 0 ? 1 : -1;

  // Step 2: Compute the distance between consecutive x or y grid lines.
  const deltaDistX =
    Math.abs(dirX) < RAY_DIRECTION_EPSILON ? Infinity : Math.abs(1 / dirX);
  const deltaDistY =
    Math.abs(dirY) < RAY_DIRECTION_EPSILON ? Infinity : Math.abs(1 / dirY);

  // Step 3: Compute the distance to the first x or y grid boundary.
  let sideDistX =
    dirX >= 0 ? (mapX + 1 - posX) * deltaDistX : (posX - mapX) * deltaDistX;
  let sideDistY =
    dirY >= 0 ? (mapY + 1 - posY) * deltaDistY : (posY - mapY) * deltaDistY;

  // Step 4: Step through the grid until a wall is hit or the grid is left.
  let side: 0 | 1 = 0;
  let steps = 0;
  while (steps < MAX_DDA_STEPS) {
    if (mapX < 0 || mapX >= gridWidth || mapY < 0 || mapY >= gridHeight) {
      break;
    }

    if (grid[mapX][mapY] !== 0) {
      break;
    }

    if (sideDistX < sideDistY) {
      side = 0;
      sideDistX += deltaDistX;
      mapX += stepX;
    } else {
      side = 1;
      sideDistY += deltaDistY;
      mapY += stepY;
    }
    steps++;
  }

  // Step 5: Resolve the perpendicular distance from the camera plane.
  const safeDirX =
    Math.abs(dirX) < RAY_DIRECTION_EPSILON ? RAY_DIRECTION_EPSILON : dirX;
  const safeDirY =
    Math.abs(dirY) < RAY_DIRECTION_EPSILON ? RAY_DIRECTION_EPSILON : dirY;
  const perpWallDist =
    side === 0
      ? (mapX - posX + (1 - stepX) / 2) / safeDirX
      : (mapY - posY + (1 - stepY) / 2) / safeDirY;

  return { perpWallDist, side, mapX, mapY };
}
