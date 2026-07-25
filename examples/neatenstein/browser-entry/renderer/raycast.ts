/**
 * DDA ray casting for the Neatenstein neon renderer.
 *
 * This module provides the grid traversal primitive used by the renderer to
 * find the first wall cell intersected by a ray. The implementation uses the
 * digital differential analyzer (DDA) algorithm commonly used by Wolfenstein-
 * style raycasters.
 *
 * The canonical Neatenstein map is stored as a flat row-major `Uint8Array`.
 * For performance, {@link castRayDDAFromFlatMap} traverses that flat map
 * directly and should be preferred in renderer hot paths. The 2D-grid variant,
 * {@link castRayDDA}, is retained for tests, tooling, and compatibility with
 * callers that already expose a `grid[x][y]` structure.
 *
 * @module
 */

import { buildNeatensteinMap } from './map';

export { buildNeatensteinMap };

/**
 * Result of a single DDA ray cast.
 *
 * In well-formed Neatenstein maps, where the arena perimeter is closed, rays
 * should always hit a wall and `perpWallDist` should be finite.
 *
 * For malformed maps, out-of-bounds starts, or zero-length directions, the
 * functions return a safe miss sentinel with `perpWallDist` set to
 * `Number.POSITIVE_INFINITY`. This preserves the historical return shape while
 * avoiding misleading finite wall distances.
 */
export interface CastRayDDAHit {
  /**
   * Perpendicular distance from the camera plane to the wall hit.
   *
   * A value of `Number.POSITIVE_INFINITY` means no valid wall hit was found.
   * Renderers can naturally treat that as a zero-height projected wall column.
   */
  perpWallDist: number;

  /**
   * Wall side that was hit:
   *
   * - `0` = X-side, meaning an east/west wall face was crossed
   * - `1` = Y-side, meaning a north/south wall face was crossed
   */
  side: 0 | 1;

  /** Grid X coordinate of the hit cell, or the final attempted cell on miss. */
  mapX: number;

  /** Grid Y coordinate of the hit cell, or the final attempted cell on miss. */
  mapY: number;
}

/**
 * Direction component below this magnitude is treated as zero.
 *
 * This avoids unstable reciprocal values for rays that are effectively
 * axis-aligned.
 */
const RAY_DIRECTION_EPSILON = 1e-9;

/**
 * Compatibility safety cap for Neatenstein-sized maps.
 *
 * The runtime traversal limit is derived from the input grid dimensions so
 * well-formed rectangular grids terminate naturally after leaving the map.
 * This constant remains exported for existing imports and tests.
 */
export const NEATENSTEIN_DDA_MAX_STEPS = 256;

/**
 * Return `true` when a direction component is effectively zero.
 *
 * @param value - Direction component to test.
 * @returns Whether the component is close enough to zero to be ignored.
 */
function isNearlyZero(value: number): boolean {
  return Math.abs(value) < RAY_DIRECTION_EPSILON;
}

/**
 * Build a safe no-hit result without throwing from renderer hot paths.
 *
 * @param mapX - Final or attempted grid X coordinate.
 * @param mapY - Final or attempted grid Y coordinate.
 * @param side - Last side selected by the DDA traversal.
 * @returns A miss sentinel compatible with {@link CastRayDDAHit}.
 */
function createMissResult(
  mapX: number,
  mapY: number,
  side: 0 | 1 = 0,
): CastRayDDAHit {
  return {
    perpWallDist: Number.POSITIVE_INFINITY,
    side,
    mapX,
    mapY,
  };
}

/**
 * Return `true` when a grid coordinate lies inside the given rectangular bounds.
 *
 * @param mapX - Grid X coordinate.
 * @param mapY - Grid Y coordinate.
 * @param gridWidth - Grid width in cells.
 * @param gridHeight - Grid height in cells.
 * @returns Whether the cell is within the grid.
 */
function isInsideGrid(
  mapX: number,
  mapY: number,
  gridWidth: number,
  gridHeight: number,
): boolean {
  return mapX >= 0 && mapX < gridWidth && mapY >= 0 && mapY < gridHeight;
}

/**
 * Compute the maximum number of DDA cell transitions required to leave a grid.
 *
 * Each DDA iteration advances by one cell along either X or Y. Starting inside
 * a finite rectangular grid, a ray can cross at most roughly `width + height`
 * cell boundaries before it leaves the grid. The small margin keeps edge cases
 * around exact grid-line starts from terminating too aggressively.
 *
 * @param gridWidth - Grid width in cells.
 * @param gridHeight - Grid height in cells.
 * @returns Safe traversal limit for the supplied grid.
 */
function getTraversalStepLimit(gridWidth: number, gridHeight: number): number {
  return Math.max(1, gridWidth + gridHeight + 2);
}

/**
 * Compute perpendicular wall distance for the final DDA hit.
 *
 * The formula mirrors classic camera-plane raycasting. It intentionally uses
 * the original ray direction components rather than normalized Euclidean ray
 * length because projection code typically expects this camera-space distance
 * to avoid fish-eye distortion.
 *
 * @param mapX - Hit cell X coordinate.
 * @param mapY - Hit cell Y coordinate.
 * @param posX - Ray origin X coordinate.
 * @param posY - Ray origin Y coordinate.
 * @param dirX - Ray direction X component.
 * @param dirY - Ray direction Y component.
 * @param stepX - DDA step direction along X.
 * @param stepY - DDA step direction along Y.
 * @param side - Side selected by the final DDA step.
 * @returns Perpendicular wall distance, or `Infinity` if the denominator is invalid.
 */
function computePerpendicularWallDistance(
  mapX: number,
  mapY: number,
  posX: number,
  posY: number,
  dirX: number,
  dirY: number,
  stepX: number,
  stepY: number,
  side: 0 | 1,
): number {
  if (side === 0) {
    if (isNearlyZero(dirX)) {
      return Number.POSITIVE_INFINITY;
    }

    return (mapX - posX + (1 - stepX) / 2) / dirX;
  }

  if (isNearlyZero(dirY)) {
    return Number.POSITIVE_INFINITY;
  }

  return (mapY - posY + (1 - stepY) / 2) / dirY;
}

/**
 * Cast a single ray through a wall grid using DDA.
 *
 * The ray starts at `(posX, posY)` and travels in direction `(dirX, dirY)`
 * until it enters a cell whose value is not `0`.
 *
 * This compatibility variant accepts a 2D grid indexed as `grid[x][y]`.
 * Renderer hot paths should prefer {@link castRayDDAFromFlatMap}, which avoids
 * allocating or traversing a converted 2D grid.
 *
 * @param grid - 2D wall grid where `0` is open floor and any non-zero value is
 *   a wall. The outer array is indexed by X and each inner array by Y.
 * @param gridWidth - Horizontal cell count of the grid.
 * @param gridHeight - Vertical cell count of the grid.
 * @param posX - Ray origin X coordinate in grid units.
 * @param posY - Ray origin Y coordinate in grid units.
 * @param dirX - Ray direction X component in renderer camera-space scale.
 * @param dirY - Ray direction Y component in renderer camera-space scale.
 * @returns The first wall hit, or an `Infinity`-distance miss sentinel.
 *
 * @example
 * ```ts
 * const hit = castRayDDA(grid, 60, 60, 1.5, 1.5, 1, 0);
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
  let mapX = Math.floor(posX);
  let mapY = Math.floor(posY);

  // A zero-length ray has no meaningful direction. Return a safe miss sentinel
  // rather than producing arbitrary wall distances.
  if (isNearlyZero(dirX) && isNearlyZero(dirY)) {
    return createMissResult(mapX, mapY);
  }

  // If the ray starts outside the provided grid, no in-grid wall can be hit.
  if (!isInsideGrid(mapX, mapY, gridWidth, gridHeight)) {
    return createMissResult(mapX, mapY);
  }

  // If the ray begins inside a solid cell, report an immediate zero-distance
  // hit. This makes the function well-defined for overlapping wall starts.
  if (grid[mapX]?.[mapY] !== 0) {
    return {
      perpWallDist: 0,
      side: 0,
      mapX,
      mapY,
    };
  }

  // Decide which direction each grid coordinate moves when crossing a boundary.
  const stepX = dirX >= 0 ? 1 : -1;
  const stepY = dirY >= 0 ? 1 : -1;

  // Distance along the ray between consecutive vertical or horizontal grid
  // lines. An axis-aligned ray never crosses grid lines on the zero component.
  const deltaDistX = isNearlyZero(dirX)
    ? Number.POSITIVE_INFINITY
    : Math.abs(1 / dirX);
  const deltaDistY = isNearlyZero(dirY)
    ? Number.POSITIVE_INFINITY
    : Math.abs(1 / dirY);

  // Distance from the ray origin to the first vertical or horizontal grid line.
  let sideDistX =
    stepX > 0 ? (mapX + 1 - posX) * deltaDistX : (posX - mapX) * deltaDistX;
  let sideDistY =
    stepY > 0 ? (mapY + 1 - posY) * deltaDistY : (posY - mapY) * deltaDistY;

  let side: 0 | 1 = 0;
  const maxSteps = getTraversalStepLimit(gridWidth, gridHeight);

  for (let steps = 0; steps < maxSteps; steps++) {
    // Advance to the next grid cell along the nearest grid boundary.
    if (sideDistX < sideDistY) {
      side = 0;
      sideDistX += deltaDistX;
      mapX += stepX;
    } else {
      side = 1;
      sideDistY += deltaDistY;
      mapY += stepY;
    }

    // Leaving the grid means the supplied map did not contain a reachable wall.
    if (!isInsideGrid(mapX, mapY, gridWidth, gridHeight)) {
      return createMissResult(mapX, mapY, side);
    }

    // Any non-zero cell is solid.
    if (grid[mapX]?.[mapY] !== 0) {
      return {
        perpWallDist: computePerpendicularWallDistance(
          mapX,
          mapY,
          posX,
          posY,
          dirX,
          dirY,
          stepX,
          stepY,
          side,
        ),
        side,
        mapX,
        mapY,
      };
    }
  }

  // A well-formed rectangular traversal should leave the grid before this.
  // Reaching the limit indicates malformed dimensions or unexpected input.
  return createMissResult(mapX, mapY, side);
}

/**
 * Cast a ray through a flat row-major `Uint8Array` map.
 *
 * This is the preferred renderer-facing raycast entrypoint. It traverses the
 * canonical flat map directly and performs no per-ray grid allocation.
 *
 * The flat layout is row-major:
 *
 * ```ts
 * const index = y * side + x;
 * ```
 *
 * @param flatMap - Row-major wall grid where any non-zero value is a wall.
 * @param side - Width and height of the square grid.
 * @param posX - Ray origin X coordinate in grid units.
 * @param posY - Ray origin Y coordinate in grid units.
 * @param dirX - Ray direction X component in renderer camera-space scale.
 * @param dirY - Ray direction Y component in renderer camera-space scale.
 * @returns The first wall hit, or an `Infinity`-distance miss sentinel.
 */
export function castRayDDAFromFlatMap(
  flatMap: Uint8Array,
  side: number,
  posX: number,
  posY: number,
  dirX: number,
  dirY: number,
): CastRayDDAHit {
  let mapX = Math.floor(posX);
  let mapY = Math.floor(posY);

  // A zero-length ray cannot enter a deterministic DDA traversal.
  if (isNearlyZero(dirX) && isNearlyZero(dirY)) {
    return createMissResult(mapX, mapY);
  }

  // Treat invalid or empty square dimensions as a safe miss.
  if (!Number.isInteger(side) || side <= 0) {
    return createMissResult(mapX, mapY);
  }

  // If the typed array is shorter than the declared square map, avoid unsafe
  // reads and report a miss. Correct Neatenstein maps are exactly side * side.
  if (flatMap.length < side * side) {
    return createMissResult(mapX, mapY);
  }

  // Starting outside the arena cannot hit a valid in-map wall cell.
  if (!isInsideGrid(mapX, mapY, side, side)) {
    return createMissResult(mapX, mapY);
  }

  // Immediate hit when the origin is already inside a solid cell.
  if (flatMap[mapY * side + mapX] !== 0) {
    return {
      perpWallDist: 0,
      side: 0,
      mapX,
      mapY,
    };
  }

  // DDA step direction for each grid axis.
  const stepX = dirX >= 0 ? 1 : -1;
  const stepY = dirY >= 0 ? 1 : -1;

  // Distance between successive vertical or horizontal grid lines.
  const deltaDistX = isNearlyZero(dirX)
    ? Number.POSITIVE_INFINITY
    : Math.abs(1 / dirX);
  const deltaDistY = isNearlyZero(dirY)
    ? Number.POSITIVE_INFINITY
    : Math.abs(1 / dirY);

  // Distance from the ray origin to the first crossed grid line on each axis.
  let sideDistX =
    stepX > 0 ? (mapX + 1 - posX) * deltaDistX : (posX - mapX) * deltaDistX;
  let sideDistY =
    stepY > 0 ? (mapY + 1 - posY) * deltaDistY : (posY - mapY) * deltaDistY;

  let sideHit: 0 | 1 = 0;
  const maxSteps = getTraversalStepLimit(side, side);

  for (let steps = 0; steps < maxSteps; steps++) {
    // Step into the next map cell through the closest pending grid boundary.
    if (sideDistX < sideDistY) {
      sideHit = 0;
      sideDistX += deltaDistX;
      mapX += stepX;
    } else {
      sideHit = 1;
      sideDistY += deltaDistY;
      mapY += stepY;
    }

    // A closed Neatenstein perimeter should prevent this. If it occurs, return
    // a safe miss sentinel instead of manufacturing a finite wall distance.
    if (!isInsideGrid(mapX, mapY, side, side)) {
      return createMissResult(mapX, mapY, sideHit);
    }

    // Non-zero cells are walls.
    if (flatMap[mapY * side + mapX] !== 0) {
      return {
        perpWallDist: computePerpendicularWallDistance(
          mapX,
          mapY,
          posX,
          posY,
          dirX,
          dirY,
          stepX,
          stepY,
          sideHit,
        ),
        side: sideHit,
        mapX,
        mapY,
      };
    }
  }

  // Traversal limits should not be reached for valid square maps with closed
  // borders. Keep renderer behavior stable by returning an infinite miss.
  return createMissResult(mapX, mapY, sideHit);
}
