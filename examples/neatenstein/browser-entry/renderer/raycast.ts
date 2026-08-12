/**
 * DDA ray casting for the Neatenstein neon renderer.
 *
 * This module provides the grid traversal primitive used by the renderer to
 * find the first wall cell intersected by a ray. The implementation uses the
 * digital differential analyzer (DDA) algorithm commonly used by Wolfenstein-
 * style raycasters.
 *
 * The canonical Neatenstein map is stored as a flat row-major `Uint8Array`.
 * {@link castRayDDAFromFlatMap} traverses that flat map directly and performs
 * no per-ray grid allocation.
 *
 * @module
 */

import { buildNeatensteinMap } from './map';
import { NEATENSTEIN_RENDER_DISTANCE_CAP } from './framebuffer';
import type { Vector2 } from '../host/game/types';

export { buildNeatensteinMap };

/**
 * Result of a single DDA ray cast.
 *
 * In well-formed Neatenstein maps, where the arena perimeter is closed and the
 * camera starts inside the open arena, rays always hit a wall and
 * `perpWallDist` is finite.
 */
export interface CastRayDDAHit {
  /** Perpendicular distance from the camera plane to the wall hit. */
  perpWallDist: number;

  /**
   * Wall side that was hit:
   *
   * - `0` = X-side, meaning an east/west wall face was crossed
   * - `1` = Y-side, meaning a north/south wall face was crossed
   */
  side: 0 | 1;

  /** Grid X coordinate of the hit cell. */
  mapX: number;

  /** Grid Y coordinate of the hit cell. */
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
 * Return `true` when a direction component is effectively zero.
 *
 * @param value - Direction component to test.
 * @returns Whether the component is close enough to zero to be ignored.
 */
function isNearlyZero(value: number): boolean {
  return Math.abs(value) < RAY_DIRECTION_EPSILON;
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
 * @returns Perpendicular wall distance.
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
  // Valid Neatenstein rays reach this function only after a wall hit on the
  // selected side. The corresponding direction component is therefore non-zero
  // and the classic camera-space distance formula applies directly.
  return side === 0
    ? (mapX - posX + (1 - stepX) / 2) / dirX
    : (mapY - posY + (1 - stepY) / 2) / dirY;
}

/**
 * Cast a ray through a flat row-major `Uint8Array` map.
 *
 * This is the renderer-facing raycast entrypoint. It traverses the canonical
 * flat map directly and performs no per-ray grid allocation.
 *
 * Preconditions (satisfied by the fixed 120×120 Neatenstein map and valid
 * camera path):
 * - `flatMap.length === side * side`.
 * - `side` is a positive integer.
 * - `(posX, posY)` lies inside an open cell.
 * - `(dirX, dirY)` is not a zero-length direction.
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
 * @returns The first wall hit, or a sentinel with `perpWallDist = Infinity`
 *   when no wall is found within {@link NEATENSTEIN_RENDER_DISTANCE_CAP} cells.
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

  let sideHit: 0 | 1;
  let steps = 0;
  while (true) {
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

    steps += 1;

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

    // Hard cap: stop walking after a fixed number of empty cells.
    if (steps >= NEATENSTEIN_RENDER_DISTANCE_CAP) {
      return {
        perpWallDist: Number.POSITIVE_INFINITY,
        side: sideHit,
        mapX,
        mapY,
      };
    }
  }
}

/**
 * Check whether there is a clear line of sight between two grid-space points.
 *
 * Casts a DDA ray from `from` toward `to` and returns `true` when no wall cell
 * is intersected before reaching `to`. The direction vector is normalized so
 * that the perpendicular wall distance returned by
 * {@link castRayDDAFromFlatMap} is the Euclidean distance along the ray to the
 * first wall. When that wall distance is greater than or equal to the Euclidean
 * distance from `from` to `to`, the destination is visible.
 *
 * @param flatMap - Row-major wall grid where any non-zero value is a wall.
 * @param mapSize - Width and height of the square grid.
 * @param from - Origin point in grid units.
 * @param to - Destination point in grid units.
 * @returns `true` when no wall occludes the straight-line path from `from` to
 *   `to`; `false` when a wall cell is intersected first.
 */
export function hasLineOfSight(
  flatMap: Uint8Array,
  mapSize: number,
  from: Vector2,
  to: Vector2,
): boolean {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const dist = Math.hypot(dx, dy);
  if (dist === 0) return true; // same position — trivially visible

  const dirX = dx / dist;
  const dirY = dy / dist;

  const hit = castRayDDAFromFlatMap(
    flatMap,
    mapSize,
    from.x,
    from.y,
    dirX,
    dirY,
  );
  const wallDist = Number.isFinite(hit.perpWallDist)
    ? hit.perpWallDist
    : Infinity;

  return wallDist > dist;
}
