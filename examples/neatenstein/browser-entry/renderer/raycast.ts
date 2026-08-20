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
import { RAY_DIRECTION_EPSILON } from './renderer.raycast.constants';
import type { Vector2 } from '../host/game/types';
import type { CastRayDDAHit } from './renderer.raycast.types';

export { buildNeatensteinMap };

// Re-export constants and types for external consumers.
export type { CastRayDDAHit } from './renderer.raycast.types';

/**
 * World-space size of one DDA grid cell in world units (C1.1, Invariant §1).
 *
 * The DDA traverses integer map cells, so each cell is 1 world unit wide. This
 * MUST equal {@link NEATENSTEIN_FLOOR_GRID_SPACING_WORLD} so wall bases align
 * to floor grid lines.
 */
export const NEATENSTEIN_DDA_CELL_SIZE_WORLD = 1.0 as const;

/**
 * Resolve the initial side distance for a DDA axis, guarding against the
 * `0 * Infinity = NaN` edge case that occurs when the ray origin is exactly
 * on a grid line and the direction component is near-zero (making
 * `deltaDist` infinite).
 *
 * @param offset - Grid-line offset from the ray origin (always ≥ 0).
 * @param deltaDist - Distance between successive grid lines on this axis.
 * @returns Side distance, with `Infinity` substituted for any `NaN` result.
 */
export function resolveSideDistance(offset: number, deltaDist: number): number {
  const result = offset * deltaDist;
  return Number.isNaN(result) ? Number.POSITIVE_INFINITY : result;
}

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
 * When `out` is provided, the result is written into it and the same object
 * is returned, avoiding a per-ray object allocation on hot paths.
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
 * @param out - Optional pre-allocated hit object to write into and return,
 *   avoiding a per-call allocation on hot paths.
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
  out?: CastRayDDAHit,
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
  // resolveSideDistance guards against the 0 * Infinity = NaN edge case when
  // the ray origin is exactly on a grid line and the direction is near-zero.
  let sideDistX = resolveSideDistance(
    stepX > 0 ? mapX + 1 - posX : posX - mapX,
    deltaDistX,
  );
  let sideDistY = resolveSideDistance(
    stepY > 0 ? mapY + 1 - posY : posY - mapY,
    deltaDistY,
  );

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

    // Bounds guard: stop before reading out-of-range map cells. Open sight
    // lines can walk past the map edge before hitting the step cap, so a flat
    // array read would wrap to a false hit.
    if (mapX < 0 || mapY < 0 || mapX >= side || mapY >= side) {
      if (out) {
        out.perpWallDist = Number.POSITIVE_INFINITY;
        out.side = sideHit;
        out.mapX = mapX;
        out.mapY = mapY;
        return out;
      }
      return {
        perpWallDist: Number.POSITIVE_INFINITY,
        side: sideHit,
        mapX,
        mapY,
      };
    }

    // Non-zero cells are walls.
    if (flatMap[mapY * side + mapX] !== 0) {
      const perpWallDist = computePerpendicularWallDistance(
        mapX,
        mapY,
        posX,
        posY,
        dirX,
        dirY,
        stepX,
        stepY,
        sideHit,
      );
      if (out) {
        out.perpWallDist = perpWallDist;
        out.side = sideHit;
        out.mapX = mapX;
        out.mapY = mapY;
        return out;
      }
      return {
        perpWallDist,
        side: sideHit,
        mapX,
        mapY,
      };
    }

    // Hard cap: stop walking after a distance-proportional number of empty
    // cells. A 45° ray advances ~0.707 perpendicular units per step, so a
    // fixed 30-step cap would clip legitimate walls ~21 units away. Scale by
    // the inverse of the dominant direction component so that axis-aligned
    // rays keep the standard 30-step limit and diagonal rays get the steps
    // they need to cover the same world distance.
    const minAbsDir = Math.min(Math.abs(dirX), Math.abs(dirY));
    const maxSteps =
      minAbsDir < RAY_DIRECTION_EPSILON
        ? NEATENSTEIN_RENDER_DISTANCE_CAP
        : Math.ceil(NEATENSTEIN_RENDER_DISTANCE_CAP / minAbsDir);
    if (steps >= maxSteps) {
      if (out) {
        out.perpWallDist = Number.POSITIVE_INFINITY;
        out.side = sideHit;
        out.mapX = mapX;
        out.mapY = mapY;
        return out;
      }
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
