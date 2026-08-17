/**
 * Type definitions for the raycast module extracted from
 * {@link module:./raycast}.
 *
 * @module
 */

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
