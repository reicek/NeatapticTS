/**
 * Pure raycast utility executors extracted from the display worker.
 *
 * `castColumnRay` was previously reading the module-level `wallMap` variable
 * directly. It has been parameterized to accept `wallMap` as an explicit
 * argument so it can be tested and reused without coupling to worker state.
 *
 * @module
 */

import { NEATENSTEIN_MAP_SIZE } from '../constants';
import { castRayDDAFromFlatMap } from '../renderer/raycast';
import type { RaycastHit } from './display.worker.types';

/**
 * Cast a single camera ray for the given column index.
 *
 * @param wallMap - Flat row-major wall map (`Uint8Array`).
 * @param column - Column index in `[0, columnCount)`.
 * @param columnCount - Total number of raycast columns.
 * @param cameraPositionX - Camera X position.
 * @param cameraPositionY - Camera Y position.
 * @param cameraDirectionX - Camera direction X.
 * @param cameraDirectionY - Camera direction Y.
 * @param cameraPlaneX - Camera plane X.
 * @param cameraPlaneY - Camera plane Y.
 * @returns DDA raycast hit or miss sentinel.
 */
export function castColumnRay(
  wallMap: Uint8Array,
  column: number,
  columnCount: number,
  cameraPositionX: number,
  cameraPositionY: number,
  cameraDirectionX: number,
  cameraDirectionY: number,
  cameraPlaneX: number,
  cameraPlaneY: number,
): RaycastHit {
  // Map the column index to a -1..+1 offset on the camera plane.
  const cameraPlaneOffset = (2 * column) / columnCount - 1;

  // Combine camera forward direction with the camera plane offset.
  const rayDirectionX = cameraDirectionX + cameraPlaneX * cameraPlaneOffset;
  const rayDirectionY = cameraDirectionY + cameraPlaneY * cameraPlaneOffset;

  return castRayDDAFromFlatMap(
    wallMap,
    NEATENSTEIN_MAP_SIZE,
    cameraPositionX,
    cameraPositionY,
    rayDirectionX,
    rayDirectionY,
  );
}
