/**
 * Per-column depth buffer for sprite and pulse occlusion in the Neatenstein
 * neon raycaster.
 *
 * The wall pass writes one perpendicular wall distance per renderer column into
 * a `Float32Array`. Sprite and pulse passes then compare their projected
 * distance against that buffer so they only draw where they are closer than the
 * wall already rendered in the same screen column.
 *
 * @module
 */

import { NEATENSTEIN_ZBUFFER_EMPTY } from './renderer.zbuffer.constants';
import type { NeatensteinSpriteClip } from './renderer.zbuffer.types';

// Re-export constants and types for external consumers.
export { NEATENSTEIN_ZBUFFER_EMPTY } from './renderer.zbuffer.constants';
export type { NeatensteinSpriteClip } from './renderer.zbuffer.types';

/**
 * Return a canonical empty clip for invisible or invalid spans.
 *
 * @returns Empty sprite clip.
 */
function createEmptySpriteClip(): NeatensteinSpriteClip {
  return {
    left: 0,
    right: -1,
    visibleColumns: [],
  };
}

/**
 * Normalize a requested column count into a valid typed-array length.
 *
 * Invalid, negative, or fractional values are treated as zero. This keeps the
 * builder total and prevents low-level `RangeError` exceptions from leaking out
 * of typed-array construction.
 *
 * @param columnCount - Requested z-buffer column count.
 * @returns Safe non-negative integer column count.
 */
function normalizeZBufferColumnCount(columnCount: number): number {
  if (!Number.isFinite(columnCount) || columnCount <= 0) {
    return 0;
  }

  return Math.floor(columnCount);
}

/**
 * Return whether a wall distance should occlude sprites.
 *
 * Positive finite distances represent real wall hits. Positive infinity is the
 * explicit empty sentinel. Zero, negative, and `NaN` values are treated as
 * empty because they do not represent a valid wall in front of the camera.
 *
 * @param distance - Candidate wall distance.
 * @returns Whether the distance is a positive finite wall hit.
 */
function isValidWallDistance(distance: number): boolean {
  return Number.isFinite(distance) && distance > 0;
}

/**
 * Return whether a sprite distance can participate in z-buffer comparisons.
 *
 * @param spriteDistance - Perpendicular sprite distance.
 * @returns Whether the sprite is in front of the camera at a finite distance.
 */
function isValidSpriteDistance(spriteDistance: number): boolean {
  return Number.isFinite(spriteDistance) && spriteDistance > 0;
}

/**
 * Allocate a fresh per-column z-buffer.
 *
 * The buffer is initialized to {@link NEATENSTEIN_ZBUFFER_EMPTY} so sprites and
 * pulses are visible in every column until the wall pass writes real distances.
 *
 * Invalid or negative `columnCount` values produce an empty buffer.
 *
 * @param columnCount - Number of renderer columns.
 * @returns A new `Float32Array` sized to the normalized column count.
 *
 * @example
 * ```ts
 * const zBuffer = buildNeatensteinZBuffer(320);
 * ```
 */
export function buildNeatensteinZBuffer(columnCount: number): Float32Array {
  const safeColumnCount = normalizeZBufferColumnCount(columnCount);

  return new Float32Array(safeColumnCount).fill(NEATENSTEIN_ZBUFFER_EMPTY);
}

/**
 * Copy wall distances into the z-buffer.
 *
 * Positive finite wall distances are copied directly. Zero, negative, `NaN`,
 * and missing entries are stored as {@link NEATENSTEIN_ZBUFFER_EMPTY}, keeping
 * sprites visible in those columns.
 *
 * The entire z-buffer is written every call. If `wallDistances` is shorter
 * than `zBuffer`, the remaining columns are reset to the empty sentinel so no
 * stale values leak across frames.
 *
 * @param zBuffer - The depth buffer to fill.
 * @param wallDistances - Perpendicular wall distance per column.
 *
 * @example
 * ```ts
 * fillNeatensteinZBuffer(zBuffer, frame.wallDistances);
 * ```
 */
export function fillNeatensteinZBuffer(
  zBuffer: Float32Array,
  wallDistances: Readonly<Float32Array>,
): void {
  for (let index = 0; index < zBuffer.length; index += 1) {
    const distance =
      index < wallDistances.length
        ? wallDistances[index]
        : NEATENSTEIN_ZBUFFER_EMPTY;

    zBuffer[index] = isValidWallDistance(distance)
      ? distance
      : NEATENSTEIN_ZBUFFER_EMPTY;
  }
}

/**
 * Decide whether a single sprite column should be drawn.
 *
 * A sprite column is visible when the sprite's perpendicular distance is
 * strictly less than the wall distance stored in the z-buffer at the same
 * column. Empty columns contain `Infinity`, so any finite positive sprite
 * distance is visible there.
 *
 * @param zBuffer - Per-column depth buffer.
 * @param column - Integer screen column index to test.
 * @param spriteDistance - Perpendicular distance from the camera plane to the
 *   sprite.
 * @returns `true` when the sprite column is not occluded by a wall.
 */
export function isNeatensteinSpriteColumnVisible(
  zBuffer: Readonly<Float32Array>,
  column: number,
  spriteDistance: number,
): boolean {
  if (
    !Number.isInteger(column) ||
    column < 0 ||
    column >= zBuffer.length ||
    !isValidSpriteDistance(spriteDistance)
  ) {
    return false;
  }

  return spriteDistance < zBuffer[column];
}

/**
 * Invoke a callback for each visible column in a projected sprite span.
 *
 * This allocation-free helper is useful in hot paths where callers want to
 * render columns directly without first building a `visibleColumns` array.
 *
 * @param zBuffer - Per-column depth buffer filled by the wall pass.
 * @param screenLeft - Left edge of the projected span in screen pixels.
 * @param screenRight - Right edge of the projected span in screen pixels.
 * @param spriteDistance - Perpendicular distance from camera plane to sprite.
 * @param onVisibleColumn - Callback invoked with each visible integer column.
 */
export function forEachVisibleNeatensteinSpriteColumn(
  zBuffer: Readonly<Float32Array>,
  screenLeft: number,
  screenRight: number,
  spriteDistance: number,
  onVisibleColumn: (column: number) => void,
): void {
  if (
    zBuffer.length === 0 ||
    !Number.isFinite(screenLeft) ||
    !Number.isFinite(screenRight) ||
    screenRight < screenLeft ||
    !isValidSpriteDistance(spriteDistance)
  ) {
    return;
  }

  const left = Math.max(0, Math.floor(screenLeft));
  const right = Math.min(zBuffer.length - 1, Math.ceil(screenRight) - 1);

  if (left > right) {
    return;
  }

  for (let column = left; column <= right; column += 1) {
    if (spriteDistance < zBuffer[column]) {
      onVisibleColumn(column);
    }
  }
}

/**
 * Clip a sprite's projected screen span against the z-buffer.
 *
 * Only columns where `spriteDistance < zBuffer[column]` are returned. Columns
 * outside the canvas bounds are ignored, so callers can project sprites whose
 * edges fall partially off-screen without extra clamping logic.
 *
 * Invalid spans or invalid sprite distances return an empty clip:
 *
 * ```ts
 * { left: 0, right: -1, visibleColumns: [] }
 * ```
 *
 * @param zBuffer - Per-column depth buffer filled by the wall pass.
 * @param screenLeft - Left edge of the sprite in screen pixels.
 * @param screenRight - Right edge of the sprite in screen pixels.
 * @param spriteDistance - Perpendicular distance from camera plane to sprite.
 * @returns The clamped span and list of visible column indices.
 *
 * @example
 * ```ts
 * const { visibleColumns } = clipNeatensteinSpriteSpan(
 *   zBuffer,
 *   projection.left,
 *   projection.right,
 *   projection.perpDist,
 * );
 * ```
 */
export function clipNeatensteinSpriteSpan(
  zBuffer: Readonly<Float32Array>,
  screenLeft: number,
  screenRight: number,
  spriteDistance: number,
): NeatensteinSpriteClip {
  if (
    zBuffer.length === 0 ||
    !Number.isFinite(screenLeft) ||
    !Number.isFinite(screenRight) ||
    screenRight < screenLeft ||
    !isValidSpriteDistance(spriteDistance)
  ) {
    return createEmptySpriteClip();
  }

  const left = Math.max(0, Math.floor(screenLeft));
  const right = Math.min(zBuffer.length - 1, Math.ceil(screenRight) - 1);

  if (left > right) {
    return createEmptySpriteClip();
  }

  const visibleColumns: number[] = [];

  for (let column = left; column <= right; column += 1) {
    if (spriteDistance < zBuffer[column]) {
      visibleColumns.push(column);
    }
  }

  return {
    left,
    right,
    visibleColumns,
  };
}
