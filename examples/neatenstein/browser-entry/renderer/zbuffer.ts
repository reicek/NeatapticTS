/**
 * Per-column depth buffer for sprite and pulse occlusion in the Neatenstein
 * neon raycaster.
 *
 * The wall pass writes the perpendicular wall distance for every renderer column
 * into a `Float32Array`. The sprite pass then clips each sprite's projected
 * screen span against that buffer so that sprite columns are only drawn where the
 * sprite is closer than the wall at that column.
 *
 * @module
 */

/**
 * Sentinel value meaning "no wall in this column".
 *
 * Sprites are always visible against an empty column because any finite
 * distance is closer than infinity.
 */
export const NEATENSTEIN_ZBUFFER_EMPTY = Number.POSITIVE_INFINITY;

/**
 * Result of clipping a sprite's screen span against the per-column z-buffer.
 */
export interface NeatensteinSpriteClip {
  /** Inclusive first screen column covered by the sprite (clamped to canvas). */
  left: number;
  /** Inclusive last screen column covered by the sprite (clamped to canvas). */
  right: number;
  /** Screen column indices where the sprite is closer than the stored wall. */
  visibleColumns: number[];
}

/**
 * Allocate a fresh per-column z-buffer.
 *
 * The buffer is initialized to {@link NEATENSTEIN_ZBUFFER_EMPTY} so that sprites
 * are visible in every column until the wall pass writes real distances.
 *
 * @param columnCount - Number of renderer columns (must be non-negative).
 * @returns A new `Float32Array` sized to `columnCount`.
 *
 * @example
 * ```ts
 * const zBuffer = buildNeatensteinZBuffer(320);
 * ```
 */
export function buildNeatensteinZBuffer(columnCount: number): Float32Array {
  return new Float32Array(columnCount).fill(NEATENSTEIN_ZBUFFER_EMPTY);
}

/**
 * Copy wall distances into the z-buffer, replacing empty values with the empty
 * sentinel so sprites stay visible where no wall was hit.
 *
 * @param zBuffer - The depth buffer to fill.
 * @param wallDistances - Perpendicular wall distance per column. Zero or
 *   negative entries are treated as "no wall" and stored as infinity.
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
  const count = Math.min(zBuffer.length, wallDistances.length);
  for (let index = 0; index < count; index++) {
    const distance = wallDistances[index];
    zBuffer[index] = distance > 0 ? distance : NEATENSTEIN_ZBUFFER_EMPTY;
  }
}

/**
 * Decide whether a single sprite column should be drawn.
 *
 * A sprite column is visible when the sprite's perpendicular distance is
 * strictly less than the wall distance stored in the z-buffer at that column.
 * If the stored value is the empty sentinel, the sprite is always visible.
 *
 * @param zBuffer - Per-column depth buffer.
 * @param column - Screen column index to test.
 * @param spriteDistance - Perpendicular distance from the camera plane to the
 *   sprite.
 * @returns `true` when the sprite column is not occluded by a wall.
 */
export function isNeatensteinSpriteColumnVisible(
  zBuffer: Readonly<Float32Array>,
  column: number,
  spriteDistance: number,
): boolean {
  if (column < 0 || column >= zBuffer.length) return false;
  return spriteDistance < zBuffer[column];
}

/**
 * Clip a sprite's projected screen span against the z-buffer.
 *
 * Only columns where `spriteDistance < zBuffer[column]` are returned. Columns
 * outside the canvas bounds are ignored, so callers can project sprites whose
 * edges fall partially off-screen without extra clamping logic.
 *
 * @param zBuffer - Per-column depth buffer filled by the wall pass.
 * @param screenLeft - Left edge of the sprite in screen pixels (may be
 *   fractional or negative).
 * @param screenRight - Right edge of the sprite in screen pixels (may be
 *   fractional or beyond the canvas).
 * @param spriteDistance - Perpendicular distance from camera plane to sprite.
 * @returns The clamped span and the list of visible column indices.
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
  const columnCount = zBuffer.length;
  const left = Math.max(0, Math.floor(screenLeft));
  const right = Math.min(columnCount - 1, Math.ceil(screenRight) - 1);

  const visibleColumns: number[] = [];
  for (let column = left; column <= right; column++) {
    if (isNeatensteinSpriteColumnVisible(zBuffer, column, spriteDistance)) {
      visibleColumns.push(column);
    }
  }

  return { left, right, visibleColumns };
}
