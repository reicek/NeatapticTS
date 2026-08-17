/**
 * Type definitions for the z-buffer module extracted from
 * {@link module:./zbuffer}.
 *
 * @module
 */

/**
 * Result of clipping a sprite's screen span against the per-column z-buffer.
 */
export interface NeatensteinSpriteClip {
  /** Inclusive first screen column covered by the sprite, clamped to canvas. */
  left: number;
  /** Inclusive last screen column covered by the sprite, clamped to canvas. */
  right: number;
  /** Screen column indices where the sprite is closer than the stored wall. */
  visibleColumns: number[];
}
