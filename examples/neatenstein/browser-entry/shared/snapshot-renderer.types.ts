/**
 * @module snapshot-renderer.types
 *
 * Type definitions for the 8-direction voxel snapshot renderer, extracted
 * from `snapshot-renderer.ts` so other modules can import them without a
 * circular dependency on the renderer implementation.
 */

/**
 * Output of the directional voxel snapshot renderer, containing the image
 * dimensions and a flat RGBA buffer in row-major order.
 */
export interface VoxelSnapshot {
  /** Output width in pixels. */
  width: number;
  /** Output height in pixels. */
  height: number;
  /** Flat RGBA pixel buffer in row-major order. */
  data: Uint8ClampedArray;
}

/**
 * Optional configuration for `renderVoxelSnapshot`, including output
 * dimensions and a pre-computed occupancy grid for faster repeated renders.
 */
export interface SnapshotOptions {
  /** Output width in pixels. Defaults to 128. */
  width?: number;
  /** Output height in pixels. Defaults to 128. */
  height?: number;
  /**
   * Optional pre-computed occupancy grid for the voxel grid. When supplied, the
   * renderer skips rebuilding the grid, which is significantly faster when many
   * frames are rendered from the same grid. The grid is a dense `Uint8Array`
   * indexed by `z * (height * width) + y * width + x`.
   */
  occupancy?: Uint8Array;
}
