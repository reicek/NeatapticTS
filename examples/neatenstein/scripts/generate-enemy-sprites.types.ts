/**
 * @module generate-enemy-sprites.types
 *
 * Type definitions for the enemy sprite-sheet and reference-snapshot
 * generator. Extracted from `generate-enemy-sprites.ts` so utility modules
 * can import them without creating a circular dependency on the orchestrator.
 */

import type { EnemyAnimationState } from './enemy-animator.types';

/**
 * Decoded RGBA image extracted from an 8-bit PNG file by `decodePng`.
 */
export interface DecodedPng {
  /** Image width in pixels. */
  width: number;
  /** Image height in pixels. */
  height: number;
  /** Flat RGBA buffer in row-major order. */
  data: Buffer;
}

/**
 * Configuration options accepted by the enemy sprite sheet generator function.
 */
export interface SpriteSheetOptions {
  /** Override the output directory. */
  outputDir?: string;
}

/**
 * Describes one 128×128 frame cell inside the combined enemy sprite atlas.
 */
export interface SpriteFrameDescriptor {
  /** Yaw direction index (0–7). */
  direction: number;
  /** Animation state. */
  state: EnemyAnimationState;
  /** Frame index within the state. */
  frameIndex: number;
  /** Horizontal offset in the atlas. */
  x: number;
  /** Vertical offset in the atlas. */
  y: number;
  /** Cell width in pixels. */
  width: number;
  /** Cell height in pixels. */
  height: number;
}

/**
 * Metadata describing a generated enemy sprite atlas and its JSON manifest.
 */
export interface SpriteSheetResult {
  /** Absolute path to the combined atlas PNG. */
  atlasPath: string;
  /** Absolute path to the JSON manifest. */
  manifestPath: string;
  /** Atlas width in pixels. */
  width: number;
  /** Atlas height in pixels. */
  height: number;
  /** Per-cell frame descriptors. */
  frames: SpriteFrameDescriptor[];
}

/**
 * Absolute paths to the four generated 192×192 orthographic reference PNGs.
 */
export interface ReferenceSnapshotResult {
  /** Absolute path to the front-view PNG. */
  front: string;
  /** Absolute path to the back-view PNG. */
  back: string;
  /** Absolute path to the left-view PNG. */
  left: string;
  /** Absolute path to the right-view PNG. */
  right: string;
}

/**
 * Perceptual parity scores and opaque-pixel counts returned by the snapshot
 * comparator.
 */
export interface SnapshotComparison {
  /** Silhouette intersection-over-union in [0, 1]. */
  iou: number;
  /** Color-class overlap in [0, 1]. */
  colorSimilarity: number;
  /** Number of opaque pixels in the generated image. */
  generatedOpaque: number;
  /** Number of opaque pixels in the reference image. */
  referenceOpaque: number;
}