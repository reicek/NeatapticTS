/**
 * @module enemy-sprite.types
 *
 * Extracted types for the enemy sprite rendering module.
 */

import type { ControlledEnemy } from './enemy-controller.types';

/**
 * Camera transform consumed by the enemy billboard projector, expressed in
 * world cells and matching the classic raycasting camera plane convention.
 */
export interface NeatensteinEnemyCamera {
  /** Camera X position in world cells. */
  posX: number;
  /** Camera Y position in world cells. */
  posY: number;
  /** View direction X component. */
  dirX: number;
  /** View direction Y component. */
  dirY: number;
  /** Camera plane X component. */
  planeX: number;
  /** Camera plane Y component. */
  planeY: number;
}

/**
 * Screen-space projection result for a single enemy billboard, including
 * center coordinate, scale, perpendicular distance, and left/right edges.
 */
export interface NeatensteinEnemyProjection {
  /** Horizontal screen coordinate of the sprite center. */
  screenX: number;
  /** Projected screen size in pixels. */
  scale: number;
  /** Perpendicular distance from the camera plane to the sprite. */
  perpDist: number;
  /** Left edge of the projected sprite in screen pixels. */
  left: number;
  /** Right edge of the projected sprite in screen pixels. */
  right: number;
  /** `false` when the sprite is behind the camera or otherwise invisible. */
  visible: boolean;
}

/**
 * Directional light source applied to enemy billboard pixels, pairing a
 * normalized light direction with diffuse intensity and an optional ambient term.
 */
export interface NeatensteinDirectionalLight {
  /** Light direction X component (should be normalized). */
  dirX: number;
  /** Light direction Y component (should be normalized). */
  dirY: number;
  /** Intensity multiplier for the diffuse term. */
  intensity: number;
  /** Optional ambient term in `[0, 1]`; defaults to 0.25. */
  ambient?: number;
}

/**
 * Runtime enemy descriptor plus the extra timing fields the renderer needs for
 * spawn and death effects.
 */
export interface NeatensteinEnemyBillboard {
  /** Controlled enemy AI/animation descriptor. */
  enemy: ControlledEnemy;
  /** Milliseconds since this enemy spawned; drives the teal force-field. */
  spawnElapsedMs: number;
  /** Milliseconds to use for non-death animation frame selection. */
  animationElapsedMs: number;
}

/**
 * Minimal canvas-like context consumed by the CPU billboard renderer, abstracting
 * the putImageData surface so the same sprite code runs in Node tests and
 * browser workers.
 */
export interface NeatensteinSpriteRenderContext {
  /**
   * Flush an ImageData-like payload to the canvas.
   *
   * @param imageData - Object with `data`, `width`, and `height`.
   * @param dx - Destination X coordinate.
   * @param dy - Destination Y coordinate.
   */
  putImageData(
    imageData: { data: Uint8ClampedArray; width: number; height: number },
    dx: number,
    dy: number,
  ): void;
}

/**
 * Runtime sprite atlas produced by the build-time pipeline and consumed by the
 * CPU billboard renderer.
 *
 * The layout matches `generate-enemy-sprites.ts`:
 *
 * - each cell is `cellSize × cellSize` pixels
 * - frames for a state are laid out horizontally (`frameIndex * cellSize`)
 * - rows are grouped by direction: `y = (direction * stateCount + stateIndex) * cellSize`
 */
export interface NeatensteinSpriteAtlas {
  /** Atlas width in pixels. */
  width: number;
  /** Atlas height in pixels. */
  height: number;
  /** Pixel size of one frame cell. */
  cellSize: number;
  /** Number of yaw directions in the atlas. */
  directions: number;
  /** Number of animation states in the atlas. */
  states: number;
  /** Flat RGBA pixel buffer in row-major order. */
  data: Uint8ClampedArray;
}