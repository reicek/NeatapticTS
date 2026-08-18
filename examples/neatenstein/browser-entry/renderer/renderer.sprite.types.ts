/**
 * Shared sprite type definitions extracted from the sprite renderer modules.
 *
 * Centralises the interfaces and type aliases consumed by
 * {@link module:./sprites}, {@link module:./sprites.projection.utils},
 * {@link module:./sprites.column.utils}, and {@link module:./sprites.atlas.utils}.
 *
 * @module
 */

import type { VoxelSnapshot } from '../shared/snapshot-renderer';
import type { EnemyAnimationState } from '../shared/enemy-animator';

/**
 * Minimal canvas-like context consumed by the CPU sprite renderer.
 *
 * Only `putImageData` is required, mirroring the narrow context contract used
 * by the CPU wall renderer.
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
 * Camera transform consumed by the sprite projector.
 */
export interface NeatensteinCamera {
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
 * World-space sprite to project.
 */
export interface NeatensteinSprite {
  /** Sprite X position in world cells. */
  worldX: number;
  /** Sprite Y position in world cells. */
  worldY: number;
  /** Sprite yaw in radians. 0 points toward +X; used together with the camera position to select a pre-rendered voxel frame. */
  facing?: number;
  /** Current animation state used to pick a frame from the voxel atlas. */
  animationState?: EnemyAnimationState;
  /** Animation frame index. Currently only frame 0 is pre-rendered in the runtime atlas. */
  frameIndex?: number;
  /** Optional enemy type index for hue selection by higher-level callers. */
  type?: number;
  /** Sim tick counter driving the walk cycle (stand → walk1 → stand → walk2). When provided, overrides position-based walk alternation. */
  walkTick?: number;
  /** Remaining shoot-blink ticks; when > 0 the renderer composites the shoot upper body over the walk lower body. */
  shootBlinkTicks?: number;
  /** Optional team color [r, g, b] to swap palette indices 5/6/7 at runtime, preserving alpha. */
  teamColor?: readonly [number, number, number];
  /** Elapsed milliseconds since the de-rez death animation began (death state only). */
  deRezElapsedMs?: number;
  /** Total de-rez animation duration in milliseconds (death state only). */
  deRezDurationMs?: number;
  /** Per-enemy deterministic seed for the scattered de-rez dissolution pattern. */
  seed?: number;
}

/**
 * De-rez death animation state passed through the sprite render pipeline.
 *
 * When provided, the column renderer dissolves pixels whose deterministic
 * noise hash falls below the animation progress `t`, and tints surviving
 * pixels toward the enemy death color.
 */
export interface NeatensteinDerezState {
  /** Elapsed milliseconds since the de-rez death animation began. */
  elapsedMs: number;
  /** Total de-rez animation duration in milliseconds. */
  durationMs: number;
  /** Per-enemy deterministic seed for the scattered dissolution pattern. */
  seed: number;
}

/**
 * Encoded robot sprite frame: rows of palette indices into
 * {@link ../../robot-sprite-data.js}.
 */
export type EncodedRobotSpriteFrame = readonly (readonly number[])[];

/**
 * Renderable sprite source: a pre-rendered voxel snapshot, an encoded robot
 * frame, or a legacy color string (ignored).
 */
export type NeatensteinSpriteSource =
  VoxelSnapshot | EncodedRobotSpriteFrame | string;

/**
 * Screen-space projection result for a single sprite.
 */
export interface NeatensteinSpriteProjection {
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
 * Resolved framebuffer dimensions.
 */
export interface ResolvedSpriteFramebufferSize {
  /** Framebuffer width in pixels. */
  width: number;
  /** Framebuffer height in pixels. */
  height: number;
}
