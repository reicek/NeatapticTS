/**
 * Enemy wireframe sprite rendering for the Neatenstein neon raycaster.
 *
 * Each sprite is projected from world space into screen space using the inverse
 * camera matrix, clipped against the per-column z-buffer, and drawn as a
 * vertical neon bar for every visible screen column.
 *
 * This module is intentionally CPU-friendly:
 *
 * - projection is scalar and deterministic
 * - clipping is delegated to the z-buffer helper
 * - color parsing is cached
 * - framebuffer writes are direct RGBA byte writes
 * - the full framebuffer is flushed once per rendered sprite
 *
 * The sprite renderer uses the same camera matrix convention as classic
 * Wolfenstein-style raycasters: `dir` is the forward vector and `plane` is the
 * camera plane vector.
 *
 * @module
 */

import { clampInt } from './framebuffer';
import {
  clipNeatensteinSpriteSpan,
  type NeatensteinSpriteClip,
} from './zbuffer';
import { NEATENSTEIN_FLOOR_FOV_RADIANS } from './floor';
import { buildVoxelEnemy } from '../../../neatenstein/scripts/voxel-enemy';
import {
  renderVoxelSnapshot,
  type VoxelSnapshot,
} from '../../../neatenstein/scripts/snapshot-renderer';
import {
  ENEMY_ANIMATION_FRAME_COUNTS,
  type EnemyAnimationState,
} from '../../../neatenstein/scripts/enemy-animator';

/**
 * Number of RGBA channels per framebuffer pixel.
 */
const NEATENSTEIN_RGBA_CHANNELS = 4;

/**
 * Minimum absolute determinant accepted for the inverse camera matrix.
 *
 * A determinant close to zero means the camera direction and plane are
 * degenerate, which would make projection unstable.
 */
const NEATENSTEIN_CAMERA_DETERMINANT_EPSILON = 1e-9;

/**
 * World-space size of a sprite in grid cells.
 *
 * The projected screen size is:
 *
 * ```ts
 * focalLength / perpDist * NEATENSTEIN_SPRITE_WORLD_SIZE
 * ```
 *
 * where `focalLength` is derived from {@link NEATENSTEIN_FLOOR_FOV_RADIANS}.
 * Keeping this value in world units makes enemies scale consistently as they
 * move toward or away from the camera.
 */
export const NEATENSTEIN_SPRITE_WORLD_SIZE = 0.5;

/**
 * Minimum perpendicular distance at which a sprite is drawn.
 *
 * Sprites closer than this are considered too close to the camera plane and are
 * culled to avoid division-by-zero and unstable projection.
 */
export const NEATENSTEIN_SPRITE_NEAR_CLIP = 0.1;

/**
 * Fraction of projected sprite scale used as horizontal sprite thickness.
 *
 * A value of 1.0 makes the projected sprite width match its height, which
 * matches the square aspect of the 192×192 reference robot silhouettes.
 */
export const NEATENSTEIN_SPRITE_THICKNESS_RATIO = 1.0;

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
  /** Sprite yaw in radians. 0 points toward +X; a {@link resolveNeatensteinEnemyFrame} helper uses this together with the camera position to select a pre-rendered voxel frame. */
  facing?: number;
  /** Current animation state used to pick a frame from the voxel atlas. */
  animationState?: EnemyAnimationState;
  /** Animation frame index. Currently only frame 0 is pre-rendered in the runtime atlas. */
  frameIndex?: number;
  /** Optional enemy type index for hue selection by higher-level callers. */
  type?: number;
}

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
interface ResolvedSpriteFramebufferSize {
  /** Framebuffer width in pixels. */
  width: number;
  /** Framebuffer height in pixels. */
  height: number;
}

/**
 * Return whether a number is a positive finite integer dimension.
 *
 * @param value - Candidate dimension.
 * @returns Whether the value is usable as a framebuffer/canvas dimension.
 */
function isPositiveIntegerDimension(value: number): boolean {
  return Number.isInteger(value) && value > 0;
}

/**
 * Return whether a number is finite and greater than zero.
 *
 * @param value - Candidate scalar.
 * @returns Whether the value is positive and finite.
 */
function isPositiveFinite(value: number): boolean {
  return Number.isFinite(value) && value > 0;
}

/**
 * Return a reusable invisible projection result.
 *
 * @param perpDist - Perpendicular distance associated with the rejected sprite.
 * @returns Invisible sprite projection.
 */
function createInvisibleSpriteProjection(
  perpDist = Number.POSITIVE_INFINITY,
): NeatensteinSpriteProjection {
  return {
    screenX: -1,
    scale: 0,
    perpDist,
    left: -1,
    right: -1,
    visible: false,
  };
}

/**
 * Resolve framebuffer dimensions from explicit dimensions or available buffers.
 *
 * A flat RGBA buffer length alone cannot uniquely identify rectangular
 * dimensions. Therefore:
 *
 * - explicit width/height are preferred when provided
 * - the z-buffer length is used as width when available
 * - legacy square inference is used only as a final fallback
 *
 * @param framebuffer - Flat RGBA framebuffer.
 * @param preferredWidth - Optional known framebuffer width.
 * @param preferredHeight - Optional known framebuffer height.
 * @returns Resolved framebuffer dimensions.
 */
function resolveFramebufferSize(
  framebuffer: Uint8ClampedArray,
  preferredWidth?: number,
  preferredHeight?: number,
): ResolvedSpriteFramebufferSize {
  const pixelCount = Math.floor(framebuffer.length / NEATENSTEIN_RGBA_CHANNELS);

  // Best case: caller supplied exact dimensions.
  if (
    preferredWidth !== undefined &&
    preferredHeight !== undefined &&
    isPositiveIntegerDimension(preferredWidth) &&
    isPositiveIntegerDimension(preferredHeight) &&
    preferredWidth * preferredHeight <= pixelCount
  ) {
    return {
      width: preferredWidth,
      height: preferredHeight,
    };
  }

  // Common sprite-render path: z-buffer length gives the canvas width.
  if (
    preferredWidth !== undefined &&
    isPositiveIntegerDimension(preferredWidth) &&
    pixelCount >= preferredWidth
  ) {
    return {
      width: preferredWidth,
      height: Math.floor(pixelCount / preferredWidth),
    };
  }

  // Less common case: height is known but width is not.
  if (
    preferredHeight !== undefined &&
    isPositiveIntegerDimension(preferredHeight) &&
    pixelCount >= preferredHeight
  ) {
    return {
      width: Math.floor(pixelCount / preferredHeight),
      height: preferredHeight,
    };
  }

  // Legacy fallback for older square-framebuffer callers.
  const side = Math.floor(Math.sqrt(pixelCount));
  return {
    width: side,
    height: side,
  };
}

/**
 * Return whether a framebuffer size can be drawn into safely.
 *
 * @param size - Resolved framebuffer dimensions.
 * @returns Whether width and height are drawable.
 */
function isDrawableFramebufferSize(
  size: ResolvedSpriteFramebufferSize,
): boolean {
  return (
    isPositiveIntegerDimension(size.width) &&
    isPositiveIntegerDimension(size.height)
  );
}

/**
 * Return whether a projection has usable finite screen-space values.
 *
 * @param projection - Projection to validate.
 * @returns Whether the projection can be used for clipping/rendering.
 */
function isDrawableSpriteProjection(
  projection: NeatensteinSpriteProjection,
): boolean {
  return (
    projection.visible &&
    Number.isFinite(projection.screenX) &&
    isPositiveFinite(projection.scale) &&
    isPositiveFinite(projection.perpDist) &&
    Number.isFinite(projection.left) &&
    Number.isFinite(projection.right) &&
    projection.right >= projection.left
  );
}

/**
 * Discrete yaw directions stored in the enemy voxel atlas.
 */
const NEATENSTEIN_VOXEL_ATLAS_YAW_STEPS = 8;

/**
 * Full rotation in radians.
 */
const NEATENSTEIN_FULL_ROTATION_RADIANS = Math.PI * 2;

/**
 * Angular step between adjacent yaw atlas entries, in radians.
 */
const NEATENSTEIN_VOXEL_ATLAS_YAW_STEP_RADIANS =
  NEATENSTEIN_FULL_ROTATION_RADIANS / NEATENSTEIN_VOXEL_ATLAS_YAW_STEPS;

/**
 * Size in pixels of each pre-rendered voxel frame in the runtime atlas.
 */
const NEATENSTEIN_VOXEL_ATLAS_FRAME_SIZE = 128;

/**
 * Pre-rendered runtime atlas: one yaw strip per animation state, each strip
 * holding {@link NEATENSTEIN_VOXEL_ATLAS_YAW_STEPS} frames at index 0.
 */
const NEATENSTEIN_VOXEL_ATLAS: Record<
  EnemyAnimationState,
  readonly VoxelSnapshot[]
> = buildNeatensteinVoxelAtlas();

/**
 * Build the canonical enemy voxel grid once at module load time.
 *
 * The runtime atlas holds 8 camera-relative yaw snapshots for every animation
 * state, which keeps the per-frame render path allocation-free.
 *
 * @returns Record mapping each animation state to its pre-rendered yaw frames.
 */
function buildNeatensteinVoxelAtlas(): Record<
  EnemyAnimationState,
  readonly VoxelSnapshot[]
> {
  const grid = buildVoxelEnemy();
  const states = Object.keys(
    ENEMY_ANIMATION_FRAME_COUNTS,
  ) as EnemyAnimationState[];

  const atlas = {} as Record<EnemyAnimationState, readonly VoxelSnapshot[]>;

  for (const state of states) {
    const frames: VoxelSnapshot[] = [];
    for (
      let yawIndex = 0;
      yawIndex < NEATENSTEIN_VOXEL_ATLAS_YAW_STEPS;
      yawIndex += 1
    ) {
      frames.push(
        renderVoxelSnapshot(grid, yawIndex, {
          width: NEATENSTEIN_VOXEL_ATLAS_FRAME_SIZE,
          height: NEATENSTEIN_VOXEL_ATLAS_FRAME_SIZE,
        }),
      );
    }
    atlas[state] = frames;
  }

  return atlas;
}

/**
 * Convert a relative yaw angle to the nearest atlas yaw index.
 *
 * @param relativeYaw - Camera-relative yaw in radians, already normalized to
 *   the sprite's facing direction.
 * @returns Integer index in `[0, NEATENSTEIN_VOXEL_ATLAS_YAW_STEPS - 1]`.
 */
function yawIndexFromRelativeYaw(relativeYaw: number): number {
  const normalized =
    ((relativeYaw % NEATENSTEIN_FULL_ROTATION_RADIANS) +
      NEATENSTEIN_FULL_ROTATION_RADIANS) %
    NEATENSTEIN_FULL_ROTATION_RADIANS;

  const rawIndex = Math.round(
    normalized / NEATENSTEIN_VOXEL_ATLAS_YAW_STEP_RADIANS,
  );
  return rawIndex % NEATENSTEIN_VOXEL_ATLAS_YAW_STEPS;
}

/**
 * Resolve the pre-rendered voxel frame for an enemy sprite facing a camera.
 *
 * Returns `null` when the sprite does not carry the animation or facing data
 * required by the runtime atlas.
 *
 * @param sprite - Enemy sprite with optional facing and animation fields.
 * @param camera - Camera whose position determines the relative yaw.
 * @returns The matching {@link VoxelSnapshot}, or `null` if not resolvable.
 */
export function resolveNeatensteinEnemyFrame(
  sprite: NeatensteinSprite,
  camera: NeatensteinCamera,
): VoxelSnapshot | null {
  if (
    sprite.animationState === undefined ||
    sprite.facing === undefined ||
    !Number.isFinite(sprite.facing)
  ) {
    return null;
  }

  if (
    !Number.isFinite(sprite.worldX) ||
    !Number.isFinite(sprite.worldY) ||
    !Number.isFinite(camera.posX) ||
    !Number.isFinite(camera.posY)
  ) {
    return null;
  }

  const cameraRelativeYaw = Math.atan2(
    camera.posY - sprite.worldY,
    camera.posX - sprite.worldX,
  );
  const relativeYaw = cameraRelativeYaw - sprite.facing;
  const yawIndex = yawIndexFromRelativeYaw(relativeYaw);

  const stateFrames = NEATENSTEIN_VOXEL_ATLAS[sprite.animationState];
  if (stateFrames === undefined || stateFrames[yawIndex] === undefined) {
    return null;
  }

  return stateFrames[yawIndex];
}

/**
 * Project a world-space sprite into screen coordinates.
 *
 * The projection uses the inverse camera matrix, matching the classic
 * raycasting sprite transform. Sprites behind the camera plane, too close to
 * the camera, or produced by an invalid camera matrix are marked invisible.
 *
 * @param sprite - World-space sprite position.
 * @param camera - Current camera transform.
 * @param canvasWidth - Canvas width in backing-store pixels.
 * @param canvasHeight - Canvas height in backing-store pixels.
 * @returns The screen-space projection, including the span `[left, right]`.
 *
 * @example
 * ```ts
 * const projection = projectNeatensteinSprite(
 *   { worldX: 5, worldY: 5, type: 0 },
 *   { posX: 1, posY: 1, dirX: 1, dirY: 0, planeX: 0, planeY: 0.66 },
 *   640,
 *   480,
 * );
 * ```
 */
export function projectNeatensteinSprite(
  sprite: NeatensteinSprite,
  camera: NeatensteinCamera,
  canvasWidth: number,
  canvasHeight: number,
): NeatensteinSpriteProjection {
  // Invalid render dimensions cannot produce a meaningful projection.
  if (!isPositiveFinite(canvasWidth) || !isPositiveFinite(canvasHeight)) {
    return createInvisibleSpriteProjection();
  }

  // Reject malformed camera/sprite values before they can produce NaN paths.
  if (
    !Number.isFinite(sprite.worldX) ||
    !Number.isFinite(sprite.worldY) ||
    !Number.isFinite(camera.posX) ||
    !Number.isFinite(camera.posY) ||
    !Number.isFinite(camera.dirX) ||
    !Number.isFinite(camera.dirY) ||
    !Number.isFinite(camera.planeX) ||
    !Number.isFinite(camera.planeY)
  ) {
    return createInvisibleSpriteProjection();
  }

  // Translate sprite into camera-relative world space.
  const relativeX = sprite.worldX - camera.posX;
  const relativeY = sprite.worldY - camera.posY;

  // Invert the 2x2 camera matrix. A near-zero determinant means the camera
  // direction and plane are degenerate, so projection would be unstable.
  const determinant = camera.planeX * camera.dirY - camera.dirX * camera.planeY;
  if (
    !Number.isFinite(determinant) ||
    Math.abs(determinant) < NEATENSTEIN_CAMERA_DETERMINANT_EPSILON
  ) {
    return createInvisibleSpriteProjection();
  }

  const invDet = 1 / determinant;

  const transformX =
    invDet * (camera.dirY * relativeX - camera.dirX * relativeY);
  const transformY =
    invDet * (-camera.planeY * relativeX + camera.planeX * relativeY);

  // transformY is the perpendicular camera-space depth.
  const perpDist = transformY;
  if (!isPositiveFinite(perpDist) || perpDist <= NEATENSTEIN_SPRITE_NEAR_CLIP) {
    return createInvisibleSpriteProjection(perpDist);
  }

  const focalLength =
    canvasHeight / 2 / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
  const screenX = (canvasWidth / 2) * (1 + transformX / transformY);
  const scale =
    Math.abs(focalLength / transformY) * NEATENSTEIN_SPRITE_WORLD_SIZE;
  const halfThickness = (scale * NEATENSTEIN_SPRITE_THICKNESS_RATIO) / 2;
  const left = screenX - halfThickness;
  const right = screenX + halfThickness;

  return {
    screenX,
    scale,
    perpDist,
    left,
    right,
    visible: true,
  };
}

/**
 * Build a {@link NeatensteinSpriteClip} for a projected sprite.
 *
 * This helper combines projection and z-buffer clipping into one call so the
 * renderer can iterate only the columns that are actually visible.
 *
 * @param sprite - World-space sprite.
 * @param camera - Current camera transform.
 * @param canvasWidth - Canvas width in backing-store pixels.
 * @param canvasHeight - Canvas height in backing-store pixels.
 * @param zBuffer - Per-column wall-depth buffer.
 * @returns The projection plus clipped span and visible column indices.
 */
export function clipNeatensteinSprite(
  sprite: NeatensteinSprite,
  camera: NeatensteinCamera,
  canvasWidth: number,
  canvasHeight: number,
  zBuffer: Readonly<Float32Array>,
): NeatensteinSpriteProjection & NeatensteinSpriteClip {
  const projection = projectNeatensteinSprite(
    sprite,
    camera,
    canvasWidth,
    canvasHeight,
  );

  if (!isDrawableSpriteProjection(projection) || zBuffer.length === 0) {
    return {
      ...projection,
      left: 0,
      right: -1,
      visibleColumns: [],
    };
  }

  const clip = clipNeatensteinSpriteSpan(
    zBuffer,
    projection.left,
    projection.right,
    projection.perpDist,
  );

  return { ...projection, ...clip };
}

/**
 * Return precomputed visible columns from a projection if present.
 *
 * This lets callers pass the result of {@link clipNeatensteinSprite} directly
 * into {@link renderNeatensteinSprite} without recomputing clipping.
 *
 * @param projection - Projection that may also contain clip data.
 * @returns Visible columns if already available, otherwise `null`.
 */
function getPrecomputedVisibleColumns(
  projection: NeatensteinSpriteProjection,
): readonly number[] | null {
  const maybeClip = projection as NeatensteinSpriteProjection &
    Partial<NeatensteinSpriteClip>;

  return Array.isArray(maybeClip.visibleColumns)
    ? maybeClip.visibleColumns
    : null;
}

/**
 * Copy one column of a pre-rendered voxel frame onto a screen column.
 *
 * `visibleColumns` already guarantees that the screen column is closer than
 * the wall at that X, so this helper does not re-check the z-buffer. It skips
 * transparent frame pixels (alpha == 0) and clamps vertical writes to the
 * framebuffer bounds.
 *
 * @param framebuffer - Flat RGBA framebuffer to write into.
 * @param width - Framebuffer width in pixels.
 * @param height - Framebuffer height in pixels.
 * @param screenColumn - Horizontal framebuffer column to write.
 * @param drawStart - Top screen row of the projected sprite, inclusive.
 * @param drawEnd - Bottom screen row of the projected sprite, exclusive.
 * @param frame - Pre-rendered voxel snapshot.
 * @param frameX - Column of the voxel frame to sample.
 */
function renderNeatensteinVoxelSpriteColumn(
  framebuffer: Uint8ClampedArray,
  width: number,
  height: number,
  screenColumn: number,
  drawStart: number,
  drawEnd: number,
  frame: VoxelSnapshot,
  frameX: number,
): void {
  const clampedStart = clampInt(drawStart, 0, height);
  const clampedEnd = clampInt(drawEnd, 0, height);
  if (clampedStart >= clampedEnd || screenColumn < 0 || screenColumn >= width) {
    return;
  }

  const frameHeight = frame.height;
  const frameWidth = frame.width;
  const frameData = frame.data;
  if (frameHeight === 0 || frameWidth === 0) {
    return;
  }

  const safeFrameX = clampInt(frameX, 0, frameWidth - 1);
  const spriteHeightPixels = clampedEnd - clampedStart;

  for (let rowOffset = 0; rowOffset < spriteHeightPixels; rowOffset += 1) {
    const screenY = clampedStart + rowOffset;
    const v = rowOffset / (drawEnd - drawStart);
    const frameY = Math.floor(v * (frameHeight - 1));
    const safeFrameY = clampInt(frameY, 0, frameHeight - 1);

    const frameOffset =
      (safeFrameY * frameWidth + safeFrameX) * NEATENSTEIN_RGBA_CHANNELS;
    const alpha = frameData[frameOffset + 3];
    if (alpha === 0) {
      continue;
    }

    const screenOffset =
      (screenY * width + screenColumn) * NEATENSTEIN_RGBA_CHANNELS;
    framebuffer[screenOffset] = frameData[frameOffset];
    framebuffer[screenOffset + 1] = frameData[frameOffset + 1];
    framebuffer[screenOffset + 2] = frameData[frameOffset + 2];
    framebuffer[screenOffset + 3] = alpha;
  }
}

/**
 * Render a projected sprite into the CPU ImageData framebuffer with z-buffer
 * occlusion.
 *
 * Only sprite columns closer than the wall distance at the same screen column
 * are drawn. The framebuffer is flushed with a single `putImageData` call.
 *
 * The framebuffer width is inferred from `zBuffer.length`, which correctly
 * supports rectangular render targets such as `640x480`.
 *
 * @param framebuffer - Flat RGBA framebuffer.
 * @param zBuffer - Per-column wall-depth buffer filled by the wall pass.
 * @param projection - Screen-space projection from
 *   {@link projectNeatensteinSprite} or {@link clipNeatensteinSprite}.
 * @param source - Pre-rendered {@link VoxelSnapshot} to draw, or a legacy color
 *   string. Color strings are ignored; the renderer now requires a voxel frame.
 * @param ctx - Canvas-like context with `putImageData`.
 */
export function renderNeatensteinSprite(
  framebuffer: Uint8ClampedArray,
  zBuffer: Readonly<Float32Array>,
  projection: NeatensteinSpriteProjection,
  source: string | VoxelSnapshot,
  ctx: NeatensteinSpriteRenderContext,
): void {
  if (!isDrawableSpriteProjection(projection) || zBuffer.length === 0) {
    return;
  }

  if (typeof source === 'string') {
    return;
  }

  // In the normal renderer path, zBuffer.length is the framebuffer width.
  const { width, height } = resolveFramebufferSize(
    framebuffer,
    zBuffer.length,
    undefined,
  );

  if (!isDrawableFramebufferSize({ width, height })) {
    return;
  }

  // Use precomputed clipping if the caller passed clipNeatensteinSprite(...).
  const visibleColumns =
    getPrecomputedVisibleColumns(projection) ??
    clipNeatensteinSpriteSpan(
      zBuffer,
      projection.left,
      projection.right,
      projection.perpDist,
    ).visibleColumns;

  if (visibleColumns.length === 0) {
    return;
  }

  const spanPixels = projection.right - projection.left;

  // Center sprite vertically on the horizon/midline.
  const halfScale = projection.scale / 2;
  const centerY = height / 2;
  const drawStart = Math.floor(centerY - halfScale);
  const drawEnd = Math.floor(centerY + halfScale);

  for (const column of visibleColumns) {
    const u = spanPixels > 0 ? (column - projection.left) / spanPixels : 0;
    const frameX = Math.floor(u * (source.width - 1));

    renderNeatensteinVoxelSpriteColumn(
      framebuffer,
      width,
      height,
      column,
      drawStart,
      drawEnd,
      source,
      frameX,
    );
  }

  ctx.putImageData({ data: framebuffer, width, height }, 0, 0);
}

/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyResolveFramebufferSize = resolveFramebufferSize;

/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyRenderNeatensteinVoxelSpriteColumn =
  renderNeatensteinVoxelSpriteColumn;
