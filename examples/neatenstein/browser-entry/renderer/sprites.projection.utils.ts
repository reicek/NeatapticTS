/**
 * Sprite projection executors extracted from the sprite renderer.
 *
 * Contains pure leaf functions that project world-space sprites into screen
 * space, clip them against the z-buffer, resolve framebuffer dimensions, and
 * validate projection results. These executors accept scalar parameters and
 * do not create per-pixel allocations.
 *
 * @module
 */

import { NEATENSTEIN_RENDER_DISTANCE_CAP } from './framebuffer';
import {
  clipNeatensteinSpriteSpan,
  type NeatensteinSpriteClip,
} from './zbuffer';
import { NEATENSTEIN_FLOOR_FOV_RADIANS } from './floor';
import {
  isPositiveFinite,
  isPositiveIntegerDimension,
} from './sprites.guards.utils';
import {
  RGBA_CHANNELS,
  NEATENSTEIN_CAMERA_DETERMINANT_EPSILON,
  NEATENSTEIN_SPRITE_WORLD_SIZE,
  NEATENSTEIN_SPRITE_NEAR_CLIP,
  NEATENSTEIN_SPRITE_THICKNESS_RATIO,
} from './renderer.sprite.constants';
import type {
  NeatensteinCamera,
  NeatensteinSprite,
  NeatensteinSpriteProjection,
  ResolvedSpriteFramebufferSize,
} from './renderer.sprite.types';

// Re-export previously-public symbols that moved to dedicated files.
export {
  NEATENSTEIN_SPRITE_WORLD_SIZE,
  NEATENSTEIN_SPRITE_NEAR_CLIP,
  NEATENSTEIN_SPRITE_THICKNESS_RATIO,
} from './renderer.sprite.constants';
export type { NeatensteinSpriteProjection } from './renderer.sprite.types';

/**
 * Return a reusable invisible projection result.
 *
 * @param perpDist - Perpendicular distance associated with the rejected sprite.
 * @returns Invisible sprite projection.
 */
export function createInvisibleSpriteProjection(
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
export function resolveFramebufferSize(
  framebuffer: Uint8ClampedArray,
  preferredWidth?: number,
  preferredHeight?: number,
): ResolvedSpriteFramebufferSize {
  const pixelCount = Math.floor(framebuffer.length / RGBA_CHANNELS);

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
export function isDrawableFramebufferSize(
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
export function isDrawableSpriteProjection(
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

  // Cull sprites beyond the shared render-distance cap.
  if (perpDist >= NEATENSTEIN_RENDER_DISTANCE_CAP) {
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

  return { ...projection, visibleColumns: clip.visibleColumns };
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
export function getPrecomputedVisibleColumns(
  projection: NeatensteinSpriteProjection,
): readonly number[] | null {
  const maybeClip = projection as NeatensteinSpriteProjection &
    Partial<NeatensteinSpriteClip>;

  return Array.isArray(maybeClip.visibleColumns)
    ? maybeClip.visibleColumns
    : null;
}
