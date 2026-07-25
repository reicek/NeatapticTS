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
 * Fully opaque alpha value written for neon sprite pixels.
 */
const NEATENSTEIN_SPRITE_ALPHA = 255;

/**
 * World-space size of a sprite in grid cells.
 *
 * The projected screen size is:
 *
 * ```ts
 * canvasHeight / perpDist * NEATENSTEIN_SPRITE_WORLD_SIZE
 * ```
 *
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
 * Fraction of projected sprite scale used as horizontal neon-bar thickness.
 *
 * The rendered sprite is a vertical wireframe-like bar centered on the
 * projected screen X coordinate.
 */
export const NEATENSTEIN_SPRITE_THICKNESS_RATIO = 0.4;

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
 * Parsed RGB triplet from a `#rrggbb` hex color string.
 */
interface ParsedRgb {
  /** Red channel in `[0, 255]`. */
  r: number;
  /** Green channel in `[0, 255]`. */
  g: number;
  /** Blue channel in `[0, 255]`. */
  b: number;
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
 * Cache of parsed sprite colors.
 *
 * Sprite colors are normally reused across many frames, so parsing once avoids
 * repeated string work in the render path.
 */
const SPRITE_COLOR_CACHE = new Map<string, ParsedRgb>();

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
 * Parse a strict `#rrggbb` hex color string into an RGB triplet.
 *
 * @param hex - Color string in `#rrggbb` format.
 * @returns Parsed `{ r, g, b }` channels.
 * @throws {Error} When the string is not a valid `#rrggbb` color.
 */
function parseHexColor(hex: string): ParsedRgb {
  const cached = SPRITE_COLOR_CACHE.get(hex);
  if (cached !== undefined) {
    return cached;
  }

  const match = /^#([0-9a-fA-F]{6})$/.exec(hex);
  if (match === null) {
    throw new Error(`Expected #rrggbb hex color, got "${hex}"`);
  }

  const digits = match[1];
  const rgb = {
    r: Number.parseInt(digits.slice(0, 2), 16),
    g: Number.parseInt(digits.slice(2, 4), 16),
    b: Number.parseInt(digits.slice(4, 6), 16),
  };

  SPRITE_COLOR_CACHE.set(hex, rgb);
  return rgb;
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

  const screenX = (canvasWidth / 2) * (1 + transformX / transformY);
  const scale =
    Math.abs(canvasHeight / transformY) * NEATENSTEIN_SPRITE_WORLD_SIZE;
  const halfThickness = (scale * NEATENSTEIN_SPRITE_THICKNESS_RATIO) / 2;
  const left = screenX - halfThickness;
  const right = screenX + halfThickness;

  if (
    !Number.isFinite(screenX) ||
    !isPositiveFinite(scale) ||
    !Number.isFinite(left) ||
    !Number.isFinite(right)
  ) {
    return createInvisibleSpriteProjection(perpDist);
  }

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
 * Write a vertical RGB sprite column into a flat RGBA framebuffer.
 *
 * This internal helper assumes color and framebuffer dimensions have already
 * been resolved by the caller.
 *
 * @param framebuffer - Flat RGBA framebuffer.
 * @param width - Framebuffer width in pixels.
 * @param height - Framebuffer height in pixels.
 * @param column - Horizontal column index to write.
 * @param drawStart - Top row of the sprite stripe, inclusive.
 * @param drawEnd - Bottom row of the sprite stripe, exclusive.
 * @param color - Parsed RGB sprite color.
 */
function renderNeatensteinSpriteColumnRgb(
  framebuffer: Uint8ClampedArray,
  width: number,
  height: number,
  column: number,
  drawStart: number,
  drawEnd: number,
  color: ParsedRgb,
): void {
  if (
    !isPositiveIntegerDimension(width) ||
    !isPositiveIntegerDimension(height)
  ) {
    return;
  }

  // Do not clamp invalid columns to the edge; that can create false edge pixels.
  if (!Number.isFinite(column)) {
    return;
  }

  const x = Math.trunc(column);
  if (x < 0 || x >= width) {
    return;
  }

  const clampedStart = clampInt(drawStart, 0, height);
  const clampedEnd = clampInt(drawEnd, 0, height);

  if (clampedStart >= clampedEnd) {
    return;
  }

  for (let row = clampedStart; row < clampedEnd; row += 1) {
    const offset = (row * width + x) * NEATENSTEIN_RGBA_CHANNELS;

    // Defensive guard for malformed buffers or mismatched dimensions.
    if (offset + 3 >= framebuffer.length) {
      break;
    }

    framebuffer[offset] = color.r;
    framebuffer[offset + 1] = color.g;
    framebuffer[offset + 2] = color.b;
    framebuffer[offset + 3] = NEATENSTEIN_SPRITE_ALPHA;
  }
}

/**
 * Render a single sprite column into the CPU ImageData framebuffer.
 *
 * This public helper is useful for tests and low-level callers. For full sprite
 * rendering, prefer {@link renderNeatensteinSprite}, which parses color and
 * resolves dimensions once before drawing all visible columns.
 *
 * If `framebufferWidth` and `framebufferHeight` are omitted, the function falls
 * back to legacy square framebuffer inference.
 *
 * @param framebuffer - Flat RGBA framebuffer.
 * @param column - Horizontal column index to write.
 * @param drawStart - Top row of the sprite stripe, inclusive.
 * @param drawEnd - Bottom row of the sprite stripe, exclusive.
 * @param hexColor - Sprite color as `#rrggbb`.
 * @param framebufferWidth - Optional explicit framebuffer width in pixels.
 * @param framebufferHeight - Optional explicit framebuffer height in pixels.
 *
 * @example
 * ```ts
 * renderNeatensteinSpriteColumn(framebuffer, 4, 2, 6, '#00bfff', 640, 480);
 * ```
 */
export function renderNeatensteinSpriteColumn(
  framebuffer: Uint8ClampedArray,
  column: number,
  drawStart: number,
  drawEnd: number,
  hexColor: string,
  framebufferWidth?: number,
  framebufferHeight?: number,
): void {
  const color = parseHexColor(hexColor);
  const { width, height } = resolveFramebufferSize(
    framebuffer,
    framebufferWidth,
    framebufferHeight,
  );

  renderNeatensteinSpriteColumnRgb(
    framebuffer,
    width,
    height,
    column,
    drawStart,
    drawEnd,
    color,
  );
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
 * @param color - Neon color as `#rrggbb`.
 * @param ctx - Canvas-like context with `putImageData`.
 */
export function renderNeatensteinSprite(
  framebuffer: Uint8ClampedArray,
  zBuffer: Readonly<Float32Array>,
  projection: NeatensteinSpriteProjection,
  color: string,
  ctx: NeatensteinSpriteRenderContext,
): void {
  if (!isDrawableSpriteProjection(projection) || zBuffer.length === 0) {
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

  // Parse once per sprite, not once per column.
  const parsedColor = parseHexColor(color);

  // Center sprite vertically on the horizon/midline.
  const halfScale = projection.scale / 2;
  const centerY = height / 2;
  const drawStart = Math.floor(centerY - halfScale);
  const drawEnd = Math.floor(centerY + halfScale);

  for (const column of visibleColumns) {
    renderNeatensteinSpriteColumnRgb(
      framebuffer,
      width,
      height,
      column,
      drawStart,
      drawEnd,
      parsedColor,
    );
  }

  ctx.putImageData({ data: framebuffer, width, height }, 0, 0);
}
