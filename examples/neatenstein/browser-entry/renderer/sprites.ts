/**
 * Enemy wireframe sprite rendering for the Neatenstein neon raycaster.
 *
 * Each sprite is projected from world space into screen space using the inverse
 * camera matrix, clipped against the per-column z-buffer, and drawn as a
 * vertical neon column for every visible pixel column. This keeps the sprite pass
 * cheap, deterministic, and trivially occluded by walls.
 *
 * @module
 */

import { clampInt } from './framebuffer';
import {
  clipNeatensteinSpriteSpan,
  type NeatensteinSpriteClip,
} from './zbuffer';

/**
 * World-space size of a sprite in grid cells.
 *
 * The projected screen size is `canvasHeight / perpDist * worldSize`, which
 * keeps sprites the same apparent size regardless of field of view.
 */
export const NEATENSTEIN_SPRITE_WORLD_SIZE = 0.5;

/**
 * Minimum perpendicular distance at which a sprite is drawn.
 *
 * Sprites closer than this are considered to be behind or touching the camera
 * plane and are culled to avoid division-by-zero and unstable projection.
 */
export const NEATENSTEIN_SPRITE_NEAR_CLIP = 0.1;

/**
 * Fraction of sprite scale used as a vertical neon line thickness.
 *
 * The line is centered on the projected screen X coordinate and extends
 * `scale * thickness` pixels horizontally.
 */
export const NEATENSTEIN_SPRITE_THICKNESS_RATIO = 0.4;

/**
 * Minimal canvas-like context consumed by the CPU sprite renderer.
 *
 * Only `putImageData` is required, mirroring the narrow context contract used
 * by the wall renderer so the same lightweight mock works in unit tests.
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
  /** Optional enemy type index for hue selection. */
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
  r: number;
  g: number;
  b: number;
}

/**
 * Parse a `#rrggbb` hex color string into an RGB triplet.
 *
 * @param hex - Color string in `#rrggbb` format.
 * @returns Parsed `{ r, g, b }` values in [0, 255].
 * @throws Error when the string is not a valid `#rrggbb` color.
 */
function parseHexColor(hex: string): ParsedRgb {
  if (hex.length !== 7 || hex[0] !== '#') {
    throw new Error(`Expected #rrggbb hex color, got "${hex}"`);
  }

  const r = Number.parseInt(hex.slice(1, 3), 16);
  const g = Number.parseInt(hex.slice(3, 5), 16);
  const b = Number.parseInt(hex.slice(5, 7), 16);

  if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) {
    throw new Error(`Invalid hex color components in "${hex}"`);
  }

  return { r, g, b };
}

/**
 * Resolve the square width/height of a flat RGBA framebuffer.
 *
 * @param framebuffer - Flat RGBA pixel buffer.
 * @returns The inferred `{ width, height }` dimensions.
 */
function resolveFramebufferSize(framebuffer: Uint8ClampedArray): {
  width: number;
  height: number;
} {
  const width = Math.floor(Math.sqrt(framebuffer.length / 4));
  return { width, height: width };
}

/**
 * Project a world-space sprite into screen coordinates.
 *
 * The projection uses the inverse camera matrix, the same math used by classic
 * raycasting sprite renderers. Sprites behind the camera plane or too close are
 * marked invisible so callers can skip them before touching the z-buffer.
 *
 * @param sprite - World-space sprite position.
 * @param camera - Current camera transform.
 * @param canvasWidth - Canvas width in CSS pixels.
 * @param canvasHeight - Canvas height in CSS pixels.
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
  const relativeX = sprite.worldX - camera.posX;
  const relativeY = sprite.worldY - camera.posY;

  const determinant = camera.planeX * camera.dirY - camera.dirX * camera.planeY;
  const invDet = 1.0 / determinant;

  const transformX =
    invDet * (camera.dirY * relativeX - camera.dirX * relativeY);
  const transformY =
    invDet * (-camera.planeY * relativeX + camera.planeX * relativeY);

  const perpDist = transformY;
  const visible = perpDist > NEATENSTEIN_SPRITE_NEAR_CLIP;

  if (!visible) {
    return {
      screenX: -1,
      scale: 0,
      perpDist,
      left: -1,
      right: -1,
      visible: false,
    };
  }

  const screenX = (canvasWidth / 2) * (1 + transformX / transformY);
  const scale =
    Math.abs(canvasHeight / transformY) * NEATENSTEIN_SPRITE_WORLD_SIZE;
  const halfThickness = (scale * NEATENSTEIN_SPRITE_THICKNESS_RATIO) / 2;

  return {
    screenX,
    scale,
    perpDist,
    left: screenX - halfThickness,
    right: screenX + halfThickness,
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
 * @param canvasWidth - Canvas width in CSS pixels.
 * @param canvasHeight - Canvas height in CSS pixels.
 * @param zBuffer - Per-column depth buffer.
 * @returns The clipped span and visible column indices. If the sprite is not
 *   visible, `visibleColumns` is empty.
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

  if (!projection.visible) {
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
 * Render a single sprite column into the CPU ImageData framebuffer.
 *
 * This draws a vertical neon bar for one visible column. It is intentionally
 * simple (a solid column, not a textured billboard) so that the CPU path stays
 * cheap and the z-buffer clipping remains obvious.
 *
 * @param framebuffer - Flat RGBA framebuffer.
 * @param column - Horizontal column index to write.
 * @param drawStart - Top row of the sprite stripe (inclusive).
 * @param drawEnd - Bottom row of the sprite stripe (exclusive).
 * @param hexColor - Sprite color as `#rrggbb`.
 *
 * @example
 * ```ts
 * renderNeatensteinSpriteColumn(framebuffer, 4, 2, 6, '#00bfff');
 * ```
 */
export function renderNeatensteinSpriteColumn(
  framebuffer: Uint8ClampedArray,
  column: number,
  drawStart: number,
  drawEnd: number,
  hexColor: string,
): void {
  const { r, g, b } = parseHexColor(hexColor);
  const { width, height } = resolveFramebufferSize(framebuffer);

  const clampedColumn = clampInt(column, 0, width - 1);
  const clampedStart = clampInt(drawStart, 0, height);
  const clampedEnd = clampInt(drawEnd, 0, height);

  if (clampedStart >= clampedEnd) return;

  for (let row = clampedStart; row < clampedEnd; row++) {
    const offset = (row * width + clampedColumn) * 4;
    framebuffer[offset] = r;
    framebuffer[offset + 1] = g;
    framebuffer[offset + 2] = b;
    framebuffer[offset + 3] = 255;
  }
}

/**
 * Render a projected sprite into the CPU ImageData framebuffer with z-buffer
 * occlusion.
 *
 * Only the sprite columns that are closer than the wall distance at the same
 * screen column are drawn. The whole framebuffer is flushed with a single
 * `putImageData` call, matching the CPU wall renderer's one-flush contract.
 *
 * @param framebuffer - Flat RGBA framebuffer.
 * @param zBuffer - Per-column depth buffer filled by the wall pass.
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
  const { width, height } = resolveFramebufferSize(framebuffer);
  const clip = clipNeatensteinSpriteSpan(
    zBuffer,
    projection.left,
    projection.right,
    projection.perpDist,
  );

  if (clip.visibleColumns.length === 0) return;

  const halfScale = projection.scale / 2;
  const centerY = height / 2;
  const drawStart = Math.floor(centerY - halfScale);
  const drawEnd = Math.floor(centerY + halfScale);

  for (const column of clip.visibleColumns) {
    renderNeatensteinSpriteColumn(
      framebuffer,
      column,
      drawStart,
      drawEnd,
      color,
    );
  }

  ctx.putImageData({ data: framebuffer, width, height }, 0, 0);
}
