/**
 * Pure gun overlay renderer for the Neatenstein neon raycasting demo.
 *
 * The weapon is drawn as a center-screen DOOM-style plasma cannon using a
 * palette-indexed 2D sprite decoded from {@link gun-sprite-data.js}. The
 * decoded RGBA pixels are rendered as uniform {@link GUN_SPRITE_SCALE}-sized
 * fillRect calls anchored at the bottom center of the viewport and kicked
 * upward by the current screen-space recoil offset. All rendering is pure
 * side effects on the supplied 2D context.
 *
 * @module
 */

import { RGBA_OPAQUE_ALPHA } from '../constants';
import type { GunState } from '../host/game/types';
import { decodeGunSpriteFrame, RGBA_CHANNELS } from './gun-sprite-decode';
import { GUN_SPRITE_FRAMES, GUN_SPRITE_SCALE } from '../../gun-sprite-data.js';

// Re-export constants and types for external consumers.
export { GUN_BODY_ASPECT_RATIO } from './renderer.gun.constants';
export type { EncodedGunSpriteFrame } from './renderer.gun.types';
export { NEATENSTEIN_GUN_BODY_COLOR, NEATENSTEIN_GUN_ACCENT_COLOR } from '../constants';

/**
 * Create the canonical initial {@link GunState} for a fresh episode.
 *
 * The recoil offset starts at zero so the overlay rests in its idle position.
 *
 * @returns A fresh {@link GunState} with zero recoil.
 *
 * @example
 * ```ts
 * const gun = createInitialGunState();
 * expect(gun.recoilOffset).toBe(0);
 * ```
 */
export function createInitialGunState(): GunState {
  return { recoilOffset: 0, firing: false };
}

/**
 * Render the gun overlay on top of the raycast frame.
 *
 * The gun is anchored at the bottom center of the viewport. The
 * palette-indexed sprite is decoded via {@link decodeGunSpriteFrame} into RGBA
 * pixels and drawn as uniform {@link GUN_SPRITE_SCALE}×{@link GUN_SPRITE_SCALE}
 * fillRect calls — one per non-transparent pixel — producing a chunky
 * DOOM-style plasma cannon with per-pixel palette colors and a semi-transparent
 * muzzle flash on the firing frame.
 *
 * When {@link GunState.recoilOffset} is non-zero, the entire overlay is
 * translated upward by that amount before drawing, producing a visible kick
 * that settles back to the idle position as the recoil decays.
 *
 * @param ctx - 2D canvas context to draw into.
 * @param gun - Current weapon overlay state.
 * @param width - Viewport width in CSS pixels.
 * @param height - Viewport height in CSS pixels.
 *
 * @returns Nothing; the function mutates the supplied 2D context as a side effect.
 *
 * @example
 * ```ts
 * const gun = createInitialGunState();
 * renderGunOverlay(ctx, gun, 640, 360);
 * ```
 */
export function renderGunOverlay(
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
  gun: GunState,
  width: number,
  height: number,
): void {
  const frame = gun.firing ? GUN_SPRITE_FRAMES.fire : GUN_SPRITE_FRAMES.idle;
  const decoded = decodeGunSpriteFrame(frame);

  const decodedWidth = decoded.width;
  const decodedHeight = decoded.height;
  const data = decoded.data;

  // The decoded sprite is scaled by GUN_SPRITE_SCALE from the logical grid.
  // Sample one pixel per GUN_SPRITE_SCALE×GUN_SPRITE_SCALE block (top-left
  // corner) to get the palette color for each logical cell, then draw each as
  // a GUN_SPRITE_SCALE×GUN_SPRITE_SCALE fillRect.
  const logicalWidth = decodedWidth / GUN_SPRITE_SCALE;
  const logicalHeight = decodedHeight / GUN_SPRITE_SCALE;

  // Anchor the sprite flush to the bottom center of the viewport.
  const spriteWidth = logicalWidth * GUN_SPRITE_SCALE;
  const spriteHeight = logicalHeight * GUN_SPRITE_SCALE;
  const spriteX = width / 2 - spriteWidth / 2;
  const spriteY = height - spriteHeight;

  ctx.save();
  if (gun.recoilOffset !== 0) {
    ctx.translate(0, -gun.recoilOffset);
  }

  for (let row = 0; row < logicalHeight; row += 1) {
    for (let col = 0; col < logicalWidth; col += 1) {
      // Sample the top-left pixel of each GUN_SPRITE_SCALE block. Because the
      // decoder uses nearest-neighbor scaling, every pixel in the block has
      // the same color, so sampling the corner is sufficient.
      const px = col * GUN_SPRITE_SCALE;
      const py = row * GUN_SPRITE_SCALE;
      const offset = (py * decodedWidth + px) * RGBA_CHANNELS;
      const r = data[offset];
      const g = data[offset + 1];
      const b = data[offset + 2];
      const a = data[offset + 3];

      if (a === 0) {
        continue;
      }

      if (a < RGBA_OPAQUE_ALPHA) {
        ctx.fillStyle = `rgba(${r},${g},${b},${a / RGBA_OPAQUE_ALPHA})`;
      } else {
        ctx.fillStyle = `rgb(${r},${g},${b})`;
      }
      ctx.fillRect(
        spriteX + col * GUN_SPRITE_SCALE,
        spriteY + row * GUN_SPRITE_SCALE,
        GUN_SPRITE_SCALE,
        GUN_SPRITE_SCALE,
      );
    }
  }

  ctx.restore();
}
