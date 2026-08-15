/**
 * Shared gun sprite decode helpers for the Neatenstein on-screen cannon.
 *
 * Provides the encoded frame type, palette-tinting utility, and
 * nearest-neighbor decode used by the renderer. Mirrors the
 * {@link robot-sprite-decode.ts} module so the same decode/tint/render
 * pipeline can be reused for both the robot and the gun sprites.
 *
 * @module
 */

import { GUN_SPRITE_PALETTE, GUN_SPRITE_SCALE } from '../../gun-sprite-data.js';
import { type VoxelSnapshot } from '../../../neatenstein/scripts/snapshot-renderer';
import { RGBA_CHANNELS } from '../constants';
import type { EncodedGunSpriteFrame } from './renderer.gun.types';

// Re-export constants and types for external consumers.
export { RGBA_CHANNELS } from '../constants';
export type { EncodedGunSpriteFrame } from './renderer.gun.types';

/**
 * Decode an encoded gun sprite frame into a pre-rendered RGBA snapshot.
 *
 * Each palette index is mapped through the supplied palette (defaults to
 * {@link GUN_SPRITE_PALETTE}), preserving semitransparent colors. The
 * decoded frame is scaled up by {@link GUN_SPRITE_SCALE} using
 * nearest-neighbor sampling so the renderer can draw it directly.
 *
 * @param frame - Encoded rows of palette indices.
 * @param palette - Optional palette array (defaults to GUN_SPRITE_PALETTE).
 * @returns Decoded RGBA {@link VoxelSnapshot}.
 */
export function decodeGunSpriteFrame(
  frame: EncodedGunSpriteFrame,
  palette: readonly (readonly number[])[] = GUN_SPRITE_PALETTE,
): VoxelSnapshot {
  const logicalHeight = frame.length;
  const logicalWidth = (frame[0] as readonly number[]).length;
  const width = logicalWidth * GUN_SPRITE_SCALE;
  const height = logicalHeight * GUN_SPRITE_SCALE;
  const data = new Uint8ClampedArray(width * height * RGBA_CHANNELS);

  for (let y = 0; y < height; y += 1) {
    const logicalY = Math.floor(y / GUN_SPRITE_SCALE);
    const row = frame[logicalY] as readonly number[];
    for (let x = 0; x < width; x += 1) {
      const logicalX = Math.floor(x / GUN_SPRITE_SCALE);
      const color = palette[row[logicalX]] as readonly number[];
      const offset = (y * width + x) * RGBA_CHANNELS;
      data[offset] = color[0];
      data[offset + 1] = color[1];
      data[offset + 2] = color[2];
      data[offset + 3] = color[3];
    }
  }

  return { width, height, data };
}

/**
 * Build a modified palette with a team/accent color applied to the teal
 * energy indices 5 and 6.
 *
 * The RGB channels of palette entries 5 and 6 are replaced with the
 * supplied color while their original alpha values are preserved. All
 * other palette entries (including the muzzle flash at index 7) remain
 * unchanged.
 *
 * @param accentColor - [r, g, b] accent color to apply.
 * @returns Modified palette array.
 */
export function buildGunAccentPalette(
  accentColor: readonly [number, number, number],
): readonly (readonly number[])[] {
  return GUN_SPRITE_PALETTE.map((color, i) => {
    if (i === 5 || i === 6) {
      return [accentColor[0], accentColor[1], accentColor[2], color[3]];
    }
    return color;
  });
}
