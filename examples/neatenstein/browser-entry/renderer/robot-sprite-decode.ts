/**
 * Shared robot sprite decode helpers extracted from sprites.ts.
 *
 * Provides the encoded frame type, palette-tinting utility, and
 * nearest-neighbor decode used by both the renderer and the HUD mugshot.
 *
 * @module
 */

import {
  ROBOT_SPRITE_PALETTE,
  ROBOT_SPRITE_SCALE,
} from '../../robot-sprite-data.js';
import { type VoxelSnapshot } from '../shared/snapshot-renderer';
import { RGBA_CHANNELS } from '../constants';
import type { EncodedRobotSpriteFrame } from './renderer.sprite.types';

// Re-export constants and types for external consumers.
export { RGBA_CHANNELS } from '../constants';
export type { EncodedRobotSpriteFrame } from './renderer.sprite.types';

/**
 * Decode an encoded robot sprite frame into a pre-rendered RGBA snapshot.
 *
 * Each palette index is mapped through the supplied palette (defaults to
 * {@link ROBOT_SPRITE_PALETTE}), preserving semitransparent colors. The
 * decoded frame is scaled up by {@link ROBOT_SPRITE_SCALE} using
 * nearest-neighbor sampling so the renderer can sample it directly.
 *
 * @param frame - Encoded rows of palette indices.
 * @param palette - Optional palette array (defaults to ROBOT_SPRITE_PALETTE).
 * @returns Decoded RGBA {@link VoxelSnapshot}.
 */
export function decodeRobotSpriteFrame(
  frame: EncodedRobotSpriteFrame,
  palette: readonly (readonly [
    number,
    number,
    number,
    number,
  ])[] = ROBOT_SPRITE_PALETTE,
): VoxelSnapshot {
  const logicalHeight = frame.length;
  const logicalWidth = (frame[0] as readonly number[]).length;
  const width = logicalWidth * ROBOT_SPRITE_SCALE;
  const height = logicalHeight * ROBOT_SPRITE_SCALE;
  const data = new Uint8ClampedArray(width * height * RGBA_CHANNELS);

  for (let y = 0; y < height; y += 1) {
    const logicalY = Math.floor(y / ROBOT_SPRITE_SCALE);
    const row = frame[logicalY] as readonly number[];
    for (let x = 0; x < width; x += 1) {
      const logicalX = Math.floor(x / ROBOT_SPRITE_SCALE);
      const color = palette[row[logicalX]] as [number, number, number, number];
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
 * Build a modified palette with a team color applied to indices 5/6/7.
 *
 * The RGB channels of palette entries 5, 6, and 7 are replaced with the
 * team color while their original alpha values are preserved. All other
 * palette entries remain unchanged.
 *
 * @param teamColor - [r, g, b] team color to apply.
 * @returns Modified palette array.
 */
export function buildTeamColorPalette(
  teamColor: readonly [number, number, number],
): readonly (readonly [number, number, number, number])[] {
  return ROBOT_SPRITE_PALETTE.map((color, i) => {
    if (i === 5 || i === 6 || i === 7) {
      return [teamColor[0], teamColor[1], teamColor[2], color[3]] as const;
    }
    return color;
  });
}
