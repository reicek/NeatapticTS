/**
 * Per-column sprite rendering executors extracted from the sprite renderer.
 *
 * Contains the voxel-column pixel loop body that renders a single sprite
 * column into the framebuffer. This executor accepts scalar locals (framebuffer
 * reference, dimensions, column index, draw bounds, frame reference, fog factor)
 * and does NOT create per-pixel object allocations.
 *
 * @module
 */

import { clampInt, NEATENSTEIN_BACKGROUND_RGB } from './framebuffer';
import { shouldDissolvePixel } from './derez';
import { NEATENSTEIN_ENEMY_DEATH_COLOR } from '../constants';
import { RGBA_CHANNELS } from './renderer.sprite.constants';
import type { VoxelSnapshot } from '../../../neatenstein/scripts/snapshot-renderer';
import type { NeatensteinDerezState } from './renderer.sprite.types';

/**
 * Copy one column of a pre-rendered voxel frame onto a screen column.
 *
 * `visibleColumns` already guarantees that the screen column is closer than
 * the wall at that X, so this helper does not re-check the z-buffer. It skips
 * transparent frame pixels (alpha == 0) and clamps vertical writes to the
 * framebuffer bounds.
 *
 * Distance fog linearly blends each pixel toward
 * {@link NEATENSTEIN_BACKGROUND_RGB} as the sprite's perpendicular distance
 * approaches {@link NEATENSTEIN_RENDER_DISTANCE_CAP}.
 *
 * When `derezState` is provided, each opaque frame pixel is tested against
 * the de-rez dissolution mask via {@link shouldDissolvePixel}. Dissolved
 * pixels are skipped entirely; surviving pixels are tinted toward
 * {@link NEATENSTEIN_ENEMY_DEATH_COLOR} by a factor of `t * 0.5` where
 * `t` is the normalised animation progress.
 *
 * @param framebuffer - Flat RGBA framebuffer to write into.
 * @param width - Framebuffer width in pixels.
 * @param height - Framebuffer height in pixels.
 * @param screenColumn - Horizontal framebuffer column to write.
 * @param drawStart - Top screen row of the projected sprite, inclusive.
 * @param drawEnd - Bottom screen row of the projected sprite, exclusive.
 * @param frame - Pre-rendered voxel snapshot.
 * @param frameX - Column of the voxel frame to sample.
 * @param fogFactor - Fog interpolation factor in `[0, 1]` where `0` is no fog
 *   and `1` is fully blended into the background color.
 * @param derezState - Optional de-rez death animation state. When provided,
 *   the column renderer dissolves pixels and tints survivors.
 */
export function renderNeatensteinVoxelSpriteColumn(
  framebuffer: Uint8ClampedArray,
  width: number,
  height: number,
  screenColumn: number,
  drawStart: number,
  drawEnd: number,
  frame: VoxelSnapshot,
  frameX: number,
  fogFactor: number,
  derezState?: NeatensteinDerezState,
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

  const invFog = 1 - fogFactor;
  const { r: bgR, g: bgG, b: bgB } = NEATENSTEIN_BACKGROUND_RGB;

  // Precompute de-rez constants outside the per-pixel loop.
  const derezActive = derezState !== undefined;
  const derezT = derezActive
    ? derezState!.elapsedMs / derezState!.durationMs
    : 0;
  const tintFactor = derezT * 0.5;
  const invTint = 1 - tintFactor;
  const deathR = NEATENSTEIN_ENEMY_DEATH_COLOR[0];
  const deathG = NEATENSTEIN_ENEMY_DEATH_COLOR[1];
  const deathB = NEATENSTEIN_ENEMY_DEATH_COLOR[2];

  for (let rowOffset = 0; rowOffset < spriteHeightPixels; rowOffset += 1) {
    const screenY = clampedStart + rowOffset;
    const v = (screenY - drawStart) / (drawEnd - drawStart);
    const frameY = Math.floor(v * (frameHeight - 1));
    const safeFrameY = clampInt(frameY, 0, frameHeight - 1);

    const frameOffset = (safeFrameY * frameWidth + safeFrameX) * RGBA_CHANNELS;
    const alpha = frameData[frameOffset + 3];
    if (alpha === 0) {
      continue;
    }

    // De-rez dissolution: skip pixels whose noise hash falls below t.
    if (
      derezActive &&
      shouldDissolvePixel(
        safeFrameX,
        safeFrameY,
        derezState!.seed,
        derezState!.elapsedMs,
        derezState!.durationMs,
      )
    ) {
      continue;
    }

    const screenOffset = (screenY * width + screenColumn) * RGBA_CHANNELS;

    // Apply fog, then optionally tint surviving pixels toward death color.
    const fogR = frameData[frameOffset] * invFog + bgR * fogFactor;
    const fogG = frameData[frameOffset + 1] * invFog + bgG * fogFactor;
    const fogB = frameData[frameOffset + 2] * invFog + bgB * fogFactor;

    framebuffer[screenOffset] = derezActive
      ? Math.round(fogR * invTint + deathR * tintFactor)
      : Math.round(fogR);
    framebuffer[screenOffset + 1] = derezActive
      ? Math.round(fogG * invTint + deathG * tintFactor)
      : Math.round(fogG);
    framebuffer[screenOffset + 2] = derezActive
      ? Math.round(fogB * invTint + deathB * tintFactor)
      : Math.round(fogB);
    framebuffer[screenOffset + 3] = alpha;
  }
}
