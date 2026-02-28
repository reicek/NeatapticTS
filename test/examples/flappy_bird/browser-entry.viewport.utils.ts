import {
  FLAPPY_HALF,
  FLAPPY_NORMALIZATION_EPSILON,
} from './browser-entry.constants';
import { FLAPPY_WORLD_HEIGHT_PX, FLAPPY_PIPE_WIDTH_PX } from './constants';
import type { ViewportInfo } from './browser-entry.types';

/**
 * Resolves world viewport transformation based on canvas size.
 *
 * @param canvas - Playback canvas.
 * @returns Viewport scale and offsets.
 */
export function resolveWorldViewport(canvas: HTMLCanvasElement): ViewportInfo {
  const scale = canvas.height / FLAPPY_WORLD_HEIGHT_PX;
  const contentHeightPx = FLAPPY_WORLD_HEIGHT_PX * scale;

  return {
    offsetXPx: 0,
    offsetYPx: (canvas.height - contentHeightPx) * FLAPPY_HALF,
    scale,
  };
}

/**
 * Resolves visible world width represented by the current canvas.
 *
 * @param canvas - Playback canvas.
 * @returns Visible width in world-space pixels.
 */
export function resolveVisibleWorldWidthPx(canvas: HTMLCanvasElement): number {
  const viewportScale = canvas.height / Math.max(1, FLAPPY_WORLD_HEIGHT_PX);
  return canvas.width / Math.max(FLAPPY_NORMALIZATION_EPSILON, viewportScale);
}

/**
 * Resolves the world-space x spawn position for new pipes.
 *
 * @param visibleWorldWidthPx - Current visible world width.
 * @returns Spawn x-position.
 */
export function resolvePipeSpawnXPx(visibleWorldWidthPx: number): number {
  return visibleWorldWidthPx + FLAPPY_PIPE_WIDTH_PX;
}
