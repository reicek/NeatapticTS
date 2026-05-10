import {
  FLAPPY_BIRD_VIEWPORT_X_RATIO,
  FLAPPY_BIRD_X_PX,
  FLAPPY_HALF,
  FLAPPY_PIPE_WIDTH_PX,
} from '../constants/constants';
import type { ViewportInfo } from './browser-entry.types';

/**
 * Resolves world viewport transformation based on canvas size.
 *
 * @param canvas - Playback canvas.
 * @returns Viewport scale and offsets.
 */
export function resolveWorldViewport(canvas: HTMLCanvasElement): ViewportInfo {
  const scale = 1;
  const contentHeightPx = Math.max(1, canvas.height);

  return {
    offsetXPx: 0,
    offsetYPx: (canvas.height - contentHeightPx) * FLAPPY_HALF,
    scale,
  };
}

/**
 * Resolves visible world width represented by the current canvas.
 *
 * Educational note:
 * The current viewport model uses a 1:1 mapping between canvas pixels and
 * world-space pixels, so visible width is the canvas width directly.
 *
 * @param canvas - Playback canvas.
 * @returns Visible width in world-space pixels.
 */
export function resolveVisibleWorldWidthPx(canvas: HTMLCanvasElement): number {
  return canvas.width;
}

/**
 * Resolves visible world height represented by the current canvas.
 *
 * Educational note:
 * The current viewport model uses a 1:1 mapping between canvas pixels and
 * world-space pixels, so visible height is the canvas height directly.
 *
 * @param canvas - Playback canvas.
 * @returns Visible height in world-space pixels.
 */
export function resolveVisibleWorldHeightPx(canvas: HTMLCanvasElement): number {
  return canvas.height;
}

/**
 * Resolves the world-space x spawn position for new pipes.
 *
 * @param visibleWorldWidthPx - Current visible world width.
 * @param overflowPx - Additional offset relative to the visible right edge.
 * @returns Spawn x-position.
 */
export function resolvePipeSpawnXPx(
  visibleWorldWidthPx: number,
  overflowPx = FLAPPY_PIPE_WIDTH_PX,
): number {
  return resolveVisibleWorldRightXPx(visibleWorldWidthPx) + overflowPx;
}

function resolveVisibleWorldRightXPx(visibleWorldWidthPx: number): number {
  return (
    FLAPPY_BIRD_X_PX +
    Math.max(1, visibleWorldWidthPx) * (1 - FLAPPY_BIRD_VIEWPORT_X_RATIO)
  );
}
