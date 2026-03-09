import { renderPlaybackBackgroundGroundGrid } from './ground-grid/playback.background.ground-grid';
import {
  drawPlaybackBackgroundHorizon,
  drawPlaybackBackgroundSky,
  paintPlaybackBackgroundBase,
  resolvePlaybackBackgroundSceneContext,
} from './playback.background.services';
import type { PlaybackBackgroundRequest } from './playback.background.types';

/**
 * Draws the layered playback background.
 *
 * The composition keeps the top two-thirds for the neon starfield, fills the
 * lower band with a TRON-like perspective ground grid, and separates both
 * regions with a glowing horizon divider.
 *
 * @param context - Canvas 2D drawing context.
 * @param request - Narrow render input required for background composition.
 * @returns Nothing.
 *
 * @example
 * ```ts
 * renderPlaybackBackground(context, {
 *   viewportLeftXPx: cameraLeftPx,
 *   visibleWorldWidthPx: 288,
 *   visibleWorldHeightPx: 512,
 *   scrollBasePx: frameIndex * pipeSpeedPxPerFrame,
 * });
 * ```
 */
export function renderPlaybackBackground(
  context: CanvasRenderingContext2D,
  request: PlaybackBackgroundRequest,
): void {
  // Step 1: Resolve the scene contracts needed by the background passes.
  const backgroundSceneContext = resolvePlaybackBackgroundSceneContext(request);

  // Step 2: Paint the shared base fill behind all background layers.
  paintPlaybackBackgroundBase(context, backgroundSceneContext);

  // Step 3: Render the sky-only starfield inside the horizon clip band.
  drawPlaybackBackgroundSky(context, backgroundSceneContext, request);

  // Step 4: Render the lower-band neon grid beneath the horizon seam.
  renderPlaybackBackgroundGroundGrid(context, backgroundSceneContext, {
    scrollBasePx: request.scrollBasePx,
  });

  // Step 5: Draw the horizon divider above the sky and ground layers.
  drawPlaybackBackgroundHorizon(context, backgroundSceneContext);
}
