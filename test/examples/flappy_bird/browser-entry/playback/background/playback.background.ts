/**
 * Layered playback background composition for the browser demo.
 *
 * This boundary keeps atmosphere separate from gameplay entities. The frame
 * renderer can ask for one deterministic scenic backdrop while this module owns
 * the details of sky styling, ground-grid composition, and the glowing seam
 * that ties both halves together.
 */
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
 * This boundary exists to keep atmosphere separate from gameplay entities. The
 * frame renderer should be able to ask for a complete backdrop in one call
 * without also absorbing starfield policy, horizon styling, and ground-grid
 * composition details.
 *
 * The composition is intentionally chapter-like: sky first, ground second,
 * horizon seam last. That ordering gives the playback scene a stable visual
 * identity while keeping the background deterministic and cheap to re-render.
 *
 * @param context - Canvas 2D drawing context.
 * @param request - Narrow render input required for background composition.
 * @returns Nothing.
 * @example
 * ```ts
 * renderPlaybackBackground(context, {
 *   viewportLeftXPx: cameraLeftPx,
 *   visibleWorldWidthPx: 288,
 *   visibleWorldHeightPx: 512,
 *   frameIndex,
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
    frameIndex: request.frameIndex,
    scrollBasePx: request.scrollBasePx,
  });

  // Step 5: Draw the horizon divider above the sky and ground layers.
  drawPlaybackBackgroundHorizon(context, backgroundSceneContext);
}
