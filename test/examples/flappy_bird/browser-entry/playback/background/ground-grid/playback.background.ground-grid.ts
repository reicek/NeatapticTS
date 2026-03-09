import { drawPlaybackGroundGrid } from './playback.background.ground-grid.services';
import { FLAPPY_BACKGROUND_GROUND_GRID_STYLE } from './playback.background.ground-grid.constants';
import type {
  PlaybackBackgroundGroundGridRequest,
  PlaybackBackgroundGroundGridResolvedScene,
  PlaybackBackgroundGroundGridSourceScene,
} from './playback.background.ground-grid.types';
import {
  resolvePlaybackGroundGridGeometry,
  resolvePlaybackGroundGridSceneContext,
} from './playback.background.ground-grid.utils';

/**
 * Draws the neon lower-band ground grid beneath the horizon.
 *
 * The grid is intentionally stylized rather than physically realistic: fixed
 * horizontal depth bands compress toward the horizon, while moving perspective
 * rays slide sideways but still converge to the centered vanishing point.
 *
 * @param context - Canvas 2D drawing context.
 * @param sourceScene - Shared lower-band geometry from the background module.
 * @param request - Shared parallax scroll input for the current frame.
 * @returns Nothing.
 */
export function renderPlaybackBackgroundGroundGrid(
  context: CanvasRenderingContext2D,
  sourceScene: PlaybackBackgroundGroundGridSourceScene,
  request: PlaybackBackgroundGroundGridRequest,
): void {
  // Step 1: Adapt the shared background scene into the narrow grid contract.
  const sceneContext = resolvePlaybackGroundGridSceneContext(sourceScene);

  // Step 2: Resolve theme-owned styling for the lower-band grid.
  const resolvedScene: PlaybackBackgroundGroundGridResolvedScene = {
    sceneContext,
    style: FLAPPY_BACKGROUND_GROUND_GRID_STYLE,
  };

  // Step 3: Build all grid geometry before touching the canvas state.
  const geometry = resolvePlaybackGroundGridGeometry(
    sceneContext,
    request.frameIndex,
    request.scrollBasePx,
  );

  // Step 4: Draw the resolved lower-band neon grid.
  drawPlaybackGroundGrid(context, resolvedScene, geometry);
}
