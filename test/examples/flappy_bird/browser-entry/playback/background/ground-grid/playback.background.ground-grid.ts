/**
 * Neon ground-grid composition for the playback background.
 *
 * This sub-boundary turns lower-band scene geometry into the synthwave-style
 * motion cue that anchors the browser demo's visual identity. Geometry helpers,
 * pulse timing, batching, and styling all serve one teaching goal: let the
 * lower third imply depth and travel without distracting from the birds.
 */
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
 * The ground grid is the playback background's strongest style cue, but it also
 * serves a structural purpose: it gives the lower third of the screen a sense
 * of forward motion without competing with the pipes and birds for attention.
 *
 * This module owns the fold from shared lower-band scene data to one finished
 * grid pass. Geometry, pulse timing, batching, and styling all exist so the
 * final effect reads as depth and motion rather than as a collection of loose
 * line helpers.
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
