import {
  FLAPPY_BIRD_RADIUS_PX,
  FLAPPY_BIRD_X_PX,
} from '../../../constants/constants';
import type {
  PopulationRenderState,
  TrailState,
} from '../../browser-entry.types';
import { resolveChampionBirdIndex } from '../playback.render.utils';
import { pushChampionTrailPoint } from '../playback.trail.utils';
import {
  beginPlaybackFrameViewportTransform,
  finalizePlaybackFrameCanvas,
  preparePlaybackFrameCanvas,
  renderPlaybackFrameBackground,
  renderPlaybackFrameBirds,
  renderPlaybackFramePipes,
  renderPlaybackFrameTrails,
  resolvePlaybackFrameSceneContext,
} from './playback.frame-render.services';
import {
  drawTrail,
  renderPlaybackBird,
  resolvePlaybackTrailStyle,
} from './playback.frame-render.utils';

/**
 * Draws one simulation frame for the current population state.
 *
 * @param context - Canvas 2D drawing context.
 * @param renderState - Mutable simulation state snapshot.
 * @param trailState - Leader trail render cache.
 * @returns Nothing.
 */
export function renderPopulationFrame(
  context: CanvasRenderingContext2D,
  renderState: PopulationRenderState,
  trailState: TrailState,
): void {
  // Step 1: Resolve scene geometry and champion render state.
  const sceneContext = resolvePlaybackFrameSceneContext(context, renderState);

  // Step 2: Reset canvas state and enter the viewport transform.
  preparePlaybackFrameCanvas(context);
  beginPlaybackFrameViewportTransform(context, sceneContext);

  // Step 3: Draw the split background, pipes, birds, and trails in order.
  renderPlaybackFrameBackground(context, renderState, sceneContext);
  renderPlaybackFramePipes(context, renderState, sceneContext);
  renderPlaybackFrameBirds(
    context,
    renderState,
    sceneContext,
    renderPlaybackBird,
  );
  renderPlaybackFrameTrails(
    context,
    renderState,
    trailState,
    sceneContext,
    resolvePlaybackTrailStyle,
    drawTrail,
  );

  // Step 4: Restore the caller canvas state after viewport-space drawing.
  finalizePlaybackFrameCanvas(context);
}

/**
 * Updates the trail cache from the latest frame snapshot.
 *
 * @param trailState - Mutable trail state.
 * @param renderState - Current render state.
 * @returns Nothing.
 */
export function updateTrailState(
  trailState: TrailState,
  renderState: PopulationRenderState,
): void {
  // Step 1: Resolve the current champion so only one short trail is retained.
  const championBirdIndex = resolveChampionBirdIndex(renderState);

  // Step 2: Reset all non-champion trails to avoid per-frame trail work.
  renderState.birds.forEach((bird, birdIndex) => {
    if (!trailState.birdTrailsY[birdIndex]) {
      trailState.birdTrailsY[birdIndex] = [];
    }
    const birdTrail = trailState.birdTrailsY[birdIndex];

    if (bird.done || birdIndex !== championBirdIndex) {
      birdTrail.length = 0;
      return;
    }

    // Step 3: Keep a short trail only for the current champion bird.
    pushChampionTrailPoint(birdTrail, renderState.frameIndex, bird.yPx);
  });
}
