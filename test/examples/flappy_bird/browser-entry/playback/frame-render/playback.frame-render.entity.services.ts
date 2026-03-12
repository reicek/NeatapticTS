import {
  FLAPPY_BIRD_RADIUS_PX,
  FLAPPY_BIRD_X_PX,
  FLAPPY_PIPE_WIDTH_PX,
} from '../../../constants/constants';
import type {
  PopulationRenderState,
  TrailState,
} from '../../browser-entry.types';
import { resolvePlaybackGroundGridPipeConnectionProfile } from '../background/ground-grid/playback.background.ground-grid.math.utils';
import { renderPlaybackBackground } from '../background/playback.background';
import { drawPipeNeonOutline } from '../playback.render.service';
import type {
  PlaybackBirdRenderer,
  PlaybackFrameSceneContext,
  PlaybackTrailRenderer,
  PlaybackTrailStyleResolver,
} from './playback.frame-render.types';

/**
 * Entity-layer rendering helpers for playback frames.
 *
 * These services paint the world-space contents of a frame once the scene and
 * canvas transforms have already been resolved.
 */

/**
 * Draws the split playback background for the current world viewport.
 *
 * Background painting is delegated to the dedicated background subsystem so this
 * layer can stay focused on frame composition order.
 *
 * @param context - Canvas 2D drawing context.
 * @param renderState - Mutable simulation state snapshot.
 * @param sceneContext - Shared scene geometry for the frame.
 * @returns Nothing.
 */
export function renderPlaybackFrameBackground(
  context: CanvasRenderingContext2D,
  renderState: PopulationRenderState,
  sceneContext: PlaybackFrameSceneContext,
): void {
  // Step 1: Render the world-space background behind all gameplay entities.
  renderPlaybackBackground(context, {
    viewportLeftXPx: sceneContext.cameraLeftPx,
    visibleWorldWidthPx: sceneContext.visibleWorldWidthPx,
    visibleWorldHeightPx: sceneContext.visibleWorldHeightPx,
    frameIndex: renderState.frameIndex,
    scrollBasePx: renderState.cumulativePipeTravelPx,
  });
}

/**
 * Draws all visible pipe segments and their neon outlines for the frame.
 *
 * Pipes are rendered as upper and lower segments connected to the projected
 * floor profile used by the background grid.
 *
 * @param context - Canvas 2D drawing context.
 * @param renderState - Mutable simulation state snapshot.
 * @param sceneContext - Shared scene geometry for the frame.
 * @returns Nothing.
 */
export function renderPlaybackFramePipes(
  context: CanvasRenderingContext2D,
  renderState: PopulationRenderState,
  sceneContext: PlaybackFrameSceneContext,
): void {
  // Step 1: Resolve the lifted lower-pipe floor from the shared grid projection.
  const pipeConnectionProfile = resolvePlaybackGroundGridPipeConnectionProfile(
    sceneContext.visibleWorldHeightPx,
  );

  // Step 2: Render each pipe pair against the projected floor height.
  for (const pipe of renderState.pipes) {
    const gapHalfPx = pipe.gapSizePx * 0.5;
    const gapTopPx = pipe.gapCenterYPx - gapHalfPx;
    const gapBottomPx = pipe.gapCenterYPx + gapHalfPx;

    drawPipeNeonOutline(context, pipe.xPx, 0, FLAPPY_PIPE_WIDTH_PX, gapTopPx);

    drawPipeNeonOutline(
      context,
      pipe.xPx,
      gapBottomPx,
      FLAPPY_PIPE_WIDTH_PX,
      pipeConnectionProfile.pipeFloorYPx - gapBottomPx,
    );
  }
}

/**
 * Draws all active birds for the current frame.
 *
 * Only live birds are painted so the playback frame reflects the active
 * population rather than leaving ghost bodies behind.
 *
 * @param context - Canvas 2D drawing context.
 * @param renderState - Mutable simulation state snapshot.
 * @param sceneContext - Shared scene geometry for the frame.
 * @param renderBird - Bird body renderer owned by the detailed utility layer.
 * @returns Nothing.
 */
export function renderPlaybackFrameBirds(
  context: CanvasRenderingContext2D,
  renderState: PopulationRenderState,
  sceneContext: PlaybackFrameSceneContext,
  renderBird: PlaybackBirdRenderer,
): void {
  // Step 1: Render only currently active birds.
  renderState.birds.forEach((bird, birdIndex) => {
    if (bird.done) {
      return;
    }

    renderBird(context, bird.yPx, birdIndex, sceneContext.championBirdIndex);
  });
}

/**
 * Draws stepped trails for all active birds in the frame.
 *
 * In practice the trail cache usually contains only the champion trail, but the
 * renderer stays generic and asks the trail-style policy how each active bird
 * should be painted.
 *
 * @param context - Canvas 2D drawing context.
 * @param renderState - Mutable simulation state snapshot.
 * @param trailState - Leader trail render cache.
 * @param sceneContext - Shared scene geometry for the frame.
 * @param resolveTrailStyle - Trail style resolver owned by the detailed utility layer.
 * @param renderTrail - Trail segment renderer owned by the detailed utility layer.
 * @returns Nothing.
 */
export function renderPlaybackFrameTrails(
  context: CanvasRenderingContext2D,
  renderState: PopulationRenderState,
  trailState: TrailState,
  sceneContext: PlaybackFrameSceneContext,
  resolveTrailStyle: PlaybackTrailStyleResolver,
  renderTrail: PlaybackTrailRenderer,
): void {
  // Step 1: Render only trails for active birds with available trail data.
  renderState.birds.forEach((bird, birdIndex) => {
    if (bird.done) {
      return;
    }

    const birdTrailPoints = trailState.birdTrailsY[birdIndex];
    if (!birdTrailPoints || birdTrailPoints.length === 0) {
      return;
    }

    const { baseOpacity, trailColor } = resolveTrailStyle(
      birdIndex,
      sceneContext.championBirdIndex,
    );
    renderTrail(
      context,
      birdTrailPoints,
      trailColor,
      FLAPPY_BIRD_X_PX - FLAPPY_BIRD_RADIUS_PX,
      baseOpacity,
      sceneContext.edgeBounds,
    );
  });
}
