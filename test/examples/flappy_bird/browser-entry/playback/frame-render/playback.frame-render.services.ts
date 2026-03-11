import {
  FLAPPY_BIRD_RADIUS_PX,
  FLAPPY_BIRD_VIEWPORT_X_RATIO,
  FLAPPY_BIRD_X_PX,
  FLAPPY_PIPE_SPEED_PX_PER_FRAME,
  FLAPPY_PIPE_WIDTH_PX,
} from '../../../constants/constants';
import type {
  PopulationRenderState,
  TrailPoint,
  TrailState,
} from '../../browser-entry.types';
import { resolveWorldViewport } from '../../browser-entry.viewport.utils';
import { renderPlaybackBackground } from '../background/playback.background';
import { drawPipeNeonOutline } from '../playback.render.service';
import { resolveChampionBirdIndex } from '../playback.render.utils';
import type {
  PlaybackFrameSceneContext,
  PlaybackTrailRenderStyle,
} from './playback.frame-render.types';

type PlaybackBirdRenderer = (
  context: CanvasRenderingContext2D,
  birdYPx: number,
  birdIndex: number,
  championBirdIndex: number,
) => void;

type PlaybackTrailStyleResolver = (
  birdIndex: number,
  championBirdIndex: number,
) => PlaybackTrailRenderStyle;

type PlaybackTrailRenderer = (
  context: CanvasRenderingContext2D,
  trailPoints: TrailPoint[],
  color: string,
  anchorX: number,
  baseOpacity: number,
  edgeBounds: PlaybackFrameSceneContext['edgeBounds'],
) => void;

/**
 * Resolves the shared scene contract used by one frame render pass.
 *
 * @param context - Canvas 2D drawing context.
 * @param renderState - Mutable simulation state snapshot.
 * @returns Viewport, camera, and edge-bounds state for the frame.
 */
export function resolvePlaybackFrameSceneContext(
  context: CanvasRenderingContext2D,
  renderState: PopulationRenderState,
): PlaybackFrameSceneContext {
  // Step 1: Resolve viewport scaling and visible world bounds.
  const viewport = resolveWorldViewport(context.canvas);
  const visibleWorldWidthPx = Math.max(1, renderState.visibleWorldWidthPx);
  const visibleWorldHeightPx = Math.max(1, renderState.visibleWorldHeightPx);

  // Step 2: Resolve camera position and champion index for this frame.
  const desiredBirdScreenXPx =
    visibleWorldWidthPx * FLAPPY_BIRD_VIEWPORT_X_RATIO;
  const cameraLeftPx = FLAPPY_BIRD_X_PX - desiredBirdScreenXPx;
  const championBirdIndex = resolveChampionBirdIndex(renderState);

  // Step 3: Return the shared frame scene context.
  return {
    viewport,
    visibleWorldWidthPx,
    visibleWorldHeightPx,
    cameraLeftPx,
    championBirdIndex,
    edgeBounds: {
      leftXPx: cameraLeftPx,
      rightXPx: cameraLeftPx + visibleWorldWidthPx,
      topYPx: 0,
      bottomYPx: visibleWorldHeightPx,
    },
  };
}

/**
 * Resets the target canvas and base paint state before frame drawing begins.
 *
 * @param context - Canvas 2D drawing context.
 * @returns Nothing.
 */
export function preparePlaybackFrameCanvas(
  context: CanvasRenderingContext2D,
): void {
  // Step 1: Reset base paint state to a predictable source-over draw pass.
  context.globalAlpha = 1;
  context.globalCompositeOperation = 'source-over';
  context.shadowBlur = 0;
  context.shadowColor = 'transparent';
  context.shadowOffsetX = 0;
  context.shadowOffsetY = 0;

  // Step 2: Clear the full backing canvas before drawing the next frame.
  context.clearRect(0, 0, context.canvas.width, context.canvas.height);
}

/**
 * Applies the viewport transform used for world-space frame rendering.
 *
 * @param context - Canvas 2D drawing context.
 * @param sceneContext - Shared scene geometry for the frame.
 * @returns Nothing.
 */
export function beginPlaybackFrameViewportTransform(
  context: CanvasRenderingContext2D,
  sceneContext: PlaybackFrameSceneContext,
): void {
  // Step 1: Save the caller state before applying viewport-space transforms.
  context.save();

  // Step 2: Translate and scale into the current world viewport.
  context.translate(
    sceneContext.viewport.offsetXPx,
    sceneContext.viewport.offsetYPx,
  );
  context.scale(sceneContext.viewport.scale, sceneContext.viewport.scale);
  context.translate(-sceneContext.cameraLeftPx, 0);
}

/**
 * Draws the split playback background for the current world viewport.
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
    scrollBasePx: renderState.frameIndex * FLAPPY_PIPE_SPEED_PX_PER_FRAME,
  });
}

/**
 * Draws all visible pipe segments and their neon outlines for the frame.
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
  // Step 1: Render each pipe pair against the current world height.
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
      sceneContext.visibleWorldHeightPx - gapBottomPx,
    );
  }
}

/**
 * Draws all active birds for the current frame.
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

/**
 * Restores the caller canvas state after viewport-space frame drawing.
 *
 * @param context - Canvas 2D drawing context.
 * @returns Nothing.
 */
export function finalizePlaybackFrameCanvas(
  context: CanvasRenderingContext2D,
): void {
  // Step 1: Reset global alpha before restoring the caller transform state.
  context.globalAlpha = 1;

  // Step 2: Restore the caller state saved before viewport transforms.
  context.restore();
}
