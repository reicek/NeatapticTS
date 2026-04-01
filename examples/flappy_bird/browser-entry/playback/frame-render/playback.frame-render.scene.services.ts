import {
  FLAPPY_BIRD_VIEWPORT_X_RATIO,
  FLAPPY_BIRD_X_PX,
} from '../../../constants/constants';
import type { PopulationRenderState } from '../../browser-entry.types';
import { resolveWorldViewport } from '../../browser-entry.viewport.utils';
import { resolveChampionBirdIndex } from '../playback.render.utils';
import type { PlaybackFrameSceneContext } from './playback.frame-render.types';

/**
 * Scene-resolution helpers for playback frame rendering.
 *
 * Before anything can be painted, the renderer needs a camera-relative view of
 * the world: viewport scale, visible bounds, camera origin, and which bird is
 * currently the champion.
 */

/**
 * Resolves the shared scene contract used by one frame render pass.
 *
 * The camera is anchored so the focal bird region stays at a readable screen
 * position while the world scrolls beneath it.
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
