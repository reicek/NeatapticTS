import type { PlaybackFrameSceneContext } from './playback.frame-render.types';

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