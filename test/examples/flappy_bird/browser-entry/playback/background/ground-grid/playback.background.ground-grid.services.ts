import { FLAPPY_BACKGROUND_COMPOSITE_SOURCE_OVER } from '../playback.background.constants';
import type {
  PlaybackBackgroundGroundGridResolvedScene,
  PlaybackGroundGridGeometry,
} from './playback.background.ground-grid.types';
import {
  drawGroundGridSegmentBatch,
  drawGroundGridSegmentBatches,
} from './playback.background.ground-grid.batch.services';
import {
  drawGroundGridFog,
  drawGroundGridPulse,
} from './playback.background.ground-grid.layer.services';

/**
 * Draws the resolved neon ground grid inside the lower background band.
 *
 * @param context - Canvas 2D drawing context.
 * @param resolvedScene - Geometry and style for the current viewport.
 * @param geometry - Precomputed horizontal and vertical line segments.
 * @returns Nothing.
 */
export function drawPlaybackGroundGrid(
  context: CanvasRenderingContext2D,
  resolvedScene: PlaybackBackgroundGroundGridResolvedScene,
  geometry: PlaybackGroundGridGeometry,
): void {
  context.save();
  context.translate(resolvedScene.sceneContext.viewportOffsetXPx, 0);
  context.globalCompositeOperation = FLAPPY_BACKGROUND_COMPOSITE_SOURCE_OVER;
  context.beginPath();
  context.rect(
    0,
    resolvedScene.sceneContext.lowerBandTopYPx,
    resolvedScene.sceneContext.visibleWorldWidthPx,
    resolvedScene.sceneContext.lowerBandHeightPx,
  );
  context.clip();

  drawGroundGridFog(context, resolvedScene);
  drawGroundGridSegmentBatches(
    context,
    geometry.verticalLineBatches,
    resolvedScene.style.lineColor,
  );
  drawGroundGridSegmentBatches(
    context,
    geometry.horizontalLineBatches,
    resolvedScene.style.lineColor,
  );
  drawGroundGridPulse(
    context,
    geometry.pulse,
    resolvedScene.style.pulseFillColor,
  );
  context.restore();
}
export {
  drawGroundGridSegmentBatch,
  drawGroundGridSegmentBatches,
} from './playback.background.ground-grid.batch.services';
export { drawGroundGridFog, drawGroundGridPulse } from './playback.background.ground-grid.layer.services';
