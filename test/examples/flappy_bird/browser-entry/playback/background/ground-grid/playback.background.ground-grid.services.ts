import {
  FLAPPY_GROUND_GRID_FOG_ALPHA,
  FLAPPY_GROUND_GRID_FOG_HEIGHT_RATIO,
} from './playback.background.ground-grid.constants';
import {
  FLAPPY_BACKGROUND_COMPOSITE_SOURCE_OVER,
} from '../playback.background.constants';
import {
  resolveCachedGroundGridFogGradient,
  resolveGroundGridSceneCacheKey,
} from './playback.background.ground-grid.cache.services';
import type {
  PlaybackBackgroundGroundGridResolvedScene,
  PlaybackGroundGridGeometry,
  PlaybackGroundGridPulse,
  PlaybackGroundGridSegmentBatch,
} from './playback.background.ground-grid.types';

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

/**
 * Draws the lower-band atmospheric wash behind the neon line work.
 *
 * @param context - Canvas 2D drawing context.
 * @param resolvedScene - Geometry and style for the current viewport.
 * @returns Nothing.
 */
export function drawGroundGridFog(
  context: CanvasRenderingContext2D,
  resolvedScene: PlaybackBackgroundGroundGridResolvedScene,
): void {
  const fogHeightPx =
    resolvedScene.sceneContext.lowerBandHeightPx *
    FLAPPY_GROUND_GRID_FOG_HEIGHT_RATIO;
  const sceneCacheKey = resolveGroundGridSceneCacheKey(
    resolvedScene.sceneContext,
  );
  const fogGradient = resolveCachedGroundGridFogGradient(
    context,
    sceneCacheKey,
    resolvedScene.sceneContext,
    resolvedScene.style.fogColor,
  );

  context.save();
  context.globalAlpha = FLAPPY_GROUND_GRID_FOG_ALPHA;
  context.fillStyle = fogGradient;
  context.fillRect(
    0,
    resolvedScene.sceneContext.lowerBandTopYPx,
    resolvedScene.sceneContext.visibleWorldWidthPx,
    fogHeightPx,
  );
  context.restore();
}

/**
 * Draws one ordered collection of neon segment batches.
 *
 * @param context - Canvas 2D drawing context.
 * @param batches - Ordered line-segment batches to render.
 * @param lineColor - Core neon stroke color.
 * @returns Nothing.
 */
export function drawGroundGridSegmentBatches(
  context: CanvasRenderingContext2D,
  batches: readonly PlaybackGroundGridSegmentBatch[],
  lineColor: string,
): void {
  context.save();
  context.strokeStyle = lineColor;

  for (const batch of batches) {
    drawGroundGridSegmentBatch(context, batch);
  }

  context.restore();
}

/**
 * Draws one batch of neon line segments that share one render style.
 *
 * @param context - Canvas 2D drawing context.
 * @param batch - Ordered line-segment batch that shares one render style.
 * @returns Nothing.
 */
export function drawGroundGridSegmentBatch(
  context: CanvasRenderingContext2D,
  batch: PlaybackGroundGridSegmentBatch,
): void {
  context.globalAlpha = batch.alpha;
  context.lineWidth = batch.thicknessPx;
  strokePlaybackGroundGridBatch(context, batch);
}

/**
 * Strokes one ground-grid batch using a cached path when the environment supports it.
 *
 * @param context - Canvas 2D drawing context.
 * @param batch - Ordered line-segment batch that shares one render style.
 * @returns Nothing.
 */
function strokePlaybackGroundGridBatch(
  context: CanvasRenderingContext2D,
  batch: PlaybackGroundGridSegmentBatch,
): void {
  if (batch.path) {
    context.stroke(batch.path);
    return;
  }

  context.beginPath();
  for (const segment of batch.segments) {
    context.moveTo(segment.startXPx, segment.startYPx);
    context.lineTo(segment.endXPx, segment.endYPx);
  }
  context.stroke();
}

/**
 * Draws one pulse square above the grid lines and below gameplay entities.
 *
 * @param context - Canvas 2D drawing context.
 * @param pulse - Visible pulse square for the current frame.
 * @param fillColor - Core neon fill color.
 * @returns Nothing.
 */
export function drawGroundGridPulse(
  context: CanvasRenderingContext2D,
  pulse: PlaybackGroundGridPulse | null,
  fillColor: string,
): void {
  if (!pulse) {
    return;
  }

  const halfSizePx = pulse.sizePx * 0.5;

  context.save();
  context.globalCompositeOperation = FLAPPY_BACKGROUND_COMPOSITE_SOURCE_OVER;
  context.fillStyle = fillColor;
  context.globalAlpha = pulse.alpha;
  context.fillRect(
    pulse.centerXPx - halfSizePx,
    pulse.centerYPx - halfSizePx,
    pulse.sizePx,
    pulse.sizePx,
  );
  context.restore();
}
