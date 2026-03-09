import {
  FLAPPY_GROUND_GRID_FOG_ALPHA,
  FLAPPY_GROUND_GRID_FOG_HEIGHT_RATIO,
} from './playback.background.ground-grid.constants';
import { FLAPPY_BACKGROUND_COMPOSITE_SOURCE_OVER } from '../playback.background.constants';
import type {
  PlaybackBackgroundGroundGridResolvedScene,
  PlaybackGroundGridGeometry,
  PlaybackGroundGridLineSegment,
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
  context.globalCompositeOperation = FLAPPY_BACKGROUND_COMPOSITE_SOURCE_OVER;
  context.beginPath();
  context.rect(
    resolvedScene.sceneContext.viewportLeftXPx,
    resolvedScene.sceneContext.lowerBandTopYPx,
    resolvedScene.sceneContext.visibleWorldWidthPx,
    resolvedScene.sceneContext.lowerBandHeightPx,
  );
  context.clip();

  drawGroundGridFog(context, resolvedScene);
  drawGroundGridSegments(
    context,
    geometry.verticalLines,
    resolvedScene.style.lineColor,
    resolvedScene.style.glowColor,
  );
  drawGroundGridSegments(
    context,
    geometry.horizontalLines,
    resolvedScene.style.lineColor,
    resolvedScene.style.glowColor,
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
  const fogGradient = context.createLinearGradient(
    0,
    resolvedScene.sceneContext.lowerBandTopYPx,
    0,
    resolvedScene.sceneContext.lowerBandTopYPx + fogHeightPx,
  );

  fogGradient.addColorStop(0, resolvedScene.style.fogColor);
  fogGradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

  context.save();
  context.globalAlpha = FLAPPY_GROUND_GRID_FOG_ALPHA;
  context.fillStyle = fogGradient;
  context.fillRect(
    resolvedScene.sceneContext.viewportLeftXPx,
    resolvedScene.sceneContext.lowerBandTopYPx,
    resolvedScene.sceneContext.visibleWorldWidthPx,
    fogHeightPx,
  );
  context.restore();
}

/**
 * Draws one ordered collection of neon line segments.
 *
 * @param context - Canvas 2D drawing context.
 * @param segments - Ordered line segments to render.
 * @param lineColor - Core neon stroke color.
 * @param glowColor - Outer glow color used for bloom.
 * @returns Nothing.
 */
export function drawGroundGridSegments(
  context: CanvasRenderingContext2D,
  segments: readonly PlaybackGroundGridLineSegment[],
  lineColor: string,
  glowColor: string,
): void {
  for (const segment of segments) {
    drawGroundGridSegment(context, segment, lineColor, glowColor);
  }
}

/**
 * Draws one neon line segment with a glow pass and crisp core line.
 *
 * @param context - Canvas 2D drawing context.
 * @param segment - One resolved line segment.
 * @param lineColor - Core neon stroke color.
 * @param glowColor - Outer glow color used for bloom.
 * @returns Nothing.
 */
export function drawGroundGridSegment(
  context: CanvasRenderingContext2D,
  segment: PlaybackGroundGridLineSegment,
  lineColor: string,
  glowColor: string,
): void {
  context.save();
  context.strokeStyle = lineColor;
  context.shadowColor = glowColor;
  context.shadowBlur = segment.blurPx;
  context.globalAlpha = segment.alpha;
  context.lineWidth = segment.thicknessPx;
  context.beginPath();
  context.moveTo(segment.startXPx, segment.startYPx);
  context.lineTo(segment.endXPx, segment.endYPx);
  context.stroke();

  context.shadowBlur = 0;
  context.globalAlpha = Math.min(1, segment.alpha + 0.18);
  context.beginPath();
  context.moveTo(segment.startXPx, segment.startYPx);
  context.lineTo(segment.endXPx, segment.endYPx);
  context.stroke();
  context.restore();
}