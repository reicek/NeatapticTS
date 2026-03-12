import type { PlaybackGroundGridSegmentBatch } from './playback.background.ground-grid.types';

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