import {
  FLAPPY_GROUND_GRID_PULSE_MIN_ELIGIBLE_THICKNESS_PX,
  FLAPPY_GROUND_GRID_PULSE_PREFERRED_HORIZONTAL_START_RATIO,
} from './playback.background.ground-grid.constants';
import type {
  PlaybackGroundGridLineSegment,
  PlaybackGroundGridPulsePath,
  PlaybackGroundGridSegmentBatch,
} from './playback.background.ground-grid.types';

/**
 * Groups line segments into ordered style batches for lower-overhead drawing.
 *
 * @param segments - Ordered line segments that should preserve draw grouping.
 * @returns Ordered style batches that can be stroked with fewer state changes.
 */
export function groupPlaybackGroundGridSegmentsByStyle(
  segments: readonly PlaybackGroundGridLineSegment[],
): readonly PlaybackGroundGridSegmentBatch[] {
  const groupedSegmentsByStyle = new Map<
    string,
    PlaybackGroundGridLineSegment[]
  >();

  for (const segment of segments) {
    const styleKey = `${segment.alpha}:${segment.blurPx}:${segment.thicknessPx}`;
    const existingGroup = groupedSegmentsByStyle.get(styleKey);
    if (existingGroup) {
      existingGroup.push(segment);
      continue;
    }

    groupedSegmentsByStyle.set(styleKey, [segment]);
  }

  const segmentBatches: PlaybackGroundGridSegmentBatch[] = [];
  for (const groupedSegments of groupedSegmentsByStyle.values()) {
    const firstSegment = groupedSegments[0];
    segmentBatches.push({
      alpha: firstSegment.alpha,
      blurPx: firstSegment.blurPx,
      path: resolvePlaybackGroundGridBatchPath(groupedSegments),
      thicknessPx: firstSegment.thicknessPx,
      segments: groupedSegments,
    });
  }

  return segmentBatches;
}

/**
 * Prefers the nearer, thicker horizontal tracks when picking a pulse lane.
 *
 * @param horizontalLines - Visible horizontal grid bands.
 * @returns Pulse-eligible horizontal paths biased toward the foreground.
 */
export function resolvePlaybackGroundGridPreferredHorizontalPulsePaths(
  horizontalLines: readonly PlaybackGroundGridLineSegment[],
): readonly PlaybackGroundGridPulsePath[] {
  const eligibleHorizontalPulsePaths: PlaybackGroundGridPulsePath[] = [];

  for (const horizontalLine of horizontalLines) {
    if (
      horizontalLine.thicknessPx <
      FLAPPY_GROUND_GRID_PULSE_MIN_ELIGIBLE_THICKNESS_PX
    ) {
      continue;
    }

    eligibleHorizontalPulsePaths.push({
      orientation: 'horizontal',
      startXPx: horizontalLine.startXPx,
      startYPx: horizontalLine.startYPx,
      endXPx: horizontalLine.endXPx,
      endYPx: horizontalLine.endYPx,
      thicknessPx: horizontalLine.thicknessPx,
    });
  }

  if (eligibleHorizontalPulsePaths.length === 0) {
    return eligibleHorizontalPulsePaths;
  }

  const preferredStartIndex = Math.min(
    eligibleHorizontalPulsePaths.length - 1,
    Math.floor(
      eligibleHorizontalPulsePaths.length *
        FLAPPY_GROUND_GRID_PULSE_PREFERRED_HORIZONTAL_START_RATIO,
    ),
  );
  return eligibleHorizontalPulsePaths.slice(preferredStartIndex);
}

/**
 * Resolves one cached draw-ready path for a grouped segment batch.
 *
 * @param segments - Ordered line segments that belong to one style batch.
 * @returns Cached Path2D when available, otherwise null.
 */
function resolvePlaybackGroundGridBatchPath(
  segments: readonly PlaybackGroundGridLineSegment[],
): Path2D | null {
  if (typeof Path2D !== 'function') {
    return null;
  }

  const batchPath = new Path2D();
  for (const segment of segments) {
    batchPath.moveTo(segment.startXPx, segment.startYPx);
    batchPath.lineTo(segment.endXPx, segment.endYPx);
  }

  return batchPath;
}