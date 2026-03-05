import {
  FLAPPY_TRAIL_EDGE_FADE_DISTANCE_PX,
  FLAPPY_TRAIL_MAX_POINTS,
} from '../../constants/constants';
import type { PlaybackEdgeBounds } from './playback.types';
import type { TrailPoint } from '../browser-entry.types';

/**
 * Appends one trail point while enforcing max retained history length.
 *
 * @param trailPoints - Mutable trail collection.
 * @param frameIndex - Source frame index.
 * @param yPosition - Bird y position.
 * @returns Nothing.
 */
export function pushTrailPoint(
  trailPoints: TrailPoint[],
  frameIndex: number,
  yPosition: number,
): void {
  trailPoints.push({ frameIndex, yPx: yPosition });
  const maxTrailPoints = FLAPPY_TRAIL_MAX_POINTS;
  if (trailPoints.length > maxTrailPoints) {
    trailPoints.splice(0, trailPoints.length - maxTrailPoints);
  }
}

/**
 * Converts distance-to-edge into a normalized opacity factor.
 *
 * Returns 0 exactly on or beyond an edge and rises to 1 once distance exceeds
 * the configured fade band.
 *
 * @param pointXPx - Point x position.
 * @param pointYPx - Point y position.
 * @param edgeBounds - Visible world bounds used for edge distance checks.
 * @returns Opacity multiplier in [0, 1].
 */
export function resolveEdgeOpacityFactor(
  pointXPx: number,
  pointYPx: number,
  edgeBounds: PlaybackEdgeBounds,
): number {
  const distanceToLeftEdgePx = pointXPx - edgeBounds.leftXPx;
  const distanceToRightEdgePx = edgeBounds.rightXPx - pointXPx;
  const distanceToTopEdgePx = pointYPx - edgeBounds.topYPx;
  const distanceToBottomEdgePx = edgeBounds.bottomYPx - pointYPx;

  const nearestEdgeDistancePx = Math.min(
    distanceToLeftEdgePx,
    distanceToRightEdgePx,
    distanceToTopEdgePx,
    distanceToBottomEdgePx,
  );
  const fadeProgress =
    nearestEdgeDistancePx / FLAPPY_TRAIL_EDGE_FADE_DISTANCE_PX;
  return clamp01(fadeProgress);
}

/**
 * Converts trail age into a normalized opacity factor.
 *
 * Oldest retained history approaches 0 opacity; newest approaches 1.
 *
 * @param frameOffset - Frames between this point and newest trail point.
 * @param maxTrailFrameOffset - Oldest age offset currently retained by trail.
 * @returns Opacity multiplier in [0, 1].
 */
export function resolveTrailLifetimeOpacityFactor(
  frameOffset: number,
  maxTrailFrameOffset: number,
): number {
  const normalizedLifetimeProgress =
    1 - frameOffset / Math.max(1, maxTrailFrameOffset);
  return clamp01(normalizedLifetimeProgress);
}

/**
 * Clamps a number to the inclusive [0, 1] range.
 *
 * @param value - Candidate value.
 * @returns Clamped value.
 */
export function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}
