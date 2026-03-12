import { FLAPPY_TRAIL_EDGE_FADE_DISTANCE_PX } from '../../../constants/constants';
import type { PlaybackEdgeBounds } from '../playback.types';

/**
 * Converts distance-to-edge into a normalized opacity factor.
 *
 * Trail points fade as they approach the viewport border so the rendered path
 * feels cropped by the camera instead of abruptly chopped off. This mirrors the
 * common animation principle of easing visual intensity near a frame boundary.
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
 * This helper implements the other half of the afterimage effect: recent trail
 * samples should read as energetic and bright, while older samples should fade
 * away smoothly so the viewer's eye stays anchored to the current flock motion.
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
 * The trail renderer combines several normalized fade factors, so keeping this
 * utility local to the module makes the intent obvious: every opacity channel
 * must remain safe for direct canvas alpha use.
 *
 * @param value - Candidate value.
 * @returns Clamped value.
 */
export function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}
