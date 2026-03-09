import {
  FLAPPY_GROUND_GRID_DEPTH_CURVE_EXPONENT,
  FLAPPY_GROUND_GRID_MAX_ALPHA,
  FLAPPY_GROUND_GRID_MAX_BLUR_PX,
  FLAPPY_GROUND_GRID_MAX_THICKNESS_PX,
  FLAPPY_GROUND_GRID_MIN_ALPHA,
  FLAPPY_GROUND_GRID_MIN_BLUR_PX,
  FLAPPY_GROUND_GRID_MIN_THICKNESS_PX,
} from './playback.background.ground-grid.constants';

/**
 * Small point value used when interpolating positions along one grid ray.
 */
export type PlaybackGroundGridPoint = {
  xPx: number;
  yPx: number;
};

/**
 * Maps a normalized depth ratio into a stronger synthwave spacing curve.
 *
 * @param depthRatio - Normalized 0..1 depth where 0 is far and 1 is near.
 * @returns Curved depth ratio used for line placement and styling.
 */
export function resolvePlaybackGroundGridDepthCurve(
  depthRatio: number,
): number {
  return Math.pow(depthRatio, FLAPPY_GROUND_GRID_DEPTH_CURVE_EXPONENT);
}

/**
 * Resolves neon alpha for one line based on its normalized depth.
 *
 * @param depthRatio - Normalized 0..1 depth where 0 is far and 1 is near.
 * @returns Opacity for the rendered line.
 */
export function resolvePlaybackGroundGridLineAlpha(depthRatio: number): number {
  return (
    FLAPPY_GROUND_GRID_MIN_ALPHA +
    (FLAPPY_GROUND_GRID_MAX_ALPHA - FLAPPY_GROUND_GRID_MIN_ALPHA) * depthRatio
  );
}

/**
 * Resolves glow blur for one line based on its normalized depth.
 *
 * @param depthRatio - Normalized 0..1 depth where 0 is far and 1 is near.
 * @returns Blur radius for the rendered line.
 */
export function resolvePlaybackGroundGridLineBlur(depthRatio: number): number {
  return (
    FLAPPY_GROUND_GRID_MAX_BLUR_PX -
    (FLAPPY_GROUND_GRID_MAX_BLUR_PX - FLAPPY_GROUND_GRID_MIN_BLUR_PX) *
      depthRatio
  );
}

/**
 * Resolves stroke thickness for one line based on its normalized depth.
 *
 * @param depthRatio - Normalized 0..1 depth where 0 is far and 1 is near.
 * @returns Stroke width in pixels.
 */
export function resolvePlaybackGroundGridLineThickness(
  depthRatio: number,
): number {
  return (
    FLAPPY_GROUND_GRID_MIN_THICKNESS_PX +
    (FLAPPY_GROUND_GRID_MAX_THICKNESS_PX -
      FLAPPY_GROUND_GRID_MIN_THICKNESS_PX) *
      depthRatio
  );
}

/**
 * Resolves normalized depth from a vertical distance away from the horizon.
 *
 * @param distanceToHorizonPx - Vertical distance from the vanishing horizon.
 * @param maximumDistanceToHorizonPx - Largest visible vertical horizon distance.
 * @returns Normalized 0..1 depth where 0 is at the horizon and 1 is nearest.
 */
export function resolvePlaybackGroundGridDepthFromHorizonDistance(
  distanceToHorizonPx: number,
  maximumDistanceToHorizonPx: number,
): number {
  if (maximumDistanceToHorizonPx <= 0) {
    return 0;
  }

  return Math.min(
    1,
    Math.max(0, distanceToHorizonPx / maximumDistanceToHorizonPx),
  );
}

/**
 * Interpolates one point along a perspective ray.
 *
 * @param startXPx - Bottom anchor x-position.
 * @param startYPx - Bottom anchor y-position.
 * @param endXPx - Vanishing-point x-position.
 * @param endYPx - Vanishing-point y-position.
 * @param interpolationRatio - Normalized 0..1 position along the ray.
 * @returns Interpolated point on the perspective ray.
 */
export function interpolatePlaybackGroundGridPoint(
  startXPx: number,
  startYPx: number,
  endXPx: number,
  endYPx: number,
  interpolationRatio: number,
): PlaybackGroundGridPoint {
  return {
    xPx: startXPx + (endXPx - startXPx) * interpolationRatio,
    yPx: startYPx + (endYPx - startYPx) * interpolationRatio,
  };
}
