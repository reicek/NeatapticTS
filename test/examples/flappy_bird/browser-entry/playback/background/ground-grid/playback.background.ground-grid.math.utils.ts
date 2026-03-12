import {
  FLAPPY_GROUND_GRID_DEPTH_CURVE_EXPONENT,
  FLAPPY_GROUND_GRID_HORIZONTAL_LINE_COUNT,
  FLAPPY_GROUND_GRID_MAX_ALPHA,
  FLAPPY_GROUND_GRID_MAX_BLUR_PX,
  FLAPPY_GROUND_GRID_MAX_THICKNESS_PX,
  FLAPPY_GROUND_GRID_MIN_ALPHA,
  FLAPPY_GROUND_GRID_MIN_BLUR_PX,
  FLAPPY_GROUND_GRID_MIN_THICKNESS_PX,
  FLAPPY_GROUND_GRID_PIPE_CONNECTION_LINE_OFFSET_FROM_BOTTOM,
  FLAPPY_GROUND_GRID_SCROLL_RATIO,
} from './playback.background.ground-grid.constants';
import { resolvePlaybackBackgroundLayout } from '../playback.background.utils';

const cachedPipeConnectionProfileByHeight = new Map<
  number,
  PlaybackGroundGridPipeConnectionProfile
>();

/**
 * Small point value used when interpolating positions along one grid ray.
 */
export type PlaybackGroundGridPoint = {
  xPx: number;
  yPx: number;
};

/**
 * Shared pipe-floor projection resolved from the lower ground-grid geometry.
 */
export type PlaybackGroundGridPipeConnectionProfile = {
  pipeFloorYPx: number;
  matchedRayScrollRatio: number;
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
 * Resolves the shared lower-pipe floor and matched grid-ray scroll ratio.
 *
 * The lower pipe is visually clipped to the first usable horizontal grid band
 * above the bottom edge. The returned scroll ratio then speeds up the moving
 * perspective rays so their lateral motion matches the pipe speed exactly at
 * that same projected height.
 *
 * @param visibleWorldHeightPx - Current visible world height in pixels.
 * @returns Pipe-floor y-position plus the matching vertical-ray scroll ratio.
 */
export function resolvePlaybackGroundGridPipeConnectionProfile(
  visibleWorldHeightPx: number,
): PlaybackGroundGridPipeConnectionProfile {
  const cachedProfile = cachedPipeConnectionProfileByHeight.get(
    visibleWorldHeightPx,
  );
  if (cachedProfile) {
    return cachedProfile;
  }

  // Step 1: Resolve the lower-band layout used by both the grid and pipe illusion.
  const backgroundLayout = resolvePlaybackBackgroundLayout(visibleWorldHeightPx);
  const connectionLineIndex = Math.max(
    0,
    FLAPPY_GROUND_GRID_HORIZONTAL_LINE_COUNT -
      FLAPPY_GROUND_GRID_PIPE_CONNECTION_LINE_OFFSET_FROM_BOTTOM -
      1,
  );
  const depthRatio =
    (connectionLineIndex + 1) / FLAPPY_GROUND_GRID_HORIZONTAL_LINE_COUNT;
  const curvedDepthRatio = resolvePlaybackGroundGridDepthCurve(depthRatio);
  const pipeFloorYPx =
    backgroundLayout.lowerBandTopYPx +
    curvedDepthRatio * backgroundLayout.lowerBandHeightPx;

  // Step 2: Resolve how much anchor motion survives at the projected floor height.
  const vanishingPointYPx = Math.max(0, visibleWorldHeightPx * 0.5);
  const verticalTravelPx =
    vanishingPointYPx - backgroundLayout.lowerBandBottomYPx;
  const interpolationRatio =
    Math.abs(verticalTravelPx) < Number.EPSILON
      ? 0
      : (pipeFloorYPx - backgroundLayout.lowerBandBottomYPx) /
        verticalTravelPx;
  const clampedInterpolationRatio = Math.min(
    0.999,
    Math.max(0, interpolationRatio),
  );

  // Step 3: Return the floor height and the ray-scroll multiplier needed there.
  const pipeConnectionProfile = {
    pipeFloorYPx,
    matchedRayScrollRatio:
      1 /
      Math.max(
        Number.EPSILON,
        FLAPPY_GROUND_GRID_SCROLL_RATIO * (1 - clampedInterpolationRatio),
      ),
  };

  cachedPipeConnectionProfileByHeight.set(
    visibleWorldHeightPx,
    pipeConnectionProfile,
  );
  return pipeConnectionProfile;
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
