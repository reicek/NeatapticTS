import {
  FLAPPY_GROUND_GRID_DEPTH_CURVE_EXPONENT,
  FLAPPY_GROUND_GRID_HORIZONTAL_LINE_COUNT,
  FLAPPY_GROUND_GRID_MAX_ALPHA,
  FLAPPY_GROUND_GRID_MAX_BLUR_PX,
  FLAPPY_GROUND_GRID_MAX_THICKNESS_PX,
  FLAPPY_GROUND_GRID_MIN_ALPHA,
  FLAPPY_GROUND_GRID_MIN_BLUR_PX,
  FLAPPY_GROUND_GRID_MIN_VERTICAL_LINE_COUNT,
  FLAPPY_GROUND_GRID_MIN_THICKNESS_PX,
  FLAPPY_GROUND_GRID_SCROLL_RATIO,
  FLAPPY_GROUND_GRID_TARGET_VERTICAL_LINE_SPACING_PX,
  FLAPPY_GROUND_GRID_TARGET_VERTICAL_SEGMENT_HEIGHT_PX,
  FLAPPY_GROUND_GRID_VERTICAL_OVERFLOW_COUNT,
} from './playback.background.ground-grid.constants';
import { positiveModulo } from '../../playback.starfield.utils';
import type {
  PlaybackBackgroundGroundGridSceneContext,
  PlaybackGroundGridGeometry,
  PlaybackGroundGridLineSegment,
} from './playback.background.ground-grid.types';

/**
 * Resolves the shared scene context used by the ground-grid renderer.
 *
 * @param sceneContext - Lower-band geometry provided by the background module.
 * @returns Narrow scene contract consumed by grid-specific helpers.
 */
export function resolvePlaybackGroundGridSceneContext(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
): PlaybackBackgroundGroundGridSceneContext {
  return sceneContext;
}

/**
 * Builds the line geometry for the neon ground grid.
 *
 * @param sceneContext - Lower-band geometry for the current viewport.
 * @param scrollBasePx - Shared world scroll used for parallax motion.
 * @returns Horizontal depth bands and perspective rays for the current frame.
 */
export function resolvePlaybackGroundGridGeometry(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
  scrollBasePx: number,
): PlaybackGroundGridGeometry {
  // Step 1: Resolve the fixed horizontal depth bands for the lower plane.
  const horizontalLines = resolvePlaybackGroundGridHorizontalLines(
    sceneContext,
  );

  // Step 2: Resolve the moving perspective rays with wrapped anchor spacing.
  const verticalLines = resolvePlaybackGroundGridVerticalLines(
    sceneContext,
    scrollBasePx,
  );

  // Step 3: Return the pure geometry bundle used by the canvas renderer.
  return {
    horizontalLines,
    verticalLines,
  };
}

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
export function resolvePlaybackGroundGridLineAlpha(
  depthRatio: number,
): number {
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
export function resolvePlaybackGroundGridLineBlur(
  depthRatio: number,
): number {
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
 * Builds the screen-horizontal depth bands for the lower neon plane.
 *
 * @param sceneContext - Lower-band geometry for the current viewport.
 * @returns Ordered far-to-near line segments.
 */
function resolvePlaybackGroundGridHorizontalLines(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
): readonly PlaybackGroundGridLineSegment[] {
  const maximumDistanceToHorizonPx = Math.max(
    1,
    sceneContext.lowerBandBottomYPx - sceneContext.alignedHorizonYPx,
  );

  return Array.from(
    { length: FLAPPY_GROUND_GRID_HORIZONTAL_LINE_COUNT },
    (_unusedValue, lineIndex) => {
      // Step 1: Resolve normalized depth so the first line hugs the horizon.
      const depthRatio =
        (lineIndex + 1) / FLAPPY_GROUND_GRID_HORIZONTAL_LINE_COUNT;
      const curvedDepthRatio = resolvePlaybackGroundGridDepthCurve(depthRatio);
      const lineYPx =
        sceneContext.lowerBandTopYPx +
        curvedDepthRatio * sceneContext.lowerBandHeightPx;
      const thicknessDepthRatio =
        resolvePlaybackGroundGridDepthFromHorizonDistance(
          lineYPx - sceneContext.alignedHorizonYPx,
          maximumDistanceToHorizonPx,
        );

      // Step 2: Strengthen the near lines while softening the far ones.
      return {
        startXPx: sceneContext.viewportLeftXPx,
        startYPx: lineYPx,
        endXPx: sceneContext.viewportLeftXPx + sceneContext.visibleWorldWidthPx,
        endYPx: lineYPx,
        alpha: resolvePlaybackGroundGridLineAlpha(depthRatio),
        blurPx: resolvePlaybackGroundGridLineBlur(depthRatio),
        thicknessPx: resolvePlaybackGroundGridLineThickness(thicknessDepthRatio),
      };
    },
  );
}

/**
 * Builds the perspective rays that converge to the centered horizon point.
 *
 * @param sceneContext - Lower-band geometry for the current viewport.
 * @param scrollBasePx - Shared world scroll used for parallax motion.
 * @returns Wrapped left-to-right perspective rays.
 */
function resolvePlaybackGroundGridVerticalLines(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
  scrollBasePx: number,
): readonly PlaybackGroundGridLineSegment[] {
  const horizonLeftXPx = sceneContext.viewportLeftXPx;
  const horizonRightXPx =
    sceneContext.viewportLeftXPx + sceneContext.visibleWorldWidthPx;
  const visibleAnchorBounds = resolvePlaybackGroundGridAnchorBounds({
    horizonLeftXPx,
    horizonRightXPx,
    sceneContext,
  });
  const totalVisibleLaneCount = Math.max(
    FLAPPY_GROUND_GRID_MIN_VERTICAL_LINE_COUNT,
    Math.ceil(
      visibleAnchorBounds.anchorSpanPx /
        FLAPPY_GROUND_GRID_TARGET_VERTICAL_LINE_SPACING_PX,
    ) + 1,
  );
  const laneSpacingPx =
    totalVisibleLaneCount > 1
      ? visibleAnchorBounds.anchorSpanPx / (totalVisibleLaneCount - 1)
      : visibleAnchorBounds.anchorSpanPx;
  const safeLaneSpacingPx = Math.max(1, laneSpacingPx);
  const wrappedOffsetPx = positiveModulo(
    scrollBasePx * FLAPPY_GROUND_GRID_SCROLL_RATIO,
    safeLaneSpacingPx,
  );
  const firstAnchorXPx =
    visibleAnchorBounds.leftAnchorXPx -
    FLAPPY_GROUND_GRID_VERTICAL_OVERFLOW_COUNT * safeLaneSpacingPx -
    wrappedOffsetPx;
  const totalRayCount =
    totalVisibleLaneCount + FLAPPY_GROUND_GRID_VERTICAL_OVERFLOW_COUNT * 2 + 1;
  const maximumLateralDistancePx =
    Math.max(
      Math.abs(visibleAnchorBounds.leftAnchorXPx - sceneContext.vanishingPointXPx),
      Math.abs(visibleAnchorBounds.rightAnchorXPx - sceneContext.vanishingPointXPx),
    ) + FLAPPY_GROUND_GRID_VERTICAL_OVERFLOW_COUNT * safeLaneSpacingPx;
  const maximumDistanceToHorizonPx = Math.max(
    1,
    sceneContext.lowerBandBottomYPx - sceneContext.alignedHorizonYPx,
  );
  const verticalSegmentCount = Math.max(
    6,
    Math.ceil(
      maximumDistanceToHorizonPx /
        FLAPPY_GROUND_GRID_TARGET_VERTICAL_SEGMENT_HEIGHT_PX,
    ),
  );

  return Array.from({ length: totalRayCount }, (_unusedValue, rayIndex) => {
    // Step 1: Resolve the bottom anchor and its centered strength.
    const anchorXPx = firstAnchorXPx + rayIndex * safeLaneSpacingPx;
    const lateralDistancePx = Math.abs(
      anchorXPx - sceneContext.vanishingPointXPx,
    );
    const centeredStrength =
      1 - Math.min(1, lateralDistancePx / maximumLateralDistancePx);
    const lineDepthRatio = 0.45 + centeredStrength * 0.55;

    // Step 2: Split the ray into short segments so width can taper to 1px.
    return resolvePlaybackGroundGridVerticalLineSegments({
      anchorXPx,
      centeredStrength,
      lineDepthRatio,
      maximumDistanceToHorizonPx,
      sceneContext,
      verticalSegmentCount,
    });
  }).flat();
}

type PlaybackGroundGridVerticalSegmentsInput = {
  anchorXPx: number;
  centeredStrength: number;
  lineDepthRatio: number;
  maximumDistanceToHorizonPx: number;
  sceneContext: PlaybackBackgroundGroundGridSceneContext;
  verticalSegmentCount: number;
};

/**
 * Builds tapered style segments for one perspective ray.
 *
 * @param input - Geometry and depth context for one ray.
 * @returns Ordered near-to-far segments for one perspective ray.
 */
function resolvePlaybackGroundGridVerticalLineSegments(
  input: PlaybackGroundGridVerticalSegmentsInput,
): readonly PlaybackGroundGridLineSegment[] {
  return Array.from({ length: input.verticalSegmentCount }, (_unusedValue, segmentIndex) => {
    // Step 1: Resolve the interpolation bounds for this sub-segment.
    const segmentStartRatio = segmentIndex / input.verticalSegmentCount;
    const segmentEndRatio = (segmentIndex + 1) / input.verticalSegmentCount;
    const segmentMidpointRatio = (segmentStartRatio + segmentEndRatio) * 0.5;
    const segmentStartPoint = interpolatePlaybackGroundGridPoint(
      input.anchorXPx,
      input.sceneContext.lowerBandBottomYPx,
      input.sceneContext.vanishingPointXPx,
      input.sceneContext.vanishingPointYPx,
      segmentStartRatio,
    );
    const segmentEndPoint = interpolatePlaybackGroundGridPoint(
      input.anchorXPx,
      input.sceneContext.lowerBandBottomYPx,
      input.sceneContext.vanishingPointXPx,
      input.sceneContext.vanishingPointYPx,
      segmentEndRatio,
    );

    // Step 2: Resolve style from the segment midpoint distance to the horizon.
    const segmentMidpoint = interpolatePlaybackGroundGridPoint(
      input.anchorXPx,
      input.sceneContext.lowerBandBottomYPx,
      input.sceneContext.vanishingPointXPx,
      input.sceneContext.vanishingPointYPx,
      segmentMidpointRatio,
    );
    const segmentDepthRatio = resolvePlaybackGroundGridDepthFromHorizonDistance(
      segmentMidpoint.yPx - input.sceneContext.alignedHorizonYPx,
      input.maximumDistanceToHorizonPx,
    );
    const combinedAlphaDepthRatio =
      segmentDepthRatio * 0.7 + input.lineDepthRatio * 0.3;

    // Step 3: Return the tapered segment so the ray narrows toward the horizon.
    return {
      startXPx: segmentStartPoint.xPx,
      startYPx: segmentStartPoint.yPx,
      endXPx: segmentEndPoint.xPx,
      endYPx: segmentEndPoint.yPx,
      alpha: resolvePlaybackGroundGridLineAlpha(combinedAlphaDepthRatio),
      blurPx: resolvePlaybackGroundGridLineBlur(segmentDepthRatio),
      thicknessPx: resolvePlaybackGroundGridLineThickness(segmentDepthRatio),
    };
  });
}

type PlaybackGroundGridAnchorBoundsInput = {
  horizonLeftXPx: number;
  horizonRightXPx: number;
  sceneContext: PlaybackBackgroundGroundGridSceneContext;
};

type PlaybackGroundGridAnchorBounds = {
  leftAnchorXPx: number;
  rightAnchorXPx: number;
  anchorSpanPx: number;
};

/**
 * Projects the visible horizon span back onto the floor anchor line.
 *
 * @param input - Visible horizon bounds and scene geometry.
 * @returns Bottom-anchor bounds required to cover the full visible horizon.
 */
function resolvePlaybackGroundGridAnchorBounds(
  input: PlaybackGroundGridAnchorBoundsInput,
): PlaybackGroundGridAnchorBounds {
  const leftAnchorXPx = projectPlaybackGroundGridHorizonXToAnchorX({
    horizonXPx: input.horizonLeftXPx,
    sceneContext: input.sceneContext,
  });
  const rightAnchorXPx = projectPlaybackGroundGridHorizonXToAnchorX({
    horizonXPx: input.horizonRightXPx,
    sceneContext: input.sceneContext,
  });

  return {
    leftAnchorXPx,
    rightAnchorXPx,
    anchorSpanPx: Math.max(1, rightAnchorXPx - leftAnchorXPx),
  };
}

type PlaybackGroundGridAnchorProjectionInput = {
  horizonXPx: number;
  sceneContext: PlaybackBackgroundGroundGridSceneContext;
};

/**
 * Projects one horizon x-position down to the required floor anchor x-position.
 *
 * @param input - Horizon target and scene geometry.
 * @returns Bottom anchor x-position whose ray reaches the target horizon x.
 */
function projectPlaybackGroundGridHorizonXToAnchorX(
  input: PlaybackGroundGridAnchorProjectionInput,
): number {
  const verticalProjectionDenominatorPx =
    input.sceneContext.alignedHorizonYPx - input.sceneContext.vanishingPointYPx;
  if (Math.abs(verticalProjectionDenominatorPx) < Number.EPSILON) {
    return input.horizonXPx;
  }

  const verticalProjectionRatio =
    (input.sceneContext.lowerBandBottomYPx - input.sceneContext.vanishingPointYPx) /
    verticalProjectionDenominatorPx;
  return (
    input.sceneContext.vanishingPointXPx +
    (input.horizonXPx - input.sceneContext.vanishingPointXPx) *
      verticalProjectionRatio
  );
}

type PlaybackGroundGridPoint = {
  xPx: number;
  yPx: number;
};

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
function interpolatePlaybackGroundGridPoint(
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