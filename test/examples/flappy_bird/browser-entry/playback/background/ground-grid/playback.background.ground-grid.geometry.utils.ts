import {
  FLAPPY_GROUND_GRID_HORIZONTAL_LINE_COUNT,
  FLAPPY_GROUND_GRID_MIN_VERTICAL_LINE_COUNT,
  FLAPPY_GROUND_GRID_SCROLL_RATIO,
  FLAPPY_GROUND_GRID_TARGET_VERTICAL_LINE_SPACING_PX,
  FLAPPY_GROUND_GRID_TARGET_VERTICAL_SEGMENT_HEIGHT_PX,
  FLAPPY_GROUND_GRID_VERTICAL_OVERFLOW_COUNT,
} from './playback.background.ground-grid.constants';
import { positiveModulo } from '../../playback.starfield.utils';
import {
  interpolatePlaybackGroundGridPoint,
  resolvePlaybackGroundGridDepthCurve,
  resolvePlaybackGroundGridDepthFromHorizonDistance,
  resolvePlaybackGroundGridLineAlpha,
  resolvePlaybackGroundGridLineBlur,
  resolvePlaybackGroundGridLineThickness,
} from './playback.background.ground-grid.math.utils';
import type {
  PlaybackBackgroundGroundGridSceneContext,
  PlaybackGroundGridLineSegment,
  PlaybackGroundGridPulsePath,
} from './playback.background.ground-grid.types';

type PlaybackGroundGridVerticalSegmentsInput = PlaybackGroundGridPulsePath & {
  anchorXPx: number;
  centeredStrength: number;
  lineDepthRatio: number;
  maximumDistanceToHorizonPx: number;
  sceneContext: PlaybackBackgroundGroundGridSceneContext;
  verticalSegmentCount: number;
};

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

type PlaybackGroundGridAnchorProjectionInput = {
  horizonXPx: number;
  sceneContext: PlaybackBackgroundGroundGridSceneContext;
};

/**
 * Builds the screen-horizontal depth bands for the lower neon plane.
 *
 * @param sceneContext - Lower-band geometry for the current viewport.
 * @returns Ordered far-to-near line segments.
 */
export function resolvePlaybackGroundGridHorizontalLines(
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
        thicknessPx:
          resolvePlaybackGroundGridLineThickness(thicknessDepthRatio),
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
export function resolvePlaybackGroundGridVerticalLines(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
  scrollBasePx: number,
): {
  verticalLines: readonly PlaybackGroundGridLineSegment[];
  verticalPulsePaths: readonly PlaybackGroundGridPulsePath[];
} {
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
      Math.abs(
        visibleAnchorBounds.leftAnchorXPx - sceneContext.vanishingPointXPx,
      ),
      Math.abs(
        visibleAnchorBounds.rightAnchorXPx - sceneContext.vanishingPointXPx,
      ),
    ) +
    FLAPPY_GROUND_GRID_VERTICAL_OVERFLOW_COUNT * safeLaneSpacingPx;
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

  const verticalPulsePaths = Array.from(
    { length: totalRayCount },
    (_unusedValue, rayIndex) => {
      // Step 1: Resolve the bottom anchor and its centered strength.
      const anchorXPx = firstAnchorXPx + rayIndex * safeLaneSpacingPx;
      const lateralDistancePx = Math.abs(
        anchorXPx - sceneContext.vanishingPointXPx,
      );
      const centeredStrength =
        1 - Math.min(1, lateralDistancePx / maximumLateralDistancePx);
      const lineDepthRatio = 0.45 + centeredStrength * 0.55;

      // Step 2: Split the ray into short segments so width can taper to 1px.
      return {
        orientation: 'vertical' as const,
        anchorXPx,
        endXPx: sceneContext.vanishingPointXPx,
        endYPx: sceneContext.vanishingPointYPx,
        startXPx: anchorXPx,
        startYPx: sceneContext.lowerBandBottomYPx,
        thicknessPx: resolvePlaybackGroundGridLineThickness(1),
        centeredStrength,
        lineDepthRatio,
        maximumDistanceToHorizonPx,
        sceneContext,
        verticalSegmentCount,
      };
    },
  );
  const verticalLines = verticalPulsePaths.flatMap((verticalPulsePath) =>
    resolvePlaybackGroundGridVerticalLineSegments(verticalPulsePath),
  );

  return {
    verticalLines,
    verticalPulsePaths,
  };
}

/**
 * Builds tapered style segments for one perspective ray.
 *
 * @param input - Geometry and depth context for one ray.
 * @returns Ordered near-to-far segments for one perspective ray.
 */
function resolvePlaybackGroundGridVerticalLineSegments(
  input: PlaybackGroundGridVerticalSegmentsInput,
): readonly PlaybackGroundGridLineSegment[] {
  return Array.from(
    { length: input.verticalSegmentCount },
    (_unusedValue, segmentIndex) => {
      // Step 1: Resolve the interpolation bounds for this sub-segment.
      const segmentStartRatio = segmentIndex / input.verticalSegmentCount;
      const segmentEndRatio = (segmentIndex + 1) / input.verticalSegmentCount;
      const segmentMidpointRatio = (segmentStartRatio + segmentEndRatio) * 0.5;
      const segmentStartPoint = interpolatePlaybackGroundGridPoint(
        input.startXPx,
        input.startYPx,
        input.endXPx,
        input.endYPx,
        segmentStartRatio,
      );
      const segmentEndPoint = interpolatePlaybackGroundGridPoint(
        input.startXPx,
        input.startYPx,
        input.endXPx,
        input.endYPx,
        segmentEndRatio,
      );

      // Step 2: Resolve style from the segment midpoint distance to the horizon.
      const segmentMidpoint = interpolatePlaybackGroundGridPoint(
        input.startXPx,
        input.startYPx,
        input.endXPx,
        input.endYPx,
        segmentMidpointRatio,
      );
      const segmentDepthRatio =
        resolvePlaybackGroundGridDepthFromHorizonDistance(
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
    },
  );
}

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
    (input.sceneContext.lowerBandBottomYPx -
      input.sceneContext.vanishingPointYPx) /
    verticalProjectionDenominatorPx;
  return (
    input.sceneContext.vanishingPointXPx +
    (input.horizonXPx - input.sceneContext.vanishingPointXPx) *
      verticalProjectionRatio
  );
}
