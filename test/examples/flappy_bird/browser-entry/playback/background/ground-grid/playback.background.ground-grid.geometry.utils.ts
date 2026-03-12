import {
  FLAPPY_GROUND_GRID_HORIZONTAL_LINE_COUNT,
  FLAPPY_GROUND_GRID_TARGET_VERTICAL_LINE_SPACING_PX,
  FLAPPY_GROUND_GRID_TARGET_VERTICAL_SEGMENT_HEIGHT_PX,
  FLAPPY_GROUND_GRID_VERTICAL_OVERFLOW_COUNT,
} from './playback.background.ground-grid.constants';
import {
  resolvePlaybackGroundGridDepthCurve,
  resolvePlaybackGroundGridDepthFromHorizonDistance,
  resolvePlaybackGroundGridLineAlpha,
  resolvePlaybackGroundGridLineBlur,
  resolvePlaybackGroundGridLineThickness,
} from './playback.background.ground-grid.math.utils';
import {
  ensureGroundGridViewportCacheValidity,
  resolveCachedGroundGridHorizontalGeometry,
  resolveCachedGroundGridVerticalGeometry,
  resolveCachedGroundGridVerticalSceneMetrics,
  resolveGroundGridSceneCacheKey,
  resolveGroundGridVerticalCycleCacheKey,
} from './playback.background.ground-grid.cache.services';
import {
  groupPlaybackGroundGridSegmentsByStyle,
  resolvePlaybackGroundGridPreferredHorizontalPulsePaths,
} from './playback.background.ground-grid.geometry.batch.utils';
import {
  buildPlaybackGroundGridVerticalSceneMetrics,
  isPlaybackGroundGridVerticalPulsePathVisible,
  resolvePlaybackGroundGridVerticalCycleContext,
} from './playback.background.ground-grid.geometry.layout.utils';
import type {
  PlaybackBackgroundGroundGridSceneContext,
  PlaybackGroundGridHorizontalGeometry,
  PlaybackGroundGridHorizontalGeometryFactory,
  PlaybackGroundGridLineSegment,
  PlaybackGroundGridPulsePath,
  PlaybackGroundGridSegmentBatch,
  PlaybackGroundGridVerticalGeometry,
  PlaybackGroundGridVerticalRayInput,
  PlaybackGroundGridVerticalSceneMetrics,
} from './playback.background.ground-grid.types';

/**
 * Resolves cached screen-horizontal depth bands for the lower neon plane.
 *
 * @param sceneContext - Lower-band geometry for the current viewport.
 * @returns Ordered far-to-near line segments and pulse subsets.
 */
export function resolvePlaybackGroundGridHorizontalGeometry(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
): PlaybackGroundGridHorizontalGeometry {
  ensureGroundGridViewportCacheValidity(sceneContext);
  const sceneCacheKey = resolveGroundGridSceneCacheKey(sceneContext);

  return resolveCachedGroundGridHorizontalGeometry(sceneCacheKey, () =>
    buildPlaybackGroundGridHorizontalGeometry(sceneContext),
  );
}

/**
 * Resolves cached perspective rays that converge to the centered horizon point.
 *
 * @param sceneContext - Lower-band geometry for the current viewport.
 * @param scrollBasePx - Shared world scroll used for parallax motion.
 * @returns Wrapped left-to-right perspective rays and pulse subsets.
 */
export function resolvePlaybackGroundGridVerticalGeometry(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
  scrollBasePx: number,
): PlaybackGroundGridVerticalGeometry {
  ensureGroundGridViewportCacheValidity(sceneContext);

  const sceneCacheKey = resolveGroundGridSceneCacheKey(sceneContext);
  const verticalSceneMetrics = resolveCachedGroundGridVerticalSceneMetrics(
    sceneCacheKey,
    () => buildPlaybackGroundGridVerticalSceneMetrics(sceneContext),
  );
  const verticalCycleContext = resolvePlaybackGroundGridVerticalCycleContext(
    verticalSceneMetrics.safeLaneSpacingPx,
    sceneContext.lowerBandBottomYPx,
    scrollBasePx,
  );
  const cycleCacheKey = resolveGroundGridVerticalCycleCacheKey(
    sceneCacheKey,
    verticalCycleContext.wrappedOffsetPx,
  );

  return resolveCachedGroundGridVerticalGeometry(cycleCacheKey, () =>
    buildPlaybackGroundGridVerticalGeometry(
      sceneContext,
      verticalSceneMetrics,
      verticalCycleContext.wrappedOffsetPx,
    ),
  );
}

/**
 * Builds the screen-horizontal depth bands for the lower neon plane.
 *
 * @param sceneContext - Lower-band geometry for the current viewport.
 * @returns Ordered far-to-near line segments and pulse subsets.
 */
function buildPlaybackGroundGridHorizontalGeometry(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
): PlaybackGroundGridHorizontalGeometry {
  const maximumDistanceToHorizonPx = Math.max(
    1,
    sceneContext.lowerBandBottomYPx - sceneContext.alignedHorizonYPx,
  );
  const horizontalLines = new Array<PlaybackGroundGridLineSegment>(
    FLAPPY_GROUND_GRID_HORIZONTAL_LINE_COUNT,
  );

  for (
    let lineIndex = 0;
    lineIndex < FLAPPY_GROUND_GRID_HORIZONTAL_LINE_COUNT;
    lineIndex += 1
  ) {
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
    horizontalLines[lineIndex] = {
      startXPx: 0,
      startYPx: lineYPx,
      endXPx: sceneContext.visibleWorldWidthPx,
      endYPx: lineYPx,
      alpha: resolvePlaybackGroundGridLineAlpha(depthRatio),
      blurPx: resolvePlaybackGroundGridLineBlur(depthRatio),
      thicknessPx: resolvePlaybackGroundGridLineThickness(thicknessDepthRatio),
    };
  }

  return {
    horizontalLineBatches:
      groupPlaybackGroundGridSegmentsByStyle(horizontalLines),
    horizontalLines,
    preferredHorizontalPulsePaths:
      resolvePlaybackGroundGridPreferredHorizontalPulsePaths(horizontalLines),
  };
}

/**
 * Builds the perspective rays that converge to the centered horizon point.
 *
 * @param sceneContext - Lower-band geometry for the current viewport.
 * @param safeLaneSpacingPx - Stable lane spacing used for ray anchors.
 * @param wrappedOffsetPx - Wrapped offset used for cache reuse and ray placement.
 * @returns Wrapped left-to-right perspective rays and pulse subsets.
 */
function buildPlaybackGroundGridVerticalGeometry(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
  verticalSceneMetrics: PlaybackGroundGridVerticalSceneMetrics,
  wrappedOffsetPx: number,
): PlaybackGroundGridVerticalGeometry {
  const firstAnchorXPx =
    verticalSceneMetrics.visibleAnchorBounds.leftAnchorXPx -
    FLAPPY_GROUND_GRID_VERTICAL_OVERFLOW_COUNT *
      verticalSceneMetrics.safeLaneSpacingPx -
    wrappedOffsetPx;
  const totalRayCount =
    verticalSceneMetrics.totalVisibleLaneCount +
    FLAPPY_GROUND_GRID_VERTICAL_OVERFLOW_COUNT * 2 +
    1;
  const maximumLateralDistancePx =
    Math.max(
      Math.abs(
        verticalSceneMetrics.visibleAnchorBounds.leftAnchorXPx -
          sceneContext.vanishingPointXPx,
      ),
      Math.abs(
        verticalSceneMetrics.visibleAnchorBounds.rightAnchorXPx -
          sceneContext.vanishingPointXPx,
      ),
    ) +
    FLAPPY_GROUND_GRID_VERTICAL_OVERFLOW_COUNT *
      verticalSceneMetrics.safeLaneSpacingPx;
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
  const verticalPulsePaths = new Array<PlaybackGroundGridPulsePath>(
    totalRayCount,
  );
  const verticalLineSegments = new Array<PlaybackGroundGridLineSegment>(
    totalRayCount * verticalSegmentCount,
  );
  const visibleVerticalPulsePaths: PlaybackGroundGridPulsePath[] = [];
  let segmentWriteIndex = 0;

  for (let rayIndex = 0; rayIndex < totalRayCount; rayIndex += 1) {
    // Step 1: Resolve the bottom anchor and its centered strength.
    const anchorXPx =
      firstAnchorXPx + rayIndex * verticalSceneMetrics.safeLaneSpacingPx;
    const lateralDistancePx = Math.abs(
      anchorXPx - sceneContext.vanishingPointXPx,
    );
    const centeredStrength =
      1 - Math.min(1, lateralDistancePx / maximumLateralDistancePx);
    const lineDepthRatio = 0.45 + centeredStrength * 0.55;
    const verticalPulsePath: PlaybackGroundGridPulsePath = {
      orientation: 'vertical',
      startXPx: anchorXPx,
      startYPx: sceneContext.lowerBandBottomYPx,
      endXPx: sceneContext.vanishingPointXPx,
      endYPx: sceneContext.vanishingPointYPx,
      thicknessPx: resolvePlaybackGroundGridLineThickness(1),
    };

    verticalPulsePaths[rayIndex] = verticalPulsePath;
    if (
      isPlaybackGroundGridVerticalPulsePathVisible(
        verticalPulsePath,
        sceneContext,
      )
    ) {
      visibleVerticalPulsePaths.push(verticalPulsePath);
    }

    segmentWriteIndex = appendPlaybackGroundGridVerticalLineSegments(
      verticalLineSegments,
      segmentWriteIndex,
      {
        ...verticalPulsePath,
        lineDepthRatio,
        maximumDistanceToHorizonPx,
        sceneContext,
        verticalSegmentCount,
      },
    );
  }

  return {
    verticalLineBatches:
      groupPlaybackGroundGridSegmentsByStyle(verticalLineSegments),
    verticalPulsePaths,
    visibleVerticalPulsePaths,
  };
}

/**
 * Appends tapered style segments for one perspective ray.
 *
 * @param targetSegments - Target line-segment buffer.
 * @param startIndex - Current insertion index within the target buffer.
 * @param input - Geometry and depth context for one ray.
 * @returns Next insertion index after all ray segments have been written.
 */
function appendPlaybackGroundGridVerticalLineSegments(
  targetSegments: PlaybackGroundGridLineSegment[],
  startIndex: number,
  input: PlaybackGroundGridVerticalRayInput,
): number {
  const rayDeltaXPx = input.endXPx - input.startXPx;
  const rayDeltaYPx = input.endYPx - input.startYPx;
  let writeIndex = startIndex;

  for (
    let segmentIndex = 0;
    segmentIndex < input.verticalSegmentCount;
    segmentIndex += 1
  ) {
    // Step 1: Resolve the interpolation bounds for this sub-segment.
    const segmentStartRatio = segmentIndex / input.verticalSegmentCount;
    const segmentEndRatio = (segmentIndex + 1) / input.verticalSegmentCount;
    const segmentMidpointRatio = (segmentStartRatio + segmentEndRatio) * 0.5;
    const segmentStartXPx = input.startXPx + rayDeltaXPx * segmentStartRatio;
    const segmentStartYPx = input.startYPx + rayDeltaYPx * segmentStartRatio;
    const segmentEndXPx = input.startXPx + rayDeltaXPx * segmentEndRatio;
    const segmentEndYPx = input.startYPx + rayDeltaYPx * segmentEndRatio;

    // Step 2: Resolve style from the segment midpoint distance to the horizon.
    const segmentMidpointYPx =
      input.startYPx + rayDeltaYPx * segmentMidpointRatio;
    const segmentDepthRatio = resolvePlaybackGroundGridDepthFromHorizonDistance(
      segmentMidpointYPx - input.sceneContext.alignedHorizonYPx,
      input.maximumDistanceToHorizonPx,
    );
    const combinedAlphaDepthRatio =
      segmentDepthRatio * 0.7 + input.lineDepthRatio * 0.3;

    // Step 3: Append the tapered segment so the ray narrows toward the horizon.
    targetSegments[writeIndex] = {
      startXPx: segmentStartXPx,
      startYPx: segmentStartYPx,
      endXPx: segmentEndXPx,
      endYPx: segmentEndYPx,
      alpha: resolvePlaybackGroundGridLineAlpha(combinedAlphaDepthRatio),
      blurPx: resolvePlaybackGroundGridLineBlur(segmentDepthRatio),
      thicknessPx: resolvePlaybackGroundGridLineThickness(segmentDepthRatio),
    };
    writeIndex += 1;
  }

  return writeIndex;
}

