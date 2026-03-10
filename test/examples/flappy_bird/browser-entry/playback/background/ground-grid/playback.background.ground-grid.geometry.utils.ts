import {
  FLAPPY_GROUND_GRID_HORIZONTAL_LINE_COUNT,
  FLAPPY_GROUND_GRID_MIN_VERTICAL_LINE_COUNT,
  FLAPPY_GROUND_GRID_PULSE_MIN_ELIGIBLE_THICKNESS_PX,
  FLAPPY_GROUND_GRID_PULSE_PREFERRED_HORIZONTAL_START_RATIO,
  FLAPPY_GROUND_GRID_PULSE_VISIBLE_VIEWPORT_INSET_PX,
  FLAPPY_GROUND_GRID_SCROLL_RATIO,
  FLAPPY_GROUND_GRID_TARGET_VERTICAL_LINE_SPACING_PX,
  FLAPPY_GROUND_GRID_TARGET_VERTICAL_SEGMENT_HEIGHT_PX,
  FLAPPY_GROUND_GRID_VERTICAL_OVERFLOW_COUNT,
  FLAPPY_GROUND_GRID_VERTICAL_PULSE_END_RATIO,
  FLAPPY_GROUND_GRID_VERTICAL_PULSE_START_RATIO,
} from './playback.background.ground-grid.constants';
import { positiveModulo } from '../../playback.starfield.utils';
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
  resolveGroundGridSceneCacheKey,
  resolveGroundGridVerticalCycleCacheKey,
} from './playback.background.ground-grid.cache.services';
import type {
  PlaybackGroundGridAnchorBounds,
  PlaybackGroundGridAnchorBoundsInput,
  PlaybackGroundGridAnchorProjectionInput,
  PlaybackBackgroundGroundGridSceneContext,
  PlaybackGroundGridHorizontalGeometry,
  PlaybackGroundGridHorizontalGeometryFactory,
  PlaybackGroundGridLineSegment,
  PlaybackGroundGridPulsePath,
  PlaybackGroundGridSegmentBatch,
  PlaybackGroundGridVerticalCycleContext,
  PlaybackGroundGridVerticalGeometry,
  PlaybackGroundGridVerticalRayInput,
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
  const verticalCycleContext = resolvePlaybackGroundGridVerticalCycleContext(
    sceneContext,
    scrollBasePx,
  );
  const cycleCacheKey = resolveGroundGridVerticalCycleCacheKey(
    sceneCacheKey,
    verticalCycleContext.quantizedWrappedOffsetPx,
  );

  return resolveCachedGroundGridVerticalGeometry(cycleCacheKey, () =>
    buildPlaybackGroundGridVerticalGeometry(
      sceneContext,
      verticalCycleContext.safeLaneSpacingPx,
      verticalCycleContext.quantizedWrappedOffsetPx,
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
 * Resolves the wrapped vertical-geometry cycle for the current scroll value.
 *
 * @param sceneContext - Lower-band geometry for the current viewport.
 * @param scrollBasePx - Shared world scroll used for parallax motion.
 * @returns Quantized wrapped offset and safe lane spacing for cache lookups.
 */
function resolvePlaybackGroundGridVerticalCycleContext(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
  scrollBasePx: number,
): PlaybackGroundGridVerticalCycleContext {
  const visibleAnchorBounds = resolvePlaybackGroundGridAnchorBounds({
    horizonLeftXPx: 0,
    horizonRightXPx: sceneContext.visibleWorldWidthPx,
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

  return {
    quantizedWrappedOffsetPx: Math.round(wrappedOffsetPx),
    safeLaneSpacingPx,
  };
}

/**
 * Builds the perspective rays that converge to the centered horizon point.
 *
 * @param sceneContext - Lower-band geometry for the current viewport.
 * @param safeLaneSpacingPx - Stable lane spacing used for ray anchors.
 * @param quantizedWrappedOffsetPx - Quantized wrapped offset used for cache reuse.
 * @returns Wrapped left-to-right perspective rays and pulse subsets.
 */
function buildPlaybackGroundGridVerticalGeometry(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
  safeLaneSpacingPx: number,
  quantizedWrappedOffsetPx: number,
): PlaybackGroundGridVerticalGeometry {
  const visibleAnchorBounds = resolvePlaybackGroundGridAnchorBounds({
    horizonLeftXPx: 0,
    horizonRightXPx: sceneContext.visibleWorldWidthPx,
    sceneContext,
  });
  const totalVisibleLaneCount = Math.max(
    FLAPPY_GROUND_GRID_MIN_VERTICAL_LINE_COUNT,
    Math.ceil(
      visibleAnchorBounds.anchorSpanPx /
        FLAPPY_GROUND_GRID_TARGET_VERTICAL_LINE_SPACING_PX,
    ) + 1,
  );
  const firstAnchorXPx =
    visibleAnchorBounds.leftAnchorXPx -
    FLAPPY_GROUND_GRID_VERTICAL_OVERFLOW_COUNT * safeLaneSpacingPx -
    quantizedWrappedOffsetPx;
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
  const verticalPulsePaths = new Array<PlaybackGroundGridPulsePath>(
    totalRayCount,
  );
  const verticalLineSegments = new Array<PlaybackGroundGridLineSegment>(
    totalRayCount * verticalSegmentCount,
  );
  const visibleVerticalPulsePaths: PlaybackGroundGridPulsePath[] = [];
  const midTravelRatio =
    FLAPPY_GROUND_GRID_VERTICAL_PULSE_START_RATIO +
    (FLAPPY_GROUND_GRID_VERTICAL_PULSE_END_RATIO -
      FLAPPY_GROUND_GRID_VERTICAL_PULSE_START_RATIO) *
      0.5;
  let segmentWriteIndex = 0;

  for (let rayIndex = 0; rayIndex < totalRayCount; rayIndex += 1) {
    // Step 1: Resolve the bottom anchor and its centered strength.
    const anchorXPx = firstAnchorXPx + rayIndex * safeLaneSpacingPx;
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
        midTravelRatio,
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

/**
 * Groups line segments into ordered style batches for lower-overhead drawing.
 *
 * @param segments - Ordered line segments that should preserve draw grouping.
 * @returns Ordered style batches that can be stroked with fewer state changes.
 */
function groupPlaybackGroundGridSegmentsByStyle(
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
 * Resolves one cached draw-ready path for a grouped segment batch.
 *
 * Browsers can stroke a reused Path2D more cheaply than replaying dozens of
 * moveTo/lineTo calls every frame. Test environments may not provide Path2D,
 * so callers must tolerate a null fallback and replay raw segments instead.
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

/**
 * Prefers the nearer, thicker horizontal tracks when picking a pulse lane.
 *
 * @param horizontalLines - Visible horizontal grid bands.
 * @returns Pulse-eligible horizontal paths biased toward the foreground.
 */
function resolvePlaybackGroundGridPreferredHorizontalPulsePaths(
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
 * Resolves whether one vertical pulse path is safely visible in the viewport.
 *
 * @param pulsePath - Candidate vertical pulse path.
 * @param sceneContext - Current lower-band scene geometry.
 * @param midTravelRatio - Midpoint travel ratio used for visibility checks.
 * @returns True when the pulse midpoint stays inside the visible ground band.
 */
function isPlaybackGroundGridVerticalPulsePathVisible(
  pulsePath: PlaybackGroundGridPulsePath,
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
  midTravelRatio: number,
): boolean {
  const pulseMidpointXPx =
    pulsePath.startXPx +
    (pulsePath.endXPx - pulsePath.startXPx) * midTravelRatio;
  const visibleLeftXPx = FLAPPY_GROUND_GRID_PULSE_VISIBLE_VIEWPORT_INSET_PX;
  const visibleRightXPx =
    sceneContext.visibleWorldWidthPx -
    FLAPPY_GROUND_GRID_PULSE_VISIBLE_VIEWPORT_INSET_PX;

  return (
    pulseMidpointXPx >= visibleLeftXPx && pulseMidpointXPx <= visibleRightXPx
  );
}
