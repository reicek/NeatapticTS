import {
  FLAPPY_GROUND_GRID_MIN_VERTICAL_LINE_COUNT,
  FLAPPY_GROUND_GRID_PULSE_VISIBLE_VIEWPORT_INSET_PX,
  FLAPPY_GROUND_GRID_SCROLL_OFFSET_QUANTIZATION_DECIMALS,
  FLAPPY_GROUND_GRID_SCROLL_RATIO,
  FLAPPY_GROUND_GRID_TARGET_VERTICAL_LINE_SPACING_PX,
  FLAPPY_GROUND_GRID_VERTICAL_PULSE_END_RATIO,
  FLAPPY_GROUND_GRID_VERTICAL_PULSE_START_RATIO,
} from './playback.background.ground-grid.constants';
import { resolvePlaybackGroundGridPipeConnectionProfile } from './playback.background.ground-grid.math.utils';
import { positiveModulo } from '../../playback.starfield.utils';
import type {
  PlaybackGroundGridAnchorBounds,
  PlaybackGroundGridAnchorBoundsInput,
  PlaybackBackgroundGroundGridSceneContext,
  PlaybackGroundGridPulsePath,
  PlaybackGroundGridVerticalCycleContext,
  PlaybackGroundGridVerticalSceneMetrics,
} from './playback.background.ground-grid.types';

/**
 * Builds the static scene metrics reused across one viewport-sized grid layout.
 *
 * @param sceneContext - Lower-band geometry for the current viewport.
 * @returns Stable anchor bounds and lane spacing for vertical-ray reuse.
 */
export function buildPlaybackGroundGridVerticalSceneMetrics(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
): PlaybackGroundGridVerticalSceneMetrics {
  const visibleAnchorBounds = resolvePlaybackGroundGridAnchorBounds({
    horizonLeftXPx: 0,
    horizonRightXPx: sceneContext.visibleWorldWidthPx,
    sceneContext,
  });
  const pipeConnectionProfile = resolvePlaybackGroundGridPipeConnectionProfile(
    sceneContext.lowerBandBottomYPx,
  );
  const retainedWidthRatioAtPipeConnection =
    resolvePlaybackGroundGridRetainedWidthRatioAtYPx(
      sceneContext.lowerBandBottomYPx,
      sceneContext.vanishingPointYPx,
      pipeConnectionProfile.pipeFloorYPx,
    );
  const safeLaneSpacingPx = Math.max(
    1,
    FLAPPY_GROUND_GRID_TARGET_VERTICAL_LINE_SPACING_PX /
      Math.max(Number.EPSILON, retainedWidthRatioAtPipeConnection),
  );
  const totalVisibleLaneCount = Math.max(
    FLAPPY_GROUND_GRID_MIN_VERTICAL_LINE_COUNT,
    Math.ceil(visibleAnchorBounds.anchorSpanPx / safeLaneSpacingPx) + 1,
  );

  return {
    visibleAnchorBounds,
    totalVisibleLaneCount,
    safeLaneSpacingPx,
  };
}

/**
 * Resolves the wrapped vertical-geometry cycle for the current scroll value.
 *
 * @param safeLaneSpacingPx - Stable lane spacing used by current viewport metrics.
 * @param lowerBandBottomYPx - Lower edge of the visible ground-grid band.
 * @param scrollBasePx - Shared world scroll used for parallax motion.
 * @returns Quantized wrapped offset and safe lane spacing for cache lookups.
 */
export function resolvePlaybackGroundGridVerticalCycleContext(
  safeLaneSpacingPx: number,
  lowerBandBottomYPx: number,
  scrollBasePx: number,
): PlaybackGroundGridVerticalCycleContext {
  const pipeConnectionProfile =
    resolvePlaybackGroundGridPipeConnectionProfile(lowerBandBottomYPx);
  const wrappedOffsetPx = positiveModulo(
    scrollBasePx *
      FLAPPY_GROUND_GRID_SCROLL_RATIO *
      pipeConnectionProfile.matchedRayScrollRatio,
    safeLaneSpacingPx,
  );
  const wrappedOffsetPrecisionScale = Math.pow(
    10,
    FLAPPY_GROUND_GRID_SCROLL_OFFSET_QUANTIZATION_DECIMALS,
  );

  return {
    wrappedOffsetPx:
      Math.round(wrappedOffsetPx * wrappedOffsetPrecisionScale) /
      wrappedOffsetPrecisionScale,
    safeLaneSpacingPx,
  };
}

/**
 * Projects one visible horizontal span back onto the floor anchor line.
 *
 * @param input - Visible span bounds and scene geometry.
 * @returns Bottom-anchor bounds required to cover the chosen projected span.
 */
export function resolvePlaybackGroundGridAnchorBounds(
  input: PlaybackGroundGridAnchorBoundsInput,
): PlaybackGroundGridAnchorBounds {
  const leftAnchorXPx = projectPlaybackGroundGridProjectedXToAnchorX({
    projectedXPx: input.horizonLeftXPx,
    projectedYPx: input.sceneContext.alignedHorizonYPx,
    sceneContext: input.sceneContext,
  });
  const rightAnchorXPx = projectPlaybackGroundGridProjectedXToAnchorX({
    projectedXPx: input.horizonRightXPx,
    projectedYPx: input.sceneContext.alignedHorizonYPx,
    sceneContext: input.sceneContext,
  });

  return {
    leftAnchorXPx,
    rightAnchorXPx,
    anchorSpanPx: Math.max(1, rightAnchorXPx - leftAnchorXPx),
  };
}

/**
 * Resolves whether one vertical pulse path is safely visible in the viewport.
 *
 * @param pulsePath - Candidate vertical pulse path.
 * @param sceneContext - Current lower-band scene geometry.
 * @returns True when the pulse midpoint stays inside the visible ground band.
 */
export function isPlaybackGroundGridVerticalPulsePathVisible(
  pulsePath: PlaybackGroundGridPulsePath,
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
): boolean {
  const midTravelRatio =
    FLAPPY_GROUND_GRID_VERTICAL_PULSE_START_RATIO +
    (FLAPPY_GROUND_GRID_VERTICAL_PULSE_END_RATIO -
      FLAPPY_GROUND_GRID_VERTICAL_PULSE_START_RATIO) *
      0.5;
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

/**
 * Projects one visible x-position at an arbitrary y-level to the anchor line.
 *
 * @param input - Projected target and scene geometry.
 * @returns Bottom anchor x-position whose ray reaches the projected point.
 */
function projectPlaybackGroundGridProjectedXToAnchorX(input: {
  projectedXPx: number;
  projectedYPx: number;
  sceneContext: PlaybackBackgroundGroundGridSceneContext;
}): number {
  const verticalProjectionDenominatorPx =
    input.projectedYPx - input.sceneContext.vanishingPointYPx;
  if (Math.abs(verticalProjectionDenominatorPx) < Number.EPSILON) {
    return input.projectedXPx;
  }

  const verticalProjectionRatio =
    (input.sceneContext.lowerBandBottomYPx -
      input.sceneContext.vanishingPointYPx) /
    verticalProjectionDenominatorPx;
  return (
    input.sceneContext.vanishingPointXPx +
    (input.projectedXPx - input.sceneContext.vanishingPointXPx) *
      verticalProjectionRatio
  );
}

/**
 * Resolves how much adjacent-ray spacing remains at one projected y-position.
 *
 * Perspective rays linearly collapse toward the vanishing point, so the local
 * lane width at any y-position is just the bottom-anchor spacing multiplied by
 * the remaining width ratio between the bottom edge and the vanishing point.
 *
 * @param lowerBandBottomYPx - Bottom edge of the visible ground band.
 * @param vanishingPointYPx - Shared vanishing-point y-position.
 * @param targetYPx - Projected y-position whose retained width should be measured.
 * @returns Width-retention ratio in the inclusive `[0, 1]` range.
 */
function resolvePlaybackGroundGridRetainedWidthRatioAtYPx(
  lowerBandBottomYPx: number,
  vanishingPointYPx: number,
  targetYPx: number,
): number {
  const verticalTravelPx = vanishingPointYPx - lowerBandBottomYPx;
  if (Math.abs(verticalTravelPx) < Number.EPSILON) {
    return 1;
  }

  const interpolationRatio =
    (targetYPx - lowerBandBottomYPx) / verticalTravelPx;
  return Math.max(0, Math.min(1, 1 - interpolationRatio));
}
