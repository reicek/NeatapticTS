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
  PlaybackGroundGridAnchorProjectionInput,
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

  return {
    visibleAnchorBounds,
    totalVisibleLaneCount,
    safeLaneSpacingPx: Math.max(1, laneSpacingPx),
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
 * Projects the visible horizon span back onto the floor anchor line.
 *
 * @param input - Visible horizon bounds and scene geometry.
 * @returns Bottom-anchor bounds required to cover the full visible horizon.
 */
export function resolvePlaybackGroundGridAnchorBounds(
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
