import {
  FLAPPY_GROUND_GRID_PULSE_ALPHA,
  FLAPPY_GROUND_GRID_PULSE_MAX_SIZE_PX,
  FLAPPY_GROUND_GRID_PULSE_MIN_SIZE_PX,
} from './playback.background.ground-grid.constants';
import {
  interpolatePlaybackGroundGridPoint,
  resolvePlaybackGroundGridDepthFromHorizonDistance,
  resolvePlaybackGroundGridLineThickness,
} from './playback.background.ground-grid.math.utils';
import {
  rememberPlaybackGroundGridVerticalPulseSelection,
  resolvePlaybackGroundGridVerticalPulseSelection,
} from './playback.background.ground-grid.pulse.selection.utils';
import {
  resolvePlaybackGroundGridHorizontalPulsePath,
  resolvePlaybackGroundGridPulseOrientation,
  resolvePlaybackGroundGridPulseTiming,
  resolvePlaybackGroundGridPulseTravelRatio,
  resolvePlaybackGroundGridUnitHash,
} from './playback.background.ground-grid.pulse.timing.utils';
import type {
  PlaybackBackgroundGroundGridSceneContext,
  PlaybackGroundGridPulse,
  PlaybackGroundGridPulseInput,
  PlaybackGroundGridPulsePath,
  PlaybackGroundGridPulseTrackThicknessInput,
} from './playback.background.ground-grid.types';

/**
 * Resolves one rare, deterministic pulse square for the current frame.
 *
 * @param input - Current frame timing and visible pulse path candidates.
 * @returns Visible pulse square, or null when the current slot is inactive.
 */
export function resolvePlaybackGroundGridPulse(
  input: PlaybackGroundGridPulseInput,
): PlaybackGroundGridPulse | null {
  const pulseTiming = resolvePlaybackGroundGridPulseTiming(input.frameIndex);
  if (!pulseTiming) {
    return null;
  }

  const pulseOrientation = resolvePlaybackGroundGridPulseOrientation(
    pulseTiming.pulseSlotIndex,
  );

  const directionIsForward =
    resolvePlaybackGroundGridUnitHash(pulseTiming.pulseSlotIndex, 29) >= 0.5;
  const travelProgressRatio = resolvePlaybackGroundGridPulseTravelRatio({
    directionIsForward,
    lifetimeProgressRatio: pulseTiming.lifetimeProgressRatio,
    orientation: pulseOrientation,
  });
  const selectedPulsePath =
    pulseOrientation === 'horizontal'
      ? resolvePlaybackGroundGridHorizontalPulsePath(
          input.horizontalPulsePaths,
          pulseTiming.pulseSlotIndex,
        )
      : resolvePlaybackGroundGridVerticalPulseSelection({
          verticalPulsePaths: input.verticalPulsePaths,
          visibleVerticalPulsePaths: input.visibleVerticalPulsePaths,
          pulseSlotIndex: pulseTiming.pulseSlotIndex,
          frameIndex: input.frameIndex,
          travelProgressRatio,
          resolveUnitHash: resolvePlaybackGroundGridUnitHash,
        });
  if (!selectedPulsePath) {
    return null;
  }

  const pulseCenter = interpolatePlaybackGroundGridPoint(
    selectedPulsePath.startXPx,
    selectedPulsePath.startYPx,
    selectedPulsePath.endXPx,
    selectedPulsePath.endYPx,
    travelProgressRatio,
  );

  if (pulseOrientation === 'vertical') {
    rememberPlaybackGroundGridVerticalPulseSelection(
      pulseTiming.pulseSlotIndex,
      {
        centerXPx: pulseCenter.xPx,
        centerYPx: pulseCenter.yPx,
        frameIndex: input.frameIndex,
      },
    );
  }

  const pulseTrackThicknessPx = resolvePlaybackGroundGridPulseTrackThickness({
    pulseCenterYPx: pulseCenter.yPx,
    pulsePath: selectedPulsePath,
    sceneContext: input.sceneContext,
  });

  return {
    centerXPx: pulseCenter.xPx,
    centerYPx: pulseCenter.yPx,
    sizePx: Math.max(
      FLAPPY_GROUND_GRID_PULSE_MIN_SIZE_PX,
      Math.min(FLAPPY_GROUND_GRID_PULSE_MAX_SIZE_PX, pulseTrackThicknessPx),
    ),
    alpha: FLAPPY_GROUND_GRID_PULSE_ALPHA,
  };
}

/**
 * Resolves the local track thickness at the pulse position.
 *
 * @param input - Pulse position, path, and scene geometry.
 * @returns Thickness of the current line under the pulse.
 */
function resolvePlaybackGroundGridPulseTrackThickness(
  input: PlaybackGroundGridPulseTrackThicknessInput,
): number {
  if (input.pulsePath.orientation === 'horizontal') {
    return input.pulsePath.thicknessPx;
  }

  const maximumDistanceToHorizonPx = Math.max(
    1,
    input.sceneContext.lowerBandBottomYPx -
      input.sceneContext.alignedHorizonYPx,
  );
  const depthRatio = resolvePlaybackGroundGridDepthFromHorizonDistance(
    input.pulseCenterYPx - input.sceneContext.alignedHorizonYPx,
    maximumDistanceToHorizonPx,
  );
  return resolvePlaybackGroundGridLineThickness(depthRatio);
}
