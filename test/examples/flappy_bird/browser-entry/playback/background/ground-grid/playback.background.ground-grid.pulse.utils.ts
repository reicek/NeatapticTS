import {
  FLAPPY_GROUND_GRID_APPROX_FRAME_DURATION_MS,
  FLAPPY_GROUND_GRID_PULSE_ALPHA,
  FLAPPY_GROUND_GRID_PULSE_GLOW_BLUR_PX,
  FLAPPY_GROUND_GRID_PULSE_INTERVAL_MS,
  FLAPPY_GROUND_GRID_PULSE_LIFETIME_MS,
  FLAPPY_GROUND_GRID_PULSE_MAX_SIZE_PX,
  FLAPPY_GROUND_GRID_PULSE_MIN_ELIGIBLE_THICKNESS_PX,
  FLAPPY_GROUND_GRID_PULSE_MIN_SIZE_PX,
  FLAPPY_GROUND_GRID_PULSE_PREFERRED_HORIZONTAL_START_RATIO,
  FLAPPY_GROUND_GRID_PULSE_VISIBLE_VIEWPORT_INSET_PX,
  FLAPPY_GROUND_GRID_UNSIGNED_NORMALIZATION_DIVISOR,
  FLAPPY_GROUND_GRID_VERTICAL_PULSE_END_RATIO,
  FLAPPY_GROUND_GRID_VERTICAL_PULSE_START_RATIO,
} from './playback.background.ground-grid.constants';
import {
  interpolatePlaybackGroundGridPoint,
  resolvePlaybackGroundGridDepthFromHorizonDistance,
  resolvePlaybackGroundGridLineThickness,
} from './playback.background.ground-grid.math.utils';
import type {
  PlaybackBackgroundGroundGridSceneContext,
  PlaybackGroundGridLineSegment,
  PlaybackGroundGridPulse,
  PlaybackGroundGridPulseOrientation,
  PlaybackGroundGridPulsePath,
} from './playback.background.ground-grid.types';

type PlaybackGroundGridPulseInput = {
  frameIndex: number;
  horizontalLines: readonly PlaybackGroundGridLineSegment[];
  sceneContext: PlaybackBackgroundGroundGridSceneContext;
  verticalPulsePaths: readonly PlaybackGroundGridPulsePath[];
};

type PlaybackGroundGridPulseTravelRatioInput = {
  directionIsForward: boolean;
  lifetimeProgressRatio: number;
  orientation: PlaybackGroundGridPulseOrientation;
};

type PlaybackGroundGridPulseTrackThicknessInput = {
  pulseCenterYPx: number;
  pulsePath: PlaybackGroundGridPulsePath;
  sceneContext: PlaybackBackgroundGroundGridSceneContext;
};

/**
 * Resolves one rare, deterministic pulse square for the current frame.
 *
 * @param input - Current frame timing and visible pulse path candidates.
 * @returns Visible pulse square, or null when the current slot is inactive.
 */
export function resolvePlaybackGroundGridPulse(
  input: PlaybackGroundGridPulseInput,
): PlaybackGroundGridPulse | null {
  const currentTimeMs =
    input.frameIndex * FLAPPY_GROUND_GRID_APPROX_FRAME_DURATION_MS;
  const pulseSlotIndex = Math.floor(
    currentTimeMs / FLAPPY_GROUND_GRID_PULSE_INTERVAL_MS,
  );
  const pulseElapsedMs =
    currentTimeMs - pulseSlotIndex * FLAPPY_GROUND_GRID_PULSE_INTERVAL_MS;
  if (pulseElapsedMs > FLAPPY_GROUND_GRID_PULSE_LIFETIME_MS) {
    return null;
  }

  const pulseOrientation =
    resolvePlaybackGroundGridPulseOrientation(pulseSlotIndex);
  const selectedPulsePath =
    pulseOrientation === 'horizontal'
      ? resolvePlaybackGroundGridHorizontalPulsePath(
          input.horizontalLines,
          pulseSlotIndex,
        )
      : resolvePlaybackGroundGridVerticalPulsePath(
          input.verticalPulsePaths,
          pulseSlotIndex,
          input.sceneContext,
        );
  if (!selectedPulsePath) {
    return null;
  }

  const directionIsForward =
    resolvePlaybackGroundGridUnitHash(pulseSlotIndex, 29) >= 0.5;
  const lifetimeProgressRatio =
    pulseElapsedMs / FLAPPY_GROUND_GRID_PULSE_LIFETIME_MS;
  const travelProgressRatio = resolvePlaybackGroundGridPulseTravelRatio({
    directionIsForward,
    lifetimeProgressRatio,
    orientation: pulseOrientation,
  });
  const pulseCenter = interpolatePlaybackGroundGridPoint(
    selectedPulsePath.startXPx,
    selectedPulsePath.startYPx,
    selectedPulsePath.endXPx,
    selectedPulsePath.endYPx,
    travelProgressRatio,
  );
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
    glowBlurPx: FLAPPY_GROUND_GRID_PULSE_GLOW_BLUR_PX,
  };
}

/**
 * Resolves pulse orientation for one deterministic pulse slot.
 *
 * @param pulseSlotIndex - Zero-based pulse slot index.
 * @returns Horizontal or vertical pulse travel orientation.
 */
function resolvePlaybackGroundGridPulseOrientation(
  pulseSlotIndex: number,
): PlaybackGroundGridPulseOrientation {
  return resolvePlaybackGroundGridUnitHash(pulseSlotIndex, 11) >= 0.5
    ? 'horizontal'
    : 'vertical';
}

/**
 * Selects one thick-enough horizontal band for the current pulse slot.
 *
 * @param horizontalLines - Visible horizontal grid bands.
 * @param pulseSlotIndex - Zero-based pulse slot index.
 * @returns Horizontal pulse path, or null when none are suitable.
 */
function resolvePlaybackGroundGridHorizontalPulsePath(
  horizontalLines: readonly PlaybackGroundGridLineSegment[],
  pulseSlotIndex: number,
): PlaybackGroundGridPulsePath | null {
  const eligibleHorizontalLines =
    resolvePlaybackGroundGridPreferredHorizontalPulseLines(horizontalLines);
  if (eligibleHorizontalLines.length === 0) {
    return null;
  }

  const selectedLineIndex = Math.min(
    eligibleHorizontalLines.length - 1,
    Math.floor(
      resolvePlaybackGroundGridUnitHash(pulseSlotIndex, 17) *
        eligibleHorizontalLines.length,
    ),
  );
  const selectedLine = eligibleHorizontalLines[selectedLineIndex];
  return {
    orientation: 'horizontal',
    startXPx: selectedLine.startXPx,
    startYPx: selectedLine.startYPx,
    endXPx: selectedLine.endXPx,
    endYPx: selectedLine.endYPx,
    thicknessPx: selectedLine.thicknessPx,
  };
}

/**
 * Resolves whether a horizontal line is thick enough to carry a visible pulse.
 *
 * @param horizontalLine - Candidate horizontal ground-grid line.
 * @returns True when the line should be considered pulse-eligible.
 */
function isPlaybackGroundGridHorizontalPulseLineEligible(
  horizontalLine: PlaybackGroundGridLineSegment,
): boolean {
  return (
    horizontalLine.thicknessPx >=
    FLAPPY_GROUND_GRID_PULSE_MIN_ELIGIBLE_THICKNESS_PX
  );
}

/**
 * Prefers the nearer, thicker horizontal tracks when picking a pulse lane.
 *
 * @param horizontalLines - Visible horizontal grid bands.
 * @returns Pulse-eligible horizontal lines biased toward the foreground.
 */
function resolvePlaybackGroundGridPreferredHorizontalPulseLines(
  horizontalLines: readonly PlaybackGroundGridLineSegment[],
): readonly PlaybackGroundGridLineSegment[] {
  const eligibleHorizontalLines = horizontalLines.filter(
    isPlaybackGroundGridHorizontalPulseLineEligible,
  );
  if (eligibleHorizontalLines.length === 0) {
    return eligibleHorizontalLines;
  }

  const preferredStartIndex = Math.min(
    eligibleHorizontalLines.length - 1,
    Math.floor(
      eligibleHorizontalLines.length *
        FLAPPY_GROUND_GRID_PULSE_PREFERRED_HORIZONTAL_START_RATIO,
    ),
  );
  return eligibleHorizontalLines.slice(preferredStartIndex);
}

/**
 * Selects one sparse vertical pulse path for the current pulse slot.
 *
 * @param verticalPulsePaths - Full vertical ray paths.
 * @param pulseSlotIndex - Zero-based pulse slot index.
 * @param sceneContext - Current lower-band scene geometry.
 * @returns Vertical pulse path, or null when none are available.
 */
function resolvePlaybackGroundGridVerticalPulsePath(
  verticalPulsePaths: readonly PlaybackGroundGridPulsePath[],
  pulseSlotIndex: number,
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
): PlaybackGroundGridPulsePath | null {
  if (verticalPulsePaths.length === 0) {
    return null;
  }

  const visibleVerticalPulsePaths = verticalPulsePaths.filter(
    (verticalPulsePath) =>
      isPlaybackGroundGridVerticalPulsePathVisible(
        verticalPulsePath,
        sceneContext,
      ),
  );
  const candidatePulsePaths =
    visibleVerticalPulsePaths.length > 0
      ? visibleVerticalPulsePaths
      : verticalPulsePaths;

  const selectedPathIndex = Math.min(
    candidatePulsePaths.length - 1,
    Math.floor(
      resolvePlaybackGroundGridUnitHash(pulseSlotIndex, 23) *
        candidatePulsePaths.length,
    ),
  );
  return candidatePulsePaths[selectedPathIndex];
}

/**
 * Resolves whether one vertical pulse path is safely visible in the viewport.
 *
 * @param pulsePath - Candidate vertical pulse path.
 * @param sceneContext - Current lower-band scene geometry.
 * @returns True when the pulse midpoint stays inside the visible ground band.
 */
function isPlaybackGroundGridVerticalPulsePathVisible(
  pulsePath: PlaybackGroundGridPulsePath,
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
): boolean {
  const midTravelRatio =
    FLAPPY_GROUND_GRID_VERTICAL_PULSE_START_RATIO +
    (FLAPPY_GROUND_GRID_VERTICAL_PULSE_END_RATIO -
      FLAPPY_GROUND_GRID_VERTICAL_PULSE_START_RATIO) *
      0.5;
  const pulseMidpoint = interpolatePlaybackGroundGridPoint(
    pulsePath.startXPx,
    pulsePath.startYPx,
    pulsePath.endXPx,
    pulsePath.endYPx,
    midTravelRatio,
  );
  const visibleLeftXPx =
    sceneContext.viewportLeftXPx +
    FLAPPY_GROUND_GRID_PULSE_VISIBLE_VIEWPORT_INSET_PX;
  const visibleRightXPx =
    sceneContext.viewportLeftXPx +
    sceneContext.visibleWorldWidthPx -
    FLAPPY_GROUND_GRID_PULSE_VISIBLE_VIEWPORT_INSET_PX;

  return (
    pulseMidpoint.xPx >= visibleLeftXPx && pulseMidpoint.xPx <= visibleRightXPx
  );
}

/**
 * Resolves the pulse travel ratio along its chosen line.
 *
 * @param input - Pulse timing direction and orientation.
 * @returns Normalized 0..1 travel ratio along the chosen line.
 */
function resolvePlaybackGroundGridPulseTravelRatio(
  input: PlaybackGroundGridPulseTravelRatioInput,
): number {
  const baseTravelProgressRatio = input.directionIsForward
    ? input.lifetimeProgressRatio
    : 1 - input.lifetimeProgressRatio;
  if (input.orientation === 'horizontal') {
    return baseTravelProgressRatio;
  }

  return (
    FLAPPY_GROUND_GRID_VERTICAL_PULSE_START_RATIO +
    (FLAPPY_GROUND_GRID_VERTICAL_PULSE_END_RATIO -
      FLAPPY_GROUND_GRID_VERTICAL_PULSE_START_RATIO) *
      baseTravelProgressRatio
  );
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

/**
 * Resolves a deterministic unit-interval hash from a slot index and salt.
 *
 * @param seed - Slot-local seed value.
 * @param salt - Small integer salt used to pick a stable random stream.
 * @returns Stable random value in the range 0..1.
 */
function resolvePlaybackGroundGridUnitHash(seed: number, salt: number): number {
  let hashedValue = (seed + 1) ^ Math.imul(salt + 1, 374_761_393);
  hashedValue = Math.imul(hashedValue ^ (hashedValue >>> 13), 1_274_126_177);
  hashedValue ^= hashedValue >>> 16;
  return (
    (hashedValue >>> 0) / FLAPPY_GROUND_GRID_UNSIGNED_NORMALIZATION_DIVISOR
  );
}
