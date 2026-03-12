import {
  FLAPPY_GROUND_GRID_APPROX_FRAME_DURATION_MS,
  FLAPPY_GROUND_GRID_PULSE_INTERVAL_MS,
  FLAPPY_GROUND_GRID_PULSE_LIFETIME_MS,
  FLAPPY_GROUND_GRID_UNSIGNED_NORMALIZATION_DIVISOR,
  FLAPPY_GROUND_GRID_VERTICAL_PULSE_END_RATIO,
  FLAPPY_GROUND_GRID_VERTICAL_PULSE_START_RATIO,
} from './playback.background.ground-grid.constants';
import type {
  PlaybackGroundGridPulseOrientation,
  PlaybackGroundGridPulsePath,
  PlaybackGroundGridPulseTimingState,
  PlaybackGroundGridPulseTravelRatioInput,
} from './playback.background.ground-grid.types';

/**
 * Resolves timing state for the currently active deterministic pulse slot.
 *
 * @param frameIndex - Current deterministic playback frame index.
 * @returns Pulse timing state, or null when no pulse is active in this frame.
 */
export function resolvePlaybackGroundGridPulseTiming(
  frameIndex: number,
): PlaybackGroundGridPulseTimingState | null {
  const currentTimeMs =
    frameIndex * FLAPPY_GROUND_GRID_APPROX_FRAME_DURATION_MS;
  const pulseSlotIndex = Math.floor(
    currentTimeMs / FLAPPY_GROUND_GRID_PULSE_INTERVAL_MS,
  );
  const pulseElapsedMs =
    currentTimeMs - pulseSlotIndex * FLAPPY_GROUND_GRID_PULSE_INTERVAL_MS;
  if (pulseElapsedMs > FLAPPY_GROUND_GRID_PULSE_LIFETIME_MS) {
    return null;
  }

  return {
    pulseSlotIndex,
    pulseElapsedMs,
    lifetimeProgressRatio:
      pulseElapsedMs / FLAPPY_GROUND_GRID_PULSE_LIFETIME_MS,
  };
}

/**
 * Resolves pulse orientation for one deterministic pulse slot.
 *
 * @param pulseSlotIndex - Zero-based pulse slot index.
 * @returns Horizontal or vertical pulse travel orientation.
 */
export function resolvePlaybackGroundGridPulseOrientation(
  pulseSlotIndex: number,
): PlaybackGroundGridPulseOrientation {
  return resolvePlaybackGroundGridUnitHash(pulseSlotIndex, 11) >= 0.5
    ? 'horizontal'
    : 'vertical';
}

/**
 * Selects one thick-enough horizontal band for the current pulse slot.
 *
 * @param horizontalPulsePaths - Cached horizontal pulse paths eligible for travel.
 * @param pulseSlotIndex - Zero-based pulse slot index.
 * @returns Horizontal pulse path, or null when none are suitable.
 */
export function resolvePlaybackGroundGridHorizontalPulsePath(
  horizontalPulsePaths: readonly PlaybackGroundGridPulsePath[],
  pulseSlotIndex: number,
): PlaybackGroundGridPulsePath | null {
  if (horizontalPulsePaths.length === 0) {
    return null;
  }

  const selectedLineIndex = Math.min(
    horizontalPulsePaths.length - 1,
    Math.floor(
      resolvePlaybackGroundGridUnitHash(pulseSlotIndex, 17) *
        horizontalPulsePaths.length,
    ),
  );
  return horizontalPulsePaths[selectedLineIndex];
}

/**
 * Resolves the pulse travel ratio along its chosen line.
 *
 * @param input - Pulse timing direction and orientation.
 * @returns Normalized 0..1 travel ratio along the chosen line.
 */
export function resolvePlaybackGroundGridPulseTravelRatio(
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
 * Resolves a deterministic unit-interval hash from a slot index and salt.
 *
 * @param seed - Slot-local seed value.
 * @param salt - Small integer salt used to pick a stable random stream.
 * @returns Stable random value in the range 0..1.
 */
export function resolvePlaybackGroundGridUnitHash(
  seed: number,
  salt: number,
): number {
  let hashedValue = (seed + 1) ^ Math.imul(salt + 1, 374_761_393);
  hashedValue = Math.imul(hashedValue ^ (hashedValue >>> 13), 1_274_126_177);
  hashedValue ^= hashedValue >>> 16;
  return (
    (hashedValue >>> 0) / FLAPPY_GROUND_GRID_UNSIGNED_NORMALIZATION_DIVISOR
  );
}