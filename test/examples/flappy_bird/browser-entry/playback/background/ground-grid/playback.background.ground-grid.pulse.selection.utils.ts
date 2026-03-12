import { FLAPPY_GROUND_GRID_VERTICAL_OVERFLOW_COUNT } from './playback.background.ground-grid.constants';
import { interpolatePlaybackGroundGridPoint } from './playback.background.ground-grid.math.utils';
import type {
  PlaybackGroundGridPulsePath,
  PlaybackGroundGridVerticalPulseContinuationState,
} from './playback.background.ground-grid.types';

const cachedVerticalPulseContinuationBySlot = new Map<
  number,
  PlaybackGroundGridVerticalPulseContinuationState
>();

/**
 * Resolves the current vertical pulse path while preserving per-slot continuity.
 *
 * A vertical pulse should stay attached to one moving ray for its whole
 * lifetime, even though the frame-local ray array is rebuilt as scroll wraps.
 * This helper first prefers the nearest continuation of the previous frame's
 * pulse position, then falls back to deterministic slot-based selection.
 *
 * @param verticalPulsePaths - Full vertical ray paths for the current frame.
 * @param visibleVerticalPulsePaths - Visible subset preferred for on-screen pulses.
 * @param pulseSlotIndex - Zero-based pulse slot index.
 * @param frameIndex - Current deterministic frame index.
 * @param travelProgressRatio - Current travel ratio along the chosen line.
 * @param resolveUnitHash - Deterministic unit-hash helper used for fallback picks.
 * @returns Vertical pulse path, or null when none are available.
 */
export function resolvePlaybackGroundGridVerticalPulseSelection(input: {
  verticalPulsePaths: readonly PlaybackGroundGridPulsePath[];
  visibleVerticalPulsePaths: readonly PlaybackGroundGridPulsePath[];
  pulseSlotIndex: number;
  frameIndex: number;
  travelProgressRatio: number;
  resolveUnitHash: (seed: number, salt: number) => number;
}): PlaybackGroundGridPulsePath | null {
  if (input.verticalPulsePaths.length === 0) {
    return null;
  }

  trimCachedVerticalPulseContinuationState(input.pulseSlotIndex);

  const candidatePulsePaths = resolveStableVerticalPulsePathCandidates(
    input.verticalPulsePaths,
    input.visibleVerticalPulsePaths,
  );
  const continuedPulsePath = resolveContinuedVerticalPulsePath({
    candidatePulsePaths,
    pulseSlotIndex: input.pulseSlotIndex,
    frameIndex: input.frameIndex,
    travelProgressRatio: input.travelProgressRatio,
  });
  if (continuedPulsePath) {
    return continuedPulsePath;
  }

  const selectedPathIndex = Math.min(
    candidatePulsePaths.length - 1,
    Math.floor(
      input.resolveUnitHash(input.pulseSlotIndex, 23) *
        candidatePulsePaths.length,
    ),
  );
  return candidatePulsePaths[selectedPathIndex];
}

/**
 * Stores the resolved pulse center for continuation on the next frame.
 *
 * @param pulseSlotIndex - Zero-based pulse slot index.
 * @param continuationState - Latest visible pulse position for the slot.
 * @returns Nothing.
 */
export function rememberPlaybackGroundGridVerticalPulseSelection(
  pulseSlotIndex: number,
  continuationState: PlaybackGroundGridVerticalPulseContinuationState,
): void {
  cachedVerticalPulseContinuationBySlot.set(pulseSlotIndex, continuationState);
}

/**
 * Resolves the nearest continued vertical pulse path for an active slot.
 *
 * @param input - Continuation input for the current frame.
 * @returns Continued pulse path when one can be matched, otherwise null.
 */
function resolveContinuedVerticalPulsePath(input: {
  candidatePulsePaths: readonly PlaybackGroundGridPulsePath[];
  pulseSlotIndex: number;
  frameIndex: number;
  travelProgressRatio: number;
}): PlaybackGroundGridPulsePath | null {
  const cachedContinuationState = cachedVerticalPulseContinuationBySlot.get(
    input.pulseSlotIndex,
  );
  if (!cachedContinuationState) {
    return null;
  }

  if (input.frameIndex <= cachedContinuationState.frameIndex) {
    return null;
  }

  let closestPulsePath: PlaybackGroundGridPulsePath | null = null;
  let closestDistanceSquared = Number.POSITIVE_INFINITY;

  for (const candidatePulsePath of input.candidatePulsePaths) {
    const candidateCenter = interpolatePlaybackGroundGridPoint(
      candidatePulsePath.startXPx,
      candidatePulsePath.startYPx,
      candidatePulsePath.endXPx,
      candidatePulsePath.endYPx,
      input.travelProgressRatio,
    );
    const deltaXPx = candidateCenter.xPx - cachedContinuationState.centerXPx;
    const deltaYPx = candidateCenter.yPx - cachedContinuationState.centerYPx;
    const distanceSquared = deltaXPx * deltaXPx + deltaYPx * deltaYPx;

    if (distanceSquared < closestDistanceSquared) {
      closestDistanceSquared = distanceSquared;
      closestPulsePath = candidatePulsePath;
    }
  }

  return closestPulsePath;
}

/**
 * Trims cached continuation state so only the current or previous pulse slots remain.
 *
 * @param currentPulseSlotIndex - Pulse slot currently being resolved.
 * @returns Nothing.
 */
function trimCachedVerticalPulseContinuationState(
  currentPulseSlotIndex: number,
): void {
  for (const cachedPulseSlotIndex of cachedVerticalPulseContinuationBySlot.keys()) {
    if (cachedPulseSlotIndex < currentPulseSlotIndex - 1) {
      cachedVerticalPulseContinuationBySlot.delete(cachedPulseSlotIndex);
    }
  }
}

/**
 * Resolves a stable vertical pulse-candidate set for one frame.
 *
 * @param verticalPulsePaths - Full vertical ray paths for the current frame.
 * @param visibleVerticalPulsePaths - Midpoint-visible subset used as fallback.
 * @returns Stable candidate set for deterministic vertical pulse selection.
 */
function resolveStableVerticalPulsePathCandidates(
  verticalPulsePaths: readonly PlaybackGroundGridPulsePath[],
  visibleVerticalPulsePaths: readonly PlaybackGroundGridPulsePath[],
): readonly PlaybackGroundGridPulsePath[] {
  const stableCandidatePaths = verticalPulsePaths.slice(
    FLAPPY_GROUND_GRID_VERTICAL_OVERFLOW_COUNT,
    Math.max(
      FLAPPY_GROUND_GRID_VERTICAL_OVERFLOW_COUNT,
      verticalPulsePaths.length - FLAPPY_GROUND_GRID_VERTICAL_OVERFLOW_COUNT,
    ),
  );
  if (stableCandidatePaths.length > 0) {
    return stableCandidatePaths;
  }

  if (visibleVerticalPulsePaths.length > 0) {
    return visibleVerticalPulsePaths;
  }

  return verticalPulsePaths;
}
