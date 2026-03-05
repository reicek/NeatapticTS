import {
  FLAPPY_MEMORY_ACTION_WINDOW_STEPS,
  FLAPPY_MEMORY_CORE_FEATURE_COUNT,
  FLAPPY_MEMORY_STACKED_FRAME_COUNT,
} from '../constants/constants';
import { resolveCoreObservationVectorFromFeatures } from './simulation-shared.observation.utils';
import type {
  SharedObservationFeatures,
  SharedObservationMemoryState,
} from './simulation-shared.types';

/**
 * Creates an empty temporal observation memory state.
 *
 * @returns Fresh mutable memory buffers for one bird/controller.
 */
export function createSharedObservationMemoryState(): SharedObservationMemoryState {
  return {
    previousCoreObservationFrames: [],
    recentFlapActions: [],
  };
}

/**
 * Builds the temporal policy input vector (stacked observation + action memory).
 *
 * Output layout:
 * 1) current core observation frame
 * 2) previous core frames (newest to oldest) with zero padding
 * 3) last-action channel
 * 4) recent flap-rate channel over a fixed window
 *
 * @param features - Structured observation features for the current decision step.
 * @param observationMemoryState - Mutable temporal memory for the active bird.
 * @returns Ordered temporal input vector for policy activation.
 */
export function resolveTemporalObservationVector(
  features: SharedObservationFeatures,
  observationMemoryState: SharedObservationMemoryState,
): number[] {
  const currentCoreObservationFrame =
    resolveCoreObservationVectorFromFeatures(features);
  const stackedFrames = [
    currentCoreObservationFrame,
    ...resolvePreviousCoreFramesWithPadding(observationMemoryState),
  ];
  const flattenedStackedFrames = stackedFrames.flat();
  const lastActionChannel = observationMemoryState.recentFlapActions[0] ?? 0;
  const recentFlapRateChannel =
    observationMemoryState.recentFlapActions.length === 0
      ? 0
      : observationMemoryState.recentFlapActions.reduce(
          (actionSum, actionValue) => actionSum + actionValue,
          0,
        ) / observationMemoryState.recentFlapActions.length;

  return [...flattenedStackedFrames, lastActionChannel, recentFlapRateChannel];
}

/**
 * Commits one observation-action step into temporal memory.
 *
 * @param observationMemoryState - Mutable temporal memory for the active bird.
 * @param features - Structured observation features used for the decision.
 * @param didFlap - Decision taken at this step.
 * @returns Nothing.
 */
export function commitSharedObservationMemoryStep(
  observationMemoryState: SharedObservationMemoryState,
  features: SharedObservationFeatures,
  didFlap: boolean,
): void {
  // Step 1: Persist current core frame for future stacked observations.
  const currentCoreObservationFrame =
    resolveCoreObservationVectorFromFeatures(features);
  const previousFrameCapacity = Math.max(
    0,
    FLAPPY_MEMORY_STACKED_FRAME_COUNT - 1,
  );
  observationMemoryState.previousCoreObservationFrames = [
    currentCoreObservationFrame,
    ...observationMemoryState.previousCoreObservationFrames,
  ].slice(0, previousFrameCapacity);

  // Step 2: Persist latest action into a fixed-size action history window.
  const actionScalar = didFlap ? 1 : 0;
  observationMemoryState.recentFlapActions = [
    actionScalar,
    ...observationMemoryState.recentFlapActions,
  ].slice(0, FLAPPY_MEMORY_ACTION_WINDOW_STEPS);
}

/**
 * Resolves previous core frames (newest-first) with deterministic zero padding.
 *
 * @param observationMemoryState - Mutable temporal memory for the active bird.
 * @returns Previous core frame list with fixed target length.
 */
function resolvePreviousCoreFramesWithPadding(
  observationMemoryState: SharedObservationMemoryState,
): number[][] {
  const previousFrameTargetCount = Math.max(
    0,
    FLAPPY_MEMORY_STACKED_FRAME_COUNT - 1,
  );
  const previousCoreFrames =
    observationMemoryState.previousCoreObservationFrames
      .slice(0, previousFrameTargetCount)
      .map((coreFrame) =>
        coreFrame.length === FLAPPY_MEMORY_CORE_FEATURE_COUNT
          ? coreFrame
          : resolveZeroCoreObservationFrame(),
      );

  while (previousCoreFrames.length < previousFrameTargetCount) {
    previousCoreFrames.push(resolveZeroCoreObservationFrame());
  }

  return previousCoreFrames;
}

/**
 * Builds a zero-valued core frame with canonical length.
 *
 * @returns Zero core frame.
 */
function resolveZeroCoreObservationFrame(): number[] {
  return Array.from({ length: FLAPPY_MEMORY_CORE_FEATURE_COUNT }, () => 0);
}
