import {
  FLAPPY_MEMORY_ACTION_WINDOW_STEPS,
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
 * @example
 * ```ts
 * const memoryState = createSharedObservationMemoryState();
 * ```
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
 * Builds the controller input vector for one decision step.
 *
 * Educational note:
 * This helper keeps the public observation API stable while making the
 * effective controller input just the current normalized frame. That removes
 * hand-authored memory from all architectures so recurrent profiles must learn
 * temporal state internally instead of receiving it as extra inputs.
 *
 * @param features - Structured observation features for the current decision step.
 * @param observationMemoryState - Mutable temporal memory for the active bird.
 * @returns Ordered controller input vector for policy activation.
 */
export function resolveTemporalObservationVector(
  features: SharedObservationFeatures,
  _observationMemoryState: SharedObservationMemoryState,
): number[] {
  return resolveCoreObservationVectorFromFeatures(features);
}

/**
 * Commits one observation-action step into temporal memory.
 *
 * The memory update happens after the decision is made so the next step can see
 * both the recent observation context and the action history that produced the
 * current trajectory.
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
  const previousFrameCapacity = Math.max(
    0,
    FLAPPY_MEMORY_STACKED_FRAME_COUNT - 1,
  );
  const actionWindowCapacity = Math.max(0, FLAPPY_MEMORY_ACTION_WINDOW_STEPS);

  if (previousFrameCapacity === 0 && actionWindowCapacity === 0) {
    return;
  }

  // Step 1: Persist current core frame for future stacked observations.
  const currentCoreObservationFrame =
    resolveCoreObservationVectorFromFeatures(features);
  observationMemoryState.previousCoreObservationFrames = [
    currentCoreObservationFrame,
    ...observationMemoryState.previousCoreObservationFrames,
  ].slice(0, previousFrameCapacity);

  // Step 2: Persist latest action into a fixed-size action history window.
  const actionScalar = didFlap ? 1 : 0;
  observationMemoryState.recentFlapActions = [
    actionScalar,
    ...observationMemoryState.recentFlapActions,
  ].slice(0, actionWindowCapacity);
}
