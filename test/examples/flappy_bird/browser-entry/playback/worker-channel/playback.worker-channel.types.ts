import type { EvolutionPlaybackStepMessage } from '../../browser-entry.types';

/**
 * Playback-specific worker-channel contracts.
 *
 * These types sit above the lower-level browser worker protocol and describe the
 * request budgeting plus summary data flow used by the playback loop.
 *
 * The key idea is cadence smoothing: the browser renders on animation frames,
 * while the worker advances simulation in step batches. These contracts describe
 * how a fractional render-time budget becomes a concrete worker request.
 *
 * Tiny example:
 * a budget of `2.4` frames this render usually becomes `2` simulation steps now
 * plus `0.4` carried forward to the next render tick.
 */

/**
 * Request payload for one playback-step worker call.
 *
 * The browser asks the worker to advance simulation by a small batch of steps
 * and to package the result for the current viewport dimensions.
 */
export interface PlaybackStepRequest {
  simulationSteps: number;
  visibleWorldWidthPx: number;
  visibleWorldHeightPx: number;
}

/**
 * Input used to resolve the next playback-step request and budget remainder.
 *
 * Playback uses a fractional frame budget so browser render cadence and worker
 * simulation cadence can be smoothed together over time.
 */
export interface ResolvePlaybackStepRequestInput {
  simulationFrameBudget: number;
  visibleWorldWidthPx: number;
  visibleWorldHeightPx: number;
  emulationSpeedMultiplier: number;
}

/**
 * Output for the resolved playback-step request and frame-budget remainder.
 *
 * The resolved request records both the integer step batch to send now and the
 * leftover fractional budget to carry into the next render tick.
 */
export interface ResolvePlaybackStepRequestResult {
  simulationStepsThisRender: number;
  simulationFrameBudgetRemainder: number;
  playbackStepRequest: PlaybackStepRequest;
}

/**
 * Shared alias for the worker playback-step payload.
 *
 * This keeps the playback worker-channel modules focused on playback semantics
 * instead of long imported protocol names.
 */
export type PlaybackStepPayload = EvolutionPlaybackStepMessage['payload'];
