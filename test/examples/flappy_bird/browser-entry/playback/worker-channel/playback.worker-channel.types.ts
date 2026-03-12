import type {
  EvolutionPlaybackStepMessage,
} from '../../browser-entry.types';

/**
 * Request payload for one playback-step worker call.
 */
export interface PlaybackStepRequest {
  simulationSteps: number;
  visibleWorldWidthPx: number;
  visibleWorldHeightPx: number;
}

/**
 * Input used to resolve the next playback-step request and budget remainder.
 */
export interface ResolvePlaybackStepRequestInput {
  simulationFrameBudget: number;
  visibleWorldWidthPx: number;
  visibleWorldHeightPx: number;
  emulationSpeedMultiplier: number;
}

/**
 * Output for the resolved playback-step request and frame-budget remainder.
 */
export interface ResolvePlaybackStepRequestResult {
  simulationStepsThisRender: number;
  simulationFrameBudgetRemainder: number;
  playbackStepRequest: PlaybackStepRequest;
}

/**
 * Shared alias for the worker playback-step payload.
 */
export type PlaybackStepPayload = EvolutionPlaybackStepMessage['payload'];