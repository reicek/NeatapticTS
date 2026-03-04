import type {
  EvolutionPlaybackStepMessage,
  PlaybackFrameStats,
} from '../browser-entry.types';

/**
 * Request payload for one playback-step worker call.
 */
export interface PlaybackStepRequest {
  simulationSteps: number;
  visibleWorldWidthPx: number;
  visibleWorldHeightPx: number;
}

/**
 * Input used to resolve next playback-step request and budget remainder.
 */
export interface ResolvePlaybackStepRequestInput {
  simulationFrameBudget: number;
  visibleWorldWidthPx: number;
  visibleWorldHeightPx: number;
  emulationSpeedMultiplier: number;
}

/**
 * Output for resolved playback-step request and frame-budget remainder.
 */
export interface ResolvePlaybackStepRequestResult {
  simulationStepsThisRender: number;
  simulationFrameBudgetRemainder: number;
  playbackStepRequest: PlaybackStepRequest;
}

/**
 * Shared alias for worker playback-step payload.
 */
export type PlaybackStepPayload = EvolutionPlaybackStepMessage['payload'];

/**
 * Resolves step count and request payload for the next worker playback batch.
 *
 * @param input - Current frame budget and viewport dimensions.
 * @returns Request payload plus carried-over fractional frame budget.
 */
export function resolvePlaybackStepRequest(
  input: ResolvePlaybackStepRequestInput,
): ResolvePlaybackStepRequestResult {
  const updatedSimulationFrameBudget =
    input.simulationFrameBudget + input.emulationSpeedMultiplier;
  const simulationStepsThisRender = Math.max(
    1,
    Math.floor(updatedSimulationFrameBudget),
  );
  const simulationFrameBudgetRemainder =
    updatedSimulationFrameBudget - simulationStepsThisRender;

  return {
    simulationStepsThisRender,
    simulationFrameBudgetRemainder,
    playbackStepRequest: {
      simulationSteps: simulationStepsThisRender,
      visibleWorldWidthPx: input.visibleWorldWidthPx,
      visibleWorldHeightPx: input.visibleWorldHeightPx,
    },
  };
}

/**
 * Resolves HUD playback frame stats from worker payload and leader metrics.
 *
 * @param playbackStepPayload - Playback payload returned by worker.
 * @param frameIndex - Current render frame index.
 * @param activeBirdCount - Number of alive birds in current frame.
 * @param leaderPipesPassed - Current frame leader pipes passed.
 * @param leaderFramesSurvived - Current frame leader survived frames.
 * @returns Normalized per-frame HUD telemetry payload.
 */
export function resolvePlaybackFrameStats(
  playbackStepPayload: PlaybackStepPayload,
  frameIndex: number,
  activeBirdCount: number,
  leaderPipesPassed: number,
  leaderFramesSurvived: number,
): PlaybackFrameStats {
  return {
    frameIndex,
    activeBirdCount,
    leaderPipesPassed,
    leaderFramesSurvived,
    activationCallsPerFrame:
      playbackStepPayload.instrumentation?.activationCallsPerFrame ?? 0,
    simulationStepsPerRaf:
      playbackStepPayload.instrumentation?.simulationStepsPerRaf ?? 0,
  };
}

/**
 * Resolves final playback summary values when worker reports completion.
 *
 * @param playbackStepPayload - Playback payload returned by worker.
 * @param latestLeaderPipesPassed - Last observed leader pipes passed fallback.
 * @param latestLeaderFramesSurvived - Last observed leader frames fallback.
 * @returns Final aggregate playback summary.
 */
export function resolvePlaybackCompletionSummary(
  playbackStepPayload: PlaybackStepPayload,
  latestLeaderPipesPassed: number,
  latestLeaderFramesSurvived: number,
): {
  averagePipesPassed: number;
  p90FramesSurvived: number;
  winnerPipesPassed: number;
  winnerFramesSurvived: number;
} {
  return {
    averagePipesPassed: playbackStepPayload.averagePipesPassed ?? 0,
    p90FramesSurvived: playbackStepPayload.p90FramesSurvived ?? 0,
    winnerPipesPassed:
      playbackStepPayload.winnerPipesPassed ?? latestLeaderPipesPassed,
    winnerFramesSurvived:
      playbackStepPayload.winnerFramesSurvived ?? latestLeaderFramesSurvived,
  };
}
