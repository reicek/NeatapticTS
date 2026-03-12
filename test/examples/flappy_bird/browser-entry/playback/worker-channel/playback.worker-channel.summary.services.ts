import type { PlaybackFrameStats } from '../../browser-entry.types';
import type { PlaybackStepPayload } from './playback.worker-channel.types';

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
 * Resolves final playback summary values when the worker reports completion.
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