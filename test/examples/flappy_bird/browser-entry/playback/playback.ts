import {
  resolveAliveBirdCount,
  resolveLeaderPipesPassed,
} from '../browser-entry.observation.utils';
import type {
  PlaybackFrameStats,
  PopulationRenderState,
  TrailState,
} from '../browser-entry.types';
import {
  resolveVisibleWorldHeightPx,
  resolveVisibleWorldWidthPx,
} from '../browser-entry.viewport.utils';
import { requestWorkerPlaybackStep } from '../worker-channel/worker-channel';
import { FLAPPY_EMULATION_SPEED_MULTIPLIER } from '../../constants/constants';
import {
  renderPopulationFrame,
  updateTrailState,
} from './playback.frame-render.service';
import { nextAnimationFrame } from './playback.loop.service';
import {
  applyPlaybackSnapshot,
  resolveLeaderFramesSurvived,
} from './playback.snapshot.utils';
import {
  resolvePlaybackCompletionSummary,
  resolvePlaybackFrameStats,
  resolvePlaybackStepRequest,
} from './playback.worker-channel.utils';

export type PlaybackEpisodeSummary = {
  averagePipesPassed: number;
  p90FramesSurvived: number;
  winnerPipesPassed: number;
  winnerFramesSurvived: number;
};

/**
 * Public playback entry point used by browser runtime orchestration.
 *
 * @param canvas - Target playback canvas.
 * @param context - Canvas 2D context.
 * @param evolutionWorker - Worker owning playback simulation state.
 * @param onFrameStats - Callback receiving per-frame playback telemetry.
 * @returns Aggregate playback summary for the current episode.
 */
export async function animatePopulationEpisode(
  canvas: HTMLCanvasElement,
  context: CanvasRenderingContext2D,
  evolutionWorker: Worker,
  onFrameStats: (stats: PlaybackFrameStats) => void,
): Promise<PlaybackEpisodeSummary> {
  return animatePopulationEpisodeInternal(
    canvas,
    context,
    evolutionWorker,
    onFrameStats,
  );
}

/**
 * Internal playback orchestration entry retained for compatibility re-exports.
 *
 * @param canvas - Target playback canvas.
 * @param context - Canvas 2D context.
 * @param evolutionWorker - Worker owning playback simulation state.
 * @param onFrameStats - Callback receiving per-frame playback telemetry.
 * @returns Aggregate playback summary for the current episode.
 */
export async function animatePopulationEpisodeInternal(
  canvas: HTMLCanvasElement,
  context: CanvasRenderingContext2D,
  evolutionWorker: Worker,
  onFrameStats: (stats: PlaybackFrameStats) => void,
): Promise<PlaybackEpisodeSummary> {
  // Step 1: Initialize worker playback state with current viewport size.
  evolutionWorker.postMessage({
    type: 'start-playback',
    payload: {
      visibleWorldWidthPx: resolveVisibleWorldWidthPx(canvas),
      visibleWorldHeightPx: resolveVisibleWorldHeightPx(canvas),
    },
  });

  // Step 2: Initialize local render and trail state mirrors.
  const renderState: PopulationRenderState = {
    frameIndex: 0,
    visibleWorldWidthPx: resolveVisibleWorldWidthPx(canvas),
    visibleWorldHeightPx: resolveVisibleWorldHeightPx(canvas),
    nextPipeId: 0,
    lastSpawnedPipeGapPx: 0,
    lastSpawnedPipeGapCenterYPx: 0,
    lastSpawnedPipeSpawnIntervalFrames: 0,
    framesUntilNextPipeSpawn: 0,
    pipes: [],
    birds: [],
  };
  const trailState: TrailState = {
    birdTrailsY: [],
  };
  let simulationFrameBudget = 0;
  let finished = false;
  let averagePipesPassed = 0;
  let p90FramesSurvived = 0;
  let winnerPipesPassed = 0;
  let winnerFramesSurvived = 0;
  let latestLeaderPipesPassed = 0;
  let latestLeaderFramesSurvived = 0;

  // Step 3: Run playback batches until the worker reports completion.
  while (!finished) {
    // Step 3.1: Resolve frame budget and request one playback step batch.
    renderState.visibleWorldWidthPx = resolveVisibleWorldWidthPx(canvas);
    renderState.visibleWorldHeightPx = resolveVisibleWorldHeightPx(canvas);
    const { simulationFrameBudgetRemainder, playbackStepRequest } =
      resolvePlaybackStepRequest({
        simulationFrameBudget,
        visibleWorldWidthPx: renderState.visibleWorldWidthPx,
        visibleWorldHeightPx: renderState.visibleWorldHeightPx,
        emulationSpeedMultiplier: FLAPPY_EMULATION_SPEED_MULTIPLIER,
      });
    simulationFrameBudget = simulationFrameBudgetRemainder;

    const playbackStepPayload = await requestWorkerPlaybackStep(
      evolutionWorker,
      playbackStepRequest,
    );

    // Step 3.2: Apply worker snapshot and refresh local trail state.
    applyPlaybackSnapshot(renderState, playbackStepPayload.snapshot);
    updateTrailState(trailState, renderState);

    // Step 3.3: Resolve leader metrics and emit frame telemetry.
    const leaderPipesPassed = resolveLeaderPipesPassed(renderState.birds);
    const leaderFramesSurvived = resolveLeaderFramesSurvived(renderState);
    latestLeaderPipesPassed = leaderPipesPassed;
    latestLeaderFramesSurvived = leaderFramesSurvived;
    onFrameStats(
      resolvePlaybackFrameStats(
        playbackStepPayload,
        renderState.frameIndex,
        resolveAliveBirdCount(renderState.birds),
        leaderPipesPassed,
        leaderFramesSurvived,
      ),
    );

    // Step 3.4: Fold final playback aggregates when the worker is done.
    if (playbackStepPayload.done) {
      const playbackCompletionSummary = resolvePlaybackCompletionSummary(
        playbackStepPayload,
        latestLeaderPipesPassed,
        latestLeaderFramesSurvived,
      );
      finished = true;
      averagePipesPassed = playbackCompletionSummary.averagePipesPassed;
      p90FramesSurvived = playbackCompletionSummary.p90FramesSurvived;
      winnerPipesPassed = playbackCompletionSummary.winnerPipesPassed;
      winnerFramesSurvived = playbackCompletionSummary.winnerFramesSurvived;
    }

    // Step 3.5: Render the current frame and yield to RAF while active.
    renderPopulationFrame(context, renderState, trailState);
    if (!finished) {
      await nextAnimationFrame();
    }
  }

  // Step 4: Render the final settled frame and return aggregate summary.
  renderPopulationFrame(context, renderState, trailState);

  return {
    averagePipesPassed,
    p90FramesSurvived,
    winnerPipesPassed,
    winnerFramesSurvived,
  };
}
