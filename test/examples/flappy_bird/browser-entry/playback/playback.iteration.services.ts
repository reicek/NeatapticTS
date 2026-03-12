import {
  resolveAliveBirdCount,
  resolveLeaderPipesPassed,
} from '../browser-entry.observation.utils';
import { requestWorkerPlaybackStep } from '../worker-channel/worker-channel';
import { FLAPPY_EMULATION_SPEED_MULTIPLIER } from '../../constants/constants';
import {
  renderPopulationFrame,
  updateTrailState,
} from './frame-render/playback.frame-render.service';
import { nextAnimationFrame } from './playback.loop.service';
import {
  applyPlaybackSnapshot,
  resolveLeaderFramesSurvived,
} from './playback.snapshot.utils';
import type {
  PlaybackIterationContext,
  PlaybackLoopState,
  PlaybackSessionContext,
} from './playback.orchestration.types';
import { syncPlaybackViewportDimensions } from './playback.session.services';
import {
  resolvePlaybackCompletionSummary,
  resolvePlaybackFrameStats,
  resolvePlaybackStepRequest,
} from './playback.worker-channel.utils';

/**
 * Runs playback iterations until the worker reports that the episode is done.
 *
 * @param iterationContext - Shared loop dependencies and mutable playback state.
 * @returns Nothing.
 */
export async function runPlaybackLoop(
  iterationContext: PlaybackIterationContext,
): Promise<void> {
  // Step 1: Continue iterating until the mutable loop state reports completion.
  while (!iterationContext.sessionContext.loopState.finished) {
    await runPlaybackIteration(iterationContext);
  }
}

/**
 * Executes one playback iteration from viewport sync through render pacing.
 *
 * @param iterationContext - Shared loop dependencies and mutable playback state.
 * @returns Nothing.
 */
export async function runPlaybackIteration(
  iterationContext: PlaybackIterationContext,
): Promise<void> {
  // Step 1: Sync viewport size and request the next worker playback step.
  syncPlaybackViewportDimensions(
    iterationContext.canvas,
    iterationContext.sessionContext.renderState,
  );
  const playbackStepPayload =
    await requestPlaybackStepPayload(iterationContext);

  // Step 2: Fold the worker snapshot into local render and trail state.
  applyPlaybackStepSnapshot(
    iterationContext.sessionContext,
    playbackStepPayload.snapshot,
  );

  // Step 3: Resolve telemetry metrics and emit the frame stats callback.
  emitPlaybackFrameStats(iterationContext, playbackStepPayload);

  // Step 4: Update loop completion state when the worker marks playback done.
  updatePlaybackLoopCompletion(
    iterationContext.sessionContext.loopState,
    playbackStepPayload,
  );

  // Step 5: Render the frame and yield to the browser when playback continues.
  renderPopulationFrame(
    iterationContext.context,
    iterationContext.sessionContext.renderState,
    iterationContext.sessionContext.trailState,
  );
  if (!iterationContext.sessionContext.loopState.finished) {
    await nextAnimationFrame();
  }
}

/**
 * Requests one playback step batch from the evolution worker.
 *
 * @param iterationContext - Shared loop dependencies and mutable playback state.
 * @returns Worker playback step payload for the current iteration.
 */
export async function requestPlaybackStepPayload(
  iterationContext: PlaybackIterationContext,
): Promise<Awaited<ReturnType<typeof requestWorkerPlaybackStep>>> {
  // Step 1: Resolve the worker request from the current loop budget and viewport.
  const { renderState, loopState } = iterationContext.sessionContext;
  const { simulationFrameBudgetRemainder, playbackStepRequest } =
    resolvePlaybackStepRequest({
      simulationFrameBudget: loopState.simulationFrameBudget,
      visibleWorldWidthPx: renderState.visibleWorldWidthPx,
      visibleWorldHeightPx: renderState.visibleWorldHeightPx,
      emulationSpeedMultiplier: FLAPPY_EMULATION_SPEED_MULTIPLIER,
    });
  loopState.simulationFrameBudget = simulationFrameBudgetRemainder;

  // Step 2: Request the next playback step from the worker.
  return requestWorkerPlaybackStep(
    iterationContext.evolutionWorker,
    playbackStepRequest,
  );
}

/**
 * Applies the latest worker snapshot to render state and trail caches.
 *
 * @param sessionContext - Shared mutable playback session state.
 * @param snapshot - Worker snapshot for the current playback batch.
 * @returns Nothing.
 */
export function applyPlaybackStepSnapshot(
  sessionContext: PlaybackSessionContext,
  snapshot: Parameters<typeof applyPlaybackSnapshot>[1],
): void {
  // Step 1: Fold the worker snapshot into the mutable render state.
  applyPlaybackSnapshot(sessionContext.renderState, snapshot);

  // Step 2: Refresh the bird trail cache from the updated render state.
  updateTrailState(sessionContext.trailState, sessionContext.renderState);
}

/**
 * Resolves leader telemetry and emits the public frame-stats callback.
 *
 * @param iterationContext - Shared loop dependencies and mutable playback state.
 * @param playbackStepPayload - Worker playback result for the current iteration.
 * @returns Nothing.
 */
export function emitPlaybackFrameStats(
  iterationContext: PlaybackIterationContext,
  playbackStepPayload: Awaited<ReturnType<typeof requestWorkerPlaybackStep>>,
): void {
  // Step 1: Resolve leader metrics from the updated render state.
  const { renderState, loopState } = iterationContext.sessionContext;
  const leaderPipesPassed = resolveLeaderPipesPassed(renderState.birds);
  const leaderFramesSurvived = resolveLeaderFramesSurvived(renderState);
  loopState.summary.latestLeaderPipesPassed = leaderPipesPassed;
  loopState.summary.latestLeaderFramesSurvived = leaderFramesSurvived;

  // Step 2: Emit frame telemetry for HUD and runtime consumers.
  iterationContext.onFrameStats(
    resolvePlaybackFrameStats(
      playbackStepPayload,
      renderState.frameIndex,
      resolveAliveBirdCount(renderState.birds),
      leaderPipesPassed,
      leaderFramesSurvived,
    ),
  );
}

/**
 * Updates the loop summary when the worker reports playback completion.
 *
 * @param loopState - Mutable playback loop state.
 * @param playbackStepPayload - Worker playback result for the current iteration.
 * @returns Nothing.
 */
export function updatePlaybackLoopCompletion(
  loopState: PlaybackLoopState,
  playbackStepPayload: Awaited<ReturnType<typeof requestWorkerPlaybackStep>>,
): void {
  // Step 1: Skip summary folding until the worker marks playback complete.
  if (!playbackStepPayload.done) {
    return;
  }

  // Step 2: Fold the final aggregate summary into the mutable loop state.
  const playbackCompletionSummary = resolvePlaybackCompletionSummary(
    playbackStepPayload,
    loopState.summary.latestLeaderPipesPassed,
    loopState.summary.latestLeaderFramesSurvived,
  );
  loopState.finished = true;
  loopState.summary.averagePipesPassed =
    playbackCompletionSummary.averagePipesPassed;
  loopState.summary.p90FramesSurvived =
    playbackCompletionSummary.p90FramesSurvived;
  loopState.summary.winnerPipesPassed =
    playbackCompletionSummary.winnerPipesPassed;
  loopState.summary.winnerFramesSurvived =
    playbackCompletionSummary.winnerFramesSurvived;
}
