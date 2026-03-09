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
} from './frame-render/playback.frame-render.service';
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

type PlaybackMutableSummary = PlaybackEpisodeSummary & {
  latestLeaderPipesPassed: number;
  latestLeaderFramesSurvived: number;
};

type PlaybackLoopState = {
  simulationFrameBudget: number;
  finished: boolean;
  summary: PlaybackMutableSummary;
};

type PlaybackSessionContext = {
  renderState: PopulationRenderState;
  trailState: TrailState;
  loopState: PlaybackLoopState;
};

type PlaybackIterationContext = {
  canvas: HTMLCanvasElement;
  context: CanvasRenderingContext2D;
  evolutionWorker: Worker;
  onFrameStats: (stats: PlaybackFrameStats) => void;
  sessionContext: PlaybackSessionContext;
};

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
  // Step 1: Initialize worker playback state and local render mirrors.
  const sessionContext = initializePlaybackSessionContext(
    canvas,
    evolutionWorker,
  );

  // Step 2: Run playback iterations until the worker reports completion.
  await runPlaybackLoop({
    canvas,
    context,
    evolutionWorker,
    onFrameStats,
    sessionContext,
  });

  // Step 3: Render the final settled frame for the completed episode.
  renderPopulationFrame(
    context,
    sessionContext.renderState,
    sessionContext.trailState,
  );

  // Step 4: Fold the mutable loop summary into the public return shape.
  return resolvePlaybackEpisodeSummary(sessionContext.loopState.summary);
}

/**
 * Initializes worker playback and local state mirrors for one episode.
 *
 * @param canvas - Target playback canvas.
 * @param evolutionWorker - Worker owning playback simulation state.
 * @returns Session context shared across the playback loop.
 */
function initializePlaybackSessionContext(
  canvas: HTMLCanvasElement,
  evolutionWorker: Worker,
): PlaybackSessionContext {
  // Step 1: Resolve the current viewport dimensions.
  const viewportDimensions = resolvePlaybackViewportDimensions(canvas);

  // Step 2: Start playback in the worker using the current viewport.
  evolutionWorker.postMessage({
    type: 'start-playback',
    payload: viewportDimensions,
  });

  // Step 3: Build local render, trail, and loop state mirrors.
  return {
    renderState: createInitialRenderState(viewportDimensions),
    trailState: createInitialTrailState(),
    loopState: createInitialPlaybackLoopState(),
  };
}

/**
 * Runs playback iterations until the worker reports that the episode is done.
 *
 * @param iterationContext - Shared loop dependencies and mutable playback state.
 * @returns Nothing.
 */
async function runPlaybackLoop(
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
async function runPlaybackIteration(
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
 * Resolves the current visible playback viewport dimensions from the canvas.
 *
 * @param canvas - Target playback canvas.
 * @returns Visible world width and height in pixels.
 */
function resolvePlaybackViewportDimensions(canvas: HTMLCanvasElement): {
  visibleWorldWidthPx: number;
  visibleWorldHeightPx: number;
} {
  // Step 1: Read the current world-space viewport dimensions.
  return {
    visibleWorldWidthPx: resolveVisibleWorldWidthPx(canvas),
    visibleWorldHeightPx: resolveVisibleWorldHeightPx(canvas),
  };
}

/**
 * Creates the initial render state used before the first worker snapshot.
 *
 * @param viewportDimensions - Current visible world dimensions.
 * @returns Initialized population render state.
 */
function createInitialRenderState(viewportDimensions: {
  visibleWorldWidthPx: number;
  visibleWorldHeightPx: number;
}): PopulationRenderState {
  // Step 1: Seed render fields with viewport dimensions and empty entities.
  return {
    frameIndex: 0,
    visibleWorldWidthPx: viewportDimensions.visibleWorldWidthPx,
    visibleWorldHeightPx: viewportDimensions.visibleWorldHeightPx,
    nextPipeId: 0,
    lastSpawnedPipeGapPx: 0,
    lastSpawnedPipeGapCenterYPx: 0,
    lastSpawnedPipeSpawnIntervalFrames: 0,
    framesUntilNextPipeSpawn: 0,
    pipes: [],
    birds: [],
  };
}

/**
 * Creates the initial trail state used before any snapshots have been applied.
 *
 * @returns Empty trail state for all birds.
 */
function createInitialTrailState(): TrailState {
  // Step 1: Initialize the trail cache as empty arrays.
  return {
    birdTrailsY: [],
  };
}

/**
 * Creates the mutable loop state used while processing playback steps.
 *
 * @returns Initialized loop state and aggregate summary values.
 */
function createInitialPlaybackLoopState(): PlaybackLoopState {
  // Step 1: Initialize frame-budget and completion tracking.
  return {
    simulationFrameBudget: 0,
    finished: false,
    summary: {
      averagePipesPassed: 0,
      p90FramesSurvived: 0,
      winnerPipesPassed: 0,
      winnerFramesSurvived: 0,
      latestLeaderPipesPassed: 0,
      latestLeaderFramesSurvived: 0,
    },
  };
}

/**
 * Synchronizes the render state viewport fields with the current canvas size.
 *
 * @param canvas - Target playback canvas.
 * @param renderState - Mutable render state updated in place.
 * @returns Nothing.
 */
function syncPlaybackViewportDimensions(
  canvas: HTMLCanvasElement,
  renderState: PopulationRenderState,
): void {
  // Step 1: Resolve current viewport dimensions from the canvas.
  const viewportDimensions = resolvePlaybackViewportDimensions(canvas);

  // Step 2: Store the current viewport dimensions on the mutable render state.
  renderState.visibleWorldWidthPx = viewportDimensions.visibleWorldWidthPx;
  renderState.visibleWorldHeightPx = viewportDimensions.visibleWorldHeightPx;
}

/**
 * Requests one playback step batch from the evolution worker.
 *
 * @param iterationContext - Shared loop dependencies and mutable playback state.
 * @returns Worker playback step payload for the current iteration.
 */
async function requestPlaybackStepPayload(
  iterationContext: PlaybackIterationContext,
) {
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
function applyPlaybackStepSnapshot(
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
function emitPlaybackFrameStats(
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
function updatePlaybackLoopCompletion(
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

/**
 * Folds the mutable loop summary into the public playback summary shape.
 *
 * @param summary - Mutable loop summary accumulated during playback.
 * @returns Public playback episode summary.
 */
function resolvePlaybackEpisodeSummary(
  summary: PlaybackMutableSummary,
): PlaybackEpisodeSummary {
  // Step 1: Return only the public aggregate fields.
  return {
    averagePipesPassed: summary.averagePipesPassed,
    p90FramesSurvived: summary.p90FramesSurvived,
    winnerPipesPassed: summary.winnerPipesPassed,
    winnerFramesSurvived: summary.winnerFramesSurvived,
  };
}
