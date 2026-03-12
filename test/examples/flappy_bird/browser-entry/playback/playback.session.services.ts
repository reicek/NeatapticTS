import type {
  PopulationRenderState,
  TrailState,
} from '../browser-entry.types';
import {
  resolveVisibleWorldHeightPx,
  resolveVisibleWorldWidthPx,
} from '../browser-entry.viewport.utils';
import type {
  PlaybackEpisodeSummary,
  PlaybackLoopState,
  PlaybackMutableSummary,
  PlaybackSessionContext,
} from './playback.orchestration.types';

/**
 * Resolves the current visible playback viewport dimensions from the canvas.
 *
 * @param canvas - Target playback canvas.
 * @returns Visible world width and height in pixels.
 */
export function resolvePlaybackViewportDimensions(canvas: HTMLCanvasElement): {
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
export function createInitialRenderState(viewportDimensions: {
  visibleWorldWidthPx: number;
  visibleWorldHeightPx: number;
}): PopulationRenderState {
  // Step 1: Seed render fields with viewport dimensions and empty entities.
  return {
    frameIndex: 0,
    cumulativePipeTravelPx: 0,
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
export function createInitialTrailState(): TrailState {
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
export function createInitialPlaybackLoopState(): PlaybackLoopState {
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
 * Initializes worker playback and local state mirrors for one episode.
 *
 * @param canvas - Target playback canvas.
 * @param evolutionWorker - Worker owning playback simulation state.
 * @returns Session context shared across the playback loop.
 */
export function initializePlaybackSessionContext(
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
 * Synchronizes the render state viewport fields with the current canvas size.
 *
 * @param canvas - Target playback canvas.
 * @param renderState - Mutable render state updated in place.
 * @returns Nothing.
 */
export function syncPlaybackViewportDimensions(
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
 * Folds the mutable loop summary into the public playback summary shape.
 *
 * @param summary - Mutable loop summary accumulated during playback.
 * @returns Public playback episode summary.
 */
export function resolvePlaybackEpisodeSummary(
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