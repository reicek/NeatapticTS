import type { PopulationRenderState, TrailState } from '../browser-entry.types';
import {
  FLAPPY_WORLD_HEIGHT_PX,
  FLAPPY_WORLD_WIDTH_PX,
} from '../../constants/constants';
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

const FLAPPY_BROWSER_PLAYBACK_LOG_PREFIX = '[flappy-browser]';
const FLAPPY_PLAYBACK_MIN_READY_VIEWPORT_PX = 2;
const SHOULD_LOG_FLAPPY_PLAYBACK_STARTUP =
  resolveNodeEnvForRuntimeLogs() !== 'test';

/**
 * Session initialization and summary-folding helpers for playback.
 *
 * These services answer three orchestration questions:
 * 1. What viewport is the browser currently showing?
 * 2. What local mirror state should exist before the first worker snapshot?
 * 3. How should mutable loop state be folded back into a public summary?
 */

/**
 * Resolves the current visible playback viewport dimensions from the canvas.
 *
 * Playback normally mirrors the live canvas dimensions, but the browser host
 * intentionally seeds that canvas at `1x1` before the resize hook applies the
 * real backing size. When that placeholder size is still active, the playback
 * worker uses the canonical Flappy world dimensions instead of waiting.
 *
 * @param canvas - Target playback canvas.
 * @returns Visible world width and height in pixels.
 */
export function resolvePlaybackViewportDimensions(canvas: HTMLCanvasElement): {
  visibleWorldWidthPx: number;
  visibleWorldHeightPx: number;
} {
  const rawViewportDimensions = resolveRawPlaybackViewportDimensions(canvas);

  // Step 1: Fall back to the canonical Flappy world while the canvas still has its 1x1 placeholder size.
  if (!isPlaybackViewportReady(rawViewportDimensions)) {
    return resolveFixedPlaybackStartupViewportDimensions();
  }

  // Step 2: Use the live canvas size once layout has applied a real viewport.
  return rawViewportDimensions;
}

/**
 * Creates the initial render state used before the first worker snapshot.
 *
 * The browser starts from an empty-but-shaped render state so rendering helpers
 * can assume the object graph exists even before the worker has emitted any
 * population geometry.
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
 * Trails are purely visual history, so they begin empty and accumulate only as
 * playback frames are observed.
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
 * This is the browser's running notebook for one episode: budget, completion
 * flag, and the latest known aggregate outcome metrics.
 *
 * @returns Initialized loop state and aggregate summary values.
 */
export function createInitialPlaybackLoopState(): PlaybackLoopState {
  // Step 1: Initialize frame-budget and completion tracking.
  return {
    simulationFrameBudget: 0,
    finished: false,
    currentChampionBirdIndex: -1,
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
 * This is the point where the browser and worker agree on a fresh episode. The
 * browser sends the initial viewport dimensions to the worker, then builds the
 * local render and summary mirrors that will be updated as snapshots arrive.
 *
 * @param canvas - Target playback canvas.
 * @param evolutionWorker - Worker owning playback simulation state.
 * @returns Session context shared across the playback loop.
 */
export function initializePlaybackSessionContext(
  canvas: HTMLCanvasElement,
  evolutionWorker: Worker,
): PlaybackSessionContext {
  const rawViewportDimensions = resolveRawPlaybackViewportDimensions(canvas);

  // Step 1: Resolve the playback viewport dimensions for the worker/session boundary.
  const viewportDimensions = resolvePlaybackViewportDimensions(canvas);

  if (
    SHOULD_LOG_FLAPPY_PLAYBACK_STARTUP &&
    !isPlaybackViewportReady(rawViewportDimensions)
  ) {
    console.info(
      `${FLAPPY_BROWSER_PLAYBACK_LOG_PREFIX} using fixed playback startup viewport while canvas settles`,
      {
        rawViewportDimensions,
        startupViewportDimensions: viewportDimensions,
      },
    );
  }

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
 * Playback can continue while the canvas size changes, so the browser refreshes
 * its local viewport mirror rather than assuming dimensions stay fixed.
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
 * The public summary is intentionally smaller than the internal loop state. It
 * exposes the outcome, not the browser's intermediate bookkeeping.
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

function isPlaybackViewportReady(viewportDimensions: {
  visibleWorldWidthPx: number;
  visibleWorldHeightPx: number;
}): boolean {
  return (
    viewportDimensions.visibleWorldWidthPx >=
      FLAPPY_PLAYBACK_MIN_READY_VIEWPORT_PX &&
    viewportDimensions.visibleWorldHeightPx >=
      FLAPPY_PLAYBACK_MIN_READY_VIEWPORT_PX
  );
}

function resolveFixedPlaybackStartupViewportDimensions(): {
  visibleWorldWidthPx: number;
  visibleWorldHeightPx: number;
} {
  return {
    visibleWorldWidthPx: FLAPPY_WORLD_WIDTH_PX,
    visibleWorldHeightPx: FLAPPY_WORLD_HEIGHT_PX,
  };
}

function resolveRawPlaybackViewportDimensions(canvas: HTMLCanvasElement): {
  visibleWorldWidthPx: number;
  visibleWorldHeightPx: number;
} {
  return {
    visibleWorldWidthPx: resolveVisibleWorldWidthPx(canvas),
    visibleWorldHeightPx: resolveVisibleWorldHeightPx(canvas),
  };
}

function resolveNodeEnvForRuntimeLogs(): string | undefined {
  return (globalThis as { process?: { env?: { NODE_ENV?: string } } }).process
    ?.env?.NODE_ENV;
}
