import type {
  PlaybackFrameStats,
  PopulationRenderState,
  TrailState,
} from '../browser-entry.types';

/**
 * Playback orchestration contracts for the Flappy Bird browser demo.
 *
 * These types describe the moving pieces of one playback episode: the public
 * summary returned at the end, the mutable loop bookkeeping used while frames
 * are streaming, and the session context mirrored locally in the browser.
 */

/**
 * Public aggregate playback summary returned after one episode completes.
 *
 * The summary captures the headline outcomes of the just-finished population
 * run without exposing all internal frame-by-frame details.
 */
export type PlaybackEpisodeSummary = {
  averagePipesPassed: number;
  p90FramesSurvived: number;
  winnerPipesPassed: number;
  winnerFramesSurvived: number;
  winnerNetworkJson?: Record<string, unknown>;
};

/**
 * Event emitted when the current playback champion changes.
 *
 * The event identifies which playback bird is currently highlighted as the red
 * bird so the side-panel network view can stay synchronized with the renderer.
 */
export type PlaybackChampionChangedEvent = {
  championBirdIndex: number;
};

/**
 * Mutable playback summary extended with latest leader telemetry fallbacks.
 *
 * During playback the browser may need temporary "latest known" values before
 * the worker emits final aggregate statistics, so the mutable form carries both
 * final fields and rolling fallbacks.
 */
export type PlaybackMutableSummary = PlaybackEpisodeSummary & {
  latestLeaderPipesPassed: number;
  latestLeaderFramesSurvived: number;
};

/**
 * Mutable loop bookkeeping shared across playback iterations.
 *
 * This is the browser-side state machine for the playback loop: how much
 * simulation budget is being requested, whether the episode has finished, and
 * what aggregate summary has been observed so far.
 */
export type PlaybackLoopState = {
  simulationFrameBudget: number;
  finished: boolean;
  currentChampionBirdIndex: number;
  summary: PlaybackMutableSummary;
};

/**
 * Shared mutable playback state mirrored locally while worker playback runs.
 *
 * The worker remains the source of truth for simulation, but the browser keeps
 * lightweight mirrored state for rendering, trail accumulation, and loop
 * orchestration.
 */
export type PlaybackSessionContext = {
  renderState: PopulationRenderState;
  trailState: TrailState;
  loopState: PlaybackLoopState;
};

/**
 * Shared dependencies and mutable state used by one playback iteration.
 *
 * Grouping these fields into one context object keeps the iteration services
 * declarative and avoids long parameter lists across the playback loop.
 */
export type PlaybackIterationContext = {
  canvas: HTMLCanvasElement;
  context: CanvasRenderingContext2D;
  evolutionWorker: Worker;
  onFrameStats: (stats: PlaybackFrameStats) => void;
  onChampionChanged?: (event: PlaybackChampionChangedEvent) => void;
  sessionContext: PlaybackSessionContext;
};
