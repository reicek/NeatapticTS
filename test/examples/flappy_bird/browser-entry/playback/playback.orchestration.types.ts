import type {
  PlaybackFrameStats,
  PopulationRenderState,
  TrailState,
} from '../browser-entry.types';

/**
 * Public aggregate playback summary returned after one episode completes.
 */
export type PlaybackEpisodeSummary = {
  averagePipesPassed: number;
  p90FramesSurvived: number;
  winnerPipesPassed: number;
  winnerFramesSurvived: number;
};

/**
 * Mutable playback summary extended with latest leader telemetry fallbacks.
 */
export type PlaybackMutableSummary = PlaybackEpisodeSummary & {
  latestLeaderPipesPassed: number;
  latestLeaderFramesSurvived: number;
};

/**
 * Mutable loop bookkeeping shared across playback iterations.
 */
export type PlaybackLoopState = {
  simulationFrameBudget: number;
  finished: boolean;
  summary: PlaybackMutableSummary;
};

/**
 * Shared mutable playback state mirrored locally while worker playback runs.
 */
export type PlaybackSessionContext = {
  renderState: PopulationRenderState;
  trailState: TrailState;
  loopState: PlaybackLoopState;
};

/**
 * Shared dependencies and mutable state used by one playback iteration.
 */
export type PlaybackIterationContext = {
  canvas: HTMLCanvasElement;
  context: CanvasRenderingContext2D;
  evolutionWorker: Worker;
  onFrameStats: (stats: PlaybackFrameStats) => void;
  sessionContext: PlaybackSessionContext;
};