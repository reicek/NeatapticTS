/** Loose JSON-compatible network payload used by worker messages. */
export type SerializedNetwork = Record<string, unknown>;

/** Renderable pipe state snapshot emitted by the playback worker. */
export interface PopulationPipe {
  id: number;
  xPx: number;
  gapCenterYPx: number;
  gapSizePx: number;
}

/** Renderable bird state snapshot emitted by the playback worker. */
export interface PopulationBird {
  color: string;
  yPx: number;
  pipesPassed: number;
  framesSurvived: number;
  done: boolean;
}

/** Worker payload describing evolved generation summary values. */
export interface EvolutionGenerationPayload {
  generation: number;
  bestFitness: number;
  bestNetworkJson?: SerializedNetwork;
}

/** Worker message emitted when a generation has completed evolving. */
export interface EvolutionGenerationReadyMessage {
  type: 'generation-ready';
  payload: EvolutionGenerationPayload;
}

/** Worker message emitted for simulation/playback errors. */
export interface EvolutionWorkerErrorMessage {
  type: 'error';
  payload: {
    message: string;
  };
}

/** Per-frame snapshot received from the worker playback channel. */
export interface EvolutionPlaybackStepSnapshot {
  frameIndex: number;
  visibleWorldWidthPx: number;
  pipes: PopulationPipe[];
  birds: PopulationBird[];
}

/** Worker message carrying one playback step and aggregate markers. */
export interface EvolutionPlaybackStepMessage {
  type: 'playback-step';
  payload: {
    snapshot: EvolutionPlaybackStepSnapshot;
    instrumentation?: {
      activationCallsPerFrame: number;
      simulationStepsPerRaf: number;
    };
    done: boolean;
    averagePipesPassed?: number;
    p90FramesSurvived?: number;
    winnerPipesPassed?: number;
    winnerFramesSurvived?: number;
  };
}

/** Union of all supported worker messages consumed by browser entry. */
export type EvolutionWorkerMessage =
  | EvolutionGenerationReadyMessage
  | EvolutionPlaybackStepMessage
  | EvolutionWorkerErrorMessage;

/** Lightweight per-frame telemetry emitted to HUD update callback. */
export interface PlaybackFrameStats {
  frameIndex: number;
  leaderPipesPassed: number;
  leaderFramesSurvived: number;
  activationCallsPerFrame: number;
  simulationStepsPerRaf: number;
}
