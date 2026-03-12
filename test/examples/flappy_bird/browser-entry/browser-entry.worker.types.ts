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
  yPx: number;
  pipesPassed: number;
  framesSurvived: number;
  done: boolean;
}

/** Packed typed-array payload for playback pipe snapshot transport. */
export interface PackedPlaybackPipeSnapshot {
  xPositionsPx: Float32Array;
  gapCenterYPositionsPx: Float32Array;
  gapSizesPx: Float32Array;
}

/** Packed typed-array payload for playback bird snapshot transport. */
export interface PackedPlaybackBirdSnapshot {
  yPositionsPx: Float32Array;
  pipesPassed: Uint32Array;
  framesSurvived: Uint32Array;
  doneFlags: Uint8Array;
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
  format: 'packed-v1';
  frameIndex: number;
  cumulativePipeTravelPx: number;
  visibleWorldWidthPx: number;
  visibleWorldHeightPx: number;
  pipeCount: number;
  birdCount: number;
  pipes: PackedPlaybackPipeSnapshot;
  birds: PackedPlaybackBirdSnapshot;
}

/** Worker message carrying one playback step and aggregate markers. */
export interface EvolutionPlaybackStepMessage {
  type: 'playback-step';
  payload: {
    requestId: number;
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
  activeBirdCount: number;
  leaderPipesPassed: number;
  leaderFramesSurvived: number;
  activationCallsPerFrame: number;
  simulationStepsPerRaf: number;
}
