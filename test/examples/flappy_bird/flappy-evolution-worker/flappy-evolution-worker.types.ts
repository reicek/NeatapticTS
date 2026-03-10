import type Network from '../../../../src/architecture/network';
import type {
  SharedObservationFeatures,
  SharedObservationMemoryState,
} from '../flappy.simulation.shared.utils';

/** Loose JSON-compatible network payload used by worker messages. */
export type SerializedNetwork = Record<string, unknown>;

/** Mutable pipe state tracked by the worker playback simulation. */
export interface WorkerPopulationPipe {
  id: number;
  xPx: number;
  gapCenterYPx: number;
  gapSizePx: number;
}

/** Mutable bird state tracked by the worker playback simulation. */
export interface WorkerPopulationBird {
  network: Network;
  observationMemoryState: SharedObservationMemoryState;
  yPx: number;
  velocityYPxPerFrame: number;
  pipesPassed: number;
  framesSurvived: number;
  passedPipeIds: Set<number>;
  done: boolean;
  doneReason?: 'collision' | 'out_of_bounds';
}

/** Mutable simulation state stored between worker playback requests. */
export interface WorkerPlaybackState {
  frameIndex: number;
  visibleWorldWidthPx: number;
  visibleWorldHeightPx: number;
  nextPipeId: number;
  lastSpawnedPipeGapPx: number;
  lastSpawnedPipeGapCenterYPx: number;
  lastSpawnedPipeSpawnIntervalFrames: number;
  framesUntilNextPipeSpawn: number;
  pipes: WorkerPopulationPipe[];
  birds: WorkerPopulationBird[];
}

/** Render-only bird snapshot DTO posted to host. */
export interface WorkerFrameBirdSnapshot {
  yPx: number;
  pipesPassed: number;
  framesSurvived: number;
  done: boolean;
}

/** Render-only pipe snapshot DTO posted to host. */
export interface WorkerFramePipeSnapshot {
  id: number;
  xPx: number;
  gapCenterYPx: number;
  gapSizePx: number;
}

/** Packed typed-array payload for playback pipe snapshot transport. */
export interface WorkerPackedPlaybackPipeSnapshot {
  xPositionsPx: Float32Array;
  gapCenterYPositionsPx: Float32Array;
  gapSizesPx: Float32Array;
}

/** Packed typed-array payload for playback bird snapshot transport. */
export interface WorkerPackedPlaybackBirdSnapshot {
  yPositionsPx: Float32Array;
  pipesPassed: Uint32Array;
  framesSurvived: Uint32Array;
  doneFlags: Uint8Array;
}

/** Full frame snapshot payload posted to host. */
export interface WorkerPlaybackFrameSnapshot {
  format: 'packed-v1';
  frameIndex: number;
  visibleWorldWidthPx: number;
  visibleWorldHeightPx: number;
  pipeCount: number;
  birdCount: number;
  pipes: WorkerPackedPlaybackPipeSnapshot;
  birds: WorkerPackedPlaybackBirdSnapshot;
}

/** Worker init request message. */
export interface WorkerInitMessage {
  type: 'init';
  payload: {
    populationSize: number;
    elitismCount: number;
    rngSeed: number;
  };
}

/** Worker request asking to evolve one generation. */
export interface WorkerRequestGenerationMessage {
  type: 'request-generation';
}

/** Worker request asking to initialize playback state. */
export interface WorkerStartPlaybackMessage {
  type: 'start-playback';
  payload: {
    visibleWorldWidthPx: number;
    visibleWorldHeightPx: number;
  };
}

/** Worker request asking to advance playback by N simulation steps. */
export interface WorkerRequestPlaybackStepMessage {
  type: 'request-playback-step';
  payload: {
    simulationSteps: number;
    visibleWorldWidthPx: number;
    visibleWorldHeightPx: number;
  };
}

/** Worker stop request message. */
export interface WorkerStopMessage {
  type: 'stop';
}

/** Union of inbound worker request messages. */
export type WorkerRequestMessage =
  | WorkerInitMessage
  | WorkerRequestGenerationMessage
  | WorkerStartPlaybackMessage
  | WorkerRequestPlaybackStepMessage
  | WorkerStopMessage;

/** Worker generation-ready response message. */
export interface WorkerGenerationReadyMessage {
  type: 'generation-ready';
  payload: {
    generation: number;
    bestFitness: number;
    bestNetworkJson?: SerializedNetwork;
  };
}

/** Worker playback-step response message. */
export interface WorkerPlaybackStepMessage {
  type: 'playback-step';
  payload: {
    snapshot: WorkerPlaybackFrameSnapshot;
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

/** Worker error response message. */
export interface WorkerErrorMessage {
  type: 'error';
  payload: {
    message: string;
  };
}

/** Union of outbound worker response messages. */
export type WorkerResponseMessage =
  | WorkerGenerationReadyMessage
  | WorkerPlaybackStepMessage
  | WorkerErrorMessage;

/** Structured features used by heuristic generation-0 teacher policy. */
export type WorkerHeuristicObservationFeatures = SharedObservationFeatures;
