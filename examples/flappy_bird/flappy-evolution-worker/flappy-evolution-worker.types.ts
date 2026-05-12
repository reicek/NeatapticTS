import type Network from '../../../src/architecture/network';
import type {
  InferenceChannel,
  TransferableInferencePayload,
} from '../../../src/neataptic';
import type {
  SharedObservationFeatures,
  SharedObservationMemoryState,
} from '../flappy.simulation.shared.utils';
import type { ExampleArchitectureProfileId } from '../../architectureProfiles';

/**
 * Loose JSON-compatible network payload used by worker messages.
 *
 * The worker never posts live `Network` instances back to the browser host.
 * The transferable inference payload now owns playback-friendly transport,
 * while this JSON bridge remains only for the browser-side network-view cache.
 */
export type SerializedNetwork = Record<string, unknown>;

/**
 * Mutable pipe state tracked by the worker playback simulation.
 *
 * These objects exist only inside the worker runtime. The host later receives a
 * packed snapshot derived from them rather than these live mutable records.
 */
export interface WorkerPopulationPipe {
  id: number;
  xPx: number;
  gapCenterYPx: number;
  gapSizePx: number;
}

/**
 * Mutable bird state tracked by the worker playback simulation.
 *
 * Educational note:
 * Each bird keeps both physics state and policy state. The
 * `observationMemoryState` field stays on the bird so worker playback shares
 * the same control-state shape as evaluation and browser helpers. The current
 * controller input does not read external history, but the aligned state shelf
 * keeps future opt-in experiments from forking the runtime contracts.
 */
export interface WorkerPopulationBird {
  inferenceChannel?: InferenceChannel;
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

/**
 * Mutable simulation state stored between worker playback requests.
 *
 * A `start-playback` message creates this state once, and each
 * `request-playback-step` message advances it by a host-selected number of
 * simulation steps.
 */
export interface WorkerPlaybackState {
  frameIndex: number;
  cumulativePipeTravelPx: number;
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

/**
 * Render-only bird snapshot DTO posted to host.
 *
 * This shape is useful conceptually, but the current transport uses the packed
 * typed-array variant for lower allocation and transfer cost.
 */
export interface WorkerFrameBirdSnapshot {
  yPx: number;
  pipesPassed: number;
  framesSurvived: number;
  done: boolean;
}

/**
 * Render-only pipe snapshot DTO posted to host.
 *
 * Like `WorkerFrameBirdSnapshot`, this documents the logical payload shape even
 * though the worker currently sends the packed transport form.
 */
export interface WorkerFramePipeSnapshot {
  id: number;
  xPx: number;
  gapCenterYPx: number;
  gapSizePx: number;
}

/**
 * Packed typed-array payload for playback pipe snapshot transport.
 *
 * Packing the per-pipe fields into column-oriented typed arrays makes the
 * browser/worker boundary cheaper than sending large arrays of object literals
 * on every animation frame.
 */
export interface WorkerPackedPlaybackPipeSnapshot {
  xPositionsPx: Float32Array;
  gapCenterYPositionsPx: Float32Array;
  gapSizesPx: Float32Array;
}

/**
 * Packed typed-array payload for playback bird snapshot transport.
 *
 * The host can reconstruct renderer-friendly bird views from these arrays while
 * the worker keeps the authoritative mutable simulation objects private.
 */
export interface WorkerPackedPlaybackBirdSnapshot {
  yPositionsPx: Float32Array;
  pipesPassed: Uint32Array;
  framesSurvived: Uint32Array;
  doneFlags: Uint8Array;
}

/**
 * Full frame snapshot payload posted to host.
 *
 * Educational note:
 * `packed-v1` is a transport contract, not a rendering primitive. The versioned
 * format string gives the browser host a stable way to decode snapshots even if
 * the worker later gains additional packed fields or alternate transport modes.
 *
 * @example
 * ```ts
 * const message = {
 *   type: 'playback-step',
 *   payload: {
 *     requestId: 7,
 *     snapshot,
 *     done: false,
 *   },
 * };
 * ```
 */
export interface WorkerPlaybackFrameSnapshot {
  format: 'packed-v1';
  frameIndex: number;
  cumulativePipeTravelPx: number;
  visibleWorldWidthPx: number;
  visibleWorldHeightPx: number;
  pipeCount: number;
  birdCount: number;
  pipes: WorkerPackedPlaybackPipeSnapshot;
  birds: WorkerPackedPlaybackBirdSnapshot;
}

/**
 * Worker init request message.
 *
 * This is the first message the host should send. It seeds deterministic RNG
 * state and configures the worker-local NEAT runtime.
 */
export interface WorkerInitMessage {
  type: 'init';
  payload: {
    architectureProfileId?: ExampleArchitectureProfileId;
    championNetworkJson?: SerializedNetwork;
    populationSize: number;
    elitismCount: number;
    rngSeed: number;
  };
}

/**
 * Worker request asking for the next playable generation payload.
 *
 * The first response may release the bounded generation-zero warm-start
 * population before a full recurrent evolution pass. Later responses publish
 * normally evolved populations.
 */
export interface WorkerRequestGenerationMessage {
  type: 'request-generation';
}

/**
 * Worker request asking to initialize playback state.
 *
 * This materializes the mutable world state for the current evolved population.
 * After this message succeeds, the host can begin issuing playback-step
 * requests.
 */
export interface WorkerStartPlaybackMessage {
  type: 'start-playback';
  payload: {
    visibleWorldWidthPx: number;
    visibleWorldHeightPx: number;
  };
}

/**
 * Worker request asking to advance playback by N simulation steps.
 *
 * The host typically sends this once per animation frame and chooses
 * `simulationSteps` based on how much simulation throughput it wants relative to
 * rendering smoothness.
 */
export interface WorkerRequestPlaybackStepMessage {
  type: 'request-playback-step';
  payload: {
    requestId: number;
    simulationSteps: number;
    visibleWorldWidthPx: number;
    visibleWorldHeightPx: number;
  };
}

/**
 * Worker stop request message.
 *
 * This is a cooperative shutdown signal. Long-running worker flows can observe
 * the stopped flag and fail fast instead of continuing work the UI no longer
 * cares about.
 */
export interface WorkerStopMessage {
  type: 'stop';
}

/**
 * Union of inbound worker request messages.
 *
 * Reading this union top-to-bottom is the quickest way to understand the worker
 * protocol: initialize, evolve, start playback, step playback, then stop.
 */
export type WorkerRequestMessage =
  | WorkerInitMessage
  | WorkerRequestGenerationMessage
  | WorkerStartPlaybackMessage
  | WorkerRequestPlaybackStepMessage
  | WorkerStopMessage;

/**
 * Worker generation-ready response message.
 *
 * The browser host uses this message to refresh HUD state, render the current
 * best network visualization, and start playback for the current population.
 * Generation zero can be a startup release after warm-start rather than a full
 * post-selection NEAT generation.
 */
export interface WorkerGenerationReadyMessage {
  type: 'generation-ready';
  payload: {
    architectureProfileId: ExampleArchitectureProfileId;
    generation: number;
    bestFitness: number;
    bestNetworkPayload?: TransferableInferencePayload;
    bestNetworkJson?: SerializedNetwork;
    populationNetworkPayloads?: TransferableInferencePayload[];
    populationNetworksJson?: SerializedNetwork[];
  };
}

/**
 * Worker playback-step response message.
 *
 * The message carries the packed frame snapshot plus optional instrumentation
 * and end-of-run summary statistics when the whole simulated population has
 * been eliminated.
 *
 * The split between per-frame snapshot data and end-of-run summary fields keeps
 * the hot path compact while still giving the host enough telemetry to update
 * HUD metrics when a playback session completes.
 */
export interface WorkerPlaybackStepMessage {
  type: 'playback-step';
  payload: {
    requestId: number;
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

/**
 * Worker error response message.
 *
 * Errors are normalized into a display-safe string so the host UI can surface
 * failures without depending on worker-specific exception classes.
 */
export interface WorkerErrorMessage {
  type: 'error';
  payload: {
    message: string;
  };
}

/**
 * Union of outbound worker response messages.
 *
 * Together with `WorkerRequestMessage`, this forms the full host/worker
 * protocol contract for the demo.
 */
export type WorkerResponseMessage =
  | WorkerGenerationReadyMessage
  | WorkerPlaybackStepMessage
  | WorkerErrorMessage;

/**
 * Structured features used by heuristic generation-0 teacher policy.
 *
 * The warm-start service reuses the same high-level observation semantics as the
 * real policy inference path, which keeps the heuristic teacher aligned with the
 * features evolved networks will later see.
 */
export type WorkerHeuristicObservationFeatures = SharedObservationFeatures;
