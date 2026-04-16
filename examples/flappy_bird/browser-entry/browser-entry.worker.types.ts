import type { ExampleArchitectureProfileId } from '../../architectureProfiles';

/**
 * Worker transport contracts for the Flappy Bird browser runtime.
 *
 * The browser UI and the evolution worker communicate through a deliberately
 * explicit message protocol. The goal is educational as well as practical: it
 * makes it obvious which values are computed off-thread, which snapshots are
 * transferred frame-by-frame, and which events advance the demo state.
 *
 * If you want background reading, the Wikipedia article on "message passing"
 * provides a useful conceptual frame for this boundary.
 */

/** Loose JSON-compatible network payload used by worker messages. */
export type SerializedNetwork = Record<string, unknown>;

/**
 * Renderable pipe state snapshot emitted by the playback worker.
 *
 * This is the smallest pipe shape the browser renderer needs for one frame:
 * horizontal position plus the vertical corridor geometry.
 */
export interface PopulationPipe {
  id: number;
  xPx: number;
  gapCenterYPx: number;
  gapSizePx: number;
}

/**
 * Renderable bird state snapshot emitted by the playback worker.
 *
 * The browser does not receive full neural state here. It only gets the fields
 * needed for presentation and HUD summaries, which keeps per-frame transport
 * light.
 */
export interface PopulationBird {
  yPx: number;
  pipesPassed: number;
  framesSurvived: number;
  done: boolean;
}

/**
 * Packed typed-array payload for playback pipe snapshot transport.
 *
 * Typed arrays keep frame payloads compact and predictable, which matters when
 * the worker is streaming many birds and pipes across animation frames.
 */
export interface PackedPlaybackPipeSnapshot {
  xPositionsPx: Float32Array;
  gapCenterYPositionsPx: Float32Array;
  gapSizesPx: Float32Array;
}

/**
 * Packed typed-array payload for playback bird snapshot transport.
 *
 * This mirrors the pipe packing strategy so playback can move large population
 * snapshots with less allocation pressure than object-per-bird messages.
 */
export interface PackedPlaybackBirdSnapshot {
  yPositionsPx: Float32Array;
  pipesPassed: Uint32Array;
  framesSurvived: Uint32Array;
  doneFlags: Uint8Array;
}

/**
 * Worker payload describing evolved generation summary values.
 *
 * This is the browser-facing summary of one completed NEAT generation: what
 * generation finished, how fit the best genome was, and optionally the best
 * network for visualization or playback.
 */
export interface EvolutionGenerationPayload {
  architectureProfileId: ExampleArchitectureProfileId;
  generation: number;
  bestFitness: number;
  bestNetworkJson?: SerializedNetwork;
  populationNetworksJson?: SerializedNetwork[];
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

/**
 * Per-frame snapshot received from the worker playback channel.
 *
 * A snapshot combines geometry, packed population state, and lightweight world
 * metadata so the browser can render a deterministic frame without rerunning
 * the simulation locally.
 */
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

/**
 * Worker message carrying one playback step and aggregate markers.
 *
 * Besides the frame snapshot itself, this message also carries summary values
 * used by the HUD so the browser can show performance and progress without
 * recomputing population-wide statistics on the main thread.
 */
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

/**
 * Union of all supported worker messages consumed by browser entry.
 *
 * A closed union keeps the main-thread message handler explicit and easy to
 * audit when the protocol evolves.
 */
export type EvolutionWorkerMessage =
  | EvolutionGenerationReadyMessage
  | EvolutionPlaybackStepMessage
  | EvolutionWorkerErrorMessage;

/**
 * Lightweight per-frame telemetry emitted to HUD update callback.
 *
 * These values are the browser-friendly metrics shown in the live status panel:
 * how many birds remain, how far the leader has progressed, and how expensive
 * the current playback cadence is.
 */
export interface PlaybackFrameStats {
  frameIndex: number;
  activeBirdCount: number;
  leaderPipesPassed: number;
  leaderFramesSurvived: number;
  activationCallsPerFrame: number;
  simulationStepsPerRaf: number;
}
