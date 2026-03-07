import type {
  EvolutionGenerationPayload,
  EvolutionPlaybackStepMessage,
  EvolutionWorkerMessage,
} from '../browser-entry.types';

/**
 * Aliases the shared worker message union for worker-channel modules.
 */
export type WorkerChannelMessage = EvolutionWorkerMessage;

/**
 * Aliases the generation payload contract returned by the worker.
 */
export type WorkerChannelGenerationPayload = EvolutionGenerationPayload;

/**
 * Aliases the playback-step payload contract returned by the worker.
 */
export type WorkerChannelPlaybackStepPayload =
  EvolutionPlaybackStepMessage['payload'];

/**
 * Request payload sent when asking the worker to advance playback simulation.
 */
export interface WorkerChannelPlaybackStepRequest {
  simulationSteps: number;
  visibleWorldWidthPx: number;
  visibleWorldHeightPx: number;
}
