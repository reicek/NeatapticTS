import type {
  EvolutionGenerationPayload,
  EvolutionPlaybackStepMessage,
  EvolutionWorkerMessage,
} from '../browser-entry.types';

/**
 * Shared protocol contracts for the browser-entry worker channel.
 *
 * These types define the thin translation layer between generic worker messages
 * and the specific request/response flows used by the Flappy Bird browser UI.
 *
 * If you want background reading, the Wikipedia article on "message passing"
 * gives the right mental model for this boundary.
 */

/**
 * Aliases the shared worker message union for worker-channel modules.
 *
 * Keeping the alias local makes submodules read as protocol-focused code rather
 * than browser-entry plumbing.
 */
export type WorkerChannelMessage = EvolutionWorkerMessage;

/**
 * Aliases the generation payload contract returned by the worker.
 *
 * This payload arrives when one NEAT generation has finished evolving and the
 * browser is ready to update its "best so far" view.
 */
export type WorkerChannelGenerationPayload = EvolutionGenerationPayload;

/**
 * Aliases the playback-step payload contract returned by the worker.
 *
 * This is the browser-side shape of one streamed playback frame plus its
 * accompanying aggregate telemetry.
 */
export type WorkerChannelPlaybackStepPayload =
  EvolutionPlaybackStepMessage['payload'];

/**
 * Request payload sent when asking the worker to advance playback simulation.
 *
 * The request declares both simulation budget and viewport dimensions so the
 * worker can package a frame that already matches the browser's current canvas
 * world.
 */
export interface WorkerChannelPlaybackStepRequest {
  simulationSteps: number;
  visibleWorldWidthPx: number;
  visibleWorldHeightPx: number;
}
