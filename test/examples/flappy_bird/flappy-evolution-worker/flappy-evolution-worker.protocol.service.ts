import type {
  WorkerInitMessage,
  WorkerRequestMessage,
  WorkerResponseMessage,
  WorkerStartPlaybackMessage,
  WorkerRequestPlaybackStepMessage,
} from './flappy-evolution-worker.types';
import {
  createWorkerErrorMessage,
  FLAPPY_WORKER_PLAYBACK_START_REQUIRES_GENERATION_ERROR_MESSAGE,
  FLAPPY_WORKER_PLAYBACK_STEP_REQUIRES_START_ERROR_MESSAGE,
} from './flappy-evolution-worker.errors';

/**
 * Callback bundle used by worker protocol routing.
 *
 * Each callback corresponds to one legal transition in the worker message
 * protocol. Keeping the router dependent on this narrow interface makes the
 * protocol easy to read in generated docs and easy to test independently from
 * the worker-global `self.onmessage` hook.
 */
export interface WorkerProtocolHandlers {
  markStopped: () => void;
  beginInitialization: (payload: WorkerInitMessage['payload']) => void;
  beginGenerationRequest: () => void;
  startPlayback: (payload: WorkerStartPlaybackMessage['payload']) => void;
  processPlaybackStep: (
    payload: WorkerRequestPlaybackStepMessage['payload'],
  ) => void;
  hasPopulation: () => boolean;
  hasPlaybackState: () => boolean;
  postWorkerMessage: (
    workerMessage: WorkerResponseMessage,
    transferList?: Transferable[],
  ) => void;
}

/**
 * Routes one inbound worker request message to the corresponding runtime action.
 *
 * Educational note:
 * This router is the protocol gatekeeper for the worker. It enforces the two
 * important sequencing rules in the demo:
 * - playback requires a previously evolved population,
 * - playback stepping requires an active playback session.
 *
 * In practice this acts like a tiny finite-state machine. If you want a quick
 * conceptual refresher, the Wikipedia article on "finite-state machine" maps
 * well onto the worker's init -> evolve -> start playback -> step playback flow.
 *
 * @example
 * ```ts
 * routeWorkerProtocolMessage(
 *   { type: 'request-generation' },
 *   workerProtocolHandlers,
 * );
 * ```
 *
 * @param workerMessage - Inbound worker request payload.
 * @param handlers - Runtime action callbacks and state probes.
 * @returns Nothing.
 */
export function routeWorkerProtocolMessage(
  workerMessage: WorkerRequestMessage,
  handlers: WorkerProtocolHandlers,
): void {
  // Step 1: Handle stop requests first so long-running operations can observe it.
  if (workerMessage.type === 'stop') {
    handlers.markStopped();
    return;
  }

  // Step 2: Initialize runtime once and capture initialization failures.
  if (workerMessage.type === 'init') {
    handlers.beginInitialization(workerMessage.payload);
    return;
  }

  // Step 3: Evolve one generation and publish generation-ready payload.
  if (workerMessage.type === 'request-generation') {
    handlers.beginGenerationRequest();
    return;
  }

  // Step 4: Create a fresh playback simulation from the current evolved population.
  if (workerMessage.type === 'start-playback') {
    if (!handlers.hasPopulation()) {
      handlers.postWorkerMessage(
        createWorkerErrorMessage(
          FLAPPY_WORKER_PLAYBACK_START_REQUIRES_GENERATION_ERROR_MESSAGE,
        ),
      );
      return;
    }

    handlers.startPlayback(workerMessage.payload);
    return;
  }

  // Step 5: Advance playback only when a playback state already exists.
  if (workerMessage.type === 'request-playback-step') {
    if (!handlers.hasPlaybackState()) {
      handlers.postWorkerMessage(
        createWorkerErrorMessage(
          FLAPPY_WORKER_PLAYBACK_STEP_REQUIRES_START_ERROR_MESSAGE,
        ),
      );
      return;
    }

    handlers.processPlaybackStep(workerMessage.payload);
  }
}
