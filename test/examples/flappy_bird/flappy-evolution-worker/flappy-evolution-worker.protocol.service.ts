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
