import type {
  EvolutionGenerationPayload,
  EvolutionPlaybackStepMessage,
  EvolutionWorkerMessage,
} from './browser-entry.types';

/**
 * Creates the evolution worker used to keep heavy NEAT compute off the UI thread.
 *
 * @returns Initialized worker instance.
 */
export function createEvolutionWorker(): Worker {
  // Step 1: Locate the currently loaded browser bundle script element.
  const scriptElements = document.querySelectorAll('script[src]');
  const currentBundleScript = Array.from(scriptElements)
    .map((scriptElement) => scriptElement as HTMLScriptElement)
    .find((scriptElement) =>
      scriptElement.src.includes('flappy-bird.bundle.js'),
    );

  // Step 2: Resolve worker URL relative to current bundle (or page URL fallback).
  const workerBaseUrl = currentBundleScript?.src ?? window.location.href;
  const workerUrl = new URL(
    'flappy-evolution.worker.bundle.js',
    workerBaseUrl,
  ).toString();
  // Step 3: Create worker instance from resolved URL.
  return new Worker(workerUrl);
}

/**
 * Waits for the next generation payload emitted by the evolution worker.
 *
 * @param evolutionWorker - Worker emitting generation-ready messages.
 * @returns Next generation payload.
 */
export function requestWorkerGeneration(
  evolutionWorker: Worker,
): Promise<EvolutionGenerationPayload> {
  return new Promise((resolve, reject) => {
    // Step 1: Handle worker messages and route by message type.
    const handleMessage = (
      event: MessageEvent<EvolutionWorkerMessage>,
    ): void => {
      const workerMessage = event.data;
      if (workerMessage.type === 'generation-ready') {
        cleanup();
        resolve(workerMessage.payload);
        return;
      }

      if (workerMessage.type === 'error') {
        cleanup();
        reject(new Error(workerMessage.payload.message));
      }
    };

    // Step 2: Handle worker-level execution errors.
    const handleError = (event: ErrorEvent): void => {
      cleanup();
      reject(
        event.error instanceof Error ? event.error : new Error(event.message),
      );
    };

    // Step 3: Remove transient listeners once request settles.
    const cleanup = (): void => {
      evolutionWorker.removeEventListener(
        'message',
        handleMessage as EventListener,
      );
      evolutionWorker.removeEventListener(
        'error',
        handleError as EventListener,
      );
    };

    // Step 4: Register listeners and issue generation request.
    evolutionWorker.addEventListener('message', handleMessage as EventListener);
    evolutionWorker.addEventListener('error', handleError as EventListener);
    evolutionWorker.postMessage({ type: 'request-generation' });
  });
}

/**
 * Requests one playback batch step from the worker.
 *
 * @param evolutionWorker - Worker that owns playback simulation state.
 * @param playbackStepRequest - Requested simulation budget and viewport width.
 * @returns Playback-step payload including snapshot and completion marker.
 */
export function requestWorkerPlaybackStep(
  evolutionWorker: Worker,
  playbackStepRequest: {
    simulationSteps: number;
    visibleWorldWidthPx: number;
    visibleWorldHeightPx: number;
  },
): Promise<EvolutionPlaybackStepMessage['payload']> {
  return new Promise((resolve, reject) => {
    // Step 1: Handle worker messages and route by message type.
    const handleMessage = (
      event: MessageEvent<EvolutionWorkerMessage>,
    ): void => {
      const workerMessage = event.data;
      if (workerMessage.type === 'playback-step') {
        cleanup();
        resolve(workerMessage.payload);
        return;
      }

      if (workerMessage.type === 'error') {
        cleanup();
        reject(new Error(workerMessage.payload.message));
      }
    };

    // Step 2: Handle worker-level execution errors.
    const handleError = (event: ErrorEvent): void => {
      cleanup();
      reject(
        event.error instanceof Error ? event.error : new Error(event.message),
      );
    };

    // Step 3: Remove transient listeners once request settles.
    const cleanup = (): void => {
      evolutionWorker.removeEventListener(
        'message',
        handleMessage as EventListener,
      );
      evolutionWorker.removeEventListener(
        'error',
        handleError as EventListener,
      );
    };

    // Step 4: Register listeners and issue playback-step request.
    evolutionWorker.addEventListener('message', handleMessage as EventListener);
    evolutionWorker.addEventListener('error', handleError as EventListener);
    evolutionWorker.postMessage({
      type: 'request-playback-step',
      payload: playbackStepRequest,
    });
  });
}
