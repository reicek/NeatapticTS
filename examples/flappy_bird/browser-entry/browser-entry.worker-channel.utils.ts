import type {
  WorkerChannelGenerationPayload,
  WorkerChannelPlaybackStepPayload,
  WorkerChannelPlaybackStepRequest,
} from './worker-channel/worker-channel.types';
import { resolveEvolutionWorkerBundleUrl } from './worker-channel/worker-channel.url.service';
import { requestWorkerGeneration as requestWorkerGenerationService } from './worker-channel/worker-channel.generation.service';
import { requestWorkerPlaybackStep as requestWorkerPlaybackStepService } from './worker-channel/worker-channel.playback.service';

const FLAPPY_BROWSER_WORKER_LOG_PREFIX = '[flappy-browser]';
const SHOULD_LOG_FLAPPY_BROWSER_WORKERS =
  resolveNodeEnvForRuntimeLogs() !== 'test';

/**
 * Creates the evolution worker used to keep heavy NEAT compute off the UI thread.
 *
 * @returns Initialized worker instance.
 */
export function createEvolutionWorker(): Worker {
  // Step 1: Resolve worker URL from active bundle context.
  const workerUrl = resolveEvolutionWorkerBundleUrl();

  if (SHOULD_LOG_FLAPPY_BROWSER_WORKERS) {
    console.info(
      `${FLAPPY_BROWSER_WORKER_LOG_PREFIX} starting evolution Web Worker from ${workerUrl}`,
    );
  }

  // Step 2: Create worker instance from resolved URL.
  return new Worker(workerUrl);
}

function resolveNodeEnvForRuntimeLogs(): string | undefined {
  return (globalThis as { process?: { env?: { NODE_ENV?: string } } }).process
    ?.env?.NODE_ENV;
}

/**
 * Waits for the next generation payload emitted by the evolution worker.
 *
 * @param evolutionWorker - Worker emitting generation-ready messages.
 * @returns Next generation payload.
 */
export function requestWorkerGeneration(
  evolutionWorker: Worker,
): Promise<WorkerChannelGenerationPayload> {
  return requestWorkerGenerationService(evolutionWorker);
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
  playbackStepRequest: WorkerChannelPlaybackStepRequest,
): Promise<WorkerChannelPlaybackStepPayload> {
  return requestWorkerPlaybackStepService(evolutionWorker, playbackStepRequest);
}
