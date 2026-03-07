/// <reference lib="webworker" />

import { Neat } from '../../../../src/neataptic';
import Network from '../../../../src/architecture/network';
import { createXorshift32 } from '../rng';
import type {
  WorkerInitMessage,
  WorkerPlaybackState,
  WorkerRequestMessage,
  WorkerRequestPlaybackStepMessage,
  WorkerResponseMessage,
} from './flappy-evolution-worker.types';
import { createInitializedWorkerRuntime } from './flappy-evolution-worker.runtime.service';
import { FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION } from '../constants/constants';
import {
  createWorkerErrorMessage,
  createWorkerErrorMessageFromUnknown,
  FLAPPY_WORKER_INIT_FAILED_ERROR_MESSAGE,
} from './flappy-evolution-worker.errors';
import { routeWorkerProtocolMessage } from './flappy-evolution-worker.protocol.service';
import { evolveAndBuildGenerationReadyMessage } from './flappy-evolution-worker.evolution.service';
import {
  beginWorkerPlaybackSession,
  processWorkerPlaybackStep,
} from './flappy-evolution-worker.playback.service';
import { createWorkerPlaybackSnapshot } from './flappy-evolution-worker.snapshot.utils';
import {
  createWorkerPopulationRenderState,
  stepWorkerPopulationFrame,
} from './flappy-evolution-worker.simulation.utils';
import { warmStartWorkerGenerationZeroIfNeeded } from './flappy-evolution-worker.warm-start.service';

let stopped = false;
let neatRuntime: Neat | undefined;
let currentPopulation: Network[] = [];
let currentPlaybackState: WorkerPlaybackState | undefined;
let currentPlaybackRng: ReturnType<typeof createXorshift32> | undefined;
let playbackWinnerIndex = -1;
let initializationPromise: Promise<void> | undefined;
let workerInitSeed = 0;
let generationZeroWarmStartApplied = false;

/**
 * Main worker message router.
 *
 * Educational note:
 * Web Workers are message-driven runtimes. Instead of calling methods directly,
 * the host posts typed messages, and the worker folds those messages into state.
 * This keeps rendering and heavy simulation/evolution work decoupled.
 *
 * Routing policy in this worker:
 * - `init`: create NEAT runtime and seed deterministic RNG state.
 * - `request-generation`: run one evolution cycle and publish summary payload.
 * - `start-playback`: materialize simulation state from current population.
 * - `request-playback-step`: advance playback and stream telemetry snapshots.
 * - `stop`: mark worker as stopped (future generation requests fail fast).
 */
self.onmessage = (event: MessageEvent<WorkerRequestMessage>) => {
  // Step 1: Decode the incoming discriminated-union message.
  const workerMessage = event.data;

  // Step 2: Route inbound protocol message to worker lifecycle handlers.
  routeWorkerProtocolMessage(workerMessage, {
    markStopped: () => {
      stopped = true;
    },
    beginInitialization: (payload) => {
      initializationPromise = initializeRuntime(payload).catch(
        (error: unknown) => {
          postWorkerMessage(createWorkerErrorMessageFromUnknown(error));
          throw error;
        },
      );
      void initializationPromise.catch(() => {
        postWorkerMessage(
          createWorkerErrorMessage(FLAPPY_WORKER_INIT_FAILED_ERROR_MESSAGE),
        );
      });
    },
    beginGenerationRequest: () => {
      void evolveAndPublishGeneration().catch((error: unknown) => {
        postWorkerMessage(createWorkerErrorMessageFromUnknown(error));
      });
    },
    hasPopulation: () =>
      Array.isArray(currentPopulation) && currentPopulation.length > 0,
    startPlayback: (payload) => {
      const nextPlaybackSessionState = beginWorkerPlaybackSession({
        currentPopulation,
        payload,
        createPopulationRenderState: createWorkerPopulationRenderState,
      });
      currentPlaybackState = nextPlaybackSessionState.currentPlaybackState;
      currentPlaybackRng = nextPlaybackSessionState.currentPlaybackRng;
      playbackWinnerIndex = nextPlaybackSessionState.playbackWinnerIndex;
    },
    hasPlaybackState: () => currentPlaybackState != null,
    processPlaybackStep,
    postWorkerMessage,
  });
};

/**
 * Initializes the worker-local NEAT runtime used by browser evolution playback.
 *
 * Educational note:
 * The runtime is configured once with deterministic RNG state and a lightweight
 * early-termination fitness rollout. Keeping this setup centralized helps ensure
 * reproducibility between runs and keeps host<->worker contracts simple.
 *
 * @param initPayload - Initialization values from the browser host.
 * @returns Promise resolved when runtime setup is complete.
 */
async function initializeRuntime(
  initPayload: WorkerInitMessage['payload'],
): Promise<void> {
  // Step 1: Persist the worker seed so warm-start logic can reuse it deterministically.
  workerInitSeed = initPayload.rngSeed;

  // Step 2: Build and configure the NEAT runtime controller.
  neatRuntime = createInitializedWorkerRuntime(initPayload);
}

/**
 * Evolves one generation and publishes the best-network summary message.
 *
 * Educational note:
 * This method is the orchestration seam between evolutionary search and
 * browser rendering: it runs evolution, snapshots the population, and emits
 * a compact payload for UI state updates.
 *
 * @returns Promise resolved after generation payload is posted.
 */
async function evolveAndPublishGeneration(): Promise<void> {
  const generationPayload = await evolveAndBuildGenerationReadyMessage({
    initializationPromise,
    neatRuntime,
    isStopped: () => stopped,
    warmStartGenerationZeroIfNeeded: (neatController) =>
      warmStartWorkerGenerationZeroIfNeeded(neatController, {
        workerInitSeed,
        generationZeroWarmStartApplied,
      }).then(() => {
        generationZeroWarmStartApplied = true;
      }),
    setCurrentPopulation: (nextPopulation) => {
      currentPopulation = nextPopulation;
    },
  });

  postWorkerMessage(generationPayload);
}

/**
 * Advances playback by a host-requested number of simulation steps.
 *
 * Educational note:
 * The browser host can request multiple simulation steps per RAF to trade visual
 * smoothness against throughput. This function keeps that loop deterministic and
 * emits one compact snapshot payload per request.
 *
 * @param playbackStepPayload - Host-selected simulation-step budget and viewport.
 * @returns Nothing.
 */
function processPlaybackStep(
  playbackStepPayload: WorkerRequestPlaybackStepMessage['payload'],
): void {
  if (!currentPlaybackState || !currentPlaybackRng) {
    throw new Error(
      'Playback random source is unavailable. Start playback first.',
    );
  }

  const nextPlaybackStepState = processWorkerPlaybackStep({
    playbackStepPayload,
    currentPlaybackState,
    currentPlaybackRng,
    currentPopulation,
    neatRuntime,
    stepPopulationFrame: stepWorkerPopulationFrame,
    createPlaybackSnapshot: createWorkerPlaybackSnapshot,
    postWorkerMessage,
  });

  currentPlaybackState = nextPlaybackStepState.currentPlaybackState;
  currentPlaybackRng = nextPlaybackStepState.currentPlaybackRng;
  currentPopulation = nextPlaybackStepState.currentPopulation;
  playbackWinnerIndex = nextPlaybackStepState.playbackWinnerIndex;
}

/**
 * Posts a typed message from worker to host.
 *
 * @param workerMessage - Outbound worker response payload.
 * @returns Nothing.
 */
function postWorkerMessage(workerMessage: WorkerResponseMessage): void {
  self.postMessage(workerMessage);
}
