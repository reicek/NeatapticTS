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
  WorkerStartPlaybackMessage,
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
import { createWorkerPopulationRenderState } from './flappy-evolution-worker.simulation.utils';
import { stepWorkerPopulationFrame } from './flappy-evolution-worker.simulation.frame.service';
import { warmStartWorkerGenerationZeroIfNeeded } from './flappy-evolution-worker.warm-start.service';

const FLAPPY_WORKER_INITIAL_SEED = 0;
const FLAPPY_WORKER_INITIAL_WINNER_INDEX = -1;

type WorkerMutableRuntimeState = {
  stopped: boolean;
  neatRuntime: Neat | undefined;
  currentPopulation: Network[];
  currentPlaybackState: WorkerPlaybackState | undefined;
  currentPlaybackRng: ReturnType<typeof createXorshift32> | undefined;
  playbackWinnerIndex: number;
  initializationPromise: Promise<void> | undefined;
  workerInitSeed: number;
  generationZeroWarmStartApplied: boolean;
};

const workerMutableRuntimeState = createWorkerMutableRuntimeState();

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
self.onmessage = createWorkerMessageHandler(workerMutableRuntimeState);

/**
 * Creates the mutable worker runtime state container.
 *
 * @returns Mutable worker runtime state.
 */
function createWorkerMutableRuntimeState(): WorkerMutableRuntimeState {
  // Step 1: Initialize the worker state with empty runtime and playback fields.
  return {
    stopped: false,
    neatRuntime: undefined,
    currentPopulation: [],
    currentPlaybackState: undefined,
    currentPlaybackRng: undefined,
    playbackWinnerIndex: FLAPPY_WORKER_INITIAL_WINNER_INDEX,
    initializationPromise: undefined,
    workerInitSeed: FLAPPY_WORKER_INITIAL_SEED,
    generationZeroWarmStartApplied: false,
  };
}

/**
 * Creates the top-level worker message handler.
 *
 * @param workerMutableRuntimeState - Mutable worker runtime state.
 * @returns Worker message handler.
 */
function createWorkerMessageHandler(
  workerMutableRuntimeState: WorkerMutableRuntimeState,
): (event: MessageEvent<WorkerRequestMessage>) => void {
  // Step 1: Capture the protocol handlers so each message reuses the same stateful callbacks.
  const workerProtocolHandlers = createWorkerProtocolHandlers(
    workerMutableRuntimeState,
  );

  // Step 2: Return the thin orchestration handler used by `self.onmessage`.
  return (event: MessageEvent<WorkerRequestMessage>): void => {
    const workerMessage = event.data;
    routeWorkerProtocolMessage(workerMessage, workerProtocolHandlers);
  };
}

/**
 * Creates protocol handlers bound to the mutable worker runtime state.
 *
 * @param workerMutableRuntimeState - Mutable worker runtime state.
 * @returns Protocol handler bundle.
 */
function createWorkerProtocolHandlers(
  workerMutableRuntimeState: WorkerMutableRuntimeState,
) {
  // Step 1: Bind each protocol operation to the shared mutable runtime state.
  return {
    markStopped: (): void => {
      workerMutableRuntimeState.stopped = true;
    },
    beginInitialization: (payload: WorkerInitMessage['payload']): void => {
      beginWorkerInitialization(workerMutableRuntimeState, payload);
    },
    beginGenerationRequest: (): void => {
      beginWorkerGenerationRequest(workerMutableRuntimeState);
    },
    hasPopulation: (): boolean =>
      workerMutableRuntimeState.currentPopulation.length > 0,
    startPlayback: (payload: WorkerStartPlaybackMessage['payload']): void => {
      beginWorkerPlayback(workerMutableRuntimeState, payload);
    },
    hasPlaybackState: (): boolean =>
      workerMutableRuntimeState.currentPlaybackState != null,
    processPlaybackStep: (
      payload: WorkerRequestPlaybackStepMessage['payload'],
    ): void => {
      processWorkerPlaybackStepRequest(workerMutableRuntimeState, payload);
    },
    postWorkerMessage,
  };
}

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
  workerMutableRuntimeState: WorkerMutableRuntimeState,
  initPayload: WorkerInitMessage['payload'],
): Promise<void> {
  // Step 1: Persist the worker seed so warm-start logic can reuse it deterministically.
  workerMutableRuntimeState.workerInitSeed = initPayload.rngSeed;

  // Step 2: Build and configure the NEAT runtime controller.
  workerMutableRuntimeState.neatRuntime =
    createInitializedWorkerRuntime(initPayload);
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
async function evolveAndPublishGeneration(
  workerMutableRuntimeState: WorkerMutableRuntimeState,
): Promise<void> {
  const generationPayload = await evolveAndBuildGenerationReadyMessage({
    initializationPromise: workerMutableRuntimeState.initializationPromise,
    neatRuntime: workerMutableRuntimeState.neatRuntime,
    isStopped: () => workerMutableRuntimeState.stopped,
    warmStartGenerationZeroIfNeeded: (neatController) =>
      warmStartWorkerGenerationZeroIfNeeded(neatController, {
        workerInitSeed: workerMutableRuntimeState.workerInitSeed,
        generationZeroWarmStartApplied:
          workerMutableRuntimeState.generationZeroWarmStartApplied,
      }).then(() => {
        workerMutableRuntimeState.generationZeroWarmStartApplied = true;
      }),
    setCurrentPopulation: (nextPopulation) => {
      workerMutableRuntimeState.currentPopulation = nextPopulation;
    },
  });

  postWorkerMessage(generationPayload);
}

/**
 * Begins worker initialization and captures asynchronous failures.
 *
 * @param workerMutableRuntimeState - Mutable worker runtime state.
 * @param initPayload - Initialization payload.
 * @returns Nothing.
 */
function beginWorkerInitialization(
  workerMutableRuntimeState: WorkerMutableRuntimeState,
  initPayload: WorkerInitMessage['payload'],
): void {
  // Step 1: Start initialization and retain the promise for later generation requests.
  workerMutableRuntimeState.initializationPromise = initializeRuntime(
    workerMutableRuntimeState,
    initPayload,
  ).catch((error: unknown) => {
    postWorkerMessage(createWorkerErrorMessageFromUnknown(error));
    throw error;
  });

  // Step 2: Emit a stable init-failed error if the initialization promise rejects.
  void workerMutableRuntimeState.initializationPromise.catch(() => {
    postWorkerMessage(
      createWorkerErrorMessage(FLAPPY_WORKER_INIT_FAILED_ERROR_MESSAGE),
    );
  });
}

/**
 * Begins one asynchronous generation request and captures failures.
 *
 * @param workerMutableRuntimeState - Mutable worker runtime state.
 * @returns Nothing.
 */
function beginWorkerGenerationRequest(
  workerMutableRuntimeState: WorkerMutableRuntimeState,
): void {
  // Step 1: Run the generation pipeline and publish worker errors on failure.
  void evolveAndPublishGeneration(workerMutableRuntimeState).catch(
    (error: unknown) => {
      postWorkerMessage(createWorkerErrorMessageFromUnknown(error));
    },
  );
}

/**
 * Begins a new playback session from the current evolved population.
 *
 * @param workerMutableRuntimeState - Mutable worker runtime state.
 * @param payload - Playback start payload.
 * @returns Nothing.
 */
function beginWorkerPlayback(
  workerMutableRuntimeState: WorkerMutableRuntimeState,
  payload: { visibleWorldWidthPx: number; visibleWorldHeightPx: number },
): void {
  // Step 1: Create the next playback session from the current population snapshot.
  const nextPlaybackSessionState = beginWorkerPlaybackSession({
    currentPopulation: workerMutableRuntimeState.currentPopulation,
    payload,
    createPopulationRenderState: createWorkerPopulationRenderState,
  });

  // Step 2: Persist the next playback state, RNG, and winner index.
  workerMutableRuntimeState.currentPlaybackState =
    nextPlaybackSessionState.currentPlaybackState;
  workerMutableRuntimeState.currentPlaybackRng =
    nextPlaybackSessionState.currentPlaybackRng;
  workerMutableRuntimeState.playbackWinnerIndex =
    nextPlaybackSessionState.playbackWinnerIndex;
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
function processWorkerPlaybackStepRequest(
  workerMutableRuntimeState: WorkerMutableRuntimeState,
  playbackStepPayload: WorkerRequestPlaybackStepMessage['payload'],
): void {
  // Step 1: Guard against playback-step requests before playback state exists.
  if (
    !workerMutableRuntimeState.currentPlaybackState ||
    !workerMutableRuntimeState.currentPlaybackRng
  ) {
    throw new Error(
      'Playback random source is unavailable. Start playback first.',
    );
  }

  // Step 2: Advance playback and build the next compact snapshot payload.
  const nextPlaybackStepState = processWorkerPlaybackStep({
    playbackStepPayload,
    currentPlaybackState: workerMutableRuntimeState.currentPlaybackState,
    currentPlaybackRng: workerMutableRuntimeState.currentPlaybackRng,
    currentPopulation: workerMutableRuntimeState.currentPopulation,
    neatRuntime: workerMutableRuntimeState.neatRuntime,
    stepPopulationFrame: stepWorkerPopulationFrame,
    createPlaybackSnapshot: createWorkerPlaybackSnapshot,
    postWorkerMessage,
  });

  // Step 3: Persist the advanced playback state for the next host request.
  workerMutableRuntimeState.currentPlaybackState =
    nextPlaybackStepState.currentPlaybackState;
  workerMutableRuntimeState.currentPlaybackRng =
    nextPlaybackStepState.currentPlaybackRng;
  workerMutableRuntimeState.currentPopulation =
    nextPlaybackStepState.currentPopulation;
  workerMutableRuntimeState.playbackWinnerIndex =
    nextPlaybackStepState.playbackWinnerIndex;
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
