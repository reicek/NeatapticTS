/**
 * Off-thread evolution and playback authority for the Flappy Bird browser demo.
 *
 * This worker boundary exists to keep the browser honest. The main thread owns
 * explanation, HUD rendering, and network inspection, while the worker owns the
 * hot path: evolving generations, materializing playback state, advancing the
 * simulation, and packaging compact snapshots back to the host.
 *
 * That separation is doing two jobs at once. It protects the browser from
 * heavy simulation work, and it turns responsibility into something a reader
 * can see directly because every cross-thread handoff must become an explicit
 * typed message.
 *
 * Read this folder as the protocol chapter between the browser host and the
 * deterministic runtime that actually evolves and replays the flock. The
 * important design question is not merely "how do Web Workers run code?" It is
 * "which side should own each piece of truth when evolution, playback, and
 * inspection all need the same population?"
 *
 * In this example, the answer is deliberate:
 *
 * - the host owns controls, HUD state, and network visualization,
 * - the worker owns generation requests, playback stepping, and winner
 *   selection,
 * - packed snapshots are the narrow bridge between those two worlds.
 *
 * ## What This Folder Is Trying To Teach
 *
 * The worker chapter is organized around four reader questions:
 *
 * 1. How does the browser request evolution work without becoming the
 *    simulation authority?
 * 2. How does one evolved population become a replayable playback session?
 * 3. Why are snapshots packed into typed arrays instead of posted as nested
 *    render objects?
 * 4. Where should you read next when the protocol is clear but one runtime step
 *    still feels opaque?
 *
 * ## Core Worker Map
 *
 * ```mermaid
 * flowchart LR
 *     Host["browser-entry/\nhost UI and controls"] --> Protocol["protocol service\nlegal message transitions"]
 *     Protocol --> Runtime["runtime service\nworker-local NEAT controller"]
 *     Runtime --> Evolution["evolution service\nadvance one generation"]
 *     Runtime --> Playback["playback service\nmaterialize and step population"]
 *     Playback --> Snapshot["snapshot utils\npacked typed-array transport"]
 *     Snapshot --> Host
 *
 *     WarmStart["warm-start service\ngeneration zero bootstrap"] -.-> Evolution
 *     Types["worker types\nmessage and DTO contracts"] -.-> Protocol
 *
 *     classDef boundary fill:#001522,stroke:#0fb5ff,color:#9fdcff,stroke-width:2px;
 *     classDef runtime fill:#03111f,stroke:#00e5ff,color:#d8f6ff,stroke-width:2px;
 *     classDef highlight fill:#2a1029,stroke:#ff4a8d,color:#ffd7e8,stroke-width:3px;
 *
 *     class Host,Protocol,Types boundary;
 *     class Runtime,Evolution,Playback,Snapshot runtime;
 *     class Snapshot highlight;
 * ```
 *
 * Read the diagram left to right. The host is allowed to request work, but it
 * never takes ownership of the worker's mutable simulation state. The worker
 * can then optimize for determinism and throughput while the browser optimizes
 * for explanation.
 *
 * ## Choose Your Route
 *
 * - Start with `flappy-evolution-worker.ts` if you want the high-level message
 *   lifecycle.
 * - Read `flappy-evolution-worker.protocol.service.ts` next if you want the
 *   legal sequencing rules.
 * - Read `flappy-evolution-worker.runtime.service.ts` if you want the worker's
 *   NEAT runtime setup.
 * - Read `flappy-evolution-worker.playback.service.ts` and
 *   `flappy-evolution-worker.simulation.frame.service.ts` if you want the hot
 *   playback path.
 * - Read `flappy-evolution-worker.snapshot.utils.ts` if you want the typed-array
 *   transport story.
 *
 * Minimal host-side sketch:
 *
 * ```ts
 * worker.postMessage({
 *   type: 'init',
 *   payload: { populationSize: 50, elitismCount: 10, rngSeed: 12345 },
 * });
 * worker.postMessage({ type: 'request-generation' });
 * worker.postMessage({
 *   type: 'start-playback',
 *   payload: { visibleWorldWidthPx: 1280, visibleWorldHeightPx: 720 },
 * });
 * worker.postMessage({
 *   type: 'request-playback-step',
 *   payload: {
 *     requestId: 1,
 *     simulationSteps: 2,
 *     visibleWorldWidthPx: 1280,
 *     visibleWorldHeightPx: 720,
 *   },
 * });
 * ```
 *
 * If you want background reading before the symbol shelf, the MDN Web Workers
 * guide is the fastest practical reference for why this example pushes both
 * evolution and playback off the main thread.
 */
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
import {
  createWorkerPlaybackSnapshot,
  resolveWorkerPlaybackSnapshotTransferList,
} from './flappy-evolution-worker.snapshot.utils';
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
 * If you want background reading, the MDN article on Web Workers is the most
 * practical reference for understanding why this example moves evolution and
 * playback simulation off the main thread.
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
 * Educational note:
 * The worker keeps one small mutable state bag instead of scattering globals.
 * That makes the protocol flow easier to explain and lets the entrypoint pass a
 * single dependency object through the orchestration helpers.
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
 * The returned function is intentionally thin. All protocol decisions are
 * delegated to the router service so the worker entrypoint stays readable as a
 * high-level orchestration module.
 *
 * @example
 * ```ts
 * self.onmessage = createWorkerMessageHandler(workerMutableRuntimeState);
 * ```
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
 * This helper is the bridge between the pure protocol router and the impure
 * worker runtime. Each callback closes over the same mutable state bag so the
 * protocol layer can remain small and declarative.
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
 * This function only prepares the evolutionary controller. It does not start
 * playback and it does not evolve a generation yet; those remain separate
 * protocol steps so the host can control them explicitly.
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
 * The worker retains the initialization promise so later generation requests can
 * await setup completion instead of racing against it.
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
 * This is intentionally fire-and-forget from the protocol perspective. The
 * actual completion signal is the later `generation-ready` or `error` message
 * posted back to the host.
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
 * A playback session is a deterministic simulation snapshot seeded from the
 * current population. Each new session resets playback RNG and world state so
 * the host can replay generations cleanly.
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
 * Importantly, the worker remains authoritative for deciding when the run is
 * over and which bird should be treated as the playback winner.
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
    resolvePlaybackSnapshotTransferList:
      resolveWorkerPlaybackSnapshotTransferList,
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
 * This is the narrowest possible transport helper: all message construction is
 * done elsewhere so the README can point to one stable worker-to-host boundary.
 *
 * @param workerMessage - Outbound worker response payload.
 * @param transferList - Optional transferable buffers moved with the payload.
 * @returns Nothing.
 */
function postWorkerMessage(
  workerMessage: WorkerResponseMessage,
  transferList?: Transferable[],
): void {
  self.postMessage(workerMessage, transferList ?? []);
}
