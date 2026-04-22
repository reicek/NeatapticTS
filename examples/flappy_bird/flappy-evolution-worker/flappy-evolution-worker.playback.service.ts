import type Network from '../../../src/architecture/network';
import { createXorshift32 } from '../rng';
import {
  hasAliveBirds,
  resolveFramePrimaryWinnerIndex,
  resolveLeaderPipesPassed,
} from '../browser-entry/browser-entry.observation.utils';
import {
  FLAPPY_BROWSER_SUCCESS_DOWNSHIFT_ELITISM_COUNT,
  FLAPPY_BROWSER_SUCCESS_DOWNSHIFT_POPULATION_SIZE,
  FLAPPY_BROWSER_SUCCESS_PIPE_TARGET,
  FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION,
} from '../constants/constants';
import {
  resolveAdaptiveDifficultyProfile,
  type SharedDifficultyProfile,
} from '../flappy.simulation.shared.utils';
import type {
  WorkerPlaybackFrameSnapshot,
  WorkerPlaybackState,
  WorkerRequestPlaybackStepMessage,
  WorkerResponseMessage,
  WorkerStartPlaybackMessage,
} from './flappy-evolution-worker.types';
import type { Neat } from '../../../src/neataptic';

/**
 * Creates a fresh worker playback session state from the current evolved population.
 *
 * Educational note:
 * Evolution and playback are intentionally separated. Evolution produces a new
 * population, then playback freezes that population into a deterministic
 * simulation state that the host can step frame-by-frame for rendering.
 *
 * @example
 * ```ts
 * const session = beginWorkerPlaybackSession({
 *   currentPopulation,
 *   payload: { visibleWorldWidthPx: 1280, visibleWorldHeightPx: 720 },
 *   createPopulationRenderState,
 * });
 * ```
 *
 * @param currentPopulation - Current evolved population.
 * @param payload - Playback start viewport payload.
 * @param createPopulationRenderState - Callback that builds initial simulation state.
 * @returns Playback runtime state and deterministic RNG.
 */
export function beginWorkerPlaybackSession(options: {
  currentPopulation: Network[];
  payload: WorkerStartPlaybackMessage['payload'];
  createPopulationRenderState: (
    networks: Network[],
    rng: ReturnType<typeof createXorshift32>,
    initialVisibleWorldWidthPx: number,
    initialVisibleWorldHeightPx: number,
  ) => WorkerPlaybackState;
}): {
  currentPlaybackState: WorkerPlaybackState;
  currentPlaybackRng: ReturnType<typeof createXorshift32>;
  playbackWinnerIndex: number;
} {
  const { currentPopulation, payload, createPopulationRenderState } = options;
  const playbackRng = createXorshift32(0xabcdef01);
  const currentPlaybackState = createPopulationRenderState(
    currentPopulation,
    playbackRng,
    payload.visibleWorldWidthPx,
    payload.visibleWorldHeightPx,
  );

  return {
    currentPlaybackState,
    currentPlaybackRng: playbackRng,
    playbackWinnerIndex: -1,
  };
}

/**
 * Processes one worker playback-step request including completion/finalization logic.
 *
 * Educational note:
 * One playback request may advance multiple simulation steps. This lets the
 * host trade visual smoothness against throughput while keeping the worker in
 * control of simulation correctness, winner selection, and packed snapshot
 * publishing.
 *
 * @param options - Playback step dependencies and mutable runtime state.
 * @returns Updated playback runtime state after processing this step.
 */
export function processWorkerPlaybackStep(options: {
  playbackStepPayload: WorkerRequestPlaybackStepMessage['payload'];
  currentPlaybackState: WorkerPlaybackState;
  currentPlaybackRng: ReturnType<typeof createXorshift32>;
  currentPopulation: Network[];
  neatRuntime: Neat | undefined;
  stepPopulationFrame: (
    renderState: WorkerPlaybackState,
    rng: ReturnType<typeof createXorshift32>,
    difficultyProfile: SharedDifficultyProfile,
  ) => number;
  createPlaybackSnapshot: (
    playbackState: WorkerPlaybackState,
  ) => WorkerPlaybackFrameSnapshot;
  resolvePlaybackSnapshotTransferList: (
    snapshot: WorkerPlaybackFrameSnapshot,
  ) => Transferable[];
  postWorkerMessage: (
    workerMessage: WorkerResponseMessage,
    transferList?: Transferable[],
  ) => void;
}): {
  currentPlaybackState: WorkerPlaybackState | undefined;
  currentPlaybackRng: ReturnType<typeof createXorshift32> | undefined;
  currentPopulation: Network[];
  playbackWinnerIndex: number;
} {
  const {
    playbackStepPayload,
    currentPlaybackState,
    currentPlaybackRng,
    currentPopulation,
    neatRuntime,
    stepPopulationFrame,
    createPlaybackSnapshot,
    resolvePlaybackSnapshotTransferList,
    postWorkerMessage,
  } = options;

  currentPlaybackState.visibleWorldWidthPx =
    playbackStepPayload.visibleWorldWidthPx;
  currentPlaybackState.visibleWorldHeightPx =
    playbackStepPayload.visibleWorldHeightPx;
  const simulationSteps = Math.max(
    1,
    Math.trunc(playbackStepPayload.simulationSteps),
  );
  let executedSimulationSteps = 0;
  let totalActivationCalls = 0;

  for (
    let simulationStepIndex = 0;
    simulationStepIndex < simulationSteps &&
    hasAliveBirds(currentPlaybackState.birds);
    simulationStepIndex++
  ) {
    const difficultyProfile = resolveAdaptiveDifficultyProfile(
      resolveLeaderPipesPassed(currentPlaybackState.birds),
      1,
    );
    totalActivationCalls += stepPopulationFrame(
      currentPlaybackState,
      currentPlaybackRng,
      difficultyProfile,
    );
    executedSimulationSteps += 1;
  }

  const activationCallsPerFrame =
    executedSimulationSteps > 0
      ? totalActivationCalls / executedSimulationSteps
      : 0;
  const instrumentationPayload = FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION
    ? {
        activationCallsPerFrame,
        simulationStepsPerRaf: executedSimulationSteps,
      }
    : undefined;

  const snapshot = createPlaybackSnapshot(currentPlaybackState);
  const snapshotTransferList = resolvePlaybackSnapshotTransferList(snapshot);

  if (hasAliveBirds(currentPlaybackState.birds)) {
    postWorkerMessage(
      {
        type: 'playback-step',
        payload: {
          requestId: playbackStepPayload.requestId,
          snapshot,
          instrumentation: instrumentationPayload,
          done: false,
        },
      },
      snapshotTransferList,
    );
    return {
      currentPlaybackState,
      currentPlaybackRng,
      currentPopulation,
      playbackWinnerIndex: -1,
    };
  }

  const playbackWinnerIndex = resolveFramePrimaryWinnerIndex(
    currentPlaybackState.birds,
    false,
  );
  const winnerBird =
    playbackWinnerIndex >= 0
      ? currentPlaybackState.birds[playbackWinnerIndex]
      : undefined;

  if (winnerBird && currentPopulation.length > 0) {
    currentPopulation[0] = winnerBird.network.clone();
    if (neatRuntime) {
      const runtimeNeat = neatRuntime as unknown as { population?: Network[] };
      if (
        Array.isArray(runtimeNeat.population) &&
        runtimeNeat.population.length > 0
      ) {
        runtimeNeat.population[0] = winnerBird.network.clone();
      }
    }
  }

  const averagePipesPassed =
    currentPlaybackState.birds.reduce(
      (totalPipesPassed, bird) => totalPipesPassed + bird.pipesPassed,
      0,
    ) / Math.max(1, currentPlaybackState.birds.length);
  const sortedFramesSurvived = currentPlaybackState.birds
    .map((bird) => bird.framesSurvived)
    .toSorted((leftFrames, rightFrames) => leftFrames - rightFrames);
  const p90FrameIndex = Math.min(
    sortedFramesSurvived.length - 1,
    Math.floor(sortedFramesSurvived.length * 0.9),
  );
  const p90FramesSurvived =
    sortedFramesSurvived.length > 0 ? sortedFramesSurvived[p90FrameIndex] : 0;

  // Step 4: Downshift future browser generations once this architecture has clearly solved pipes.
  maybeDownshiftSuccessfulBrowserPopulation(
    neatRuntime,
    winnerBird?.pipesPassed ?? 0,
  );

  postWorkerMessage(
    {
      type: 'playback-step',
      payload: {
        requestId: playbackStepPayload.requestId,
        snapshot,
        instrumentation: instrumentationPayload,
        done: true,
        averagePipesPassed,
        p90FramesSurvived,
        winnerPipesPassed: winnerBird?.pipesPassed ?? 0,
        winnerFramesSurvived: winnerBird?.framesSurvived ?? 0,
      },
    },
    snapshotTransferList,
  );

  return {
    currentPlaybackState: undefined,
    currentPlaybackRng: undefined,
    currentPopulation,
    playbackWinnerIndex,
  };
}

/**
 * Downshifts the worker's future browser population budget after a successful playback run.
 *
 * The current generation has already been evaluated, so the savings apply to
 * future generations only. This keeps the demo responsive once an architecture
 * has already demonstrated that it can clear the live pipe target.
 *
 * @param neatRuntime - Worker-local NEAT runtime.
 * @param winnerPipesPassed - Winning playback pipe count for the completed generation.
 * @returns Nothing.
 */
function maybeDownshiftSuccessfulBrowserPopulation(
  neatRuntime: Neat | undefined,
  winnerPipesPassed: number,
): void {
  if (!neatRuntime || winnerPipesPassed < FLAPPY_BROWSER_SUCCESS_PIPE_TARGET) {
    return;
  }

  const runtimeNeat = neatRuntime as unknown as {
    options?: {
      popsize?: number;
      elitism?: number;
    };
  };
  const currentPopulationSize = Math.max(
    0,
    runtimeNeat.options?.popsize ?? 0,
  );
  if (
    !runtimeNeat.options ||
    currentPopulationSize <= FLAPPY_BROWSER_SUCCESS_DOWNSHIFT_POPULATION_SIZE
  ) {
    return;
  }

  runtimeNeat.options.popsize =
    FLAPPY_BROWSER_SUCCESS_DOWNSHIFT_POPULATION_SIZE;
  runtimeNeat.options.elitism = Math.min(
    FLAPPY_BROWSER_SUCCESS_DOWNSHIFT_ELITISM_COUNT,
    FLAPPY_BROWSER_SUCCESS_DOWNSHIFT_POPULATION_SIZE,
  );
}
