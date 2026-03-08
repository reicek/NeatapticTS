import type Network from '../../../../src/architecture/network';
import { createXorshift32 } from '../rng';
import {
  hasAliveBirds,
  resolveFramePrimaryWinnerIndex,
  resolveLeaderPipesPassed,
} from '../browser-entry/browser-entry.observation.utils';
import { FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION } from '../constants/constants';
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
import type { Neat } from '../../../../src/neataptic';

/**
 * Creates a fresh worker playback session state from the current evolved population.
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

  return {
    currentPlaybackState: createPopulationRenderState(
      currentPopulation,
      playbackRng,
      payload.visibleWorldWidthPx,
      payload.visibleWorldHeightPx,
    ),
    currentPlaybackRng: playbackRng,
    playbackWinnerIndex: -1,
  };
}

/**
 * Processes one worker playback-step request including completion/finalization logic.
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
  postWorkerMessage: (workerMessage: WorkerResponseMessage) => void;
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

  if (hasAliveBirds(currentPlaybackState.birds)) {
    postWorkerMessage({
      type: 'playback-step',
      payload: {
        snapshot,
        instrumentation: instrumentationPayload,
        done: false,
      },
    });
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

  postWorkerMessage({
    type: 'playback-step',
    payload: {
      snapshot,
      instrumentation: instrumentationPayload,
      done: true,
      averagePipesPassed,
      p90FramesSurvived,
      winnerPipesPassed: winnerBird?.pipesPassed ?? 0,
      winnerFramesSurvived: winnerBird?.framesSurvived ?? 0,
    },
  });

  return {
    currentPlaybackState: undefined,
    currentPlaybackRng: undefined,
    currentPopulation,
    playbackWinnerIndex,
  };
}
