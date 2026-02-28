/// <reference lib="webworker" />

import { Neat, methods } from '../../../src/neataptic';
import Architect from '../../../src/architecture/architect';
import Network from '../../../src/architecture/network';
import { evaluateFlappyFitness } from './flappyEvaluation';
import { createXorshift32 } from './rng';
import {
  clamp,
  createBirdColor,
  hasAliveBirds,
  resolveDifficultyProfile,
  resolveFlapDecision,
  resolveFramePrimaryWinnerIndex,
  resolveLeaderPipesPassed,
  resolveNextSpawnGapCenterY,
  resolveNextSpawnGapSize,
  resolveNextSpawnIntervalFrames,
  resolveObservationVector,
  resolvePipeSpawnXPx,
  sampleGapCenterY,
} from './browser-entry.utils';
import {
  FLAPPY_BIRD_RADIUS_PX,
  FLAPPY_BIRD_X_PX,
  FLAPPY_CONTROL_SUBSTEPS_PER_FRAME,
  FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION,
  FLAPPY_FLAP_VELOCITY_PX_PER_FRAME,
  FLAPPY_GRAVITY_PX_PER_FRAME2,
  FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
  FLAPPY_NETWORK_HIDDEN_LAYER_SIZES,
  FLAPPY_NETWORK_INPUT_SIZE,
  FLAPPY_NETWORK_OUTPUT_SIZE,
  FLAPPY_PIPE_WIDTH_PX,
  FLAPPY_PIPE_SPAWN_INTERVAL_FRAMES,
  FLAPPY_WORLD_HEIGHT_PX,
} from './constants';

type SerializedNetwork = Record<string, unknown>;

interface WorkerPopulationPipe {
  id: number;
  xPx: number;
  gapCenterYPx: number;
  gapSizePx: number;
}

interface WorkerPopulationBird {
  network: Network;
  color: string;
  yPx: number;
  velocityYPxPerFrame: number;
  pipesPassed: number;
  framesSurvived: number;
  passedPipeIds: Set<number>;
  done: boolean;
  doneReason?: 'collision' | 'out_of_bounds';
}

interface WorkerPlaybackState {
  frameIndex: number;
  visibleWorldWidthPx: number;
  nextPipeId: number;
  lastSpawnedPipeGapPx: number;
  lastSpawnedPipeGapCenterYPx: number;
  lastSpawnedPipeSpawnIntervalFrames: number;
  framesUntilNextPipeSpawn: number;
  pipes: WorkerPopulationPipe[];
  birds: WorkerPopulationBird[];
}

interface WorkerFrameBirdSnapshot {
  color: string;
  yPx: number;
  pipesPassed: number;
  framesSurvived: number;
  done: boolean;
}

interface WorkerFramePipeSnapshot {
  id: number;
  xPx: number;
  gapCenterYPx: number;
  gapSizePx: number;
}

interface WorkerPlaybackFrameSnapshot {
  frameIndex: number;
  visibleWorldWidthPx: number;
  pipes: WorkerFramePipeSnapshot[];
  birds: WorkerFrameBirdSnapshot[];
}

interface WorkerInitMessage {
  type: 'init';
  payload: {
    populationSize: number;
    elitismCount: number;
    rngSeed: number;
  };
}

interface WorkerRequestGenerationMessage {
  type: 'request-generation';
}

interface WorkerStartPlaybackMessage {
  type: 'start-playback';
  payload: {
    visibleWorldWidthPx: number;
  };
}

interface WorkerRequestPlaybackStepMessage {
  type: 'request-playback-step';
  payload: {
    simulationSteps: number;
    visibleWorldWidthPx: number;
  };
}

interface WorkerStopMessage {
  type: 'stop';
}

type WorkerRequestMessage =
  | WorkerInitMessage
  | WorkerRequestGenerationMessage
  | WorkerStartPlaybackMessage
  | WorkerRequestPlaybackStepMessage
  | WorkerStopMessage;

interface WorkerGenerationReadyMessage {
  type: 'generation-ready';
  payload: {
    generation: number;
    bestFitness: number;
    bestNetworkJson?: SerializedNetwork;
  };
}

interface WorkerPlaybackStepMessage {
  type: 'playback-step';
  payload: {
    snapshot: WorkerPlaybackFrameSnapshot;
    instrumentation?: {
      activationCallsPerFrame: number;
      simulationStepsPerRaf: number;
    };
    done: boolean;
    averagePipesPassed?: number;
    p90FramesSurvived?: number;
    winnerPipesPassed?: number;
    winnerFramesSurvived?: number;
  };
}

interface WorkerErrorMessage {
  type: 'error';
  payload: {
    message: string;
  };
}

type WorkerResponseMessage =
  | WorkerGenerationReadyMessage
  | WorkerPlaybackStepMessage
  | WorkerErrorMessage;

let stopped = false;
let neatRuntime: Neat | undefined;
let currentPopulation: Network[] = [];
let currentPlaybackState: WorkerPlaybackState | undefined;
let currentPlaybackRng: ReturnType<typeof createXorshift32> | undefined;
let playbackWinnerIndex = -1;
let initializationPromise: Promise<void> | undefined;

self.onmessage = (event: MessageEvent<WorkerRequestMessage>) => {
  const workerMessage = event.data;

  if (workerMessage.type === 'stop') {
    stopped = true;
    return;
  }

  if (workerMessage.type === 'init') {
    initializationPromise = initializeRuntime(workerMessage.payload).catch(
      (error: unknown) => {
        postWorkerMessage({
          type: 'error',
          payload: {
            message: String((error as Error)?.message ?? error),
          },
        });
        throw error;
      },
    );
    void initializationPromise.catch(() => {
      postWorkerMessage({
        type: 'error',
        payload: {
          message: 'Failed to initialize Flappy evolution worker runtime.',
        },
      });
    });
    return;
  }

  if (workerMessage.type === 'request-generation') {
    void evolveAndPublishGeneration().catch((error: unknown) => {
      postWorkerMessage({
        type: 'error',
        payload: {
          message: String((error as Error)?.message ?? error),
        },
      });
    });
    return;
  }

  if (workerMessage.type === 'start-playback') {
    if (!Array.isArray(currentPopulation) || currentPopulation.length === 0) {
      postWorkerMessage({
        type: 'error',
        payload: {
          message:
            'Cannot start playback before a generation is available. Request a generation first.',
        },
      });
      return;
    }

    currentPlaybackState = createPopulationRenderState(
      currentPopulation,
      createXorshift32(0xabcdef01),
      workerMessage.payload.visibleWorldWidthPx,
    );
    currentPlaybackRng = createXorshift32(0xabcdef01);
    playbackWinnerIndex = -1;
    return;
  }

  if (workerMessage.type === 'request-playback-step') {
    if (!currentPlaybackState) {
      postWorkerMessage({
        type: 'error',
        payload: {
          message:
            'Playback step requested before playback was initialized. Send start-playback first.',
        },
      });
      return;
    }

    processPlaybackStep(workerMessage.payload);
  }
};

async function initializeRuntime(
  initPayload: WorkerInitMessage['payload'],
): Promise<void> {
  const inputSize = FLAPPY_NETWORK_INPUT_SIZE;
  const outputSize = FLAPPY_NETWORK_OUTPUT_SIZE;

  neatRuntime = new Neat(inputSize, outputSize, () => 0, {
    popsize: initPayload.populationSize,
    elitism: initPayload.elitismCount,
    mutationRate: 0.75,
    mutationAmount: 2,
    mutation: methods.mutation.FFW,
    network: Architect.perceptron(
      inputSize,
      ...FLAPPY_NETWORK_HIDDEN_LAYER_SIZES,
      outputSize,
    ),
    speciation: true,
    multiObjective: { enabled: false },
    novelty: { enabled: false },
  });

  neatRuntime.fitness = (network) =>
    evaluateFlappyFitness(network, {
      enableEarlyTermination: true,
      maxFrames: FLAPPY_PIPE_SPAWN_INTERVAL_FRAMES,
    });

  neatRuntime.restoreRNGState(initPayload.rngSeed);
}

async function evolveAndPublishGeneration(): Promise<void> {
  if (initializationPromise) {
    await initializationPromise;
  }

  if (!neatRuntime || stopped) {
    throw new Error('Evolution worker runtime is not initialized.');
  }

  const bestNetwork = (await neatRuntime.evolve()) as Network;
  const runtimeNeat = neatRuntime as unknown as { population?: Network[] };
  currentPopulation = Array.isArray(runtimeNeat.population)
    ? runtimeNeat.population
    : [bestNetwork];

  const generationPayload: WorkerGenerationReadyMessage = {
    type: 'generation-ready',
    payload: {
      generation: neatRuntime.generation,
      bestFitness: Number(bestNetwork.score ?? 0),
      bestNetworkJson: bestNetwork.toJSON(),
    },
  };

  postWorkerMessage(generationPayload);
}

function processPlaybackStep(
  playbackStepPayload: WorkerRequestPlaybackStepMessage['payload'],
): void {
  if (!currentPlaybackState) return;
  if (!currentPlaybackRng) {
    throw new Error(
      'Playback random source is unavailable. Start playback first.',
    );
  }

  currentPlaybackState.visibleWorldWidthPx =
    playbackStepPayload.visibleWorldWidthPx;
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
    const difficultyProfile = resolveDifficultyProfile(
      resolveLeaderPipesPassed(currentPlaybackState.birds),
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
    return;
  }

  playbackWinnerIndex = resolveFramePrimaryWinnerIndex(
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

  currentPlaybackState = undefined;
  currentPlaybackRng = undefined;
}

function createPlaybackSnapshot(
  playbackState: WorkerPlaybackState,
): WorkerPlaybackFrameSnapshot {
  return {
    frameIndex: playbackState.frameIndex,
    visibleWorldWidthPx: playbackState.visibleWorldWidthPx,
    pipes: playbackState.pipes.map((pipe) => ({
      id: pipe.id,
      xPx: pipe.xPx,
      gapCenterYPx: pipe.gapCenterYPx,
      gapSizePx: pipe.gapSizePx,
    })),
    birds: playbackState.birds.map((bird) => ({
      color: bird.color,
      yPx: bird.yPx,
      pipesPassed: bird.pipesPassed,
      framesSurvived: bird.framesSurvived,
      done: bird.done,
    })),
  };
}

function postWorkerMessage(workerMessage: WorkerResponseMessage): void {
  self.postMessage(workerMessage);
}

function createPopulationRenderState(
  networks: Network[],
  rng: ReturnType<typeof createXorshift32>,
  initialVisibleWorldWidthPx: number,
): WorkerPlaybackState {
  const initialDifficultyProfile = resolveDifficultyProfile(0);
  const initialGapCenterYPx = sampleGapCenterY(rng);
  const initialGapSizePx = resolveNextSpawnGapSize(
    undefined,
    initialDifficultyProfile,
    rng,
  );
  const initialSpawnIntervalFrames = resolveNextSpawnIntervalFrames(
    undefined,
    initialDifficultyProfile,
  );
  const birds = networks.map((network, networkIndex) => ({
    network,
    color: createBirdColor(networkIndex, networks.length),
    yPx: FLAPPY_WORLD_HEIGHT_PX * 0.5,
    velocityYPxPerFrame: 0,
    pipesPassed: 0,
    framesSurvived: 0,
    passedPipeIds: new Set<number>(),
    done: false,
  }));

  return {
    frameIndex: 0,
    visibleWorldWidthPx: initialVisibleWorldWidthPx,
    nextPipeId: 2,
    lastSpawnedPipeGapPx: initialGapSizePx,
    lastSpawnedPipeGapCenterYPx: initialGapCenterYPx,
    lastSpawnedPipeSpawnIntervalFrames: initialSpawnIntervalFrames,
    framesUntilNextPipeSpawn: initialSpawnIntervalFrames,
    pipes: [
      {
        id: 1,
        xPx: resolvePipeSpawnXPx(initialVisibleWorldWidthPx),
        gapCenterYPx: initialGapCenterYPx,
        gapSizePx: initialGapSizePx,
      },
    ],
    birds,
  };
}

function stepPopulationFrame(
  renderState: WorkerPlaybackState,
  rng: ReturnType<typeof createXorshift32>,
  difficultyProfile: {
    pipeGapPx: number;
    pipeSpeedPxPerFrame: number;
    pipeSpawnIntervalFrames: number;
  },
): number {
  const controlSubstepCount = Math.max(1, FLAPPY_CONTROL_SUBSTEPS_PER_FRAME);
  const controlSubstepDelta = 1 / controlSubstepCount;
  let activationCallsThisFrame = 0;

  renderState.birds.forEach((bird) => {
    if (bird.done) return;
    bird.framesSurvived += 1;
  });

  for (
    let controlSubstepIndex = 0;
    controlSubstepIndex < controlSubstepCount;
    controlSubstepIndex++
  ) {
    renderState.birds.forEach((bird) => {
      if (bird.done) return;

      const observation = resolveObservationVector(
        bird.yPx,
        bird.velocityYPxPerFrame,
        renderState.pipes,
        renderState.visibleWorldWidthPx,
        difficultyProfile,
        renderState.lastSpawnedPipeSpawnIntervalFrames,
      );
      const outputs = bird.network.activate(observation) as unknown;
      if (FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION) {
        activationCallsThisFrame += 1;
      }
      const shouldFlap = resolveFlapDecision(outputs);

      if (shouldFlap) {
        bird.velocityYPxPerFrame = FLAPPY_FLAP_VELOCITY_PX_PER_FRAME;
      }
    });

    renderState.birds.forEach((bird) => {
      if (bird.done) return;

      bird.velocityYPxPerFrame = clamp(
        bird.velocityYPxPerFrame +
          FLAPPY_GRAVITY_PX_PER_FRAME2 * controlSubstepDelta,
        -Infinity,
        FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
      );
      bird.yPx += bird.velocityYPxPerFrame * controlSubstepDelta;
    });

    renderState.pipes.forEach((pipe) => {
      pipe.xPx -= difficultyProfile.pipeSpeedPxPerFrame * controlSubstepDelta;
    });
    renderState.pipes = renderState.pipes.filter(
      (pipe) => pipe.xPx + FLAPPY_PIPE_WIDTH_PX > 0,
    );

    renderState.framesUntilNextPipeSpawn -= controlSubstepDelta;
    if (renderState.framesUntilNextPipeSpawn <= 0) {
      const nextGapSizePx = resolveNextSpawnGapSize(
        renderState.lastSpawnedPipeGapPx,
        difficultyProfile,
        rng,
      );
      const nextSpawnIntervalFrames = resolveNextSpawnIntervalFrames(
        renderState.lastSpawnedPipeSpawnIntervalFrames,
        difficultyProfile,
      );
      const nextGapCenterYPx = resolveNextSpawnGapCenterY(
        renderState.lastSpawnedPipeGapCenterYPx,
        rng,
      );
      renderState.pipes.push({
        id: renderState.nextPipeId++,
        xPx: resolvePipeSpawnXPx(renderState.visibleWorldWidthPx),
        gapCenterYPx: nextGapCenterYPx,
        gapSizePx: nextGapSizePx,
      });
      renderState.lastSpawnedPipeGapPx = nextGapSizePx;
      renderState.lastSpawnedPipeGapCenterYPx = nextGapCenterYPx;
      renderState.lastSpawnedPipeSpawnIntervalFrames = nextSpawnIntervalFrames;
      renderState.framesUntilNextPipeSpawn += nextSpawnIntervalFrames;
    }

    renderState.birds.forEach((bird) => {
      if (bird.done) return;

      const birdTop = bird.yPx - FLAPPY_BIRD_RADIUS_PX;
      const birdBottom = bird.yPx + FLAPPY_BIRD_RADIUS_PX;

      if (birdTop <= 0 || birdBottom >= FLAPPY_WORLD_HEIGHT_PX) {
        bird.done = true;
        bird.doneReason = 'out_of_bounds';
        return;
      }

      const birdLeft = FLAPPY_BIRD_X_PX - FLAPPY_BIRD_RADIUS_PX;
      const birdRight = FLAPPY_BIRD_X_PX + FLAPPY_BIRD_RADIUS_PX;

      for (const pipe of renderState.pipes) {
        const pipeLeft = pipe.xPx;
        const pipeRight = pipe.xPx + FLAPPY_PIPE_WIDTH_PX;
        const overlapsHorizontally =
          birdRight >= pipeLeft && birdLeft <= pipeRight;

        if (overlapsHorizontally) {
          const gapHalf = pipe.gapSizePx * 0.5;
          const gapTop = pipe.gapCenterYPx - gapHalf;
          const gapBottom = pipe.gapCenterYPx + gapHalf;
          const isInsideGap = birdTop >= gapTop && birdBottom <= gapBottom;

          if (!isInsideGap) {
            bird.done = true;
            bird.doneReason = 'collision';
            break;
          }
        }

        if (pipeRight < FLAPPY_BIRD_X_PX && !bird.passedPipeIds.has(pipe.id)) {
          bird.passedPipeIds.add(pipe.id);
          bird.pipesPassed += 1;
        }
      }
    });
  }

  renderState.frameIndex += 1;
  return activationCallsThisFrame;
}
