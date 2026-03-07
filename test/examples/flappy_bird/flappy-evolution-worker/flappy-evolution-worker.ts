/// <reference lib="webworker" />

import { Neat } from '../../../../src/neataptic';
import Network from '../../../../src/architecture/network';
import { createXorshift32 } from '../rng';
import {
  commitObservationMemoryStep,
  resolveFlapDecision,
  resolveObservationVector,
} from '../browser-entry/browser-entry.observation.utils';
import {
  createBirdColor,
  resolveDifficultyProfile,
  sampleGapCenterY,
} from '../browser-entry/browser-entry.spawn.utils';
import type {
  WorkerHeuristicObservationFeatures,
  WorkerInitMessage,
  WorkerPlaybackFrameSnapshot,
  WorkerPopulationPipe,
  WorkerPlaybackState,
  WorkerRequestMessage,
  WorkerRequestPlaybackStepMessage,
  WorkerResponseMessage,
} from './flappy-evolution-worker.types';
import { createInitializedWorkerRuntime } from './flappy-evolution-worker.runtime.service';
import { FLAPPY_BIRD_VIEWPORT_X_RATIO } from '../constants/constants';
import {
  FLAPPY_BIRD_RADIUS_PX,
  FLAPPY_BIRD_X_PX,
  FLAPPY_PIPE_COLLISION_ENTRANCE_EXPAND_PX,
  FLAPPY_PIPE_COLLISION_SIDE_EXPAND_PX,
  FLAPPY_CONTROL_SUBSTEPS_PER_FRAME,
  FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION,
  FLAPPY_FLAP_VELOCITY_PX_PER_FRAME,
  FLAPPY_GRAVITY_PX_PER_FRAME2,
  FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
  FLAPPY_PIPE_WIDTH_PX,
  FLAPPY_WORLD_HEIGHT_PX,
} from '../constants/constants';
import {
  FLAPPY_WORKER_GEN0_PRETRAIN_BATCH_SIZE,
  FLAPPY_WORKER_GEN0_PRETRAIN_BIAS_NOISE_STDDEV,
  FLAPPY_WORKER_GEN0_PRETRAIN_ITERATIONS,
  FLAPPY_WORKER_GEN0_PRETRAIN_RATE,
  FLAPPY_WORKER_GEN0_PRETRAIN_SAMPLE_COUNT,
  FLAPPY_WORKER_GEN0_PRETRAIN_VISIBLE_WORLD_WIDTH_PX,
  FLAPPY_WORKER_GEN0_PRETRAIN_WEIGHT_NOISE_STDDEV,
} from './flappy-evolution-worker.constants';
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
import { createSharedObservationMemoryState } from '../flappy.simulation.shared.utils';
import {
  createWorkerPopulationRenderState,
  stepWorkerPopulationFrame,
} from './flappy-evolution-worker.simulation.utils';

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
    warmStartGenerationZeroIfNeeded,
    setCurrentPopulation: (nextPopulation) => {
      currentPopulation = nextPopulation;
    },
  });

  postWorkerMessage(generationPayload);
}

/**
 * Applies a one-time generation-0 warm-start to improve initial demo quality.
 *
 * Educational note:
 * NEAT in this setup starts from cloned copies of one seed architecture.
 * A short supervised warm-up can reduce the "all-random" first-generation
 * behavior while preserving evolutionary search in later generations.
 *
 * Strategy:
 * 1) Build heuristic-labeled synthetic observations.
 * 2) Train one template network quickly.
 * 3) Copy template parameters to each genome with small Gaussian noise.
 *
 * @param neatController - Initialized NEAT runtime.
 * @returns Promise resolved when warm-start decision completes.
 */
async function warmStartGenerationZeroIfNeeded(
  neatController: Neat,
): Promise<void> {
  // Step 1: Exit fast when warm-start is already processed.
  if (generationZeroWarmStartApplied) return;

  // Step 2: Warm-start only generation 0 to avoid skewing later evolution.
  if (neatController.generation !== 0) {
    generationZeroWarmStartApplied = true;
    return;
  }

  // Step 3: Validate population availability.
  const population = neatController.population;
  if (!Array.isArray(population) || population.length === 0) {
    generationZeroWarmStartApplied = true;
    return;
  }

  // Step 1: Build a small synthetic dataset labeled by a simple heuristic.
  // The goal is NOT to solve Flappy here, only to avoid a totally random first generation.
  const warmStartRng = createXorshift32(workerInitSeed ^ 0x9e37_79b9);
  const trainingSet = buildHeuristicPretrainSet(
    warmStartRng,
    FLAPPY_WORKER_GEN0_PRETRAIN_SAMPLE_COUNT,
  );

  // Step 2: Train a single template network (fast), then seed the whole population from it.
  const templateNetwork = population[0]?.clone();
  if (!templateNetwork) {
    generationZeroWarmStartApplied = true;
    return;
  }

  try {
    templateNetwork.train(trainingSet, {
      iterations: FLAPPY_WORKER_GEN0_PRETRAIN_ITERATIONS,
      rate: FLAPPY_WORKER_GEN0_PRETRAIN_RATE,
      batchSize: FLAPPY_WORKER_GEN0_PRETRAIN_BATCH_SIZE,
      optimizer: 'adam',
      // Keep the training loop simple and deterministic.
      mixedPrecision: false,
    });
  } catch {
    // If training fails for any reason (e.g., incompatible topology), fall back to pure noise seeding.
  }

  // Step 3: Copy trained weights/biases into each genome with small noise for diversity.
  for (const genome of population) {
    applyTemplateWeightsWithNoise(genome, templateNetwork, warmStartRng, {
      weightStdDev: FLAPPY_WORKER_GEN0_PRETRAIN_WEIGHT_NOISE_STDDEV,
      biasStdDev: FLAPPY_WORKER_GEN0_PRETRAIN_BIAS_NOISE_STDDEV,
    });
    (genome as unknown as { score?: number }).score = undefined;
  }

  generationZeroWarmStartApplied = true;
}

/**
 * Builds synthetic supervised samples for generation-0 behavior cloning.
 *
 * Educational note:
 * Samples are generated from plausible Flappy states and passed through the
 * exact observation encoder used during runtime. This keeps feature semantics
 * aligned between pretraining and live simulation.
 *
 * @param rng - Deterministic random source.
 * @param sampleCount - Requested number of synthetic samples.
 * @returns Supervised dataset of { input, output } pairs.
 */
function buildHeuristicPretrainSet(
  rng: ReturnType<typeof createXorshift32>,
  sampleCount: number,
): Array<{ input: number[]; output: number[] }> {
  // Step 1: Resolve bounded sample count and a baseline difficulty profile.
  const clampedSampleCount = Math.max(8, Math.trunc(sampleCount));
  const difficultyProfile = resolveDifficultyProfile(0);
  const trainingSet: Array<{ input: number[]; output: number[] }> = [];

  // Step 2: Sample state tuples, encode observations, and attach heuristic labels.
  for (let sampleIndex = 0; sampleIndex < clampedSampleCount; sampleIndex++) {
    const pipeGapPx = difficultyProfile.pipeGapPx;
    const gapHalfPx = pipeGapPx * 0.5;

    const birdYPx = rng.nextFloat01() * FLAPPY_WORLD_HEIGHT_PX;
    const velocityYPxPerFrame =
      (rng.nextFloat01() * 2 - 1) * FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME;

    const pipeXMinPx = FLAPPY_BIRD_X_PX + 40;
    const pipeXMaxPx =
      FLAPPY_BIRD_X_PX +
      FLAPPY_WORKER_GEN0_PRETRAIN_VISIBLE_WORLD_WIDTH_PX * 0.92;
    const pipeXSamplePx =
      pipeXMinPx + rng.nextFloat01() * Math.max(1, pipeXMaxPx - pipeXMinPx);

    const minGapCenterYPx = gapHalfPx;
    const maxGapCenterYPx = Math.max(
      minGapCenterYPx,
      FLAPPY_WORLD_HEIGHT_PX - gapHalfPx,
    );
    const gapCenterYPx =
      minGapCenterYPx + rng.nextFloat01() * (maxGapCenterYPx - minGapCenterYPx);

    const pipes: WorkerPopulationPipe[] = [
      {
        id: 1,
        xPx: pipeXSamplePx,
        gapCenterYPx,
        gapSizePx: pipeGapPx,
      },
    ];

    const observationMemoryState = createSharedObservationMemoryState();
    const observation = resolveObservationVector(
      birdYPx,
      velocityYPxPerFrame,
      pipes,
      FLAPPY_WORKER_GEN0_PRETRAIN_VISIBLE_WORLD_WIDTH_PX,
      FLAPPY_WORLD_HEIGHT_PX,
      difficultyProfile,
      difficultyProfile.pipeSpawnIntervalFrames,
      observationMemoryState,
    );

    const shouldFlap = resolveHeuristicTeacherFlapDecision(
      observation.observationFeatures,
    );
    trainingSet.push({
      input: observation.observationVector,
      output: shouldFlap ? [0, 1] : [1, 0],
    });
  }

  // Step 3: Return immutable training pairs consumed by `Network.train`.
  return trainingSet;
}

/**
 * Heuristic teacher policy used to label synthetic pretraining samples.
 *
 * Educational note:
 * The teacher is intentionally simple and interpretable, acting as a rough
 * "stay near next gap center" controller. The objective is biasing the initial
 * policy away from chaotic behavior, not producing expert trajectories.
 *
 * @param features - Structured observation features for one synthetic state.
 * @returns True when the teacher says "flap".
 */
function resolveHeuristicTeacherFlapDecision(
  features: WorkerHeuristicObservationFeatures,
): boolean {
  // Heuristic intent:
  // - If the bird is below the upcoming gap center, flap to correct upward.
  // - Avoid spamming flap when already moving upward quickly.
  // - Prefer corrections closer to the pipe (or when urgency is high).
  const isBelowNextGapCenter = features.normalizedDeltaToNextGap > 0.035;
  const isNotAlreadyRisingFast = features.normalizedVelocity > -0.25;
  const isNearGapEntry = features.normalizedFramesToGapEntry < 0.8;
  const isUrgent = features.normalizedEntryUrgency > 0.18;

  return (
    isBelowNextGapCenter &&
    isNotAlreadyRisingFast &&
    (isNearGapEntry || isUrgent)
  );
}

/**
 * Copies template parameters into a genome and injects small Gaussian noise.
 *
 * Educational note:
 * Copying one trained template without noise would collapse diversity. Adding
 * mild deterministic jitter keeps the population near a useful baseline while
 * preserving variation for selection and mutation.
 *
 * @param genome - Target genome to mutate in-place.
 * @param template - Trained template source network.
 * @param rng - Deterministic random source for noise sampling.
 * @param noise - Standard deviations for weight and bias perturbations.
 * @returns Nothing.
 */
function applyTemplateWeightsWithNoise(
  genome: Network,
  template: Network,
  rng: ReturnType<typeof createXorshift32>,
  noise: { weightStdDev: number; biasStdDev: number },
): void {
  // Step 1: Clamp noise scales to valid non-negative values.
  const weightStdDev = Math.max(0, noise.weightStdDev);
  const biasStdDev = Math.max(0, noise.biasStdDev);

  // Step 2: Copy node biases with additive Gaussian noise.
  const genomeNodes = genome.nodes;
  const templateNodes = template.nodes;
  const nodeCopyCount = Math.min(genomeNodes.length, templateNodes.length);
  for (let nodeIndex = 0; nodeIndex < nodeCopyCount; nodeIndex++) {
    const templateBias = templateNodes[nodeIndex]?.bias ?? 0;
    genomeNodes[nodeIndex].bias =
      templateBias + sampleGaussian(rng) * biasStdDev;
  }

  // Step 3: Copy connection weights with additive Gaussian noise.
  const genomeConnections = genome.connections;
  const templateConnections = template.connections;
  const connectionCopyCount = Math.min(
    genomeConnections.length,
    templateConnections.length,
  );
  for (
    let connectionIndex = 0;
    connectionIndex < connectionCopyCount;
    connectionIndex++
  ) {
    const templateWeight = templateConnections[connectionIndex]?.weight ?? 0;
    genomeConnections[connectionIndex].weight =
      templateWeight + sampleGaussian(rng) * weightStdDev;
  }
}

/**
 * Samples one standard-normal value using the Box–Muller transform.
 *
 * Educational note:
 * Uniform RNG is easy to produce, but Gaussian noise is usually a better fit
 * for small parameter perturbations because it concentrates probability around
 * zero while still occasionally exploring larger offsets.
 *
 * @param rng - Deterministic random source.
 * @returns One approximately standard-normal random value.
 */
function sampleGaussian(rng: ReturnType<typeof createXorshift32>): number {
  // Box–Muller transform (deterministic given rng).
  // Step 1: Avoid `log(0)` by flooring uniforms away from zero.
  const epsilon = 1e-12;
  const uniformA = Math.max(epsilon, rng.nextFloat01());
  const uniformB = Math.max(epsilon, rng.nextFloat01());

  // Step 2: Convert two uniforms into one normal sample.
  const magnitude = Math.sqrt(-2 * Math.log(uniformA));
  const angle = 2 * Math.PI * uniformB;
  return magnitude * Math.cos(angle);
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
