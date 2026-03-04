/// <reference lib="webworker" />

import { Neat, methods } from '../../../src/neataptic';
import Architect from '../../../src/architecture/architect';
import Network from '../../../src/architecture/network';
import { evaluateFlappyFitness } from './flappyEvaluation';
import { createXorshift32 } from './rng';
import {
  clamp,
  commitObservationMemoryStep,
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
} from './browser-entry/browser-entry.utils';
import {
  createSharedObservationMemoryState,
  type SharedObservationMemoryState,
  type SharedObservationFeatures,
} from './flappy.simulation.shared.utils';
import { FLAPPY_BIRD_VIEWPORT_X_RATIO } from './constants/constants';
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
  FLAPPY_NETWORK_HIDDEN_LAYER_SIZES,
  FLAPPY_NETWORK_INPUT_SIZE,
  FLAPPY_NETWORK_OUTPUT_SIZE,
  FLAPPY_PIPE_WIDTH_PX,
  FLAPPY_PIPE_SPAWN_INTERVAL_FRAMES,
  FLAPPY_WORLD_HEIGHT_PX,
} from './constants/constants';

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
  observationMemoryState: SharedObservationMemoryState;
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
  visibleWorldHeightPx: number;
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

/**
 * Resolves the current left-edge of the visible world in world-space pixels.
 *
 * The browser renderer anchors the bird at a fixed viewport ratio, which
 * effectively shifts the camera. The simulation must use the same camera-left
 * when deciding when pipes have fully exited the screen; otherwise pipes will
 * disappear early when the camera-left becomes negative.
 *
 * @param visibleWorldWidthPx - Current visible world width.
 * @returns Left edge x-position in world coordinates.
 */
function resolveCameraLeftXPx(visibleWorldWidthPx: number): number {
  // Step 1: Guard against degenerate viewport widths.
  const clampedVisibleWorldWidthPx = Math.max(1, visibleWorldWidthPx);

  // Step 2: Project bird's anchored screen-x from viewport ratio.
  const desiredBirdScreenXPx =
    clampedVisibleWorldWidthPx * FLAPPY_BIRD_VIEWPORT_X_RATIO;

  // Step 3: Convert from screen anchor to world-space camera-left.
  return FLAPPY_BIRD_X_PX - desiredBirdScreenXPx;
}

interface WorkerPlaybackFrameSnapshot {
  frameIndex: number;
  visibleWorldWidthPx: number;
  visibleWorldHeightPx: number;
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
    visibleWorldHeightPx: number;
  };
}

interface WorkerRequestPlaybackStepMessage {
  type: 'request-playback-step';
  payload: {
    simulationSteps: number;
    visibleWorldWidthPx: number;
    visibleWorldHeightPx: number;
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
let workerInitSeed = 0;
let generationZeroWarmStartApplied = false;

const FLAPPY_GEN0_PRETRAIN_SAMPLE_COUNT = 512;
const FLAPPY_GEN0_PRETRAIN_VISIBLE_WORLD_WIDTH_PX = 720;
const FLAPPY_GEN0_PRETRAIN_ITERATIONS = 60;
const FLAPPY_GEN0_PRETRAIN_BATCH_SIZE = 32;
const FLAPPY_GEN0_PRETRAIN_RATE = 0.02;
const FLAPPY_GEN0_PRETRAIN_WEIGHT_NOISE_STDDEV = 0.08;
const FLAPPY_GEN0_PRETRAIN_BIAS_NOISE_STDDEV = 0.03;

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

  // Step 2: Handle stop requests first so long-running operations can observe it.
  if (workerMessage.type === 'stop') {
    stopped = true;
    return;
  }

  // Step 3: Initialize runtime once and capture initialization failures.
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

  // Step 4: Evolve one generation and publish generation-ready payload.
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

  // Step 5: Create a fresh playback simulation from the current evolved population.
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
      workerMessage.payload.visibleWorldHeightPx,
    );
    currentPlaybackRng = createXorshift32(0xabcdef01);
    playbackWinnerIndex = -1;
    return;
  }

  // Step 6: Advance playback only when a playback state already exists.
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
  const inputSize = FLAPPY_NETWORK_INPUT_SIZE;
  const outputSize = FLAPPY_NETWORK_OUTPUT_SIZE;

  // Step 2: Build the NEAT controller with feed-forward mutation policy.
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

  // Step 3: Attach a per-genome fitness function used by NEAT selection.
  neatRuntime.fitness = (network) =>
    evaluateFlappyFitness(network, {
      enableEarlyTermination: true,
      maxFrames: FLAPPY_PIPE_SPAWN_INTERVAL_FRAMES,
    });

  // Step 4: Restore deterministic RNG state for reproducible experiments.
  neatRuntime.restoreRNGState(initPayload.rngSeed);
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
  // Step 1: Ensure initialization is complete before evolution starts.
  if (initializationPromise) {
    await initializationPromise;
  }

  // Step 2: Guard against invalid lifecycle states.
  if (!neatRuntime || stopped) {
    throw new Error('Evolution worker runtime is not initialized.');
  }

  // Step 3: Apply one-time generation-0 warm-start before first evolve call.
  await warmStartGenerationZeroIfNeeded(neatRuntime);

  // Step 4: Evolve, then snapshot the new population for playback use.
  const bestNetwork = (await neatRuntime.evolve()) as Network;
  const runtimeNeat = neatRuntime as unknown as { population?: Network[] };
  currentPopulation = Array.isArray(runtimeNeat.population)
    ? runtimeNeat.population
    : [bestNetwork];

  // Step 5: Publish compact generation metadata and best network JSON.
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
    FLAPPY_GEN0_PRETRAIN_SAMPLE_COUNT,
  );

  // Step 2: Train a single template network (fast), then seed the whole population from it.
  const templateNetwork = population[0]?.clone();
  if (!templateNetwork) {
    generationZeroWarmStartApplied = true;
    return;
  }

  try {
    templateNetwork.train(trainingSet, {
      iterations: FLAPPY_GEN0_PRETRAIN_ITERATIONS,
      rate: FLAPPY_GEN0_PRETRAIN_RATE,
      batchSize: FLAPPY_GEN0_PRETRAIN_BATCH_SIZE,
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
      weightStdDev: FLAPPY_GEN0_PRETRAIN_WEIGHT_NOISE_STDDEV,
      biasStdDev: FLAPPY_GEN0_PRETRAIN_BIAS_NOISE_STDDEV,
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
      FLAPPY_BIRD_X_PX + FLAPPY_GEN0_PRETRAIN_VISIBLE_WORLD_WIDTH_PX * 0.92;
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
      FLAPPY_GEN0_PRETRAIN_VISIBLE_WORLD_WIDTH_PX,
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
  features: SharedObservationFeatures,
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
  // Step 1: Validate playback runtime state.
  if (!currentPlaybackState) return;
  if (!currentPlaybackRng) {
    throw new Error(
      'Playback random source is unavailable. Start playback first.',
    );
  }

  // Step 2: Apply host viewport and normalize requested simulation-step count.
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

  // Step 3: Run simulation until budget is exhausted or all birds are done.
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

  // Step 4: Build optional runtime instrumentation.
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

  // Step 5: Snapshot current frame for host rendering.
  const snapshot = createPlaybackSnapshot(currentPlaybackState);

  // Step 6: Fast path: continue playback while any bird is still alive.
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

  // Step 7: Resolve winner and optionally preserve it in the next population seed.
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

  // Step 8: Compute aggregate end-of-playback telemetry.
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

  // Step 9: Publish final playback payload and clear transient playback state.
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

/**
 * Creates a serializable snapshot of current playback state.
 *
 * Educational note:
 * Workers should send only plain structured-clone-safe objects. This method
 * strips runtime-only references (e.g., network instances, sets) and keeps just
 * renderer-relevant fields.
 *
 * @param playbackState - Current mutable playback state.
 * @returns Immutable frame snapshot for the host.
 */
function createPlaybackSnapshot(
  playbackState: WorkerPlaybackState,
): WorkerPlaybackFrameSnapshot {
  // Step 1: Project pipes and birds to lightweight render DTOs.
  return {
    frameIndex: playbackState.frameIndex,
    visibleWorldWidthPx: playbackState.visibleWorldWidthPx,
    visibleWorldHeightPx: playbackState.visibleWorldHeightPx,
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

/**
 * Posts a typed message from worker to host.
 *
 * @param workerMessage - Outbound worker response payload.
 * @returns Nothing.
 */
function postWorkerMessage(workerMessage: WorkerResponseMessage): void {
  self.postMessage(workerMessage);
}

/**
 * Creates initial playback state for a population of networks.
 *
 * Educational note:
 * This state builder mirrors runtime simulation assumptions (initial bird pose,
 * first pipe spawn, and spawn cadence memory), so playback behaves like a
 * continuation of normal game dynamics rather than a synthetic benchmark.
 *
 * @param networks - Population to visualize.
 * @param rng - Deterministic random source.
 * @param initialVisibleWorldWidthPx - Initial viewport width from host.
 * @param initialVisibleWorldHeightPx - Initial viewport height from host.
 * @returns Fresh mutable playback state.
 */
function createPopulationRenderState(
  networks: Network[],
  rng: ReturnType<typeof createXorshift32>,
  initialVisibleWorldWidthPx: number,
  initialVisibleWorldHeightPx: number,
): WorkerPlaybackState {
  // Step 1: Resolve initial difficulty-dependent spawn characteristics.
  const initialDifficultyProfile = resolveDifficultyProfile(0);
  const initialGapCenterYPx = sampleGapCenterY(
    rng,
    initialVisibleWorldHeightPx,
  );
  const initialGapSizePx = resolveNextSpawnGapSize(
    undefined,
    initialDifficultyProfile,
    rng,
  );
  const initialSpawnIntervalFrames = resolveNextSpawnIntervalFrames(
    undefined,
    initialDifficultyProfile,
  );

  // Step 2: Materialize birds from networks with fresh temporal memory.
  const birds = networks.map((network, networkIndex) => ({
    network,
    color: createBirdColor(networkIndex, networks.length),
    observationMemoryState: createSharedObservationMemoryState(),
    yPx: initialVisibleWorldHeightPx * 0.5,
    velocityYPxPerFrame: 0,
    pipesPassed: 0,
    framesSurvived: 0,
    passedPipeIds: new Set<number>(),
    done: false,
  }));

  // Step 3: Return full playback state with one initial pipe in front of the bird.
  return {
    frameIndex: 0,
    visibleWorldWidthPx: initialVisibleWorldWidthPx,
    visibleWorldHeightPx: initialVisibleWorldHeightPx,
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

/**
 * Advances the whole population simulation by one logical frame.
 *
 * Educational note:
 * A logical frame is subdivided into control/physics substeps. Each substep runs
 * policy inference, integrates motion, updates pipes, and evaluates collisions.
 * This makes control cadence finer while preserving frame-based telemetry.
 *
 * @param renderState - Mutable simulation state.
 * @param rng - Deterministic random source for spawn variation.
 * @param difficultyProfile - Active dynamic difficulty profile.
 * @returns Number of policy activation calls made in this frame.
 */
function stepPopulationFrame(
  renderState: WorkerPlaybackState,
  rng: ReturnType<typeof createXorshift32>,
  difficultyProfile: {
    pipeGapPx: number;
    pipeSpeedPxPerFrame: number;
    pipeSpawnIntervalFrames: number;
  },
): number {
  // Step 1: Resolve substep integration constants.
  const controlSubstepCount = Math.max(1, FLAPPY_CONTROL_SUBSTEPS_PER_FRAME);
  const controlSubstepDelta = 1 / controlSubstepCount;
  let activationCallsThisFrame = 0;

  // Step 2: Increment survival frames for alive birds once per logical frame.
  renderState.birds.forEach((bird) => {
    if (bird.done) return;
    bird.framesSurvived += 1;
  });

  // Step 3: Execute control/physics substeps.
  for (
    let controlSubstepIndex = 0;
    controlSubstepIndex < controlSubstepCount;
    controlSubstepIndex++
  ) {
    // Step 3.1: Run policy inference and optional flap impulse.
    renderState.birds.forEach((bird) => {
      if (bird.done) return;

      const observation = resolveObservationVector(
        bird.yPx,
        bird.velocityYPxPerFrame,
        renderState.pipes,
        renderState.visibleWorldWidthPx,
        renderState.visibleWorldHeightPx,
        difficultyProfile,
        renderState.lastSpawnedPipeSpawnIntervalFrames,
        bird.observationMemoryState,
      );
      const outputs = bird.network.activate(
        observation.observationVector,
      ) as unknown;
      if (FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION) {
        activationCallsThisFrame += 1;
      }
      const shouldFlap = resolveFlapDecision(outputs);
      commitObservationMemoryStep(
        bird.observationMemoryState,
        observation.observationFeatures,
        shouldFlap,
      );

      if (shouldFlap) {
        bird.velocityYPxPerFrame = FLAPPY_FLAP_VELOCITY_PX_PER_FRAME;
      }
    });

    // Step 3.2: Integrate bird motion with gravity and velocity clamp.
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

    // Step 3.3: Advance existing pipes in world space.
    renderState.pipes.forEach((pipe) => {
      pipe.xPx -= difficultyProfile.pipeSpeedPxPerFrame * controlSubstepDelta;
    });

    // Step 3.4: Culling uses camera-left so off-screen logic matches renderer framing.
    const cameraLeftXPx = resolveCameraLeftXPx(renderState.visibleWorldWidthPx);
    renderState.pipes = renderState.pipes.filter(
      (pipe) => pipe.xPx + FLAPPY_PIPE_WIDTH_PX > cameraLeftXPx,
    );

    // Step 3.5: Run spawn countdown and create a new pipe when interval elapses.
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
        renderState.visibleWorldHeightPx,
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

    // Step 3.6: Evaluate out-of-bounds, collisions, and pipes-passed counters.
    renderState.birds.forEach((bird) => {
      if (bird.done) return;

      const birdTop = bird.yPx - FLAPPY_BIRD_RADIUS_PX;
      const birdBottom = bird.yPx + FLAPPY_BIRD_RADIUS_PX;

      if (birdTop <= 0 || birdBottom >= renderState.visibleWorldHeightPx) {
        bird.done = true;
        bird.doneReason = 'out_of_bounds';
        return;
      }

      const birdLeft = FLAPPY_BIRD_X_PX - FLAPPY_BIRD_RADIUS_PX;
      const birdRight = FLAPPY_BIRD_X_PX + FLAPPY_BIRD_RADIUS_PX;

      for (const pipe of renderState.pipes) {
        const pipeLeft = pipe.xPx - FLAPPY_PIPE_COLLISION_SIDE_EXPAND_PX;
        const pipeRight =
          pipe.xPx +
          FLAPPY_PIPE_WIDTH_PX +
          FLAPPY_PIPE_COLLISION_SIDE_EXPAND_PX;
        const overlapsHorizontally =
          birdRight >= pipeLeft && birdLeft <= pipeRight;

        if (overlapsHorizontally) {
          const gapHalf = pipe.gapSizePx * 0.5;
          const gapTop =
            pipe.gapCenterYPx -
            gapHalf +
            FLAPPY_PIPE_COLLISION_ENTRANCE_EXPAND_PX;
          const gapBottom =
            pipe.gapCenterYPx +
            gapHalf -
            FLAPPY_PIPE_COLLISION_ENTRANCE_EXPAND_PX;
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

  // Step 4: Commit logical frame index and return instrumentation count.
  renderState.frameIndex += 1;
  return activationCallsThisFrame;
}
