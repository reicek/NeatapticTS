import type { Neat } from '../../../src/neataptic';
import type Network from '../../../src/architecture/network';
import type { ExampleArchitectureProfileId } from '../../architectureProfiles';
import { createXorshift32 } from '../rng';
import {
  rolloutEpisode,
  type FlappyEpisodeResult,
  type FlappyRolloutOptions,
  type FlappySeedBatchEvaluation,
} from '../flappyEvaluation';
import { resolveObservationVector } from '../browser-entry/browser-entry.observation.utils';
import type { WorkerPopulationPipe } from './flappy-evolution-worker.types';
import {
  FLAPPY_BIRD_X_PX,
  FLAPPY_MAX_FRAMES_PER_EPISODE,
  FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
  FLAPPY_NETWORK_INPUT_SIZE,
  FLAPPY_WORLD_HEIGHT_PX,
} from '../constants/constants';
import {
  FLAPPY_WORKER_GEN0_PRETRAIN_BATCH_SIZE,
  FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_BIAS_STDDEV_END,
  FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_BIAS_STDDEV_START,
  FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_OPTIMIZATION_STEPS,
  FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_SEED_COUNT,
  FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_WEIGHT_STDDEV_END,
  FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_WEIGHT_STDDEV_START,
  FLAPPY_WORKER_GEN0_PRETRAIN_BIAS_NOISE_STDDEV,
  FLAPPY_WORKER_GEN0_PRETRAIN_ITERATIONS,
  FLAPPY_WORKER_GEN0_PRETRAIN_RATE,
  FLAPPY_WORKER_GEN0_PRETRAIN_SAMPLE_COUNT,
  FLAPPY_WORKER_GEN0_PRETRAIN_VISIBLE_WORLD_WIDTH_PX,
  FLAPPY_WORKER_GEN0_PRETRAIN_WEIGHT_NOISE_STDDEV,
} from './flappy-evolution-worker.constants';
import {
  createSharedObservationMemoryState,
  computeMean,
  computePercentile,
  computePopulationStandardDeviation,
  resolveAdaptiveDifficultyProfile,
} from '../flappy.simulation.shared.utils';
import { FLAPPY_EVALUATION_ROBUST_STDDEV_PENALTY } from '../evaluation/evaluation.constants';

const FLAPPY_WORKER_NARX_WARM_START_ROLLOUT_SEED_COUNT = 7;
const FLAPPY_WORKER_NARX_WARM_START_OPTIMIZATION_STEPS = 20;
const FLAPPY_WORKER_NARX_WARM_START_PIPE_PROGRESS_WEIGHT = 10_000;
const FLAPPY_WORKER_NARX_WARM_START_STABILITY_STDDEV_WEIGHT = 0.5;
const FLAPPY_WORKER_GRU_WARM_START_ROLLOUT_SEED_COUNT = 4;
const FLAPPY_WORKER_GRU_WARM_START_OPTIMIZATION_STEPS = 8;
const FLAPPY_WORKER_LSTM_WARM_START_ROLLOUT_SEED_COUNT = 6;
const FLAPPY_WORKER_LSTM_WARM_START_OPTIMIZATION_STEPS = 16;

/**
 * Hard worker-local assist limit for recurrent generation-zero warm-start.
 *
 * The browser demo should start evolving promptly: recurrent warm-start is a
 * helpful prior, not a gate that must fully finish before NEAT can continue.
 */
const FLAPPY_WORKER_RECURRENT_WARM_START_TIME_LIMIT_MS = 10_000;

/**
 * Callable shape used to evaluate one warm-start rollout candidate.
 *
 * Tests can inject this seam to observe deadline hooks without running the full
 * Flappy simulator, while production uses the real rollout service.
 */
type WorkerWarmStartRolloutRunner = (
  templateNetwork: Network,
  rolloutOptions: FlappyRolloutOptions,
) => FlappyEpisodeResult;

const FLAPPY_WARM_START_OBSERVATION_INDEX = {
  birdYPx: 0,
  velocity: 1,
  distanceToNextPipe: 2,
  deltaToNextGap: 3,
  nextGapTop: 4,
  nextGapBottom: 5,
} as const;

/**
 * State carried between generation requests for one worker runtime.
 *
 * The warm-start service is intentionally one-shot. These fields let the worker
 * remember whether generation 0 has already been bootstrapped and which initial
 * RNG seed should be reused for deterministic synthetic sample generation.
 */
export interface WorkerWarmStartState {
  architectureProfileId: ExampleArchitectureProfileId;
  workerInitSeed: number;
  generationZeroWarmStartApplied: boolean;
}

/**
 * Dependency bag for generation-0 warm-start orchestration.
 *
 * The production path uses the real heuristic dataset builder and rollout-guided
 * template refinement. Tests can override these seams to keep assertions small
 * and deterministic.
 */
export interface WorkerWarmStartDependencies {
  /** Build heuristic supervised samples for feed-forward warm-start families. */
  buildHeuristicPretrainSet: (
    rng: ReturnType<typeof createXorshift32>,
    sampleCount: number,
  ) => Array<{ input: number[]; output: number[] }>;
  /** Optional rollout-guided template optimizer used by tests and production. */
  optimizeWarmStartTemplateNetwork?: (
    templateNetwork: Network,
    workerInitSeed: number,
    architectureProfileId: ExampleArchitectureProfileId,
    warmStartDeadline: WorkerWarmStartDeadline,
    runWarmStartRollout: WorkerWarmStartRolloutRunner,
  ) => Network;
  /** Optional rollout runner injected to observe or replace rollout scoring. */
  rolloutEpisode?: WorkerWarmStartRolloutRunner;
  /** Clock source used to create and query warm-start deadlines. */
  resolveCurrentTimeMs?: () => number;
}

/** Deadline contract used by recurrent rollout refinement. */
export interface WorkerWarmStartDeadline {
  /** Absolute timestamp at which warm-start refinement should yield. */
  expiresAtMs: number | undefined;
  /** Clock source used for deadline checks. */
  resolveCurrentTimeMs: () => number;
}

/** Rollout-refinement budget resolved for one warm-start architecture profile. */
export type WorkerWarmStartRolloutOptimizationPlan = {
  /** Number of deterministic rollout seeds used for candidate comparison. */
  rolloutSeedCount: number;
  /** Maximum number of topology-fixed optimization perturbations to try. */
  optimizationStepCount: number;
  /** Optional wall-clock assist budget for recurrent warm-start refinement. */
  timeLimitMs?: number;
};

const DEFAULT_WORKER_WARM_START_DEPENDENCIES: WorkerWarmStartDependencies = {
  buildHeuristicPretrainSet,
  optimizeWarmStartTemplateNetwork: optimizeWarmStartTemplateNetwork,
};

/**
 * Applies a one-time generation-0 warm-start to improve initial demo quality.
 *
 * Educational note:
 * The worker entry should stay protocol-first. This service owns the short
 * supervised bootstrap pass that nudges generation 0 away from pure noise while
 * preserving the later NEAT-driven search loop.
 *
 * Conceptually this is a lightweight behavior-cloning pass. If you want more
 * background, the Wikipedia article on "imitation learning" is a helpful bridge
 * between the heuristic teacher used here and the later evolutionary search.
 *
 * @param neatController - Initialized NEAT runtime.
 * @param warmStartState - Mutable warm-start lifecycle state.
 * @param dependencies - Injectable warm-start seams for tests and runtime customization.
 * @returns Promise resolved when warm-start evaluation finishes.
 * @example
 * ```ts
 * await warmStartWorkerGenerationZeroIfNeeded(neatRuntime, {
 *   workerInitSeed: 123,
 *   generationZeroWarmStartApplied: false,
 * });
 * ```
 */
export async function warmStartWorkerGenerationZeroIfNeeded(
  neatController: Neat,
  warmStartState: WorkerWarmStartState,
  dependencies: WorkerWarmStartDependencies = DEFAULT_WORKER_WARM_START_DEPENDENCIES,
): Promise<void> {
  // Step 1: Exit fast when warm-start is already processed.
  if (warmStartState.generationZeroWarmStartApplied) return;

  try {
    applyWarmStartGenerationZero(neatController, warmStartState, dependencies);
  } catch {
    // Warm-start is best-effort; the first NEAT generation should still evolve.
  } finally {
    warmStartState.generationZeroWarmStartApplied = true;
  }
}

/**
 * Applies the generation-zero warm-start body when the runtime is still eligible.
 *
 * @param neatController - Initialized NEAT runtime.
 * @param warmStartState - Mutable warm-start lifecycle state.
 * @param dependencies - Injectable warm-start seams.
 * @returns Nothing.
 */
function applyWarmStartGenerationZero(
  neatController: Neat,
  warmStartState: WorkerWarmStartState,
  dependencies: WorkerWarmStartDependencies,
): void {
  // Step 2: Warm-start only generation 0 to avoid skewing later evolution.
  if (neatController.generation !== 0) {
    return;
  }

  // Step 3: Validate population availability.
  const population = neatController.population;
  if (!Array.isArray(population) || population.length === 0) {
    return;
  }

  // Step 4: Build synthetic labels only for profiles that use teacher fitting.
  const warmStartRng = createXorshift32(
    warmStartState.workerInitSeed ^ 0x9e37_79b9,
  );
  const teacherStrategy = resolveWorkerWarmStartTeacherStrategy(
    warmStartState.architectureProfileId,
  );
  const trainingSet =
    teacherStrategy === 'rollout-only'
      ? []
      : dependencies.buildHeuristicPretrainSet(
          warmStartRng,
          FLAPPY_WORKER_GEN0_PRETRAIN_SAMPLE_COUNT,
        );

  // Step 5: Train a single template network when the selected profile supports it.
  const templateNetwork = population[0]?.clone();
  if (!templateNetwork) {
    return;
  }

  applyTeacherWarmStart(
    templateNetwork,
    trainingSet,
    warmStartState.architectureProfileId,
  );

  // Step 6: Refine the trained template with a short rollout-guided hill-climb.
  const resolveCurrentTimeMs = dependencies.resolveCurrentTimeMs ?? Date.now;
  const warmStartDeadline = createWarmStartDeadline(
    warmStartState.architectureProfileId,
    resolveCurrentTimeMs,
  );
  const optimizeWarmStartTemplate =
    dependencies.optimizeWarmStartTemplateNetwork ??
    optimizeWarmStartTemplateNetwork;
  const runWarmStartRollout = dependencies.rolloutEpisode ?? rolloutEpisode;
  const optimizedTemplateNetwork = optimizeWarmStartTemplate(
    templateNetwork,
    warmStartState.workerInitSeed,
    warmStartState.architectureProfileId,
    warmStartDeadline,
    runWarmStartRollout,
  );

  // Step 7: Copy trained weights/biases into each genome with small noise for diversity.
  for (const genome of population) {
    applyTemplateWeightsWithNoise(
      genome,
      optimizedTemplateNetwork,
      warmStartRng,
      {
        weightStdDev: FLAPPY_WORKER_GEN0_PRETRAIN_WEIGHT_NOISE_STDDEV,
        biasStdDev: FLAPPY_WORKER_GEN0_PRETRAIN_BIAS_NOISE_STDDEV,
      },
    );
    (genome as unknown as { score?: number }).score = undefined;
  }
}

type WorkerWarmStartTeacherStrategy =
  'feed-forward-teacher-fit' | 'rollout-only';

/**
 * Applies the teacher phase that best matches the selected architecture family.
 *
 * @param templateNetwork - Template network cloned from the current population.
 * @param trainingSet - Synthetic heuristic dataset.
 * @param architectureProfileId - Selected shared Flappy profile id.
 * @returns Nothing.
 */
function applyTeacherWarmStart(
  templateNetwork: Network,
  trainingSet: Array<{ input: number[]; output: number[] }>,
  architectureProfileId: ExampleArchitectureProfileId,
): void {
  const teacherStrategy = resolveWorkerWarmStartTeacherStrategy(
    architectureProfileId,
  );

  if (teacherStrategy === 'rollout-only') {
    return;
  }

  try {
    if (teacherStrategy === 'feed-forward-teacher-fit') {
      templateNetwork.train(trainingSet, {
        iterations: FLAPPY_WORKER_GEN0_PRETRAIN_ITERATIONS,
        rate: FLAPPY_WORKER_GEN0_PRETRAIN_RATE,
        batchSize: FLAPPY_WORKER_GEN0_PRETRAIN_BATCH_SIZE,
        optimizer: 'adam',
        mixedPrecision: false,
      });
    }
  } catch {
    // If teacher fitting fails for any reason, fall back to rollout-only refinement.
  }
}

/**
 * Resolves which teacher path should run before rollout refinement.
 *
 * @param architectureProfileId - Selected shared Flappy profile id.
 * @returns Teacher strategy best matched to the architecture family.
 */
export function resolveWorkerWarmStartTeacherStrategy(
  architectureProfileId: ExampleArchitectureProfileId,
): WorkerWarmStartTeacherStrategy {
  switch (architectureProfileId) {
    case 'mlp':
    case 'random-sparse':
      return 'feed-forward-teacher-fit';

    case 'narx':
    case 'gru':
    case 'lstm':
      return 'rollout-only';
  }
}

/**
 * Builds synthetic supervised samples for generation-0 behavior cloning.
 *
 * Educational note:
 * These samples are not recorded gameplay traces. They are synthetic states
 * generated from the same observation pipeline used during real playback so the
 * teacher labels and the evolved policy inputs stay in the same feature space.
 * The critical rule is that the teacher must only read the same compact
 * current-frame controller shelf that the live network will later receive.
 *
 * @param rng - Deterministic random source.
 * @param sampleCount - Requested number of synthetic samples.
 * @returns Supervised dataset of input/output pairs.
 */
function buildHeuristicPretrainSet(
  rng: ReturnType<typeof createXorshift32>,
  sampleCount: number,
): Array<{ input: number[]; output: number[] }> {
  // Step 1: Resolve bounded sample count and a baseline difficulty profile.
  const clampedSampleCount = Math.max(8, Math.trunc(sampleCount));
  const difficultyProfile = resolveAdaptiveDifficultyProfile(0, 1);
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

    const shouldFlap = resolveHeuristicTeacherFlapDecisionFromObservationVector(
      observation.observationVector,
    );
    trainingSet.push({
      input: observation.observationVector,
      output: shouldFlap ? [0, 1] : [1, 0],
    });
  }

  // Step 3: Return immutable training pairs consumed by Network.train.
  return trainingSet;
}

/**
 * Refines the generation-0 template against real Flappy rollouts.
 *
 * The warm-start teacher gets the template out of pure-random territory, but it
 * still only imitates a simple flap heuristic. This refinement pass keeps the
 * topology fixed and searches the parameter surface directly against actual
 * rollout fitness so the first visible generation starts closer to competent
 * control.
 *
 * @param templateNetwork - Heuristic-pretrained template network.
 * @param workerInitSeed - Deterministic worker seed.
 * @param architectureProfileId - Selected shared Flappy profile id.
 * @param warmStartDeadline - Optional recurrent assist deadline.
 * @param runWarmStartRollout - Rollout runner used to score candidate templates.
 * @returns Best rollout-refined template found within the bounded budget.
 */
function optimizeWarmStartTemplateNetwork(
  templateNetwork: Network,
  workerInitSeed: number,
  architectureProfileId: ExampleArchitectureProfileId,
  warmStartDeadline: WorkerWarmStartDeadline = createWarmStartDeadline(
    architectureProfileId,
    Date.now,
  ),
  runWarmStartRollout: WorkerWarmStartRolloutRunner = rolloutEpisode,
): Network {
  const rolloutOptimizationPlan = resolveWarmStartRolloutOptimizationPlan(
    architectureProfileId,
  );

  // Step 1: Build deterministic shared rollout seeds for candidate comparison.
  const rolloutSeedRng = createXorshift32(workerInitSeed ^ 0xa341_316c);
  const sharedRolloutSeeds = buildWarmStartRolloutSeedBatch(
    rolloutSeedRng,
    rolloutOptimizationPlan.rolloutSeedCount,
  );

  // Step 2: Start from the teacher-fitted template as the current best policy.
  const optimizationRng = createXorshift32(workerInitSeed ^ 0xc801_3ea4);
  let bestTemplateNetwork = templateNetwork;
  let bestTemplateEvaluation = evaluateWarmStartTemplateAcrossRollouts(
    bestTemplateNetwork,
    sharedRolloutSeeds,
    warmStartDeadline,
    runWarmStartRollout,
  );

  // Step 3: Run a bounded topology-fixed hill-climb on actual rollout fitness.
  for (
    let optimizationStepIndex = 0;
    optimizationStepIndex < rolloutOptimizationPlan.optimizationStepCount;
    optimizationStepIndex++
  ) {
    if (isWarmStartDeadlineExpired(warmStartDeadline)) {
      break;
    }

    const candidateTemplateNetwork = bestTemplateNetwork.clone();
    const annealRatio = resolveWarmStartAnnealRatio(
      optimizationStepIndex,
      rolloutOptimizationPlan.optimizationStepCount,
    );

    perturbNetworkParametersInPlace(candidateTemplateNetwork, optimizationRng, {
      weightStdDev: interpolateValue(
        FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_WEIGHT_STDDEV_START,
        FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_WEIGHT_STDDEV_END,
        annealRatio,
      ),
      biasStdDev: interpolateValue(
        FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_BIAS_STDDEV_START,
        FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_BIAS_STDDEV_END,
        annealRatio,
      ),
    });

    const candidateTemplateEvaluation = evaluateWarmStartTemplateAcrossRollouts(
      candidateTemplateNetwork,
      sharedRolloutSeeds,
      warmStartDeadline,
      runWarmStartRollout,
    );
    if (
      !isWarmStartEvaluationBetter(
        candidateTemplateEvaluation,
        bestTemplateEvaluation,
        architectureProfileId,
      )
    ) {
      continue;
    }

    bestTemplateNetwork = candidateTemplateNetwork;
    bestTemplateEvaluation = candidateTemplateEvaluation;
  }

  return bestTemplateNetwork;
}

/**
 * Creates the optional deadline used by recurrent warm-start refinement.
 *
 * @param architectureProfileId - Selected shared Flappy profile id.
 * @param resolveCurrentTimeMs - Clock source used for deadline checks.
 * @returns Warm-start deadline contract.
 */
function createWarmStartDeadline(
  architectureProfileId: ExampleArchitectureProfileId,
  resolveCurrentTimeMs: () => number,
): WorkerWarmStartDeadline {
  const rolloutOptimizationPlan = resolveWarmStartRolloutOptimizationPlan(
    architectureProfileId,
  );
  const expiresAtMs = rolloutOptimizationPlan.timeLimitMs
    ? resolveCurrentTimeMs() + rolloutOptimizationPlan.timeLimitMs
    : undefined;

  return {
    expiresAtMs,
    resolveCurrentTimeMs,
  };
}

/**
 * Resolves whether rollout refinement should yield to regular NEAT evolution.
 *
 * @param warmStartDeadline - Deadline contract for the current warm-start pass.
 * @returns True when the warm-start assist has spent its allowed budget.
 */
function isWarmStartDeadlineExpired(
  warmStartDeadline: WorkerWarmStartDeadline,
): boolean {
  return (
    warmStartDeadline.expiresAtMs !== undefined &&
    warmStartDeadline.resolveCurrentTimeMs() >= warmStartDeadline.expiresAtMs
  );
}

/**
 * Builds the deterministic shared rollout seed batch used during warm-start refinement.
 *
 * @param rng - Deterministic random source.
 * @param seedCount - Requested seed count.
 * @returns Shared rollout seed batch.
 */
function buildWarmStartRolloutSeedBatch(
  rng: ReturnType<typeof createXorshift32>,
  seedCount: number,
): number[] {
  // Step 1: Clamp the request and sample uint32-compatible rollout seeds.
  const clampedSeedCount = Math.max(1, Math.trunc(seedCount));
  return Array.from({ length: clampedSeedCount }, () =>
    rng.nextInt(0, 0x1_0000_0000),
  );
}

/**
 * Evaluates one warm-start template across the shared rollout seed batch.
 *
 * @param templateNetwork - Candidate template to score.
 * @param sharedRolloutSeeds - Shared rollout seeds used for stable comparison.
 * @param warmStartDeadline - Deadline that can stop seed evaluation early.
 * @param runWarmStartRollout - Rollout runner used to evaluate each seed.
 * @returns Aggregate shared-seed evaluation.
 */
function evaluateWarmStartTemplateAcrossRollouts(
  templateNetwork: Network,
  sharedRolloutSeeds: readonly number[],
  warmStartDeadline: WorkerWarmStartDeadline,
  runWarmStartRollout: WorkerWarmStartRolloutRunner,
): FlappySeedBatchEvaluation {
  // Step 1: Reset network state when the runtime exposes a clear hook.
  const maybeClearableNetwork = templateNetwork as Network & {
    clear?: () => void;
  };
  maybeClearableNetwork.clear?.();

  // Step 2: Score as many shared seeds as the current deadline can still afford.
  const episodeResults: FlappyEpisodeResult[] = [];
  for (const seedValue of sharedRolloutSeeds) {
    if (isWarmStartDeadlineExpired(warmStartDeadline)) {
      break;
    }

    maybeClearableNetwork.clear?.();
    episodeResults.push(
      runWarmStartRollout(templateNetwork, {
        enableEarlyTermination: true,
        maxFrames: FLAPPY_MAX_FRAMES_PER_EPISODE,
        seed: seedValue,
        shouldStop: () => isWarmStartDeadlineExpired(warmStartDeadline),
      }),
    );
  }

  // Step 3: Collapse the completed seed evidence into the same aggregate shape.
  return composeWarmStartSeedBatchEvaluation(episodeResults);
}

/**
 * Composes the completed warm-start rollouts into aggregate evidence.
 *
 * @param episodeResults - Completed rollout results before the deadline fired.
 * @returns Aggregate warm-start evaluation metrics.
 */
function composeWarmStartSeedBatchEvaluation(
  episodeResults: readonly FlappyEpisodeResult[],
): FlappySeedBatchEvaluation {
  const rolloutFitnessValues = episodeResults.map(
    (rolloutResult) => rolloutResult.fitness,
  );
  const fitnessMean = computeMean(rolloutFitnessValues);
  const fitnessStdDev = computePopulationStandardDeviation(
    rolloutFitnessValues,
    fitnessMean,
  );
  const medianFitness =
    rolloutFitnessValues.length === 0
      ? 0
      : computePercentile(rolloutFitnessValues, 0.5);
  const p90Fitness =
    rolloutFitnessValues.length === 0
      ? 0
      : computePercentile(rolloutFitnessValues, 0.9);

  return {
    seedCount: episodeResults.length,
    meanFitness: fitnessMean,
    medianFitness,
    p90Fitness,
    fitnessStdDev,
    robustFitness:
      fitnessMean - fitnessStdDev * FLAPPY_EVALUATION_ROBUST_STDDEV_PENALTY,
    meanPipesPassed: computeMean(
      episodeResults.map((episodeResult) => episodeResult.pipesPassed),
    ),
    meanFramesSurvived: computeMean(
      episodeResults.map((episodeResult) => episodeResult.framesSurvived),
    ),
  };
}

/**
 * Resolves whether the candidate batch evaluation beats the current best one.
 *
 * Robust fitness is the primary signal. Mean pipe progress and mean frame
 * survival act as deterministic tie-breakers so upgrades remain stable when the
 * robust score is identical.
 *
 * @param candidateEvaluation - Newly scored candidate aggregate.
 * @param bestEvaluation - Current best aggregate.
 * @param architectureProfileId - Selected shared Flappy profile id.
 * @returns True when the candidate should replace the incumbent template.
 */
function isWarmStartEvaluationBetter(
  candidateEvaluation: FlappySeedBatchEvaluation,
  bestEvaluation: FlappySeedBatchEvaluation,
  architectureProfileId: ExampleArchitectureProfileId,
): boolean {
  // Step 1: Prefer the architecture-specific warm-start scalar.
  const candidateScore = resolveWarmStartEvaluationScore(
    candidateEvaluation,
    architectureProfileId,
  );
  const bestScore = resolveWarmStartEvaluationScore(
    bestEvaluation,
    architectureProfileId,
  );

  if (candidateScore !== bestScore) {
    return candidateScore > bestScore;
  }

  // Step 2: Break ties with more practical gameplay progress.
  if (candidateEvaluation.meanPipesPassed !== bestEvaluation.meanPipesPassed) {
    return candidateEvaluation.meanPipesPassed > bestEvaluation.meanPipesPassed;
  }

  return (
    candidateEvaluation.meanFramesSurvived > bestEvaluation.meanFramesSurvived
  );
}

/**
 * Resolves the rollout-refinement budget for one warm-start architecture profile.
 *
 * NARX gets a stronger rollout pass, while GRU keeps a smaller bounded pass so
 * the browser worker stays responsive after the Flappy-specific readout
 * shortcut expands the recurrent seed.
 *
 * @param architectureProfileId - Selected shared Flappy profile id.
 * @returns Shared-seed count, optimization-step budget, and optional time cap.
 */
export function resolveWarmStartRolloutOptimizationPlan(
  architectureProfileId: ExampleArchitectureProfileId,
): WorkerWarmStartRolloutOptimizationPlan {
  if (architectureProfileId === 'narx') {
    return {
      rolloutSeedCount: FLAPPY_WORKER_NARX_WARM_START_ROLLOUT_SEED_COUNT,
      optimizationStepCount: FLAPPY_WORKER_NARX_WARM_START_OPTIMIZATION_STEPS,
      timeLimitMs: FLAPPY_WORKER_RECURRENT_WARM_START_TIME_LIMIT_MS,
    };
  }

  if (architectureProfileId === 'gru') {
    return {
      rolloutSeedCount: FLAPPY_WORKER_GRU_WARM_START_ROLLOUT_SEED_COUNT,
      optimizationStepCount: FLAPPY_WORKER_GRU_WARM_START_OPTIMIZATION_STEPS,
      timeLimitMs: FLAPPY_WORKER_RECURRENT_WARM_START_TIME_LIMIT_MS,
    };
  }

  if (architectureProfileId === 'lstm') {
    return {
      rolloutSeedCount: FLAPPY_WORKER_LSTM_WARM_START_ROLLOUT_SEED_COUNT,
      optimizationStepCount: FLAPPY_WORKER_LSTM_WARM_START_OPTIMIZATION_STEPS,
      timeLimitMs: FLAPPY_WORKER_RECURRENT_WARM_START_TIME_LIMIT_MS,
    };
  }

  return {
    rolloutSeedCount: FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_SEED_COUNT,
    optimizationStepCount:
      FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_OPTIMIZATION_STEPS,
  };
}

/**
 * Resolves the architecture-specific scalar used during warm-start rollout refinement.
 *
 * NARX, GRU, and LSTM use a pipe-first scalar so rollout refinement prefers
 * real pipe progress over a dense-shaping local optimum before the browser
 * NEAT loop begins.
 *
 * @param aggregateEvaluation - Shared-seed rollout evidence for one template.
 * @param architectureProfileId - Selected shared Flappy profile id.
 * @returns Scalar score used for candidate comparison.
 */
export function resolveWarmStartEvaluationScore(
  aggregateEvaluation: FlappySeedBatchEvaluation,
  architectureProfileId: ExampleArchitectureProfileId,
): number {
  if (
    architectureProfileId === 'narx' ||
    architectureProfileId === 'gru' ||
    architectureProfileId === 'lstm'
  ) {
    const pipeProgressScore =
      aggregateEvaluation.meanPipesPassed *
      FLAPPY_WORKER_NARX_WARM_START_PIPE_PROGRESS_WEIGHT;
    const survivalScore = aggregateEvaluation.meanFramesSurvived;
    const stabilityPenalty =
      aggregateEvaluation.fitnessStdDev *
      FLAPPY_WORKER_NARX_WARM_START_STABILITY_STDDEV_WEIGHT;

    return pipeProgressScore + survivalScore - stabilityPenalty;
  }

  return aggregateEvaluation.robustFitness;
}

/**
 * Resolves the annealing ratio for rollout-guided warm-start refinement.
 *
 * @param optimizationStepIndex - Zero-based optimization step index.
 * @param totalOptimizationSteps - Total number of optimization steps.
 * @returns Clamped ratio in the inclusive range [0, 1].
 */
function resolveWarmStartAnnealRatio(
  optimizationStepIndex: number,
  totalOptimizationSteps: number,
): number {
  // Step 1: Collapse one-step schedules to the broad-search endpoint.
  if (totalOptimizationSteps <= 1) {
    return 0;
  }

  // Step 2: Convert the step index into a stable cooling ratio.
  return Math.min(
    1,
    Math.max(0, optimizationStepIndex / (totalOptimizationSteps - 1)),
  );
}

/**
 * Applies additive Gaussian noise to an existing network in-place.
 *
 * Unlike the later population seeding copy step, this helper perturbs the
 * candidate template directly so the rollout optimizer can evaluate one local
 * parameter move at a time while keeping the topology unchanged.
 *
 * @param network - Candidate template to perturb.
 * @param rng - Deterministic random source.
 * @param noise - Standard deviations for weight and bias perturbations.
 * @returns Nothing.
 */
function perturbNetworkParametersInPlace(
  network: Network,
  rng: ReturnType<typeof createXorshift32>,
  noise: { weightStdDev: number; biasStdDev: number },
): void {
  // Step 1: Clamp noise scales to valid non-negative values.
  const weightStdDev = Math.max(0, noise.weightStdDev);
  const biasStdDev = Math.max(0, noise.biasStdDev);

  // Step 2: Perturb node biases in-place.
  for (const networkNode of network.nodes) {
    networkNode.bias += sampleGaussian(rng) * biasStdDev;
  }

  // Step 3: Perturb connection weights in-place.
  for (const networkConnection of network.connections) {
    networkConnection.weight += sampleGaussian(rng) * weightStdDev;
  }
}

/**
 * Linearly interpolates between two scalar values.
 *
 * @param startValue - Value at ratio `0`.
 * @param endValue - Value at ratio `1`.
 * @param ratio - Interpolation ratio.
 * @returns Interpolated value.
 */
function interpolateValue(
  startValue: number,
  endValue: number,
  ratio: number,
): number {
  // Step 1: Clamp the ratio before blending.
  const clampedRatio = Math.min(1, Math.max(0, ratio));
  return startValue + (endValue - startValue) * clampedRatio;
}

/**
 * Heuristic teacher policy used to label synthetic pretraining samples.
 *
 * The rule intentionally stays simple and interpretable: flap when the bird is
 * meaningfully below the next gap center, not already rising fast, and either
 * near the next pipe or drifting close to the lower edge of the current gap.
 *
 * The key constraint is architectural consistency: this teacher reads only the
 * same compact 6-value controller vector used by regular training and playback.
 * Extra legacy-derived features must not influence generation-zero labels.
 *
 * @param observationVector - Compact current-frame controller input vector.
 * @returns True when the teacher says to flap.
 */
export function resolveHeuristicTeacherFlapDecisionFromObservationVector(
  observationVector: readonly number[],
): boolean {
  // Step 1: Ignore incomplete vectors so warm-start never invents hidden channels.
  if (observationVector.length < FLAPPY_NETWORK_INPUT_SIZE) {
    return false;
  }

  const birdYPx =
    observationVector[FLAPPY_WARM_START_OBSERVATION_INDEX.birdYPx] ?? 0;
  const velocity =
    observationVector[FLAPPY_WARM_START_OBSERVATION_INDEX.velocity] ?? 0;
  const distanceToNextPipe =
    observationVector[FLAPPY_WARM_START_OBSERVATION_INDEX.distanceToNextPipe] ??
    0;
  const deltaToNextGap =
    observationVector[FLAPPY_WARM_START_OBSERVATION_INDEX.deltaToNextGap] ?? 0;
  const nextGapTop =
    observationVector[FLAPPY_WARM_START_OBSERVATION_INDEX.nextGapTop] ?? 0;
  const nextGapBottom =
    observationVector[FLAPPY_WARM_START_OBSERVATION_INDEX.nextGapBottom] ?? 0;
  // Step 2: Resolve the few interpretable teacher conditions from the live shelf.
  const isBelowNextGapCenter = deltaToNextGap > 0.035;
  const isNotAlreadyRisingFast = velocity > -0.25;
  const isNearNextPipe = distanceToNextPipe < 0.42;
  const isNearLowerGapEdge = birdYPx > nextGapBottom - 0.08;
  const hasRoomAboveGapFloor = birdYPx > nextGapTop;

  // Step 3: Fold the teacher decision from those current-frame conditions only.
  return (
    isBelowNextGapCenter &&
    isNotAlreadyRisingFast &&
    hasRoomAboveGapFloor &&
    (isNearNextPipe || isNearLowerGapEdge)
  );
}

/**
 * Copies template parameters into a genome and injects small Gaussian noise.
 *
 * Educational note:
 * The template network gives generation 0 a shared prior, while the noise terms
 * restore diversity so the population is still worth evolving.
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
 * Samples one standard-normal value using the Box-Muller transform.
 *
 * If you are unfamiliar with the transform, the Wikipedia article on
 * "Box-Muller transform" is a useful short background read. The worker uses it
 * here because it is deterministic, dependency-light, and good enough for small
 * noise injection during warm-start diversification.
 *
 * @param rng - Deterministic random source.
 * @returns One approximately standard-normal random value.
 */
function sampleGaussian(rng: ReturnType<typeof createXorshift32>): number {
  // Step 1: Avoid log(0) by flooring uniforms away from zero.
  const epsilon = 1e-12;
  const uniformA = Math.max(epsilon, rng.nextFloat01());
  const uniformB = Math.max(epsilon, rng.nextFloat01());

  // Step 2: Convert two uniforms into one normal sample.
  const magnitude = Math.sqrt(-2 * Math.log(uniformA));
  const angle = 2 * Math.PI * uniformB;
  return magnitude * Math.cos(angle);
}
