import Network from '../../../src/architecture/network';
import { Neat, methods } from '../../../src/neataptic';
import {
  evaluateFlappyFitness,
  evaluateFlappyFitnessAcrossSeeds,
} from '../flappyEvaluation';
import type { FlappySeedBatchEvaluation } from '../flappyEvaluation';
import { FlappyEvaluationWorkerPool } from '../evaluation/evaluation.worker-pool';
import type { WorkerInitMessage } from './flappy-evolution-worker.types';
import {
  FLAPPY_MAX_FRAMES_PER_EPISODE,
  FLAPPY_NETWORK_INPUT_SIZE,
  FLAPPY_NETWORK_OUTPUT_SIZE,
} from '../constants/constants';
import { createXorshift32 } from '../rng';
import {
  DEFAULT_FLAPPY_ARCHITECTURE_PROFILE_ID,
  buildExampleArchitectureProfileNetwork,
  resolveExampleArchitectureProfile,
} from '../../architectureProfiles';

const FLAPPY_WORKER_RUNTIME_LOG_PREFIX = '[flappy-worker]';
const SHOULD_LOG_FLAPPY_WORKER_RUNTIME =
  resolveNodeEnvForRuntimeLogs() !== 'test';
const FLAPPY_RECURRENT_MUTATION_METHODS = [
  ...methods.mutation.FFW,
  methods.mutation.ADD_BACK_CONN,
  methods.mutation.SUB_BACK_CONN,
  methods.mutation.ADD_SELF_CONN,
  methods.mutation.SUB_SELF_CONN,
];
const FLAPPY_WORKER_DEFAULT_PIPE_FIRST_SHARED_ROLLOUT_SEED_COUNT = 3;
const FLAPPY_WORKER_LSTM_PIPE_FIRST_SHARED_ROLLOUT_SEED_COUNT = 4;
const FLAPPY_WORKER_PIPE_PROGRESS_TARGET = 12;
const FLAPPY_WORKER_PIPE_PROGRESS_WEIGHT = 10_000;
const FLAPPY_WORKER_STABILITY_STDDEV_WEIGHT = 0.5;
const FLAPPY_WORKER_SHARED_ROLLOUT_GENERATION_XOR_SALT = 0x85eb_ca6b;

/** Minimum first-seed pipe progress required before spending the full recurrent seed batch. */
const FLAPPY_WORKER_MIN_PIPE_PROGRESS_FOR_FULL_BATCH = 0;

interface WorkerPipeFirstEvaluationPlan {
  sharedRolloutSeedCount: number;
}

interface WorkerRuntimeDependencies {
  workerPool?: FlappyEvaluationWorkerPool;
}

type WorkerPopulationLike = Network[];

/**
 * Creates and configures the worker-local NEAT runtime used by browser evolution playback.
 *
 * Educational note:
 * The browser worker reuses the same core NeatapticTS runtime as the Node-side
 * trainer, but trims configuration down to the pieces needed for an interactive
 * example: deterministic seeding, feed-forward mutation policy, and a fitness
 * function that favors quick browser-visible iteration.
 *
 * The resulting runtime is both the evolution engine and the source of the
 * population that later playback requests visualize.
 *
 * For background reading, the Wikipedia article on "Neuroevolution of
 * augmenting topologies" is a useful overview of the family of ideas this demo
 * is exercising, even though the repository implements its own detailed runtime
 * behavior and modern extensions.
 *
 * @example
 * ```ts
 * const neatRuntime = createInitializedWorkerRuntime({
 *   populationSize: 50,
 *   elitismCount: 10,
 *   rngSeed: 12345,
 * });
 * ```
 *
 * @param initPayload - Initialization values from the browser host.
 * @returns Initialized NEAT runtime.
 */
export function createInitializedWorkerRuntime(
  initPayload: WorkerInitMessage['payload'],
  workerRuntimeDependencies: WorkerRuntimeDependencies = {},
): Neat {
  // Step 1: Resolve the selected architecture profile and its seed network.
  const inputSize = FLAPPY_NETWORK_INPUT_SIZE;
  const outputSize = FLAPPY_NETWORK_OUTPUT_SIZE;
  const selectedArchitectureProfile = resolveExampleArchitectureProfile(
    'flappy-bird',
    initPayload.architectureProfileId ?? DEFAULT_FLAPPY_ARCHITECTURE_PROFILE_ID,
  );
  const seedNetwork = resolveWorkerSeedNetwork(
    initPayload,
    selectedArchitectureProfile.id,
  );

  // Step 2: Create the worker-local NEAT runtime around that fixed seed.
  const neatRuntime = new Neat(inputSize, outputSize, () => 0, {
    popsize: initPayload.populationSize,
    elitism: initPayload.elitismCount,
    mutationRate: 0.75,
    mutationAmount: 2,
    fitnessPopulation: shouldUseWorkerPopulationFitness(
      selectedArchitectureProfile.id,
      workerRuntimeDependencies,
    ),
    allowRecurrent: selectedArchitectureProfile.recurrent,
    mutation: selectedArchitectureProfile.recurrent
      ? FLAPPY_RECURRENT_MUTATION_METHODS
      : methods.mutation.FFW,
    network: seedNetwork,
    speciation: true,
    multiObjective: { enabled: false },
    novelty: { enabled: false },
  });

  const evaluateWorkerFitness = createWorkerFitnessEvaluator(
    selectedArchitectureProfile.id,
    initPayload.rngSeed,
    () => neatRuntime.generation,
    workerRuntimeDependencies,
  );

  // Step 3: Install the profile-aware worker fitness function and RNG state.
  neatRuntime.fitness = evaluateWorkerFitness;

  neatRuntime.restoreRNGState(initPayload.rngSeed);
  return neatRuntime;
}

/**
 * Resolves the worker seed network from a saved champion override or the shared profile template.
 *
 * @param initPayload - Initialization values from the browser host.
 * @param architectureProfileId - Resolved architecture profile id.
 * @returns Seed network for the worker-local NEAT runtime.
 */
function resolveWorkerSeedNetwork(
  initPayload: WorkerInitMessage['payload'],
  architectureProfileId: NonNullable<
    WorkerInitMessage['payload']['architectureProfileId']
  >,
): Network {
  if (initPayload.championNetworkJson) {
    try {
      return Network.fromJSON(initPayload.championNetworkJson);
    } catch {
      // Fall back to the shared profile template when browser-local champion state is stale.
    }
  }

  return buildExampleArchitectureProfileNetwork(
    'flappy-bird',
    architectureProfileId,
  );
}

/**
 * Builds the worker fitness evaluator for the selected architecture profile.
 *
 * NARX, GRU, and LSTM benefit from a slightly stricter browser objective
 * because the tiny interactive population is otherwise too willing to overfit
 * one lucky rollout and stall at a zero-pipe local optimum.
 *
 * @param architectureProfileId - Resolved worker architecture profile id.
 * @param workerInitSeed - Deterministic worker seed.
 * @returns Worker-local scalar fitness function.
 */
function createWorkerFitnessEvaluator(
  architectureProfileId: NonNullable<
    WorkerInitMessage['payload']['architectureProfileId']
  >,
  workerInitSeed: number,
  resolveCurrentGeneration: () => number,
  workerRuntimeDependencies: WorkerRuntimeDependencies,
): Neat['fitness'] {
  const usesPipeFirstSharedSeeds =
    architectureProfileId === 'narx' ||
    architectureProfileId === 'gru' ||
    architectureProfileId === 'lstm';
  const workerPool = workerRuntimeDependencies.workerPool;

  if (workerPool && usesPipeFirstSharedSeeds) {
    return createWorkerPopulationFitnessEvaluator(
      architectureProfileId,
      workerInitSeed,
      resolveCurrentGeneration,
      workerPool,
    );
  }

  logWorkerFitnessTransportMode(
    architectureProfileId,
    usesPipeFirstSharedSeeds,
    false,
  );

  // Step 1: Keep the default browser objective unchanged for profiles that do not need pipe-first pressure.
  if (
    architectureProfileId !== 'narx' &&
    architectureProfileId !== 'gru' &&
    architectureProfileId !== 'lstm'
  ) {
    return async (network) =>
      evaluateFlappyFitness(network, {
        enableEarlyTermination: true,
        maxFrames: FLAPPY_MAX_FRAMES_PER_EPISODE,
      });
  }

  // Step 2: Reuse one deterministic seed batch per generation so same-generation comparisons stay fair without overfitting one eternal rollout lane.
  const evaluationPlan = resolveWorkerPipeFirstEvaluationPlan(
    architectureProfileId,
  );
  let cachedGeneration = -1;
  let cachedSharedRolloutSeeds: number[] = [];

  return async (network) =>
    scorePipeFirstWorkerAggregateEvaluation(
      evaluateWorkerNetworkAcrossProgressiveSeeds(
        network,
        resolveSharedRolloutSeedsForGeneration(),
      ),
    );

  /**
   * Evaluates one recurrent genome with a cheap first-seed gate before full scoring.
   *
   * @param network - Genome being scored by the browser worker.
   * @param sharedRolloutSeeds - Deterministic seed batch for the current generation.
   * @returns Aggregate evaluation from either the first seed or the full batch.
   */
  function evaluateWorkerNetworkAcrossProgressiveSeeds(
    network: Network,
    sharedRolloutSeeds: readonly number[],
  ): FlappySeedBatchEvaluation {
    const firstSeedBatch =
      resolveFirstSharedRolloutSeedBatch(sharedRolloutSeeds);
    const firstSeedEvaluation = evaluateFlappyFitnessAcrossSeeds(
      network,
      firstSeedBatch,
      resolveWorkerPipeFirstRolloutOptions(),
    );

    if (
      firstSeedBatch.length === sharedRolloutSeeds.length ||
      !shouldSpendFullWorkerSeedBatch(firstSeedEvaluation)
    ) {
      return firstSeedEvaluation;
    }

    return evaluateFlappyFitnessAcrossSeeds(
      network,
      sharedRolloutSeeds,
      resolveWorkerPipeFirstRolloutOptions(),
    );
  }

  /**
   * Resolves the deterministic shared rollout seeds for the current worker generation.
   *
   * @returns Shared rollout seeds for the current generation.
   */
  function resolveSharedRolloutSeedsForGeneration(): number[] {
    // Step 1: Rebuild the seed batch only when evolution advances to a new generation.
    const currentGeneration = resolveCurrentGeneration();
    if (currentGeneration !== cachedGeneration) {
      cachedGeneration = currentGeneration;
      cachedSharedRolloutSeeds = buildWorkerSharedRolloutSeedBatch(
        workerInitSeed,
        currentGeneration,
        evaluationPlan.sharedRolloutSeedCount,
      );
    }

    return cachedSharedRolloutSeeds;
  }
}

function createWorkerPopulationFitnessEvaluator(
  architectureProfileId: NonNullable<
    WorkerInitMessage['payload']['architectureProfileId']
  >,
  workerInitSeed: number,
  resolveCurrentGeneration: () => number,
  workerPool: FlappyEvaluationWorkerPool,
): (population: WorkerPopulationLike) => Promise<void> {
  const evaluationPlan = resolveWorkerPipeFirstEvaluationPlan(
    architectureProfileId,
  );
  let cachedGeneration = -1;
  let cachedSharedRolloutSeeds: number[] = [];

  logWorkerFitnessTransportMode(architectureProfileId, true, true);

  return async (population) => {
    const aggregateByGenome =
      await evaluateWorkerPopulationAcrossProgressiveSeeds(
        population,
        resolveSharedRolloutSeedsForGeneration(),
        workerPool,
      );

    for (const genome of population) {
      const aggregateEvaluation = aggregateByGenome.get(genome);

      if (!aggregateEvaluation) {
        throw new Error(
          'Worker evaluation pool did not resolve every recurrent genome.',
        );
      }

      genome.score =
        scorePipeFirstWorkerAggregateEvaluation(aggregateEvaluation);
    }
  };

  /**
   * Evaluates a recurrent population through the shared pool with a first-seed gate.
   *
   * @param population - Ordered population shelf to score.
   * @param sharedRolloutSeeds - Deterministic seed batch for the current generation.
   * @param workerPool - Shared-memory pool used for candidate evaluation.
   * @returns Aggregate evidence keyed by genome.
   */
  async function evaluateWorkerPopulationAcrossProgressiveSeeds(
    population: WorkerPopulationLike,
    sharedRolloutSeeds: readonly number[],
    workerPool: FlappyEvaluationWorkerPool,
  ): Promise<Map<Network, FlappySeedBatchEvaluation>> {
    const rolloutOptions = resolveWorkerPipeFirstRolloutOptions();
    const firstSeedBatch =
      resolveFirstSharedRolloutSeedBatch(sharedRolloutSeeds);
    const firstAggregateByGenome = await workerPool.evaluateGenomesAcrossSeeds(
      population,
      firstSeedBatch,
      rolloutOptions,
    );
    const fullBatchPopulation = population.filter((genome) => {
      const firstAggregate = firstAggregateByGenome.get(genome);
      return firstAggregate
        ? shouldSpendFullWorkerSeedBatch(firstAggregate)
        : false;
    });

    if (
      firstSeedBatch.length === sharedRolloutSeeds.length ||
      fullBatchPopulation.length === 0
    ) {
      return firstAggregateByGenome;
    }

    const fullAggregateByGenome = await workerPool.evaluateGenomesAcrossSeeds(
      fullBatchPopulation,
      sharedRolloutSeeds,
      rolloutOptions,
    );

    return new Map(
      population.map((genome) => [
        genome,
        fullAggregateByGenome.get(genome) ??
          resolveRequiredFirstSeedAggregate(firstAggregateByGenome, genome),
      ]),
    );
  }

  function resolveSharedRolloutSeedsForGeneration(): number[] {
    // Step 1: Rebuild the seed batch only when evolution advances to a new generation.
    const currentGeneration = resolveCurrentGeneration();
    if (currentGeneration !== cachedGeneration) {
      cachedGeneration = currentGeneration;
      cachedSharedRolloutSeeds = buildWorkerSharedRolloutSeedBatch(
        workerInitSeed,
        currentGeneration,
        evaluationPlan.sharedRolloutSeedCount,
      );
    }

    return cachedSharedRolloutSeeds;
  }
}

/**
 * Resolves shared rollout options for the pipe-first browser objective.
 *
 * @returns Rollout options used by recurrent worker scoring.
 */
function resolveWorkerPipeFirstRolloutOptions(): {
  enableEarlyTermination: true;
  maxFrames: number;
  normalizeFitness: true;
  pipeProgressTarget: number;
} {
  return {
    enableEarlyTermination: true,
    maxFrames: FLAPPY_MAX_FRAMES_PER_EPISODE,
    normalizeFitness: true,
    pipeProgressTarget: FLAPPY_WORKER_PIPE_PROGRESS_TARGET,
  };
}

/**
 * Resolves the cheap first-seed batch used before full recurrent scoring.
 *
 * @param sharedRolloutSeeds - Full deterministic seed batch for the generation.
 * @returns One-seed batch used for the progressive gate.
 */
function resolveFirstSharedRolloutSeedBatch(
  sharedRolloutSeeds: readonly number[],
): number[] {
  const firstSharedRolloutSeed = sharedRolloutSeeds[0];
  return firstSharedRolloutSeed === undefined ? [] : [firstSharedRolloutSeed];
}

/**
 * Resolves whether one genome should receive the full recurrent seed batch.
 *
 * @param aggregateEvaluation - First-seed evidence for one genome.
 * @returns True when the genome showed enough pipe progress to justify full scoring.
 */
function shouldSpendFullWorkerSeedBatch(
  aggregateEvaluation: FlappySeedBatchEvaluation,
): boolean {
  return (
    aggregateEvaluation.meanPipesPassed >
    FLAPPY_WORKER_MIN_PIPE_PROGRESS_FOR_FULL_BATCH
  );
}

/**
 * Resolves the required first-seed aggregate for a genome.
 *
 * @param aggregateByGenome - First-seed aggregate map.
 * @param genome - Genome whose evidence should exist.
 * @returns Aggregate evaluation for the genome.
 */
function resolveRequiredFirstSeedAggregate(
  aggregateByGenome: ReadonlyMap<Network, FlappySeedBatchEvaluation>,
  genome: Network,
): FlappySeedBatchEvaluation {
  const aggregateEvaluation = aggregateByGenome.get(genome);

  if (!aggregateEvaluation) {
    throw new Error('Progressive worker scoring lost first-seed evidence.');
  }

  return aggregateEvaluation;
}

/**
 * Logs the worker-side fitness transport selection.
 *
 * @param architectureProfileId - Resolved worker architecture profile id.
 * @param usesPipeFirstSharedSeeds - Whether the profile uses the shared-seed aggregate evaluator.
 * @param inferenceChannelWorkerUrl - Nested worker bundle URL when persistent channels are available.
 * @returns Nothing.
 */
function logWorkerFitnessTransportMode(
  architectureProfileId: NonNullable<
    WorkerInitMessage['payload']['architectureProfileId']
  >,
  usesPipeFirstSharedSeeds: boolean,
  usesParallelWorkerPool: boolean,
): void {
  const workerTransportMode = usesParallelWorkerPool
    ? 'shared-memory worker pool evaluation'
    : 'worker-local direct network.activate evaluation';
  const evaluationShape = usesPipeFirstSharedSeeds
    ? 'shared-seed aggregate evaluation'
    : 'single-rollout evaluation';

  if (SHOULD_LOG_FLAPPY_WORKER_RUNTIME) {
    console.info(
      `${FLAPPY_WORKER_RUNTIME_LOG_PREFIX} fitness transport=${workerTransportMode} profile=${architectureProfileId} evaluator=${evaluationShape}`,
    );
  }
}

function shouldUseWorkerPopulationFitness(
  architectureProfileId: NonNullable<
    WorkerInitMessage['payload']['architectureProfileId']
  >,
  workerRuntimeDependencies: WorkerRuntimeDependencies,
): boolean {
  return Boolean(
    workerRuntimeDependencies.workerPool &&
    (architectureProfileId === 'narx' ||
      architectureProfileId === 'gru' ||
      architectureProfileId === 'lstm'),
  );
}

function resolveNodeEnvForRuntimeLogs(): string | undefined {
  return (globalThis as { process?: { env?: { NODE_ENV?: string } } }).process
    ?.env?.NODE_ENV;
}

/**
 * Resolves the recurrent worker evaluation plan for one browser profile.
 *
 * LSTM gets a slightly broader shared-seed batch than the other recurrent
 * profiles because the heavier gated controller was still regressing after
 * generation-zero warm-start when browser selection only saw one static lane.
 *
 * @param architectureProfileId - Resolved worker architecture profile id.
 * @returns Shared-seed batch size for worker fitness.
 */
function resolveWorkerPipeFirstEvaluationPlan(
  architectureProfileId: NonNullable<
    WorkerInitMessage['payload']['architectureProfileId']
  >,
): WorkerPipeFirstEvaluationPlan {
  if (architectureProfileId === 'lstm') {
    return {
      sharedRolloutSeedCount:
        FLAPPY_WORKER_LSTM_PIPE_FIRST_SHARED_ROLLOUT_SEED_COUNT,
    };
  }

  return {
    sharedRolloutSeedCount:
      FLAPPY_WORKER_DEFAULT_PIPE_FIRST_SHARED_ROLLOUT_SEED_COUNT,
  };
}

/**
 * Builds the deterministic rollout seed batch used by the pipe-first worker objective.
 *
 * @param workerInitSeed - Deterministic worker seed.
 * @returns Shared rollout seed batch.
 */
function buildWorkerSharedRolloutSeedBatch(
  workerInitSeed: number,
  generation: number,
  sharedRolloutSeedCount: number,
): number[] {
  // Step 1: Mix both worker seed and generation so browser selection stays fair within a generation without freezing onto one rollout forever.
  const generationMixedSeed =
    workerInitSeed ^
    0x9f38_51de ^
    Math.imul(generation + 1, FLAPPY_WORKER_SHARED_ROLLOUT_GENERATION_XOR_SALT);
  const rolloutSeedRng = createXorshift32(generationMixedSeed);

  // Step 2: Sample a tiny shared batch that remains cheap for the interactive demo.
  return Array.from({ length: sharedRolloutSeedCount }, () =>
    rolloutSeedRng.nextInt(0, 0x1_0000_0000),
  );
}

/**
 * Scores one aggregate with a pipe-first browser selection scalar.
 *
 * The interactive worker population is tiny, so escaping the first-pipe local
 * optimum matters more than preserving the exact trainer ranking stack. This
 * scalar therefore promotes mean pipe progress first, then uses survival and
 * stability as tie-breakers inside the same shared-seed batch.
 *
 * @param aggregateEvaluation - Shared-seed evaluation evidence.
 * @returns Scalar fitness consumed by the browser worker NEAT loop.
 */
function scorePipeFirstWorkerAggregateEvaluation(
  aggregateEvaluation: FlappySeedBatchEvaluation,
): number {
  // Step 1: Convert mean pipe progress into the dominant escape signal.
  const pipeProgressScore =
    aggregateEvaluation.meanPipesPassed * FLAPPY_WORKER_PIPE_PROGRESS_WEIGHT;

  // Step 2: Use robust normalized fitness as the centering-quality signal.
  // Raw meanFramesSurvived is intentionally avoided here: birds that survive
  // thousands of frames without passing any pipes would otherwise outscore
  // shorter but more controlled centering attempts. The normalized fitness
  // already incorporates centering quality, velocity stability, and terminal
  // alignment — all of which reflect real policy quality, not just endurance.
  const normalizedQualityScore = aggregateEvaluation.robustFitness;
  const stabilityPenalty =
    aggregateEvaluation.fitnessStdDev * FLAPPY_WORKER_STABILITY_STDDEV_WEIGHT;

  return pipeProgressScore + normalizedQualityScore - stabilityPenalty;
}
