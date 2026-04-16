import { Neat, methods } from '../../../src/neataptic';
import {
  evaluateFlappyFitness,
  evaluateFlappyFitnessAcrossSeeds,
} from '../flappyEvaluation';
import type { FlappySeedBatchEvaluation } from '../flappyEvaluation';
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

const FLAPPY_RECURRENT_MUTATION_METHODS = [
  ...methods.mutation.FFW,
  methods.mutation.ADD_BACK_CONN,
  methods.mutation.SUB_BACK_CONN,
  methods.mutation.ADD_SELF_CONN,
  methods.mutation.SUB_SELF_CONN,
];
const FLAPPY_WORKER_PIPE_FIRST_SHARED_ROLLOUT_SEED_COUNT = 1;
const FLAPPY_WORKER_PIPE_PROGRESS_TARGET = 12;
const FLAPPY_WORKER_PIPE_PROGRESS_WEIGHT = 10_000;
const FLAPPY_WORKER_STABILITY_STDDEV_WEIGHT = 0.5;

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
): Neat {
  // Step 1: Resolve the selected architecture profile and its seed network.
  const inputSize = FLAPPY_NETWORK_INPUT_SIZE;
  const outputSize = FLAPPY_NETWORK_OUTPUT_SIZE;
  const selectedArchitectureProfile = resolveExampleArchitectureProfile(
    'flappy-bird',
    initPayload.architectureProfileId ??
      DEFAULT_FLAPPY_ARCHITECTURE_PROFILE_ID,
  );
  const seedNetwork = buildExampleArchitectureProfileNetwork(
    'flappy-bird',
    selectedArchitectureProfile.id,
  );
  const evaluateWorkerFitness = createWorkerFitnessEvaluator(
    selectedArchitectureProfile.id,
    initPayload.rngSeed,
  );

  // Step 2: Create the worker-local NEAT runtime around that fixed seed.
  const neatRuntime = new Neat(inputSize, outputSize, () => 0, {
    popsize: initPayload.populationSize,
    elitism: initPayload.elitismCount,
    mutationRate: 0.75,
    mutationAmount: 2,
    allowRecurrent: selectedArchitectureProfile.recurrent,
    mutation: selectedArchitectureProfile.recurrent
      ? FLAPPY_RECURRENT_MUTATION_METHODS
      : methods.mutation.FFW,
    network: seedNetwork,
    speciation: true,
    multiObjective: { enabled: false },
    novelty: { enabled: false },
  });

  // Step 3: Install the profile-aware worker fitness function and RNG state.
  neatRuntime.fitness = evaluateWorkerFitness;

  neatRuntime.restoreRNGState(initPayload.rngSeed);
  return neatRuntime;
}

/**
 * Builds the worker fitness evaluator for the selected architecture profile.
 *
 * NARX and GRU benefit from a slightly stricter browser objective because the
 * tiny interactive population is otherwise too willing to overfit one lucky
 * rollout and stall at a zero-pipe local optimum.
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
): (network: Parameters<Neat['fitness']>[0]) => number {
  // Step 1: Keep the default browser objective unchanged for profiles that do not need pipe-first pressure.
  if (architectureProfileId !== 'narx' && architectureProfileId !== 'gru') {
    return (network) =>
      evaluateFlappyFitness(network, {
        enableEarlyTermination: true,
        maxFrames: FLAPPY_MAX_FRAMES_PER_EPISODE,
      });
  }

  // Step 2: Reuse one deterministic seed batch so same-generation comparisons stay fair.
  const sharedRolloutSeeds = buildWorkerSharedRolloutSeedBatch(workerInitSeed);
  return (network) =>
    scorePipeFirstWorkerAggregateEvaluation(
      evaluateFlappyFitnessAcrossSeeds(network, sharedRolloutSeeds, {
        enableEarlyTermination: true,
        maxFrames: FLAPPY_MAX_FRAMES_PER_EPISODE,
        normalizeFitness: true,
        pipeProgressTarget: FLAPPY_WORKER_PIPE_PROGRESS_TARGET,
      }),
    );
}

/**
 * Builds the deterministic rollout seed batch used by the pipe-first worker objective.
 *
 * @param workerInitSeed - Deterministic worker seed.
 * @returns Shared rollout seed batch.
 */
function buildWorkerSharedRolloutSeedBatch(workerInitSeed: number): number[] {
  // Step 1: Mix the worker seed so browser playback and worker selection do not reuse the same stream.
  const rolloutSeedRng = createXorshift32(workerInitSeed ^ 0x9f38_51de);

  // Step 2: Sample a tiny shared batch that remains cheap for the interactive demo.
  return Array.from(
    { length: FLAPPY_WORKER_PIPE_FIRST_SHARED_ROLLOUT_SEED_COUNT },
    () => rolloutSeedRng.nextInt(0, 0x1_0000_0000),
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

  // Step 2: Reward longer stable survival while penalizing noisy aggregates.
  const survivalScore = aggregateEvaluation.meanFramesSurvived;
  const stabilityPenalty =
    aggregateEvaluation.fitnessStdDev * FLAPPY_WORKER_STABILITY_STDDEV_WEIGHT;

  return pipeProgressScore + survivalScore - stabilityPenalty;
}
