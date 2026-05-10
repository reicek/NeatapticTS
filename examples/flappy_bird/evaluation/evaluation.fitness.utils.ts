import type Network from '../../../src/architecture/network';
import {
  exportTransferableInferencePayload,
  openInferenceChannel,
  type SharedInferenceWorker,
} from '../../../src/neataptic';
import { FLAPPY_EVALUATION_ROBUST_STDDEV_PENALTY } from './evaluation.constants';
import {
  rolloutEpisode,
  rolloutEpisodeWithPredictor,
} from './evaluation.rollout.service';
import type {
  FlappyNetworkLike,
  FlappyRolloutOptions,
  FlappySeedBatchEvaluation,
} from './evaluation.types';
import {
  computeMean,
  computePercentile,
  computePopulationStandardDeviation,
} from '../flappy.simulation.shared.utils';

/**
 * Evaluate a network on a single deterministic Flappy Bird episode.
 *
 * This is the simplest evaluation entrypoint: one policy, one rollout, one
 * scalar fitness.
 *
 * @param network - Genome/network to evaluate.
 * @param rolloutOptions - Optional rollout controls.
 * @returns Fitness score (higher is better).
 */
export function evaluateFlappyFitness(
  network: FlappyNetworkLike,
  rolloutOptions: FlappyRolloutOptions = {},
): number {
  return runClearedRolloutEpisode(network, rolloutOptions).fitness;
}

/**
 * Evaluate a network on a shared batch of deterministic seeds.
 *
 * Educational note:
 * Shared-seed evaluation reduces luck. Every genome in the same comparison set
 * sees the same rollout seeds, which makes the aggregate statistics much more
 * useful for selection than a single lucky episode.
 *
 * @example
 * ```ts
 * const aggregate = evaluateFlappyFitnessAcrossSeeds(network, [11, 22, 33], {
 *   normalizeFitness: true,
 * });
 * ```
 *
 * @param network - Genome/network to evaluate.
 * @param sharedSeeds - Shared deterministic seeds used for all genomes.
 * @param rolloutOptions - Optional rollout controls.
 * @returns Robust aggregate metrics for selection/ranking.
 */
export function evaluateFlappyFitnessAcrossSeeds(
  network: FlappyNetworkLike,
  sharedSeeds: readonly number[],
  rolloutOptions: FlappyRolloutOptions = {},
): FlappySeedBatchEvaluation {
  const episodeResults = sharedSeeds.map((seedValue) =>
    runClearedRolloutEpisode(network, {
      ...rolloutOptions,
      seed: seedValue,
    }),
  );
  return composeSeedBatchEvaluation(sharedSeeds.length, episodeResults);
}

/**
 * Evaluate a network through one persistent inference channel on a single seeded episode.
 *
 * This browser-worker-oriented helper reuses one worker-side predictor instead
 * of calling `network.activate(...)` directly on the hot rollout path.
 *
 * @param network - Network to evaluate through one persistent inference channel.
 * @param options - Rollout controls plus the browser worker bundle URL.
 * @returns Fitness score (higher is better).
 */
export async function evaluateFlappyFitnessWithInferenceChannel(
  network: Network,
  options: {
    rolloutOptions?: FlappyRolloutOptions;
    workerUrl: string;
  },
): Promise<number> {
  const networkId = typeof network._id === 'number' ? network._id : undefined;
  const inferenceChannel = openInferenceChannel(
    exportTransferableInferencePayload(network),
    {
      workerUrl: options.workerUrl,
    },
  );

  try {
    const rolloutResult = await runResetChannelRolloutEpisode(
      inferenceChannel,
      networkId,
      options.rolloutOptions ?? {},
    );

    return rolloutResult.fitness;
  } finally {
    await inferenceChannel.close();
  }
}

/**
 * Evaluate a network through one persistent inference channel across shared seeds.
 *
 * The same worker-side predictor is reset between seeded episodes so the
 * browser worker can reuse warm transport state without leaking recurrent
 * memory across rollout boundaries.
 *
 * @param network - Network to evaluate through one persistent inference channel.
 * @param sharedSeeds - Shared deterministic seeds used for all genomes.
 * @param options - Rollout controls plus the browser worker bundle URL.
 * @returns Robust aggregate metrics for selection/ranking.
 */
export async function evaluateFlappyFitnessAcrossSeedsWithInferenceChannel(
  network: Network,
  sharedSeeds: readonly number[],
  options: {
    rolloutOptions?: FlappyRolloutOptions;
    workerUrl: string;
  },
): Promise<FlappySeedBatchEvaluation> {
  const networkId = typeof network._id === 'number' ? network._id : undefined;
  const inferenceChannel = openInferenceChannel(
    exportTransferableInferencePayload(network),
    {
      workerUrl: options.workerUrl,
    },
  );

  try {
    const episodeResults = await Promise.all(
      sharedSeeds.map((seedValue) =>
        runResetChannelRolloutEpisode(inferenceChannel, networkId, {
          ...(options.rolloutOptions ?? {}),
          seed: seedValue,
        }),
      ),
    );

    return composeSeedBatchEvaluation(sharedSeeds.length, episodeResults);
  } finally {
    await inferenceChannel.close();
  }
}

/**
 * Evaluate one shared-memory predictor across a deterministic seed batch.
 *
 * This helper keeps one `SharedInferenceWorker` warm across the whole seed set
 * so the caller can parallelize across genomes without paying one bootstrap
 * cost per seeded rollout.
 *
 * @param sharedInferenceWorker - Persistent shared-memory predictor for one genome.
 * @param sharedSeeds - Shared deterministic seeds used for the evaluation batch.
 * @param options - Optional rollout controls plus a stable network id for seed mixing.
 * @returns Robust aggregate metrics for selection/ranking.
 */
export async function evaluateFlappyFitnessAcrossSeedsWithSharedInferenceWorker(
  sharedInferenceWorker: SharedInferenceWorker,
  sharedSeeds: readonly number[],
  options: {
    networkId?: number;
    rolloutOptions?: FlappyRolloutOptions;
  } = {},
): Promise<FlappySeedBatchEvaluation> {
  const episodeResults = [];

  // Step 1: Reuse one shared-memory predictor across the seeded rollout batch.
  for (const seedValue of sharedSeeds) {
    const rolloutResult = await runResetSharedWorkerRolloutEpisode(
      sharedInferenceWorker,
      options.networkId,
      {
        ...(options.rolloutOptions ?? {}),
        seed: seedValue,
      },
    );
    episodeResults.push(rolloutResult);
  }

  // Step 2: Collapse the seeded episode shelf into one ranking aggregate.
  return composeSeedBatchEvaluation(sharedSeeds.length, episodeResults);
}

function composeSeedBatchEvaluation(
  seedCount: number,
  episodeResults: Array<{
    fitness: number;
    pipesPassed: number;
    framesSurvived: number;
  }>,
): FlappySeedBatchEvaluation {
  const rolloutFitnessValues = episodeResults.map(
    (rolloutResult) => rolloutResult.fitness,
  );

  const fitnessMean = computeMean(rolloutFitnessValues);
  const fitnessStdDev = computePopulationStandardDeviation(
    rolloutFitnessValues,
    fitnessMean,
  );

  const meanPipesPassed = computeMean(
    episodeResults.map((episodeResult) => episodeResult.pipesPassed),
  );
  const meanFramesSurvived = computeMean(
    episodeResults.map((episodeResult) => episodeResult.framesSurvived),
  );

  return {
    seedCount,
    meanFitness: fitnessMean,
    medianFitness: computePercentile(rolloutFitnessValues, 0.5),
    p90Fitness: computePercentile(rolloutFitnessValues, 0.9),
    fitnessStdDev,
    robustFitness:
      fitnessMean - fitnessStdDev * FLAPPY_EVALUATION_ROBUST_STDDEV_PENALTY,
    meanPipesPassed,
    meanFramesSurvived,
  };
}

async function runResetChannelRolloutEpisode(
  inferenceChannel: ReturnType<typeof openInferenceChannel>,
  networkId: number | undefined,
  rolloutOptions: FlappyRolloutOptions,
) {
  // Step 1: Reset carried predictor state before the new seeded episode begins.
  await inferenceChannel.reset();

  // Step 2: Run the deterministic rollout from a clean predictor state.
  return rolloutEpisodeWithPredictor({
    predict: async (observationVector: number[]) =>
      inferenceChannel.predict(observationVector),
    rolloutOptions,
    networkId,
  });
}

async function runResetSharedWorkerRolloutEpisode(
  sharedInferenceWorker: SharedInferenceWorker,
  networkId: number | undefined,
  rolloutOptions: FlappyRolloutOptions,
) {
  // Step 1: Reset carried predictor state before the new seeded episode begins.
  await sharedInferenceWorker.reset();

  // Step 2: Run the deterministic rollout from a clean predictor state.
  return rolloutEpisodeWithPredictor({
    predict: async (observationVector: number[]) =>
      sharedInferenceWorker.infer(observationVector),
    rolloutOptions,
    networkId,
  });
}

/**
 * Runs one rollout after resetting any carried recurrent network state.
 *
 * Stateful builders such as NARX, GRU, and LSTM must start each deterministic
 * Flappy rollout from a clean memory slate. Feed-forward networks ignore the
 * optional `clear()` hook, but recurrent networks use it to avoid leaking state
 * across shared-seed evaluations.
 *
 * @param network - Network being evaluated.
 * @param rolloutOptions - Rollout controls for this episode.
 * @returns One deterministic episode result.
 */
function runClearedRolloutEpisode(
  network: FlappyNetworkLike,
  rolloutOptions: FlappyRolloutOptions,
) {
  // Step 1: Reset carried network state before the new episode begins.
  network.clear?.();

  // Step 2: Run the deterministic rollout from a clean controller state.
  return rolloutEpisode(network, rolloutOptions);
}
