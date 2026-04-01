import { FLAPPY_EVALUATION_ROBUST_STDDEV_PENALTY } from './evaluation.constants';
import { rolloutEpisode } from './evaluation.rollout.service';
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
  return rolloutEpisode(network, rolloutOptions).fitness;
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
    rolloutEpisode(network, {
      ...rolloutOptions,
      seed: seedValue,
    }),
  );
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
    seedCount: sharedSeeds.length,
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
