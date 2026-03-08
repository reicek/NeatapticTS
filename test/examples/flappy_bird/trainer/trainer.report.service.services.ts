import {
  evaluateFlappyFitnessAcrossSeeds,
  rolloutEpisode,
  type FlappyRolloutOptions,
  type FlappySeedBatchEvaluation,
} from '../flappyEvaluation';
import {
  FLAPPY_TRAINER_DUMMY_FLAP_OUTPUT,
  FLAPPY_TRAINER_DUMMY_NETWORK_ID,
  FLAPPY_TRAINER_DUMMY_NO_FLAP_OUTPUT,
} from './trainer.constants';
import type { FlappyTrainerNetwork } from './trainer.types';

/**
 * Aggregate and representative rollout resolved for the best genome.
 *
 * Keeping these values together lets the report facade stay focused on
 * orchestration while this helper module owns cache fallback behavior.
 */
export interface ResolvedBestGenerationDetails {
  bestAggregate: FlappySeedBatchEvaluation;
  bestEpisode: ReturnType<typeof rolloutEpisode>;
}

/**
 * Collects only finite scores from the current population.
 *
 * Unevaluated or invalid scores are intentionally skipped so percentile and
 * standard deviation calculations operate on stable numeric inputs only.
 *
 * @param population - Current population.
 * @returns Finite scores in population order.
 * @example
 * ```ts
 * const scores = collectFiniteGenomeScores(population);
 * ```
 */
export function collectFiniteGenomeScores(
  population: readonly FlappyTrainerNetwork[],
): number[] {
  const finiteScores: number[] = [];

  for (const genome of population) {
    const genomeScore = genome.score ?? Number.NEGATIVE_INFINITY;
    if (Number.isFinite(genomeScore)) {
      finiteScores.push(genomeScore);
    }
  }

  return finiteScores;
}

/**
 * Resolves cached or fallback best-of-generation details for reporting.
 *
 * The report layer needs both aggregate seed statistics and one representative
 * episode. This helper centralizes the fallback rules so the service facade can
 * remain a thin orchestration layer.
 *
 * @param population - Current population.
 * @param bestGenome - Genome selected as generation best.
 * @param aggregateByGenome - Cached aggregate evaluations keyed by genome.
 * @param fallbackSeeds - Seeds used when the aggregate must be recomputed.
 * @param fallbackRolloutOptions - Rollout options for fallback evaluation.
 * @returns Aggregate metrics and a representative best-genome episode.
 * @example
 * ```ts
 * const { bestAggregate, bestEpisode } = resolveBestGenerationDetails(
 *   population,
 *   bestGenome,
 *   aggregateByGenome,
 *   reevaluationSeeds,
 *   reevaluationRolloutOptions,
 * );
 * ```
 */
export function resolveBestGenerationDetails(
  population: readonly FlappyTrainerNetwork[],
  bestGenome: FlappyTrainerNetwork | undefined,
  aggregateByGenome: ReadonlyMap<
    FlappyTrainerNetwork,
    FlappySeedBatchEvaluation
  >,
  fallbackSeeds: readonly number[],
  fallbackRolloutOptions: FlappyRolloutOptions,
): ResolvedBestGenerationDetails {
  return {
    bestAggregate: resolveBestAggregate(
      population,
      bestGenome,
      aggregateByGenome,
      fallbackSeeds,
      fallbackRolloutOptions,
    ),
    bestEpisode: resolveBestEpisode(
      population,
      bestGenome,
      fallbackSeeds,
      fallbackRolloutOptions,
    ),
  };
}

function resolveBestAggregate(
  population: readonly FlappyTrainerNetwork[],
  bestGenome: FlappyTrainerNetwork | undefined,
  aggregateByGenome: ReadonlyMap<
    FlappyTrainerNetwork,
    FlappySeedBatchEvaluation
  >,
  fallbackSeeds: readonly number[],
  fallbackRolloutOptions: FlappyRolloutOptions,
): FlappySeedBatchEvaluation {
  const fallbackGenome = resolveFallbackGenome(population, bestGenome);
  if (!fallbackGenome) {
    return buildEmptySeedBatchEvaluation();
  }

  const cachedAggregate = aggregateByGenome.get(fallbackGenome);
  if (cachedAggregate) {
    return cachedAggregate;
  }

  return evaluateFlappyFitnessAcrossSeeds(
    fallbackGenome,
    fallbackSeeds,
    fallbackRolloutOptions,
  );
}

function resolveBestEpisode(
  population: readonly FlappyTrainerNetwork[],
  bestGenome: FlappyTrainerNetwork | undefined,
  fallbackSeeds: readonly number[],
  fallbackRolloutOptions: FlappyRolloutOptions,
): ReturnType<typeof rolloutEpisode> {
  const fallbackGenome = resolveFallbackGenome(population, bestGenome);
  if (!fallbackGenome) {
    return rolloutEpisode(resolveDummyNetwork(), fallbackRolloutOptions);
  }

  const episodeSeed = fallbackSeeds[0];
  return rolloutEpisode(fallbackGenome, {
    ...fallbackRolloutOptions,
    seed: episodeSeed,
  });
}

function resolveFallbackGenome(
  population: readonly FlappyTrainerNetwork[],
  bestGenome: FlappyTrainerNetwork | undefined,
): FlappyTrainerNetwork | undefined {
  return bestGenome ?? population[0];
}

function buildEmptySeedBatchEvaluation(): FlappySeedBatchEvaluation {
  return {
    seedCount: 0,
    meanFitness: 0,
    medianFitness: 0,
    p90Fitness: 0,
    fitnessStdDev: 0,
    robustFitness: 0,
    meanPipesPassed: 0,
    meanFramesSurvived: 0,
  };
}

function resolveDummyNetwork(): FlappyTrainerNetwork {
  return {
    activate: activateWithoutFlap,
    _id: FLAPPY_TRAINER_DUMMY_NETWORK_ID,
  };
}

function activateWithoutFlap(): number[] {
  return [
    FLAPPY_TRAINER_DUMMY_NO_FLAP_OUTPUT,
    FLAPPY_TRAINER_DUMMY_FLAP_OUTPUT,
  ];
}
