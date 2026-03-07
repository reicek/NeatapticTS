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
  FLAPPY_TRAINER_LOG_PARTS_DELIMITER,
  FLAPPY_TRAINER_SCORE_MEDIAN_PERCENTILE,
  FLAPPY_TRAINER_SCORE_P90_PERCENTILE,
} from './trainer.constants';
import type { FlappyMutationSchedule } from './trainer.evaluation-plan.utils';
import { buildGenerationLogParts } from './trainer.reporting.utils';
import { resolveBestGenomeByScore } from './trainer.selection.utils';
import {
  computeMean,
  computePercentile,
  computePopulationStandardDeviation,
} from './trainer.statistics.utils';
import type {
  FlappyGenerationEvaluationPlan,
  FlappyGenerationReport,
  FlappyTrainerNetwork,
} from './trainer.types';

/**
 * Builds a compact report for the current generation.
 *
 * @param population - Current population.
 * @param aggregateByGenome - Aggregate evaluation results keyed by genome.
 * @param generationEvaluationPlan - Per-generation staged evaluation plan.
 * @returns Aggregated generation report.
 */
export function buildGenerationReport(
  population: readonly FlappyTrainerNetwork[],
  aggregateByGenome: ReadonlyMap<
    FlappyTrainerNetwork,
    FlappySeedBatchEvaluation
  >,
  generationEvaluationPlan: FlappyGenerationEvaluationPlan,
): FlappyGenerationReport {
  const finalScores = collectFiniteGenomeScores(population);
  const scoreMean = computeMean(finalScores);
  const scoreStdDev = computePopulationStandardDeviation(
    finalScores,
    scoreMean,
  );
  const bestGenome = resolveBestGenomeByScore(population);
  const bestAggregate = resolveBestAggregate(
    population,
    bestGenome,
    aggregateByGenome,
    generationEvaluationPlan.reevaluationSeeds,
    generationEvaluationPlan.reevaluationRolloutOptions,
  );
  const bestEpisode = resolveBestEpisode(
    population,
    bestGenome,
    generationEvaluationPlan.reevaluationSeeds,
    generationEvaluationPlan.reevaluationRolloutOptions,
  );

  return {
    generationIndex: generationEvaluationPlan.generationIndex,
    difficultyScale: generationEvaluationPlan.difficultyScale,
    mutationRate: generationEvaluationPlan.mutationRate,
    mutationAmount: generationEvaluationPlan.mutationAmount,
    quickSeedCount: generationEvaluationPlan.quickSeeds.length,
    fullSeedCount: generationEvaluationPlan.fullSeeds.length,
    reevaluationSeedCount: generationEvaluationPlan.reevaluationSeeds.length,
    evaluatedPopulationSize: population.length,
    scoreMean,
    scoreMedian: computePercentile(
      finalScores,
      FLAPPY_TRAINER_SCORE_MEDIAN_PERCENTILE,
    ),
    scoreP90: computePercentile(
      finalScores,
      FLAPPY_TRAINER_SCORE_P90_PERCENTILE,
    ),
    scoreStdDev,
    bestRobustFitness: bestGenome?.score ?? Number.NaN,
    bestMeanFitness: bestAggregate.meanFitness,
    bestPipesPassed: bestEpisode.pipesPassed,
    bestFramesSurvived: bestEpisode.framesSurvived,
  };
}

/**
 * Emits one compact generation log line.
 *
 * @param generationLabel - Current generation label.
 * @param mutationSchedule - Active mutation schedule.
 * @param report - Optional aggregated generation report.
 * @param fittestGenome - Fittest genome returned by the NEAT controller.
 * @param fallbackEpisode - Fallback representative rollout episode.
 * @returns Nothing.
 */
export function logGenerationSummary(
  generationLabel: number,
  mutationSchedule: FlappyMutationSchedule,
  report: FlappyGenerationReport | undefined,
  fittestGenome: FlappyTrainerNetwork,
  fallbackEpisode: ReturnType<typeof rolloutEpisode>,
): void {
  const bestFitness =
    report?.bestRobustFitness ??
    (fittestGenome.score as number) ??
    fallbackEpisode.fitness;
  const bestPipesPassed =
    report?.bestPipesPassed ?? fallbackEpisode.pipesPassed;
  const bestFramesSurvived =
    report?.bestFramesSurvived ?? fallbackEpisode.framesSurvived;

  const logParts = buildGenerationLogParts(
    generationLabel,
    bestFitness,
    bestPipesPassed,
    bestFramesSurvived,
    report,
    mutationSchedule,
  );

  // eslint-disable-next-line no-console
  console.log(logParts.join(FLAPPY_TRAINER_LOG_PARTS_DELIMITER));
}

function collectFiniteGenomeScores(
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
