import {
  rolloutEpisode,
  type FlappySeedBatchEvaluation,
} from '../flappyEvaluation';
import {
  FLAPPY_TRAINER_LOG_PARTS_DELIMITER,
  FLAPPY_TRAINER_SCORE_MEDIAN_PERCENTILE,
  FLAPPY_TRAINER_SCORE_P90_PERCENTILE,
} from './trainer.constants';
import type { FlappyMutationSchedule } from './trainer.evaluation-plan.utils';
import {
  collectFiniteGenomeScores,
  resolveBestGenerationDetails,
} from './trainer.report.service.services';
import { buildGenerationLogParts } from './trainer.reporting.utils';
import { resolveBestGenomeByScore } from './trainer.selection.utils';
import {
  computeMean,
  computePercentile,
  computePopulationStandardDeviation,
} from '../flappy.simulation.shared.utils';
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
  // Step 1: Collect stable numeric inputs for distribution statistics.
  const finalScores = collectFiniteGenomeScores(population);
  const scoreMean = computeMean(finalScores);
  const scoreStdDev = computePopulationStandardDeviation(
    finalScores,
    scoreMean,
  );

  // Step 2: Resolve best-genome derived metrics through the report helper boundary.
  const bestGenome = resolveBestGenomeByScore(population);
  const { bestAggregate, bestEpisode } = resolveBestGenerationDetails(
    population,
    bestGenome,
    aggregateByGenome,
    generationEvaluationPlan.reevaluationSeeds,
    generationEvaluationPlan.reevaluationRolloutOptions,
  );

  // Step 3: Fold the resolved metrics into the compact generation report.
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
