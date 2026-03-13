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
 * Educational note:
 * The trainer logs more than a single best score because single-number progress
 * can hide instability. Mean, median, $p90$, and standard deviation reveal
 * whether a generation is broadly improving or whether one lucky genome is
 * masking a weak population.
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
 * The emitted line is designed for long-running terminal sessions: dense enough
 * to be useful, but stable enough that humans can visually scan progress over
 * hundreds of generations.
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

  console.log(logParts.join(FLAPPY_TRAINER_LOG_PARTS_DELIMITER));
}
