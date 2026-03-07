import {
  evaluateFlappyFitnessAcrossSeeds,
  type FlappyRolloutOptions,
  type FlappySeedBatchEvaluation,
} from '../flappyEvaluation';
import {
  FLAPPY_TRAINER_FRAME_PRIMARY_BASE_SCORE,
  FLAPPY_TRAINER_FRAME_PRIMARY_PIPE_WEIGHT,
  FLAPPY_TRAINER_FRAME_PRIMARY_SURVIVAL_WEIGHT,
  FLAPPY_TRAINER_FRAME_STABILITY_STDDEV_WEIGHT,
  FLAPPY_TRAINER_FULL_PASS_ELITISM_MULTIPLIER,
  FLAPPY_TRAINER_FULL_PASS_POPULATION_FRACTION,
  FLAPPY_TRAINER_PIPE_FALLBACK_PIPE_WEIGHT,
  FLAPPY_TRAINER_PIPE_FILTER_TOLERANCE,
  FLAPPY_TRAINER_REEVALUATION_MIN_CANDIDATE_COUNT,
} from './trainer.constants';
import { selectTopGenomesByScore } from './trainer.selection.utils';
import type {
  FlappyGenerationEvaluationPlan,
  FlappyTrainerNetwork,
} from './trainer.types';

/**
 * Executes the quick evaluation stage over the full population.
 *
 * @param population - Current population.
 * @param generationEvaluationPlan - Per-generation staged evaluation plan.
 * @param aggregateByGenome - Mutable aggregate cache keyed by genome.
 * @param provisionalScoresByGenome - Mutable provisional score map.
 * @returns Nothing.
 */
export function evaluatePopulationQuickStage(
  population: readonly FlappyTrainerNetwork[],
  generationEvaluationPlan: FlappyGenerationEvaluationPlan,
  aggregateByGenome: Map<FlappyTrainerNetwork, FlappySeedBatchEvaluation>,
  provisionalScoresByGenome: Map<FlappyTrainerNetwork, number>,
): void {
  evaluateSpecificGenomesAcrossSeeds(
    population,
    generationEvaluationPlan.quickSeeds,
    generationEvaluationPlan.quickRolloutOptions,
    aggregateByGenome,
  );

  assignFramePrimaryScores(
    population,
    aggregateByGenome,
    provisionalScoresByGenome,
  );
}

/**
 * Executes the full evaluation stage over the top provisional candidates.
 *
 * @param population - Current population.
 * @param generationEvaluationPlan - Per-generation staged evaluation plan.
 * @param aggregateByGenome - Mutable aggregate cache keyed by genome.
 * @param provisionalScoresByGenome - Mutable provisional score map.
 * @param elitismCount - Configured elitism count.
 * @returns Nothing.
 */
export function evaluatePopulationFullStage(
  population: readonly FlappyTrainerNetwork[],
  generationEvaluationPlan: FlappyGenerationEvaluationPlan,
  aggregateByGenome: Map<FlappyTrainerNetwork, FlappySeedBatchEvaluation>,
  provisionalScoresByGenome: Map<FlappyTrainerNetwork, number>,
  elitismCount: number,
): void {
  const fullPassCandidateCount = resolveFullPassCandidateCount(
    population.length,
    elitismCount,
  );
  const fullPassCandidates = selectTopGenomesByScore(
    population,
    provisionalScoresByGenome,
    fullPassCandidateCount,
  );

  evaluateSpecificGenomesAcrossSeeds(
    fullPassCandidates,
    generationEvaluationPlan.fullSeeds,
    generationEvaluationPlan.fullRolloutOptions,
    aggregateByGenome,
  );

  assignFramePrimaryScores(
    population,
    aggregateByGenome,
    provisionalScoresByGenome,
  );
}

/**
 * Executes the large-seed reevaluation stage over top candidates.
 *
 * @param population - Current population.
 * @param generationEvaluationPlan - Per-generation staged evaluation plan.
 * @param aggregateByGenome - Mutable aggregate cache keyed by genome.
 * @param provisionalScoresByGenome - Mutable provisional score map.
 * @param elitismCount - Configured elitism count.
 * @returns Nothing.
 */
export function evaluatePopulationReevaluationStage(
  population: readonly FlappyTrainerNetwork[],
  generationEvaluationPlan: FlappyGenerationEvaluationPlan,
  aggregateByGenome: Map<FlappyTrainerNetwork, FlappySeedBatchEvaluation>,
  provisionalScoresByGenome: Map<FlappyTrainerNetwork, number>,
  elitismCount: number,
): void {
  const reevaluationCount = Math.max(
    elitismCount,
    FLAPPY_TRAINER_REEVALUATION_MIN_CANDIDATE_COUNT,
  );
  const reevaluationCandidates = selectTopGenomesByScore(
    population,
    provisionalScoresByGenome,
    reevaluationCount,
  );

  evaluateSpecificGenomesAcrossSeeds(
    reevaluationCandidates,
    generationEvaluationPlan.reevaluationSeeds,
    generationEvaluationPlan.reevaluationRolloutOptions,
    aggregateByGenome,
  );

  assignFramePrimaryScores(
    population,
    aggregateByGenome,
    provisionalScoresByGenome,
  );
}

/**
 * Commits provisional scores to genome score fields.
 *
 * @param population - Current population.
 * @param provisionalScoresByGenome - Final provisional score map.
 * @returns Nothing.
 */
export function commitPopulationScores(
  population: readonly FlappyTrainerNetwork[],
  provisionalScoresByGenome: ReadonlyMap<FlappyTrainerNetwork, number>,
): void {
  for (const genome of population) {
    genome.score =
      provisionalScoresByGenome.get(genome) ?? Number.NEGATIVE_INFINITY;
  }
}

function evaluateSpecificGenomesAcrossSeeds(
  genomes: readonly FlappyTrainerNetwork[],
  sharedSeeds: readonly number[],
  rolloutOptions: FlappyRolloutOptions,
  aggregateByGenome: Map<FlappyTrainerNetwork, FlappySeedBatchEvaluation>,
): void {
  for (const genome of genomes) {
    const aggregate = evaluateFlappyFitnessAcrossSeeds(
      genome,
      sharedSeeds,
      rolloutOptions,
    );
    aggregateByGenome.set(genome, aggregate);
  }
}

function assignFramePrimaryScores(
  population: readonly FlappyTrainerNetwork[],
  aggregateByGenome: ReadonlyMap<
    FlappyTrainerNetwork,
    FlappySeedBatchEvaluation
  >,
  provisionalScoresByGenome: Map<FlappyTrainerNetwork, number>,
): void {
  const aggregateValues = collectAggregateValues(population, aggregateByGenome);
  const maximumMeanPipesPassed = resolveMaximumMeanPipesPassed(aggregateValues);

  for (const genome of population) {
    const aggregate = aggregateByGenome.get(genome);
    if (!aggregate) {
      provisionalScoresByGenome.set(genome, Number.NEGATIVE_INFINITY);
      continue;
    }

    provisionalScoresByGenome.set(
      genome,
      scoreAggregateFramePrimary(aggregate, maximumMeanPipesPassed),
    );
  }
}

function collectAggregateValues(
  population: readonly FlappyTrainerNetwork[],
  aggregateByGenome: ReadonlyMap<
    FlappyTrainerNetwork,
    FlappySeedBatchEvaluation
  >,
): FlappySeedBatchEvaluation[] {
  const aggregateValues: FlappySeedBatchEvaluation[] = [];

  for (const genome of population) {
    const aggregate = aggregateByGenome.get(genome);
    if (aggregate) {
      aggregateValues.push(aggregate);
    }
  }

  return aggregateValues;
}

function resolveMaximumMeanPipesPassed(
  aggregateValues: readonly FlappySeedBatchEvaluation[],
): number {
  return aggregateValues.reduce(
    (bestPipeProgress, aggregate) =>
      Math.max(bestPipeProgress, aggregate.meanPipesPassed),
    0,
  );
}

function scoreAggregateFramePrimary(
  aggregate: FlappySeedBatchEvaluation,
  maximumMeanPipesPassed: number,
): number {
  const frameStabilityPenalty =
    aggregate.fitnessStdDev * FLAPPY_TRAINER_FRAME_STABILITY_STDDEV_WEIGHT;
  const passesPipeFilter =
    aggregate.meanPipesPassed >=
    maximumMeanPipesPassed - FLAPPY_TRAINER_PIPE_FILTER_TOLERANCE;

  if (passesPipeFilter) {
    return (
      FLAPPY_TRAINER_FRAME_PRIMARY_BASE_SCORE +
      aggregate.meanFramesSurvived *
        FLAPPY_TRAINER_FRAME_PRIMARY_SURVIVAL_WEIGHT +
      aggregate.meanPipesPassed * FLAPPY_TRAINER_FRAME_PRIMARY_PIPE_WEIGHT -
      frameStabilityPenalty
    );
  }

  return (
    aggregate.meanPipesPassed * FLAPPY_TRAINER_PIPE_FALLBACK_PIPE_WEIGHT +
    aggregate.meanFramesSurvived -
    frameStabilityPenalty
  );
}

function resolveFullPassCandidateCount(
  populationSize: number,
  elitismCount: number,
): number {
  return Math.max(
    elitismCount * FLAPPY_TRAINER_FULL_PASS_ELITISM_MULTIPLIER,
    Math.floor(populationSize * FLAPPY_TRAINER_FULL_PASS_POPULATION_FRACTION),
  );
}
