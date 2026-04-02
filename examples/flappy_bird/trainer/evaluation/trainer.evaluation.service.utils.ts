import type { FlappySeedBatchEvaluation } from '../../flappyEvaluation';
import {
  FLAPPY_TRAINER_FRAME_PRIMARY_BASE_SCORE,
  FLAPPY_TRAINER_FRAME_PRIMARY_PIPE_WEIGHT,
  FLAPPY_TRAINER_FRAME_PRIMARY_SURVIVAL_WEIGHT,
  FLAPPY_TRAINER_FRAME_STABILITY_STDDEV_WEIGHT,
  FLAPPY_TRAINER_PIPE_FALLBACK_PIPE_WEIGHT,
  FLAPPY_TRAINER_PIPE_FILTER_TOLERANCE,
} from '../trainer.constants';
import type { FlappyTrainerNetwork } from '../trainer.types';
import {
  FLAPPY_TRAINER_MIN_PIPE_PROGRESS,
  FLAPPY_TRAINER_NEGATIVE_INFINITY_SCORE,
} from './trainer.evaluation.service.constants';
import type { PopulationAggregateScoringContext } from './trainer.evaluation.service.types';

/**
 * Assigns refreshed frame-primary scores to the current population.
 *
 * Educational note:
 * The trainer does not rank genomes purely by one raw metric. It combines pipe
 * progress, survival, and stability into a provisional score so early-stage
 * selection remains robust when several genomes are close in quality.
 *
 * @param population - Current population.
 * @param aggregateByGenome - Aggregate cache keyed by genome.
 * @param provisionalScoresByGenome - Mutable provisional score map.
 * @returns Nothing.
 */
export function assignFramePrimaryScores(
  population: readonly FlappyTrainerNetwork[],
  aggregateByGenome: ReadonlyMap<
    FlappyTrainerNetwork,
    FlappySeedBatchEvaluation
  >,
  provisionalScoresByGenome: Map<FlappyTrainerNetwork, number>,
): void {
  // Step 1: Resolve the aggregate scoring context shared by all genomes.
  const populationAggregateScoringContext =
    resolvePopulationAggregateScoringContext(population, aggregateByGenome);

  // Step 2: Score each genome using the refreshed aggregate context.
  for (const genome of population) {
    const aggregate = aggregateByGenome.get(genome);
    if (!aggregate) {
      provisionalScoresByGenome.set(
        genome,
        FLAPPY_TRAINER_NEGATIVE_INFINITY_SCORE,
      );
      continue;
    }

    provisionalScoresByGenome.set(
      genome,
      scoreAggregateFramePrimary(
        aggregate,
        populationAggregateScoringContext.maximumMeanPipesPassed,
      ),
    );
  }
}

/**
 * Collects all currently available aggregate values.
 *
 * Only genomes with completed aggregate results are included. That lets the
 * scoring helpers distinguish between genuinely weak genomes and genomes that
 * simply have not yet reached a later stage.
 *
 * @param population - Current population.
 * @param aggregateByGenome - Aggregate cache keyed by genome.
 * @returns Collected aggregate values.
 */
export function collectAggregateValues(
  population: readonly FlappyTrainerNetwork[],
  aggregateByGenome: ReadonlyMap<
    FlappyTrainerNetwork,
    FlappySeedBatchEvaluation
  >,
): FlappySeedBatchEvaluation[] {
  const aggregateValues: FlappySeedBatchEvaluation[] = [];

  // Step 1: Include only genomes that already have an aggregate in the cache.
  for (const genome of population) {
    const aggregate = aggregateByGenome.get(genome);
    if (aggregate) {
      aggregateValues.push(aggregate);
    }
  }

  return aggregateValues;
}

/**
 * Resolves the leading mean pipe-progress value across available aggregates.
 *
 * Mean pipe progress acts as the leading indicator for the frame-primary score:
 * if a genome is far behind the current pipe leader, it falls back to a simpler
 * progress-first score.
 *
 * @param aggregateValues - Aggregate values currently available.
 * @returns Highest mean pipe-progress value.
 */
export function resolveMaximumMeanPipesPassed(
  aggregateValues: readonly FlappySeedBatchEvaluation[],
): number {
  // Step 1: Resolve the leading mean pipe-progress value across all aggregates.
  return aggregateValues.reduce(
    (bestPipeProgress, aggregate) =>
      Math.max(bestPipeProgress, aggregate.meanPipesPassed),
    FLAPPY_TRAINER_MIN_PIPE_PROGRESS,
  );
}

/**
 * Scores one aggregate using the frame-primary heuristic.
 *
 * Educational note:
 * The heuristic intentionally mixes progress and stability. A genome that passes
 * many pipes but has wildly inconsistent fitness across seeds is treated more
 * cautiously than a similarly strong but steadier genome.
 *
 * @param aggregate - Aggregate evaluation result.
 * @param maximumMeanPipesPassed - Best mean pipe progress in the population.
 * @returns Provisional score.
 */
export function scoreAggregateFramePrimary(
  aggregate: FlappySeedBatchEvaluation,
  maximumMeanPipesPassed: number,
): number {
  // Step 1: Resolve the stability penalty and pipe-progress eligibility filter.
  const frameStabilityPenalty =
    aggregate.fitnessStdDev * FLAPPY_TRAINER_FRAME_STABILITY_STDDEV_WEIGHT;
  const passesPipeFilter =
    aggregate.meanPipesPassed >=
    maximumMeanPipesPassed - FLAPPY_TRAINER_PIPE_FILTER_TOLERANCE;

  // Step 2: Use the full frame-primary score when the genome remains near the pipe leader.
  if (passesPipeFilter) {
    return (
      FLAPPY_TRAINER_FRAME_PRIMARY_BASE_SCORE +
      aggregate.meanFramesSurvived *
        FLAPPY_TRAINER_FRAME_PRIMARY_SURVIVAL_WEIGHT +
      aggregate.meanPipesPassed * FLAPPY_TRAINER_FRAME_PRIMARY_PIPE_WEIGHT -
      frameStabilityPenalty
    );
  }

  // Step 3: Fall back to the pipe-progress-first score for trailing genomes.
  return (
    aggregate.meanPipesPassed * FLAPPY_TRAINER_PIPE_FALLBACK_PIPE_WEIGHT +
    aggregate.meanFramesSurvived -
    frameStabilityPenalty
  );
}

/**
 * Resolves the aggregate scoring context used by frame-primary scoring.
 *
 * This precomputation step keeps the per-genome scoring loop lean and avoids
 * recomputing population-wide maxima for every genome.
 *
 * @param population - Current population.
 * @param aggregateByGenome - Aggregate cache keyed by genome.
 * @returns Aggregate scoring context.
 */
function resolvePopulationAggregateScoringContext(
  population: readonly FlappyTrainerNetwork[],
  aggregateByGenome: ReadonlyMap<
    FlappyTrainerNetwork,
    FlappySeedBatchEvaluation
  >,
): PopulationAggregateScoringContext {
  // Step 1: Collect the currently available aggregate values.
  const aggregateValues = collectAggregateValues(population, aggregateByGenome);

  // Step 2: Resolve the highest mean pipe-progress value in the population.
  return {
    aggregateValues,
    maximumMeanPipesPassed: resolveMaximumMeanPipesPassed(aggregateValues),
  };
}
