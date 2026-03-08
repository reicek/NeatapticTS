import type { FlappySeedBatchEvaluation } from '../../flappyEvaluation';
import {
  FLAPPY_TRAINER_FULL_PASS_ELITISM_MULTIPLIER,
  FLAPPY_TRAINER_FULL_PASS_POPULATION_FRACTION,
  FLAPPY_TRAINER_REEVALUATION_MIN_CANDIDATE_COUNT,
} from '../trainer.constants';
import type {
  FlappyGenerationEvaluationPlan,
  FlappyTrainerNetwork,
} from '../trainer.types';
import { FLAPPY_TRAINER_NEGATIVE_INFINITY_SCORE } from './trainer.evaluation.service.constants';
import {
  evaluatePopulationSelectedCandidateStage,
  evaluateSpecificGenomesAcrossSeeds,
} from './trainer.evaluation.service.services';
import { assignFramePrimaryScores } from './trainer.evaluation.service.utils';
import type { PopulationStageEvaluationRequest } from './trainer.evaluation.service.types';

/**
 * Executes the quick evaluation stage over the full population.
 *
 * @param population - Current population.
 * @param generationEvaluationPlan - Per-generation staged evaluation plan.
 * @param aggregateByGenome - Mutable aggregate cache keyed by genome.
 * @param provisionalScoresByGenome - Mutable provisional score map.
 * @returns Nothing.
 *
 * @example
 * ```ts
 * evaluatePopulationQuickStage(
 *   population,
 *   generationEvaluationPlan,
 *   aggregateByGenome,
 *   provisionalScoresByGenome,
 * );
 * ```
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
  // Step 1: Resolve the candidate-stage request for the full evaluation pass.
  const populationStageEvaluationRequest = {
    candidateCount: resolveFullPassCandidateCount(
      population.length,
      elitismCount,
    ),
    sharedSeeds: generationEvaluationPlan.fullSeeds,
    rolloutOptions: generationEvaluationPlan.fullRolloutOptions,
  } satisfies PopulationStageEvaluationRequest;

  // Step 2: Evaluate the selected candidates and refresh provisional scores.
  evaluatePopulationSelectedCandidateStage(
    population,
    populationStageEvaluationRequest,
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
  // Step 1: Resolve the candidate-stage request for reevaluation.
  const populationStageEvaluationRequest = {
    candidateCount: Math.max(
      elitismCount,
      FLAPPY_TRAINER_REEVALUATION_MIN_CANDIDATE_COUNT,
    ),
    sharedSeeds: generationEvaluationPlan.reevaluationSeeds,
    rolloutOptions: generationEvaluationPlan.reevaluationRolloutOptions,
  } satisfies PopulationStageEvaluationRequest;

  // Step 2: Evaluate the selected candidates and refresh provisional scores.
  evaluatePopulationSelectedCandidateStage(
    population,
    populationStageEvaluationRequest,
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
  // Step 1: Copy provisional scores into the mutable genome score fields.
  for (const genome of population) {
    genome.score =
      provisionalScoresByGenome.get(genome) ??
      FLAPPY_TRAINER_NEGATIVE_INFINITY_SCORE;
  }
}

/**
 * Resolves how many genomes should advance to the full-pass stage.
 *
 * @param populationSize - Population size.
 * @param elitismCount - Configured elitism count.
 * @returns Full-pass candidate count.
 */
function resolveFullPassCandidateCount(
  populationSize: number,
  elitismCount: number,
): number {
  // Step 1: Use the larger of the elitism-based and population-fraction candidate budgets.
  return Math.max(
    elitismCount * FLAPPY_TRAINER_FULL_PASS_ELITISM_MULTIPLIER,
    Math.floor(populationSize * FLAPPY_TRAINER_FULL_PASS_POPULATION_FRACTION),
  );
}
