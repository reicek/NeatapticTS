import {
  evaluateFlappyFitnessAcrossSeeds,
  type FlappyRolloutOptions,
  type FlappySeedBatchEvaluation,
} from '../../flappyEvaluation';
import { selectTopGenomesByScore } from '../trainer.selection.utils';
import type { FlappyTrainerNetwork } from '../trainer.types';
import { assignFramePrimaryScores } from './trainer.evaluation.service.utils';
import type { PopulationStageEvaluationRequest } from './trainer.evaluation.service.types';

/**
 * Evaluates a selected candidate subset for a population stage.
 *
 * @param population - Current population.
 * @param populationStageEvaluationRequest - Candidate-stage evaluation request.
 * @param aggregateByGenome - Mutable aggregate cache keyed by genome.
 * @param provisionalScoresByGenome - Mutable provisional score map.
 * @returns Nothing.
 */
export function evaluatePopulationSelectedCandidateStage(
  population: readonly FlappyTrainerNetwork[],
  populationStageEvaluationRequest: PopulationStageEvaluationRequest,
  aggregateByGenome: Map<FlappyTrainerNetwork, FlappySeedBatchEvaluation>,
  provisionalScoresByGenome: Map<FlappyTrainerNetwork, number>,
): void {
  // Step 1: Select the top-scoring candidates for the requested stage.
  const selectedCandidates = selectTopGenomesByScore(
    population,
    provisionalScoresByGenome,
    populationStageEvaluationRequest.candidateCount,
  );

  // Step 2: Evaluate the selected candidates across the stage's shared seeds.
  evaluateSpecificGenomesAcrossSeeds(
    selectedCandidates,
    populationStageEvaluationRequest.sharedSeeds,
    populationStageEvaluationRequest.rolloutOptions,
    aggregateByGenome,
  );

  // Step 3: Refresh frame-primary scores for the full population.
  assignFramePrimaryScores(
    population,
    aggregateByGenome,
    provisionalScoresByGenome,
  );
}

/**
 * Evaluates a specific genome subset across shared seeds.
 *
 * @param genomes - Genomes selected for evaluation.
 * @param sharedSeeds - Shared deterministic seeds.
 * @param rolloutOptions - Rollout options for this stage.
 * @param aggregateByGenome - Mutable aggregate cache keyed by genome.
 * @returns Nothing.
 */
export function evaluateSpecificGenomesAcrossSeeds(
  genomes: readonly FlappyTrainerNetwork[],
  sharedSeeds: readonly number[],
  rolloutOptions: FlappyRolloutOptions,
  aggregateByGenome: Map<FlappyTrainerNetwork, FlappySeedBatchEvaluation>,
): void {
  // Step 1: Evaluate each selected genome independently across the shared batch.
  for (const genome of genomes) {
    const aggregate = evaluateFlappyFitnessAcrossSeeds(
      genome,
      sharedSeeds,
      rolloutOptions,
    );
    aggregateByGenome.set(genome, aggregate);
  }
}
