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
 * Educational note:
 * This helper is the workhorse behind the full and reevaluation stages. It
 * turns a stage request into three steps: pick candidates, evaluate them across
 * shared seeds, then refresh the provisional ranking for the whole population.
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
 * Shared seeds are the fairness mechanism in this trainer. Every selected genome
 * sees the same randomized episode batch for the stage, so comparisons are much
 * less noisy than per-genome private seed sampling.
 *
 * For background reading, the Wikipedia article on "control variates" is a good
 * intuition pump for why holding part of the randomness fixed can reduce
 * variance when comparing alternatives.
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
