import type {
  FlappyRolloutOptions,
  FlappySeedBatchEvaluation,
} from '../../flappyEvaluation';

/**
 * Candidate-stage request used by the staged population evaluator.
 *
 * Keeping this internal contract narrow lets the orchestration service choose
 * a candidate budget without coupling the execution helpers to generation-plan
 * details.
 */
export type PopulationStageEvaluationRequest = {
  candidateCount: number;
  sharedSeeds: readonly number[];
  rolloutOptions: FlappyRolloutOptions;
};

/**
 * Aggregate scoring context shared while computing frame-primary scores.
 */
export type PopulationAggregateScoringContext = {
  aggregateValues: FlappySeedBatchEvaluation[];
  maximumMeanPipesPassed: number;
};
