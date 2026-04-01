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
 *
 * This shape is intentionally stage-agnostic: quick, full, and reevaluation can
 * all use the same execution helper by changing only candidate count, seed set,
 * and rollout budget.
 */
export type PopulationStageEvaluationRequest = {
  candidateCount: number;
  sharedSeeds: readonly number[];
  rolloutOptions: FlappyRolloutOptions;
};

/**
 * Aggregate scoring context shared while computing frame-primary scores.
 *
 * The context precomputes population-wide reference values so per-genome scoring
 * can stay simple and deterministic.
 */
export type PopulationAggregateScoringContext = {
  aggregateValues: FlappySeedBatchEvaluation[];
  maximumMeanPipesPassed: number;
};
