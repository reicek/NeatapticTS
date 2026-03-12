import type {
  FlappyNetworkLike,
  FlappyRolloutOptions,
} from '../flappyEvaluation';

/**
 * Network shape expected by the Flappy trainer.
 *
 * The trainer only needs the evaluation-facing subset of a full network plus an
 * optional score field used by staged ranking helpers.
 */
export interface FlappyTrainerNetwork extends FlappyNetworkLike {
  score?: number;
}

/**
 * Local typed view for population-level fitness mode used by this trainer.
 *
 * This is intentionally narrower than the full `Neat` runtime API. The trainer
 * documents only the methods and mutable options it actually depends on.
 */
export interface FlappyTrainerNeatController {
  generation: number;
  options: {
    mutationRate: number;
    mutationAmount: number;
    fitnessPopulation?: boolean;
  };
  fitness: (population: FlappyTrainerNetwork[]) => Promise<void>;
  evolve(): Promise<FlappyTrainerNetwork>;
  restoreRNGState(seed: number): void;
}

/**
 * Compact generation report used for training logs.
 *
 * The report is shaped for longitudinal monitoring rather than raw storage. It
 * collects the distribution and best-run details needed to judge whether a
 * generation improved robustly.
 */
export interface FlappyGenerationReport {
  generationIndex: number;
  difficultyScale: number;
  mutationRate: number;
  mutationAmount: number;
  quickSeedCount: number;
  fullSeedCount: number;
  reevaluationSeedCount: number;
  evaluatedPopulationSize: number;
  scoreMean: number;
  scoreMedian: number;
  scoreP90: number;
  scoreStdDev: number;
  bestRobustFitness: number;
  bestMeanFitness: number;
  bestPipesPassed: number;
  bestFramesSurvived: number;
}

/**
 * Trainer runtime state shared by orchestration helpers.
 *
 * Only mutable cross-step values live here: stop intent and the most recent
 * generation report.
 */
export interface FlappyTrainerRuntimeState {
  shouldStop: boolean;
  latestGenerationReport?: FlappyGenerationReport;
}

/**
 * Immutable trainer setup values.
 *
 * These values define the static training shape before runtime state and staged
 * evaluation are attached.
 */
export interface FlappyTrainerSetup {
  inputSize: number;
  outputSize: number;
  populationSize: number;
  elitismCount: number;
}

/**
 * Generation-level rollout plans for staged evaluation.
 *
 * Each generation resolves one plan that answers three questions: how strong is
 * the current mutation schedule, which shared seeds belong to each stage, and
 * what rollout budget each stage is allowed to spend.
 */
export interface FlappyGenerationEvaluationPlan {
  generationIndex: number;
  mutationRate: number;
  mutationAmount: number;
  difficultyScale: number;
  quickSeeds: number[];
  fullSeeds: number[];
  reevaluationSeeds: number[];
  quickRolloutOptions: FlappyRolloutOptions;
  fullRolloutOptions: FlappyRolloutOptions;
  reevaluationRolloutOptions: FlappyRolloutOptions;
}

/**
 * Score carrier used for deterministic ordering helpers.
 *
 * Wrapping a genome together with its score makes ranking utilities easier to
 * write and keeps tie-breaking logic explicit.
 */
export interface ScoredGenomeEntry {
  genome: FlappyTrainerNetwork;
  score: number;
}
