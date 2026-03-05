import type {
  FlappyNetworkLike,
  FlappyRolloutOptions,
} from '../flappyEvaluation';

/** Network shape expected by the Flappy trainer. */
export interface FlappyTrainerNetwork extends FlappyNetworkLike {
  score?: number;
}

/** Local typed view for population-level fitness mode used by this trainer. */
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

/** Compact generation report used for training logs. */
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

/** Trainer runtime state shared by orchestration helpers. */
export interface FlappyTrainerRuntimeState {
  shouldStop: boolean;
  latestGenerationReport?: FlappyGenerationReport;
}

/** Immutable trainer setup values. */
export interface FlappyTrainerSetup {
  inputSize: number;
  outputSize: number;
  populationSize: number;
  elitismCount: number;
}

/** Generation-level rollout plans for staged evaluation. */
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

/** Score carrier used for deterministic ordering helpers. */
export interface ScoredGenomeEntry {
  genome: FlappyTrainerNetwork;
  score: number;
}
