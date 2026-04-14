/**
 * Public type map for the Flappy Bird trainer boundary.
 *
 * The trainer coordinates several moving pieces at once: a generic NEAT
 * controller, staged rollout plans, compact longitudinal reporting, and a very
 * small amount of mutable runtime state. Keeping those contracts together makes
 * the trainer easier to read because the important nouns live in one place.
 *
 * A useful reading order is:
 *
 * 1. `FlappyTrainerSetup` for static configuration.
 * 2. `FlappyTrainerNeatController` for the runtime API shape.
 * 3. `FlappyGenerationEvaluationPlan` for staged evaluation policy.
 * 4. `FlappyGenerationReport` for what the loop emits back out.
 *
 * Type relationship map:
 * ```mermaid
 * flowchart TB
 *     Setup["FlappyTrainerSetup\nstatic sizes and counts"] --> Controller["FlappyTrainerNeatController\nmutable NEAT runtime"]
 *     Controller --> Plan["FlappyGenerationEvaluationPlan\nseeds, rollout budgets, annealing"]
 *     Plan --> Report["FlappyGenerationReport\ngeneration-level summary"]
 *     Runtime["FlappyTrainerRuntimeState\nstop intent + latest report"] --> Report
 *     Network["FlappyTrainerNetwork\nscore-carrying genome"] --> Controller
 *     Network --> Report
 *     ScoreEntry["ScoredGenomeEntry\nranking helper"] --> Network
 * ```
 */
import type {
  FlappyNetworkLike,
  FlappyRolloutOptions,
} from '../flappyEvaluation';
import type { ExampleArchitectureProfileId } from '../../architectureProfiles';

/**
 * Network shape expected by the Flappy trainer.
 *
 * The trainer only needs the evaluation-facing subset of a full network plus an
 * optional score field used by staged ranking helpers. That narrow shape keeps
 * the trainer decoupled from most of the broader network implementation.
 */
export interface FlappyTrainerNetwork extends FlappyNetworkLike {
  score?: number;
}

/**
 * Local typed view for population-level fitness mode used by this trainer.
 *
 * This is intentionally narrower than the full `Neat` runtime API. The trainer
 * documents only the methods and mutable options it actually depends on, which
 * makes the orchestration code read more like policy and less like framework
 * plumbing.
 */
export interface FlappyTrainerNeatController {
  generation: number;
  options: {
    allowRecurrent?: boolean;
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
 * generation improved robustly instead of producing one lucky outlier.
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
 * generation report. That tiny scope is deliberate because it keeps the rest of
 * the trainer easy to reason about during long-running sessions.
 */
export interface FlappyTrainerRuntimeState {
  shouldStop: boolean;
  latestGenerationReport?: FlappyGenerationReport;
}

/**
 * Immutable trainer setup values.
 *
 * These values define the static training shape before runtime state and staged
 * evaluation are attached. Once created, the rest of the trainer can treat this
 * as a stable configuration shelf rather than scattered ad hoc constants.
 */
export interface FlappyTrainerSetup {
  architectureProfileId: ExampleArchitectureProfileId;
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
 * what rollout budget each stage is allowed to spend. This is the contract that
 * makes quick screening, full passes, and reevaluation feel like one coherent
 * policy instead of three unrelated helper calls.
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
 * write and keeps score extraction explicit at sort time instead of letting it
 * leak across multiple helpers.
 */
export interface ScoredGenomeEntry {
  genome: FlappyTrainerNetwork;
  score: number;
}
