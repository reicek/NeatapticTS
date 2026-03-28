/**
 * Adapter layer that teaches a generic NEAT controller how to score an entire
 * Flappy population fairly.
 *
 * A plain controller only knows that it needs a fitness callback. This service
 * turns that loose contract into the trainer's staged policy: screen everybody
 * cheaply, spend more budget on the survivors, then reevaluate the finalists so
 * ranking is less sensitive to luck.
 *
 * Fitness orchestration map:
 * ```mermaid
 * flowchart LR
 *     Population["population"] --> Plan["resolveGenerationEvaluationPlan()"]
 *     Plan --> Quick["evaluatePopulationQuickStage()"]
 *     Quick --> Full["evaluatePopulationFullStage()"]
 *     Full --> Reeval["evaluatePopulationReevaluationStage()"]
 *     Reeval --> Commit["commitPopulationScores()"]
 *     Commit --> Report["buildGenerationReport()"]
 * ```
 */
import type { FlappySeedBatchEvaluation } from '../flappyEvaluation';
import type {
  FlappyGenerationEvaluationPlan,
  FlappyGenerationReport,
  FlappyTrainerNeatController,
  FlappyTrainerNetwork,
  FlappyTrainerRuntimeState,
} from './trainer.types';

/**
 * Callback dependencies required by the trainer fitness orchestration service.
 *
 * Educational note:
 * The trainer evaluates whole populations in staged passes. This dependency bag
 * keeps the top-level service declarative and makes each stage independently
 * replaceable without rewriting the orchestration logic.
 */
export interface TrainerFitnessServiceDependencies {
  resolveGenerationEvaluationPlan: (
    generationIndex: number,
  ) => FlappyGenerationEvaluationPlan;
  evaluatePopulationQuickStage: (
    population: readonly FlappyTrainerNetwork[],
    generationEvaluationPlan: FlappyGenerationEvaluationPlan,
    aggregateByGenome: Map<FlappyTrainerNetwork, FlappySeedBatchEvaluation>,
    provisionalScoresByGenome: Map<FlappyTrainerNetwork, number>,
  ) => void;
  evaluatePopulationFullStage: (
    population: readonly FlappyTrainerNetwork[],
    generationEvaluationPlan: FlappyGenerationEvaluationPlan,
    aggregateByGenome: Map<FlappyTrainerNetwork, FlappySeedBatchEvaluation>,
    provisionalScoresByGenome: Map<FlappyTrainerNetwork, number>,
    elitismCount: number,
  ) => void;
  evaluatePopulationReevaluationStage: (
    population: readonly FlappyTrainerNetwork[],
    generationEvaluationPlan: FlappyGenerationEvaluationPlan,
    aggregateByGenome: Map<FlappyTrainerNetwork, FlappySeedBatchEvaluation>,
    provisionalScoresByGenome: Map<FlappyTrainerNetwork, number>,
    elitismCount: number,
  ) => void;
  commitPopulationScores: (
    population: readonly FlappyTrainerNetwork[],
    provisionalScoresByGenome: Map<FlappyTrainerNetwork, number>,
  ) => void;
  buildGenerationReport: (
    population: readonly FlappyTrainerNetwork[],
    aggregateByGenome: ReadonlyMap<
      FlappyTrainerNetwork,
      FlappySeedBatchEvaluation
    >,
    generationEvaluationPlan: FlappyGenerationEvaluationPlan,
  ) => FlappyGenerationReport;
}

/**
 * Attaches population-level staged evaluator to the NEAT controller.
 *
 * This is the moment where the generic NEAT controller becomes a
 * Flappy-specific trainer: a plain controller receives the staged population
 * evaluator that understands shared-seed screening, full-pass scoring, and
 * reevaluation.
 *
 * @param neatController - Trainer NEAT controller.
 * @param trainerRuntimeState - Mutable trainer runtime state.
 * @param elitismCount - Number of elite genomes preserved each generation.
 * @param dependencies - Pure/impure helper callbacks used by the evaluator.
 * @returns Nothing.
 */
export function attachPopulationFitnessEvaluator(
  neatController: FlappyTrainerNeatController,
  trainerRuntimeState: FlappyTrainerRuntimeState,
  elitismCount: number,
  dependencies: TrainerFitnessServiceDependencies,
): void {
  neatController.fitness = createPopulationFitnessEvaluator(
    neatController,
    trainerRuntimeState,
    elitismCount,
    dependencies,
  );
}

/**
 * Creates the asynchronous population fitness evaluator.
 *
 * Educational note:
 * The trainer uses staged evaluation to reduce luck. Genomes are first screened
 * quickly, then the most promising ones receive more expensive evaluation, and
 * the best candidates are reevaluated again for robustness.
 *
 * That strategy is closer to tournament design than to naive one-shot scoring:
 * the same generation budget is spent unevenly so weak genomes are filtered out
 * early and strong genomes are compared more carefully.
 *
 * @param neatController - Trainer NEAT controller.
 * @param trainerRuntimeState - Mutable trainer runtime state.
 * @param elitismCount - Number of elite genomes preserved each generation.
 * @param dependencies - Pure/impure helper callbacks used by the evaluator.
 * @returns Evaluator callback assigned to `neatController.fitness`.
 */
export function createPopulationFitnessEvaluator(
  neatController: FlappyTrainerNeatController,
  trainerRuntimeState: FlappyTrainerRuntimeState,
  elitismCount: number,
  dependencies: TrainerFitnessServiceDependencies,
): (population: FlappyTrainerNetwork[]) => Promise<void> {
  async function evaluatePopulationFitness(
    population: FlappyTrainerNetwork[],
  ): Promise<void> {
    const generationEvaluationPlan =
      dependencies.resolveGenerationEvaluationPlan(neatController.generation);
    const provisionalScoresByGenome = new Map<FlappyTrainerNetwork, number>();
    const aggregateByGenome = new Map<
      FlappyTrainerNetwork,
      FlappySeedBatchEvaluation
    >();

    dependencies.evaluatePopulationQuickStage(
      population,
      generationEvaluationPlan,
      aggregateByGenome,
      provisionalScoresByGenome,
    );

    dependencies.evaluatePopulationFullStage(
      population,
      generationEvaluationPlan,
      aggregateByGenome,
      provisionalScoresByGenome,
      elitismCount,
    );

    dependencies.evaluatePopulationReevaluationStage(
      population,
      generationEvaluationPlan,
      aggregateByGenome,
      provisionalScoresByGenome,
      elitismCount,
    );

    dependencies.commitPopulationScores(population, provisionalScoresByGenome);

    trainerRuntimeState.latestGenerationReport =
      dependencies.buildGenerationReport(
        population,
        aggregateByGenome,
        generationEvaluationPlan,
      );
  }

  return evaluatePopulationFitness;
}
