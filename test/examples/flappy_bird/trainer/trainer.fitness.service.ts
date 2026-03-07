import type { FlappySeedBatchEvaluation } from '../flappyEvaluation';
import type {
  FlappyGenerationEvaluationPlan,
  FlappyGenerationReport,
  FlappyTrainerNeatController,
  FlappyTrainerNetwork,
  FlappyTrainerRuntimeState,
} from './trainer.types';

/** Callback dependencies required by the trainer fitness orchestration service. */
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
 * @param neatController - Trainer NEAT controller.
 * @param trainerRuntimeState - Mutable trainer runtime state.
 * @param elitismCount - Number of elite genomes preserved each generation.
 * @param dependencies - Pure/impure helper callbacks used by the evaluator.
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
