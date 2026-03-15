import type {
  GenomeForEvaluation,
  NeatControllerForEval,
} from '../shared/evaluate.types';

/**
 * Fitness-evaluation helpers for the NEAT evaluate chapter.
 *
 * This chapter owns the first part of evaluation: decide whether the fitness
 * delegate runs once for the whole population or once per genome, and clear
 * per-genome runtime state when the controller requests it.
 */

/**
 * Run the configured fitness delegate.
 *
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 * @returns Promise that resolves after fitness evaluation completes.
 */
export async function runFitnessEvaluation(
  controller: NeatControllerForEval,
  evaluationOptions: NeatControllerForEval['options'],
): Promise<void> {
  // Step 1: Use the population-level delegate when requested.
  if (evaluationOptions.fitnessPopulation) {
    clearGenomeStateIfRequested(controller, evaluationOptions, (genome) =>
      genome.clear?.(),
    );
    await controller.fitness(controller.population);
    return;
  }

  // Step 2: Otherwise evaluate each genome independently.
  for (const genome of controller.population) {
    if (evaluationOptions.clear && genome.clear) genome.clear();
    const fitnessValue = await controller.fitness(genome);
    genome.score = fitnessValue as number;
  }
}

/**
 * Clear cached genome state before evaluation when configured.
 *
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 * @param clearAction - Action that clears a genome's internal state.
 */
function clearGenomeStateIfRequested(
  controller: NeatControllerForEval,
  evaluationOptions: NeatControllerForEval['options'],
  clearAction: (genome: GenomeForEvaluation) => void,
): void {
  // Step 1: Clear genome state only when the option is enabled.
  if (!evaluationOptions.clear) return;
  controller.population.forEach((genome) => clearAction(genome));
}
