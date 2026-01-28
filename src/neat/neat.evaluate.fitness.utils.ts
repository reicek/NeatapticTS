import type {
  GenomeForEvaluation,
  NeatControllerForEval,
} from './neat.evaluate.types.utils';

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 * @returns Promise<void> after fitness evaluation completes.
 */
export async function runFitnessEvaluation(
  controller: NeatControllerForEval,
  evaluationOptions: NeatControllerForEval['options'],
): Promise<void> {
  // Step 1: Select population-level or per-genome fitness evaluation.
  if (evaluationOptions.fitnessPopulation) {
    // Step 2: Clear genome state when requested.
    clearGenomeStateIfRequested(controller, evaluationOptions, (genome) =>
      genome.clear?.(),
    );
    // Step 3: Execute population-level fitness.
    await controller.fitness(controller.population);
    return;
  }

  // Step 2: Evaluate each genome individually.
  for (const genome of controller.population) {
    if (evaluationOptions.clear && genome.clear) genome.clear();
    const fitnessValue = await controller.fitness(genome);
    genome.score = fitnessValue as number;
  }
}

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 * @param clearAction - Action that clears a genome's internal state.
 * @returns void.
 */
function clearGenomeStateIfRequested(
  controller: NeatControllerForEval,
  evaluationOptions: NeatControllerForEval['options'],
  clearAction: (genome: GenomeForEvaluation) => void,
): void {
  // Step 1: Clear genome state when the option is enabled.
  if (!evaluationOptions.clear) return;
  controller.population.forEach((genome) => clearAction(genome));
}
