import { rolloutEpisode } from '../flappyEvaluation';
import { FLAPPY_MAX_FRAMES_PER_EPISODE } from '../constants/constants';
import {
  resolveMutationSchedule,
  type FlappyMutationSchedule,
} from './trainer.evaluation-plan.utils';
import type {
  FlappyGenerationReport,
  FlappyTrainerNeatController,
  FlappyTrainerNetwork,
  FlappyTrainerRuntimeState,
} from './trainer.types';

/** Callback signature for one-line generation logging. */
export type LogGenerationSummaryCallback = (
  generationLabel: number,
  mutationSchedule: FlappyMutationSchedule,
  report: FlappyGenerationReport | undefined,
  fittestGenome: FlappyTrainerNetwork,
  fallbackEpisode: ReturnType<typeof rolloutEpisode>,
) => void;

/**
 * Runs the outer evolution loop until runtime stop is requested.
 *
 * @param neatController - Trainer NEAT controller.
 * @param trainerRuntimeState - Mutable trainer runtime state.
 * @param logGenerationSummary - Callback that emits compact generation logs.
 */
export async function runTrainerEvolutionLoop(
  neatController: FlappyTrainerNeatController,
  trainerRuntimeState: FlappyTrainerRuntimeState,
  logGenerationSummary: LogGenerationSummaryCallback,
): Promise<void> {
  while (!trainerRuntimeState.shouldStop) {
    const mutationSchedule = resolveMutationSchedule(neatController.generation);
    applyMutationSchedule(neatController, mutationSchedule);

    const fittestGenome = await neatController.evolve();
    const fallbackEpisode = rolloutEpisode(fittestGenome, {
      normalizeFitness: true,
      maxFrames: FLAPPY_MAX_FRAMES_PER_EPISODE,
    });

    logGenerationSummary(
      neatController.generation,
      mutationSchedule,
      trainerRuntimeState.latestGenerationReport,
      fittestGenome,
      fallbackEpisode,
    );
  }
}

/**
 * Applies mutation schedule values to the NEAT controller options.
 *
 * @param neatController - Trainer NEAT controller.
 * @param mutationSchedule - Mutation schedule for current generation.
 */
function applyMutationSchedule(
  neatController: FlappyTrainerNeatController,
  mutationSchedule: FlappyMutationSchedule,
): void {
  neatController.options.mutationRate = mutationSchedule.mutationRate;
  neatController.options.mutationAmount = mutationSchedule.mutationAmount;
}
