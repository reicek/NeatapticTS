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

/**
 * Callback signature for one-line generation logging.
 *
 * The loop owns evolution cadence, while the callback owns presentation.
 * Keeping those concerns separate makes it easy to reuse the loop with richer
 * reporting later.
 */
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
 * Educational note:
 * This is the trainer's main heartbeat: resolve the current mutation schedule,
 * evolve one generation, run a representative fallback rollout for logging, and
 * emit a compact summary.
 *
 * @param neatController - Trainer NEAT controller.
 * @param trainerRuntimeState - Mutable trainer runtime state.
 * @param logGenerationSummary - Callback that emits compact generation logs.
 * @returns Promise resolved when the trainer has been stopped.
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
 * The schedule is resolved outside this helper so the loop can read as a clean
 * "resolve -> apply -> evolve -> report" flow.
 *
 * @param neatController - Trainer NEAT controller.
 * @param mutationSchedule - Mutation schedule for current generation.
 * @returns Nothing.
 */
function applyMutationSchedule(
  neatController: FlappyTrainerNeatController,
  mutationSchedule: FlappyMutationSchedule,
): void {
  neatController.options.mutationRate = mutationSchedule.mutationRate;
  neatController.options.mutationAmount = mutationSchedule.mutationAmount;
}
