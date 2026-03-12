import { pathToFileURL } from 'node:url';
import { rolloutEpisode } from '../flappyEvaluation';
import type {
  FlappyTrainerNetwork,
  FlappyTrainerRuntimeState,
  FlappyTrainerSetup,
} from './trainer.types';
import {
  FLAPPY_TRAINER_DEFAULT_RNG_SEED,
  FLAPPY_TRAINER_STOPPED_MESSAGE,
} from './trainer.constants';
import {
  commitPopulationScores,
  evaluatePopulationFullStage,
  evaluatePopulationQuickStage,
  evaluatePopulationReevaluationStage,
} from './trainer.evaluation.service';
import { formatTrainerErrorMessage } from './trainer.errors';
import { resolveGenerationEvaluationPlan } from './trainer.evaluation-plan.utils';
import {
  createNeatController,
  createTrainerRuntimeState,
  createTrainerSetup,
} from './trainer.setup.service';
import { attachPopulationFitnessEvaluator } from './trainer.fitness.service';
import { runTrainerEvolutionLoop } from './trainer.loop.service';
import {
  buildGenerationReport,
  logGenerationSummary,
} from './trainer.report.service';
import { registerTrainerStopSignals } from './trainer.signals.service';

/**
 * Flappy Bird neuroevolution demo.
 *
 * This script runs a small NEAT population where each genome controls a bird.
 * The network sees a temporal observation (38 floats) and outputs two competing
 * action scores (`no flap` vs `flap`).
 *
 * Educational note:
 * The trainer is intentionally orchestration-first. It wires together setup,
 * staged population evaluation, the outer evolution loop, graceful shutdown,
 * and compact generation logging without burying those responsibilities inside a
 * single monolithic file.
 *
 * The mutation schedule gradually cools over early generations. If you want a
 * conceptual parallel, the Wikipedia article on "simulated annealing" is a
 * useful mental model for why early exploration is broader and later updates are
 * more conservative.
 *
 * Run (from repo root):
 * `npx ts-node test/examples/flappy_bird/trainFlappyBird.ts`
 *
 * @example
 * ```ts
 * await runTrainer();
 * ```
 */
export async function runTrainer(): Promise<void> {
  const trainerSetup = createTrainerSetup();
  const trainerRuntimeState = createTrainerRuntimeState();
  const neatController = createNeatController(trainerSetup);

  attachPopulationFitnessEvaluator(
    neatController,
    trainerRuntimeState,
    trainerSetup.elitismCount,
    {
      resolveGenerationEvaluationPlan,
      evaluatePopulationQuickStage,
      evaluatePopulationFullStage,
      evaluatePopulationReevaluationStage,
      commitPopulationScores,
      buildGenerationReport,
    },
  );
  neatController.restoreRNGState(FLAPPY_TRAINER_DEFAULT_RNG_SEED);
  registerTrainerStopSignals(trainerRuntimeState);

  await runTrainerEvolutionLoop(
    neatController,
    trainerRuntimeState,
    logGenerationSummary,
  );

  // eslint-disable-next-line no-console
  console.log(FLAPPY_TRAINER_STOPPED_MESSAGE);
}

/**
 * Handles fatal `main` rejection path.
 *
 * The trainer keeps this boundary small so unexpected failures are formatted in
 * one consistent place before reaching the CLI.
 *
 * @param error - Unknown rejection reason from trainer execution.
 * @returns Nothing.
 */
export function handleTrainerMainError(error: unknown): void {
  // eslint-disable-next-line no-console
  console.error(formatTrainerErrorMessage(error));
  process.exitCode = 1;
}

if (isDirectTrainerExecution()) {
  runTrainer().catch(handleTrainerMainError);
}

/**
 * Resolves whether this module is the direct Node entrypoint.
 *
 * @returns `true` when Node launched this file directly.
 */
function isDirectTrainerExecution(): boolean {
  const entryScriptPath = process.argv[1];
  if (!entryScriptPath) {
    return false;
  }
  return import.meta.url === pathToFileURL(entryScriptPath).href;
}
