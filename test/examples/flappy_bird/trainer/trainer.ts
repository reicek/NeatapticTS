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
 * Run (from repo root):
 * `npx ts-node test/examples/flappy_bird/trainFlappyBird.ts`
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
 */
export function handleTrainerMainError(error: unknown): void {
  // eslint-disable-next-line no-console
  console.error(formatTrainerErrorMessage(error));
  process.exitCode = 1;
}

if (isDirectTrainerExecution()) {
  runTrainer().catch(handleTrainerMainError);
}

function isDirectTrainerExecution(): boolean {
  const entryScriptPath = process.argv[1];
  if (!entryScriptPath) {
    return false;
  }
  return import.meta.url === pathToFileURL(entryScriptPath).href;
}
