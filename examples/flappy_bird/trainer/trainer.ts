/**
 * Node-facing entry shelf for the Flappy Bird trainer.
 *
 * This file is the chapter opening for the trainer as a whole. If you want the
 * fastest mental model for how the Flappy Bird training stack behaves, start
 * here before reading the narrower type, planning, fitness, or reporting
 * helpers.
 *
 * The trainer exists to turn a generic NEAT controller into a fair,
 * repeatable, Flappy-specific training program. That means the entry boundary
 * has to do more than just call `evolve()`: it restores deterministic random
 * state, installs staged population scoring, keeps shutdown cooperative, and
 * hands each generation to a compact reporting pipeline that makes progress
 * easy to inspect from the terminal.
 *
 * This file does not own the scoring math, report formatting, or rollout
 * mechanics directly. Its job is orchestration. Keeping that policy wiring in
 * one place makes the trainer easier to reason about because the reader can see
 * which responsibilities are static setup, which belong to the runtime loop,
 * and which are delegated to specialized helpers.
 *
 * A practical reading order is:
 *
 * 1. read this file to understand the runtime spine,
 * 2. continue with [trainer.types.ts](./trainer.types.ts) to learn the shared
 *    nouns,
 * 3. move to [trainer.evaluation-plan.utils.ts](./trainer.evaluation-plan.utils.ts)
 *    for the staged curriculum and mutation schedule,
 * 4. finish with [trainer.fitness.service.ts](./trainer.fitness.service.ts) and
 *    [trainer.loop.service.ts](./trainer.loop.service.ts) to see how each
 *    generation is actually evaluated and advanced.
 *
 * Read the rest of the trainer folder as supporting shelves beneath this
 * entrypoint: types define the vocabulary, evaluation planning defines the
 * budget and curriculum, fitness helpers define how populations are scored, and
 * the loop turns all of that into a long-running evolutionary session.
 *
 * Trainer startup map:
 * ```mermaid
 * flowchart LR
 *     Entry["runTrainer()"] --> Setup["createTrainerSetup()\nstatic training shape"]
 *     Entry --> Runtime["createTrainerRuntimeState()\nmutable stop + latest report"]
 *     Entry --> Controller["createNeatController()\nbase NEAT runtime"]
 *     Controller --> Fitness["attachPopulationFitnessEvaluator()\nstaged population scoring"]
 *     Entry --> RNG["restoreRNGState()\ndeterministic run"]
 *     Entry --> Signals["registerTrainerStopSignals()\ncooperative shutdown"]
 *     Fitness --> Loop["runTrainerEvolutionLoop()\ngeneration heartbeat"]
 *     Signals --> Loop
 *     Loop --> Summary["logGenerationSummary()\ncompact terminal output"]
 * ```
 */
import { pathToFileURL } from 'node:url';
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
 * The function is intentionally orchestration-first. It answers one practical
 * question: what has to be connected so a generic NEAT controller turns into a
 * fair, repeatable, Flappy-specific trainer?
 *
 * Educational note:
 * The trainer is intentionally orchestration-first. It wires together setup,
 * staged population evaluation, the outer evolution loop, graceful shutdown,
 * and compact generation logging without burying those responsibilities inside a
 * single monolithic file.
 *
 * The mutation schedule gradually cools over early generations. If you want a
 * conceptual parallel, the Wikipedia article on
 * [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing) is
 * a useful mental model for why early exploration is broader and later updates
 * are more conservative.
 *
 * Run (from repo root):
 * `npx ts-node examples/flappy_bird/trainFlappyBird.ts`
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

  console.log(FLAPPY_TRAINER_STOPPED_MESSAGE);
}

/**
 * Handles fatal `main` rejection path.
 *
 * The trainer keeps this boundary small so unexpected failures are formatted in
 * one consistent place before reaching the CLI. That keeps shutdown behavior
 * and terminal messaging consistent whether the failure came from setup,
 * evaluation, or the loop itself.
 *
 * @param error - Unknown rejection reason from trainer execution.
 * @returns Nothing.
 */
export function handleTrainerMainError(error: unknown): void {
  console.error(formatTrainerErrorMessage(error));
  process.exitCode = 1;
}

if (isDirectTrainerExecution()) {
  runTrainer().catch(handleTrainerMainError);
}

/**
 * Resolves whether this module is the direct Node entrypoint.
 *
 * This lets the file behave as both a reusable module and a runnable script
 * without duplicating the startup boundary in a second wrapper file.
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
