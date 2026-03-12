/**
 * Node entrypoint for running the Flappy Bird trainer directly.
 *
 * Educational note:
 * This file stays intentionally small so the runnable script remains easy to
 * discover. The actual trainer orchestration lives under `trainer/`, while this
 * entrypoint handles the "run if invoked directly" Node workflow.
 */
import { pathToFileURL } from 'node:url';
import { handleTrainerMainError, runTrainer } from './trainer/trainer';

export { handleTrainerMainError, runTrainer } from './trainer/trainer';

if (isDirectTrainerExecution()) {
  runTrainer().catch(handleTrainerMainError);
}

/**
 * Resolves whether this module is being executed as the direct Node entrypoint.
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
