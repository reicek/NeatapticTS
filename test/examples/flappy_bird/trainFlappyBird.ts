import { pathToFileURL } from 'node:url';
import { handleTrainerMainError, runTrainer } from './trainer/trainer';

export { handleTrainerMainError, runTrainer } from './trainer/trainer';

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
