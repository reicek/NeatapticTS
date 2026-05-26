/*
 * Orchestrates the docs pipeline with one docs-scripts build and parallelized
 * independent steps.
 *
 * This keeps the user-facing `npm run docs` and `npm run docs:folders`
 * commands simple while removing repeated `docs:build-scripts` work and
 * parallelizing the safe parts of the pipeline.
 *
 * Heavy logic lives in `scripts/run-docs/`. This file is intentionally a thin
 * router — it resolves the mode from `process.argv[2]`, validates it, and
 * delegates to the matching workflow.
 */

import { ALL_MODE } from './run-docs/run-docs.constants.js';
import {
  ensureSupportedMode,
  runFullDocsWorkflow,
  runFoldersWorkflow,
} from './run-docs/run-docs.workflows.js';

async function main(): Promise<void> {
  const requestedMode = process.argv[2] ?? ALL_MODE;
  ensureSupportedMode(requestedMode);

  if (requestedMode === 'folders') {
    await runFoldersWorkflow();
    return;
  }

  await runFullDocsWorkflow();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
