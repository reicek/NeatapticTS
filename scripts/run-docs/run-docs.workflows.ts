/*
 * High-level workflow functions and mode validation for the docs pipeline.
 *
 * Each workflow maps to one public `npm run docs*` command. Steps within a
 * workflow are ordered by dependency: parallel batches that share no outputs
 * run simultaneously; sequential steps wait for prior batches to finish.
 *
 * `ensureSupportedMode` is an assertion function so TypeScript narrows the
 * mode union after the call — callers need no extra type guard.
 */

import {
  ALL_MODE,
  FOLDERS_MODE,
  SUPPORTED_MODES,
} from './run-docs.constants.js';
import { runScriptTask, runScriptTasksInParallel } from './run-docs.runner.js';

/**
 * Runs the complete docs-generation pipeline:
 *
 * 1. Builds all browser example bundles and the semantic snapshot in parallel.
 * 2. Generates all folder-level READMEs in parallel.
 * 3. Copies example browser entrypoints and static docs into the published docs
 *    tree (sequential — must happen after step 2 so hand-written docs are not
 *    overwritten by generated folder output).
 * 4. Renders the final HTML site (sequential — depends on steps 2 and 3).
 *
 * @returns A promise that resolves when the full pipeline completes.
 */
export async function runFullDocsWorkflow(): Promise<void> {
  await runScriptTasksInParallel([
    { label: 'Hello Network bundle', scriptName: 'build:hello-network' },
    { label: 'Evolve XOR bundle', scriptName: 'build:evolve-xor' },
    { label: 'Sequence Reset bundle', scriptName: 'build:sequence-reset' },
    { label: 'ASCII Maze bundles', scriptName: 'build:ascii-maze' },
    { label: 'Flappy Bird bundles', scriptName: 'build:flappy-bird' },
    { label: 'NEATchat bundle', scriptName: 'build:neat-chat' },
    {
      label: 'Racing Curriculum bundle',
      scriptName: 'build:racing-curriculum',
    },
    {
      label: 'Neatenstein bundles',
      scriptName: 'build:neatenstein',
    },
    { label: 'Semantic snapshot', scriptName: 'index:build-snapshot' },
  ]);

  await runScriptTasksInParallel([
    { label: 'Source folder docs', scriptName: 'docs:folders:src:built' },
    {
      label: 'ASCII Maze folder docs',
      scriptName: 'docs:folders:asciiMaze:built',
    },
    {
      label: 'Flappy Bird folder docs',
      scriptName: 'docs:folders:flappy-bird:built',
    },
    {
      label: 'Racing Curriculum folder docs',
      scriptName: 'docs:folders:racing-curriculum:built',
    },
  ]);

  await runScriptTask({
    label: 'Examples copy',
    scriptName: 'docs:examples:built',
  });

  await runScriptTask({
    label: 'HTML docs render',
    scriptName: 'docs:html:built',
  });
}

/**
 * Runs only the folder-level README generation pipeline.
 *
 * Generates `src/`, ASCII Maze, and Flappy Bird folder docs in parallel.
 * Skips example bundling and HTML rendering, making this significantly faster
 * than the full workflow when only source documentation has changed.
 *
 * @returns A promise that resolves when all folder docs are generated.
 */
export async function runFoldersWorkflow(): Promise<void> {
  await runScriptTasksInParallel([
    { label: 'Source folder docs', scriptName: 'docs:folders:src:built' },
    {
      label: 'ASCII Maze folder docs',
      scriptName: 'docs:folders:asciiMaze:built',
    },
    {
      label: 'Flappy Bird folder docs',
      scriptName: 'docs:folders:flappy-bird:built',
    },
    {
      label: 'Racing Curriculum folder docs',
      scriptName: 'docs:folders:racing-curriculum:built',
    },
  ]);
}

/**
 * Asserts that `requestedMode` is a recognised docs pipeline mode.
 *
 * Acts as a TypeScript assertion function so that callers receive the narrowed
 * `'all' | 'folders'` union after the call without a separate type guard.
 *
 * @param requestedMode - The raw string from `process.argv[2]`.
 * @throws {Error} When `requestedMode` is not in `SUPPORTED_MODES`.
 */
export function ensureSupportedMode(
  requestedMode: string,
): asserts requestedMode is typeof ALL_MODE | typeof FOLDERS_MODE {
  if (!SUPPORTED_MODES.has(requestedMode)) {
    throw new Error(
      `Unsupported docs mode "${requestedMode}". Expected one of: ${[...SUPPORTED_MODES].join(', ')}`,
    );
  }
}
