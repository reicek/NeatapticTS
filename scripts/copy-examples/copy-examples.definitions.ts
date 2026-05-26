/*
 * Registry of all examples that the copy-examples workflow publishes.
 *
 * Add or remove entries here to control which examples appear in docs/examples/.
 * The order of entries does not affect landing-page rendering — categories are
 * sorted alphabetically within each section by the HTML builders.
 */

import path from 'node:path';
import type { ExampleDefinition } from './copy-examples.types.js';

/**
 * Complete set of example publishing definitions processed by the
 * copy-examples workflow.
 *
 * Each entry describes one example that the pipeline probes, copies (or
 * generates a source-first page for), and registers on the landing page.
 * The `sourceDir` field is resolved from the repository root so that the
 * script is location-independent when invoked via `npm run`.
 */
export const EXAMPLE_DEFINITIONS: readonly ExampleDefinition[] = [
  {
    category: 'starter',
    description:
      'The smallest public-network walkthrough: build one compact feed-forward network, run one inference pass, and inspect the result shape immediately.',
    dirName: 'helloNetwork',
    label: 'helloNetwork',
    runCommand: 'npm run example:hello-network',
    title: 'Hello Network (NeatapticTS)',
    sourceDir: path.resolve('examples', 'helloNetwork'),
  },
  {
    category: 'starter',
    description:
      'A bounded feed-forward NEAT run on XOR that now reaches a solved state and shows the smallest end-to-end evolutionary loop in the repo.',
    dirName: 'evolveXor',
    label: 'evolveXor',
    runCommand: 'npm run example:evolve-xor',
    title: 'Evolve XOR (NeatapticTS)',
    sourceDir: path.resolve('examples', 'evolveXor'),
  },
  {
    category: 'starter',
    description:
      'A tiny LSTM example that feeds the same input sequence three times to show state accumulation, the effect of clear() after a run, and the carryover effect when clear() is skipped.',
    dirName: 'sequenceReset',
    label: 'sequenceReset',
    runCommand: 'npx tsx examples/sequenceReset/run.ts',
    title: 'Sequence Reset (NeatapticTS)',
    sourceDir: path.resolve('examples', 'sequenceReset'),
  },
  {
    category: 'flagship',
    description:
      'A compact navigation lab with browser playback, reward shaping, telemetry-rich search, and a deliberately small observation budget.',
    dirName: 'asciiMaze',
    label: 'asciiMaze',
    title: 'ASCII Maze (NeatapticTS)',
    sourceDir: path.resolve('examples', 'asciiMaze'),
  },
  {
    category: 'flagship',
    description:
      'A fast browser neuroevolution system with worker-backed playback, temporal observations, and a full inspectable runtime architecture.',
    dirName: 'flappy_bird',
    label: 'flappy_bird',
    title: 'Flappy Bird (NeatapticTS)',
    sourceDir: path.resolve('examples', 'flappy_bird'),
  },
  {
    category: 'flagship',
    description:
      'A published browser contract preview for the tiny sequence-learning chat demo, with a visible chat shell, staged progress, and the reused Flappy visualizer boundary.',
    dirName: 'neatChat',
    label: 'neatChat',
    runCommand: 'npx tsx examples/neatChat/run.ts',
    title: 'NEATchat (NeatapticTS)',
    sourceDir: path.resolve('examples', 'neatChat'),
  },
];
