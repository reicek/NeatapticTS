/**
 * ASCII Maze example public shelf.
 *
 * This file is the easiest programmatic on-ramp into the ASCII Maze example.
 * It re-exports the pieces a reader is most likely to want without forcing
 * them to memorize the folder split first: maze utilities, policy helpers,
 * visualization tools, and the high-level evolution facade.
 *
 * Educational note:
 * Treat this file as the example's outside-facing shelf, not as the place where
 * the main architecture is explained. For the engine story, read
 * `evolutionEngine.ts`. For the browser story, read
 * `browser-entry/browser-entry.ts`. For one-episode policy behavior, read
 * `mazeMovement.ts` and `fitness.ts`.
 *
 * @example
 * ```ts
 * import { EvolutionEngine, MazeUtils } from './index';
 *
 * const maze = ['#####', '#S..#', '#.#E#', '#####'];
 * const encoded = MazeUtils.encodeMaze(maze);
 * console.log(encoded.length);
 *
 * const result = await EvolutionEngine.runMazeEvolution({
 *   mazeConfig: { maze },
 *   agentSimConfig: { maxSteps: 50 },
 *   evolutionAlgorithmConfig: { popSize: 50, maxGenerations: 5 },
 *   reportingConfig: { dashboardManager: { update() {} } },
 * });
 * console.log(result.exitReason);
 * ```
 */

// Re-export core utility and logic classes for maze solving.
export { MazeUtils } from './mazeUtils';
export { MazeVision } from './mazeVision';
export { MazeMovement } from './mazeMovement';
export { MazeVisualization } from './mazeVisualization';
export { NetworkVisualization } from './networkVisualization';
export { DashboardManager } from './dashboardManager';
export { TerminalUtility } from './terminalUtility';
export { FitnessEvaluator } from './fitness';
export { EvolutionEngine } from './evolutionEngine';
export { NetworkRefinement } from './networkRefinement';

// Re-export interfaces and configuration objects for external use.
export * from './interfaces';
export { colors } from './colors';
export * from './mazes';
