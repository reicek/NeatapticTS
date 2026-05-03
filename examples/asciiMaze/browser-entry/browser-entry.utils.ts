import { MazeGenerator } from '../mazes';
import { BROWSER_ENTRY_CONSTANTS as C } from './browser-entry.constants';
import type {
  BrowserEntryEvolutionSettings,
  BrowserEntryHostElements,
} from './browser-entry.types';

/**
 * Resolve the browser host elements used by the demo logger and dashboard.
 *
 * @param container - Element id or host element provided by the caller.
 * @returns Resolved host, archive, live, and resize-observer targets.
 */
export const resolveBrowserEntryHostElements = (
  container: string | HTMLElement,
): BrowserEntryHostElements => {
  const hostElement =
    typeof container === 'string'
      ? document.getElementById(container)
      : container;
  const archiveElement = hostElement
    ? (hostElement.querySelector('#ascii-maze-archive') as HTMLElement | null)
    : null;
  const liveElement = hostElement
    ? (hostElement.querySelector('#ascii-maze-live') as HTMLElement | null)
    : null;
  const networkCanvasElement = hostElement
    ? (hostElement.querySelector('#ascii-maze-network-canvas') as HTMLCanvasElement | null)
    : null;

  return {
    hostElement,
    archiveElement,
    liveElement,
    networkCanvasElement,
    observeTarget:
      hostElement ?? document.getElementById(C.DEFAULT_CONTAINER_ID),
  };
};

/**
 * Build immutable evolution settings for a single maze dimension.
 *
 * @param dimension - Side length in cells for the procedural square maze.
 * @returns Per-phase evolution settings consumed by the curriculum runtime.
 */
export const createBrowserEvolutionSettings = (
  dimension: number,
): BrowserEntryEvolutionSettings => {
  return {
    agentMaxSteps: C.AGENT_MAX_STEPS,
    popSize: C.POPULATION_SIZE,
    maxStagnantGenerations: C.DEFAULT_MAX_STAGNANT_GENERATIONS,
    maxGenerations: C.DEFAULT_MAX_GENERATIONS,
    lamarckianIterations: C.LAMARCKIAN_ITERATIONS,
    lamarckianSampleSize: C.LAMARCKIAN_SAMPLE_SIZE,
    mazeFactory: () => new MazeGenerator(dimension, dimension).generate(),
  };
};

/**
 * Schedule follow-up curriculum work on the next animation tick when possible.
 *
 * @param callback - Follow-up phase callback.
 */
export const scheduleBrowserEntryFrame = (callback: () => void): void => {
  try {
    if (typeof requestAnimationFrame === 'function') {
      requestAnimationFrame(callback);
      return;
    }
  } catch {
    // Fall through to setTimeout when the browser scheduler is unavailable.
  }

  setTimeout(callback, 0);
};

/**
 * Determine whether a reported progress value counts as solved for curriculum advancement.
 *
 * @param progress - Runtime progress emitted by the evolution layer.
 * @returns Whether the maze phase should advance to the next dimension.
 */
export const didSolveBrowserMaze = (progress: unknown): boolean => {
  return typeof progress === 'number' && progress >= C.MIN_PROGRESS_TO_PASS;
};

/**
 * Advance the procedural maze dimension without exceeding the configured maximum.
 *
 * @param currentDimension - Current maze side length.
 * @returns Next side length to use.
 */
export const getNextBrowserMazeDimension = (
  currentDimension: number,
): number => {
  return Math.min(
    currentDimension + C.MAZE_DIMENSION_INCREMENT,
    C.MAX_MAZE_DIMENSION,
  );
};
