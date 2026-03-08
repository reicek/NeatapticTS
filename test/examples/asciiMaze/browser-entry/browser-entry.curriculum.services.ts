import type Network from '../../../../src/architecture/network';
import { EvolutionEngine } from '../evolutionEngine';
import type { INetwork } from '../interfaces';
import { NetworkRefinement } from '../networkRefinement';
import { BROWSER_ENTRY_CONSTANTS as C } from './browser-entry.constants';
import type {
  BrowserEntryCurriculumContext,
  RuntimeEvolutionResult,
} from './browser-entry.types';
import {
  createBrowserEvolutionSettings,
  didSolveBrowserMaze,
  getNextBrowserMazeDimension,
  scheduleBrowserEntryFrame,
} from './browser-entry.utils';

/**
 * Browser curriculum-runtime service boundary for the ASCII Maze browser entry.
 *
 * This module isolates maze progression, solve detection, and cross-phase
 * winner carry-over so the compatibility facade stays orchestration-first.
 */

/**
 * Run the progressive browser curriculum across increasingly larger mazes.
 *
 * @param context - Runtime dashboard, cancellation, and completion callbacks for one browser session.
 * @returns Nothing.
 */
export const runBrowserEntryCurriculum = (
  context: BrowserEntryCurriculumContext,
): void => {
  let currentDimension: number = C.INITIAL_MAZE_DIMENSION;
  let previousBestNetwork: INetwork | undefined;

  void runCurrentPhase();

  async function runCurrentPhase(): Promise<void> {
    if (context.isCancelled()) {
      context.finish();
      return;
    }

    const settings = createBrowserEvolutionSettings(currentDimension);
    const mazeLayout = settings.mazeFactory();
    let solved = false;

    try {
      const result = await EvolutionEngine.runMazeEvolution({
        mazeConfig: { maze: mazeLayout },
        agentSimConfig: { maxSteps: settings.agentMaxSteps },
        evolutionAlgorithmConfig: {
          allowRecurrent: true,
          popSize: settings.popSize,
          maxStagnantGenerations: settings.maxStagnantGenerations,
          minProgressToPass: C.MIN_PROGRESS_TO_PASS,
          maxGenerations: settings.maxGenerations,
          autoPauseOnSolve: false,
          stopOnlyOnSolve: false,
          lamarckianIterations: settings.lamarckianIterations,
          lamarckianSampleSize: settings.lamarckianSampleSize,
          initialBestNetwork: previousBestNetwork,
        },
        reportingConfig: {
          dashboardManager: context.dashboard,
          logEvery: C.PER_GENERATION_LOG_FREQUENCY,
          label: `browser-procedural-${currentDimension}x${currentDimension}`,
          paceEveryGeneration: true,
        },
        cancellation: { isCancelled: () => context.isCancelled() },
        signal: context.combinedSignal,
      });
      const runtimeResult = result as unknown as RuntimeEvolutionResult;
      const progress = runtimeResult.bestResult?.progress;

      previousBestNetwork = refineBrowserEntryBestNetwork(
        runtimeResult.bestNetwork,
        previousBestNetwork,
      );
      solved = didSolveBrowserMaze(progress);

      try {
        console.log(
          '[asciiMaze] maze complete',
          currentDimension,
          'solved?',
          solved,
          'progress',
          progress,
        );
      } catch {
        // Ignore console failures in minimal or proxied environments.
      }
    } catch (error: unknown) {
      console.error(
        'Error while running procedural maze',
        currentDimension,
        error,
      );
    }

    if (
      !context.isCancelled() &&
      solved &&
      currentDimension < C.MAX_MAZE_DIMENSION
    ) {
      currentDimension = getNextBrowserMazeDimension(currentDimension);
      scheduleBrowserEntryFrame(() => {
        void runCurrentPhase();
      });
      return;
    }

    context.finish();
  }
};

/**
 * Refine the winning network before seeding the next curriculum phase.
 *
 * @param bestNetwork - Network returned by the latest evolution phase.
 * @param previousBestNetwork - Previously carried curriculum seed.
 * @returns Refined winner or the best available carry-over network.
 */
function refineBrowserEntryBestNetwork(
  bestNetwork: INetwork | undefined,
  previousBestNetwork: INetwork | undefined,
): INetwork | undefined {
  if (!bestNetwork) {
    return previousBestNetwork;
  }

  try {
    const refinedNetwork = NetworkRefinement.refineWinnerWithBackprop(
      bestNetwork as unknown as Network,
    );
    return (refinedNetwork as unknown as INetwork) || bestNetwork;
  } catch {
    return bestNetwork;
  }
}
