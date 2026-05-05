import { EvolutionEngine } from '../evolutionEngine';
import { resolveMazeEvolutionPhaseOutcome } from '../evolutionEngine';
import type { INetwork } from '../interfaces';
import {
  buildExampleArchitectureProfileNetwork,
  DEFAULT_ASCII_MAZE_ARCHITECTURE_PROFILE_ID,
} from '../../architectureProfiles';
import { BROWSER_ENTRY_CONSTANTS as C } from './browser-entry.constants';
import type { BrowserEntryCurriculumContext } from './browser-entry.types';
import {
  createBrowserEvolutionSettings,
  getNextBrowserMazeDimension,
  scheduleBrowserEntryFrame,
} from './browser-entry.utils';

/**
 * Browser curriculum-runtime service boundary for the ASCII Maze browser entry.
 *
 * This module now owns browser-only curriculum progression concerns: dimension
 * scheduling, frame pacing, and lifecycle completion. Evolution-phase result
 * interpretation and winner carry-over refinement live behind the engine-owned
 * curriculum helper so browser-entry stays focused on host runtime behavior.
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
    const warmStartNetwork = resolvePhaseWarmStartNetwork(
      context,
      previousBestNetwork,
    );
    let solved = false;

    try {
      const result = await EvolutionEngine.runMazeEvolution({
        mazeConfig: { maze: mazeLayout },
        agentSimConfig: { maxSteps: settings.agentMaxSteps },
        evolutionAlgorithmConfig: {
          allowRecurrent: settings.allowRecurrent,
          adaptiveMutation: settings.adaptiveMutation,
          popSize: settings.popSize,
          maxStagnantGenerations: settings.maxStagnantGenerations,
          minProgressToPass: C.MIN_PROGRESS_TO_PASS,
          maxGenerations: settings.maxGenerations,
          autoPauseOnSolve: false,
          stopOnlyOnSolve: false,
          lamarckianIterations: settings.lamarckianIterations,
          lamarckianSampleSize: settings.lamarckianSampleSize,
          initialBestNetwork: warmStartNetwork,
          architectureProfileId: context.architectureProfileId,
        },
        reportingConfig: {
          dashboardManager: context.dashboard,
          hostAdapter: context.hostAdapter,
          logEvery: C.PER_GENERATION_LOG_FREQUENCY,
          label: `browser-procedural-${context.architectureProfileId ?? 'mlp'}-${currentDimension}x${currentDimension}`,
          paceEveryGeneration: true,
        },
        cancellation: { isCancelled: () => context.isCancelled() },
        signal: context.combinedSignal,
      });
      const phaseOutcome = resolveMazeEvolutionPhaseOutcome(
        result,
        previousBestNetwork,
        C.MIN_PROGRESS_TO_PASS,
      );
      const progress = phaseOutcome.progress;

      previousBestNetwork = phaseOutcome.nextBestNetwork;
      solved = phaseOutcome.solved;

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
 * Resolve a phase warm-start network.
 *
 * Step 1: Reuse the previous phase winner when one is available.
 * Step 2: Otherwise seed the first phase with a deterministic profile-built
 * network so generation zero is less noisy than a pure cold start.
 */
function resolvePhaseWarmStartNetwork(
  context: BrowserEntryCurriculumContext,
  previousBestNetwork: INetwork | undefined,
): INetwork {
  if (previousBestNetwork) {
    return previousBestNetwork;
  }

  const profileId =
    context.architectureProfileId ?? DEFAULT_ASCII_MAZE_ARCHITECTURE_PROFILE_ID;
  return buildExampleArchitectureProfileNetwork('ascii-maze', profileId);
}
