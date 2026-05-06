import { BROWSER_ENTRY_CONSTANTS as C } from './browser-entry.constants';
import { animateSolvedMazeInBrowserHost } from './browser-entry.solved-maze-animation.services';
import type {
  BrowserEntryStartFunction,
  RuntimeWindow,
} from './browser-entry.types';
import type { EvolutionHostAdapter } from '../evolutionEngine/evolutionEngine.types';

/**
 * Browser globals-compatibility service boundary for the ASCII Maze browser entry.
 *
 * This module isolates script-loader compatibility and guarded auto-start
 * behavior so runtime orchestration can stay focused on session lifecycle.
 */

/**
 * Create the browser-owned engine host adapter used for pause polling and solve notifications.
 *
 * The adapter keeps three host-specific concerns outside the engine:
 *
 * 1. polling browser pause state,
 * 2. running the solved-path reveal in the live maze panel,
 * 3. dispatching a browser event only after the host-side reveal finishes.
 *
 * That ordering is intentional. The engine decides that a maze is solved, but
 * the browser host decides how a human should see that solve.
 *
 * @returns Host adapter that keeps browser globals and DOM events out of engine internals.
 */
export const createBrowserEntryEvolutionHostAdapter = (
  input: {
    liveElement?: HTMLElement | null;
    runtimeWindow?: RuntimeWindow | null;
  } = {},
): EvolutionHostAdapter => ({
  isPauseRequested: (): boolean => {
    const runtimeWindow = resolveRuntimeWindow(input.runtimeWindow);
    if (!runtimeWindow) {
      return false;
    }

    return runtimeWindow.asciiMazePaused === true;
  },
  handleStop: async ({
    reason,
    maze,
    completedGenerations,
    result,
    progress,
    requestHostPause,
  }): Promise<void> => {
    const runtimeWindow = resolveRuntimeWindow(input.runtimeWindow);
    if (!runtimeWindow) {
      return;
    }

    if (reason === 'solved' && requestHostPause) {
      runtimeWindow.asciiMazePaused = true;
    }

    if (reason !== 'solved') {
      return;
    }

    await animateSolvedMazeInBrowserHost({
      liveElement: input.liveElement ?? null,
      maze,
      path: result?.path,
    });

    try {
      runtimeWindow.dispatchEvent(
        new CustomEvent('asciiMazeSolved', {
          detail: {
            maze,
            generations: completedGenerations,
            progress,
          },
        }),
      );
    } catch {
      // Ignore CustomEvent dispatch failures in restricted browser runtimes.
    }
  },
});

/**
 * Install browser globals and one-time auto-start compatibility hooks.
 *
 * @param start - Public browser entry function to expose on the window namespace.
 * @param runtimeWindow - Optional runtime window override used by tests or embedding hosts.
 * @returns Nothing.
 */
export const installBrowserEntryGlobals = (
  start: BrowserEntryStartFunction,
  runtimeWindow?: RuntimeWindow | null,
): void => {
  const globalWindow = resolveRuntimeWindow(runtimeWindow);
  if (!globalWindow?.document) {
    return;
  }
  globalWindow.asciiMaze = globalWindow.asciiMaze || {};
  globalWindow.asciiMaze.start = start;

  if (!globalWindow.asciiMazeStart) {
    globalWindow.asciiMazeStart = (containerElement?: unknown) => {
      console.warn(
        '[asciiMaze] window.asciiMazeStart is deprecated; use import { start } ... or window.asciiMaze.start',
      );
      return start(containerElement as string | HTMLElement | undefined);
    };
  }

  if (globalWindow.asciiMaze._autoStarted) {
    return;
  }

  globalWindow.asciiMaze._autoStarted = true;
  setTimeout(() => {
    try {
      if (document.getElementById(C.DEFAULT_CONTAINER_ID)) {
        void start();
      }
    } catch {
      // Ignore auto-start failures so browser embedding can recover manually.
    }
  }, C.AUTO_START_DELAY_MS);
};

function resolveRuntimeWindow(
  runtimeWindow?: RuntimeWindow | null,
): RuntimeWindow | undefined {
  if (runtimeWindow !== undefined) {
    return runtimeWindow ?? undefined;
  }

  return globalThis.window as unknown as RuntimeWindow | undefined;
}
