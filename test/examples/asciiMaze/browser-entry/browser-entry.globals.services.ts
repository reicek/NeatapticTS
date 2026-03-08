import { BROWSER_ENTRY_CONSTANTS as C } from './browser-entry.constants';
import type {
  BrowserEntryStartFunction,
  RuntimeWindow,
} from './browser-entry.types';

/**
 * Browser globals-compatibility service boundary for the ASCII Maze browser entry.
 *
 * This module isolates script-loader compatibility and guarded auto-start
 * behavior so runtime orchestration can stay focused on session lifecycle.
 */

/**
 * Install browser globals and one-time auto-start compatibility hooks.
 *
 * @param start - Public browser entry function to expose on the window namespace.
 * @returns Nothing.
 */
export const installBrowserEntryGlobals = (
  start: BrowserEntryStartFunction,
): void => {
  if (typeof window === 'undefined' || !(window as RuntimeWindow).document) {
    return;
  }

  const globalWindow = window as RuntimeWindow;
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
