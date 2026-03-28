/**
 * Public browser entry facade for the ASCII Maze demo module boundary.
 *
 * The folder now owns host bootstrap, runtime orchestration, globals
 * compatibility, and resize handling behind focused helpers. This facade keeps
 * the public API stable while presenting a small orchestration-first surface.
 *
 * Educational note:
 * `index.html` is only the browser shell that loads the prebuilt bundle and
 * forwards into this API through window globals. If you want the real browser
 * host boundary, start here rather than with the HTML loader.
 */

import { BROWSER_ENTRY_CONSTANTS as C } from './browser-entry.constants';
import {
  composeBrowserEntryAbortSignal,
  createBrowserEntryEvolutionHostAdapter,
  createBrowserEntryHostServices,
  installBrowserEntryGlobals,
  runBrowserEntryCurriculum,
} from './browser-entry.services';
import type {
  AsciiMazeRunHandle,
  BrowserEntryStartOptions,
} from './browser-entry.types';
import { resolveBrowserEntryHostElements } from './browser-entry.utils';

/**
 * Start the browser-hosted ASCII Maze curriculum demo.
 *
 * @remarks
 * The run starts with a small procedural maze and carries the best refined
 * network forward into larger mazes until the configured maximum size is
 * reached or cancellation stops the curriculum.
 *
 * @example
 * ```ts
 * const handle = await start('ascii-maze-output');
 * handle.onTelemetry((telemetry) => console.log(telemetry));
 * await handle.done;
 * ```
 *
 * @example
 * ```ts
 * const abortController = new AbortController();
 * const handle = await start('ascii-maze-output', {
 *   signal: abortController.signal,
 * });
 *
 * setTimeout(() => abortController.abort(), 1_000);
 * await handle.done;
 * ```
 *
 * @param container - Element id or host element for the browser demo.
 * @param opts - Optional cooperative cancellation settings.
 * @returns Lifecycle handle for stop, status, completion, and telemetry access.
 */
export const start = async (
  container: string | HTMLElement = C.DEFAULT_CONTAINER_ID,
  opts: BrowserEntryStartOptions = {},
): Promise<AsciiMazeRunHandle> => {
  // Step 1: Resolve host elements and attach browser-specific services.
  const hostElements = resolveBrowserEntryHostElements(container);
  const hostServices = createBrowserEntryHostServices(hostElements);
  const hostAdapter = createBrowserEntryEvolutionHostAdapter();

  // Step 2: Create lifecycle state and compose cooperative cancellation.
  let cancelled = false;
  let running = true;
  let finalized = false;
  const internalController = new AbortController();
  let resolveDonePromise = () => {};
  const done = new Promise<void>((resolve) => {
    resolveDonePromise = resolve;
  });
  const finalizeRun = () => {
    if (finalized) {
      return;
    }

    finalized = true;
    running = false;
    hostServices.disposeResizeHandling();
    resolveDonePromise();
  };
  const combinedSignal = composeBrowserEntryAbortSignal(
    internalController,
    opts.signal,
  );

  if (combinedSignal.aborted) {
    cancelled = true;
    finalizeRun();
  } else {
    try {
      combinedSignal.addEventListener(
        'abort',
        () => {
          cancelled = true;
          finalizeRun();
        },
        { once: true },
      );
    } catch {
      // Ignore listener wiring failures in older or test environments.
    }
  }

  // Step 3: Start curriculum orchestration only when the run is still active.
  if (!cancelled) {
    runBrowserEntryCurriculum({
      dashboard: hostServices.dashboard,
      combinedSignal,
      isCancelled: () => cancelled,
      finish: finalizeRun,
      hostAdapter,
    });
  }

  // Step 4: Return the stable lifecycle handle for embedding hosts.
  return {
    stop: () => {
      cancelled = true;
      finalizeRun();
      try {
        internalController.abort();
      } catch {
        // Ignore duplicate aborts or unsupported environments.
      }
    },
    isRunning: () => running && !cancelled && !combinedSignal.aborted,
    done,
    onTelemetry: (telemetryCallback) =>
      hostServices.telemetryHub.add(telemetryCallback),
    getTelemetry: () => hostServices.runtimeDashboard.getLastTelemetry?.(),
  };
};

installBrowserEntryGlobals(start);

export type {
  AsciiMazeRunHandle,
  BrowserEntryStartFunction,
  BrowserEntryStartOptions,
} from './browser-entry.types';
