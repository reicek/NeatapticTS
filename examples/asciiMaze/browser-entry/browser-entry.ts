/**
 * Browser-hosted curriculum boundary for the ASCII Maze example.
 *
 * This folder is where a long-running maze experiment becomes a browser
 * experience a human can actually steer and inspect. The evolution engine still
 * owns search, scoring, and curriculum advancement. The browser-entry boundary
 * owns host elements, resize behavior, telemetry fan-out, globals
 * compatibility, and the lifecycle handle that embedding code talks to.
 *
 * The browser host also owns the parts of the experience that should feel
 * understandable to a human observer rather than merely correct to the engine.
 * When a phase solves a maze, this boundary reveals the winning path step by
 * step in the live panel before it lets the curriculum advance. That small
 * presentation delay matters because the browser demo is trying to teach route
 * discovery, not just report that a solved result existed.
 *
 * Read it as a boundary between two clocks. One clock belongs to the maze
 * curriculum that carries refined winners into larger procedural mazes. The
 * other clock belongs to the browser host that has to paint dashboards, react
 * to cancellation, and stay polite to resize or unload events. `browser-entry/`
 * exists so those clocks can cooperate without collapsing into one monolithic
 * demo script.
 *
 * That separation matters because `index.html` is intentionally thin. The page
 * only loads the published bundle from `docs/assets`, exposes a globals bridge,
 * and then hands off to this start surface. If you want the real host/runtime
 * seam, start here instead of with the HTML shell.
 *
 * A second useful mental model is ownership. This folder does not own maze
 * fitness, winner refinement, or solve thresholds. Those stay in
 * `evolutionEngine/`. The host boundary owns container resolution, dashboard
 * plumbing, cooperative abort wiring, and the stable run handle that browser
 * callers can stop, await, or subscribe to.
 *
 * The first tuning stop for the hosted curriculum is
 * `browser-entry.constants.ts`. That constants table controls the starting maze
 * size, maximum maze size, dimension increment between solved phases, and the
 * per-maze step budget. Read it as the host-facing control shelf for how the
 * browser curriculum should feel, while the deeper engine folders continue to
 * own how evolution itself works.
 *
 * Read the chapter in three passes. Start with `browser-entry.ts` for the
 * public `start(...)` surface. Continue to `browser-entry.services.ts` for host
 * assembly, globals wiring, and curriculum hand-off. Finish with the constants,
 * curriculum, and host helper files when you want the browser-specific
 * mechanics rather than the public lifecycle contract.
 *
 * ```mermaid
 * flowchart LR
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   HtmlShell["index.html\nbundle loader"]:::base --> Globals["window globals\ncompatibility bridge"]:::base
 *   Globals --> Start["start(...)\npublic browser entry"]:::accent
 *   Start --> Host["host services\ndashboard + resize + container"]:::base
 *   Start --> Curriculum["runBrowserEntryCurriculum\nphase orchestration"]:::base
 *   Curriculum --> Engine["Evolution engine\nsearch and solve logic"]:::base
 *   Host --> Handle["AsciiMazeRunHandle\nstop done telemetry"]:::base
 *   Curriculum --> Handle
 * ```
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   BrowserEntry["browser-entry/"]:::accent --> PublicApi["browser-entry.ts\npublic start surface"]:::base
 *   BrowserEntry --> Services["browser-entry.services.ts\nhost and globals assembly"]:::base
 *   BrowserEntry --> Curriculum["browser-entry.curriculum.services.ts\nphase hand-off"]:::base
 *   BrowserEntry --> HostUtils["browser-entry.host.services.ts\nand utils"]:::base
 *   BrowserEntry --> Types["browser-entry.types.ts\nrun-handle contracts"]:::base
 * ```
 *
 * For background reading on the cooperative stop side of the boundary, see
 * MDN, [AbortController](https://developer.mozilla.org/en-US/docs/Web/API/AbortController),
 * which is the browser primitive this entry layer uses to compose internal and
 * caller-provided cancellation without moving DOM concerns into the engine.
 *
 * Example: boot the browser demo from embedding code and stop it later.
 *
 * ```ts
 * const handle = await start('ascii-maze-output');
 *
 * setTimeout(() => handle.stop(), 5_000);
 * await handle.done;
 * ```
 *
 * Example: subscribe to telemetry while the curriculum advances.
 *
 * ```ts
 * const handle = await start('ascii-maze-output');
 * const unsubscribe = handle.onTelemetry((telemetry) => {
 *   console.log(telemetry.generation, telemetry.bestFitness);
 * });
 *
 * await handle.done;
 * unsubscribe();
 * ```
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

// Prevents duplicate concurrent curriculum runs when both the HTML onload handler
// and the bundle's auto-start timer call start() within the same page load.
let _activeRunHandle: AsciiMazeRunHandle | null = null;

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
  if (_activeRunHandle !== null && _activeRunHandle.isRunning()) {
    return _activeRunHandle;
  }

  // Step 1: Resolve host elements and attach browser-specific services.
  const hostElements = resolveBrowserEntryHostElements(container);
  const hostServices = createBrowserEntryHostServices(hostElements);
  const hostAdapter = createBrowserEntryEvolutionHostAdapter({
    liveElement: hostElements.liveElement,
  });

  // Step 2: Create lifecycle state and compose cooperative cancellation.
  let cancelled = false;
  let running = true;
  let finalized = false;
  const internalController = new AbortController();
  let resolveDonePromise!: () => void;
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
  const handle: AsciiMazeRunHandle = {
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

  _activeRunHandle = handle;
  void handle.done.then(() => {
    if (_activeRunHandle === handle) {
      _activeRunHandle = null;
    }
  });

  return handle;
};

installBrowserEntryGlobals(start);

export type {
  AsciiMazeRunHandle,
  BrowserEntryStartFunction,
  BrowserEntryStartOptions,
} from './browser-entry.types';
