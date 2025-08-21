import { BrowserTerminalUtility } from './browserTerminalUtility';
import { createBrowserLogger } from './browserLogger';
import { DashboardManager } from './dashboardManager';
import { EvolutionEngine } from './evolutionEngine';
import * as mazes from './mazes';

// Small helper to create UI controls (start/stop) — minimal, unobtrusive
// No UI controls in embedded mode; the demo will run automatically.

/**
 * Handle returned by {@link start} providing lifecycle & telemetry access.
 *
 * Consumers embedding the ASCII Maze demo can use this object to:
 * - Stop the evolutionary curriculum early (`stop()`)
 * - Check whether evolution is still active (`isRunning()`)
 * - Await natural completion (`done` Promise resolves when curriculum ends or stop() called)
 * - Subscribe to lightweight per-generation telemetry (`onTelemetry(cb)` returning an unsubscribe)
 * - Pull the latest snapshot on demand (`getTelemetry()`)
 */
export interface AsciiMazeRunHandle {
  stop: () => void;
  isRunning: () => boolean;
  /** internal promise that resolves when current curriculum finishes or is stopped */
  done: Promise<void>;
  /** Subscribe to telemetry events (per generation); returns unsubscribe */
  onTelemetry: (cb: (t: any) => void) => () => void;
  /** Get the latest telemetry snapshot */
  getTelemetry: () => any;
}

/**
 * Start the ASCII Maze evolutionary demo.
 *
 * This boots a multi‑maze curriculum (in a fixed order) that evolves a NEAT population
 * repeatedly until each maze is solved (or `stop()` is invoked). The function is
 * idempotent per page load for convenience when used via the UMD wrapper; calling it
 * multiple times creates independent runs (each with its own handle).
 *
 * Modern ESM usage:
 * ```ts
 * import { start } from '.../asciiMaze/browser-entry.js';
 * const run = await start('#ascii-maze-output');
 * run.onTelemetry(t => console.log(t.generation, t.bestFitness));
 * // later -> run.stop();
 * ```
 *
 * UMD (legacy) usage still supported:
 * ```html
 * <script src="/assets/ascii-maze.bundle.js"></script>
 * <script>window.asciiMaze.start();</script>
 * ```
 *
 * @param container Element id or HTMLElement acting as host (defaults to 'ascii-maze-output').
 * @param opts Optional configuration.
 * @param opts.signal AbortSignal to cooperatively cancel the entire curriculum (ES2023 idiom). `stop()` will also trigger an abort.
 * @returns A {@link AsciiMazeRunHandle} exposing lifecycle & telemetry controls.
 */
/**
 * Start the ASCII Maze evolutionary demo.
 *
 * Public API: boots a curriculum of mazes and returns a handle that can be
 * used to stop the run, subscribe to telemetry, or query the last snapshot.
 *
 * @param container - Element id or HTMLElement to host the demo (defaults to 'ascii-maze-output')
 * @param opts - Optional configuration object. `opts.signal` (AbortSignal) can be supplied
 *               by the caller to cooperatively cancel the curriculum.
 * @returns A {@link AsciiMazeRunHandle} exposing lifecycle controls and telemetry hooks.
 */
export async function start(
  container: string | HTMLElement = 'ascii-maze-output',
  opts: { signal?: AbortSignal } = {}
): Promise<AsciiMazeRunHandle> {
  const host =
    typeof container === 'string'
      ? document.getElementById(container)
      : container;
  const archiveEl = host
    ? (host.querySelector('#ascii-maze-archive') as HTMLElement)
    : null;
  const liveEl = host
    ? (host.querySelector('#ascii-maze-live') as HTMLElement)
    : null;

  // clearer will clear only the live area; archive remains
  const clearFn = BrowserTerminalUtility.createTerminalClearer(
    liveEl ?? undefined
  );
  const liveLogFn = createBrowserLogger(liveEl ?? undefined);
  const archiveLogFn = createBrowserLogger(archiveEl ?? undefined);

  // DashboardManager will use live logger for ongoing redraws and archive logger to append solved blocks
  const dashboard = new DashboardManager(
    clearFn,
    liveLogFn as any,
    archiveLogFn as any
  );
  // Local telemetry listener list
  const telemetryListeners: Array<(t: any) => void> = [];
  // Inject hook so DashboardManager can call into listeners
  // Dashboard calls this hook for each generation. We clone the listener
  // list using spread to avoid issues if a listener unsubscribes itself
  // during iteration (safe and low-cost because the listener list is small).
  (dashboard as any)._telemetryHook = (t: any) => {
    [...telemetryListeners].forEach((fn) => {
      try {
        fn(t);
      } catch {}
    });
  };

  // Responsive resize: re-render dashboard when host width changes significantly.
  try {
    const hostEl = host || document.getElementById('ascii-maze-output');
    if (hostEl && typeof ResizeObserver !== 'undefined') {
      let lastWidth = hostEl.clientWidth;
      const ro = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const w = entry.contentRect.width;
          if (Math.abs(w - lastWidth) > 8) {
            // threshold to avoid noisy redraws
            lastWidth = w;
            try {
              // Force a redraw using last known best state; dashboard.update triggers redraw normally
              (dashboard as any).redraw?.([], undefined);
            } catch {
              /* ignore */
            }
          }
        }
      });
      ro.observe(hostEl);
    } else if (hostEl) {
      // Fallback: window resize listener
      let debounce: any = null;
      const handler = () => {
        if (debounce) clearTimeout(debounce);
        debounce = setTimeout(() => {
          try {
            (dashboard as any).redraw?.([], undefined);
          } catch {}
        }, 120);
      };
      window.addEventListener('resize', handler);
    }
  } catch {
    /* ignore resize wiring errors */
  }

  // Inner runner (previously assigned to window.asciiMazeStart). Kept internal for ESM API.
  let cancelled = false;
  const internalController = new AbortController();
  const externalSignal = opts.signal;
  const combinedSignal: AbortSignal | undefined = (() => {
    if (!externalSignal) return internalController.signal;
    if ((externalSignal as any).aborted) return externalSignal; // already aborted
    // Compose by listening to external and aborting internal when it fires
    try {
      externalSignal.addEventListener(
        'abort',
        () => {
          try {
            internalController.abort();
          } catch {}
        },
        { once: true }
      );
    } catch {}
    return internalController.signal;
  })();
  let running = true;
  const runCurriculum = async () => {
    // Run mazes in the same curriculum order as the e2e test and mirror its
    // evolution settings where practical. We intentionally disable the
    // post-evolution backprop refinement (lamarckian iterations = 0) for
    // the browser demo as requested.
    const order = [
      'tiny',
      'spiralSmall',
      'spiral',
      'small',
      'medium',
      'medium2',
      'large',
      'minotaur',
    ];

    // Carry the winning network forward between phases (curriculum transfer)
    let lastBestNetwork: any = undefined;

    for (const key of order) {
      if (cancelled) break;
      const maze = (mazes as any)[key] as string[];
      if (!Array.isArray(maze)) continue; // skip missing exports

      // Per-phase settings copied from the e2e test with lamarckianIterations=0
      let agentMaxSteps = 1000;
      let maxGenerations = 500;
      switch (key) {
        case 'tiny':
          agentMaxSteps = 100;
          maxGenerations = 200;
          break;
        case 'spiralSmall':
          agentMaxSteps = 100;
          maxGenerations = 200;
          break;
        case 'spiral':
          agentMaxSteps = 150;
          maxGenerations = 300;
          break;
        case 'small':
          agentMaxSteps = 50;
          maxGenerations = 300;
          break;
        case 'medium':
          agentMaxSteps = 250;
          maxGenerations = 400;
          break;
        case 'medium2':
          agentMaxSteps = 300;
          maxGenerations = 400;
          break;
        case 'large':
          agentMaxSteps = 400;
          maxGenerations = 500;
          break;
        case 'minotaur':
          agentMaxSteps = 700;
          maxGenerations = 600;
          break;
      }

      try {
        const result = await EvolutionEngine.runMazeEvolution({
          mazeConfig: { maze },
          agentSimConfig: { maxSteps: agentMaxSteps },
          evolutionAlgorithmConfig: {
            allowRecurrent: true,
            popSize: 40,
            // Run indefinitely until solved; remove stagnation pressure for demo clarity
            maxStagnantGenerations: Number.POSITIVE_INFINITY,
            minProgressToPass: 99,
            maxGenerations: Number.POSITIVE_INFINITY,
            stopOnlyOnSolve: false,
            autoPauseOnSolve: false,
            // Disable Lamarckian/backprop refinement for browser runs per request
            lamarckianIterations: 0,
            lamarckianSampleSize: 0,
            // seed previous winner if available
            initialBestNetwork: lastBestNetwork,
          },
          reportingConfig: {
            dashboardManager: dashboard,
            logEvery: 1,
            label: `browser-${key}`,
          },
          cancellation: { isCancelled: () => cancelled },
          signal: combinedSignal,
        });
        try {
          console.log(
            '[asciiMaze] maze solved',
            key,
            (result as any)?.bestResult?.progress
          );
        } catch {}
        if (result && (result as any).bestNetwork)
          lastBestNetwork = (result as any).bestNetwork;
      } catch (e) {
        console.error('Error while running maze', key, e);
      }
    }
    running = false;
  };

  // Kick off asynchronous curriculum immediately for convenience
  const donePromise = runCurriculum();

  const handle: AsciiMazeRunHandle = {
    stop: () => {
      cancelled = true;
      try {
        internalController.abort();
      } catch {}
    },
    isRunning: () => running && !cancelled,
    done: Promise.resolve(donePromise).catch(() => {}) as Promise<void>,
    onTelemetry: (cb) => {
      telemetryListeners.push(cb);
      return () => {
        const i = telemetryListeners.indexOf(cb);
        if (i >= 0) telemetryListeners.splice(i, 1);
      };
    },
    getTelemetry: () => (dashboard as any).getLastTelemetry?.(),
  };

  // (Pause UI removed; external host can manage pause via a future API if needed.)
  return handle;
}

// UMD-style compatibility + deprecated global.
// If loaded directly (no module loader), expose window.asciiMaze.start() and legacy asciiMazeStart().
declare const __webpack_require__: any; // silence TS if bundler injects
if (typeof window !== 'undefined' && (window as any).document) {
  const g: any = window as any;
  g.asciiMaze = g.asciiMaze || {};
  g.asciiMaze.start = start;
  if (!g.asciiMazeStart) {
    g.asciiMazeStart = (el?: any) => {
      console.warn(
        '[asciiMaze] window.asciiMazeStart is deprecated; use import { start } ... or window.asciiMaze.start'
      );
      return start(el);
    };
  }
  // Guard against duplicate auto-start
  if (!g.asciiMaze._autoStarted) {
    g.asciiMaze._autoStarted = true;
    setTimeout(() => {
      try {
        if (document.getElementById('ascii-maze-output')) start();
      } catch {
        /* ignore */
      }
    }, 20);
  }
}
