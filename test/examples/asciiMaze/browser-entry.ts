import { BrowserTerminalUtility } from './browserTerminalUtility';
import { createBrowserLogger } from './browserLogger';
import { DashboardManager } from './dashboardManager';
import { EvolutionEngine } from './evolutionEngine';
import * as mazes from './mazes';
import { INetwork } from './interfaces';

/** Default host container id used when a string is supplied to `start`. */
const DEFAULT_CONTAINER_ID = 'ascii-maze-output';
/** Width change (px) that triggers a dashboard redraw. */
const RESIZE_WIDTH_THRESHOLD = 8;
/** Debounce for fallback window.resize handler (ms). */
const RESIZE_DEBOUNCE_MS = 120;
/** Delay before auto-starting the demo when loaded as a script (ms). */
const AUTO_START_DELAY_MS = 20;

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
  /** Stop the running curriculum. This will also abort the internal signal. */
  stop: () => void;
  /** Whether the curriculum is currently active (not finished or stopped). */
  isRunning: () => boolean;
  /** Promise that resolves when the curriculum naturally finishes or is stopped. */
  done: Promise<void>;
  /** Subscribe to per-generation telemetry events. Returns an unsubscribe function. */
  onTelemetry: (cb: (telemetry: Record<string, unknown>) => void) => () => void;
  /** Return the last telemetry snapshot produced by the dashboard, if any. */
  getTelemetry: () => unknown;
}

/**
 * Start the ASCII Maze evolutionary demo.
 *
 * Boots a multi-maze curriculum (in a fixed order) that evolves a NEAT population
 * repeatedly until each maze is solved (or `stop()` is invoked). The returned
 * handle provides lifecycle controls and lightweight telemetry hooks.
 *
 * @param container - Element id or HTMLElement to host the demo (defaults to 'ascii-maze-output').
 * @param opts - Optional configuration. `opts.signal` (AbortSignal) can be supplied by the caller to
 *               cooperatively cancel the curriculum. Calling `stop()` will also trigger an abort.
 * @returns A {@link AsciiMazeRunHandle} exposing lifecycle controls and telemetry hooks.
 */
export async function start(
  container: string | HTMLElement = 'ascii-maze-output',
  opts: { signal?: AbortSignal } = {}
): Promise<AsciiMazeRunHandle> {
  const hostElement =
    typeof container === 'string'
      ? document.getElementById(container)
      : container;

  const archiveElement = hostElement
    ? (hostElement.querySelector('#ascii-maze-archive') as HTMLElement)
    : null;
  const liveElement = hostElement
    ? (hostElement.querySelector('#ascii-maze-live') as HTMLElement)
    : null;

  // clearer will clear only the live area; archive remains
  const clearer = BrowserTerminalUtility.createTerminalClearer(
    liveElement ?? undefined
  );
  const liveLogger = createBrowserLogger(liveElement ?? undefined);
  const archiveLogger = createBrowserLogger(archiveElement ?? undefined);

  // DashboardManager will use live logger for ongoing redraws and archive logger to append solved blocks
  const dashboard = new DashboardManager(
    clearer,
    liveLogger as any,
    archiveLogger as any
  );

  // Telemetry listeners stored in a Set to simplify add/remove and avoid array splices
  const telemetryListeners = new Set<
    (telemetry: Record<string, unknown>) => void
  >();

  // Inject hook so DashboardManager can call into listeners. We iterate a snapshot
  // to avoid reentrancy if a listener unsubscribes during iteration.
  (dashboard as any)._telemetryHook = (telemetry: Record<string, unknown>) => {
    const snapshot = Array.from(telemetryListeners);
    for (const listener of snapshot) {
      try {
        listener(telemetry);
      } catch {
        // Keep the loop robust: telemetry listeners must not be able to break evolution
      }
    }
  };

  // Responsive resize: re-render dashboard when host width changes significantly.
  try {
    const observeTarget =
      hostElement ?? document.getElementById('ascii-maze-output');
    if (observeTarget && typeof ResizeObserver !== 'undefined') {
      let lastObservedWidth = observeTarget.clientWidth;
      const resizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const width = entry.contentRect.width;
          if (Math.abs(width - lastObservedWidth) > RESIZE_WIDTH_THRESHOLD) {
            // threshold to avoid noisy redraws
            lastObservedWidth = width;
            try {
              (dashboard as any).redraw?.([], undefined);
            } catch {
              // ignore redraw errors
            }
          }
        }
      });
      resizeObserver.observe(observeTarget);
    } else if (observeTarget) {
      // Fallback: window resize listener (debounced)
      let debounceTimer: number | undefined = undefined;
      const handler = () => {
        if (typeof debounceTimer === 'number') clearTimeout(debounceTimer);
        debounceTimer = window.setTimeout(() => {
          try {
            (dashboard as any).redraw?.([], undefined);
          } catch {
            // ignore
          }
        }, 120);
      };
      window.addEventListener('resize', handler);
    }
  } catch {
    // ignore resize wiring errors
  }

  // Inner runner (previously assigned to window.asciiMazeStart). Kept internal for ESM API.
  let cancelled = false;
  const internalController = new AbortController();
  const externalSignal = opts.signal;

  /**
   * Compose an external signal with the internal controller. If the external
   * signal aborts, the internal controller is aborted. If no external signal
   * is provided, the internal signal is returned directly.
   */
  const composeAbortSignal = (maybeExternal?: AbortSignal): AbortSignal => {
    if (!maybeExternal) return internalController.signal;
    if ((maybeExternal as any).aborted) return maybeExternal; // already aborted
    try {
      maybeExternal.addEventListener(
        'abort',
        () => {
          try {
            internalController.abort();
          } catch {
            // ignore
          }
        },
        { once: true }
      );
    } catch {
      // ignore event wiring errors
    }
    return internalController.signal;
  };

  const combinedSignal = composeAbortSignal(externalSignal);
  let running = true;

  const runCurriculum = async () => {
    // Run mazes in the same curriculum order as the e2e test and mirror its
    // evolution settings where practical. We intentionally disable the
    // post-evolution backprop refinement (lamarckian iterations = 0) for
    // the browser demo as requested.
    const curriculumOrder = [
      'tiny',
      'spiralSmall',
      'spiral',
      'small',
      'medium',
      'medium2',
      'large',
      'minotaur',
    ];

    // Centralized phase settings table to avoid switch/case and magic numbers.
    const PHASE_SETTINGS: Record<
      string,
      { agentMaxSteps: number; maxGenerations: number }
    > = {
      tiny: { agentMaxSteps: 100, maxGenerations: 200 },
      spiralSmall: { agentMaxSteps: 100, maxGenerations: 200 },
      spiral: { agentMaxSteps: 150, maxGenerations: 300 },
      small: { agentMaxSteps: 50, maxGenerations: 300 },
      medium: { agentMaxSteps: 250, maxGenerations: 400 },
      medium2: { agentMaxSteps: 300, maxGenerations: 400 },
      large: { agentMaxSteps: 400, maxGenerations: 500 },
      minotaur: { agentMaxSteps: 700, maxGenerations: 600 },
    };

    // Carry the winning network forward between phases (curriculum transfer)
    let lastBestNetwork: INetwork | undefined = undefined;

    for (const phaseKey of curriculumOrder) {
      if (cancelled) break;
      const maze = (mazes as any)[phaseKey] as string[];
      if (!Array.isArray(maze)) continue; // skip missing exports

      const phaseSettings = PHASE_SETTINGS[phaseKey] ?? {
        agentMaxSteps: 1000,
        maxGenerations: 500,
      };
      const agentMaxSteps = phaseSettings.agentMaxSteps;
      const maxGenerations = phaseSettings.maxGenerations;

      try {
        const result = await EvolutionEngine.runMazeEvolution({
          mazeConfig: { maze },
          agentSimConfig: { maxSteps: agentMaxSteps },
          evolutionAlgorithmConfig: {
            allowRecurrent: true,
            popSize: 40,
            // Run indefinitely until solved; remove stagnation pressure for demo clarity
            maxStagnantGenerations: Infinity,
            minProgressToPass: 99,
            maxGenerations: Infinity,
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
            label: `browser-${phaseKey}`,
          },
          cancellation: { isCancelled: () => cancelled },
          signal: combinedSignal,
        });

        try {
          console.log(
            '[asciiMaze] maze solved',
            phaseKey,
            (result as any)?.bestResult?.progress
          );
        } catch {
          // ignore logging errors
        }

        if (result && (result as any).bestNetwork)
          lastBestNetwork = (result as any).bestNetwork;
      } catch (error) {
        console.error('Error while running maze', phaseKey, error);
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
      } catch {
        // ignore
      }
    },
    isRunning: () => running && !cancelled,
    done: Promise.resolve(donePromise).catch(() => {}) as Promise<void>,
    onTelemetry: (cb) => {
      telemetryListeners.add(cb as any);
      return () => {
        telemetryListeners.delete(cb as any);
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
