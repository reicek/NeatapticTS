import { BrowserTerminalUtility } from './browserTerminalUtility';
import { createBrowserLogger } from './browserLogger';
import { DashboardManager } from './dashboardManager';
import { EvolutionEngine } from './evolutionEngine';
import { INetwork } from './interfaces';
import { MazeGenerator } from './mazes';
import { refineWinnerWithBackprop } from './refineWinner';

/** Default host container id used when a string is supplied to `start`. */
const DEFAULT_CONTAINER_ID = 'ascii-maze-output';
/** Width delta (px) that triggers a dashboard redraw to avoid noisy renders. */
const RESIZE_WIDTH_THRESHOLD = 8;
/** Debounce for fallback window.resize handler (ms). */
const RESIZE_DEBOUNCE_MS = 120;
/** Delay before auto-starting the demo when loaded as a script (ms). */
const AUTO_START_DELAY_MS = 20;
/** Minimum progress percentage required to consider a maze solved (mirrors e2e test). */
const MIN_PROGRESS_TO_PASS = 90;
/** Default stagnation generation threshold used across most curriculum phases. */
const DEFAULT_MAX_STAGNANT_GENERATIONS = 50;
/** Default max generations (hard cap) for most curriculum phases. */
const DEFAULT_MAX_GENERATIONS = 100;
/** Per-generation log/telemetry frequency for interactive demo (always 1). */
const PER_GENERATION_LOG_FREQUENCY = 1;
/** Initial side length (cells) of the generated procedural maze. */
const INITIAL_MAZE_DIMENSION = 8;
/** Maximum side length (cells) to grow the maze to. */
const MAX_MAZE_DIMENSION = 40;
/** Dimension increment (cells per axis) applied after each solved maze. */
const MAZE_DIMENSION_INCREMENT = 4;
/** Maximum agent steps before termination (scaled mazes). */
const AGENT_MAX_STEPS = 600;
/** Population size for evolution across maze scalings. */
const POPULATION_SIZE = 20;

/**
 * Create immutable evolution settings for a given maze dimension.
 *
 * @param dimension - Maze side length in cells (square maze).
 * @returns Readonly configuration object consumed by a single evolution run.
 */
function createEvolutionSettings(dimension: number) {
  return {
    agentMaxSteps: AGENT_MAX_STEPS,
    popSize: POPULATION_SIZE,
    maxStagnantGenerations: DEFAULT_MAX_STAGNANT_GENERATIONS,
    maxGenerations: DEFAULT_MAX_GENERATIONS,
    lamarckianIterations: 4,
    lamarckianSampleSize: 12,
    mazeFactory: () => new MazeGenerator(dimension, dimension).generate(),
  } as const;
}

/**
 * Lightweight telemetry hub using a Set + snapshot iteration (micro-optimized for small listener counts).
 * EventTarget would work here, but Set keeps call overhead extremely low and avoids string event names.
 */
class TelemetryHub<TTelemetry extends Record<string, unknown>> {
  /** Registered listener callbacks (unique). */
  #listeners = new Set<(payload: TTelemetry) => void>();

  /** Add a listener and return an unsubscribe function. */
  add(listener: (payload: TTelemetry) => void): () => void {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  }

  /** Dispatch to a snapshot of listeners so mutations during iteration are safe. */
  dispatch(payload: TTelemetry): void {
    // Step: Snapshot listeners (defensive against unsubscribe inside callback)
    const snapshot = Array.from(this.#listeners);
    for (const listener of snapshot) {
      try {
        listener(payload);
      } catch {
        // Listener exceptions are isolated so evolution cannot be disrupted.
      }
    }
  }
}

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
  onTelemetry: (
    listener: (telemetry: Record<string, unknown>) => void
  ) => () => void;
  /** Return the last telemetry snapshot produced by the dashboard, if any. */
  getTelemetry: () => unknown;
}

/**
 * Start the ASCII Maze evolutionary demo (progressively larger procedural mazes).
 *
 * Steps:
 * 1. Generate an initial procedural maze (20x20) and evolve a NEAT population.
 * 2. Emit telemetry each generation (logEvery=1) and pace via requestAnimationFrame for UI responsiveness.
 * 3. When solved (progress >= MIN_PROGRESS_TO_PASS) grow maze size by +2 on each axis (up to 40x40) and repeat.
 * 4. Continue until maximum dimension reached or `stop()` / external abort invoked.
 *
 * This progressive curriculum demonstrates transfer of learned structure to larger mazes via fresh evolution runs.
 *
 * @param container - Element id or HTMLElement to host the demo (defaults to 'ascii-maze-output').
 * @param opts - Optional configuration. `opts.signal` (AbortSignal) can be supplied by the caller to
 *               cooperatively cancel the curriculum. Calling `stop()` will also trigger an abort.
 * @returns A {@link AsciiMazeRunHandle} exposing lifecycle controls and telemetry hooks.
 */
export async function start(
  container: string | HTMLElement = DEFAULT_CONTAINER_ID,
  opts: { signal?: AbortSignal } = {}
): Promise<AsciiMazeRunHandle> {
  // Step 0: Resolve host elements & loggers
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

  // Telemetry hub mediating dashboard -> external listeners
  const telemetryHub = new TelemetryHub<Record<string, unknown>>();
  (dashboard as any)._telemetryHook = (telemetry: Record<string, unknown>) =>
    telemetryHub.dispatch(telemetry);

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
        }, RESIZE_DEBOUNCE_MS);
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
   * Compose the internal abort controller signal with an optional external signal.
   * If the external signal aborts first we propagate that abort to the internal controller.
   *
   * @param maybeExternal - Optional external AbortSignal provided by the caller.
   * @returns A signal that will abort when either the internal controller or the external signal aborts.
   */
  const composeAbortSignal = (maybeExternal?: AbortSignal): AbortSignal => {
    // Modern path: AbortSignal.any is available (2023+ browsers & recent Node) -> compose both.
    if (maybeExternal) {
      if ((maybeExternal as any).aborted) return maybeExternal;
      if (typeof (AbortSignal as any).any === 'function') {
        try {
          return (AbortSignal as any).any([
            maybeExternal,
            internalController.signal,
          ]);
        } catch {
          // fall through to manual wiring
        }
      }
      try {
        maybeExternal.addEventListener(
          'abort',
          () => {
            try {
              internalController.abort();
            } catch {
              /* ignore */
            }
          },
          { once: true }
        );
      } catch {
        // ignore event wiring errors
      }
    }
    return internalController.signal;
  };

  const combinedSignal = composeAbortSignal(externalSignal);
  let running = true;
  // Immediate abort reaction: ensure handle.isRunning() reflects abort promptly (before
  // heavy generation loop completes). This decouples external cancellation latency from
  // potentially long per-generation work inside EvolutionEngine, improving test robustness.
  try {
    combinedSignal.addEventListener(
      'abort',
      () => {
        // Step: mark cancelled state so cooperative checks exit early.
        cancelled = true;
        // Step: reflect non-running state immediately for consumers polling isRunning().
        running = false;
        // Step: resolve done Promise eagerly; underlying evolution will short‑circuit soon.
        try {
          resolveDone?.();
        } catch {
          /* ignore */
        }
      },
      { once: true }
    );
  } catch {
    /* ignore listener wiring errors */
  }

  // Progressive scaling state
  let currentDimension = INITIAL_MAZE_DIMENSION;
  let resolveDone: (() => void) | undefined;
  const donePromise = new Promise<void>((resolve) => (resolveDone = resolve));

  // --- Cross-maze curriculum transfer state ---
  // Holds the best network from the most recently completed evolution run.
  // Updated after each run finishes and passed as `initialBestNetwork` into the next
  // run to encourage structural transfer to larger mazes. Starts undefined so the
  // first maze evolves from a fresh random population.
  let previousBestNetwork: INetwork | undefined;

  /** Schedule a callback on the next animation frame with a setTimeout(0) fallback. */
  const scheduleNextMaze = (cb: () => void) => {
    try {
      if (typeof requestAnimationFrame === 'function')
        requestAnimationFrame(cb);
      else setTimeout(cb, 0);
    } catch {
      setTimeout(cb, 0);
    }
  };

  const runEvolution = async () => {
    if (cancelled) {
      running = false;
      resolveDone?.();
      return;
    }
    // Best network carried forward across curriculum phases (progressively larger mazes).
    // This lets the next maze seed its population with the prior best to encourage transfer.
    const settings = createEvolutionSettings(currentDimension);
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
          minProgressToPass: MIN_PROGRESS_TO_PASS,
          maxGenerations: settings.maxGenerations,
          autoPauseOnSolve: false,
          stopOnlyOnSolve: false,
          lamarckianIterations: settings.lamarckianIterations,
          lamarckianSampleSize: settings.lamarckianSampleSize,
          initialBestNetwork: previousBestNetwork,
        },
        reportingConfig: {
          dashboardManager: dashboard,
          logEvery: PER_GENERATION_LOG_FREQUENCY,
          label: `browser-procedural-${currentDimension}x${currentDimension}`,
          paceEveryGeneration: true, // custom flag (consumed if supported) to yield between generations
        },
        cancellation: { isCancelled: () => cancelled },
        signal: combinedSignal,
      });
      const progress = (result as any)?.bestResult?.progress;
      // Capture & refine best network for seeding next curriculum phase (if any).
      try {
        const bestNet = (result as any)?.bestNetwork as INetwork | undefined;
        if (bestNet) {
          const refined = refineWinnerWithBackprop(bestNet as any);
          previousBestNetwork = (refined as any) || bestNet;
        }
      } catch {
        /* ignore refinement */
      }
      solved = typeof progress === 'number' && progress >= MIN_PROGRESS_TO_PASS;
      try {
        console.log(
          '[asciiMaze] maze complete',
          currentDimension,
          'solved?',
          solved,
          'progress',
          progress
        );
      } catch {
        /* ignore */
      }
    } catch (error) {
      console.error(
        'Error while running procedural maze',
        currentDimension,
        error
      );
    }

    if (!cancelled && solved && currentDimension < MAX_MAZE_DIMENSION) {
      currentDimension = Math.min(
        currentDimension + MAZE_DIMENSION_INCREMENT,
        MAX_MAZE_DIMENSION
      );
      scheduleNextMaze(() => runEvolution());
    } else {
      running = false;
      resolveDone?.();
    }
  };

  // Kick off first maze immediately
  runEvolution();

  const handle: AsciiMazeRunHandle = {
    stop: () => {
      cancelled = true;
      try {
        internalController.abort();
      } catch {
        // ignore
      }
      // Reflect stopped state immediately.
      running = false;
    },
    // Include AbortSignal aborted state so external aborts flip isRunning() without relying solely on listener side-effects.
    isRunning: () => running && !cancelled && !combinedSignal.aborted,
    done: Promise.resolve(donePromise).catch(() => {}) as Promise<void>,
    onTelemetry: (telemetryCallback) =>
      telemetryHub.add(telemetryCallback as any),
    getTelemetry: () => (dashboard as any).getLastTelemetry?.(),
  };

  // (Pause UI removed; external host can manage pause via a future API if needed.)
  return handle;
}

// UMD-style compatibility + deprecated global.
// If loaded directly (no module loader), expose window.asciiMaze.start() and legacy asciiMazeStart().
declare const __webpack_require__: any; // silence TS if bundler injects
if (typeof window !== 'undefined' && (window as any).document) {
  const globalWindow: any = window as any;
  globalWindow.asciiMaze = globalWindow.asciiMaze || {};
  globalWindow.asciiMaze.start = start;
  if (!globalWindow.asciiMazeStart) {
    globalWindow.asciiMazeStart = (containerElement?: any) => {
      console.warn(
        '[asciiMaze] window.asciiMazeStart is deprecated; use import { start } ... or window.asciiMaze.start'
      );
      return start(containerElement);
    };
  }
  // Guard against duplicate auto-start
  if (!globalWindow.asciiMaze._autoStarted) {
    globalWindow.asciiMaze._autoStarted = true;
    setTimeout(() => {
      try {
        if (document.getElementById(DEFAULT_CONTAINER_ID)) start();
      } catch {
        /* ignore */
      }
    }, AUTO_START_DELAY_MS);
  }
}
