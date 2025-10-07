/**
 * setupHelpers.ts
 *
 * Environment and setup utilities for the evolution engine.
   // Return  return async (): Promise<void> => {
    // Polling loop: after each tick, if the cooperative pause flag is set, wait another tick.
    // This keeps CPU usage minimal while allowing the host to pause/resume the evolution loop.
    while (true) {
      await preferredTick();
      // Note: using a permissive read of the global pause flag; undefined => not paused.
      if (!(globalThis as Record<string, unknown>).asciiMazePaused) return;
      // otherwise continue and await another tick before re-checking
    }
  };
};flush function used by the evolution loop.
  return async (): Promise<void> => {
    // Polling loop: after each tick, if the cooperative pause flag is set, wait another tick.
    // This keeps CPU usage minimal while allowing the host to pause/resume the evolution loop.
    while (true) {
      await preferredTick();
      // Note: using a permissive read of the global pause flag; undefined => not paused.
      if (!(globalThis as Record<string, unknown>).asciiMazePaused) return;
      // otherwise continue and await another tick before re-checking
    }
  };
};ibilities:
 * - Create cooperative frame-yielding functions for async evolution loops
 * - Initialize Node.js filesystem persistence helpers (fs, path)
 * - Build resilient logging writers with fallback chains
 *
 * All functions are pure/side-effect-free except where explicitly documented.
 *
 * @module setupHelpers
 */

/**
 * Minimal filesystem module shape for type safety (Node.js fs module subset).
 */
export interface FilesystemModule {
  existsSync?: (path: string) => boolean;
  mkdirSync?: (path: string, options?: { recursive?: boolean }) => void;
  writeFileSync?: (path: string, data: string | Buffer) => void;
  [key: string]: unknown;
}

/**
 * Minimal path module shape for type safety (Node.js path module subset).
 */
export interface PathModule {
  join?: (...paths: string[]) => string;
  resolve?: (...paths: string[]) => string;
  [key: string]: unknown;
}

/**
 * Dashboard manager shape for logging (optional log function).
 */
export interface DashboardManagerLike {
  logFunction?: (msg: string) => void;
  [key: string]: unknown;
}

/**
 * Create a cooperative frame-yielding function used by the evolution loop.
 *
 * Behaviour:
 * - Prefers `requestAnimationFrame` when available (browser hosts)
 * - Falls back to `setImmediate` when available (Node) or `setTimeout(...,0)` otherwise
 * - Respects a cooperative pause flag (`globalThis.asciiMazePaused`) by polling between ticks
 *   without busy-waiting. The returned function resolves once a single new frame/tick is available
 *   and the pause flag is not set
 *
 * Steps:
 * 1. Choose the preferred tick function based on the host runtime
 * 2. When called, await the preferred tick; if `asciiMazePaused` is true poll again after the tick
 * 3. Resolve once a tick passed while not paused
 *
 * @returns A function that yields cooperatively to the next animation frame / tick.
 *
 * @example
 * const flushToFrame = makeFlushToFrame();
 * await flushToFrame(); // yields to next frame/tick
 */
export const makeFlushToFrame = (): (() => Promise<void>) => {
  // Helper factories for the three tick primitives; each returns a Promise that resolves on the next tick.
  const rafTick = () =>
    new Promise<void>((resolve) =>
      (globalThis as Record<string, unknown>).requestAnimationFrame
        ? (
            (globalThis as Record<string, unknown>).requestAnimationFrame as (
              callback: () => void,
            ) => number
          )(() => resolve())
        : setTimeout(() => resolve(), 0),
    );
  const immediateTick = () =>
    new Promise<void>((resolve) =>
      typeof setImmediate === 'function'
        ? setImmediate(resolve)
        : setTimeout(resolve, 0),
    );
  const timeoutTick = () =>
    new Promise<void>((resolve) => setTimeout(resolve, 0));

  // Pick the most appropriate tick primitive for this host.
  const preferredTick =
    typeof (globalThis as Record<string, unknown>).requestAnimationFrame ===
    'function'
      ? rafTick
      : typeof setImmediate === 'function'
        ? immediateTick
        : timeoutTick;

  // Return the async flush function used by the evolution loop.
  return async (): Promise<void> => {
    // Polling loop: after each tick, if the cooperative pause flag is set, wait another tick.
    // This keeps CPU usage minimal while allowing the host to pause/resume the evolution loop.
    while (true) {
      await preferredTick();
      // Note: using a permissive read of the global pause flag; undefined => not paused.
      if (!(globalThis as Record<string, unknown>).asciiMazePaused) return;
      // otherwise continue and await another tick before re-checking
    }
  };
};

/**
 * Initialize persistence helpers (Node `fs` & `path`) when available and ensure the target
 * directory exists. This helper intentionally does nothing in browser-like hosts.
 *
 * Steps:
 * 1. Detect whether a Node-like `require` is available and attempt to load `fs` and `path`
 * 2. If both modules are available and `persistDir` is provided, ensure the directory exists
 *    by creating it recursively when necessary
 * 3. Return an object containing the (possibly null) `{ fs, path }` references for callers to use
 *
 * Notes:
 * - This helper deliberately performs defensive checks and swallows synchronous errors because
 *   persistence is optional in many host environments (tests, browser demos)
 * - No large allocations are performed here; the function returns lightweight references
 *
 * @param persistDir - Optional directory to ensure exists. If falsy, no filesystem mutations are attempted.
 * @returns Object with `{ fs, path }` where each value may be `null` when unavailable.
 *
 * @example
 * const { fs, path } = initPersistence('./snapshots');
 * if (fs && path) {
 *   fs.writeFileSync(path.join(dir, 'snapshot.json'), data);
 * }
 */
export const initPersistence = (
  persistDir: string | undefined,
): {
  fs: FilesystemModule | null;
  path: PathModule | null;
} => {
  let fs: FilesystemModule | null = null;
  let path: PathModule | null = null;

  // Step 1: Safe detection of Node-style `require` without crashing bundlers that rewrite `require`.
  try {
    const maybeRequire =
      (globalThis as Record<string, unknown>).require ??
      (typeof require === 'function' ? require : null);
    if (maybeRequire) {
      try {
        fs = (maybeRequire as (moduleName: string) => unknown)(
          'fs',
        ) as FilesystemModule;
        path = (maybeRequire as (moduleName: string) => unknown)(
          'path',
        ) as PathModule;
      } catch {
        // module not available or require denied; leave as null
      }
    }
  } catch {
    // Defensive: any host restriction => treat as not available.
  }

  // Step 2: Ensure directory exists if possible and requested.
  if (fs && typeof fs.existsSync === 'function' && persistDir) {
    try {
      if (!fs.existsSync(persistDir)) {
        // Use recursive mkdir where supported.
        if (typeof fs.mkdirSync === 'function') {
          fs.mkdirSync(persistDir, { recursive: true });
        }
      }
    } catch {
      // Best-effort: ignore filesystem permission errors or path issues.
    }
  }

  // Step 3: Return module references (may be null in browser-like hosts).
  return { fs, path };
};

/**
 * Build a resilient writer that attempts to write to Node stdout, then a provided
 * dashboard logger, and finally `console.log` as a last resort.
 *
 * Steps:
 * 1. If Node `process.stdout.write` is available, use it (no trailing newline forced)
 * 2. Else if `dashboardManager.logFunction` exists, call it
 * 3. Else fall back to `console.log` and trim the message
 *
 * Notes:
 * - Errors are swallowed; logging must never throw and disrupt the evolution loop
 * - This factory is allocation-light; the returned function only creates a trimmed string
 *   when falling back to `console.log`
 *
 * @param dashboardManager - Optional manager exposing `logFunction(msg:string)` used in some UIs.
 * @returns A function accepting a single string message to write.
 *
 * @example
 * const safeWrite = makeSafeWriter(dashboardManager);
 * safeWrite('[INFO] Generation 42 complete\n');
 */
export const makeSafeWriter = (
  dashboardManager: DashboardManagerLike | undefined,
): ((msg: string) => void) => {
  // Capture local references to avoid repeated property lookups at call time.
  const hasProcessStdout = (() => {
    try {
      return (
        typeof process !== 'undefined' &&
        process &&
        process.stdout &&
        typeof process.stdout.write === 'function'
      );
    } catch {
      return false;
    }
  })();

  const dashboardLogFn = (() => {
    try {
      return dashboardManager && dashboardManager.logFunction
        ? dashboardManager.logFunction.bind(dashboardManager)
        : null;
    } catch {
      return null;
    }
  })();

  return (msg: string) => {
    if (!msg && msg !== '') return; // ignore undefined/null

    // Fast path: Node stdout writer
    if (hasProcessStdout) {
      try {
        (
          process as unknown as { stdout: { write: (msg: string) => void } }
        ).stdout.write(msg);
        return;
      } catch {
        /* swallow and fall through */
      }
    }

    // Dashboard logger path
    if (dashboardLogFn) {
      try {
        dashboardLogFn(msg);
        return;
      } catch {
        /* swallow and fall through */
      }
    }

    // Final fallback: console.log with trimmed message to avoid accidental trailing whitespace
    try {
      if (typeof console !== 'undefined' && typeof console.log === 'function') {
        console.log((msg as string).trim());
      }
    } catch {
      /* swallow all logging errors */
    }
  };
};
