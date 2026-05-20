/**
 * Browser-host bundle stub for the legacy Node multithreading facade.
 *
 * ASCII Maze uses the worker-payload evaluator for browser scoring instead of
 * `Network.evolve(...)`'s older `child_process`-backed TestWorker path. The
 * host bundle aliases the legacy facade to this empty shelf so esbuild can keep
 * Node-only worker code out of the browser asset while preserving the shared
 * source API for Node builds.
 */

interface BrowserEntryMultiStub {
  /** No legacy TestWorker loader is available inside the browser host bundle. */
  readonly workers?: undefined;
}

/** Empty browser-safe replacement for `src/multithreading/multi`. */
const BROWSER_ENTRY_MULTI_STUB: BrowserEntryMultiStub = {};

export default BROWSER_ENTRY_MULTI_STUB;
