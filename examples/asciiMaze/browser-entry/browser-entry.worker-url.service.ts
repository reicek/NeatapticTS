/**
 * Resolve the ASCII Maze evaluation worker bundle URL next to the browser bundle.
 *
 * The demo build emits the browser bundle and the worker bundle side-by-side
 * under `docs/assets`. Resolving the worker relative to the active browser
 * bundle keeps the example portable across docs hosting and local static runs.
 *
 * @returns Absolute URL string for `ascii-maze-evaluation.worker.bundle.js`.
 */
export function resolveAsciiMazeEvaluationWorkerBundleUrl(): string {
  const scriptElements = document.querySelectorAll('script[src]');
  const currentBundleScript = Array.from(scriptElements)
    .map((scriptElement) => scriptElement as HTMLScriptElement)
    .find((scriptElement) =>
      scriptElement.src.includes('ascii-maze.bundle.js'),
    );

  const workerBaseUrl = currentBundleScript?.src ?? window.location.href;
  const workerUrl = new URL(
    'ascii-maze-evaluation.worker.bundle.js',
    workerBaseUrl,
  );
  const bundleSearch = currentBundleScript
    ? new URL(currentBundleScript.src).search
    : '';

  if (bundleSearch) {
    workerUrl.search = bundleSearch;
  }

  return workerUrl.toString();
}
