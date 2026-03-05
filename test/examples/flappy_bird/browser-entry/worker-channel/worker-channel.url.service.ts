/**
 * Resolves the evolution worker bundle URL relative to the active browser-entry bundle.
 *
 * @returns Absolute URL string for `flappy-evolution.worker.bundle.js`.
 */
export function resolveEvolutionWorkerBundleUrl(): string {
  // Step 1: Locate the currently loaded browser bundle script element.
  const scriptElements = document.querySelectorAll('script[src]');
  const currentBundleScript = Array.from(scriptElements)
    .map((scriptElement) => scriptElement as HTMLScriptElement)
    .find((scriptElement) =>
      scriptElement.src.includes('flappy-bird.bundle.js'),
    );

  // Step 2: Resolve worker URL relative to current bundle (or page URL fallback).
  const workerBaseUrl = currentBundleScript?.src ?? window.location.href;
  return new URL('flappy-evolution.worker.bundle.js', workerBaseUrl).toString();
}
