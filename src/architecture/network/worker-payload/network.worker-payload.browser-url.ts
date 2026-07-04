type BrowserDocumentLike = {
  currentScript?: { src?: string | null } | null;
};

/**
 * Options for resolving a browser worker asset URL relative to the current script or an explicit base URL override.
 *
 * When `baseUrl` is omitted the helper falls back to `document.currentScript.src`
 * and then `location.href`, keeping the worker and host bundle co-located by default.
 */
export interface BrowserWorkerAssetUrlOptions {
  /**
   * Explicit browser bundle URL to resolve the worker asset against.
   *
   * Use this when the worker should be anchored to a known host bundle instead
   * of the ambient `document.currentScript` or `location.href` context.
   */
  readonly baseUrl?: string;
}

/**
 * Resolves a browser worker asset beside the current script or page URL.
 *
 * This helper keeps the bundler boundary explicit: callers still name the
 * emitted worker asset they expect, while the library handles the common URL
 * math for browser demos, nested workers, and side-by-side bundle delivery.
 *
 * @example
 * ```ts
 * const sharedWorkerUrl = resolveBrowserWorkerAssetUrl(
 *   'shared-inference.worker.bundle.js',
 * );
 * ```
 *
 * @param workerAssetPath - Relative worker asset path emitted by the bundler.
 * @param options - Optional explicit base URL override.
 * @returns Absolute worker asset URL when a browser base URL is available.
 */
export function resolveBrowserWorkerAssetUrl(
  workerAssetPath: string,
  options: BrowserWorkerAssetUrlOptions = {},
): string | undefined {
  const browserBaseUrl =
    options.baseUrl ??
    resolveCurrentBrowserWorkerBaseUrl(
      globalThis.document as BrowserDocumentLike | undefined,
      globalThis.location?.href,
    );

  if (!browserBaseUrl) {
    return undefined;
  }

  return new URL(workerAssetPath, browserBaseUrl).toString();
}

function resolveCurrentBrowserWorkerBaseUrl(
  browserDocument: BrowserDocumentLike | undefined,
  currentLocationHref: string | undefined,
): string | undefined {
  return browserDocument?.currentScript?.src ?? currentLocationHref;
}
