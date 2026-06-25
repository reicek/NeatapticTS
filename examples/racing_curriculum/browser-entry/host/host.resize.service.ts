/**
 * Racing curriculum network canvas resize service.
 *
 * Mirrors the Flappy Bird responsive sizing surface so the network canvas
 * backing store stays in sync with the host element and viewport.
 */

/**
 * Installs racing network canvas resize listeners.
 *
 * On every window `resize` event the service measures the host element and
 * updates the canvas backing store, then invokes `onResize` so the caller can
 * schedule a network redraw.
 *
 * If the host element has no CSS layout yet (for example in jsdom tests), the
 * service falls back to `window.innerWidth * 0.25` by `window.innerHeight * 0.35`
 * so the canvas still has a positive backing size. The returned `redraw` handle
 * is intentionally a no-op; the host schedules redraws through `onResize`.
 *
 * @param networkCanvas - Network visualization canvas to resize.
 * @param networkCanvasHost - Host element whose bounds drive the canvas size.
 * @param onResize - Callback invoked after the canvas backing size changes.
 * @returns Uninstall, resize, and redraw handles.
 *
 * @example
 * ```ts
 * const { uninstall, resize, redraw } = installRacingNetworkResize(
 *   networkCanvas,
 *   networkCanvasHost,
 *   () => renderNetworkView(networkCanvas, graph, options),
 * );
 * resize();
 * window.addEventListener('beforeunload', uninstall);
 * ```
 */
export function installRacingNetworkResize(
  networkCanvas: HTMLCanvasElement,
  networkCanvasHost: HTMLElement,
  onResize: () => void,
): { uninstall: () => void; resize: () => void; redraw: () => void } {
  const applySize = (): void => {
    const hostWidth = networkCanvasHost.clientWidth;
    const hostHeight = networkCanvasHost.clientHeight;
    const fallbackWidth = Math.max(1, Math.floor(window.innerWidth * 0.25));
    const fallbackHeight = Math.max(1, Math.floor(window.innerHeight * 0.35));
    const widthPx = hostWidth > 0 ? hostWidth : fallbackWidth;
    const heightPx = hostHeight > 0 ? hostHeight : fallbackHeight;
    if (networkCanvas.width !== widthPx || networkCanvas.height !== heightPx) {
      networkCanvas.width = widthPx;
      networkCanvas.height = heightPx;
      onResize();
    }
  };

  const handleWindowResize = (): void => {
    applySize();
  };

  applySize();
  window.addEventListener('resize', handleWindowResize);

  return {
    uninstall: () => {
      window.removeEventListener('resize', handleWindowResize);
    },
    resize: applySize,
    redraw: () => {
      // no-op; host handles redraw
    },
  };
}
