import { BrowserTerminalUtility } from '../browserTerminalUtility';
import { createBrowserLogger } from '../browserLogger';
import { DashboardManager } from '../dashboardManager';
import type {
  DashboardPresentationAdapter,
  DashboardTelemetryPayload,
} from '../dashboardManager/dashboardManager.types';
import { BROWSER_ENTRY_CONSTANTS as C } from './browser-entry.constants';
import type {
  BrowserEntryHostElements,
  BrowserEntryHostServices,
  BrowserEntryTelemetryHub,
} from './browser-entry.types';

/**
 * Browser host-service boundary for the ASCII Maze browser entry.
 *
 * This module owns the DOM-facing dashboard wiring, telemetry fan-out, and
 * resize redraw behavior used by one browser-hosted ASCII Maze session.
 */

/**
 * Create the browser host services used by one ASCII Maze demo run.
 *
 * @param hostElements - Resolved host elements for live output, archive output, and resize observation.
 * @returns Dashboard, telemetry hub, runtime dashboard adapter, and resize cleanup.
 */
export const createBrowserEntryHostServices = (
  hostElements: BrowserEntryHostElements,
): BrowserEntryHostServices => {
  const clearer = BrowserTerminalUtility.createTerminalClearer(
    hostElements.liveElement ?? undefined,
  );
  const liveLogger = createBrowserLogger(hostElements.liveElement ?? undefined);
  const archiveLogger = createBrowserLogger(
    hostElements.archiveElement ?? undefined,
  );
  const dashboard = new DashboardManager(
    clearer,
    liveLogger as unknown as (...args: unknown[]) => void,
    archiveLogger as unknown as (...args: unknown[]) => void,
  );
  const telemetryHub = createTelemetryHub<DashboardTelemetryPayload>();
  const runtimeDashboard: DashboardPresentationAdapter = dashboard;

  runtimeDashboard._telemetryHook = (telemetry: DashboardTelemetryPayload) => {
    telemetryHub.dispatch(telemetry);
  };

  return {
    dashboard,
    runtimeDashboard,
    telemetryHub,
    disposeResizeHandling: installResizeRedraw(
      hostElements.observeTarget,
      runtimeDashboard,
    ),
  };
};

/**
 * Create a minimal telemetry hub backed by a Set of listeners.
 *
 * @returns A small hub optimized for browser demo listener counts.
 */
function createTelemetryHub<
  TTelemetry extends object,
>(): BrowserEntryTelemetryHub<TTelemetry> {
  const listeners = new Set<(payload: TTelemetry) => void>();

  return {
    add(listener: (payload: TTelemetry) => void): () => void {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },

    dispatch(payload: TTelemetry): void {
      const snapshot = Array.from(listeners);
      for (const listener of snapshot) {
        try {
          listener(payload);
        } catch {
          // Ignore listener failures so telemetry never destabilizes evolution.
        }
      }
    },
  };
}

/**
 * Attach dashboard redraw behavior to host resizes and return a cleanup function.
 *
 * @param observeTarget - Element whose width should trigger redraw checks.
 * @param runtimeDashboard - Shared dashboard presentation adapter with redraw support.
 * @returns Cleanup function that removes active observers or listeners.
 */
function installResizeRedraw(
  observeTarget: HTMLElement | null,
  runtimeDashboard: DashboardPresentationAdapter,
): () => void {
  if (!observeTarget) {
    return () => {};
  }

  try {
    if (typeof ResizeObserver !== 'undefined') {
      let lastObservedWidth = observeTarget.clientWidth;
      const resizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const width = entry.contentRect.width;
          if (Math.abs(width - lastObservedWidth) > C.RESIZE_WIDTH_THRESHOLD) {
            lastObservedWidth = width;
            safelyRedrawDashboard(runtimeDashboard);
          }
        }
      });
      resizeObserver.observe(observeTarget);
      return () => resizeObserver.disconnect();
    }

    let debounceTimer: number | undefined;
    const handleResize = () => {
      if (typeof debounceTimer === 'number') {
        clearTimeout(debounceTimer);
      }

      debounceTimer = window.setTimeout(() => {
        safelyRedrawDashboard(runtimeDashboard);
      }, C.RESIZE_DEBOUNCE_MS);
    };

    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      if (typeof debounceTimer === 'number') {
        clearTimeout(debounceTimer);
      }
    };
  } catch {
    return () => {};
  }
}

/**
 * Safely request a dashboard redraw without letting host issues break the run.
 *
 * @param runtimeDashboard - Shared dashboard presentation adapter with optional redraw support.
 */
function safelyRedrawDashboard(
  runtimeDashboard: DashboardPresentationAdapter,
): void {
  try {
    runtimeDashboard.redraw?.([], undefined);
  } catch {
    // Ignore redraw failures because they are presentation-only.
  }
}
