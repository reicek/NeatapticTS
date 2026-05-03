import { BrowserTerminalUtility } from '../browserTerminalUtility';
import { createBrowserLogger } from '../browserLogger';
import { DashboardManager } from '../dashboardManager';
import type { INetwork } from '../interfaces';
import { exportVisualizationGraph } from '../../../src/architecture/network';
import { renderNetworkView } from '../../../src/visualization/visualization';
import type Network from '../../../src/architecture/network';
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

/** Minimum width reserved for the visualizer canvas during responsive layout. */
const MIN_NETWORK_CANVAS_WIDTH_PX = 280;
/** Minimum height reserved for the visualizer canvas during responsive layout. */
const MIN_NETWORK_CANVAS_HEIGHT_PX = 240;
/** Width-to-height ratio used by the ASCII maze network visualizer canvas. */
const NETWORK_CANVAS_ASPECT_RATIO = 0.6;

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
  let latestNetwork: INetwork | null = null;

  const baseDashboardUpdate = dashboard.update.bind(dashboard);
  dashboard.update = (...updateArgs: Parameters<DashboardManager['update']>) => {
    const networkCandidate = updateArgs[2] ?? null;
    latestNetwork = networkCandidate;
    baseDashboardUpdate(...updateArgs);
    renderLatestNetworkSnapshot(hostElements.networkCanvasElement, networkCandidate);
  };

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
      () => renderLatestNetworkSnapshot(hostElements.networkCanvasElement, latestNetwork),
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
  redrawNetworkSnapshot: () => void,
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
            redrawNetworkSnapshot();
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
        redrawNetworkSnapshot();
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

/**
 * Render the latest evolved network into the dedicated browser canvas panel.
 *
 * @param networkCanvasElement - Canvas target in the browser host.
 * @param networkCandidate - Current best network candidate from dashboard updates.
 */
function renderLatestNetworkSnapshot(
  networkCanvasElement: HTMLCanvasElement | null,
  networkCandidate: INetwork | null,
): void {
  if (!networkCanvasElement || !networkCandidate) {
    return;
  }

  if (!isVisualizationCompatibleNetwork(networkCandidate)) {
    return;
  }

  try {
    syncNetworkCanvasToPanel(networkCanvasElement);
    const visualizationGraph = exportVisualizationGraph(
      networkCandidate as unknown as Network,
    );
    renderNetworkView(networkCanvasElement, visualizationGraph, {
      nodeDimensions: { widthPx: 18, heightPx: 18 },
      panelPaddingPx: { topPx: 16, rightPx: 16, bottomPx: 16, leftPx: 16 },
    });
  } catch {
    // Ignore visualization-only failures; dashboard telemetry continues rendering.
  }
}

/**
 * Align canvas pixel dimensions to the responsive panel width before drawing.
 *
 * @param networkCanvasElement - Canvas target in the browser host.
 */
function syncNetworkCanvasToPanel(
  networkCanvasElement: HTMLCanvasElement,
): void {
  const measuredCanvasWidthPx = Math.floor(networkCanvasElement.clientWidth);
  const resolvedCanvasWidthPx = Math.max(
    MIN_NETWORK_CANVAS_WIDTH_PX,
    measuredCanvasWidthPx,
  );
  const resolvedCanvasHeightPx = Math.max(
    MIN_NETWORK_CANVAS_HEIGHT_PX,
    Math.floor(resolvedCanvasWidthPx * NETWORK_CANVAS_ASPECT_RATIO),
  );

  // Keep CSS size fluid while matching backing-store pixels for crisp rendering.
  networkCanvasElement.style.width = '100%';
  networkCanvasElement.style.height = `${resolvedCanvasHeightPx}px`;

  if (
    networkCanvasElement.width !== resolvedCanvasWidthPx ||
    networkCanvasElement.height !== resolvedCanvasHeightPx
  ) {
    networkCanvasElement.width = resolvedCanvasWidthPx;
    networkCanvasElement.height = resolvedCanvasHeightPx;
  }
}

/**
 * Guard that checks whether a runtime network can be exported as VisualizationGraphV1.
 *
 * @param networkCandidate - Runtime candidate from dashboard updates.
 * @returns True when the candidate exposes required visualization fields.
 */
function isVisualizationCompatibleNetwork(
  networkCandidate: INetwork,
): networkCandidate is INetwork & {
  nodes: unknown[];
  connections: unknown[];
  inputNodeIds: number[];
  outputNodeIds: number[];
  getTopologyIntent: () => 'feed-forward' | 'unconstrained';
} {
  const maybeNetwork = networkCandidate as Partial<INetwork> & {
    getTopologyIntent?: unknown;
  };

  return (
    Array.isArray(maybeNetwork.nodes) &&
    Array.isArray(maybeNetwork.connections) &&
    Array.isArray(maybeNetwork.inputNodeIds) &&
    Array.isArray(maybeNetwork.outputNodeIds) &&
    typeof maybeNetwork.getTopologyIntent === 'function'
  );
}
