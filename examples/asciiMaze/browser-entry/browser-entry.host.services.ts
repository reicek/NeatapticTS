import { BrowserTerminalUtility } from '../browserTerminalUtility';
import { createBrowserLogger } from '../browserLogger';
import { DashboardManager } from '../dashboardManager';
import type { INetwork } from '../interfaces';
import type { IMazeRunResult } from '../interfaces';
import { MazeUtils } from '../mazeUtils';
import { MazeVisualization } from '../mazeVisualization';
import {
  drawMazeNetworkVisualization,
  type MazeHitArea,
} from './network-view/network-view';
import { exportVisualizationGraph } from '../../../src/architecture/network';
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

/** ID of the DOM tooltip div used for hover tooltips over the network canvas. */
const NETWORK_TOOLTIP_ELEMENT_ID = 'maze-network-tooltip';

/** Extra hover padding used to make canvas label tooltips easier to trigger. */
const TOOLTIP_HIT_AREA_PADDING_PX = 10;

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
  resetBrowserEntryHostPresentation(hostElements);

  const clearLiveMazeOutput = hostElements.liveElement
    ? BrowserTerminalUtility.createTerminalClearer(hostElements.liveElement)
    : () => {};
  const writeLiveMazeLine = hostElements.liveElement
    ? createBrowserLogger(hostElements.liveElement)
    : () => {};
  const archiveLogger = hostElements.archiveElement
    ? createBrowserLogger(hostElements.archiveElement)
    : () => {};
  const dashboard = new DashboardManager(
    () => {},
    () => {},
    archiveLogger as unknown as (...args: unknown[]) => void,
  );
  const telemetryHub = createTelemetryHub<DashboardTelemetryPayload>();
  const runtimeDashboard: DashboardPresentationAdapter = dashboard;
  let latestMaze: string[] = [];
  let latestResult: IMazeRunResult | undefined = undefined;
  let latestNetwork: INetwork | null = null;
  let latestHitAreas: MazeHitArea[] = [];
  let hoveredNodeIndices: readonly number[] = [];

  const redrawHoveredVisualization = (
    nextHoveredNodeIndices: readonly number[],
  ): void => {
    const hoverChanged =
      nextHoveredNodeIndices.length !== hoveredNodeIndices.length ||
      nextHoveredNodeIndices.some(
        (nodeIndex, nodeIndexOffset) =>
          nodeIndex !== hoveredNodeIndices[nodeIndexOffset],
      );
    if (!hoverChanged) {
      return;
    }

    hoveredNodeIndices = [...nextHoveredNodeIndices];
    latestHitAreas = renderLatestNetworkSnapshot(
      hostElements.networkCanvasElement,
      latestNetwork,
      hoveredNodeIndices,
    );
  };

  const baseDashboardUpdate = dashboard.update.bind(dashboard);
  const baseDashboardRedraw = dashboard.redraw.bind(dashboard);
  dashboard.update = (
    ...updateArgs: Parameters<DashboardManager['update']>
  ) => {
    latestMaze = updateArgs[0];
    latestResult = updateArgs[1];
    const networkCandidate = updateArgs[2] ?? null;
    latestNetwork = networkCandidate;
    baseDashboardUpdate(...updateArgs);
    renderBrowserLiveMazeSnapshot(
      latestMaze,
      latestResult,
      clearLiveMazeOutput,
      writeLiveMazeLine,
    );
    latestHitAreas = renderLatestNetworkSnapshot(
      hostElements.networkCanvasElement,
      networkCandidate,
      hoveredNodeIndices,
    );
    hoverTooltipController.refresh();
  };

  dashboard.redraw = (
    ...redrawArgs: Parameters<DashboardManager['redraw']>
  ) => {
    latestMaze = redrawArgs[0];
    baseDashboardRedraw(...redrawArgs);
    renderBrowserLiveMazeSnapshot(
      latestMaze,
      latestResult,
      clearLiveMazeOutput,
      writeLiveMazeLine,
    );
    latestHitAreas = renderLatestNetworkSnapshot(
      hostElements.networkCanvasElement,
      latestNetwork,
      hoveredNodeIndices,
    );
    hoverTooltipController.refresh();
  };

  runtimeDashboard._telemetryHook = (telemetry: DashboardTelemetryPayload) => {
    telemetryHub.dispatch(telemetry);
  };

  // Install hover tooltip system on the canvas after services are wired.
  const hoverTooltipController = installHoverTooltip(
    hostElements.networkCanvasElement,
    () => latestHitAreas,
    redrawHoveredVisualization,
  );

  const disposeResize = installResizeRedraw(
    hostElements.observeTarget,
    runtimeDashboard,
    () => {
      latestHitAreas = renderLatestNetworkSnapshot(
        hostElements.networkCanvasElement,
        latestNetwork,
        hoveredNodeIndices,
      );
      hoverTooltipController.refresh();
    },
  );

  return {
    dashboard,
    runtimeDashboard,
    telemetryHub,
    disposeResizeHandling: () => {
      hoverTooltipController.dispose();
      disposeResize();
    },
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
 * Clears stale browser-host presentation state before a fresh session starts.
 *
 * @param hostElements - Resolved host elements for the current browser session.
 * @returns Nothing.
 */
function resetBrowserEntryHostPresentation(
  hostElements: BrowserEntryHostElements,
): void {
  hostElements.liveElement?.replaceChildren();
  hostElements.archiveElement?.replaceChildren();

  const networkContext = hostElements.networkCanvasElement?.getContext('2d');
  if (networkContext && hostElements.networkCanvasElement) {
    networkContext.clearRect(
      0,
      0,
      hostElements.networkCanvasElement.width,
      hostElements.networkCanvasElement.height,
    );
    networkContext.fillStyle = '#050a12';
    networkContext.fillRect(
      0,
      0,
      hostElements.networkCanvasElement.width,
      hostElements.networkCanvasElement.height,
    );
  }

  const tooltipElement = document.getElementById(
    NETWORK_TOOLTIP_ELEMENT_ID,
  ) as HTMLElement | null;
  if (tooltipElement) {
    hideTooltip(tooltipElement);
  }
}

/**
 * Renders the current maze into the browser live pane without the terminal dashboard frame.
 *
 * @param latestMaze - Maze currently being evolved.
 * @param latestResult - Latest run result used for path highlighting.
 * @param clearLiveMazeOutput - Live-pane clear callback.
 * @param writeLiveMazeLine - Live-pane line writer.
 * @returns Nothing.
 */
function renderBrowserLiveMazeSnapshot(
  latestMaze: string[],
  latestResult: IMazeRunResult | undefined,
  clearLiveMazeOutput: () => void,
  writeLiveMazeLine: (...args: unknown[]) => void,
): void {
  clearLiveMazeOutput();
  if (latestMaze.length === 0) {
    return;
  }

  const resolvedPath = latestResult?.path;
  const resolvedAgentPosition =
    resolvedPath?.at(-1) ?? MazeUtils.findPosition(latestMaze, 'S');
  const mazeLines = MazeVisualization.visualizeMaze(
    latestMaze,
    resolvedAgentPosition,
    resolvedPath,
  ).split('\n');

  for (const mazeLine of mazeLines) {
    writeLiveMazeLine(mazeLine);
  }
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
      let lastObservedHeight = observeTarget.clientHeight;
      const resizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const width = entry.contentRect.width;
          const height = entry.contentRect.height;
          const widthChanged =
            Math.abs(width - lastObservedWidth) > C.RESIZE_WIDTH_THRESHOLD;
          const heightChanged =
            Math.abs(height - lastObservedHeight) > C.RESIZE_WIDTH_THRESHOLD;

          if (widthChanged || heightChanged) {
            lastObservedWidth = width;
            lastObservedHeight = height;
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
 * Returns hit areas from the render so the hover system can update without
 * a re-render on every pointer event.
 *
 * @param networkCanvasElement - Canvas target in the browser host.
 * @param networkCandidate - Current best network candidate from dashboard updates.
 * @returns Hit areas for hover tooltip testing, or empty array on failure.
 */
function renderLatestNetworkSnapshot(
  networkCanvasElement: HTMLCanvasElement | null,
  networkCandidate: INetwork | null,
  hoveredNodeIndices: readonly number[] = [],
): MazeHitArea[] {
  if (!networkCanvasElement || !networkCandidate) {
    return [];
  }

  if (!isVisualizationCompatibleNetwork(networkCandidate)) {
    return [];
  }

  try {
    const visualizationGraph = exportVisualizationGraph(
      networkCandidate as unknown as Network,
    );
    const resolvedNetwork = networkCandidate as unknown as Network;
    const result = drawMazeNetworkVisualization(
      networkCanvasElement,
      resolvedNetwork,
      visualizationGraph,
      hoveredNodeIndices,
    );
    return result.hitAreas;
  } catch {
    // Ignore visualization-only failures; dashboard telemetry continues rendering.
    return [];
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

// ---------------------------------------------------------------------------
// Hover tooltip system
// ---------------------------------------------------------------------------

/**
 * Installs mousemove and mouseleave handlers on the network canvas to show
 * educational hover tooltips above the current pointer position.
 *
 * @param canvasElement  - Canvas element to attach listeners to.
 * @param getHitAreas    - Getter for the latest hit areas from the last render.
 * @returns Cleanup function that removes the installed listeners.
 */
function installHoverTooltip(
  canvasElement: HTMLCanvasElement | null,
  getHitAreas: () => MazeHitArea[],
  onHoverNodesChanged: (hoveredNodeIndices: readonly number[]) => void,
): {
  dispose: () => void;
  refresh: () => void;
} {
  if (!canvasElement) {
    return {
      dispose: () => {},
      refresh: () => {},
    };
  }

  const tooltipElement = document.getElementById(
    NETWORK_TOOLTIP_ELEMENT_ID,
  ) as HTMLElement | null;
  let lastPointerClientX = 0;
  let lastPointerClientY = 0;
  let hasActivePointer = false;

  const refreshTooltipFromPointer = (): void => {
    if (!tooltipElement || !hasActivePointer) {
      return;
    }

    const rect = canvasElement.getBoundingClientRect();
    const scaleX = canvasElement.width / rect.width;
    const scaleY = canvasElement.height / rect.height;
    const canvasX = (lastPointerClientX - rect.left) * scaleX;
    const canvasY = (lastPointerClientY - rect.top) * scaleY;
    const pointerInsideCanvas =
      lastPointerClientX >= rect.left &&
      lastPointerClientX <= rect.right &&
      lastPointerClientY >= rect.top &&
      lastPointerClientY <= rect.bottom;

    if (!pointerInsideCanvas) {
      hideTooltip(tooltipElement);
      return;
    }

    const hitArea = resolveHoveredHitArea(canvasX, canvasY, getHitAreas());
    if (hitArea && !hitArea.suppressTooltip) {
      showTooltip(
        tooltipElement,
        hitArea,
        lastPointerClientX,
        lastPointerClientY,
      );
      return;
    }

    hideTooltip(tooltipElement);
  };

  const handleMouseMove = (event: MouseEvent): void => {
    lastPointerClientX = event.clientX;
    lastPointerClientY = event.clientY;
    hasActivePointer = true;
    const rect = canvasElement.getBoundingClientRect();
    const scaleX = canvasElement.width / rect.width;
    const scaleY = canvasElement.height / rect.height;
    const canvasX = (event.clientX - rect.left) * scaleX;
    const canvasY = (event.clientY - rect.top) * scaleY;
    const hitArea = resolveHoveredHitArea(canvasX, canvasY, getHitAreas());
    onHoverNodesChanged(hitArea?.hoveredNodeIndices ?? []);
    refreshTooltipFromPointer();
  };

  const handleMouseLeave = (): void => {
    hasActivePointer = false;
    onHoverNodesChanged([]);
    if (tooltipElement) hideTooltip(tooltipElement);
  };

  canvasElement.addEventListener('mousemove', handleMouseMove);
  canvasElement.addEventListener('mouseleave', handleMouseLeave);

  return {
    refresh: refreshTooltipFromPointer,
    dispose: () => {
      hasActivePointer = false;
      onHoverNodesChanged([]);
      canvasElement.removeEventListener('mousemove', handleMouseMove);
      canvasElement.removeEventListener('mouseleave', handleMouseLeave);
      if (tooltipElement) hideTooltip(tooltipElement);
    },
  };
}

/**
 * Finds the first hit area that contains the given canvas-space point.
 *
 * @param canvasX    - X coordinate in canvas backing-store pixels.
 * @param canvasY    - Y coordinate in canvas backing-store pixels.
 * @param hitAreas   - Hit areas from the last render pass.
 * @returns First matching hit area, or undefined.
 */
function resolveHoveredHitArea(
  canvasX: number,
  canvasY: number,
  hitAreas: MazeHitArea[],
): MazeHitArea | undefined {
  const directHitArea = hitAreas.find(
    (area) =>
      canvasX >= area.leftPx - TOOLTIP_HIT_AREA_PADDING_PX &&
      canvasX <= area.leftPx + area.widthPx + TOOLTIP_HIT_AREA_PADDING_PX &&
      canvasY >= area.topPx - TOOLTIP_HIT_AREA_PADDING_PX &&
      canvasY <= area.topPx + area.heightPx + TOOLTIP_HIT_AREA_PADDING_PX,
  );

  if (directHitArea) {
    return directHitArea;
  }

  return hitAreas
    .map((area) => ({
      area,
      distancePx: resolvePointToAreaDistancePx(canvasX, canvasY, area),
    }))
    .filter(({ distancePx }) => distancePx <= TOOLTIP_HIT_AREA_PADDING_PX)
    .toSorted(
      (leftEntry, rightEntry) => leftEntry.distancePx - rightEntry.distancePx,
    )[0]?.area;
}

/**
 * Resolves the shortest Euclidean distance from a point to a rectangle.
 *
 * @param canvasX - X coordinate in canvas pixels.
 * @param canvasY - Y coordinate in canvas pixels.
 * @param hitArea - Candidate hit area rectangle.
 * @returns Distance from the point to the rectangle edge, or 0 for interior points.
 */
function resolvePointToAreaDistancePx(
  canvasX: number,
  canvasY: number,
  hitArea: MazeHitArea,
): number {
  const horizontalGapPx = Math.max(
    hitArea.leftPx - canvasX,
    0,
    canvasX - (hitArea.leftPx + hitArea.widthPx),
  );
  const verticalGapPx = Math.max(
    hitArea.topPx - canvasY,
    0,
    canvasY - (hitArea.topPx + hitArea.heightPx),
  );

  return Math.hypot(horizontalGapPx, verticalGapPx);
}

/**
 * Positions and reveals the tooltip element near the current pointer.
 *
 * @param tooltipElement - DOM tooltip div.
 * @param hitArea        - Hit area providing heading and body paragraphs.
 * @param clientX        - Pointer X in viewport coordinates.
 * @param clientY        - Pointer Y in viewport coordinates.
 */
function showTooltip(
  tooltipElement: HTMLElement,
  hitArea: MazeHitArea,
  clientX: number,
  clientY: number,
): void {
  const offsetXPx = 14;
  tooltipElement.innerHTML = resolveTooltipHtml(hitArea);
  tooltipElement.style.display = 'block';

  const tooltipWidthPx = tooltipElement.offsetWidth;
  const tooltipHeightPx = tooltipElement.offsetHeight;
  const viewportWidthPx = window.innerWidth;
  const viewportHeightPx = window.innerHeight;
  const preferredLeftPx = clientX + offsetXPx;
  const preferredTopPx = clientY - tooltipHeightPx - 12;
  const resolvedLeftPx = Math.min(
    preferredLeftPx,
    viewportWidthPx - tooltipWidthPx - 12,
  );
  const resolvedTopPx =
    preferredTopPx >= 12
      ? preferredTopPx
      : Math.min(clientY + 18, viewportHeightPx - tooltipHeightPx - 12);

  tooltipElement.style.left = `${Math.max(12, resolvedLeftPx)}px`;
  tooltipElement.style.top = `${Math.max(12, resolvedTopPx)}px`;
}

/**
 * Hides the tooltip element.
 *
 * @param tooltipElement - DOM tooltip div.
 */
function hideTooltip(tooltipElement: HTMLElement): void {
  tooltipElement.style.display = 'none';
}

/**
 * Builds the inner HTML string for a tooltip from a hit area.
 *
 * @param hitArea - Source hit area.
 * @returns Safe HTML string for the tooltip body.
 */
function resolveTooltipHtml(hitArea: MazeHitArea): string {
  const escapedHeading = escapeHtml(hitArea.heading);
  const bodyHtml = hitArea.bodyParagraphs
    .map((paragraph) => `<p>${escapeHtml(paragraph)}</p>`)
    .join('');
  return `<strong class="maze-network-tooltip-heading">${escapedHeading}</strong>${bodyHtml}`;
}

/**
 * Escapes HTML special characters in a plain-text string.
 *
 * @param text - Input text.
 * @returns HTML-safe string.
 */
function escapeHtml(text: string): string {
  return text
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}
