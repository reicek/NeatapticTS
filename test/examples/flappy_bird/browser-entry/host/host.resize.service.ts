import { clamp } from '../browser-entry.math.utils';
import {
  FLAPPY_UI_NETWORK_HOST_FIXED_HEIGHT_PX,
  FLAPPY_UI_NETWORK_HOST_INSET_PX,
  FLAPPY_VIEWPORT_MOBILE_MINIMAL_UI_BREAKPOINT_PX,
  FLAPPY_VIEWPORT_MINIMUM_SIMULATION_HEIGHT_PX,
  FLAPPY_VIEWPORT_MINIMUM_SIMULATION_HEIGHT_RATIO,
  FLAPPY_VIEWPORT_MIN_NETWORK_HEIGHT_BUDGET_PX,
  FLAPPY_VIEWPORT_NETWORK_ONLY_BREAKPOINT_PX,
  FLAPPY_VIEWPORT_SIMULATION_BOTTOM_MARGIN_PX,
  FLAPPY_VIEWPORT_VERTICAL_LAYOUT_GUTTER_PX,
} from '../../constants/constants';
import { FLAPPY_HOST_STATS_SPLIT_GAP } from './host.constants';
import {
  applyCanvasBackingSize,
  applySimulationCanvasBounds,
  resolveNetworkCanvasSizePx,
} from './host.canvas.service';

/**
 * Installs responsive viewport sizing for simulation and network canvases.
 *
 * @param canvas - Simulation canvas to resize.
 * @param containerElement - Width/height source.
 * @param mainSplitContainer - Main split panel host.
 * @param statsContainer - Stats host element.
 * @param statsSplitContainer - Stats split panel containing stats + network panes.
 * @param statsTableHost - Stats table host element.
 * @param networkCanvas - Network canvas.
 * @param networkCanvasHost - Network host element.
 * @param onNetworkResize - Callback after network resize.
 * @returns Nothing.
 */
export function installResponsiveViewportSizing(
  canvas: HTMLCanvasElement,
  containerElement: HTMLElement,
  mainSplitContainer: HTMLElement,
  statsContainer: HTMLElement,
  statsSplitContainer: HTMLElement,
  statsTableHost: HTMLElement,
  networkCanvas: HTMLCanvasElement,
  networkCanvasHost: HTMLElement,
  onNetworkResize: () => void,
): void {
  let pendingNetworkRedrawRequest = false;

  /**
   * Defers network redraw until after layout settles across two animation frames.
   *
   * @returns Nothing.
   */
  const requestNetworkRedrawAfterLayout = (): void => {
    if (pendingNetworkRedrawRequest) {
      return;
    }
    pendingNetworkRedrawRequest = true;

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        pendingNetworkRedrawRequest = false;
        onNetworkResize();
      });
    });
  };

  /**
   * Recomputes simulation/network canvas backing sizes from current layout constraints.
   *
   * @returns Nothing.
   */
  const applyCanvasSize = (): void => {
    // Step 1: Measure header and baseline layout constraints.
    const headerCanvasElement = containerElement.querySelector('canvas');
    const headerHeightPx = Math.max(
      0,
      headerCanvasElement instanceof HTMLCanvasElement
        ? headerCanvasElement.offsetHeight
        : 0,
    );

    // Step 2: Resolve stats/network panel sizing constraints.
    const preferredNetworkHeightPx = Number.parseInt(
      networkCanvasHost.dataset.preferredHeightPx ??
        `${networkCanvasHost.offsetHeight}`,
      10,
    );
    const nonNetworkStatsHeightPx = Math.max(
      0,
      statsContainer.scrollHeight - networkCanvasHost.offsetHeight,
    );

    const hardMinimumSimulationHeightPx = Math.max(
      FLAPPY_VIEWPORT_MINIMUM_SIMULATION_HEIGHT_PX,
      Math.floor(
        window.innerHeight * FLAPPY_VIEWPORT_MINIMUM_SIMULATION_HEIGHT_RATIO,
      ),
    );
    const totalCanvasBudgetPx = Math.max(
      1,
      containerElement.clientHeight - headerHeightPx,
    );
    const viewportWidthPx = Math.max(1, containerElement.clientWidth);
    const viewportHeightPx = Math.max(1, containerElement.clientHeight);
    const useMinimalMobileLayout =
      viewportWidthPx < FLAPPY_VIEWPORT_MOBILE_MINIMAL_UI_BREAKPOINT_PX;
    const useLandscapeSplitLayout = viewportWidthPx > viewportHeightPx;
    const useNetworkOnlyPanel =
      !useLandscapeSplitLayout &&
      viewportWidthPx < FLAPPY_VIEWPORT_NETWORK_ONLY_BREAKPOINT_PX;

    // Step 2.1: For mobile, hide stats/network panes and dedicate height to gameplay canvas.
    if (useMinimalMobileLayout) {
      mainSplitContainer.style.flexDirection = 'column';
      mainSplitContainer.style.gap = `${FLAPPY_VIEWPORT_VERTICAL_LAYOUT_GUTTER_PX}px`;
      statsSplitContainer.style.flexDirection = 'row';
      statsSplitContainer.style.alignItems = 'flex-start';
      statsSplitContainer.style.height = 'auto';
      statsTableHost.style.order = '0';
      networkCanvasHost.style.order = '1';

      statsContainer.style.display = 'none';
      statsContainer.style.height = '0px';
      statsContainer.style.maxHeight = '0px';
      statsContainer.style.overflowY = 'hidden';

      const mobileWidthPx = Math.max(
        1,
        Math.floor(containerElement.clientWidth),
      );
      const mobileHeightPx = Math.max(
        1,
        Math.floor(
          totalCanvasBudgetPx - FLAPPY_VIEWPORT_SIMULATION_BOTTOM_MARGIN_PX,
        ),
      );

      applyCanvasBackingSize(canvas, mobileWidthPx, mobileHeightPx);
      return;
    }

    // Step 2.2: Restore stats/network panes for non-mobile widths.
    statsContainer.style.display = 'block';

    // Step 2.3: Toggle top-level split orientation and inner panel ordering.
    mainSplitContainer.style.flexDirection = useLandscapeSplitLayout
      ? 'row'
      : 'column';
    mainSplitContainer.style.gap = `${FLAPPY_VIEWPORT_VERTICAL_LAYOUT_GUTTER_PX}px`;

    statsSplitContainer.style.flexDirection = useLandscapeSplitLayout
      ? 'column'
      : 'row';
    statsSplitContainer.style.alignItems = useLandscapeSplitLayout
      ? 'stretch'
      : 'flex-start';
    statsSplitContainer.style.height = useLandscapeSplitLayout
      ? '100%'
      : 'auto';
    statsSplitContainer.style.gap = useNetworkOnlyPanel
      ? '0'
      : FLAPPY_HOST_STATS_SPLIT_GAP;

    if (useLandscapeSplitLayout) {
      networkCanvasHost.style.order = '0';
      statsTableHost.style.order = '1';
    } else {
      statsTableHost.style.order = '0';
      networkCanvasHost.style.order = '1';
    }

    statsTableHost.style.display = useNetworkOnlyPanel ? 'none' : 'block';
    statsTableHost.style.flex = useNetworkOnlyPanel ? '0 0 auto' : '1 1 0';
    networkCanvasHost.style.flex = useNetworkOnlyPanel
      ? '1 1 100%'
      : '0 0 auto';
    networkCanvasHost.style.width = useNetworkOnlyPanel ? '100%' : 'auto';
    networkCanvasHost.style.maxWidth = '100%';

    const halfSplitTargetPx = Math.max(
      FLAPPY_VIEWPORT_MIN_NETWORK_HEIGHT_BUDGET_PX,
      Math.floor(
        (totalCanvasBudgetPx - FLAPPY_VIEWPORT_VERTICAL_LAYOUT_GUTTER_PX) * 0.5,
      ),
    );
    const maximumStatsHeightPx = Math.max(
      FLAPPY_VIEWPORT_MIN_NETWORK_HEIGHT_BUDGET_PX,
      totalCanvasBudgetPx -
        hardMinimumSimulationHeightPx -
        FLAPPY_VIEWPORT_VERTICAL_LAYOUT_GUTTER_PX,
    );
    const resolvedStatsPanelHeightPx = useLandscapeSplitLayout
      ? totalCanvasBudgetPx
      : Math.floor(
          clamp(
            halfSplitTargetPx,
            FLAPPY_VIEWPORT_MIN_NETWORK_HEIGHT_BUDGET_PX,
            maximumStatsHeightPx,
          ),
        );
    const minimumStatsPanelHeightPx =
      nonNetworkStatsHeightPx + FLAPPY_UI_NETWORK_HOST_FIXED_HEIGHT_PX;
    const adjustedStatsPanelHeightPx = useLandscapeSplitLayout
      ? resolvedStatsPanelHeightPx
      : Math.max(resolvedStatsPanelHeightPx, minimumStatsPanelHeightPx);

    const halfSplitTargetWidthPx = Math.max(
      FLAPPY_VIEWPORT_MIN_NETWORK_HEIGHT_BUDGET_PX,
      Math.floor(
        (viewportWidthPx - FLAPPY_VIEWPORT_VERTICAL_LAYOUT_GUTTER_PX) * 0.5,
      ),
    );
    const maximumStatsWidthPx = Math.max(
      FLAPPY_VIEWPORT_MIN_NETWORK_HEIGHT_BUDGET_PX,
      viewportWidthPx -
        FLAPPY_VIEWPORT_MIN_NETWORK_HEIGHT_BUDGET_PX -
        FLAPPY_VIEWPORT_VERTICAL_LAYOUT_GUTTER_PX,
    );
    const resolvedStatsPanelWidthPx = Math.floor(
      clamp(
        halfSplitTargetWidthPx,
        FLAPPY_VIEWPORT_MIN_NETWORK_HEIGHT_BUDGET_PX,
        maximumStatsWidthPx,
      ),
    );

    // Step 3: Clamp network pane height and trigger redraw when height changed.
    const resolvedNetworkHeightPx = FLAPPY_UI_NETWORK_HOST_FIXED_HEIGHT_PX;
    if (networkCanvasHost.offsetHeight !== resolvedNetworkHeightPx) {
      networkCanvasHost.style.height = `${resolvedNetworkHeightPx}px`;
      requestNetworkRedrawAfterLayout();
    }

    statsContainer.style.height = `${adjustedStatsPanelHeightPx}px`;
    statsContainer.style.maxHeight = `${adjustedStatsPanelHeightPx}px`;
    statsContainer.style.width = useLandscapeSplitLayout
      ? `${resolvedStatsPanelWidthPx}px`
      : '100%';
    statsContainer.style.maxWidth = useLandscapeSplitLayout
      ? `${resolvedStatsPanelWidthPx}px`
      : '100%';
    statsContainer.style.overflowY = useLandscapeSplitLayout
      ? 'auto'
      : 'hidden';

    // Step 4: Resolve simulation canvas size from remaining vertical budget.
    const availableWidthPx = useLandscapeSplitLayout
      ? Math.max(
          1,
          containerElement.clientWidth -
            resolvedStatsPanelWidthPx -
            FLAPPY_VIEWPORT_VERTICAL_LAYOUT_GUTTER_PX,
        )
      : Math.max(1, containerElement.clientWidth);
    const availableHeightPx = useLandscapeSplitLayout
      ? Math.max(
          1,
          totalCanvasBudgetPx - FLAPPY_VIEWPORT_SIMULATION_BOTTOM_MARGIN_PX,
        )
      : Math.max(
          1,
          totalCanvasBudgetPx -
            adjustedStatsPanelHeightPx -
            FLAPPY_VIEWPORT_VERTICAL_LAYOUT_GUTTER_PX -
            FLAPPY_VIEWPORT_SIMULATION_BOTTOM_MARGIN_PX,
        );

    // Step 5: Apply simulation canvas backing size.
    applySimulationCanvasBounds(canvas, availableWidthPx, availableHeightPx);

    // Step 6: Apply network canvas backing size and redraw when changed.
    const { widthPx, heightPx } = resolveNetworkCanvasSizePx(
      networkCanvasHost,
      FLAPPY_UI_NETWORK_HOST_INSET_PX,
    );
    if (applyCanvasBackingSize(networkCanvas, widthPx, heightPx)) {
      onNetworkResize();
    }
  };

  // Step 1: Initial sizing pass.
  applyCanvasSize();
  // Step 2: Queue first post-layout network redraw.
  requestNetworkRedrawAfterLayout();
  // Step 3: Listen to viewport resize.
  window.addEventListener('resize', applyCanvasSize);

  // Step 4: Attach container observer for responsive host resizing.
  if (typeof ResizeObserver === 'function') {
    const resizeObserver = new ResizeObserver(() => {
      applyCanvasSize();
      requestNetworkRedrawAfterLayout();
    });
    resizeObserver.observe(containerElement);
  }
}
