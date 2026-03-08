import {
  FLAPPY_UI_NETWORK_HOST_FIXED_HEIGHT_PX,
  FLAPPY_UI_NETWORK_HOST_INSET_PX,
  FLAPPY_VIEWPORT_VERTICAL_LAYOUT_GUTTER_PX,
} from '../../../constants/constants';
import { FLAPPY_HOST_STATS_SPLIT_GAP } from '../host.constants';
import {
  applyCanvasBackingSize,
  applySimulationCanvasBounds,
  resolveNetworkCanvasSizePx,
} from '../host.canvas.service';
import { FLAPPY_HOST_RESIZE_STYLE_TOKENS } from './host.resize.service.constants';
import {
  resolveMinimalMobileCanvasBounds,
  resolveSimulationCanvasBounds,
  resolveStatsPanelDimensions,
} from './host.resize.service.utils';
import type {
  DeferredNetworkRedrawController,
  ResponsiveViewportLayoutContext,
  ResponsiveViewportSizingElements,
  StatsPanelDimensions,
} from './host.resize.service.types';

/**
 * Creates a deferred redraw controller that waits for layout to settle.
 *
 * @param onNetworkResize - Callback after network resize.
 * @returns Deferred redraw controller.
 */
export function createDeferredNetworkRedrawController(
  onNetworkResize: () => void,
): DeferredNetworkRedrawController {
  let pendingNetworkRedrawRequest = false;

  return {
    requestNetworkRedrawAfterLayout: (): void => {
      // Step 1: Skip duplicate redraw requests while one is already queued.
      if (pendingNetworkRedrawRequest) {
        return;
      }
      pendingNetworkRedrawRequest = true;

      // Step 2: Wait two animation frames so flexbox layout has settled before redrawing.
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          pendingNetworkRedrawRequest = false;
          onNetworkResize();
        });
      });
    },
  };
}

/**
 * Applies the minimal mobile layout that hides the auxiliary panes.
 *
 * @param responsiveViewportSizingElements - Host elements participating in layout.
 * @param responsiveViewportLayoutContext - Responsive layout context.
 * @returns Nothing.
 */
export function applyMinimalMobileViewportLayout(
  responsiveViewportSizingElements: ResponsiveViewportSizingElements,
  responsiveViewportLayoutContext: ResponsiveViewportLayoutContext,
): void {
  // Step 1: Arrange the split containers so gameplay occupies the full vertical budget.
  responsiveViewportSizingElements.mainSplitContainer.style.flexDirection =
    FLAPPY_HOST_RESIZE_STYLE_TOKENS.columnDirection;
  responsiveViewportSizingElements.mainSplitContainer.style.gap = `${FLAPPY_VIEWPORT_VERTICAL_LAYOUT_GUTTER_PX}px`;
  responsiveViewportSizingElements.statsSplitContainer.style.flexDirection =
    FLAPPY_HOST_RESIZE_STYLE_TOKENS.rowDirection;
  responsiveViewportSizingElements.statsSplitContainer.style.alignItems =
    FLAPPY_HOST_RESIZE_STYLE_TOKENS.flexStartAlignment;
  responsiveViewportSizingElements.statsSplitContainer.style.height =
    FLAPPY_HOST_RESIZE_STYLE_TOKENS.autoSize;
  responsiveViewportSizingElements.statsTableHost.style.order =
    FLAPPY_HOST_RESIZE_STYLE_TOKENS.primaryOrder;
  responsiveViewportSizingElements.networkCanvasHost.style.order =
    FLAPPY_HOST_RESIZE_STYLE_TOKENS.secondaryOrder;

  // Step 2: Hide the stats pane so the simulation canvas gets the full available height.
  responsiveViewportSizingElements.statsContainer.style.display =
    FLAPPY_HOST_RESIZE_STYLE_TOKENS.hiddenDisplay;
  responsiveViewportSizingElements.statsContainer.style.height =
    FLAPPY_HOST_RESIZE_STYLE_TOKENS.zeroSizePx;
  responsiveViewportSizingElements.statsContainer.style.maxHeight =
    FLAPPY_HOST_RESIZE_STYLE_TOKENS.zeroSizePx;
  responsiveViewportSizingElements.statsContainer.style.overflowY =
    FLAPPY_HOST_RESIZE_STYLE_TOKENS.hiddenOverflow;

  // Step 3: Apply the simulation canvas backing size directly from the mobile viewport budget.
  const mobileCanvasBounds = resolveMinimalMobileCanvasBounds(
    responsiveViewportSizingElements.containerElement,
    responsiveViewportLayoutContext,
  );
  applyCanvasBackingSize(
    responsiveViewportSizingElements.canvas,
    mobileCanvasBounds.availableWidthPx,
    mobileCanvasBounds.availableHeightPx,
  );
}

/**
 * Applies the standard tablet and desktop layout and returns panel dimensions.
 *
 * @param responsiveViewportSizingElements - Host elements participating in layout.
 * @param responsiveViewportLayoutContext - Responsive layout context.
 * @param deferredNetworkRedrawController - Deferred redraw controller.
 * @returns Resolved stats panel dimensions.
 */
export function applyStandardViewportLayout(
  responsiveViewportSizingElements: ResponsiveViewportSizingElements,
  responsiveViewportLayoutContext: ResponsiveViewportLayoutContext,
  deferredNetworkRedrawController: DeferredNetworkRedrawController,
): StatsPanelDimensions {
  // Step 1: Restore the stats container and apply split orientation styles.
  responsiveViewportSizingElements.statsContainer.style.display =
    FLAPPY_HOST_RESIZE_STYLE_TOKENS.blockDisplay;
  applySplitContainerLayoutStyles(
    responsiveViewportSizingElements,
    responsiveViewportLayoutContext,
  );

  // Step 2: Resolve the stats panel dimensions from the current viewport budget.
  const statsPanelDimensions = resolveStatsPanelDimensions(
    responsiveViewportLayoutContext,
  );

  // Step 3: Clamp the network panel height and queue a redraw when the height changed.
  applyNetworkCanvasHostHeight(
    responsiveViewportSizingElements.networkCanvasHost,
    deferredNetworkRedrawController,
  );

  // Step 4: Apply the final stats container dimensions and scrolling rules.
  applyStatsContainerDimensions(
    responsiveViewportSizingElements.statsContainer,
    responsiveViewportLayoutContext,
    statsPanelDimensions,
  );

  return statsPanelDimensions;
}

/**
 * Applies simulation and network canvas backing sizes for the active layout.
 *
 * @param responsiveViewportSizingElements - Host elements participating in layout.
 * @param responsiveViewportLayoutContext - Responsive layout context.
 * @param statsPanelDimensions - Resolved stats panel dimensions.
 * @param onNetworkResize - Immediate network resize callback.
 * @returns Nothing.
 */
export function applyResponsiveCanvasBounds(
  responsiveViewportSizingElements: ResponsiveViewportSizingElements,
  responsiveViewportLayoutContext: ResponsiveViewportLayoutContext,
  statsPanelDimensions: StatsPanelDimensions,
  onNetworkResize: () => void,
): void {
  // Step 1: Resolve simulation canvas size from the remaining viewport budget.
  const simulationCanvasBounds = resolveSimulationCanvasBounds(
    responsiveViewportSizingElements.containerElement,
    responsiveViewportLayoutContext,
    statsPanelDimensions,
  );
  applySimulationCanvasBounds(
    responsiveViewportSizingElements.canvas,
    simulationCanvasBounds.availableWidthPx,
    simulationCanvasBounds.availableHeightPx,
  );

  // Step 2: Resize the network canvas backing store and redraw immediately when it changed.
  const { widthPx, heightPx } = resolveNetworkCanvasSizePx(
    responsiveViewportSizingElements.networkCanvasHost,
    FLAPPY_UI_NETWORK_HOST_INSET_PX,
  );
  if (
    applyCanvasBackingSize(
      responsiveViewportSizingElements.networkCanvas,
      widthPx,
      heightPx,
    )
  ) {
    onNetworkResize();
  }
}

/**
 * Installs window and container listeners for responsive host sizing.
 *
 * @param containerElement - Width and height source.
 * @param applyCanvasSize - Shared sizing callback.
 * @param deferredNetworkRedrawController - Deferred redraw controller.
 * @returns Nothing.
 */
export function installResponsiveViewportSizingListeners(
  containerElement: HTMLElement,
  applyCanvasSize: () => void,
  deferredNetworkRedrawController: DeferredNetworkRedrawController,
): void {
  // Step 1: Run the first post-layout redraw after the initial sizing pass.
  deferredNetworkRedrawController.requestNetworkRedrawAfterLayout();

  // Step 2: Recompute layout on window resize.
  window.addEventListener('resize', applyCanvasSize);

  // Step 3: Observe host-size changes when ResizeObserver is available.
  if (typeof ResizeObserver !== 'function') {
    return;
  }

  const resizeObserver = new ResizeObserver(() => {
    applyCanvasSize();
    deferredNetworkRedrawController.requestNetworkRedrawAfterLayout();
  });
  resizeObserver.observe(containerElement);
}

/**
 * Applies the split-container styles for the standard layout modes.
 *
 * @param responsiveViewportSizingElements - Host elements participating in layout.
 * @param responsiveViewportLayoutContext - Responsive layout context.
 * @returns Nothing.
 */
function applySplitContainerLayoutStyles(
  responsiveViewportSizingElements: ResponsiveViewportSizingElements,
  responsiveViewportLayoutContext: ResponsiveViewportLayoutContext,
): void {
  // Step 1: Configure the outer split direction based on landscape or stacked layout.
  responsiveViewportSizingElements.mainSplitContainer.style.flexDirection =
    responsiveViewportLayoutContext.useLandscapeSplitLayout
      ? FLAPPY_HOST_RESIZE_STYLE_TOKENS.rowDirection
      : FLAPPY_HOST_RESIZE_STYLE_TOKENS.columnDirection;
  responsiveViewportSizingElements.mainSplitContainer.style.gap = `${FLAPPY_VIEWPORT_VERTICAL_LAYOUT_GUTTER_PX}px`;

  // Step 2: Configure the inner stats/network split and alignment.
  responsiveViewportSizingElements.statsSplitContainer.style.flexDirection =
    responsiveViewportLayoutContext.useLandscapeSplitLayout
      ? FLAPPY_HOST_RESIZE_STYLE_TOKENS.columnDirection
      : FLAPPY_HOST_RESIZE_STYLE_TOKENS.rowDirection;
  responsiveViewportSizingElements.statsSplitContainer.style.alignItems =
    responsiveViewportLayoutContext.useLandscapeSplitLayout
      ? FLAPPY_HOST_RESIZE_STYLE_TOKENS.stretchAlignment
      : FLAPPY_HOST_RESIZE_STYLE_TOKENS.flexStartAlignment;
  responsiveViewportSizingElements.statsSplitContainer.style.height =
    responsiveViewportLayoutContext.useLandscapeSplitLayout
      ? FLAPPY_HOST_RESIZE_STYLE_TOKENS.fullSize
      : FLAPPY_HOST_RESIZE_STYLE_TOKENS.autoSize;
  responsiveViewportSizingElements.statsSplitContainer.style.gap =
    responsiveViewportLayoutContext.useNetworkOnlyPanel
      ? '0'
      : FLAPPY_HOST_STATS_SPLIT_GAP;

  // Step 3: Apply stats/network ordering and flex rules for the chosen layout.
  applyStatsPaneOrdering(
    responsiveViewportSizingElements.statsTableHost,
    responsiveViewportSizingElements.networkCanvasHost,
    responsiveViewportLayoutContext,
  );
}

/**
 * Applies ordering and flex styles for stats and network panes.
 *
 * @param statsTableHost - Stats table host element.
 * @param networkCanvasHost - Network canvas host element.
 * @param responsiveViewportLayoutContext - Responsive layout context.
 * @returns Nothing.
 */
function applyStatsPaneOrdering(
  statsTableHost: HTMLElement,
  networkCanvasHost: HTMLElement,
  responsiveViewportLayoutContext: ResponsiveViewportLayoutContext,
): void {
  // Step 1: Order the network panel first only for landscape split layouts.
  if (responsiveViewportLayoutContext.useLandscapeSplitLayout) {
    networkCanvasHost.style.order =
      FLAPPY_HOST_RESIZE_STYLE_TOKENS.primaryOrder;
    statsTableHost.style.order = FLAPPY_HOST_RESIZE_STYLE_TOKENS.secondaryOrder;
  } else {
    statsTableHost.style.order = FLAPPY_HOST_RESIZE_STYLE_TOKENS.primaryOrder;
    networkCanvasHost.style.order =
      FLAPPY_HOST_RESIZE_STYLE_TOKENS.secondaryOrder;
  }

  // Step 2: Collapse the stats table when the network-only compact panel is active.
  statsTableHost.style.display =
    responsiveViewportLayoutContext.useNetworkOnlyPanel
      ? FLAPPY_HOST_RESIZE_STYLE_TOKENS.hiddenDisplay
      : FLAPPY_HOST_RESIZE_STYLE_TOKENS.blockDisplay;
  statsTableHost.style.flex =
    responsiveViewportLayoutContext.useNetworkOnlyPanel
      ? FLAPPY_HOST_RESIZE_STYLE_TOKENS.autoFlex
      : FLAPPY_HOST_RESIZE_STYLE_TOKENS.fillFlex;
  networkCanvasHost.style.flex =
    responsiveViewportLayoutContext.useNetworkOnlyPanel
      ? FLAPPY_HOST_RESIZE_STYLE_TOKENS.fullFlex
      : FLAPPY_HOST_RESIZE_STYLE_TOKENS.autoFlex;
  networkCanvasHost.style.width =
    responsiveViewportLayoutContext.useNetworkOnlyPanel
      ? FLAPPY_HOST_RESIZE_STYLE_TOKENS.fullSize
      : FLAPPY_HOST_RESIZE_STYLE_TOKENS.autoSize;
  networkCanvasHost.style.maxWidth = FLAPPY_HOST_RESIZE_STYLE_TOKENS.fullSize;
}

/**
 * Applies the fixed network host height and queues redraw when it changes.
 *
 * @param networkCanvasHost - Network canvas host element.
 * @param deferredNetworkRedrawController - Deferred redraw controller.
 * @returns Nothing.
 */
function applyNetworkCanvasHostHeight(
  networkCanvasHost: HTMLElement,
  deferredNetworkRedrawController: DeferredNetworkRedrawController,
): void {
  // Step 1: Update the host height only when the fixed target changed.
  if (
    networkCanvasHost.offsetHeight === FLAPPY_UI_NETWORK_HOST_FIXED_HEIGHT_PX
  ) {
    return;
  }

  // Step 2: Apply the fixed host height and request a post-layout redraw.
  networkCanvasHost.style.height = `${FLAPPY_UI_NETWORK_HOST_FIXED_HEIGHT_PX}px`;
  deferredNetworkRedrawController.requestNetworkRedrawAfterLayout();
}

/**
 * Applies the resolved dimensions and scrolling rules to the stats container.
 *
 * @param statsContainer - Stats host element.
 * @param responsiveViewportLayoutContext - Responsive layout context.
 * @param statsPanelDimensions - Resolved stats panel dimensions.
 * @returns Nothing.
 */
function applyStatsContainerDimensions(
  statsContainer: HTMLElement,
  responsiveViewportLayoutContext: ResponsiveViewportLayoutContext,
  statsPanelDimensions: StatsPanelDimensions,
): void {
  // Step 1: Apply the resolved stats panel height.
  statsContainer.style.height = `${statsPanelDimensions.adjustedStatsPanelHeightPx}px`;
  statsContainer.style.maxHeight = `${statsPanelDimensions.adjustedStatsPanelHeightPx}px`;

  // Step 2: Apply width and scrolling rules that differ between landscape and stacked layouts.
  statsContainer.style.width =
    responsiveViewportLayoutContext.useLandscapeSplitLayout
      ? `${statsPanelDimensions.resolvedStatsPanelWidthPx}px`
      : FLAPPY_HOST_RESIZE_STYLE_TOKENS.fullSize;
  statsContainer.style.maxWidth =
    responsiveViewportLayoutContext.useLandscapeSplitLayout
      ? `${statsPanelDimensions.resolvedStatsPanelWidthPx}px`
      : FLAPPY_HOST_RESIZE_STYLE_TOKENS.fullSize;
  statsContainer.style.overflowY =
    responsiveViewportLayoutContext.useLandscapeSplitLayout
      ? FLAPPY_HOST_RESIZE_STYLE_TOKENS.autoOverflow
      : FLAPPY_HOST_RESIZE_STYLE_TOKENS.hiddenOverflow;
}
