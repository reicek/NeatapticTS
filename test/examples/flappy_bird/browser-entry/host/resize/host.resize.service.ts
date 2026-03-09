import {
  applyMinimalMobileViewportLayout,
  applyResponsiveCanvasBounds,
  applyStandardViewportLayout,
  createDeferredNetworkRedrawController,
  installResponsiveViewportSizingListeners,
} from './host.resize.service.services';
import { resolveResponsiveViewportLayoutContext } from './host.resize.service.utils';
import type {
  DeferredNetworkRedrawController,
  ResponsiveViewportSizingElements,
} from './host.resize.service.types';

/**
 * Installs responsive viewport sizing for simulation and network canvases.
 *
 * @param canvas - Simulation canvas to resize.
 * @param containerElement - Width/height source.
 * @param mainSplitContainer - Main split panel host.
 * @param statsContainer - Stats host element.
 * @param statsSplitContainer - Stats split panel containing stats and network panes.
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
  // Step 1: Capture the DOM elements involved in responsive host sizing.
  const responsiveViewportSizingElements: ResponsiveViewportSizingElements = {
    canvas,
    containerElement,
    mainSplitContainer,
    statsContainer,
    statsSplitContainer,
    statsTableHost,
    networkCanvas,
    networkCanvasHost,
  };

  // Step 2: Create the deferred redraw controller used after layout-affecting changes.
  const deferredNetworkRedrawController =
    createDeferredNetworkRedrawController(onNetworkResize);

  // Step 3: Build the shared sizing callback used by initial load and resize events.
  const applyCanvasSize = (): void => {
    applyResponsiveViewportSizing(
      responsiveViewportSizingElements,
      deferredNetworkRedrawController,
      onNetworkResize,
    );
  };

  // Step 4: Run the initial sizing pass and install resize observers.
  applyCanvasSize();
  installResponsiveViewportSizingListeners(
    containerElement,
    applyCanvasSize,
    deferredNetworkRedrawController,
  );
}

/**
 * Applies responsive sizing to the simulation and network canvases.
 *
 * @param responsiveViewportSizingElements - Host elements participating in layout.
 * @param deferredNetworkRedrawController - Deferred redraw controller.
 * @param onNetworkResize - Immediate network resize callback.
 * @returns Nothing.
 */
function applyResponsiveViewportSizing(
  responsiveViewportSizingElements: ResponsiveViewportSizingElements,
  deferredNetworkRedrawController: DeferredNetworkRedrawController,
  onNetworkResize: () => void,
): void {
  // Step 1: Measure the current viewport constraints and layout mode flags.
  const responsiveViewportLayoutContext =
    resolveResponsiveViewportLayoutContext(
      responsiveViewportSizingElements.containerElement,
      responsiveViewportSizingElements.statsContainer,
      responsiveViewportSizingElements.networkCanvasHost,
    );

  // Step 2: Route the simplest mobile layout through the dedicated minimal-flow helper.
  if (responsiveViewportLayoutContext.useMinimalMobileLayout) {
    applyMinimalMobileViewportLayout(
      responsiveViewportSizingElements,
      responsiveViewportLayoutContext,
    );
    return;
  }

  // Step 3: Apply the desktop and tablet host layout and resolve stats panel dimensions.
  const statsPanelDimensions = applyStandardViewportLayout(
    responsiveViewportSizingElements,
    responsiveViewportLayoutContext,
    deferredNetworkRedrawController,
  );

  // Step 4: Resize the simulation and network canvases for the resolved layout.
  applyResponsiveCanvasBounds(
    responsiveViewportSizingElements,
    responsiveViewportLayoutContext,
    statsPanelDimensions,
    onNetworkResize,
  );
}
