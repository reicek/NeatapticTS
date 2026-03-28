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
 * Top-level responsive sizing orchestration for the browser host.
 *
 * The resize boundary keeps one educational promise intact: the browser demo
 * should still read like one coherent instrument panel even as the viewport
 * shifts from desktop to narrow mobile layouts. That means sizing is not just a
 * cosmetic concern. It decides whether the simulation canvas, HUD table, and
 * network panel remain readable together.
 *
 * This module owns that policy at the top level: measure the current host,
 * choose the appropriate layout mode, resize canvases, and schedule any network
 * redraws needed after the geometry changes.
 *
 * Layout decision flow:
 * ```mermaid
 * flowchart LR
 *     Measure["measure container"] --> Decide{"minimal mobile\nlayout?"}
 *     Decide -->|Yes| Mobile["apply minimal mobile layout"]
 *     Decide -->|No| Standard["apply standard split layout"]
 *     Standard --> Resize["resize simulation + network canvases"]
 * ```
 */

/**
 * Installs responsive viewport sizing for simulation and network canvases.
 *
 * This is the public entrypoint used by host assembly. It captures the elements,
 * creates the deferred redraw policy, runs the first layout pass, and installs
 * ongoing resize listeners.
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
 * The resize flow is intentionally split into two branches: the minimal mobile
 * path and the richer standard path used for tablet and desktop layouts.
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
