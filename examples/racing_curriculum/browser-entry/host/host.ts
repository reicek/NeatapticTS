import type { RacingHostHandle } from './host.types';

/** Narrow viewport threshold used by the Tier 0 host layout seam. */
export const RACING_NARROW_VIEWPORT_THRESHOLD_PX = 960;

/**
 * Creates the ultra-thin racing browser shell.
 *
 * The host owns only stable DOM regions and viewport class toggling. Simulation
 * truth remains worker-owned in later steps.
 *
 * @param containerElement - Root element that receives the host tree.
 * @returns Stable handles for the Tier 0 browser shell.
 */
export function createRacingHost(
  containerElement: HTMLElement,
): RacingHostHandle {
  containerElement.replaceChildren();

  const rootElement = document.createElement('div');
  rootElement.className = 'racing-host';
  rootElement.dataset.racingLayout = 'wide';

  const canvasRegionElement = document.createElement('div');
  canvasRegionElement.className =
    'racing-host__region racing-host__region--canvas';
  canvasRegionElement.dataset.racingRegion = 'canvas-left';

  const networkRegionElement = document.createElement('div');
  networkRegionElement.className =
    'racing-host__region racing-host__region--network';
  networkRegionElement.dataset.racingRegion = 'network-right';

  const visualizerRegionElement = document.createElement('div');
  visualizerRegionElement.className =
    'racing-host__region racing-host__region--visualizer';
  visualizerRegionElement.dataset.racingRegion = 'visualizer-bottom';

  const canvasElement = document.createElement('canvas');
  canvasElement.width = 960;
  canvasElement.height = 540;
  canvasElement.setAttribute('aria-label', 'Racing curriculum playback canvas');
  canvasElement.setAttribute('role', 'img');

  const networkPlaceholderElement = document.createElement('div');
  networkPlaceholderElement.textContent = 'Network view panel';
  networkPlaceholderElement.className = 'racing-host__placeholder';

  canvasRegionElement.append(canvasElement);
  networkRegionElement.append(networkPlaceholderElement);
  rootElement.append(
    canvasRegionElement,
    networkRegionElement,
    visualizerRegionElement,
  );
  containerElement.append(rootElement);

  const applyViewportLayout = (viewportWidthPx: number): void => {
    const isNarrowViewport =
      viewportWidthPx < RACING_NARROW_VIEWPORT_THRESHOLD_PX;

    rootElement.classList.toggle('racing-host--narrow', isNarrowViewport);
    rootElement.dataset.racingLayout = isNarrowViewport ? 'narrow' : 'wide';
  };

  applyViewportLayout(window.innerWidth);

  return {
    rootElement,
    canvasElement,
    canvasRegionElement,
    networkRegionElement,
    visualizerRegionElement,
    applyViewportLayout,
  };
}
