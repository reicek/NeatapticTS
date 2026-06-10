/**
 * Stable DOM handles returned by the racing browser host.
 *
 * The Tier 0 host is intentionally narrow: canvas, network, and visualizer
 * regions in a fixed order.
 */
export interface RacingHostHandle {
  canvasElement: HTMLCanvasElement;
  canvasRegionElement: HTMLDivElement;
  networkRegionElement: HTMLDivElement;
  visualizerRegionElement: HTMLDivElement;
  rootElement: HTMLDivElement;
  applyViewportLayout: (viewportWidthPx: number) => void;
}
