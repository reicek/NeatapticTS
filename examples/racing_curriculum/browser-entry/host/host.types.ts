/**
 * Stable DOM handles returned by the racing browser host.
 *
 * The Tier 0 host is intentionally narrow: one canvas region, one network region,
 * and one visualizer region in a fixed order that later runtime layers can
 * populate without changing the shell contract.
 */
export interface RacingHostHandle {
  canvasElement: HTMLCanvasElement;
  canvasRegionElement: HTMLDivElement;
  networkRegionElement: HTMLDivElement;
  rootElement: HTMLDivElement;
  visualizerRegionElement: HTMLDivElement;
  applyViewportLayout: (viewportWidthPx: number) => void;
}
