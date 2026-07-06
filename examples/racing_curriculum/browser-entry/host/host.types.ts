/**
 * Stable DOM handles returned by the racing browser host.
 *
 * The Tier 0 host is intentionally narrow: a canvas region on the left and a
 * dedicated network-visualizer sidebar on the right. Runtime controls live below
 * the track inside the canvas region. The returned `resizeRedrawController`
 * keeps the right-sidebar network canvas in sync with the viewport.
 */

import type Network from '../../../../src/architecture/network';

/**
 * Live text nodes owned by the right-sidebar network HUD strip.
 *
 * The browser shell keeps these nodes up to date each frame so the panel header
 * shows generation/run context, the active network size, the last adaptation
 * change, and a short status line without repainting the canvas.
 */
export interface RacingNetworkHudNodes {
  /** Run/generation label (e.g., "Run 1" or "Live"). */
  titleValue: Text;
  /** Compact network size readout (e.g., "N72 / C120"). */
  sizeValue: Text;
  /** Short reason for the last controller adaptation or tuning change. */
  lastChangeValue: Text;
  /** One-word status line (e.g., "ADAPTING", "STABLE", "HOLD"). */
  statusValue: Text;
}

export interface RacingHostHandle {
  canvasElement: HTMLCanvasElement;
  canvasRegionElement: HTMLDivElement;
  networkRegionElement: HTMLDivElement;
  rootElement: HTMLDivElement;
  applyViewportLayout: (viewportWidthPx: number) => void;
  /** Dedicated host element for the right-sidebar network visualization. */
  networkCanvasHost: HTMLDivElement;
  /** Canvas element used to render the focused controller network. */
  networkCanvas: HTMLCanvasElement;
  /** 2D rendering context for {@link networkCanvas}. */
  networkContext: CanvasRenderingContext2D;
  /** Live text nodes in the neon HUD strip above the network canvas. */
  networkHud: RacingNetworkHudNodes;
  /** Static help/instruction chip strip shown beneath the network canvas. */
  networkHelpStrip: HTMLDivElement;
  /** Resize/redraw controller that keeps the network canvas in sync with the viewport. */
  resizeRedrawController: {
    uninstall: () => void;
    resize: () => void;
    redraw: () => void;
  };
  /** Renders the focused controller network architecture into the right sidebar. */
  renderNetworkArchitecture: (network: Network) => void;
}
