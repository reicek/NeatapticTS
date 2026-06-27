import type Network from '../../../../src/architecture/network';
import { FLAPPY_NEON_PALETTE } from '../../../flappy_bird/constants/constants.palette';
import { FLAPPY_MONOSPACE_FONT_FAMILY } from '../../../flappy_bird/constants/constants.frame';
import type { RacingHostHandle, RacingNetworkHudNodes } from './host.types';
import {
  FLAPPY_UI_NETWORK_CANVAS_BACKGROUND,
  FLAPPY_UI_NETWORK_HOST_BACKGROUND,
} from '../../../flappy_bird/constants/constants.layout';
import { installRacingNetworkResize } from './host.resize.service';
import {
  drawRacingNetworkVisualizationFromFrame,
  resolveRacingNetworkVisualizationFrame,
  type NetworkVisualizationResolvedFrame,
} from '../network-view/network-view';
import {
  resolveRacingNetworkTooltipScene,
  type NetworkVisualizationPositionedScene,
} from './host.network-tooltip.service';

/** Narrow viewport threshold used by the Tier 0 host layout seam. */
export const RACING_NARROW_VIEWPORT_THRESHOLD_PX = 960;

/**
 * Creates the ultra-thin racing browser shell.
 *
 * The host owns only stable DOM regions and viewport class toggling. Simulation
 * truth stays off the main thread; this file is responsible for presenting the
 * regions the worker-owned simulation and controller inference will later drive.
 *
 * Tier 0 layout mirrors the Flappy Bird parity shape: an outer frame containing a
 * left canvas region and a right sidebar network-visualizer region. Runtime
 * controls live below the track inside the canvas region; there is no
 * visualizer-bottom region.
 *
 * @param containerElement - Root element that receives the host tree.
 * @returns Stable handles for the Tier 0 browser shell.
 *
 * @example
 * ```ts
 * const host = createRacingHost(document.getElementById('racing-output')!);
 * host.applyViewportLayout(window.innerWidth);
 * ```
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

  const canvasElement = document.createElement('canvas');
  canvasElement.width = 960;
  canvasElement.height = 540;
  canvasElement.setAttribute('aria-label', 'Racing curriculum playback canvas');
  canvasElement.setAttribute('role', 'img');

  const networkCanvasHost = document.createElement('div');
  networkCanvasHost.className = 'racing-network-canvas-host';
  networkCanvasHost.style.width = '100%';
  networkCanvasHost.style.height = '100%';
  networkCanvasHost.style.boxSizing = 'border-box';
  networkCanvasHost.style.background = FLAPPY_UI_NETWORK_HOST_BACKGROUND;
  networkCanvasHost.style.position = 'relative';
  networkCanvasHost.style.overflow = 'hidden';
  networkCanvasHost.style.display = 'flex';
  networkCanvasHost.style.flexDirection = 'column';

  const networkHud = createRacingNetworkHud();
  const networkCanvasWrapper = document.createElement('div');
  networkCanvasWrapper.className = 'racing-network-canvas-wrapper';
  networkCanvasWrapper.style.position = 'relative';
  networkCanvasWrapper.style.flex = '1';
  networkCanvasWrapper.style.minHeight = '0';
  networkCanvasWrapper.style.overflow = 'hidden';
  networkCanvasWrapper.style.display = 'flex';
  networkCanvasWrapper.style.flexDirection = 'column';

  const networkCanvas = document.createElement('canvas');
  networkCanvas.className = 'racing-network-canvas';
  networkCanvas.width = 1;
  networkCanvas.height = 1;
  networkCanvas.setAttribute('aria-label', 'Focused controller network');
  networkCanvas.setAttribute('role', 'img');
  networkCanvas.style.width = '100%';
  networkCanvas.style.height = '100%';
  networkCanvas.style.display = 'block';
  networkCanvas.style.background = FLAPPY_UI_NETWORK_CANVAS_BACKGROUND;
  networkCanvas.style.cursor = 'crosshair';
  networkCanvas.style.border = 'none';
  networkCanvas.style.flex = '1';
  networkCanvas.style.minHeight = '0';

  let networkContext = networkCanvas.getContext('2d', {
    desynchronized: true,
  });
  if (networkContext === null) {
    // jsdom does not implement canvas 2D contexts; provide a minimal stub so
    // the host handle can still be exercised in browserless tests.
    networkContext = {
      canvas: networkCanvas,
    } as unknown as CanvasRenderingContext2D;
  }

  const networkHelpStrip = createRacingNetworkHelpStrip();

  canvasRegionElement.append(canvasElement);
  networkCanvasWrapper.append(networkCanvas, networkHelpStrip);
  networkCanvasHost.append(networkHud.hudElement, networkCanvasWrapper);
  networkRegionElement.append(networkCanvasHost);
  rootElement.append(canvasRegionElement, networkRegionElement);
  containerElement.append(rootElement);

  // Placeholder avoids a temporal dead-zone during the initial sizing pass, which
  // fires the resize callback before installRacingNetworkResize returns.
  const resizeRedrawController: {
    uninstall: () => void;
    resize: () => void;
    redraw: () => void;
  } = {
    uninstall: () => {},
    resize: () => {},
    redraw: () => {},
  };

  const hostNetworkVisualizationController =
    createHostNetworkVisualizationController(
      networkCanvasWrapper,
      networkCanvas,
      networkContext,
      resizeRedrawController,
    );

  const installedResizeController = installRacingNetworkResize(
    networkCanvas,
    networkCanvasWrapper,
    () => {
      resizeRedrawController.redraw();
    },
  );
  resizeRedrawController.uninstall = installedResizeController.uninstall;
  resizeRedrawController.resize = installedResizeController.resize;
  resizeRedrawController.redraw =
    hostNetworkVisualizationController.redrawCurrentNetworkArchitecture;
  resizeRedrawController.resize();

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
    networkCanvasHost,
    networkCanvas,
    networkContext,
    networkHud: networkHud.nodes,
    networkHelpStrip,
    resizeRedrawController,
    applyViewportLayout,
    renderNetworkArchitecture:
      hostNetworkVisualizationController.renderNetworkArchitecture,
  };
}

/** Internal hover/cache state kept by the host-side network visualization controller. */
type HostNetworkVisualizationState = {
  currentNetwork: Network | undefined;
  latestResolvedFrame: NetworkVisualizationResolvedFrame | undefined;
  latestPositionedScene: NetworkVisualizationPositionedScene | undefined;
  hoveredNodeIndices: number[] | undefined;
  lastPointerClientPosition: { clientX: number; clientY: number } | undefined;
  /** Cheap numeric fingerprint of the last resolved network state (weights/biases/activations). */
  networkStateFingerprint: number | undefined;
};

/** Tooltip DOM elements owned by the host network visualization controller. */
type HostNetworkVisualizationTooltipElements = {
  tooltipElement: HTMLDivElement;
  tooltipHeadingElement: HTMLDivElement;
  tooltipBodyElement: HTMLDivElement;
};

/** Controller handle returned by {@link createHostNetworkVisualizationController}. */
type HostNetworkVisualizationController = {
  renderNetworkArchitecture: (network: Network) => void;
  redrawCurrentNetworkArchitecture: () => void;
};

/**
 * Creates the host-owned network visualization controller.
 *
 * This controller keeps hover state, a resolved-frame cache, and a real redraw
 * function for the right-sidebar network panel. It mirrors the Flappy Bird
 * controller shape while staying narrow enough for the racing host.
 *
 * @param networkCanvasHost - Host element that sizes the network canvas.
 * @param networkCanvas - Network visualization canvas.
 * @param networkContext - 2D rendering context for the network canvas.
 * @param resizeRedrawController - Host resize/redraw controller used for sizing.
 * @returns Render and redraw callbacks for the network panel.
 */
function createHostNetworkVisualizationController(
  networkCanvasHost: HTMLDivElement,
  networkCanvas: HTMLCanvasElement,
  networkContext: CanvasRenderingContext2D,
  resizeRedrawController: { resize: () => void },
): HostNetworkVisualizationController {
  const tooltipElements = createRacingNetworkTooltipElements();
  networkCanvasHost.appendChild(tooltipElements.tooltipElement);

  const state: HostNetworkVisualizationState = {
    currentNetwork: undefined,
    latestResolvedFrame: undefined,
    latestPositionedScene: undefined,
    hoveredNodeIndices: undefined,
    lastPointerClientPosition: undefined,
    networkStateFingerprint: undefined,
  };

  const isNetworkPanelVisible = (): boolean => {
    const ownerDocument = networkCanvasHost.ownerDocument;
    if (ownerDocument?.hidden) {
      return false;
    }

    const defaultView = ownerDocument?.defaultView;
    if (!defaultView) {
      return true;
    }

    let ancestor: HTMLElement | null = networkCanvasHost;
    while (ancestor) {
      const style = defaultView.getComputedStyle(ancestor);
      if (style.display === 'none' || style.visibility === 'hidden') {
        return false;
      }
      ancestor = ancestor.parentElement;
    }

    return true;
  };

  const computeNetworkStateFingerprint = (network: Network): number => {
    let fingerprint = 0;

    for (const node of network.nodes) {
      fingerprint += node.bias + node.activation + node.state;
    }

    for (const connection of network.connections) {
      fingerprint += connection.weight;
    }

    return fingerprint;
  };

  const resolveLatestNetworkVisualizationFrame = ():
    NetworkVisualizationResolvedFrame | undefined => {
    const currentFingerprint = state.currentNetwork
      ? computeNetworkStateFingerprint(state.currentNetwork)
      : undefined;

    if (
      state.latestResolvedFrame &&
      state.networkStateFingerprint === currentFingerprint &&
      state.latestResolvedFrame.canvasWidthPx === networkCanvas.width &&
      state.latestResolvedFrame.canvasHeightPx === networkCanvas.height &&
      state.latestResolvedFrame.positionedScene.positionedNodes.length > 0
    ) {
      return state.latestResolvedFrame;
    }

    if (!state.currentNetwork) {
      return undefined;
    }

    const latestResolvedFrame = resolveRacingNetworkVisualizationFrame(
      networkContext,
      state.currentNetwork,
    );
    state.latestResolvedFrame = latestResolvedFrame;
    state.networkStateFingerprint = currentFingerprint;
    return latestResolvedFrame;
  };

  const syncTooltip = (
    clientX: number | undefined,
    clientY: number | undefined,
    positionedScene: NetworkVisualizationPositionedScene | undefined,
  ): void => {
    if (
      typeof clientX !== 'number' ||
      typeof clientY !== 'number' ||
      !positionedScene
    ) {
      tooltipElements.tooltipElement.style.opacity = '0';
      tooltipElements.tooltipElement.style.visibility = 'hidden';
      return;
    }

    const bounds = networkCanvas.getBoundingClientRect();
    if (
      clientX < bounds.left ||
      clientX > bounds.right ||
      clientY < bounds.top ||
      clientY > bounds.bottom
    ) {
      tooltipElements.tooltipElement.style.opacity = '0';
      tooltipElements.tooltipElement.style.visibility = 'hidden';
      return;
    }

    const scaleX = networkCanvas.width / Math.max(1, bounds.width);
    const scaleY = networkCanvas.height / Math.max(1, bounds.height);
    const canvasPoint = {
      xPx: (clientX - bounds.left) * scaleX,
      yPx: (clientY - bounds.top) * scaleY,
    };
    const scene = resolveRacingNetworkTooltipScene(
      canvasPoint,
      positionedScene,
    );

    if (!scene) {
      tooltipElements.tooltipElement.style.opacity = '0';
      tooltipElements.tooltipElement.style.visibility = 'hidden';
      return;
    }

    tooltipElements.tooltipHeadingElement.textContent = scene.heading;
    tooltipElements.tooltipBodyElement.replaceChildren(
      ...scene.bodyParagraphs.map((paragraph) => {
        const paragraphElement = document.createElement('p');
        paragraphElement.style.margin = '0';
        paragraphElement.style.whiteSpace = 'normal';
        paragraphElement.textContent = paragraph;
        return paragraphElement;
      }),
    );
    tooltipElements.tooltipElement.style.left = `${Math.min(
      Math.max(
        scene.anchorCenterXPx *
          (networkCanvas.clientWidth / Math.max(1, networkCanvas.width)),
        8,
      ),
      Math.max(
        0,
        networkCanvasHost.clientWidth -
          tooltipElements.tooltipElement.offsetWidth -
          8,
      ),
    )}px`;
    tooltipElements.tooltipElement.style.top = `${Math.max(
      8,
      scene.anchorTopPx *
        (networkCanvas.clientHeight / Math.max(1, networkCanvas.height)) -
        tooltipElements.tooltipElement.offsetHeight -
        10,
    )}px`;
    tooltipElements.tooltipElement.style.opacity = '1';
    tooltipElements.tooltipElement.style.visibility = 'visible';
  };

  const resolveHoveredNodeIndicesFromCanvasPoint = (
    canvasPoint: { xPx: number; yPx: number },
    positionedScene: NetworkVisualizationPositionedScene | undefined,
  ): number[] | undefined => {
    if (!positionedScene) {
      return undefined;
    }

    const halfNodeWidthPx = positionedScene.nodeDimensions.widthPx * 0.5;
    const halfNodeHeightPx = positionedScene.nodeDimensions.heightPx * 0.5;

    const hoveredNode = positionedScene.positionedNodes.findLast(
      (positionedNode) => {
        const nodeLeftPx = positionedNode.xPx - halfNodeWidthPx;
        const nodeRightPx = positionedNode.xPx + halfNodeWidthPx;
        const nodeTopPx = positionedNode.yPx - halfNodeHeightPx;
        const nodeBottomPx = positionedNode.yPx + halfNodeHeightPx;

        return (
          canvasPoint.xPx >= nodeLeftPx &&
          canvasPoint.xPx <= nodeRightPx &&
          canvasPoint.yPx >= nodeTopPx &&
          canvasPoint.yPx <= nodeBottomPx
        );
      },
    );

    return hoveredNode ? [hoveredNode.node.index] : undefined;
  };

  const drawCurrentNetworkVisualization = (): void => {
    if (!isNetworkPanelVisible()) {
      return;
    }

    const latestResolvedFrame = resolveLatestNetworkVisualizationFrame();
    if (!latestResolvedFrame) {
      return;
    }

    state.latestPositionedScene = drawRacingNetworkVisualizationFromFrame(
      networkContext,
      latestResolvedFrame,
      state.hoveredNodeIndices,
    );

    syncTooltip(
      state.lastPointerClientPosition?.clientX,
      state.lastPointerClientPosition?.clientY,
      state.latestPositionedScene,
    );
  };

  const updateHoveredNodeIndices = (
    nextHoveredNodeIndices: number[] | undefined,
  ): boolean => {
    if (
      arraysEqual(nextHoveredNodeIndices ?? [], state.hoveredNodeIndices ?? [])
    ) {
      return false;
    }

    state.hoveredNodeIndices = nextHoveredNodeIndices;
    return true;
  };

  const handleNetworkCanvasPointerMove = (event: PointerEvent): void => {
    state.lastPointerClientPosition = {
      clientX: event.clientX,
      clientY: event.clientY,
    };

    syncTooltip(event.clientX, event.clientY, state.latestPositionedScene);

    const bounds = networkCanvas.getBoundingClientRect();
    const scaleX = networkCanvas.width / Math.max(1, bounds.width);
    const scaleY = networkCanvas.height / Math.max(1, bounds.height);
    const canvasPoint = {
      xPx: (event.clientX - bounds.left) * scaleX,
      yPx: (event.clientY - bounds.top) * scaleY,
    };
    const resolvedHoveredNodeIndices = resolveHoveredNodeIndicesFromCanvasPoint(
      canvasPoint,
      state.latestPositionedScene,
    );

    if (!updateHoveredNodeIndices(resolvedHoveredNodeIndices)) {
      return;
    }

    drawCurrentNetworkVisualization();
  };

  const handleNetworkCanvasPointerLeave = (): void => {
    state.lastPointerClientPosition = undefined;
    syncTooltip(undefined, undefined, undefined);

    if (!state.hoveredNodeIndices?.length) {
      return;
    }

    updateHoveredNodeIndices(undefined);
    drawCurrentNetworkVisualization();
  };

  networkCanvas.addEventListener('pointermove', handleNetworkCanvasPointerMove);
  networkCanvas.addEventListener(
    'pointerleave',
    handleNetworkCanvasPointerLeave,
  );

  const renderNetworkArchitecture: HostNetworkVisualizationController['renderNetworkArchitecture'] =
    (network) => {
      state.currentNetwork = network;
      resizeRedrawController.resize();
      if (isNetworkPanelVisible()) {
        drawCurrentNetworkVisualization();
      }
    };

  const redrawCurrentNetworkArchitecture: HostNetworkVisualizationController['redrawCurrentNetworkArchitecture'] =
    () => {
      drawCurrentNetworkVisualization();
    };

  return {
    renderNetworkArchitecture,
    redrawCurrentNetworkArchitecture,
  };
}

function createRacingNetworkTooltipElements(): HostNetworkVisualizationTooltipElements {
  const tooltipElement = document.createElement('div');
  tooltipElement.className = 'racing-network-tooltip';
  tooltipElement.setAttribute('role', 'tooltip');
  tooltipElement.style.position = 'absolute';
  tooltipElement.style.left = '0';
  tooltipElement.style.top = '0';
  tooltipElement.style.opacity = '0';
  tooltipElement.style.visibility = 'hidden';
  tooltipElement.style.pointerEvents = 'none';
  tooltipElement.style.padding = '10px 12px';
  tooltipElement.style.border = `1px solid ${FLAPPY_NEON_PALETTE.hudPanelBorder}`;
  tooltipElement.style.borderRadius = '12px';
  tooltipElement.style.background = 'rgba(0, 21, 34, 0.98)';
  tooltipElement.style.color = FLAPPY_NEON_PALETTE.hudText;
  tooltipElement.style.fontFamily = FLAPPY_MONOSPACE_FONT_FAMILY;
  tooltipElement.style.zIndex = '8';

  const tooltipHeadingElement = document.createElement('div');
  tooltipHeadingElement.style.fontWeight = '700';
  tooltipHeadingElement.style.marginBottom = '6px';
  tooltipHeadingElement.style.color = FLAPPY_NEON_PALETTE.hudAccent;
  tooltipElement.appendChild(tooltipHeadingElement);

  const tooltipBodyElement = document.createElement('div');
  tooltipElement.appendChild(tooltipBodyElement);

  return {
    tooltipElement,
    tooltipHeadingElement,
    tooltipBodyElement,
  };
}

function arraysEqual(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) {
    return false;
  }

  for (let index = 0; index < a.length; index++) {
    if (a[index] !== b[index]) {
      return false;
    }
  }

  return true;
}

/**
 * Creates the neon top HUD strip for the right-sidebar network panel.
 *
 * The strip shows run context, network size, the last adaptation reason, and a
 * short status line. It lives above the network canvas and is updated through
 * the live text nodes returned here.
 *
 * @returns HUD element and mutable live text nodes.
 */
function createRacingNetworkHud(): {
  hudElement: HTMLDivElement;
  nodes: RacingNetworkHudNodes;
} {
  const titleValue = document.createTextNode('LIVE');
  const sizeValue = document.createTextNode('N0 / C0');
  const lastChangeValue = document.createTextNode('startup baseline');
  const statusValue = document.createTextNode('READY');

  const hudElement = document.createElement('div');
  hudElement.className = 'racing-network-hud';

  const titleCell = createHudCell('Run', titleValue, 'racing-network-hud__run');
  const sizeCell = createHudCell('Network', sizeValue);
  const lastChangeCell = createHudCell('Last Change', lastChangeValue);
  const statusCell = createHudCell(
    'Status',
    statusValue,
    'racing-network-hud__status',
  );

  hudElement.append(titleCell, sizeCell, lastChangeCell, statusCell);

  return {
    hudElement,
    nodes: {
      titleValue,
      sizeValue,
      lastChangeValue,
      statusValue,
    },
  };
}

/**
 * Creates one labeled HUD cell with a live-updating value.
 *
 * @param label - Small uppercase label for the cell.
 * @param valueNode - Live text node whose content is mutated each frame.
 * @param className - Optional extra CSS class for the value span.
 * @returns Completed HUD cell element.
 */
function createHudCell(
  label: string,
  valueNode: Text,
  className?: string,
): HTMLDivElement {
  const cellElement = document.createElement('div');
  cellElement.className = 'racing-network-hud__cell';

  const labelElement = document.createElement('div');
  labelElement.className = 'racing-network-hud__label';
  labelElement.textContent = label;

  const valueElement = document.createElement('div');
  valueElement.className = className
    ? `racing-network-hud__value ${className}`
    : 'racing-network-hud__value';
  valueElement.append(valueNode);

  cellElement.append(labelElement, valueElement);
  return cellElement;
}

/**
 * Creates the static help/instruction chip strip beneath the network canvas.
 *
 * These chips give first-time viewers concise cues about how to read the panel:
 * how to inspect nodes, what connection thickness means, and how the neon color
 * ramp encodes weight sign and magnitude.
 *
 * @returns Help strip element.
 */
function createRacingNetworkHelpStrip(): HTMLDivElement {
  const stripElement = document.createElement('div');
  stripElement.className = 'racing-network-help-strip';

  const chips = [
    { emoji: '👆', text: 'Hover nodes to inspect weights & bias' },
    { emoji: '⚡', text: 'Thicker lines = stronger weights' },
    { emoji: '🎨', text: 'Neon color = sign & magnitude' },
  ];

  for (const chip of chips) {
    const chipElement = document.createElement('div');
    chipElement.className = 'racing-network-help-chip';

    const emojiElement = document.createElement('span');
    emojiElement.className = 'racing-network-help-chip__emoji';
    emojiElement.textContent = chip.emoji;

    const textElement = document.createElement('span');
    textElement.className = 'racing-network-help-chip__text';
    textElement.textContent = chip.text;

    chipElement.append(emojiElement, textElement);
    stripElement.append(chipElement);
  }

  return stripElement;
}
