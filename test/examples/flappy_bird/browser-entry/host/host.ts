import type Network from '../../../../../src/architecture/network';
import {
  FLAPPY_FRAME_MONOSPACE_FONT,
  FLAPPY_HEADER_CANVAS_HEIGHT_PX,
  FLAPPY_HEADER_TITLE_TEXT,
  FLAPPY_NEON_PALETTE,
  FLAPPY_SCREEN_PADDING_PX,
  FLAPPY_UI_CANVAS_INSET_SHADOW,
  FLAPPY_UI_CONTENT_COLUMN_TOP_PADDING_PX,
  FLAPPY_UI_DOUBLE_PANEL_BORDER,
  FLAPPY_UI_NETWORK_CANVAS_BACKGROUND,
  FLAPPY_UI_NETWORK_HOST_BACKGROUND,
  FLAPPY_UI_NETWORK_HOST_FIXED_HEIGHT_PX,
  FLAPPY_UI_NETWORK_HOST_INITIAL_HEIGHT_PX,
  FLAPPY_UI_NETWORK_HOST_INSET_PX,
  FLAPPY_UI_OUTER_FRAME_BACKGROUND,
  FLAPPY_UI_OUTER_FRAME_MIN_SIDE_PADDING_PX,
  FLAPPY_UI_OUTER_FRAME_SIDE_PADDING_OFFSET_PX,
  FLAPPY_UI_UNIFIED_INSET_SHADOW,
  FLAPPY_VIEWPORT_VERTICAL_LAYOUT_GUTTER_PX,
  FLAPPY_NETWORK_INPUT_SIZE,
  FLAPPY_NETWORK_OUTPUT_SIZE,
} from '../../constants/constants';
import type {
  FlappyStatsTableCells,
  NetworkVisualizationHandle,
} from '../browser-entry.types';
import { renderStandaloneTitleBox } from '../browser-entry.text-frame.utils';
import {
  drawNetworkVisualization,
  resolveNetworkVisualizationHeightPx,
} from '../network-view/network-view';
import {
  applyCanvasBackingSize,
  resolveNetworkCanvasSizePx,
} from './host.canvas.service';
import {
  FLAPPY_HOST_PANEL_PADDING,
  FLAPPY_HOST_PANEL_TRANSITION,
  FLAPPY_HOST_STATS_SPLIT_GAP,
  FLAPPY_HOST_TABLE_HOST_PADDING,
} from './host.constants';
import { resolveRequiredCanvas2dContext } from './host.dom.service';
import { installResponsiveViewportSizing } from './resize/host.resize.service';
import {
  createAndAttachHostStatsTable,
  updateStatsTableValues as updateHostStatsTableValues,
} from './host.stats.service';
import type { CanvasHostResult, HostStatsPartialValues } from './host.types';

/**
 * Browser host assembly for the Flappy Bird demo UI.
 *
 * The host boundary is responsible for building the browser-side shell around
 * the simulation: framed title, main canvas, stats panel, and network
 * visualization panel. It does not run evolution itself; it prepares the stage
 * on which the runtime loop renders.
 */

type HostVisualPrimitives = {
  unifiedBorder: string;
  unifiedInsetShadow: string;
  sidePaddingPx: number;
};

type HostLayoutElements = {
  outerFrame: HTMLElement;
  contentColumn: HTMLDivElement;
  mainSplitContainer: HTMLDivElement;
  statsContainer: HTMLDivElement;
  statsSplitContainer: HTMLDivElement;
  statsTableHost: HTMLDivElement;
  networkCanvasHost: HTMLDivElement;
};

type HostCanvasElements = {
  headerCanvas: HTMLCanvasElement;
  headerContext: CanvasRenderingContext2D;
  canvas: HTMLCanvasElement;
  context: CanvasRenderingContext2D;
  networkCanvas: HTMLCanvasElement;
  networkContext: CanvasRenderingContext2D;
};

type HostNetworkVisualizationState = {
  previousNetworkForVisualization: Network | undefined;
  previousVisualizationInputSize: number;
  previousVisualizationOutputSize: number;
};

type HostNetworkVisualizationController = {
  renderNetworkArchitecture: NetworkVisualizationHandle['renderNetworkArchitecture'];
  resizeNetworkCanvasToHost: () => void;
  redrawCurrentNetworkArchitecture: () => void;
};

/**
 * Builds the browser demo host tree and returns rendering handles.
 *
 * This is the public host entrypoint used by the runtime startup path.
 *
 * @param containerElement - Root host container.
 * @returns Canvas handles, stats cells and network render callback.
 */
export function createCanvasHost(
  containerElement: HTMLElement,
): CanvasHostResult {
  return createCanvasHostInternal(containerElement);
}

/**
 * Builds the browser demo host tree and returns rendering handles.
 *
 * The orchestration is deliberately step-shaped: clear old DOM, build layout,
 * create canvases, wire resize behavior, render placeholders, then return the
 * handles the runtime will mutate during execution.
 *
 * @param containerElement - Root host container.
 * @returns Canvas handles, stats cells and network render callback.
 */
export function createCanvasHostInternal(
  containerElement: HTMLElement,
): CanvasHostResult {
  // Step 1: Reset the host container and resolve shared visual primitives.
  resetHostContainer(containerElement);
  const hostVisualPrimitives = resolveHostVisualPrimitives();

  // Step 2: Create layout elements, canvases, and stats table state.
  const hostLayoutElements = createHostLayoutElements(hostVisualPrimitives);
  const hostCanvasElements = createHostCanvasElements(hostVisualPrimitives);
  const statsValueByKey = createAndAttachHostStatsTable(
    hostLayoutElements.statsTableHost,
  );

  // Step 3: Create reusable header and network visualization controllers.
  const drawHeaderFrame = createHeaderFrameRenderer(
    hostCanvasElements.headerCanvas,
    hostCanvasElements.headerContext,
  );
  const hostNetworkVisualizationController =
    createHostNetworkVisualizationController(
      hostLayoutElements.networkCanvasHost,
      hostCanvasElements.networkCanvas,
      hostCanvasElements.networkContext,
    );

  // Step 4: Mount the host DOM tree in final order.
  mountCanvasHostTree(
    containerElement,
    hostLayoutElements,
    hostCanvasElements.headerCanvas,
    hostCanvasElements.canvas,
    hostCanvasElements.networkCanvas,
  );

  // Step 5: Install resize hooks that keep the header and network view in sync.
  installCanvasHostResizeHooks(
    hostCanvasElements.canvas,
    hostLayoutElements,
    hostCanvasElements.networkCanvas,
    drawHeaderFrame,
    hostNetworkVisualizationController,
  );

  // Step 6: Render the initial header and placeholder network visualization.
  renderInitialCanvasHostState(
    drawHeaderFrame,
    hostNetworkVisualizationController.renderNetworkArchitecture,
  );

  // Step 7: Return the public host handles used by the runtime.
  return {
    canvas: hostCanvasElements.canvas,
    context: hostCanvasElements.context,
    statsValueByKey,
    renderNetworkArchitecture:
      hostNetworkVisualizationController.renderNetworkArchitecture,
  };
}

/**
 * Applies partial stat updates to the rendered stats table.
 *
 * The runtime writes HUD values incrementally, so the host exposes a narrow
 * partial-update helper rather than requiring full table redraws.
 *
 * @param statsValueByKey - Lookup of stat keys to value cells.
 * @param partialValues - Subset of values to write this tick.
 * @returns Nothing.
 */
export function updateStatsTableValues(
  statsValueByKey: FlappyStatsTableCells,
  partialValues: HostStatsPartialValues,
): void {
  updateHostStatsTableValues(statsValueByKey, partialValues);
}

export { updateStatsTableValues as updateStatsTableValuesInternal };

/**
 * Clears any previous runtime DOM before rebuilding the browser host tree.
 *
 * The demo rebuilds the host from scratch on each startup so repeated runs begin
 * from a known clean DOM state.
 *
 * @param containerElement - Root host container.
 * @returns Nothing.
 */
function resetHostContainer(containerElement: HTMLElement): void {
  // Step 1: Remove any prior child content before rebuilding the host tree.
  containerElement.innerHTML = '';
}

/**
 * Resolves shared border, shadow, and padding values for host assembly.
 *
 * Centralizing these primitives keeps the DOM-building code focused on layout
 * structure instead of duplicating presentation constants everywhere.
 *
 * @returns Shared visual primitives reused across host sections.
 */
function resolveHostVisualPrimitives(): HostVisualPrimitives {
  // Step 1: Resolve the responsive outer padding used by the host frame.
  const sidePaddingPx = Math.max(
    FLAPPY_UI_OUTER_FRAME_MIN_SIDE_PADDING_PX,
    FLAPPY_SCREEN_PADDING_PX - FLAPPY_UI_OUTER_FRAME_SIDE_PADDING_OFFSET_PX,
  );

  // Step 2: Return the shared border, shadow, and padding primitives.
  return {
    unifiedBorder: FLAPPY_UI_DOUBLE_PANEL_BORDER,
    unifiedInsetShadow: FLAPPY_UI_UNIFIED_INSET_SHADOW,
    sidePaddingPx,
  };
}

/**
 * Creates the host layout elements used to assemble the browser UI tree.
 *
 * This creates the structural DOM only. Canvases, stats content, and
 * visualization wiring are layered on afterward.
 *
 * @param hostVisualPrimitives - Shared visual primitives for border and shadow styling.
 * @returns Layout elements grouped by host responsibility.
 */
function createHostLayoutElements(
  hostVisualPrimitives: HostVisualPrimitives,
): HostLayoutElements {
  // Step 1: Create and style the outer frame shell.
  const outerFrame = document.createElement('section');
  outerFrame.style.width = '100%';
  outerFrame.style.height = '100%';
  outerFrame.style.boxSizing = 'border-box';
  outerFrame.style.padding = `0 ${hostVisualPrimitives.sidePaddingPx}px ${hostVisualPrimitives.sidePaddingPx}px ${hostVisualPrimitives.sidePaddingPx}px`;
  outerFrame.style.border = hostVisualPrimitives.unifiedBorder;
  outerFrame.style.background = FLAPPY_UI_OUTER_FRAME_BACKGROUND;
  outerFrame.style.boxShadow = hostVisualPrimitives.unifiedInsetShadow;
  outerFrame.style.position = 'relative';

  // Step 2: Create the top-level content and simulation split containers.
  const contentColumn = document.createElement('div');
  contentColumn.style.width = '100%';
  contentColumn.style.height = '100%';
  contentColumn.style.display = 'flex';
  contentColumn.style.flexDirection = 'column';
  contentColumn.style.gap = '0';
  contentColumn.style.alignItems = 'stretch';
  contentColumn.style.overflow = 'hidden';
  contentColumn.style.paddingTop = `${FLAPPY_UI_CONTENT_COLUMN_TOP_PADDING_PX}px`;

  const mainSplitContainer = document.createElement('div');
  mainSplitContainer.style.width = '100%';
  mainSplitContainer.style.flex = '1 1 0';
  mainSplitContainer.style.minHeight = '0';
  mainSplitContainer.style.display = 'flex';
  mainSplitContainer.style.flexDirection = 'column';
  mainSplitContainer.style.alignItems = 'stretch';
  mainSplitContainer.style.gap = `${FLAPPY_VIEWPORT_VERTICAL_LAYOUT_GUTTER_PX}px`;

  // Step 3: Create the stats panel and its inner split layout.
  const statsContainer = document.createElement('div');
  statsContainer.style.marginTop = '0';
  statsContainer.style.boxSizing = 'border-box';
  statsContainer.style.background = FLAPPY_NEON_PALETTE.hudPanelBackground;
  statsContainer.style.border = hostVisualPrimitives.unifiedBorder;
  statsContainer.style.boxShadow = hostVisualPrimitives.unifiedInsetShadow;
  statsContainer.style.padding = FLAPPY_HOST_PANEL_PADDING;
  statsContainer.style.overflow = 'hidden';
  statsContainer.style.transition = FLAPPY_HOST_PANEL_TRANSITION;
  statsContainer.style.minHeight = '0';

  const statsSplitContainer = document.createElement('div');
  statsSplitContainer.style.display = 'flex';
  statsSplitContainer.style.flexDirection = 'row';
  statsSplitContainer.style.alignItems = 'flex-start';
  statsSplitContainer.style.gap = FLAPPY_HOST_STATS_SPLIT_GAP;
  statsSplitContainer.style.width = '100%';
  statsSplitContainer.style.boxSizing = 'border-box';

  const statsTableHost = document.createElement('div');
  statsTableHost.style.flex = '1 1 0';
  statsTableHost.style.minHeight = '120px';
  statsTableHost.style.minWidth = '0';
  statsTableHost.style.overflow = 'hidden';
  statsTableHost.style.boxSizing = 'border-box';
  statsTableHost.style.border = hostVisualPrimitives.unifiedBorder;
  statsTableHost.style.padding = FLAPPY_HOST_TABLE_HOST_PADDING;

  // Step 4: Create the network panel host.
  const networkCanvasHost = document.createElement('div');
  networkCanvasHost.style.flex = '1 1 0';
  networkCanvasHost.style.minWidth = '0';
  networkCanvasHost.style.height = `${FLAPPY_UI_NETWORK_HOST_INITIAL_HEIGHT_PX}px`;
  networkCanvasHost.style.background = FLAPPY_UI_NETWORK_HOST_BACKGROUND;
  networkCanvasHost.style.boxSizing = 'border-box';
  networkCanvasHost.style.border = hostVisualPrimitives.unifiedBorder;
  networkCanvasHost.style.padding = `${FLAPPY_UI_NETWORK_HOST_INSET_PX / 2}px`;
  networkCanvasHost.style.boxShadow = FLAPPY_UI_CANVAS_INSET_SHADOW;

  return {
    outerFrame,
    contentColumn,
    mainSplitContainer,
    statsContainer,
    statsSplitContainer,
    statsTableHost,
    networkCanvasHost,
  };
}

/**
 * Creates the canvases and 2D contexts used by the host UI.
 *
 * The host manages three canvas surfaces with different jobs: a title/header
 * frame, the main simulation view, and the side-panel network visualization.
 *
 * @param hostVisualPrimitives - Shared visual primitives for border and shadow styling.
 * @returns Simulation, header, and network canvases with required contexts.
 */
function createHostCanvasElements(
  hostVisualPrimitives: HostVisualPrimitives,
): HostCanvasElements {
  // Step 1: Create the header canvas used by the framed title renderer.
  const headerCanvas = document.createElement('canvas');
  headerCanvas.width = 1;
  headerCanvas.height = 1;
  headerCanvas.style.width = '100%';
  headerCanvas.style.height = `${FLAPPY_HEADER_CANVAS_HEIGHT_PX}px`;
  headerCanvas.style.display = 'block';
  headerCanvas.style.maxWidth = '100%';
  headerCanvas.style.boxSizing = 'border-box';
  headerCanvas.style.background = 'transparent';
  const headerContext = resolveRequiredCanvas2dContext(
    headerCanvas,
    'Header canvas 2D context unavailable',
  );

  // Step 2: Create the main simulation canvas.
  const canvas = document.createElement('canvas');
  canvas.width = 1;
  canvas.height = 1;
  canvas.style.width = '100%';
  canvas.style.height = '1px';
  canvas.style.display = 'block';
  canvas.style.maxWidth = '100%';
  canvas.style.boxSizing = 'border-box';
  canvas.style.flex = '0 0 auto';
  canvas.style.alignSelf = 'flex-start';
  canvas.style.imageRendering = 'pixelated';
  canvas.style.background = FLAPPY_NEON_PALETTE.background;
  canvas.style.border = hostVisualPrimitives.unifiedBorder;
  canvas.style.boxShadow = FLAPPY_UI_CANVAS_INSET_SHADOW;
  const context = resolveRequiredCanvas2dContext(
    canvas,
    'Canvas 2D context unavailable',
  );

  // Step 3: Create the network visualization canvas.
  const networkCanvas = document.createElement('canvas');
  networkCanvas.width = 1;
  networkCanvas.height = 1;
  networkCanvas.style.display = 'block';
  networkCanvas.style.width = '100%';
  networkCanvas.style.height = '100%';
  networkCanvas.style.maxWidth = '100%';
  networkCanvas.style.background = FLAPPY_UI_NETWORK_CANVAS_BACKGROUND;
  networkCanvas.style.border = 'none';
  networkCanvas.style.boxSizing = 'border-box';
  const networkContext = resolveRequiredCanvas2dContext(
    networkCanvas,
    'Network canvas 2D context unavailable',
  );

  return {
    headerCanvas,
    headerContext,
    canvas,
    context,
    networkCanvas,
    networkContext,
  };
}

/**
 * Creates the reusable title-frame renderer for the header canvas.
 *
 * @param headerCanvas - Header canvas element.
 * @param headerContext - Header canvas 2D context.
 * @returns Callback that redraws the framed title.
 */
function createHeaderFrameRenderer(
  headerCanvas: HTMLCanvasElement,
  headerContext: CanvasRenderingContext2D,
): () => void {
  /**
   * Redraws the title box after syncing canvas backing size to display width.
   *
   * @returns Nothing.
   */
  return () => {
    // Step 1: Sync backing resolution to the current display width.
    const displayWidthPx = Math.max(1, Math.floor(headerCanvas.clientWidth));
    const displayHeightPx = FLAPPY_HEADER_CANVAS_HEIGHT_PX;
    if (
      headerCanvas.width !== displayWidthPx ||
      headerCanvas.height !== displayHeightPx
    ) {
      headerCanvas.width = displayWidthPx;
      headerCanvas.height = displayHeightPx;
    }

    // Step 2: Render the title box using the shared framed-text renderer.
    renderStandaloneTitleBox({
      context: headerContext,
      widthPx: displayWidthPx,
      heightPx: displayHeightPx,
      titleText: FLAPPY_HEADER_TITLE_TEXT,
      glyphColor: FLAPPY_NEON_PALETTE.hudPanelBorder,
      font: FLAPPY_FRAME_MONOSPACE_FONT,
    });
  };
}

/**
 * Creates the network visualization renderer and redraw controller.
 *
 * @param networkCanvasHost - Host element wrapping the network canvas.
 * @param networkCanvas - Network visualization canvas.
 * @param networkContext - Network visualization 2D context.
 * @returns Renderer and redraw callbacks for the network panel.
 */
function createHostNetworkVisualizationController(
  networkCanvasHost: HTMLDivElement,
  networkCanvas: HTMLCanvasElement,
  networkContext: CanvasRenderingContext2D,
): HostNetworkVisualizationController {
  const hostNetworkVisualizationState: HostNetworkVisualizationState = {
    previousNetworkForVisualization: undefined,
    previousVisualizationInputSize: FLAPPY_NETWORK_INPUT_SIZE,
    previousVisualizationOutputSize: FLAPPY_NETWORK_OUTPUT_SIZE,
  };

  /**
   * Resizes the network canvas backing store to match the host element.
   *
   * @returns Nothing.
   */
  const resizeNetworkCanvasToHost = (): void => {
    // Step 1: Resolve the drawable network canvas bounds from the host panel.
    const { widthPx, heightPx } = resolveNetworkCanvasSizePx(
      networkCanvasHost,
      FLAPPY_UI_NETWORK_HOST_INSET_PX,
    );

    // Step 2: Apply backing-store dimensions to the network canvas.
    applyCanvasBackingSize(networkCanvas, widthPx, heightPx);
  };

  const renderNetworkArchitecture: NetworkVisualizationHandle['renderNetworkArchitecture'] =
    (network, inputSize, outputSize) => {
      // Step 1: Resolve preferred visualization height and lock the panel size.
      resolveNetworkVisualizationHeightPx(network, inputSize, outputSize);
      const fixedNetworkPanelHeightPx = FLAPPY_UI_NETWORK_HOST_FIXED_HEIGHT_PX;
      networkCanvasHost.dataset.preferredHeightPx = String(
        fixedNetworkPanelHeightPx,
      );
      networkCanvasHost.style.height = `${fixedNetworkPanelHeightPx}px`;

      // Step 2: Resize the canvas and persist the last rendered payload.
      resizeNetworkCanvasToHost();
      hostNetworkVisualizationState.previousNetworkForVisualization = network;
      hostNetworkVisualizationState.previousVisualizationInputSize = inputSize;
      hostNetworkVisualizationState.previousVisualizationOutputSize =
        outputSize;

      // Step 3: Draw the current network visualization payload.
      drawNetworkVisualization(networkContext, network, inputSize, outputSize);
    };

  /**
   * Redraws the most recently rendered network visualization payload.
   *
   * @returns Nothing.
   */
  const redrawCurrentNetworkArchitecture = (): void => {
    // Step 1: Re-render the last known payload after external resize changes.
    renderNetworkArchitecture(
      hostNetworkVisualizationState.previousNetworkForVisualization,
      hostNetworkVisualizationState.previousVisualizationInputSize,
      hostNetworkVisualizationState.previousVisualizationOutputSize,
    );
  };

  return {
    renderNetworkArchitecture,
    resizeNetworkCanvasToHost,
    redrawCurrentNetworkArchitecture,
  };
}

/**
 * Mounts the completed host DOM tree into the container in final order.
 *
 * @param containerElement - Root host container.
 * @param hostLayoutElements - Prepared layout containers.
 * @param headerCanvas - Header title canvas.
 * @param canvas - Main simulation canvas.
 * @param networkCanvas - Network visualization canvas.
 * @returns Nothing.
 */
function mountCanvasHostTree(
  containerElement: HTMLElement,
  hostLayoutElements: HostLayoutElements,
  headerCanvas: HTMLCanvasElement,
  canvas: HTMLCanvasElement,
  networkCanvas: HTMLCanvasElement,
): void {
  // Step 1: Mount the stats section and network panel into the stats container.
  hostLayoutElements.networkCanvasHost.appendChild(networkCanvas);
  hostLayoutElements.statsSplitContainer.appendChild(
    hostLayoutElements.statsTableHost,
  );
  hostLayoutElements.statsSplitContainer.appendChild(
    hostLayoutElements.networkCanvasHost,
  );
  hostLayoutElements.statsContainer.appendChild(
    hostLayoutElements.statsSplitContainer,
  );

  // Step 2: Mount the header, simulation canvas, and stats area into the frame.
  hostLayoutElements.contentColumn.appendChild(headerCanvas);
  hostLayoutElements.mainSplitContainer.appendChild(canvas);
  hostLayoutElements.mainSplitContainer.appendChild(
    hostLayoutElements.statsContainer,
  );
  hostLayoutElements.contentColumn.appendChild(
    hostLayoutElements.mainSplitContainer,
  );
  hostLayoutElements.outerFrame.appendChild(hostLayoutElements.contentColumn);
  containerElement.appendChild(hostLayoutElements.outerFrame);
}

/**
 * Installs responsive resize hooks for the simulation canvas and side panel.
 *
 * @param canvas - Simulation canvas.
 * @param hostLayoutElements - Prepared layout containers.
 * @param networkCanvas - Network visualization canvas.
 * @param drawHeaderFrame - Callback that redraws the header title.
 * @param hostNetworkVisualizationController - Network panel resize/redraw controller.
 * @returns Nothing.
 */
function installCanvasHostResizeHooks(
  canvas: HTMLCanvasElement,
  hostLayoutElements: HostLayoutElements,
  networkCanvas: HTMLCanvasElement,
  drawHeaderFrame: () => void,
  hostNetworkVisualizationController: HostNetworkVisualizationController,
): void {
  // Step 1: Install responsive sizing and redraw hooks across host sections.
  installResponsiveViewportSizing(
    canvas,
    hostLayoutElements.contentColumn,
    hostLayoutElements.mainSplitContainer,
    hostLayoutElements.statsContainer,
    hostLayoutElements.statsSplitContainer,
    hostLayoutElements.statsTableHost,
    networkCanvas,
    hostLayoutElements.networkCanvasHost,
    () => {
      drawHeaderFrame();
      hostNetworkVisualizationController.resizeNetworkCanvasToHost();
      hostNetworkVisualizationController.redrawCurrentNetworkArchitecture();
    },
  );
}

/**
 * Renders the initial header and placeholder network visualization state.
 *
 * @param drawHeaderFrame - Callback that redraws the header title.
 * @param renderNetworkArchitecture - Network visualization renderer.
 * @returns Nothing.
 */
function renderInitialCanvasHostState(
  drawHeaderFrame: () => void,
  renderNetworkArchitecture: NetworkVisualizationHandle['renderNetworkArchitecture'],
): void {
  // Step 1: Paint the placeholder network visualization before simulation starts.
  renderNetworkArchitecture(
    undefined,
    FLAPPY_NETWORK_INPUT_SIZE,
    FLAPPY_NETWORK_OUTPUT_SIZE,
  );

  // Step 2: Render the framed header title.
  drawHeaderFrame();
}
