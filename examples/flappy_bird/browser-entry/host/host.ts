/**
 * Browser host assembly for the Flappy Bird demo UI.
 *
 * The host boundary builds the stage that the rest of the browser runtime plays
 * on: title frame, simulation canvas, HUD table, and the network-inspection
 * panel. It is deliberately separate from runtime orchestration so DOM setup and
 * responsive layout stay understandable without also reading worker or playback
 * code.
 *
 * Read this module as a browser chapter about presentation ownership:
 * resolve the shell, mount the panels, keep the canvases sized correctly, then
 * hand the runtime narrow handles for drawing and HUD updates.
 */
import type Network from '../../../../src/architecture/network';
import {
  FLAPPY_FRAME_MONOSPACE_FONT,
  FLAPPY_HEADER_CANVAS_HEIGHT_PX,
  FLAPPY_HEADER_TITLE_TEXT,
  FLAPPY_MONOSPACE_FONT_FAMILY,
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
  NetworkVisualizationAnimatedHoveredNode,
  NetworkVisualizationHandle,
  NetworkVisualizationPositionedScene,
} from '../browser-entry.types';
import { clamp } from '../browser-entry.math.utils';
import { renderStandaloneTitleBox } from '../browser-entry.text-frame.utils';
import {
  drawResolvedNetworkVisualization,
  resolveNetworkVisualizationFrame,
  resolveNetworkVisualizationHeightPx,
  type NetworkVisualizationResolvedFrame,
} from '../network-view/network-view';
import { FLAPPY_NETWORK_HOVER_TRANSITION_DURATION_MS } from '../visualization/visualization.constants';
import {
  applyCanvasBackingSize,
  resolveNetworkCanvasSizePx,
} from './host.canvas.service';
import { createHostArchitectureSelector } from './host.architecture-selector.service';
import {
  FLAPPY_HOST_PANEL_PADDING,
  FLAPPY_HOST_PANEL_TRANSITION,
  FLAPPY_HOST_STATS_SPLIT_GAP,
  FLAPPY_HOST_TABLE_HOST_PADDING,
} from './host.constants';
import { resolveRequiredCanvas2dContext } from './host.dom.service';
import {
  resolveHoveredNetworkVisualizationTooltipScene,
  type NetworkVisualizationTooltipScene,
} from './host.network-tooltip.service';
import { installResponsiveViewportSizing } from './resize/host.resize.service';
import {
  createAndAttachHostStatsTable,
  updateStatsTableValues as updateHostStatsTableValues,
} from './host.stats.service';
import type {
  CanvasHostOptions,
  CanvasHostResult,
  HostStatsPartialValues,
} from './host.types';

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
  sidebarColumn: HTMLDivElement;
  networkCanvasHost: HTMLDivElement;
  architectureSelectorHost: HTMLDivElement;
};

type HostCanvasElements = {
  headerCanvas: HTMLCanvasElement;
  headerContext: CanvasRenderingContext2D;
  canvas: HTMLCanvasElement;
  context: CanvasRenderingContext2D;
  networkCanvas: HTMLCanvasElement;
  networkContext: CanvasRenderingContext2D;
};

type HostPointerClientPosition = {
  clientX: number;
  clientY: number;
};

type HostCanvasPoint = {
  xPx: number;
  yPx: number;
};

type HostHoveredNodeAnimationState = {
  currentIntensity: number;
  targetIntensity: number;
  lastUpdatedAtMs: number;
};

type HostNetworkVisualizationTooltipElements = {
  tooltipArrowElement: HTMLDivElement;
  tooltipBodyElement: HTMLDivElement;
  tooltipElement: HTMLDivElement;
  tooltipHeadingElement: HTMLDivElement;
};

type HostNetworkVisualizationState = {
  previousNetworkForVisualization: Network | undefined;
  previousVisualizationInputSize: number;
  previousVisualizationOutputSize: number;
  latestResolvedFrame: NetworkVisualizationResolvedFrame | undefined;
  hoveredNodeIndices: number[] | undefined;
  hoveredNodeAnimationsByNodeIndex: Map<number, HostHoveredNodeAnimationState>;
  lastPointerClientPosition: HostPointerClientPosition | undefined;
  pendingHoverAnimationFrameId: number | undefined;
  pendingRedrawAnimationFrameId: number | undefined;
  pendingRedrawSyncHoveredNodeFromPointer: boolean;
};

type HostNetworkVisualizationController = {
  renderNetworkArchitecture: NetworkVisualizationHandle['renderNetworkArchitecture'];
  resizeNetworkCanvasToHost: () => void;
  redrawCurrentNetworkArchitecture: () => void;
};

const FLAPPY_NETWORK_TOOLTIP_OFFSET_PX = 10;
const FLAPPY_NETWORK_TOOLTIP_HOST_MARGIN_PX = 8;
const FLAPPY_NETWORK_TOOLTIP_ARROW_EDGE_MARGIN_PX = 18;
const FLAPPY_NETWORK_TOOLTIP_INPUT_MIN_WIDTH_PX = 320;
const FLAPPY_NETWORK_TOOLTIP_GROUP_MIN_WIDTH_PX = 420;
const FLAPPY_NETWORK_TOOLTIP_PADDING = '10px 12px';
const FLAPPY_NETWORK_TOOLTIP_RADIUS_PX = 12;
const FLAPPY_NETWORK_TOOLTIP_HEADING_FONT_SIZE = '11px';
const FLAPPY_NETWORK_TOOLTIP_BODY_FONT_SIZE = '10px';
const FLAPPY_NETWORK_TOOLTIP_TRANSITION =
  'opacity 120ms ease-out, transform 120ms ease-out';

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
  options: CanvasHostOptions,
): CanvasHostResult {
  return createCanvasHostInternal(containerElement, options);
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
  options: CanvasHostOptions,
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
  const architectureSelectorController = createHostArchitectureSelector(
    options,
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
    architectureSelectorController.element,
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
    architectureSelectorController,
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
  statsSplitContainer.style.alignItems = 'stretch';
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

  // Step 4: Create the right-column panel hosts for network view and architecture controls.
  const sidebarColumn = document.createElement('div');
  sidebarColumn.style.flex = '1 1 0';
  sidebarColumn.style.minWidth = '0';
  sidebarColumn.style.display = 'flex';
  sidebarColumn.style.flexDirection = 'column';
  sidebarColumn.style.gap = FLAPPY_HOST_STATS_SPLIT_GAP;

  const networkCanvasHost = document.createElement('div');
  networkCanvasHost.style.flex = '0 0 auto';
  networkCanvasHost.style.minWidth = '0';
  networkCanvasHost.style.height = `${FLAPPY_UI_NETWORK_HOST_INITIAL_HEIGHT_PX}px`;
  networkCanvasHost.style.background = FLAPPY_UI_NETWORK_HOST_BACKGROUND;
  networkCanvasHost.style.boxSizing = 'border-box';
  networkCanvasHost.style.border = hostVisualPrimitives.unifiedBorder;
  networkCanvasHost.style.padding = `${FLAPPY_UI_NETWORK_HOST_INSET_PX / 2}px`;
  networkCanvasHost.style.boxShadow = FLAPPY_UI_CANVAS_INSET_SHADOW;
  networkCanvasHost.style.position = 'relative';
  networkCanvasHost.style.overflow = 'visible';

  const architectureSelectorHost = document.createElement('div');
  architectureSelectorHost.style.flex = '1 1 auto';
  architectureSelectorHost.style.minWidth = '0';
  architectureSelectorHost.style.boxSizing = 'border-box';
  architectureSelectorHost.style.background = FLAPPY_NEON_PALETTE.hudPanelBackground;
  architectureSelectorHost.style.border = hostVisualPrimitives.unifiedBorder;
  architectureSelectorHost.style.padding = FLAPPY_HOST_TABLE_HOST_PADDING;
  architectureSelectorHost.style.boxShadow = FLAPPY_UI_CANVAS_INSET_SHADOW;

  return {
    outerFrame,
    contentColumn,
    mainSplitContainer,
    statsContainer,
    statsSplitContainer,
    statsTableHost,
    sidebarColumn,
    networkCanvasHost,
    architectureSelectorHost,
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
  networkCanvas.style.cursor = 'crosshair';
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
  const networkVisualizationTooltipElements =
    createNetworkVisualizationTooltipElements();
  const hostNetworkVisualizationState: HostNetworkVisualizationState = {
    previousNetworkForVisualization: undefined,
    previousVisualizationInputSize: FLAPPY_NETWORK_INPUT_SIZE,
    previousVisualizationOutputSize: FLAPPY_NETWORK_OUTPUT_SIZE,
    latestResolvedFrame: undefined,
    hoveredNodeIndices: undefined,
    hoveredNodeAnimationsByNodeIndex: new Map<
      number,
      HostHoveredNodeAnimationState
    >(),
    lastPointerClientPosition: undefined,
    pendingHoverAnimationFrameId: undefined,
    pendingRedrawAnimationFrameId: undefined,
    pendingRedrawSyncHoveredNodeFromPointer: false,
  };
  networkCanvasHost.appendChild(networkVisualizationTooltipElements.tooltipElement);

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

    // Step 3: Drop the cached resolved frame because layout depends on canvas size.
    hostNetworkVisualizationState.latestResolvedFrame = undefined;
  };

  const resolveLatestNetworkVisualizationFrame =
    (): NetworkVisualizationResolvedFrame => {
      // Step 1: Reuse the cached frame whenever the network payload and canvas size are unchanged.
      if (hostNetworkVisualizationState.latestResolvedFrame) {
        return hostNetworkVisualizationState.latestResolvedFrame;
      }

      // Step 2: Resolve and cache a fresh frame when the payload or canvas size changed.
      const latestResolvedFrame = resolveNetworkVisualizationFrame(
        networkContext,
        hostNetworkVisualizationState.previousNetworkForVisualization,
        hostNetworkVisualizationState.previousVisualizationInputSize,
        hostNetworkVisualizationState.previousVisualizationOutputSize,
      );
      hostNetworkVisualizationState.latestResolvedFrame = latestResolvedFrame;
      return latestResolvedFrame;
    };

  const requestNetworkVisualizationRedraw = (
    syncHoveredNodeFromPointer: boolean,
  ): void => {
    // Step 1: Preserve the strongest redraw requirement while a frame is pending.
    hostNetworkVisualizationState.pendingRedrawSyncHoveredNodeFromPointer =
      hostNetworkVisualizationState.pendingRedrawSyncHoveredNodeFromPointer ||
      syncHoveredNodeFromPointer;

    // Step 2: Coalesce repeated redraw requests into one animation-frame paint.
    if (
      typeof hostNetworkVisualizationState.pendingRedrawAnimationFrameId ===
      'number'
    ) {
      return;
    }

    hostNetworkVisualizationState.pendingRedrawAnimationFrameId =
      requestAnimationFrame(() => {
        const shouldSyncHoveredNodeFromPointer =
          hostNetworkVisualizationState.pendingRedrawSyncHoveredNodeFromPointer;
        hostNetworkVisualizationState.pendingRedrawAnimationFrameId = undefined;
        hostNetworkVisualizationState.pendingRedrawSyncHoveredNodeFromPointer = false;
        drawCurrentNetworkVisualization(shouldSyncHoveredNodeFromPointer);
      });
  };

  const requestHoveredNodeAnimationFrameIfNeeded = (): void => {
    // Step 1: Skip scheduling when a frame is already pending or nothing is animating.
    if (
      typeof hostNetworkVisualizationState.pendingHoverAnimationFrameId ===
        'number' ||
      !hasUnsettledHoveredNodeAnimations(
        hostNetworkVisualizationState.hoveredNodeAnimationsByNodeIndex,
      )
    ) {
      return;
    }

    // Step 2: Advance hover fade intensities on the next frame and redraw while needed.
    hostNetworkVisualizationState.pendingHoverAnimationFrameId =
      requestAnimationFrame(() => {
        hostNetworkVisualizationState.pendingHoverAnimationFrameId = undefined;
        const animationTimestampMs = resolveHoverAnimationTimestampMs();
        const didAnimationAdvance = syncHoveredNodeAnimationsToTimestamp(
          hostNetworkVisualizationState.hoveredNodeAnimationsByNodeIndex,
          animationTimestampMs,
        );

        if (didAnimationAdvance) {
          drawCurrentNetworkVisualization(false);
          return;
        }

        requestHoveredNodeAnimationFrameIfNeeded();
      });
  };

  const updateHoveredNodeTargets = (
    nextHoveredNodeIndices: number[] | undefined,
  ): boolean => {
    // Step 1: Skip target updates when the hovered node collection is unchanged.
    if (
      hoveredNodeIndicesAreEqual(
        nextHoveredNodeIndices,
        hostNetworkVisualizationState.hoveredNodeIndices,
      )
    ) {
      return false;
    }

    // Step 2: Retarget the per-node fade state so old and new hovers can overlap.
    applyHoveredNodeAnimationTargets(
      hostNetworkVisualizationState.hoveredNodeAnimationsByNodeIndex,
      nextHoveredNodeIndices,
      resolveHoverAnimationTimestampMs(),
    );
    hostNetworkVisualizationState.hoveredNodeIndices = nextHoveredNodeIndices;
    requestHoveredNodeAnimationFrameIfNeeded();
    return true;
  };

  const drawCurrentNetworkVisualization = (
    syncHoveredNodeFromPointer: boolean,
  ): void => {
    // Step 1: Advance hover fade intensities before drawing the current frame.
    syncHoveredNodeAnimationsToTimestamp(
      hostNetworkVisualizationState.hoveredNodeAnimationsByNodeIndex,
      resolveHoverAnimationTimestampMs(),
    );

    // Step 2: Resolve the cached frame and repaint it with the current hover state.
    const latestResolvedFrame = resolveLatestNetworkVisualizationFrame();
    const latestPositionedScene = drawResolvedNetworkVisualization(
      networkContext,
      latestResolvedFrame,
      {
        hoveredNodeIndices: hostNetworkVisualizationState.hoveredNodeIndices,
        animatedHoveredNodes: resolveAnimatedHoveredNodes(
          hostNetworkVisualizationState.hoveredNodeAnimationsByNodeIndex,
        ),
      },
    );
    syncNetworkVisualizationTooltip(
      networkCanvasHost,
      networkCanvas,
      networkVisualizationTooltipElements,
      hostNetworkVisualizationState.lastPointerClientPosition,
      latestPositionedScene,
    );

    // Step 3: Re-hit-test against the new scene when the pointer is still active.
    if (!syncHoveredNodeFromPointer) {
      requestHoveredNodeAnimationFrameIfNeeded();
      return;
    }

    const resolvedHoveredNodeIndices =
      resolveHoveredNodeIndicesFromClientPosition(
        networkCanvas,
        hostNetworkVisualizationState.lastPointerClientPosition,
        latestPositionedScene,
      );
    if (!updateHoveredNodeTargets(resolvedHoveredNodeIndices)) {
      requestHoveredNodeAnimationFrameIfNeeded();
      return;
    }

    // Step 4: Redraw once so the new fade targets become visible immediately.
    drawCurrentNetworkVisualization(false);
  };

  const handleNetworkCanvasPointerMove = (event: PointerEvent): void => {
    // Step 1: Cache the latest client-space pointer coordinates.
    hostNetworkVisualizationState.lastPointerClientPosition = {
      clientX: event.clientX,
      clientY: event.clientY,
    };

    // Step 2: Sync the floating tooltip immediately from the cached frame.
    syncNetworkVisualizationTooltip(
      networkCanvasHost,
      networkCanvas,
      networkVisualizationTooltipElements,
      hostNetworkVisualizationState.lastPointerClientPosition,
      hostNetworkVisualizationState.latestResolvedFrame?.positionedScene,
    );

    // Step 3: Redraw only when the hovered node actually changed.
    const resolvedHoveredNodeIndices =
      resolveHoveredNodeIndicesFromClientPosition(
        networkCanvas,
        hostNetworkVisualizationState.lastPointerClientPosition,
        hostNetworkVisualizationState.latestResolvedFrame?.positionedScene,
      );
    if (!updateHoveredNodeTargets(resolvedHoveredNodeIndices)) {
      return;
    }

    requestNetworkVisualizationRedraw(false);
  };

  const handleNetworkCanvasPointerLeave = (): void => {
    // Step 1: Clear cached pointer state when the cursor leaves the canvas.
    hostNetworkVisualizationState.lastPointerClientPosition = undefined;
    hideNetworkVisualizationTooltip(networkVisualizationTooltipElements);
    if (!hostNetworkVisualizationState.hoveredNodeIndices?.length) {
      return;
    }

    // Step 2: Redraw once so the default no-hover styling is restored.
    updateHoveredNodeTargets(undefined);
    requestNetworkVisualizationRedraw(false);
  };

  networkCanvas.addEventListener('pointermove', handleNetworkCanvasPointerMove);
  networkCanvas.addEventListener(
    'pointerleave',
    handleNetworkCanvasPointerLeave,
  );

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
      hostNetworkVisualizationState.latestResolvedFrame = undefined;

      // Step 3: Draw the current network visualization payload.
      requestNetworkVisualizationRedraw(true);
    };

  /**
   * Redraws the most recently rendered network visualization payload.
   *
   * @returns Nothing.
   */
  const redrawCurrentNetworkArchitecture = (): void => {
    // Step 1: Re-render the last known payload after external resize changes.
    requestNetworkVisualizationRedraw(true);
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
  architectureSelectorElement: HTMLDivElement,
): void {
  // Step 1: Mount the stats section and network panel into the stats container.
  hostLayoutElements.networkCanvasHost.appendChild(networkCanvas);
  hostLayoutElements.architectureSelectorHost.appendChild(
    architectureSelectorElement,
  );
  hostLayoutElements.sidebarColumn.appendChild(hostLayoutElements.networkCanvasHost);
  hostLayoutElements.sidebarColumn.appendChild(
    hostLayoutElements.architectureSelectorHost,
  );
  hostLayoutElements.statsSplitContainer.appendChild(
    hostLayoutElements.statsTableHost,
  );
  hostLayoutElements.statsSplitContainer.appendChild(hostLayoutElements.sidebarColumn);
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

function createNetworkVisualizationTooltipElements(): HostNetworkVisualizationTooltipElements {
  const tooltipElement = document.createElement('div');
  const tooltipHeadingElement = document.createElement('div');
  const tooltipBodyElement = document.createElement('div');
  const tooltipArrowElement = document.createElement('div');

  tooltipElement.setAttribute('role', 'tooltip');
  tooltipElement.style.position = 'absolute';
  tooltipElement.style.left = '0';
  tooltipElement.style.top = '0';
  tooltipElement.style.opacity = '0';
  tooltipElement.style.visibility = 'hidden';
  tooltipElement.style.pointerEvents = 'none';
  tooltipElement.style.boxSizing = 'border-box';
  tooltipElement.style.padding = FLAPPY_NETWORK_TOOLTIP_PADDING;
  tooltipElement.style.border = `1px solid ${FLAPPY_NEON_PALETTE.hudPanelBorder}`;
  tooltipElement.style.borderRadius = `${FLAPPY_NETWORK_TOOLTIP_RADIUS_PX}px`;
  tooltipElement.style.background = 'rgba(0, 21, 34, 0.98)';
  tooltipElement.style.boxShadow = '0 0 14px rgba(15, 181, 255, 0.26)';
  tooltipElement.style.backdropFilter = 'blur(6px)';
  tooltipElement.style.transform = 'translateY(6px)';
  tooltipElement.style.transition = FLAPPY_NETWORK_TOOLTIP_TRANSITION;
  tooltipElement.style.zIndex = '8';

  tooltipArrowElement.style.position = 'absolute';
  tooltipArrowElement.style.bottom = '-6px';
  tooltipArrowElement.style.width = '12px';
  tooltipArrowElement.style.height = '12px';
  tooltipArrowElement.style.transform = 'translateX(-50%) rotate(45deg)';
  tooltipArrowElement.style.background = 'rgba(0, 21, 34, 0.98)';
  tooltipArrowElement.style.borderRight = `1px solid ${FLAPPY_NEON_PALETTE.hudPanelBorder}`;
  tooltipArrowElement.style.borderBottom = `1px solid ${FLAPPY_NEON_PALETTE.hudPanelBorder}`;
  tooltipArrowElement.style.boxShadow = '0 0 10px rgba(15, 181, 255, 0.26)';

  tooltipHeadingElement.style.fontFamily = FLAPPY_MONOSPACE_FONT_FAMILY;
  tooltipHeadingElement.style.fontSize = FLAPPY_NETWORK_TOOLTIP_HEADING_FONT_SIZE;
  tooltipHeadingElement.style.fontWeight = '700';
  tooltipHeadingElement.style.letterSpacing = '0.08em';
  tooltipHeadingElement.style.textTransform = 'uppercase';
  tooltipHeadingElement.style.color = FLAPPY_NEON_PALETTE.hudAccent;
  tooltipHeadingElement.style.marginBottom = '8px';
  tooltipHeadingElement.style.textShadow = '0 0 8px rgba(255, 154, 46, 0.35)';

  tooltipBodyElement.style.display = 'flex';
  tooltipBodyElement.style.flexDirection = 'column';
  tooltipBodyElement.style.gap = '8px';
  tooltipBodyElement.style.fontFamily = FLAPPY_MONOSPACE_FONT_FAMILY;
  tooltipBodyElement.style.fontSize = FLAPPY_NETWORK_TOOLTIP_BODY_FONT_SIZE;
  tooltipBodyElement.style.lineHeight = '1.45';
  tooltipBodyElement.style.color = FLAPPY_NEON_PALETTE.hudText;

  tooltipElement.appendChild(tooltipHeadingElement);
  tooltipElement.appendChild(tooltipBodyElement);
  tooltipElement.appendChild(tooltipArrowElement);
  return {
    tooltipArrowElement,
    tooltipBodyElement,
    tooltipElement,
    tooltipHeadingElement,
  };
}

function syncNetworkVisualizationTooltip(
  networkCanvasHost: HTMLDivElement,
  networkCanvas: HTMLCanvasElement,
  networkVisualizationTooltipElements: HostNetworkVisualizationTooltipElements,
  pointerClientPosition: HostPointerClientPosition | undefined,
  positionedScene: NetworkVisualizationPositionedScene | undefined,
): void {
  if (!pointerClientPosition || !positionedScene) {
    hideNetworkVisualizationTooltip(networkVisualizationTooltipElements);
    return;
  }

  const canvasPoint = resolveCanvasPointFromClientPosition(
    networkCanvas,
    pointerClientPosition,
  );
  if (!canvasPoint) {
    hideNetworkVisualizationTooltip(networkVisualizationTooltipElements);
    return;
  }

  const tooltipScene = resolveHoveredNetworkVisualizationTooltipScene(
    canvasPoint,
    positionedScene,
  );
  if (!tooltipScene) {
    hideNetworkVisualizationTooltip(networkVisualizationTooltipElements);
    return;
  }

  showNetworkVisualizationTooltip(
    networkCanvasHost,
    networkCanvas,
    networkVisualizationTooltipElements,
    tooltipScene,
  );
}

function showNetworkVisualizationTooltip(
  networkCanvasHost: HTMLDivElement,
  networkCanvas: HTMLCanvasElement,
  networkVisualizationTooltipElements: HostNetworkVisualizationTooltipElements,
  tooltipScene: NetworkVisualizationTooltipScene,
): void {
  const canvasDisplayWidthPx = Math.max(1, networkCanvas.clientWidth);
  const canvasDisplayHeightPx = Math.max(1, networkCanvas.clientHeight);
  const scaleXPx = canvasDisplayWidthPx / Math.max(1, networkCanvas.width);
  const scaleYPx = canvasDisplayHeightPx / Math.max(1, networkCanvas.height);
  const availableTooltipWidthPx = Math.max(
    0,
    networkCanvasHost.clientWidth - FLAPPY_NETWORK_TOOLTIP_HOST_MARGIN_PX * 2,
  );
  if (availableTooltipWidthPx <= 0) {
    hideNetworkVisualizationTooltip(networkVisualizationTooltipElements);
    return;
  }

  const minimumTooltipWidthPx =
    tooltipScene.kind === 'group'
      ? FLAPPY_NETWORK_TOOLTIP_GROUP_MIN_WIDTH_PX
      : FLAPPY_NETWORK_TOOLTIP_INPUT_MIN_WIDTH_PX;
  const anchorWidthPx = tooltipScene.anchorWidthPx * scaleXPx;
  const tooltipWidthPx = Math.min(
    availableTooltipWidthPx,
    Math.max(minimumTooltipWidthPx, anchorWidthPx),
  );
  const anchorCenterXPx =
    networkCanvas.offsetLeft + tooltipScene.anchorCenterXPx * scaleXPx;
  const anchorTopPx =
    networkCanvas.offsetTop + tooltipScene.anchorTopPx * scaleYPx;

  networkVisualizationTooltipElements.tooltipHeadingElement.textContent =
    tooltipScene.heading;
  populateNetworkVisualizationTooltipBody(
    networkVisualizationTooltipElements.tooltipBodyElement,
    tooltipScene.bodyParagraphs,
  );
  networkVisualizationTooltipElements.tooltipElement.style.width =
    `${tooltipWidthPx}px`;
  networkVisualizationTooltipElements.tooltipElement.style.visibility = 'hidden';
  const tooltipHeightPx =
    networkVisualizationTooltipElements.tooltipElement.offsetHeight;
  const maximumTooltipLeftPx = Math.max(
    FLAPPY_NETWORK_TOOLTIP_HOST_MARGIN_PX,
    networkCanvasHost.clientWidth -
      FLAPPY_NETWORK_TOOLTIP_HOST_MARGIN_PX -
      tooltipWidthPx,
  );
  const tooltipLeftPx = clamp(
    anchorCenterXPx - tooltipWidthPx * 0.5,
    FLAPPY_NETWORK_TOOLTIP_HOST_MARGIN_PX,
    maximumTooltipLeftPx,
  );
  const tooltipTopPx = Math.max(
    FLAPPY_NETWORK_TOOLTIP_HOST_MARGIN_PX,
    anchorTopPx - tooltipHeightPx - FLAPPY_NETWORK_TOOLTIP_OFFSET_PX,
  );
  const tooltipArrowLeftPx = clamp(
    anchorCenterXPx - tooltipLeftPx,
    FLAPPY_NETWORK_TOOLTIP_ARROW_EDGE_MARGIN_PX,
    tooltipWidthPx - FLAPPY_NETWORK_TOOLTIP_ARROW_EDGE_MARGIN_PX,
  );

  networkVisualizationTooltipElements.tooltipElement.style.left =
    `${tooltipLeftPx}px`;
  networkVisualizationTooltipElements.tooltipElement.style.top =
    `${tooltipTopPx}px`;
  networkVisualizationTooltipElements.tooltipArrowElement.style.left =
    `${tooltipArrowLeftPx}px`;
  networkVisualizationTooltipElements.tooltipElement.style.visibility =
    'visible';
  networkVisualizationTooltipElements.tooltipElement.style.opacity = '1';
  networkVisualizationTooltipElements.tooltipElement.style.transform =
    'translateY(0)';
}

function populateNetworkVisualizationTooltipBody(
  tooltipBodyElement: HTMLDivElement,
  tooltipBodyParagraphs: readonly string[],
): void {
  tooltipBodyElement.replaceChildren(
    ...tooltipBodyParagraphs.map((tooltipBodyParagraph) => {
      const tooltipParagraphElement = document.createElement('p');
      tooltipParagraphElement.textContent = tooltipBodyParagraph;
      tooltipParagraphElement.style.margin = '0';
      tooltipParagraphElement.style.whiteSpace = 'normal';
      return tooltipParagraphElement;
    }),
  );
}

function hideNetworkVisualizationTooltip(
  networkVisualizationTooltipElements: HostNetworkVisualizationTooltipElements,
): void {
  networkVisualizationTooltipElements.tooltipElement.style.opacity = '0';
  networkVisualizationTooltipElements.tooltipElement.style.visibility = 'hidden';
  networkVisualizationTooltipElements.tooltipElement.style.transform =
    'translateY(6px)';
}

function resolveHoveredNodeIndicesFromClientPosition(
  networkCanvas: HTMLCanvasElement,
  pointerClientPosition: HostPointerClientPosition | undefined,
  positionedScene: NetworkVisualizationPositionedScene | undefined,
): number[] | undefined {
  // Step 1: Exit early when there is no pointer or no positioned scene yet.
  if (!pointerClientPosition || !positionedScene) {
    return undefined;
  }

  // Step 2: Convert the client-space pointer into canvas-space coordinates.
  const canvasPoint = resolveCanvasPointFromClientPosition(
    networkCanvas,
    pointerClientPosition,
  );
  if (!canvasPoint) {
    return undefined;
  }

  // Step 3: Resolve the hovered node by testing against the positioned boxes.
  return resolveHoveredNodeIndicesFromCanvasPoint(canvasPoint, positionedScene);
}

function resolveCanvasPointFromClientPosition(
  networkCanvas: HTMLCanvasElement,
  pointerClientPosition: HostPointerClientPosition,
): HostCanvasPoint | undefined {
  // Step 1: Resolve the current canvas bounds in client space.
  const canvasBounds = networkCanvas.getBoundingClientRect();
  if (canvasBounds.width <= 0 || canvasBounds.height <= 0) {
    return undefined;
  }

  // Step 2: Clear hover when the cached pointer is no longer inside the canvas.
  if (
    pointerClientPosition.clientX < canvasBounds.left ||
    pointerClientPosition.clientX > canvasBounds.right ||
    pointerClientPosition.clientY < canvasBounds.top ||
    pointerClientPosition.clientY > canvasBounds.bottom
  ) {
    return undefined;
  }

  // Step 3: Convert the client-space point into backing-store coordinates.
  const scaleXPx = networkCanvas.width / canvasBounds.width;
  const scaleYPx = networkCanvas.height / canvasBounds.height;
  return {
    xPx: (pointerClientPosition.clientX - canvasBounds.left) * scaleXPx,
    yPx: (pointerClientPosition.clientY - canvasBounds.top) * scaleYPx,
  };
}

function resolveHoveredNodeIndicesFromCanvasPoint(
  canvasPoint: HostCanvasPoint,
  positionedScene: NetworkVisualizationPositionedScene,
): number[] | undefined {
  // Step 1: Prefer direct node hover when the pointer is over a node box.
  const hoveredNodeIndex = resolveHoveredNodeIndexFromCanvasPoint(
    canvasPoint,
    positionedScene,
  );
  if (typeof hoveredNodeIndex === 'number') {
    return [hoveredNodeIndex];
  }

  // Step 2: Fall back to one-input hover when the pointer is over a description row.
  const hoveredInputDescriptionNodeIndices =
    resolveHoveredInputDescriptionNodeIndicesFromCanvasPoint(
      canvasPoint,
      positionedScene,
    );
  if (hoveredInputDescriptionNodeIndices?.length) {
    return hoveredInputDescriptionNodeIndices;
  }

  // Step 3: Fall back to recurrent-column combo-hover when the pointer is over a guide chip.
  const hoveredHiddenColumnNodeIndices =
    resolveHoveredHiddenColumnNodeIndicesFromCanvasPoint(
      canvasPoint,
      positionedScene,
    );
  if (hoveredHiddenColumnNodeIndices?.length) {
    return hoveredHiddenColumnNodeIndices;
  }

  // Step 4: Fall back to input-group combo-hover when the pointer is over a category band.
  return resolveHoveredInputGroupNodeIndicesFromCanvasPoint(
    canvasPoint,
    positionedScene,
  );
}

function resolveHoveredNodeIndexFromCanvasPoint(
  canvasPoint: HostCanvasPoint,
  positionedScene: NetworkVisualizationPositionedScene,
): number | undefined {
  // Step 1: Resolve half-dimensions used by the centered positioned-node layout.
  const halfNodeWidthPx = positionedScene.nodeDimensions.widthPx * 0.5;
  const halfNodeHeightPx = positionedScene.nodeDimensions.heightPx * 0.5;

  // Step 2: Return the topmost node whose box contains the pointer coordinates.
  return positionedScene.positionedNodes.findLast((positionedNode) => {
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
  })?.node.index;
}

function resolveHoveredInputGroupNodeIndicesFromCanvasPoint(
  canvasPoint: HostCanvasPoint,
  positionedScene: NetworkVisualizationPositionedScene,
): number[] | undefined {
  return positionedScene.inputGroupLabelBandScenes.findLast(
    (inputGroupLabelBandScene) =>
      canvasPoint.xPx >= inputGroupLabelBandScene.leftPx &&
      canvasPoint.xPx <=
        inputGroupLabelBandScene.leftPx + inputGroupLabelBandScene.widthPx &&
      canvasPoint.yPx >= inputGroupLabelBandScene.topPx &&
      canvasPoint.yPx <=
        inputGroupLabelBandScene.topPx + inputGroupLabelBandScene.heightPx,
  )?.nodeIndices;
}

function resolveHoveredHiddenColumnNodeIndicesFromCanvasPoint(
  canvasPoint: HostCanvasPoint,
  positionedScene: NetworkVisualizationPositionedScene,
): number[] | undefined {
  return positionedScene.hiddenColumnLabelScenes?.findLast(
    (hiddenColumnLabelScene) =>
      canvasPoint.xPx >= hiddenColumnLabelScene.leftPx &&
      canvasPoint.xPx <=
        hiddenColumnLabelScene.leftPx + hiddenColumnLabelScene.widthPx &&
      canvasPoint.yPx >= hiddenColumnLabelScene.topPx &&
      canvasPoint.yPx <=
        hiddenColumnLabelScene.topPx + hiddenColumnLabelScene.heightPx,
  )?.nodeIndices;
}

function resolveHoveredInputDescriptionNodeIndicesFromCanvasPoint(
  canvasPoint: HostCanvasPoint,
  positionedScene: NetworkVisualizationPositionedScene,
): number[] | undefined {
  const hoveredInputDescriptionScene = positionedScene.inputDescriptionScenes.findLast(
    (inputDescriptionScene) =>
      canvasPoint.xPx >= inputDescriptionScene.leftPx &&
      canvasPoint.xPx <=
        inputDescriptionScene.leftPx + inputDescriptionScene.widthPx &&
      canvasPoint.yPx >= inputDescriptionScene.topPx &&
      canvasPoint.yPx <=
        inputDescriptionScene.topPx + inputDescriptionScene.heightPx,
  );

  return hoveredInputDescriptionScene
    ? [hoveredInputDescriptionScene.nodeIndex]
    : undefined;
}

function hoveredNodeIndicesAreEqual(
  leftNodeIndices: readonly number[] | undefined,
  rightNodeIndices: readonly number[] | undefined,
): boolean {
  if (!leftNodeIndices?.length && !rightNodeIndices?.length) {
    return true;
  }

  if (
    !leftNodeIndices ||
    !rightNodeIndices ||
    leftNodeIndices.length !== rightNodeIndices.length
  ) {
    return false;
  }

  return leftNodeIndices.every(
    (nodeIndex, nodeIndexPosition) =>
      nodeIndex === rightNodeIndices[nodeIndexPosition],
  );
}

function resolveHoverAnimationTimestampMs(): number {
  if (
    typeof performance !== 'undefined' &&
    typeof performance.now === 'function'
  ) {
    return performance.now();
  }

  return Date.now();
}

function resolveNextHoveredNodeIntensity(
  currentIntensity: number,
  targetIntensity: number,
  elapsedMs: number,
): number {
  if (elapsedMs <= 0 || currentIntensity === targetIntensity) {
    return currentIntensity;
  }

  const intensityStep = elapsedMs / FLAPPY_NETWORK_HOVER_TRANSITION_DURATION_MS;
  if (targetIntensity > currentIntensity) {
    return Math.min(targetIntensity, currentIntensity + intensityStep);
  }

  return Math.max(targetIntensity, currentIntensity - intensityStep);
}

function syncHoveredNodeAnimationsToTimestamp(
  hoveredNodeAnimationsByNodeIndex: Map<number, HostHoveredNodeAnimationState>,
  timestampMs: number,
): boolean {
  let didAnimationAdvance = false;

  for (const [
    nodeIndex,
    hoveredNodeAnimationState,
  ] of hoveredNodeAnimationsByNodeIndex) {
    const nextIntensity = resolveNextHoveredNodeIntensity(
      hoveredNodeAnimationState.currentIntensity,
      hoveredNodeAnimationState.targetIntensity,
      timestampMs - hoveredNodeAnimationState.lastUpdatedAtMs,
    );
    hoveredNodeAnimationState.lastUpdatedAtMs = timestampMs;

    if (nextIntensity !== hoveredNodeAnimationState.currentIntensity) {
      hoveredNodeAnimationState.currentIntensity = nextIntensity;
      didAnimationAdvance = true;
    }

    if (
      hoveredNodeAnimationState.currentIntensity <= 0 &&
      hoveredNodeAnimationState.targetIntensity <= 0
    ) {
      hoveredNodeAnimationsByNodeIndex.delete(nodeIndex);
    }
  }

  return didAnimationAdvance;
}

function applyHoveredNodeAnimationTargets(
  hoveredNodeAnimationsByNodeIndex: Map<number, HostHoveredNodeAnimationState>,
  hoveredNodeIndices: readonly number[] | undefined,
  timestampMs: number,
): void {
  syncHoveredNodeAnimationsToTimestamp(
    hoveredNodeAnimationsByNodeIndex,
    timestampMs,
  );

  const hoveredNodeIndexSet = hoveredNodeIndices?.length
    ? new Set(hoveredNodeIndices)
    : undefined;
  for (const [
    nodeIndex,
    hoveredNodeAnimationState,
  ] of hoveredNodeAnimationsByNodeIndex) {
    hoveredNodeAnimationState.targetIntensity = hoveredNodeIndexSet?.has(
      nodeIndex,
    )
      ? 1
      : 0;
  }

  hoveredNodeIndices?.forEach((hoveredNodeIndex) => {
    if (hoveredNodeAnimationsByNodeIndex.has(hoveredNodeIndex)) {
      return;
    }

    hoveredNodeAnimationsByNodeIndex.set(hoveredNodeIndex, {
      currentIntensity: 0,
      targetIntensity: 1,
      lastUpdatedAtMs: timestampMs,
    });
  });
}

function hasUnsettledHoveredNodeAnimations(
  hoveredNodeAnimationsByNodeIndex: Map<number, HostHoveredNodeAnimationState>,
): boolean {
  for (const hoveredNodeAnimationState of hoveredNodeAnimationsByNodeIndex.values()) {
    if (
      Math.abs(
        hoveredNodeAnimationState.currentIntensity -
          hoveredNodeAnimationState.targetIntensity,
      ) > Number.EPSILON
    ) {
      return true;
    }
  }

  return false;
}

function resolveAnimatedHoveredNodes(
  hoveredNodeAnimationsByNodeIndex: Map<number, HostHoveredNodeAnimationState>,
): NetworkVisualizationAnimatedHoveredNode[] {
  return [...hoveredNodeAnimationsByNodeIndex.entries()].flatMap(
    ([nodeIndex, hoveredNodeAnimationState]) =>
      hoveredNodeAnimationState.currentIntensity > 0
        ? [
            {
              nodeIndex,
              intensity: hoveredNodeAnimationState.currentIntensity,
            },
          ]
        : [],
  );
}
