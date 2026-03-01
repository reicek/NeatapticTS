import Network from '../../../src/architecture/network';
import { clamp } from './browser-entry.math.utils';
import {
  createFlappyStatsTableRows,
  formatArchitectureStatsValue,
} from './browser-entry.stats.utils';
import { renderStandaloneTitleBox } from './browser-entry.text-frame.utils';
import {
  drawNetworkVisualization,
  resolveNetworkVisualizationHeightPx,
} from './browser-entry.network-view.utils';
import {
  FLAPPY_FRAME_MONOSPACE_FONT,
  FLAPPY_HEADER_TITLE_TEXT,
  FLAPPY_HEADER_CANVAS_HEIGHT_PX,
  FLAPPY_HUD_INITIALIZING_TEXT,
  FLAPPY_HUD_OFF_TEXT,
  FLAPPY_HUD_ZERO_DECIMAL_TEXT,
  FLAPPY_HUD_ZERO_TEXT,
  FLAPPY_MONOSPACE_FONT_FAMILY,
  FLAPPY_UI_CANVAS_INSET_SHADOW,
  FLAPPY_UI_CONTENT_COLUMN_TOP_PADDING_PX,
  FLAPPY_UI_DOUBLE_PANEL_BORDER,
  FLAPPY_UI_NETWORK_CANVAS_BACKGROUND,
  FLAPPY_UI_NETWORK_HOST_INSET_PX,
  FLAPPY_UI_NETWORK_HOST_INITIAL_HEIGHT_PX,
  FLAPPY_UI_NETWORK_HOST_BACKGROUND,
  FLAPPY_UI_OUTER_FRAME_MIN_SIDE_PADDING_PX,
  FLAPPY_UI_OUTER_FRAME_SIDE_PADDING_OFFSET_PX,
  FLAPPY_UI_OUTER_FRAME_BACKGROUND,
  FLAPPY_UI_STATS_ROW_BORDER,
  FLAPPY_UI_UNIFIED_INSET_SHADOW,
  FLAPPY_NEON_PALETTE,
  FLAPPY_SCREEN_PADDING_PX,
  FLAPPY_STATS_KEYS,
  FLAPPY_VIEWPORT_MINIMUM_SIMULATION_HEIGHT_PX,
  FLAPPY_VIEWPORT_MINIMUM_SIMULATION_HEIGHT_RATIO,
  FLAPPY_VIEWPORT_MIN_NETWORK_HEIGHT_BUDGET_PX,
  FLAPPY_VIEWPORT_NETWORK_ONLY_BREAKPOINT_PX,
  FLAPPY_VIEWPORT_SIMULATION_BOTTOM_MARGIN_PX,
  FLAPPY_VIEWPORT_VERTICAL_LAYOUT_GUTTER_PX,
} from './browser-entry.constants';
import type {
  FlappyStatsKey,
  FlappyStatsTableCells,
  NetworkVisualizationHandle,
} from './browser-entry.types';
import {
  FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION,
  FLAPPY_NETWORK_INPUT_SIZE,
  FLAPPY_NETWORK_OUTPUT_SIZE,
} from './constants';

/**
 * Builds the browser demo host tree and returns rendering handles.
 *
 * @param containerElement - Root host container.
 * @returns Canvas handles, stats cells and network render callback.
 */
export function createCanvasHost(containerElement: HTMLElement): {
  canvas: HTMLCanvasElement;
  context: CanvasRenderingContext2D;
  statsValueByKey: FlappyStatsTableCells;
  renderNetworkArchitecture: NetworkVisualizationHandle['renderNetworkArchitecture'];
} {
  // Step 1: Reset host container and initialize shared visual primitives.
  containerElement.innerHTML = '';

  const outerFrame = document.createElement('section');
  const unifiedBorder = FLAPPY_UI_DOUBLE_PANEL_BORDER;
  const unifiedInsetShadow = FLAPPY_UI_UNIFIED_INSET_SHADOW;

  outerFrame.style.width = '100%';
  outerFrame.style.height = '100%';
  outerFrame.style.boxSizing = 'border-box';
  const sidePaddingPx = Math.max(
    FLAPPY_UI_OUTER_FRAME_MIN_SIDE_PADDING_PX,
    FLAPPY_SCREEN_PADDING_PX - FLAPPY_UI_OUTER_FRAME_SIDE_PADDING_OFFSET_PX,
  );
  outerFrame.style.padding = `0 ${sidePaddingPx}px ${sidePaddingPx}px ${sidePaddingPx}px`;
  outerFrame.style.border = unifiedBorder;
  outerFrame.style.background = FLAPPY_UI_OUTER_FRAME_BACKGROUND;
  outerFrame.style.boxShadow = unifiedInsetShadow;
  outerFrame.style.position = 'relative';

  // Step 2: Create top-level layout containers (content column + header/canvas/stats).
  const contentColumn = document.createElement('div');
  contentColumn.style.width = '100%';
  contentColumn.style.height = '100%';
  contentColumn.style.display = 'flex';
  contentColumn.style.flexDirection = 'column';
  contentColumn.style.gap = '0';
  contentColumn.style.alignItems = 'stretch';
  contentColumn.style.overflow = 'hidden';
  contentColumn.style.paddingTop = `${FLAPPY_UI_CONTENT_COLUMN_TOP_PADDING_PX}px`;

  // Step 3: Create header canvas used by the text-frame title renderer.
  const headerCanvas = document.createElement('canvas');
  headerCanvas.width = 1;
  headerCanvas.height = 1;
  headerCanvas.style.width = '100%';
  headerCanvas.style.height = `${FLAPPY_HEADER_CANVAS_HEIGHT_PX}px`;
  headerCanvas.style.display = 'block';
  headerCanvas.style.maxWidth = '100%';
  headerCanvas.style.boxSizing = 'border-box';
  headerCanvas.style.background = 'transparent';

  const headerContext = headerCanvas.getContext('2d');
  if (!headerContext) {
    throw new Error('Header canvas 2D context unavailable');
  }

  // Step 4: Create main simulation canvas.
  const canvas = document.createElement('canvas');
  canvas.width = 1;
  canvas.height = 1;
  canvas.style.width = '100%';
  canvas.style.height = '1px';
  canvas.style.display = 'block';
  canvas.style.maxWidth = '100%';
  canvas.style.boxSizing = 'border-box';
  canvas.style.background = FLAPPY_NEON_PALETTE.background;
  canvas.style.border = unifiedBorder;
  canvas.style.boxShadow = FLAPPY_UI_CANVAS_INSET_SHADOW;

  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('Canvas 2D context unavailable');
  }

  // Step 5: Create stats/network split pane container.
  const statsContainer = document.createElement('div');
  statsContainer.style.marginTop = '0';
  statsContainer.style.boxSizing = 'border-box';
  statsContainer.style.background = FLAPPY_NEON_PALETTE.hudPanelBackground;
  statsContainer.style.border = unifiedBorder;
  statsContainer.style.boxShadow = unifiedInsetShadow;
  statsContainer.style.padding = '8px';
  statsContainer.style.overflow = 'hidden';
  statsContainer.style.transition = 'max-height 120ms ease-out';
  statsContainer.style.minHeight = '0';

  const statsSplitContainer = document.createElement('div');
  statsSplitContainer.style.display = 'flex';
  statsSplitContainer.style.flexDirection = 'row';
  statsSplitContainer.style.alignItems = 'flex-start';
  statsSplitContainer.style.gap = '8px';
  statsSplitContainer.style.width = '100%';
  statsSplitContainer.style.boxSizing = 'border-box';

  const statsTableHost = document.createElement('div');
  statsTableHost.style.flex = '1 1 0';
  statsTableHost.style.minWidth = '0';
  statsTableHost.style.overflow = 'hidden';
  statsTableHost.style.boxSizing = 'border-box';
  statsTableHost.style.border = unifiedBorder;
  statsTableHost.style.padding = '6px 8px';

  // Step 6: Create stats table and initialize value cells.
  const statsTable = document.createElement('table');
  statsTable.style.borderCollapse = 'collapse';
  statsTable.style.width = '100%';
  statsTable.style.maxWidth = '100%';
  statsTable.style.tableLayout = 'fixed';
  statsTable.style.fontFamily = FLAPPY_MONOSPACE_FONT_FAMILY;
  statsTable.style.fontSize = '12px';
  statsTable.style.borderBottom = FLAPPY_UI_STATS_ROW_BORDER;

  const statsValueByKey = createFlappyStatsTableRows({
    statsTable,
    enableRuntimeInstrumentation: FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION,
    resolveCategoryColor: (
      statsKey,
    ): { keyColor: string; valueColor: string } => {
      if (statsKey === 'status') {
        return {
          keyColor: FLAPPY_NEON_PALETTE.statusText,
          valueColor: FLAPPY_NEON_PALETTE.statusText,
        };
      }
      if (statsKey.startsWith('current')) {
        return {
          keyColor: FLAPPY_NEON_PALETTE.currentRunText,
          valueColor: FLAPPY_NEON_PALETTE.currentRunText,
        };
      }
      if (statsKey.startsWith('best')) {
        return {
          keyColor: FLAPPY_NEON_PALETTE.bestRunText,
          valueColor: FLAPPY_NEON_PALETTE.bestRunText,
        };
      }
      return {
        keyColor: FLAPPY_NEON_PALETTE.hudText,
        valueColor: FLAPPY_NEON_PALETTE.hudAccent,
      };
    },
  });
  updateStatsTableValues(statsValueByKey, {
    currentHeader: 'Current run · Gen -',
    currentFrames: FLAPPY_HUD_ZERO_TEXT,
    currentPipes: FLAPPY_HUD_ZERO_TEXT,
    currentMaxFrames: FLAPPY_HUD_ZERO_TEXT,
    currentMaxPipes: FLAPPY_HUD_ZERO_TEXT,
    currentArchitecture: '-',
    telemetryHeader: 'Instrumentation',
    telemetryActivationsPerFrame: FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION
      ? FLAPPY_HUD_ZERO_DECIMAL_TEXT
      : FLAPPY_HUD_OFF_TEXT,
    telemetrySimulationStepsPerRaf: FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION
      ? FLAPPY_HUD_ZERO_DECIMAL_TEXT
      : FLAPPY_HUD_OFF_TEXT,
    telemetryHudUpdatesPerSecond: FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION
      ? FLAPPY_HUD_ZERO_DECIMAL_TEXT
      : FLAPPY_HUD_OFF_TEXT,
    telemetryMinorGcPerMinute: FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION
      ? FLAPPY_HUD_ZERO_DECIMAL_TEXT
      : FLAPPY_HUD_OFF_TEXT,
    bestHeader: 'Best run',
    bestFrames: FLAPPY_HUD_ZERO_TEXT,
    bestPipes: FLAPPY_HUD_ZERO_TEXT,
    bestMaxFrames: FLAPPY_HUD_ZERO_TEXT,
    bestMaxPipes: FLAPPY_HUD_ZERO_TEXT,
    bestArchitecture: '-',
    status: FLAPPY_HUD_INITIALIZING_TEXT,
  });
  statsTableHost.appendChild(statsTable);

  // Step 7: Create network visualization host + canvas.
  const networkCanvasHost = document.createElement('div');
  networkCanvasHost.style.flex = '1 1 0';
  networkCanvasHost.style.minWidth = '0';
  networkCanvasHost.style.height = `${FLAPPY_UI_NETWORK_HOST_INITIAL_HEIGHT_PX}px`;
  networkCanvasHost.style.background = FLAPPY_UI_NETWORK_HOST_BACKGROUND;
  networkCanvasHost.style.boxSizing = 'border-box';
  networkCanvasHost.style.border = unifiedBorder;
  networkCanvasHost.style.padding = '6px';
  networkCanvasHost.style.boxShadow = FLAPPY_UI_CANVAS_INSET_SHADOW;

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
  networkCanvasHost.appendChild(networkCanvas);

  const networkContext = networkCanvas.getContext('2d');
  if (!networkContext) {
    throw new Error('Network canvas 2D context unavailable');
  }

  // Step 8: Track last-rendered network payload for resize-triggered redraws.
  let previousNetworkForVisualization: Network | undefined;
  let previousVisualizationInputSize = FLAPPY_NETWORK_INPUT_SIZE;
  let previousVisualizationOutputSize = FLAPPY_NETWORK_OUTPUT_SIZE;

  /**
   * Resizes the network canvas backing resolution to match host CSS size.
   *
   * @returns Nothing.
   */
  const resizeNetworkCanvasToHost = (): void => {
    const networkCanvasWidthPx = Math.max(
      1,
      Math.floor(
        networkCanvasHost.clientWidth - FLAPPY_UI_NETWORK_HOST_INSET_PX,
      ),
    );
    const networkCanvasHeightPx = Math.max(
      1,
      Math.floor(
        networkCanvasHost.clientHeight - FLAPPY_UI_NETWORK_HOST_INSET_PX,
      ),
    );

    if (
      networkCanvas.width !== networkCanvasWidthPx ||
      networkCanvas.height !== networkCanvasHeightPx
    ) {
      networkCanvas.width = networkCanvasWidthPx;
      networkCanvas.height = networkCanvasHeightPx;
      networkCanvas.style.width = `${networkCanvasWidthPx}px`;
      networkCanvas.style.height = `${networkCanvasHeightPx}px`;
    }
  };

  /**
   * Renders one architecture payload into the network visualization panel.
   *
   * @param network - Runtime network instance (optional placeholder mode).
   * @param inputSize - Input width used by layout/labels.
   * @param outputSize - Output width used by layout/labels.
   * @returns Nothing.
   */
  const renderNetworkArchitecture: NetworkVisualizationHandle['renderNetworkArchitecture'] =
    (network, inputSize, outputSize) => {
      // Step 1: Resolve recommended panel height from architecture complexity.
      const recommendedNetworkCanvasHeightPx =
        resolveNetworkVisualizationHeightPx(network, inputSize, outputSize);
      networkCanvasHost.dataset.preferredHeightPx = String(
        recommendedNetworkCanvasHeightPx,
      );
      networkCanvasHost.style.height = `${recommendedNetworkCanvasHeightPx}px`;
      // Step 2: Resize canvas backing store and render current payload.
      resizeNetworkCanvasToHost();

      previousNetworkForVisualization = network;
      previousVisualizationInputSize = inputSize;
      previousVisualizationOutputSize = outputSize;
      drawNetworkVisualization(networkContext, network, inputSize, outputSize);
    };

  // Step 9: Mount host DOM tree in final order.
  statsSplitContainer.appendChild(statsTableHost);
  statsSplitContainer.appendChild(networkCanvasHost);
  statsContainer.appendChild(statsSplitContainer);

  contentColumn.appendChild(headerCanvas);
  contentColumn.appendChild(canvas);
  contentColumn.appendChild(statsContainer);
  outerFrame.appendChild(contentColumn);
  containerElement.appendChild(outerFrame);

  /**
   * Draws the standalone header title frame.
   *
   * @returns Nothing.
   */
  const drawHeaderFrame = (): void => {
    // Step 1: Sync backing resolution to display size.
    const displayWidthPx = Math.max(1, Math.floor(headerCanvas.clientWidth));
    const displayHeightPx = FLAPPY_HEADER_CANVAS_HEIGHT_PX;
    if (
      headerCanvas.width !== displayWidthPx ||
      headerCanvas.height !== displayHeightPx
    ) {
      headerCanvas.width = displayWidthPx;
      headerCanvas.height = displayHeightPx;
    }

    // Step 2: Render title box glyph frame.
    renderStandaloneTitleBox({
      context: headerContext,
      widthPx: displayWidthPx,
      heightPx: displayHeightPx,
      titleText: FLAPPY_HEADER_TITLE_TEXT,
      glyphColor: FLAPPY_NEON_PALETTE.hudPanelBorder,
      font: FLAPPY_FRAME_MONOSPACE_FONT,
    });
  };

  // Step 10: Install responsive sizing and redraw hooks.
  installResponsiveViewportSizing(
    canvas,
    contentColumn,
    statsContainer,
    statsSplitContainer,
    statsTableHost,
    networkCanvas,
    networkCanvasHost,
    () => {
      drawHeaderFrame();
      resizeNetworkCanvasToHost();
      renderNetworkArchitecture(
        previousNetworkForVisualization,
        previousVisualizationInputSize,
        previousVisualizationOutputSize,
      );
    },
  );

  // Step 11: Render initial placeholder visualization and header.
  renderNetworkArchitecture(
    undefined,
    FLAPPY_NETWORK_INPUT_SIZE,
    FLAPPY_NETWORK_OUTPUT_SIZE,
  );
  drawHeaderFrame();

  return { canvas, context, statsValueByKey, renderNetworkArchitecture };
}

/**
 * Applies partial stat updates to the rendered stats table.
 *
 * @param statsValueByKey - Lookup of stat keys to value cells.
 * @param partialValues - Subset of values to write this tick.
 * @returns Nothing.
 */
export function updateStatsTableValues(
  statsValueByKey: FlappyStatsTableCells,
  partialValues: Partial<Record<FlappyStatsKey, string>>,
): void {
  FLAPPY_STATS_KEYS.forEach((statsKey) => {
    const nextValue = partialValues[statsKey];
    if (nextValue == null) {
      return;
    }
    const statsCell = statsValueByKey[statsKey];
    if (!statsCell) {
      return;
    }
    const formattedValue =
      statsKey === 'currentArchitecture' || statsKey === 'bestArchitecture'
        ? formatArchitectureStatsValue(nextValue)
        : nextValue;
    statsCell.textContent = formattedValue;
  });
}

/**
 * Keeps canvas backing resolution synchronized with container size.
 *
 * @param canvas - Simulation canvas to resize.
 * @param containerElement - Width/height source.
 * @param statsContainer - Stats host element.
 * @param networkCanvas - Network canvas.
 * @param networkCanvasHost - Network host element.
 * @param onNetworkResize - Callback after network resize.
 * @returns Nothing.
 */
function installResponsiveViewportSizing(
  canvas: HTMLCanvasElement,
  containerElement: HTMLElement,
  statsContainer: HTMLElement,
  statsSplitContainer: HTMLElement,
  statsTableHost: HTMLElement,
  networkCanvas: HTMLCanvasElement,
  networkCanvasHost: HTMLElement,
  onNetworkResize: () => void,
): void {
  let pendingNetworkRedrawRequest = false;

  /**
   * Defers network redraw until after layout settles across two animation frames.
   *
   * @returns Nothing.
   */
  const requestNetworkRedrawAfterLayout = (): void => {
    if (pendingNetworkRedrawRequest) {
      return;
    }
    pendingNetworkRedrawRequest = true;

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        pendingNetworkRedrawRequest = false;
        onNetworkResize();
      });
    });
  };

  /**
   * Recomputes simulation/network canvas backing sizes from current layout constraints.
   *
   * @returns Nothing.
   */
  const applyCanvasSize = (): void => {
    // Step 1: Measure header and baseline layout constraints.
    const headerCanvasElement = containerElement.querySelector('canvas');
    const headerHeightPx = Math.max(
      0,
      headerCanvasElement instanceof HTMLCanvasElement
        ? headerCanvasElement.offsetHeight
        : 0,
    );

    // Step 2: Resolve stats/network panel sizing constraints.
    const preferredNetworkHeightPx = Number.parseInt(
      networkCanvasHost.dataset.preferredHeightPx ??
        `${networkCanvasHost.offsetHeight}`,
      10,
    );
    const nonNetworkStatsHeightPx = Math.max(
      0,
      statsContainer.scrollHeight - networkCanvasHost.offsetHeight,
    );

    const hardMinimumSimulationHeightPx = Math.max(
      FLAPPY_VIEWPORT_MINIMUM_SIMULATION_HEIGHT_PX,
      Math.floor(
        window.innerHeight * FLAPPY_VIEWPORT_MINIMUM_SIMULATION_HEIGHT_RATIO,
      ),
    );
    const totalCanvasBudgetPx = Math.max(
      1,
      containerElement.clientHeight - headerHeightPx,
    );
    const viewportWidthPx = Math.max(1, containerElement.clientWidth);
    const useNetworkOnlyPanel =
      viewportWidthPx < FLAPPY_VIEWPORT_NETWORK_ONLY_BREAKPOINT_PX;

    // Step 2.1: Toggle stats/network split layout for narrow viewports.
    statsTableHost.style.display = useNetworkOnlyPanel ? 'none' : 'block';
    statsSplitContainer.style.gap = useNetworkOnlyPanel ? '0' : '8px';
    networkCanvasHost.style.flex = useNetworkOnlyPanel ? '1 1 100%' : '1 1 0';
    networkCanvasHost.style.width = useNetworkOnlyPanel ? '100%' : 'auto';
    networkCanvasHost.style.maxWidth = '100%';

    const halfSplitTargetPx = Math.max(
      FLAPPY_VIEWPORT_MIN_NETWORK_HEIGHT_BUDGET_PX,
      Math.floor(
        (totalCanvasBudgetPx - FLAPPY_VIEWPORT_VERTICAL_LAYOUT_GUTTER_PX) * 0.5,
      ),
    );
    const maximumStatsHeightPx = Math.max(
      FLAPPY_VIEWPORT_MIN_NETWORK_HEIGHT_BUDGET_PX,
      totalCanvasBudgetPx -
        hardMinimumSimulationHeightPx -
        FLAPPY_VIEWPORT_VERTICAL_LAYOUT_GUTTER_PX,
    );
    const resolvedStatsPanelHeightPx = Math.floor(
      clamp(
        halfSplitTargetPx,
        FLAPPY_VIEWPORT_MIN_NETWORK_HEIGHT_BUDGET_PX,
        maximumStatsHeightPx,
      ),
    );

    // Step 3: Clamp network pane height and trigger redraw when height changed.
    const networkHeightBudgetPx = Math.max(
      FLAPPY_VIEWPORT_MIN_NETWORK_HEIGHT_BUDGET_PX,
      Math.floor(resolvedStatsPanelHeightPx - nonNetworkStatsHeightPx),
    );
    const resolvedNetworkHeightPx = Math.floor(
      clamp(
        preferredNetworkHeightPx,
        FLAPPY_VIEWPORT_MIN_NETWORK_HEIGHT_BUDGET_PX,
        networkHeightBudgetPx,
      ),
    );
    const previousNetworkHostHeightPx = networkCanvasHost.offsetHeight;
    if (networkCanvasHost.offsetHeight !== resolvedNetworkHeightPx) {
      networkCanvasHost.style.height = `${resolvedNetworkHeightPx}px`;
      if (previousNetworkHostHeightPx !== resolvedNetworkHeightPx) {
        requestNetworkRedrawAfterLayout();
      }
    }

    statsContainer.style.height = `${resolvedStatsPanelHeightPx}px`;
    statsContainer.style.maxHeight = `${resolvedStatsPanelHeightPx}px`;
    statsContainer.style.overflowY = 'hidden';

    // Step 4: Resolve simulation canvas size from remaining vertical budget.
    const availableWidthPx = Math.max(1, containerElement.clientWidth);
    const availableHeightPx = Math.max(
      1,
      totalCanvasBudgetPx -
        resolvedStatsPanelHeightPx -
        FLAPPY_VIEWPORT_VERTICAL_LAYOUT_GUTTER_PX -
        FLAPPY_VIEWPORT_SIMULATION_BOTTOM_MARGIN_PX,
    );

    // Step 5: Apply simulation canvas backing size.
    const nextWidthPx = Math.max(1, Math.floor(availableWidthPx));
    const nextHeightPx = Math.max(1, Math.floor(availableHeightPx));

    if (canvas.width !== nextWidthPx || canvas.height !== nextHeightPx) {
      canvas.width = nextWidthPx;
      canvas.height = nextHeightPx;
      canvas.style.width = `${nextWidthPx}px`;
      canvas.style.height = `${nextHeightPx}px`;
    }

    // Step 6: Apply network canvas backing size and redraw when changed.
    const networkCanvasWidthPx = Math.max(
      1,
      Math.floor(
        networkCanvasHost.clientWidth - FLAPPY_UI_NETWORK_HOST_INSET_PX,
      ),
    );
    const networkCanvasHeightPx = Math.max(
      1,
      Math.floor(
        networkCanvasHost.clientHeight - FLAPPY_UI_NETWORK_HOST_INSET_PX,
      ),
    );

    if (
      networkCanvas.width !== networkCanvasWidthPx ||
      networkCanvas.height !== networkCanvasHeightPx
    ) {
      networkCanvas.width = networkCanvasWidthPx;
      networkCanvas.height = networkCanvasHeightPx;
      networkCanvas.style.width = `${networkCanvasWidthPx}px`;
      networkCanvas.style.height = `${networkCanvasHeightPx}px`;
      onNetworkResize();
    }

    statsContainer.style.overflowY = 'hidden';
  };

  // Step 1: Initial sizing pass.
  applyCanvasSize();
  // Step 2: Queue first post-layout network redraw.
  requestNetworkRedrawAfterLayout();
  // Step 3: Listen to viewport resize.
  window.addEventListener('resize', applyCanvasSize);

  // Step 4: Attach container observer for responsive host resizing.
  if (typeof ResizeObserver === 'function') {
    const resizeObserver = new ResizeObserver(() => {
      applyCanvasSize();
      requestNetworkRedrawAfterLayout();
    });
    resizeObserver.observe(containerElement);
  }
}
