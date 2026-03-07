import type Network from '../../../../src/architecture/network';
import { renderStandaloneTitleBox } from './browser-entry.text-frame.utils';
import {
  drawNetworkVisualization,
  resolveNetworkVisualizationHeightPx,
} from './network-view/network-view';
import {
  FLAPPY_FRAME_MONOSPACE_FONT,
  FLAPPY_HEADER_TITLE_TEXT,
  FLAPPY_HEADER_CANVAS_HEIGHT_PX,
  FLAPPY_UI_CANVAS_INSET_SHADOW,
  FLAPPY_UI_CONTENT_COLUMN_TOP_PADDING_PX,
  FLAPPY_UI_DOUBLE_PANEL_BORDER,
  FLAPPY_UI_NETWORK_CANVAS_BACKGROUND,
  FLAPPY_UI_NETWORK_HOST_FIXED_HEIGHT_PX,
  FLAPPY_UI_NETWORK_HOST_INSET_PX,
  FLAPPY_UI_NETWORK_HOST_INITIAL_HEIGHT_PX,
  FLAPPY_UI_NETWORK_HOST_BACKGROUND,
  FLAPPY_UI_OUTER_FRAME_MIN_SIDE_PADDING_PX,
  FLAPPY_UI_OUTER_FRAME_SIDE_PADDING_OFFSET_PX,
  FLAPPY_UI_OUTER_FRAME_BACKGROUND,
  FLAPPY_UI_UNIFIED_INSET_SHADOW,
  FLAPPY_NEON_PALETTE,
  FLAPPY_SCREEN_PADDING_PX,
  FLAPPY_VIEWPORT_VERTICAL_LAYOUT_GUTTER_PX,
} from '../constants/constants';
import type {
  FlappyStatsKey,
  FlappyStatsTableCells,
  NetworkVisualizationHandle,
} from './browser-entry.types';
import {
  FLAPPY_HOST_PANEL_PADDING,
  FLAPPY_HOST_PANEL_TRANSITION,
  FLAPPY_HOST_STATS_SPLIT_GAP,
  FLAPPY_HOST_TABLE_HOST_PADDING,
} from './host/host.constants';
import {
  applyCanvasBackingSize,
  resolveNetworkCanvasSizePx,
} from './host/host.canvas.service';
import { resolveRequiredCanvas2dContext } from './host/host.dom.service';
import { installResponsiveViewportSizing } from './host/host.resize.service';
import {
  createAndAttachHostStatsTable,
  updateStatsTableValues,
} from './host/host.stats.service';
import type { CanvasHostResult } from './host/host.types';
import {
  FLAPPY_NETWORK_INPUT_SIZE,
  FLAPPY_NETWORK_OUTPUT_SIZE,
} from '../constants/constants';

/**
 * Builds the browser demo host tree and returns rendering handles.
 *
 * @param containerElement - Root host container.
 * @returns Canvas handles, stats cells and network render callback.
 */
export function createCanvasHostInternal(
  containerElement: HTMLElement,
): CanvasHostResult {
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

  // Step 2: Create top-level layout containers (content column + responsive main split).
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

  const headerContext = resolveRequiredCanvas2dContext(
    headerCanvas,
    'Header canvas 2D context unavailable',
  );

  // Step 4: Create main simulation canvas.
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
  canvas.style.border = unifiedBorder;
  canvas.style.boxShadow = FLAPPY_UI_CANVAS_INSET_SHADOW;

  const context = resolveRequiredCanvas2dContext(
    canvas,
    'Canvas 2D context unavailable',
  );

  // Step 5: Create stats/network split pane container.
  const statsContainer = document.createElement('div');
  statsContainer.style.marginTop = '0';
  statsContainer.style.boxSizing = 'border-box';
  statsContainer.style.background = FLAPPY_NEON_PALETTE.hudPanelBackground;
  statsContainer.style.border = unifiedBorder;
  statsContainer.style.boxShadow = unifiedInsetShadow;
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
  statsTableHost.style.border = unifiedBorder;
  statsTableHost.style.padding = FLAPPY_HOST_TABLE_HOST_PADDING;

  // Step 6: Create stats table and initialize value cells.
  const statsValueByKey = createAndAttachHostStatsTable(statsTableHost);

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

  const networkContext = resolveRequiredCanvas2dContext(
    networkCanvas,
    'Network canvas 2D context unavailable',
  );

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
    const { widthPx, heightPx } = resolveNetworkCanvasSizePx(
      networkCanvasHost,
      FLAPPY_UI_NETWORK_HOST_INSET_PX,
    );
    applyCanvasBackingSize(networkCanvas, widthPx, heightPx);
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
      const fixedNetworkPanelHeightPx = FLAPPY_UI_NETWORK_HOST_FIXED_HEIGHT_PX;
      networkCanvasHost.dataset.preferredHeightPx = String(
        fixedNetworkPanelHeightPx,
      );
      networkCanvasHost.style.height = `${fixedNetworkPanelHeightPx}px`;
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
  mainSplitContainer.appendChild(canvas);
  mainSplitContainer.appendChild(statsContainer);
  contentColumn.appendChild(mainSplitContainer);
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
    mainSplitContainer,
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
export const updateStatsTableValuesInternal = updateStatsTableValues;
