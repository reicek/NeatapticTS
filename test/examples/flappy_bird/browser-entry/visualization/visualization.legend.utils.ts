import Network from '../../../../../src/architecture/network';
import {
  FLAPPY_NETWORK_LEGEND_BOTTOM_PADDING_PX,
  FLAPPY_NETWORK_LEGEND_COMPACT_HEIGHT_THRESHOLD_PX,
  FLAPPY_NETWORK_LEGEND_COMPACT_ROW_HEIGHT_PX,
  FLAPPY_NETWORK_LEGEND_COMPACT_SECTION_GAP_PX,
  FLAPPY_NETWORK_LEGEND_COMPACT_SECTION_TITLE_HEIGHT_PX,
  FLAPPY_NETWORK_LEGEND_COMPACT_WIDTH_PX,
  FLAPPY_NETWORK_LEGEND_COMPACT_WIDTH_THRESHOLD_PX,
  FLAPPY_NETWORK_LEGEND_HEADER_HEIGHT_PX,
  FLAPPY_NETWORK_LEGEND_MARGIN_PX,
  FLAPPY_NETWORK_LEGEND_MIN_TOP_PX,
  FLAPPY_NETWORK_LEGEND_REGULAR_ROW_HEIGHT_PX,
  FLAPPY_NETWORK_LEGEND_REGULAR_SECTION_GAP_PX,
  FLAPPY_NETWORK_LEGEND_REGULAR_SECTION_TITLE_HEIGHT_PX,
  FLAPPY_NETWORK_LEGEND_REGULAR_WIDTH_PX,
  FLAPPY_NETWORK_LEGEND_TARGET_TOP_PX,
  FLAPPY_NETWORK_LEGEND_TOP_LEFT_THRESHOLD_PX,
} from '../../constants/constants';
import { clamp } from '../browser-entry.math.utils';
import type {
  ColorLegendRow,
  NetworkLegendLayout,
} from '../browser-entry.types';
import { assertFiniteLegendBound } from './visualization.errors';
import { resolveNetworkVisualizationColorScales } from './visualization.colors.utils';
import type { DynamicColorScale } from './visualization.types';

/**
 * Creates legend rows from ordered tiers.
 *
 * @param scale - Dynamic color scale containing bounds, tiers, and overflow color.
 * @param symbol - Label symbol.
 * @returns Legend rows.
 */
export function createColorLegendRows(
  scale: DynamicColorScale,
  symbol: 'w' | 'b',
): ColorLegendRow[] {
  return scale.tiers.map((tier, tierIndex) => {
    const lowerBound =
      tierIndex === 0
        ? scale.minimumValue
        : scale.tiers[tierIndex - 1].upperBound;
    return {
      label: `${formatLegendBound(lowerBound)} <= ${symbol} <= ${formatLegendBound(tier.upperBound)}`,
      color: tier.color,
      minimumValue: lowerBound,
      maximumValue: tier.upperBound,
    };
  });
}

/**
 * Resolves network legend layout from canvas constraints.
 *
 * @param context - Render context.
 * @param connectionLegendRows - Connection legend rows.
 * @param biasLegendRows - Bias legend rows.
 * @returns Computed legend layout.
 */
export function resolveNetworkLegendLayout(
  context: CanvasRenderingContext2D,
  connectionLegendRows: ColorLegendRow[],
  biasLegendRows: ColorLegendRow[],
): NetworkLegendLayout {
  const compactLegend =
    context.canvas.width < FLAPPY_NETWORK_LEGEND_COMPACT_WIDTH_THRESHOLD_PX ||
    context.canvas.height < FLAPPY_NETWORK_LEGEND_COMPACT_HEIGHT_THRESHOLD_PX;
  const legendWidthPx = compactLegend
    ? FLAPPY_NETWORK_LEGEND_COMPACT_WIDTH_PX
    : FLAPPY_NETWORK_LEGEND_REGULAR_WIDTH_PX;
  const legendHeaderHeightPx = FLAPPY_NETWORK_LEGEND_HEADER_HEIGHT_PX;
  const legendSectionTitleHeightPx = compactLegend
    ? FLAPPY_NETWORK_LEGEND_COMPACT_SECTION_TITLE_HEIGHT_PX
    : FLAPPY_NETWORK_LEGEND_REGULAR_SECTION_TITLE_HEIGHT_PX;
  const legendRowHeightPx = compactLegend
    ? FLAPPY_NETWORK_LEGEND_COMPACT_ROW_HEIGHT_PX
    : FLAPPY_NETWORK_LEGEND_REGULAR_ROW_HEIGHT_PX;
  const legendSectionGapPx = compactLegend
    ? FLAPPY_NETWORK_LEGEND_COMPACT_SECTION_GAP_PX
    : FLAPPY_NETWORK_LEGEND_REGULAR_SECTION_GAP_PX;
  const legendHeightPx =
    legendHeaderHeightPx +
    legendSectionTitleHeightPx +
    connectionLegendRows.length * legendRowHeightPx +
    legendSectionGapPx +
    legendSectionTitleHeightPx +
    biasLegendRows.length * legendRowHeightPx +
    FLAPPY_NETWORK_LEGEND_BOTTOM_PADDING_PX;
  const legendMarginPx = FLAPPY_NETWORK_LEGEND_MARGIN_PX;
  const preferTopLeft =
    context.canvas.width < FLAPPY_NETWORK_LEGEND_TOP_LEFT_THRESHOLD_PX;

  const maximumLeftPx = Math.max(
    legendMarginPx,
    context.canvas.width - legendWidthPx - legendMarginPx,
  );
  const legendLeftPx = preferTopLeft ? legendMarginPx : maximumLeftPx;

  const maximumTopPx = Math.max(
    legendMarginPx,
    context.canvas.height - legendHeightPx - legendMarginPx,
  );
  const minimumTopPx = Math.min(FLAPPY_NETWORK_LEGEND_MIN_TOP_PX, maximumTopPx);
  const legendTopPx = clamp(
    FLAPPY_NETWORK_LEGEND_TARGET_TOP_PX,
    minimumTopPx,
    maximumTopPx,
  );

  return {
    compactLegend,
    legendLeftPx,
    legendTopPx,
    legendWidthPx,
    legendHeightPx,
    legendHeaderHeightPx,
    legendSectionTitleHeightPx,
    legendRowHeightPx,
    legendSectionGapPx,
  };
}

/**
 * Resolves default legend layout from internal tier definitions.
 *
 * @param context - Render context.
 * @param network - Active network instance.
 * @returns Legend layout.
 */
export function resolveDefaultNetworkLegendLayout(
  context: CanvasRenderingContext2D,
  network: Network | undefined,
): NetworkLegendLayout {
  const colorScales = resolveNetworkVisualizationColorScales(network);
  const connectionLegendRows = createColorLegendRows(
    colorScales.connectionScale,
    'w',
  );
  const biasLegendRows = createColorLegendRows(colorScales.biasScale, 'b');
  return resolveNetworkLegendLayout(
    context,
    connectionLegendRows,
    biasLegendRows,
  );
}

function formatLegendBound(value: number): string {
  assertFiniteLegendBound(value);
  return value.toFixed(2);
}
