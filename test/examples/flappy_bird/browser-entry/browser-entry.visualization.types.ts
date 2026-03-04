import type Network from '../../../../src/architecture/network';

/** Draw callback contract for network architecture panel updates. */
export interface NetworkVisualizationHandle {
  renderNetworkArchitecture: (
    network: Network | undefined,
    inputSize: number,
    outputSize: number,
  ) => void;
}

/** Connection or bias tier used for color mapping ramps. */
export interface ColorTier {
  upperBound: number;
  color: string;
}

/** Legend row model for network visualization color legends. */
export interface ColorLegendRow {
  label: string;
  color: string;
  minimumValue: number;
  maximumValue: number;
}

/** Precomputed legend panel layout used by visualization renderer. */
export interface NetworkLegendLayout {
  compactLegend: boolean;
  legendLeftPx: number;
  legendTopPx: number;
  legendWidthPx: number;
  legendHeightPx: number;
  legendHeaderHeightPx: number;
  legendSectionTitleHeightPx: number;
  legendRowHeightPx: number;
  legendSectionGapPx: number;
}

/** Lightweight connection shape used by network visualization drawing. */
export interface VisualNetworkConnectionLike {
  from?: { index?: number };
  to?: { index?: number };
  weight?: number;
  enabled?: boolean;
}

/** Lightweight node shape used by network visualization drawing. */
export interface VisualNetworkNodeLike {
  index: number;
  type: string;
  bias: number;
  layer?: number;
}

/** Positioned node instance used by network visualization drawing. */
export interface PositionedNetworkNodeLike {
  node: VisualNetworkNodeLike;
  xPx: number;
  yPx: number;
}

/** Pixel dimensions used for network-node rectangle rendering. */
export interface NetworkNodeDimensionsLike {
  widthPx: number;
  heightPx: number;
}
