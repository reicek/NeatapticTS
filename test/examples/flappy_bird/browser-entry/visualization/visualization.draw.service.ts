import type {
  PositionedNetworkNodeLike,
  VisualNetworkConnectionLike,
  NetworkNodeDimensionsLike,
} from '../browser-entry.types';
import type {
  DynamicColorScale,
  NetworkVisualizationColorScales,
} from './visualization.types';
import {
  drawBiasNodesLayerInternal,
  drawNetworkColorLegendInternal,
  drawNetworkVisualizationHeaderInternal,
  drawWeightedConnectionsLayerInternal,
} from '../browser-entry.visualization.utils';

/**
 * Draws weighted connection lines.
 */
export function drawWeightedConnectionsLayer(
  context: CanvasRenderingContext2D,
  runtimeConnections: VisualNetworkConnectionLike[],
  positionByNodeIndex: Map<number, PositionedNetworkNodeLike>,
  connectionScale: DynamicColorScale,
): void {
  drawWeightedConnectionsLayerInternal(
    context,
    runtimeConnections,
    positionByNodeIndex,
    connectionScale,
  );
}

/**
 * Draws all network nodes with bias labels.
 */
export function drawBiasNodesLayer(
  context: CanvasRenderingContext2D,
  positionedNodes: PositionedNetworkNodeLike[],
  nodeDimensions: NetworkNodeDimensionsLike,
  biasScale: DynamicColorScale,
): void {
  drawBiasNodesLayerInternal(
    context,
    positionedNodes,
    nodeDimensions,
    biasScale,
  );
}

/**
 * Draws network architecture header text.
 */
export function drawNetworkVisualizationHeader(
  context: CanvasRenderingContext2D,
  architectureLabel: string,
): void {
  drawNetworkVisualizationHeaderInternal(context, architectureLabel);
}

/**
 * Draws the color legend for connections and node bias values.
 */
export function drawNetworkColorLegend(
  context: CanvasRenderingContext2D,
  architectureLabel: string,
  colorScales: NetworkVisualizationColorScales,
): void {
  drawNetworkColorLegendInternal(context, architectureLabel, colorScales);
}
