/**
 * Compatibility facade for browser-entry network visualization helpers.
 *
 * Legacy imports still flow through this file while the visualization subsystem
 * is organized into smaller, clearer modules under the dedicated folder.
 */
export {
  createLogDivergingColorTiers,
  resolveBiasRangeColor,
  resolveConnectionRangeColor,
  resolveNetworkVisualizationColorScales,
  resolveTierColor,
} from './visualization/visualization.colors.utils';
export {
  drawBiasNodesLayer,
  drawNetworkColorLegend,
  drawNetworkVisualizationHeader,
  drawWeightedConnectionsLayer,
} from './visualization/visualization.draw.service';
export {
  createColorLegendRows,
  resolveDefaultNetworkLegendLayout,
  resolveNetworkLegendLayout,
} from './visualization/visualization.legend.utils';
export {
  formatNodeActivationLabel,
  resolveNetworkVisualizationLayers,
} from './visualization/visualization.topology.utils';
