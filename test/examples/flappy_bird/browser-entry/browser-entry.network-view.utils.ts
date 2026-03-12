/**
 * Compatibility facade for browser-entry network-view helpers.
 *
 * Legacy imports still use this file while the network-view subsystem is split
 * into smaller topology, layout, label, and drawing modules.
 */
export {
  drawNetworkVisualization as drawNetworkVisualizationInternal,
  resolveNetworkArchitectureLabel as resolveNetworkArchitectureLabelInternal,
  resolveNetworkVisualizationHeightPx as resolveNetworkVisualizationHeightPxInternal,
} from './network-view/network-view';
export { resolveNetworkVisualizationLayers } from './network-view/network-view.topology.utils';
