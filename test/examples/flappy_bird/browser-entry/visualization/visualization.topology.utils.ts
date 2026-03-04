import Network from '../../../../../src/architecture/network';
import type { VisualNetworkNodeLike } from '../browser-entry.types';
import {
  formatNodeBiasLabelInternal,
  resolveNetworkVisualizationLayersInternal,
} from '../browser-entry.visualization.utils';

/**
 * Formats node bias labels with fixed sign and precision.
 *
 * @param nodeBias - Node bias value.
 * @returns Label text.
 */
export function formatNodeBiasLabel(nodeBias: number): string {
  return formatNodeBiasLabelInternal(nodeBias);
}

/**
 * Resolves layered node groups for visualization.
 *
 * @param network - Runtime network instance.
 * @param inputSize - Input count fallback.
 * @param outputSize - Output count fallback.
 * @returns Layered nodes for rendering.
 */
export function resolveNetworkVisualizationLayers(
  network: Network | undefined,
  inputSize: number,
  outputSize: number,
): VisualNetworkNodeLike[][] {
  return resolveNetworkVisualizationLayersInternal(network, inputSize, outputSize);
}
