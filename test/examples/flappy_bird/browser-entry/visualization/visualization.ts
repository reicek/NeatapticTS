import Network from '../../../../../src/architecture/network';
import type { VisualNetworkNodeLike } from '../browser-entry.types';
import { resolveNetworkVisualizationLayers as resolveNetworkVisualizationLayersFromTopology } from './visualization.topology.utils';

/**
 * Public visualization entry for resolving layered network node groups.
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
  return resolveNetworkVisualizationLayersFromTopology(
    network,
    inputSize,
    outputSize,
  );
}
