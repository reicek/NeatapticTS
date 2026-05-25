import type Network from '../../network/network';
import type Node from '../../node';
import type { NetworkSlabProps } from './network.slab.utils.types';

/**
 * Assigns sequential node indices used by slab packing and fast-path traversal.
 *
 * @param network - Target network.
 * @returns Nothing.
 */
export function _reindexNodes(network: Network): void {
  // Step 1: Iterate nodes and assign contiguous stable indices.
  const internalNet = network as unknown as NetworkSlabProps;
  for (let nodeIndex = 0; nodeIndex < network.nodes.length; nodeIndex++) {
    (
      network.nodes[nodeIndex] as unknown as Node & {
        index: number;
      }
    ).index = nodeIndex;
  }
  // Step 2: Mark node-index state as current.
  internalNet._nodeIndexDirty = false;
}
