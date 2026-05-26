import type Node from '../../node';
import { releaseNode as _releaseNode } from '../../nodePool/nodePool';
import { defaultMemoryManager } from '../../../memory/manager';
import type {
  NetworkRemoveProps,
  NodeRemovalContext,
} from './network.remove.utils.types';
import { FIRST_REMOVED_NODE_INDEX } from './network.remove.utils.types';

/**
 * Removes node from network storage and conditionally releases it to pool.
 *
 * @param removalContext - Immutable removal context.
 * @returns Nothing.
 */
export function removeNodeFromNetworkStorage(
  removalContext: NodeRemovalContext,
): void {
  const removedNode = spliceNodeFromNetwork(removalContext);
  releaseRemovedNodeWhenPoolingEnabled(removedNode);
}

/**
 * Marks all cached removal-sensitive network structures as dirty after node removal.
 *
 * @param internalNetwork - Internal mutable network props.
 * @returns Nothing.
 */
export function markNetworkRemovalDirtyFlags(
  internalNetwork: NetworkRemoveProps,
): void {
  internalNetwork._topoDirty = true;
  internalNetwork._nodeIndexDirty = true;
  internalNetwork._slabDirty = true;
  internalNetwork._adjDirty = true;
}

/**
 * Splices node out of network list using validated index.
 *
 * @param removalContext - Immutable removal context.
 * @returns Removed node or undefined.
 */
function spliceNodeFromNetwork(
  removalContext: NodeRemovalContext,
): Node | undefined {
  return removalContext.network.nodes.splice(removalContext.targetNodeIndex, 1)[
    FIRST_REMOVED_NODE_INDEX
  ];
}

/**
 * Releases removed node to object pool when pooling is enabled.
 *
 * @param removedNode - Removed node instance.
 * @returns Nothing.
 */
function releaseRemovedNodeWhenPoolingEnabled(
  removedNode: Node | undefined,
): void {
  const memoryConfig = defaultMemoryManager.getConfig();

  if (!memoryConfig.enableNodePooling || !removedNode) {
    return;
  }
  _releaseNode(removedNode);
}
