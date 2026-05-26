import type Network from '../../network/network';
import type Node from '../../node';
import {
  NetworkRemoveNodeNotFoundError,
  NetworkRemoveStructuralAnchorError,
} from './network.remove.errors';
import type { NodeRemovalContext } from './network.remove.utils.types';
import {
  ERROR_CANNOT_REMOVE_ANCHOR_NODE,
  ERROR_NODE_NOT_IN_NETWORK,
  NODE_NOT_FOUND_INDEX,
  NODE_TYPE_INPUT,
  NODE_TYPE_OUTPUT,
} from './network.remove.utils.types';

/**
 * Create a validated immutable context object for one node-removal operation.
 *
 * @param network - Target network.
 * @param targetNode - Node requested for removal.
 * @returns Validated removal context.
 */
export function createValidatedNodeRemovalContext(
  network: Network,
  targetNode: Node,
): NodeRemovalContext {
  const targetNodeIndex = resolveNodeIndexOrThrow(network, targetNode);
  ensureNodeIsNotStructuralAnchor(targetNode);

  return {
    network,
    internalNetwork:
      network as unknown as NodeRemovalContext['internalNetwork'],
    targetNode,
    targetNodeIndex,
  };
}

/**
 * Resolves node index and throws when missing.
 *
 * @param network - Target network.
 * @param targetNode - Node being removed.
 * @returns Node index inside network list.
 */
function resolveNodeIndexOrThrow(network: Network, targetNode: Node): number {
  const targetNodeIndex = network.nodes.indexOf(targetNode);
  if (targetNodeIndex === NODE_NOT_FOUND_INDEX) {
    throw new NetworkRemoveNodeNotFoundError(ERROR_NODE_NOT_IN_NETWORK);
  }
  return targetNodeIndex;
}

/**
 * Ensures removal target is not an input/output anchor node.
 *
 * @param targetNode - Node under validation.
 * @returns Nothing.
 */
function ensureNodeIsNotStructuralAnchor(targetNode: Node): void {
  if (isStructuralAnchorNode(targetNode)) {
    throw new NetworkRemoveStructuralAnchorError(
      ERROR_CANNOT_REMOVE_ANCHOR_NODE,
    );
  }
}

/**
 * Checks whether node is an input/output structural anchor.
 *
 * @param targetNode - Node under evaluation.
 * @returns True when node is an anchor.
 */
function isStructuralAnchorNode(targetNode: Node): boolean {
  return (
    targetNode.type === NODE_TYPE_INPUT || targetNode.type === NODE_TYPE_OUTPUT
  );
}
