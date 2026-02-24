import type Connection from '../../connection';
import type Node from '../../node';
import type { NodeRemovalContext } from './network.remove.utils.types';

/**
 * Removes gate records gated by target node and nulls their gater field.
 *
 * @param removalContext - Immutable removal context.
 * @returns Nothing.
 */
export function detachGatesOwnedByNode(
  removalContext: NodeRemovalContext,
): void {
  removalContext.network.gates = removalContext.network.gates.filter(
    (candidateConnection) =>
      keepGateConnectionAfterNodeRemoval(
        candidateConnection,
        removalContext.targetNode,
      ),
  );
}

/**
 * Filters one gate connection while clearing removed-node gater ownership.
 *
 * @param candidateConnection - Gate candidate.
 * @param removedNode - Removed node reference.
 * @returns True when gate should remain in list.
 */
function keepGateConnectionAfterNodeRemoval(
  candidateConnection: Connection,
  removedNode: Node,
): boolean {
  if (!isGatedByRemovedNode(candidateConnection, removedNode)) {
    return true;
  }

  clearConnectionGater(candidateConnection);
  return false;
}

/**
 * Checks whether a gate candidate is currently gated by removed node.
 *
 * @param candidateConnection - Gate candidate.
 * @param removedNode - Removed node reference.
 * @returns True when removed node is gater.
 */
function isGatedByRemovedNode(
  candidateConnection: Connection,
  removedNode: Node,
): boolean {
  return candidateConnection.gater === removedNode;
}

/**
 * Clears gater reference so legacy checks treat connection as ungated.
 *
 * @param candidateConnection - Connection to clear.
 * @returns Nothing.
 */
function clearConnectionGater(candidateConnection: Connection): void {
  candidateConnection.gater = null;
}
