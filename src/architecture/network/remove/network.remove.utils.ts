import type Network from '../../network/network';
import type Node from '../../node';
import { createValidatedNodeRemovalContext } from './network.remove.validation.utils';
import { detachGatesOwnedByNode } from './network.remove.gates.utils';
import {
  createNodeConnectionSnapshot,
  disconnectAllNodeConnections,
} from './network.remove.snapshot.utils';
import { reconnectBridgedPaths } from './network.remove.reconnect.utils';
import {
  markNetworkRemovalDirtyFlags,
  removeNodeFromNetworkStorage,
} from './network.remove.finalize.utils';

/**
 * Node removal utilities.
 *
 * This module provides a focused implementation for removing a single hidden node from a network
 * while attempting to preserve overall functional connectivity. The removal procedure mirrors the
 * legacy Neataptic logic but augments it with clearer documentation and explicit invariants.
 *
 * High‑level algorithm (removeNode):
 *  1. Guard: ensure the node exists and is not an input or output (those are structural anchors).
 *  2. Ungate: detach any connections gated BY the node (we don't currently reassign gater roles).
 *  3. Snapshot inbound / outbound connections (before mutation of adjacency lists).
 *  4. Disconnect all inbound, outbound, and self connections.
 *  5. Physically remove the node from the network's node array.
 *  6. Simple path repair heuristic: for every former inbound source and outbound target, add a
 *     direct connection if (a) both endpoints still exist, (b) they are distinct, and (c) no
 *     direct connection already exists. This keeps forward information flow possibilities.
 *  7. Mark topology / caches dirty so that subsequent activation / ordering passes rebuild state.
 *
 * Notes / Limitations:
 *  - We do NOT attempt to clone weights or distribute the removed node's function across new
 *    connections (more sophisticated strategies could average or compose weights).
 *  - Gating effects involving the removed node as a gater are dropped; downstream behavior may
 *    change—callers relying heavily on gating may want a custom remap strategy.
 *  - Self connections are simply removed; no attempt is made to emulate recursion via alternative
 *    structures.
 */

/**
 * Remove a hidden node from the network while minimally repairing connectivity.
 *
 * @param this Network instance (bound implicitly via method-style call).
 * @param node The node object to remove (must be of type 'hidden').
 * @throws If the node is not present or is an input / output node.
 *
 * Side Effects:
 *  - Mutates network.nodes, network.connections (via disconnect/connect calls), and network.gates.
 *  - Marks internal dirty flags so that future activation / ordering passes recompute derived state.
 */
export function removeNode(this: Network, node: Node) {
  const removalContext = createValidatedNodeRemovalContext(this, node);

  // Step 1: Remove gate assignments owned by the removed node.
  detachGatesOwnedByNode(removalContext);

  // Step 2: Snapshot connection neighborhoods before mutating graph edges.
  const snapshotContext = createNodeConnectionSnapshot(removalContext);

  // Step 3: Disconnect all edges touching the removed node.
  disconnectAllNodeConnections(removalContext, snapshotContext);

  // Step 4: Remove the node from storage and release pooled instance when enabled.
  removeNodeFromNetworkStorage(removalContext);

  // Step 5: Reconnect source-to-target paths that were bridged through the removed node.
  reconnectBridgedPaths(removalContext, snapshotContext);

  // Step 6: Mark all cached derived graph structures dirty.
  markNetworkRemovalDirtyFlags(removalContext.internalNetwork);
}

export default { removeNode };
