import type Network from '../network';
import type Node from '../node';
import { releaseNode as _releaseNode } from '../nodePool';
import { config } from '../../config';

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
  /** Cast to any to access internal dirty flags without changing public typing. */
  const internalNet = this as any;
  /** Index of the node in the network's node array (or -1 if not found). */
  const idx = this.nodes.indexOf(node);
  if (idx === -1) throw new Error('Node not in network');
  // Structural guard: inputs/outputs are fixed anchors and cannot be removed.
  if (node.type === 'input' || node.type === 'output') {
    throw new Error('Cannot remove input or output node from the network.');
  }

  // 1. Ungate any connections gated BY this node (drop gating influence).
  this.gates = this.gates.filter((c: any) => {
    if (c.gater === node) {
      (c as any).gater = null; // explicit null so legacy checks see it as ungated
      return false; // remove from gates list
    }
    return true;
  });

  /** Snapshot of inbound connections prior to mutation for reconnection heuristic. */
  const inbound = node.connections.in.slice();
  /** Snapshot of outbound connections prior to mutation for reconnection heuristic. */
  const outbound = node.connections.out.slice();

  // 2. Disconnect all inbound connections.
  inbound.forEach((c: any) => this.disconnect(c.from, c.to));
  // 3. Disconnect all outbound connections.
  outbound.forEach((c: any) => this.disconnect(c.from, c.to));
  // 4. Disconnect self connections (if any recurrent self-loop).
  node.connections.self.slice().forEach(() => this.disconnect(node, node));

  // 5. Physically remove the node from the node list (and release to pool if enabled).
  const removed = this.nodes.splice(idx, 1)[0];
  if (config.enableNodePooling && removed) {
    _releaseNode(removed as any);
  }

  // 6. Reconnect every former inbound source to every former outbound target if a direct edge is missing.
  inbound.forEach((ic: any) => {
    outbound.forEach((oc: any) => {
      if (!ic.from || !oc.to || ic.from === oc.to) return; // skip invalid or trivial (self) cases
      /** True when a direct connection between source and target already exists. */
      const exists = this.connections.some(
        (c) => c.from === ic.from && c.to === oc.to
      );
      if (!exists) this.connect(ic.from, oc.to);
    });
  });

  // 7. Mark derived structure caches dirty so they will be recomputed lazily.
  internalNet._topoDirty = true;
  internalNet._nodeIndexDirty = true;
  internalNet._slabDirty = true;
  internalNet._adjDirty = true;
}

export default { removeNode };
