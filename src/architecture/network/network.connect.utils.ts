import type Network from '../network';
import Node from '../node';
import Connection from '../connection';

/**
 * Runtime interface for accessing Network internal properties during connection operations.
 */
interface NetworkInternals {
  _enforceAcyclic?: boolean;
  _topoDirty: boolean;
  _slabDirty: boolean;
}

/**
 * Create and register one (or multiple) directed connection objects between two nodes.
 *
 * Some node types (or future composite structures) may return several low‑level connections when
 * their {@link Node.connect} is invoked (e.g., expanded recurrent templates). For that reason this
 * function always treats the result as an array and appends each edge to the appropriate collection.
 *
 * Algorithm outline:
 *  1. (Acyclic guard) If acyclicity is enforced and the source node appears after the target node in
 *     the network's node ordering, abort early and return an empty array (prevents back‑edge creation).
 *  2. Delegate to sourceNode.connect(targetNode, weight) to build the raw Connection object(s).
 *  3. For each created connection:
 *       a. If it's a self‑connection: either ignore (acyclic mode) or store in selfconns.
 *       b. Otherwise store in standard connections array.
 *  4. If any connection was added, mark structural caches dirty (_topoDirty & _slabDirty) so lazy
 *     rebuild can occur before the next forward pass.
 *
 * Complexity:
 *  - Time: O(k) where k is the number of low‑level connections returned (typically 1).
 *  - Space: O(k) new Connection instances (delegated to Node.connect).
 *
 * Edge cases & invariants:
 *  - Acyclic mode silently refuses back‑edges instead of throwing (makes evolutionary search easier).
 *  - Self‑connections are skipped entirely when acyclicity is enforced.
 *  - Weight initialization policy is delegated to Node.connect if not explicitly provided.
 *
 * @param this - Bound {@link Network} instance.
 * @param from - Source node (emits signal).
 * @param to - Target node (receives signal).
 * @param weight - Optional explicit initial weight value.
 * @returns Array of created {@link Connection} objects (possibly empty if acyclicity rejected the edge).
 * @example
 * const [edge] = net.connect(nodeA, nodeB, 0.5);
 * @remarks For bulk layer-to-layer wiring see higher-level utilities that iterate groups.
 */
export function connect(
  this: Network,
  from: Node,
  to: Node,
  weight?: number,
): Connection[] {
  const networkInternal = this as unknown as NetworkInternals;

  // Step 1: Abort early if acyclic mode rejects this edge direction.
  if (shouldRejectConnectionForAcyclicMode(this, networkInternal, from, to))
    return [];

  // Step 2: Build low-level connection instances via node delegate.
  const createdConnections = createConnectionsFromSourceNode(from, to, weight);

  // Step 3: Register created edges in network-level collections.
  registerCreatedConnections(
    this,
    networkInternal,
    from,
    to,
    createdConnections,
  );

  // Step 4: Invalidate structural caches when connection creation occurred.
  markConnectionCachesDirtyWhenNeeded(
    networkInternal,
    createdConnections.length,
  );

  return createdConnections;

  /**
   * Determine whether an edge must be rejected to preserve acyclic ordering.
   *
   * @param network - Network instance owning node ordering.
   * @param internalState - Runtime network internals used by connection pipeline.
   * @param sourceNode - Candidate source node.
   * @param targetNode - Candidate target node.
   * @returns True when edge should be rejected.
   */
  function shouldRejectConnectionForAcyclicMode(
    network: Network,
    internalState: NetworkInternals,
    sourceNode: Node,
    targetNode: Node,
  ): boolean {
    if (!internalState._enforceAcyclic) return false;
    return (
      network.nodes.indexOf(sourceNode) > network.nodes.indexOf(targetNode)
    );
  }

  /**
   * Build one or more low-level connection objects from source node to target node.
   *
   * @param sourceNode - Source node.
   * @param targetNode - Target node.
   * @param initialWeight - Optional explicit initial weight.
   * @returns Created low-level connection objects.
   */
  function createConnectionsFromSourceNode(
    sourceNode: Node,
    targetNode: Node,
    initialWeight?: number,
  ): Connection[] {
    return sourceNode.connect(targetNode, initialWeight);
  }

  /**
   * Register created connections in either normal-connection or self-connection storage.
   *
   * @param network - Network instance owning connection collections.
   * @param internalState - Runtime network internals used by connection pipeline.
   * @param sourceNode - Source node used during connection creation.
   * @param targetNode - Target node used during connection creation.
   * @param createdConnections - Created low-level connection objects.
   * @returns Nothing.
   */
  function registerCreatedConnections(
    network: Network,
    internalState: NetworkInternals,
    sourceNode: Node,
    targetNode: Node,
    createdConnections: Connection[],
  ): void {
    const isSelfConnection = sourceNode === targetNode;

    createdConnections.forEach((createdConnection) => {
      registerSingleCreatedConnection(
        network,
        internalState,
        isSelfConnection,
        createdConnection,
      );
    });
  }

  /**
   * Register one created connection in the appropriate collection.
   *
   * @param network - Network instance owning connection collections.
   * @param internalState - Runtime network internals used by connection pipeline.
   * @param isSelfConnection - Whether source and target nodes are the same.
   * @param createdConnection - Created low-level connection object.
   * @returns Nothing.
   */
  function registerSingleCreatedConnection(
    network: Network,
    internalState: NetworkInternals,
    isSelfConnection: boolean,
    createdConnection: Connection,
  ): void {
    if (!isSelfConnection) {
      network.connections.push(createdConnection);
      return;
    }

    if (internalState._enforceAcyclic) return;
    network.selfconns.push(createdConnection);
  }

  /**
   * Mark topology and slab caches dirty when connection creation occurred.
   *
   * @param internalState - Runtime network internals used by connection pipeline.
   * @param createdConnectionCount - Number of created low-level connections.
   * @returns Nothing.
   */
  function markConnectionCachesDirtyWhenNeeded(
    internalState: NetworkInternals,
    createdConnectionCount: number,
  ): void {
    if (!createdConnectionCount) return;
    internalState._topoDirty = true;
    internalState._slabDirty = true;
  }
}

/**
 * Remove (at most) one directed connection from source 'from' to target 'to'.
 *
 * Only a single direct edge is removed because typical graph configurations maintain at most
 * one logical connection between a given pair of nodes (excluding potential future multi‑edge
 * semantics). If the target edge is gated we first call {@link Network.ungate} to maintain
 * gating invariants (ensuring the gater node's internal gate list remains consistent).
 *
 * Algorithm outline:
 *  1. Choose the correct list (selfconns vs connections) based on whether from === to.
 *  2. Linear scan to find the first edge with matching endpoints.
 *  3. If gated, ungate to detach gater bookkeeping.
 *  4. Splice the edge out; exit loop (only one expected).
 *  5. Delegate per‑node cleanup via from.disconnect(to) (clears reverse references, traces, etc.).
 *  6. Mark structural caches dirty for lazy recomputation.
 *
 * Complexity:
 *  - Time: O(m) where m is length of the searched list (connections or selfconns).
 *  - Space: O(1) extra.
 *
 * Idempotence: If no such edge exists we still perform node-level disconnect and flag caches dirty –
 * this conservative approach simplifies callers (they need not pre‑check existence).
 *
 * @param this - Bound {@link Network} instance.
 * @param from - Source node.
 * @param to - Target node.
 * @example
 * net.disconnect(nodeA, nodeB);
 * @remarks For removing many edges consider higher‑level bulk utilities to avoid repeated scans.
 */
export function disconnect(this: Network, from: Node, to: Node): void {
  const networkInternal = this as unknown as NetworkInternals;

  // Step 1: Choose the collection that can contain the target edge.
  const candidateConnections = selectConnectionCollection(this, from, to);

  // Step 2: Remove first matching edge while preserving gating invariants.
  removeFirstMatchingConnection(this, candidateConnections, from, to);

  // Step 3: Perform node-level disconnect cleanup.
  disconnectNodes(from, to);

  // Step 4: Invalidate structural caches after disconnect flow.
  markStructureCachesDirty(networkInternal);

  /**
   * Select the relevant collection to search for the edge.
   *
   * @param network - Network instance owning connection collections.
   * @param sourceNode - Source node.
   * @param targetNode - Target node.
   * @returns Candidate connection collection.
   */
  function selectConnectionCollection(
    network: Network,
    sourceNode: Node,
    targetNode: Node,
  ): Connection[] {
    return sourceNode === targetNode ? network.selfconns : network.connections;
  }

  /**
   * Remove first connection that matches source and target nodes.
   *
   * @param network - Network instance used for ungating.
   * @param candidateConnections - Candidate collection to search.
   * @param sourceNode - Source node.
   * @param targetNode - Target node.
   * @returns Nothing.
   */
  function removeFirstMatchingConnection(
    network: Network,
    candidateConnections: Connection[],
    sourceNode: Node,
    targetNode: Node,
  ): void {
    const targetConnectionIndex = findConnectionIndex(
      candidateConnections,
      sourceNode,
      targetNode,
    );

    if (targetConnectionIndex < 0) return;
    removeConnectionAtIndex(
      network,
      candidateConnections,
      targetConnectionIndex,
    );
  }

  /**
   * Find index of the first connection matching source and target nodes.
   *
   * @param candidateConnections - Candidate collection to search.
   * @param sourceNode - Source node.
   * @param targetNode - Target node.
   * @returns Matching index or -1 when no edge is found.
   */
  function findConnectionIndex(
    candidateConnections: Connection[],
    sourceNode: Node,
    targetNode: Node,
  ): number {
    return candidateConnections.findIndex(
      (candidateConnection) =>
        candidateConnection.from === sourceNode &&
        candidateConnection.to === targetNode,
    );
  }

  /**
   * Remove one connection by index, ungating first if required.
   *
   * @param network - Network instance used for ungating.
   * @param candidateConnections - Candidate collection containing target index.
   * @param targetConnectionIndex - Index to remove.
   * @returns Nothing.
   */
  function removeConnectionAtIndex(
    network: Network,
    candidateConnections: Connection[],
    targetConnectionIndex: number,
  ): void {
    const targetConnection = candidateConnections[targetConnectionIndex];
    if (targetConnection.gater) network.ungate(targetConnection);
    candidateConnections.splice(targetConnectionIndex, 1);
  }

  /**
   * Delegate per-node disconnect cleanup.
   *
   * @param sourceNode - Source node.
   * @param targetNode - Target node.
   * @returns Nothing.
   */
  function disconnectNodes(sourceNode: Node, targetNode: Node): void {
    sourceNode.disconnect(targetNode);
  }

  /**
   * Mark topology/slab caches dirty after structural mutation.
   *
   * @param internalState - Runtime network internals used by connection pipeline.
   * @returns Nothing.
   */
  function markStructureCachesDirty(internalState: NetworkInternals): void {
    internalState._topoDirty = true;
    internalState._slabDirty = true;
  }
}
