# architecture/network/gating

## architecture/network/gating/network.gating.utils.ts

### assertGaterNodeBelongsToNetwork

`(network: import("C:/NeatapticTS/src/architecture/network").default, node: import("C:/NeatapticTS/src/architecture/node").default) => void`

Validate that a candidate gater node belongs to the target network.

Parameters:
- `network` - - Network performing the gating operation.
- `node` - - Candidate gater node.

Returns: Nothing.

### assertNodeRemovableAndGetIndex

`(network: import("C:/NeatapticTS/src/architecture/network").default, node: import("C:/NeatapticTS/src/architecture/node").default) => number`

Ensure a node can be removed and return its index in the network node list.

Parameters:
- `network` - - Network containing the node.
- `node` - - Node to validate.

Returns: Index of the node in the network node list.

### attachGaterToConnection

`(network: import("C:/NeatapticTS/src/architecture/network").default, node: import("C:/NeatapticTS/src/architecture/node").default, connection: import("C:/NeatapticTS/src/architecture/connection").default) => void`

Attach a gater to a connection and track that connection in the network gate list.

Parameters:
- `network` - - Network being updated.
- `node` - - Gater node to attach.
- `connection` - - Connection to gate.

Returns: Nothing.

### createBridgingConnections

`(network: import("C:/NeatapticTS/src/architecture/network").default, predecessorNodes: import("C:/NeatapticTS/src/architecture/node").default[], successorNodes: import("C:/NeatapticTS/src/architecture/node").default[]) => import("C:/NeatapticTS/src/architecture/connection").default[]`

Create bridging connections from each predecessor to each successor when valid.

Parameters:
- `network` - - Network where bridge connections are created.
- `predecessorNodes` - - Source nodes collected from inbound edges.
- `successorNodes` - - Target nodes collected from outbound edges.

Returns: Newly created bridge connections.

### detachConnectionFromGater

`(connection: import("C:/NeatapticTS/src/architecture/connection").default) => void`

Remove reverse gater bookkeeping from a connection's gater node.

Parameters:
- `connection` - - Connection to detach from its gater.

Returns: Nothing.

### disconnectInboundConnections

`(network: import("C:/NeatapticTS/src/architecture/network").default, node: import("C:/NeatapticTS/src/architecture/node").default, preservedGaters: import("C:/NeatapticTS/src/architecture/node").default[], subNodeConfig: import("C:/NeatapticTS/src/architecture/network/network.types").SubNodeMutationConfig | undefined) => import("C:/NeatapticTS/src/architecture/node").default[]`

Disconnect all inbound connections for a node while collecting predecessors.

Parameters:
- `network` - - Network being updated.
- `node` - - Node being removed.
- `preservedGaters` - - Mutable collection for gaters that should be reassigned.
- `subNodeConfig` - - Current SUB_NODE mutation settings.

Returns: Predecessor nodes that previously projected into the removed node.

### disconnectNodeSelfLoop

`(network: import("C:/NeatapticTS/src/architecture/network").default, node: import("C:/NeatapticTS/src/architecture/node").default) => void`

Disconnect a node self-loop before broader edge rewiring.

Parameters:
- `network` - - Network being updated.
- `node` - - Node whose self-loop should be removed.

Returns: Nothing.

### disconnectOutboundConnections

`(network: import("C:/NeatapticTS/src/architecture/network").default, node: import("C:/NeatapticTS/src/architecture/node").default, preservedGaters: import("C:/NeatapticTS/src/architecture/node").default[], subNodeConfig: import("C:/NeatapticTS/src/architecture/network/network.types").SubNodeMutationConfig | undefined) => import("C:/NeatapticTS/src/architecture/node").default[]`

Disconnect all outbound connections for a node while collecting successors.

Parameters:
- `network` - - Network being updated.
- `node` - - Node being removed.
- `preservedGaters` - - Mutable collection for gaters that should be reassigned.
- `subNodeConfig` - - Current SUB_NODE mutation settings.

Returns: Successor nodes that were previously targeted by the removed node.

### findGateIndex

`(network: import("C:/NeatapticTS/src/architecture/network").default, connection: import("C:/NeatapticTS/src/architecture/connection").default) => number`

Find a connection position within the network global gates list.

Parameters:
- `network` - - Network containing global gate references.
- `connection` - - Connection to locate.

Returns: Zero-based index in the gates list, or -1 when absent.

### gate

`(node: import("C:/NeatapticTS/src/architecture/node").default, connection: import("C:/NeatapticTS/src/architecture/connection").default) => void`

Attach a gater node to a connection so that the connection's effective weight
becomes dynamically modulated by the gater's activation (see {@link Node.gate} for exact math).

Validation / invariants:
 - Throws if the gater node is not part of this network (prevents cross-network corruption).
 - If the connection is already gated, function is a no-op (emits warning when enabled).

Complexity: O(1)

Parameters:
- `this` - - Bound  {@link Network} instance.
 *
- `node` - - Candidate gater node (must belong to network).
- `connection` - - Connection to gate.

### isConnectionAlreadyGated

`(connection: import("C:/NeatapticTS/src/architecture/connection").default) => boolean`

Determine whether a connection already has a gater node assigned.

Parameters:
- `connection` - - Connection candidate.

Returns: True when a gater is already set; otherwise false.

### preserveGaterForReassignment

`(connection: import("C:/NeatapticTS/src/architecture/connection").default, removedNode: import("C:/NeatapticTS/src/architecture/node").default, preservedGaters: import("C:/NeatapticTS/src/architecture/node").default[], subNodeConfig: import("C:/NeatapticTS/src/architecture/network/network.types").SubNodeMutationConfig | undefined) => void`

Preserve a gater for later reassignment when gate retention is enabled.

Parameters:
- `connection` - - Connection being detached.
- `removedNode` - - Node currently being removed.
- `preservedGaters` - - Mutable list of gaters to keep.
- `subNodeConfig` - - Current SUB_NODE mutation settings.

Returns: Nothing.

### reassignPreservedGaters

`(network: import("C:/NeatapticTS/src/architecture/network").default, preservedGaters: import("C:/NeatapticTS/src/architecture/node").default[], bridgingConnections: import("C:/NeatapticTS/src/architecture/connection").default[]) => void`

Reattach preserved gaters to randomly selected newly-created bridge connections.

Parameters:
- `network` - - Network performing reassignment.
- `preservedGaters` - - Gaters retained during node removal.
- `bridgingConnections` - - Available bridge connections for reassignment.

Returns: Nothing.

### removeGateAtIndex

`(network: import("C:/NeatapticTS/src/architecture/network").default, gateIndex: number) => void`

Remove a gated connection from the network global gate list.

Parameters:
- `network` - - Network being updated.
- `gateIndex` - - Index to remove from the gate list.

Returns: Nothing.

### removeNode

`(node: import("C:/NeatapticTS/src/architecture/node").default) => void`

Remove a hidden node from the network while attempting to preserve functional connectivity.

Algorithm outline:
 1. Reject removal if node is input/output (structural invariants) or absent (error).
 2. Optionally collect gating nodes (if keep_gates flag) from inbound & outbound connections.
 3. Remove self-loop (if present) to simplify subsequent edge handling.
 4. Disconnect all inbound edges (record their source nodes) and all outbound edges (record targets).
 5. For every (input predecessor, output successor) pair create a new connection unless:
      a. input === output (avoid trivial self loops) OR
      b. an existing projection already connects them.
 6. Reassign preserved gater nodes randomly onto newly created bridging connections.
 7. Ungate any connections that were gated BY this node (where node acted as gater).
 8. Remove node from network node list and flag node index cache as dirty.

Complexity summary:
 - Let I = number of inbound edges, O = number of outbound edges.
 - Disconnect phase: O(I + O)
 - Bridging phase: O(I * O) connection existence checks (isProjectingTo) + potential additions.
 - Gater reassignment: O(min(G, newConnections)) where G is number of preserved gaters.

Preservation rationale:
 - Reassigning gaters maintains some of the dynamic modulation capacity that would otherwise
   be lost, aiding continuity during topology simplification.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *
- `node` - - Hidden node to remove.

### removeNodeAtIndex

`(network: import("C:/NeatapticTS/src/architecture/network").default, nodeIndex: number) => void`

Remove a node from the network list and mark node indexing as dirty.

Parameters:
- `network` - - Network being mutated.
- `nodeIndex` - - Index of the node to remove.

Returns: Nothing.

### resolveSubNodeMutationConfig

`() => import("C:/NeatapticTS/src/architecture/network/network.types").SubNodeMutationConfig | undefined`

Resolve the active SUB_NODE mutation configuration shape.

Returns: Normalized SUB_NODE config when available; otherwise undefined.

### selectRandomIndex

`(length: number) => number`

Select a uniformly random integer index in the range [0, length).

Parameters:
- `length` - - Upper bound (exclusive).

Returns: Random zero-based index.

### shouldCreateBridgeConnection

`(predecessorNode: import("C:/NeatapticTS/src/architecture/node").default, successorNode: import("C:/NeatapticTS/src/architecture/node").default) => boolean`

Decide whether a predecessor-successor pair should receive a bridge connection.

Parameters:
- `predecessorNode` - - Candidate source node.
- `successorNode` - - Candidate target node.

Returns: True when the pair is distinct and no projection already exists.

### ungate

`(connection: import("C:/NeatapticTS/src/architecture/connection").default) => void`

Remove gating from a connection, restoring its static weight contribution.

Idempotent: If the connection is not currently gated, the call performs no structural changes
(and optionally logs a warning). After ungating, the connection's weight will be used directly
without modulation by a gater activation.

Complexity: O(n) where n = number of gated connections (indexOf lookup) – typically small.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *
- `connection` - - Connection to ungate.

### ungateConnectionsGatedByNode

`(network: import("C:/NeatapticTS/src/architecture/network").default, node: import("C:/NeatapticTS/src/architecture/node").default) => void`

Ungate all connections that are currently gated by the removed node.

Parameters:
- `network` - - Network performing ungate operations.
- `node` - - Node whose gated connections should be released.

Returns: Nothing.

### warnConnectionAlreadyGated

`() => void`

Emit a warning when a gate operation is skipped due to existing gating.

Returns: Nothing.

### warnMissingGateConnection

`() => void`

Emit a warning when an ungate request targets a non-tracked connection.

Returns: Nothing.
