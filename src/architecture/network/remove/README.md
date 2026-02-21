# architecture/network/remove

## architecture/network/remove/network.remove.utils.ts

### clearConnectionGater

`(candidateConnection: import("C:/NeatapticTS/src/architecture/connection").default) => void`

Clears gater reference so legacy checks treat connection as ungated.

Parameters:
- `candidateConnection` - - Connection to clear.

Returns: Nothing.

### cloneInboundConnections

`(targetNode: import("C:/NeatapticTS/src/architecture/node").default) => import("C:/NeatapticTS/src/architecture/connection").default[]`

Clones inbound connections for safe traversal after mutation.

Parameters:
- `targetNode` - - Node being removed.

Returns: Inbound connection snapshot.

### cloneOutboundConnections

`(targetNode: import("C:/NeatapticTS/src/architecture/node").default) => import("C:/NeatapticTS/src/architecture/connection").default[]`

Clones outbound connections for safe traversal after mutation.

Parameters:
- `targetNode` - - Node being removed.

Returns: Outbound connection snapshot.

### collectReconnectEndpointPairs

`(snapshotContext: import("C:/NeatapticTS/src/architecture/network/network.types").NodeConnectionSnapshotContext) => import("C:/NeatapticTS/src/architecture/network/network.types").ReconnectEndpointPairContext[]`

Collects all valid source/target reconnect endpoint pairs.

Parameters:
- `snapshotContext` - - Immutable adjacency snapshot.

Returns: Valid reconnect endpoint pairs.

### connectPairWhenMissing

`(network: import("C:/NeatapticTS/src/architecture/network").default, reconnectPair: import("C:/NeatapticTS/src/architecture/network/network.types").ReconnectEndpointPairContext) => void`

Connects one endpoint pair only when direct edge does not already exist.

Parameters:
- `network` - - Target network.
- `reconnectPair` - - Source/target pair.

Returns: Nothing.

### countSelfConnections

`(targetNode: import("C:/NeatapticTS/src/architecture/node").default) => number`

Counts self-loop connections currently attached to node.

Parameters:
- `targetNode` - - Node being removed.

Returns: Self-loop count.

### createNodeConnectionSnapshot

`(removalContext: import("C:/NeatapticTS/src/architecture/network/network.types").NodeRemovalContext) => import("C:/NeatapticTS/src/architecture/network/network.types").NodeConnectionSnapshotContext`

Creates immutable snapshots of node adjacency lists before mutation.

Parameters:
- `removalContext` - - Immutable removal context.

Returns: Snapshot context.

### createReconnectEndpointPair

`(inboundConnection: import("C:/NeatapticTS/src/architecture/connection").default, outboundConnection: import("C:/NeatapticTS/src/architecture/connection").default) => import("C:/NeatapticTS/src/architecture/network/network.types").ReconnectEndpointPairContext | undefined`

Creates one reconnect endpoint pair when endpoints are valid.

Parameters:
- `inboundConnection` - - Inbound edge from snapshot.
- `outboundConnection` - - Outbound edge from snapshot.

Returns: Reconnect pair or undefined.

### createValidatedNodeRemovalContext

`(network: import("C:/NeatapticTS/src/architecture/network").default, targetNode: import("C:/NeatapticTS/src/architecture/node").default) => import("C:/NeatapticTS/src/architecture/network/network.types").NodeRemovalContext`

Creates validated immutable context for a node-removal operation.

Parameters:
- `network` - - Target network.
- `targetNode` - - Node requested for removal.

Returns: Validated removal context.

### detachGatesOwnedByNode

`(removalContext: import("C:/NeatapticTS/src/architecture/network/network.types").NodeRemovalContext) => void`

Removes gate records gated by target node and nulls their gater field.

Parameters:
- `removalContext` - - Immutable removal context.

Returns: Nothing.

### disconnectAllNodeConnections

`(removalContext: import("C:/NeatapticTS/src/architecture/network/network.types").NodeRemovalContext, snapshotContext: import("C:/NeatapticTS/src/architecture/network/network.types").NodeConnectionSnapshotContext) => void`

Disconnects all inbound, outbound, and self-loop edges for removed node.

Parameters:
- `removalContext` - - Immutable removal context.
- `snapshotContext` - - Immutable adjacency snapshot.

Returns: Nothing.

### disconnectConnectionGroup

`(network: import("C:/NeatapticTS/src/architecture/network").default, connectionsToDisconnect: import("C:/NeatapticTS/src/architecture/connection").default[]) => void`

Disconnects each connection in a single connection list.

Parameters:
- `network` - - Target network.
- `connectionsToDisconnect` - - Connection list.

Returns: Nothing.

### disconnectSelfLoops

`(network: import("C:/NeatapticTS/src/architecture/network").default, targetNode: import("C:/NeatapticTS/src/architecture/node").default, selfConnectionCount: number) => void`

Disconnects node self-loop connections using deterministic count traversal.

Parameters:
- `network` - - Target network.
- `targetNode` - - Node whose self-loop is removed.
- `selfConnectionCount` - - Number of self-loops to remove.

Returns: Nothing.

### doesDirectConnectionExist

`(network: import("C:/NeatapticTS/src/architecture/network").default, reconnectPair: import("C:/NeatapticTS/src/architecture/network/network.types").ReconnectEndpointPairContext) => boolean`

Checks whether a direct connection already exists for reconnect pair.

Parameters:
- `network` - - Target network.
- `reconnectPair` - - Source/target pair.

Returns: True when direct edge already exists.

### ensureNodeIsNotStructuralAnchor

`(targetNode: import("C:/NeatapticTS/src/architecture/node").default) => void`

Ensures removal target is not an input/output anchor node.

Parameters:
- `targetNode` - - Node under validation.

Returns: Nothing.

### isGatedByRemovedNode

`(candidateConnection: import("C:/NeatapticTS/src/architecture/connection").default, removedNode: import("C:/NeatapticTS/src/architecture/node").default) => boolean`

Checks whether a gate candidate is currently gated by removed node.

Parameters:
- `candidateConnection` - - Gate candidate.
- `removedNode` - - Removed node reference.

Returns: True when removed node is gater.

### isReconnectPairValid

`(inboundConnection: import("C:/NeatapticTS/src/architecture/connection").default, outboundConnection: import("C:/NeatapticTS/src/architecture/connection").default) => boolean`

Validates reconnect pair endpoints.

Parameters:
- `inboundConnection` - - Inbound edge from snapshot.
- `outboundConnection` - - Outbound edge from snapshot.

Returns: True when reconnect pair should be attempted.

### isStructuralAnchorNode

`(targetNode: import("C:/NeatapticTS/src/architecture/node").default) => boolean`

Checks whether node is an input/output structural anchor.

Parameters:
- `targetNode` - - Node under evaluation.

Returns: True when node is an anchor.

### keepGateConnectionAfterNodeRemoval

`(candidateConnection: import("C:/NeatapticTS/src/architecture/connection").default, removedNode: import("C:/NeatapticTS/src/architecture/node").default) => boolean`

Filters one gate connection while clearing removed-node gater ownership.

Parameters:
- `candidateConnection` - - Gate candidate.
- `removedNode` - - Removed node reference.

Returns: True when gate should remain in list.

### markNetworkRemovalDirtyFlags

`(internalNetwork: import("C:/NeatapticTS/src/architecture/network/network.types").NetworkRemoveProps) => void`

Marks all cached removal-sensitive structures as dirty.

Parameters:
- `internalNetwork` - - Internal mutable network props.

Returns: Nothing.

### reconnectBridgedPaths

`(removalContext: import("C:/NeatapticTS/src/architecture/network/network.types").NodeRemovalContext, snapshotContext: import("C:/NeatapticTS/src/architecture/network/network.types").NodeConnectionSnapshotContext) => void`

Reconnects paths from former inbound sources to former outbound targets.

Parameters:
- `removalContext` - - Immutable removal context.
- `snapshotContext` - - Immutable adjacency snapshot.

Returns: Nothing.

### releaseRemovedNodeWhenPoolingEnabled

`(removedNode: import("C:/NeatapticTS/src/architecture/node").default | undefined) => void`

Releases removed node to object pool when pooling is enabled.

Parameters:
- `removedNode` - - Removed node instance.

Returns: Nothing.

### removeNode

`(node: import("C:/NeatapticTS/src/architecture/node").default) => void`

Node removal utilities.

This module provides a focused implementation for removing a single hidden node from a network
while attempting to preserve overall functional connectivity. The removal procedure mirrors the
legacy Neataptic logic but augments it with clearer documentation and explicit invariants.

High‑level algorithm (removeNode):
 1. Guard: ensure the node exists and is not an input or output (those are structural anchors).
 2. Ungate: detach any connections gated BY the node (we don't currently reassign gater roles).
 3. Snapshot inbound / outbound connections (before mutation of adjacency lists).
 4. Disconnect all inbound, outbound, and self connections.
 5. Physically remove the node from the network's node array.
 6. Simple path repair heuristic: for every former inbound source and outbound target, add a
    direct connection if (a) both endpoints still exist, (b) they are distinct, and (c) no
    direct connection already exists. This keeps forward information flow possibilities.
 7. Mark topology / caches dirty so that subsequent activation / ordering passes rebuild state.

Notes / Limitations:
 - We do NOT attempt to clone weights or distribute the removed node's function across new
   connections (more sophisticated strategies could average or compose weights).
 - Gating effects involving the removed node as a gater are dropped; downstream behavior may
   change—callers relying heavily on gating may want a custom remap strategy.
 - Self connections are simply removed; no attempt is made to emulate recursion via alternative
   structures.

### removeNodeFromNetworkStorage

`(removalContext: import("C:/NeatapticTS/src/architecture/network/network.types").NodeRemovalContext) => void`

Removes node from network storage and conditionally releases it to pool.

Parameters:
- `removalContext` - - Immutable removal context.

Returns: Nothing.

### resolveNodeIndexOrThrow

`(network: import("C:/NeatapticTS/src/architecture/network").default, targetNode: import("C:/NeatapticTS/src/architecture/node").default) => number`

Resolves node index and throws when missing.

Parameters:
- `network` - - Target network.
- `targetNode` - - Node being removed.

Returns: Node index inside network list.

### spliceNodeFromNetwork

`(removalContext: import("C:/NeatapticTS/src/architecture/network/network.types").NodeRemovalContext) => import("C:/NeatapticTS/src/architecture/node").default | undefined`

Splices node out of network list using validated index.

Parameters:
- `removalContext` - - Immutable removal context.

Returns: Removed node or undefined.
