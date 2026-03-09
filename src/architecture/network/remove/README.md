# architecture/network/remove

## architecture/network/remove/network.remove.utils.types.ts

### ERROR_CANNOT_REMOVE_ANCHOR_NODE

### ERROR_NODE_NOT_IN_NETWORK

### FIRST_REMOVED_NODE_INDEX

### NetworkRemoveProps

Internal network properties accessed during remove operations.

### NODE_NOT_FOUND_INDEX

### NODE_TYPE_INPUT

### NODE_TYPE_OUTPUT

### NodeConnectionSnapshotContext

Snapshot of node adjacency prior to removal.

### NodeRemovalContext

Immutable context for validated node-removal request.

### ReconnectEndpointPairContext

Endpoint pair for reconnecting bridged paths.

## architecture/network/remove/network.remove.utils.ts

### removeNode

`(node: import("src/architecture/node").default) => void`

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

## architecture/network/remove/network.remove.gates.utils.ts

### clearConnectionGater

`(candidateConnection: import("src/architecture/connection").default) => void`

Clears gater reference so legacy checks treat connection as ungated.

Parameters:
- `candidateConnection` - - Connection to clear.

Returns: Nothing.

### detachGatesOwnedByNode

`(removalContext: import("src/architecture/network/remove/network.remove.utils.types").NodeRemovalContext) => void`

Removes gate records gated by target node and nulls their gater field.

Parameters:
- `removalContext` - - Immutable removal context.

Returns: Nothing.

### isGatedByRemovedNode

`(candidateConnection: import("src/architecture/connection").default, removedNode: import("src/architecture/node").default) => boolean`

Checks whether a gate candidate is currently gated by removed node.

Parameters:
- `candidateConnection` - - Gate candidate.
- `removedNode` - - Removed node reference.

Returns: True when removed node is gater.

### keepGateConnectionAfterNodeRemoval

`(candidateConnection: import("src/architecture/connection").default, removedNode: import("src/architecture/node").default) => boolean`

Filters one gate connection while clearing removed-node gater ownership.

Parameters:
- `candidateConnection` - - Gate candidate.
- `removedNode` - - Removed node reference.

Returns: True when gate should remain in list.

## architecture/network/remove/network.remove.finalize.utils.ts

### markNetworkRemovalDirtyFlags

`(internalNetwork: import("src/architecture/network/remove/network.remove.utils.types").NetworkRemoveProps) => void`

Marks all cached removal-sensitive structures as dirty.

Parameters:
- `internalNetwork` - - Internal mutable network props.

Returns: Nothing.

### releaseRemovedNodeWhenPoolingEnabled

`(removedNode: import("src/architecture/node").default | undefined) => void`

Releases removed node to object pool when pooling is enabled.

Parameters:
- `removedNode` - - Removed node instance.

Returns: Nothing.

### removeNodeFromNetworkStorage

`(removalContext: import("src/architecture/network/remove/network.remove.utils.types").NodeRemovalContext) => void`

Removes node from network storage and conditionally releases it to pool.

Parameters:
- `removalContext` - - Immutable removal context.

Returns: Nothing.

### spliceNodeFromNetwork

`(removalContext: import("src/architecture/network/remove/network.remove.utils.types").NodeRemovalContext) => import("src/architecture/node").default | undefined`

Splices node out of network list using validated index.

Parameters:
- `removalContext` - - Immutable removal context.

Returns: Removed node or undefined.

## architecture/network/remove/network.remove.snapshot.utils.ts

### cloneInboundConnections

`(targetNode: import("src/architecture/node").default) => import("src/architecture/connection").default[]`

Clones inbound connections for safe traversal after mutation.

Parameters:
- `targetNode` - - Node being removed.

Returns: Inbound connection snapshot.

### cloneOutboundConnections

`(targetNode: import("src/architecture/node").default) => import("src/architecture/connection").default[]`

Clones outbound connections for safe traversal after mutation.

Parameters:
- `targetNode` - - Node being removed.

Returns: Outbound connection snapshot.

### countSelfConnections

`(targetNode: import("src/architecture/node").default) => number`

Counts self-loop connections currently attached to node.

Parameters:
- `targetNode` - - Node being removed.

Returns: Self-loop count.

### createNodeConnectionSnapshot

`(removalContext: import("src/architecture/network/remove/network.remove.utils.types").NodeRemovalContext) => import("src/architecture/network/remove/network.remove.utils.types").NodeConnectionSnapshotContext`

Creates immutable snapshots of node adjacency lists before mutation.

Parameters:
- `removalContext` - - Immutable removal context.

Returns: Snapshot context.

### disconnectAllNodeConnections

`(removalContext: import("src/architecture/network/remove/network.remove.utils.types").NodeRemovalContext, snapshotContext: import("src/architecture/network/remove/network.remove.utils.types").NodeConnectionSnapshotContext) => void`

Disconnects all inbound, outbound, and self-loop edges for removed node.

Parameters:
- `removalContext` - - Immutable removal context.
- `snapshotContext` - - Immutable adjacency snapshot.

Returns: Nothing.

### disconnectConnectionGroup

`(network: import("src/architecture/network").default, connectionsToDisconnect: import("src/architecture/connection").default[]) => void`

Disconnects each connection in a single connection list.

Parameters:
- `network` - - Target network.
- `connectionsToDisconnect` - - Connection list.

Returns: Nothing.

### disconnectSelfLoops

`(network: import("src/architecture/network").default, targetNode: import("src/architecture/node").default, selfConnectionCount: number) => void`

Disconnects node self-loop connections using deterministic count traversal.

Parameters:
- `network` - - Target network.
- `targetNode` - - Node whose self-loop is removed.
- `selfConnectionCount` - - Number of self-loops to remove.

Returns: Nothing.

## architecture/network/remove/network.remove.reconnect.utils.ts

### collectReconnectEndpointPairs

`(snapshotContext: import("src/architecture/network/remove/network.remove.utils.types").NodeConnectionSnapshotContext) => import("src/architecture/network/remove/network.remove.utils.types").ReconnectEndpointPairContext[]`

Collects all valid source/target reconnect endpoint pairs.

Parameters:
- `snapshotContext` - - Immutable adjacency snapshot.

Returns: Valid reconnect endpoint pairs.

### connectPairWhenMissing

`(network: import("src/architecture/network").default, reconnectPair: import("src/architecture/network/remove/network.remove.utils.types").ReconnectEndpointPairContext) => void`

Connects one endpoint pair only when direct edge does not already exist.

Parameters:
- `network` - - Target network.
- `reconnectPair` - - Source/target pair.

Returns: Nothing.

### createReconnectEndpointPair

`(inboundConnection: import("src/architecture/connection").default, outboundConnection: import("src/architecture/connection").default) => import("src/architecture/network/remove/network.remove.utils.types").ReconnectEndpointPairContext | undefined`

Creates one reconnect endpoint pair when endpoints are valid.

Parameters:
- `inboundConnection` - - Inbound edge from snapshot.
- `outboundConnection` - - Outbound edge from snapshot.

Returns: Reconnect pair or undefined.

### doesDirectConnectionExist

`(network: import("src/architecture/network").default, reconnectPair: import("src/architecture/network/remove/network.remove.utils.types").ReconnectEndpointPairContext) => boolean`

Checks whether a direct connection already exists for reconnect pair.

Parameters:
- `network` - - Target network.
- `reconnectPair` - - Source/target pair.

Returns: True when direct edge already exists.

### isReconnectPairValid

`(inboundConnection: import("src/architecture/connection").default, outboundConnection: import("src/architecture/connection").default) => boolean`

Validates reconnect pair endpoints.

Parameters:
- `inboundConnection` - - Inbound edge from snapshot.
- `outboundConnection` - - Outbound edge from snapshot.

Returns: True when reconnect pair should be attempted.

### reconnectBridgedPaths

`(removalContext: import("src/architecture/network/remove/network.remove.utils.types").NodeRemovalContext, snapshotContext: import("src/architecture/network/remove/network.remove.utils.types").NodeConnectionSnapshotContext) => void`

Reconnects paths from former inbound sources to former outbound targets.

Parameters:
- `removalContext` - - Immutable removal context.
- `snapshotContext` - - Immutable adjacency snapshot.

Returns: Nothing.

## architecture/network/remove/network.remove.validation.utils.ts

### createValidatedNodeRemovalContext

`(network: import("src/architecture/network").default, targetNode: import("src/architecture/node").default) => import("src/architecture/network/remove/network.remove.utils.types").NodeRemovalContext`

Creates validated immutable context for a node-removal operation.

Parameters:
- `network` - - Target network.
- `targetNode` - - Node requested for removal.

Returns: Validated removal context.

### ensureNodeIsNotStructuralAnchor

`(targetNode: import("src/architecture/node").default) => void`

Ensures removal target is not an input/output anchor node.

Parameters:
- `targetNode` - - Node under validation.

Returns: Nothing.

### isStructuralAnchorNode

`(targetNode: import("src/architecture/node").default) => boolean`

Checks whether node is an input/output structural anchor.

Parameters:
- `targetNode` - - Node under evaluation.

Returns: True when node is an anchor.

### resolveNodeIndexOrThrow

`(network: import("src/architecture/network").default, targetNode: import("src/architecture/node").default) => number`

Resolves node index and throws when missing.

Parameters:
- `network` - - Target network.
- `targetNode` - - Node being removed.

Returns: Node index inside network list.
