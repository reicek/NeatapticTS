# architecture/network/connect

## architecture/network/connect/network.connect.utils.types.ts

### NetworkInternals

Internal network state shape shared by connect utility helper modules.

## architecture/network/connect/network.connect.utils.ts

### connect

```ts
connect(
  from: default,
  to: default,
  weight: number | undefined,
): default[]
```

Create and register one (or multiple) directed connection objects between two nodes.

Some node types (or future composite structures) may return several low‑level connections when
their {@link Node.connect} is invoked (e.g., expanded recurrent templates). For that reason this
function always treats the result as an array and appends each edge to the appropriate collection.

Algorithm outline:
 1. (Acyclic guard) If acyclicity is enforced and the source node appears after the target node in
    the network's node ordering, abort early and return an empty array (prevents back‑edge creation).
 2. Delegate to sourceNode.connect(targetNode, weight) to build the raw Connection object(s).
 3. For each created connection:
      a. If it's a self‑connection: either ignore (acyclic mode) or store in selfconns.
      b. Otherwise store in standard connections array.
 4. If any connection was added, mark structural caches dirty (_topoDirty & _slabDirty) so lazy
    rebuild can occur before the next forward pass.

Complexity:
 - Time: O(k) where k is the number of low‑level connections returned (typically 1).
 - Space: O(k) new Connection instances (delegated to Node.connect).

Edge cases & invariants:
 - Acyclic mode silently refuses back‑edges instead of throwing (makes evolutionary search easier).
 - Self‑connections are skipped entirely when acyclicity is enforced.
 - Weight initialization policy is delegated to Node.connect if not explicitly provided.

Parameters:
- `this` - - Bound Network instance.
- `from` - - Source node (emits signal).
- `to` - - Target node (receives signal).
- `weight` - - Optional explicit initial weight value.

Returns: Array of created  {@link Connection} objects (possibly empty if acyclicity rejected the edge).

Example:

const [edge] = net.connect(nodeA, nodeB, 0.5);

### disconnect

```ts
disconnect(
  from: default,
  to: default,
): void
```

Remove (at most) one directed connection from source 'from' to target 'to'.

Only a single direct edge is removed because typical graph configurations maintain at most
one logical connection between a given pair of nodes (excluding potential future multi‑edge
semantics). If the target edge is gated we first call {@link Network.ungate} to maintain
gating invariants (ensuring the gater node's internal gate list remains consistent).

Algorithm outline:
 1. Choose the correct list (selfconns vs connections) based on whether from === to.
 2. Linear scan to find the first edge with matching endpoints.
 3. If gated, ungate to detach gater bookkeeping.
 4. Splice the edge out; exit loop (only one expected).
 5. Delegate per‑node cleanup via from.disconnect(to) (clears reverse references, traces, etc.).
 6. Mark structural caches dirty for lazy recomputation.

Complexity:
 - Time: O(m) where m is length of the searched list (connections or selfconns).
 - Space: O(1) extra.

Idempotence: If no such edge exists we still perform node-level disconnect and flag caches dirty –
this conservative approach simplifies callers (they need not pre‑check existence).

Parameters:
- `this` - - Bound Network instance.
- `from` - - Source node.
- `to` - - Target node.

Example:

net.disconnect(nodeA, nodeB);

## architecture/network/connect/network.connect.create.utils.ts

### shouldRejectConnectionForAcyclicMode

```ts
shouldRejectConnectionForAcyclicMode(
  network: default,
  internalState: ConnectNetworkInternals,
  sourceNode: default,
  targetNode: default,
): boolean
```

Determine whether an edge must be rejected to preserve acyclic ordering.

Parameters:
- `network` - - Network instance owning node ordering.
- `internalState` - - Runtime network internals used by connection pipeline.
- `sourceNode` - - Candidate source node.
- `targetNode` - - Candidate target node.

Returns: True when edge should be rejected.

### createConnectionsFromSourceNode

```ts
createConnectionsFromSourceNode(
  sourceNode: default,
  targetNode: default,
  initialWeight: number | undefined,
): default[]
```

Build one or more low-level connection objects from source node to target node.

Parameters:
- `sourceNode` - - Source node.
- `targetNode` - - Target node.
- `initialWeight` - - Optional explicit initial weight.

Returns: Created low-level connection objects.

### registerCreatedConnections

```ts
registerCreatedConnections(
  network: default,
  internalState: ConnectNetworkInternals,
  sourceNode: default,
  targetNode: default,
  createdConnections: default[],
): void
```

Register created connections in either normal-connection or self-connection storage.

Parameters:
- `network` - - Network instance owning connection collections.
- `internalState` - - Runtime network internals used by connection pipeline.
- `sourceNode` - - Source node used during connection creation.
- `targetNode` - - Target node used during connection creation.
- `createdConnections` - - Created low-level connection objects.

Returns: Nothing.

### markConnectionCachesDirtyWhenNeeded

```ts
markConnectionCachesDirtyWhenNeeded(
  internalState: ConnectNetworkInternals,
  createdConnectionCount: number,
): void
```

Mark topology and slab caches dirty when connection creation occurred.

Parameters:
- `internalState` - - Runtime network internals used by connection pipeline.
- `createdConnectionCount` - - Number of created low-level connections.

Returns: Nothing.

### registerSingleCreatedConnection

```ts
registerSingleCreatedConnection(
  network: default,
  internalState: ConnectNetworkInternals,
  isSelfConnection: boolean,
  createdConnection: default,
): void
```

Register one created connection in the appropriate collection.

Parameters:
- `network` - - Network instance owning connection collections.
- `internalState` - - Runtime network internals used by connection pipeline.
- `isSelfConnection` - - Whether source and target nodes are the same.
- `createdConnection` - - Created low-level connection object.

Returns: Nothing.

## architecture/network/connect/network.connect.remove.utils.ts

### selectConnectionCollection

```ts
selectConnectionCollection(
  network: default,
  sourceNode: default,
  targetNode: default,
): default[]
```

Select the relevant collection to search for the edge.

Parameters:
- `network` - - Network instance owning connection collections.
- `sourceNode` - - Source node.
- `targetNode` - - Target node.

Returns: Candidate connection collection.

### removeFirstMatchingConnection

```ts
removeFirstMatchingConnection(
  network: default,
  candidateConnections: default[],
  sourceNode: default,
  targetNode: default,
): void
```

Remove first connection that matches source and target nodes.

Parameters:
- `network` - - Network instance used for ungating.
- `candidateConnections` - - Candidate collection to search.
- `sourceNode` - - Source node.
- `targetNode` - - Target node.

Returns: Nothing.

### disconnectNodes

```ts
disconnectNodes(
  sourceNode: default,
  targetNode: default,
): void
```

Delegate per-node disconnect cleanup.

Parameters:
- `sourceNode` - - Source node.
- `targetNode` - - Target node.

Returns: Nothing.

### markStructureCachesDirty

```ts
markStructureCachesDirty(
  internalState: ConnectNetworkInternals,
): void
```

Mark topology/slab caches dirty after structural mutation.

Parameters:
- `internalState` - - Runtime network internals used by connection pipeline.

Returns: Nothing.

### findConnectionIndex

```ts
findConnectionIndex(
  candidateConnections: default[],
  sourceNode: default,
  targetNode: default,
): number
```

Find index of the first connection matching source and target nodes.

Parameters:
- `candidateConnections` - - Candidate collection to search.
- `sourceNode` - - Source node.
- `targetNode` - - Target node.

Returns: Matching index or -1 when no edge is found.

### removeConnectionAtIndex

```ts
removeConnectionAtIndex(
  network: default,
  candidateConnections: default[],
  targetConnectionIndex: number,
): void
```

Remove one connection by index, ungating first if required.

Parameters:
- `network` - - Network instance used for ungating.
- `candidateConnections` - - Candidate collection containing target index.
- `targetConnectionIndex` - - Index to remove.

Returns: Nothing.
