# architecture/network/topology

## architecture/network/topology/network.topology.utils.ts

### appendTopoNode

`(topoOrder: import("C:/NeatapticTS/src/architecture/node").default[], node: import("C:/NeatapticTS/src/architecture/node").default) => void`

Append one node to topological order output.

Parameters:
- `topoOrder` - Accumulated topological order.
- `node` - Node to append.

Returns: Void.

### applyIncomingEdgeCounts

`(buildContext: import("C:/NeatapticTS/src/architecture/network/network.types").TopologyBuildContext) => void`

Apply in-degree increments from non-self connections.

Parameters:
- `buildContext` - Mutable build context.

Returns: Void.

### asTopologyProps

`(network: import("C:/NeatapticTS/src/architecture/network").default) => import("C:/NeatapticTS/src/architecture/network/network.types").TopologyNetworkProps`

Cast network to internal topology props view.

Parameters:
- `network` - Network instance.

Returns: Internal topology props view.

### clearCachedTopoOrder

`(internalTopologyProps: import("C:/NeatapticTS/src/architecture/network/network.types").TopologyNetworkProps) => void`

Clear cached topological order state.

Parameters:
- `internalTopologyProps` - Internal topology props view.

Returns: Void.

### computeTopoOrder

`() => void`

Topology utilities.

Provides:
 - computeTopoOrder: Kahn-style topological sorting with graceful fallback when cycles detected.
 - hasPath: depth-first reachability query (used to prevent cycle introduction when acyclicity enforced).

Design Notes:
 - We deliberately tolerate cycles by falling back to raw node ordering instead of throwing; this
   allows callers performing interim structural mutations to proceed (e.g. during evolve phases)
   while signaling that the fast acyclic optimizations should not be used.
 - Input nodes are seeded into the queue immediately regardless of in-degree to keep them early in
   the ordering even if an unusual inbound edge was added (defensive redundancy).
 - Self loops are ignored for in-degree accounting and queue progression (they neither unlock new
   nodes nor should they block ordering completion).

### createPathSearchContext

`(from: import("C:/NeatapticTS/src/architecture/node").default, to: import("C:/NeatapticTS/src/architecture/node").default) => import("C:/NeatapticTS/src/architecture/network/network.types").PathSearchContext`

Create DFS search context.

Parameters:
- `from` - Origin node.
- `to` - Target node.

Returns: Initialized path-search context.

### createTopologyBuildContext

`(network: import("C:/NeatapticTS/src/architecture/network").default, internalTopologyProps: import("C:/NeatapticTS/src/architecture/network/network.types").TopologyNetworkProps) => import("C:/NeatapticTS/src/architecture/network/network.types").TopologyBuildContext`

Create mutable build context for Kahn traversal.

Parameters:
- `network` - Network instance.
- `internalTopologyProps` - Internal topology props view.

Returns: Initialized build context.

### decrementNodeInDegree

`(inDegreeByNode: Map<import("C:/NeatapticTS/src/architecture/node").default, number>, node: import("C:/NeatapticTS/src/architecture/node").default) => number`

Decrement node in-degree and return remaining value.

Parameters:
- `inDegreeByNode` - In-degree tally map.
- `node` - Target node.

Returns: Remaining in-degree after decrement.

### finalizeTopoOrder

`(buildContext: import("C:/NeatapticTS/src/architecture/network/network.types").TopologyBuildContext) => void`

Finalize cached order, falling back to raw node order on cycle detection.

Parameters:
- `buildContext` - Mutable build context.

Returns: Void.

### getInDegree

`(inDegreeByNode: Map<import("C:/NeatapticTS/src/architecture/node").default, number>, node: import("C:/NeatapticTS/src/architecture/node").default) => number`

Read in-degree for a node with zero fallback.

Parameters:
- `inDegreeByNode` - In-degree tally map.
- `node` - Candidate node.

Returns: In-degree value.

### hasPath

`(from: import("C:/NeatapticTS/src/architecture/node").default, to: import("C:/NeatapticTS/src/architecture/node").default) => boolean`

Depth-first reachability test (avoids infinite loops via visited set).

### hasVisitedNode

`(visitedNodes: Set<import("C:/NeatapticTS/src/architecture/node").default>, node: import("C:/NeatapticTS/src/architecture/node").default) => boolean`

Test whether a node has already been visited.

Parameters:
- `visitedNodes` - Visited-node set.
- `node` - Candidate node.

Returns: True when node is already visited.

### incrementNodeInDegree

`(inDegreeByNode: Map<import("C:/NeatapticTS/src/architecture/node").default, number>, node: import("C:/NeatapticTS/src/architecture/node").default) => void`

Increment in-degree for a node in the tally map.

Parameters:
- `inDegreeByNode` - In-degree tally map.
- `node` - Target node.

Returns: Void.

### initializeAllNodeInDegreeCounts

`(buildContext: import("C:/NeatapticTS/src/architecture/network/network.types").TopologyBuildContext) => void`

Initialize all nodes with zero in-degree.

Parameters:
- `buildContext` - Mutable build context.

Returns: Void.

### isInputNode

`(node: import("C:/NeatapticTS/src/architecture/node").default) => boolean`

Test whether a node is an input node.

Parameters:
- `node` - Candidate node.

Returns: True when node type is input.

### isQueueSeedNode

`(node: import("C:/NeatapticTS/src/architecture/node").default, inDegreeByNode: Map<import("C:/NeatapticTS/src/architecture/node").default, number>) => boolean`

Determine whether a node belongs in the initial queue.

Parameters:
- `node` - Candidate node.
- `inDegreeByNode` - In-degree tally map.

Returns: True when node is input-type or has zero in-degree.

### isSameNode

`(leftNode: import("C:/NeatapticTS/src/architecture/node").default, rightNode: import("C:/NeatapticTS/src/architecture/node").default) => boolean`

Compare node identity.

Parameters:
- `leftNode` - Left node.
- `rightNode` - Right node.

Returns: True when references are identical.

### isSelfConnection

`(from: import("C:/NeatapticTS/src/architecture/node").default, to: import("C:/NeatapticTS/src/architecture/node").default) => boolean`

Test whether a connection is a self-loop.

Parameters:
- `from` - Source node.
- `to` - Target node.

Returns: True when source and target are the same node.

### markVisited

`(visitedNodes: Set<import("C:/NeatapticTS/src/architecture/node").default>, node: import("C:/NeatapticTS/src/architecture/node").default) => void`

Mark a node as visited.

Parameters:
- `visitedNodes` - Visited-node set.
- `node` - Node to mark.

Returns: Void.

### processKahnQueue

`(buildContext: import("C:/NeatapticTS/src/architecture/network/network.types").TopologyBuildContext) => void`

Process queue until all available nodes are emitted.

Parameters:
- `buildContext` - Mutable build context.

Returns: Void.

### pushOutgoingTargets

`(nodesToVisitStack: import("C:/NeatapticTS/src/architecture/node").default[], currentNode: import("C:/NeatapticTS/src/architecture/node").default) => void`

Push non-self outgoing targets to DFS stack.

Parameters:
- `nodesToVisitStack` - DFS stack.
- `currentNode` - Current expanded node.

Returns: Void.

### relaxOutgoingEdges

`(buildContext: import("C:/NeatapticTS/src/architecture/network/network.types").TopologyBuildContext, currentNode: import("C:/NeatapticTS/src/architecture/node").default) => void`

Relax outgoing edges for one processed node.

Parameters:
- `buildContext` - Mutable build context.
- `currentNode` - Processed node.

Returns: Void.

### resolveFinalOrder

`(buildContext: import("C:/NeatapticTS/src/architecture/network/network.types").TopologyBuildContext) => import("C:/NeatapticTS/src/architecture/node").default[]`

Resolve final topological order with cycle fallback.

Parameters:
- `buildContext` - Mutable build context.

Returns: Fully valid topological order or raw node order fallback.

### seedProcessingQueue

`(buildContext: import("C:/NeatapticTS/src/architecture/network/network.types").TopologyBuildContext) => void`

Seed Kahn queue with input nodes and zero in-degree nodes.

Parameters:
- `buildContext` - Mutable build context.

Returns: Void.

### shouldUseRawNodeOrder

`(internalTopologyProps: import("C:/NeatapticTS/src/architecture/network/network.types").TopologyNetworkProps) => boolean`

Determine whether topological order should be bypassed.

Parameters:
- `internalTopologyProps` - Internal topology props view.

Returns: True when acyclic mode is disabled.

### takeNextQueueNode

`(processingQueue: import("C:/NeatapticTS/src/architecture/node").default[]) => import("C:/NeatapticTS/src/architecture/node").default`

Shift and return the next queue node.

Parameters:
- `processingQueue` - Queue of pending nodes.

Returns: Next node.

### takeNextStackNode

`(nodesToVisitStack: import("C:/NeatapticTS/src/architecture/node").default[]) => import("C:/NeatapticTS/src/architecture/node").default`

Pop and return next DFS stack node.

Parameters:
- `nodesToVisitStack` - DFS stack.

Returns: Next node to process.

### traversePathSearch

`(searchContext: import("C:/NeatapticTS/src/architecture/network/network.types").PathSearchContext) => boolean`

Traverse DFS search stack and test reachability.

Parameters:
- `searchContext` - Mutable search context.

Returns: True when target node is reachable.
