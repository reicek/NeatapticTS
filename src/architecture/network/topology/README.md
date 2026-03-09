# architecture/network/topology

## architecture/network/topology/network.topology.utils.types.ts

### IN_DEGREE_DECREMENT

### INPUT_NODE_TYPE

### PathSearchContext

Mutable context used while running iterative DFS reachability checks.

### TopologyBuildContext

Mutable context used while building Kahn topological order.

### TopologyNetwork

Network instance type used by topology helpers.

### TopologyNetworkProps

Internal topology state view carried across helper groups.

### TopologyNode

Node instance type used by topology helpers.

### ZERO_COUNT

## architecture/network/topology/network.topology.utils.ts

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

### createMLP

`(inputCount: number, hiddenCounts: number[], outputCount: number) => import("src/architecture/network").default`

Build a strictly layered and fully connected MLP network.

Parameters:
- `this` - Network constructor.
- `inputCount` - Number of input nodes.
- `hiddenCounts` - Hidden-layer node counts.
- `outputCount` - Number of output nodes.

Returns: Newly created MLP network.

### hasPath

`(from: import("src/architecture/node").default, to: import("src/architecture/node").default) => boolean`

Depth-first reachability test (avoids infinite loops via visited set).

### rebuildConnections

`(networkInstance: import("src/architecture/network").default) => void`

Rebuild the canonical connection array from per-node outgoing lists.

Parameters:
- `networkInstance` - Target network.

## architecture/network/topology/network.topology.loop.utils.ts

### appendTopoNode

`(topoOrder: import("src/architecture/node").default[], node: import("src/architecture/node").default) => void`

Append one node to topological order output.

Parameters:
- `topoOrder` - Accumulated topological order.
- `node` - Node to append.

Returns: Void.

### decrementNodeInDegree

`(buildContext: import("src/architecture/network/network.types").TopologyBuildContext, node: import("src/architecture/node").default) => number`

Decrement node in-degree and return remaining value.

Parameters:
- `buildContext` - Mutable build context.
- `node` - Target node.

Returns: Remaining in-degree after decrement.

### getInDegree

`(buildContext: import("src/architecture/network/network.types").TopologyBuildContext, node: import("src/architecture/node").default) => number`

Read in-degree for a node with zero fallback.

Parameters:
- `buildContext` - Mutable build context.
- `node` - Candidate node.

Returns: In-degree value.

### isInputNode

`(node: import("src/architecture/node").default) => boolean`

Test whether a node is an input node.

Parameters:
- `node` - Candidate node.

Returns: True when node type is input.

### isQueueSeedNode

`(node: import("src/architecture/node").default, buildContext: import("src/architecture/network/network.types").TopologyBuildContext) => boolean`

Determine whether a node belongs in the initial queue.

Parameters:
- `node` - Candidate node.
- `buildContext` - Mutable build context.

Returns: True when node is input-type or has zero in-degree.

### isSelfConnection

`(from: import("src/architecture/node").default, to: import("src/architecture/node").default) => boolean`

Test whether a connection is a self-loop.

Parameters:
- `from` - Source node.
- `to` - Target node.

Returns: True when source and target are the same node.

### processKahnQueue

`(buildContext: import("src/architecture/network/network.types").TopologyBuildContext) => void`

Process queue until all available nodes are emitted.

Parameters:
- `buildContext` - Mutable build context.

Returns: Void.

### relaxOutgoingEdges

`(buildContext: import("src/architecture/network/network.types").TopologyBuildContext, currentNode: import("src/architecture/node").default) => void`

Relax outgoing edges for one processed node.

Parameters:
- `buildContext` - Mutable build context.
- `currentNode` - Processed node.

Returns: Void.

### seedProcessingQueue

`(buildContext: import("src/architecture/network/network.types").TopologyBuildContext) => void`

Seed Kahn queue with input nodes and zero in-degree nodes.

Parameters:
- `buildContext` - Mutable build context.

Returns: Void.

### takeNextQueueNode

`(processingQueue: import("src/architecture/node").default[]) => import("src/architecture/node").default`

Shift and return the next queue node.

Parameters:
- `processingQueue` - Queue of pending nodes.

Returns: Next node.

## architecture/network/topology/network.topology.path.utils.ts

### createPathSearchContext

`(from: import("src/architecture/node").default, to: import("src/architecture/node").default) => import("src/architecture/network/network.types").PathSearchContext`

Create DFS search context.

Parameters:
- `from` - Origin node.
- `to` - Target node.

Returns: Initialized path-search context.

### hasVisitedNode

`(visitedNodes: Set<import("src/architecture/node").default>, node: import("src/architecture/node").default) => boolean`

Test whether a node has already been visited.

Parameters:
- `visitedNodes` - Visited-node set.
- `node` - Candidate node.

Returns: True when node is already visited.

### isSameNode

`(leftNode: import("src/architecture/node").default, rightNode: import("src/architecture/node").default) => boolean`

Compare node identity.

Parameters:
- `leftNode` - Left node.
- `rightNode` - Right node.

Returns: True when references are identical.

### isSelfConnection

`(from: import("src/architecture/node").default, to: import("src/architecture/node").default) => boolean`

Test whether a connection is a self-loop.

Parameters:
- `from` - Source node.
- `to` - Target node.

Returns: True when source and target are the same node.

### markVisited

`(visitedNodes: Set<import("src/architecture/node").default>, node: import("src/architecture/node").default) => void`

Mark a node as visited.

Parameters:
- `visitedNodes` - Visited-node set.
- `node` - Node to mark.

Returns: Void.

### pushOutgoingTargets

`(nodesToVisitStack: import("src/architecture/node").default[], currentNode: import("src/architecture/node").default) => void`

Push non-self outgoing targets to DFS stack.

Parameters:
- `nodesToVisitStack` - DFS stack.
- `currentNode` - Current expanded node.

Returns: Void.

### takeNextStackNode

`(nodesToVisitStack: import("src/architecture/node").default[]) => import("src/architecture/node").default`

Pop and return next DFS stack node.

Parameters:
- `nodesToVisitStack` - DFS stack.

Returns: Next node to process.

### traversePathSearch

`(searchContext: import("src/architecture/network/network.types").PathSearchContext) => boolean`

Traverse DFS search stack and test reachability.

Parameters:
- `searchContext` - Mutable search context.

Returns: True when target node is reachable.

## architecture/network/topology/network.topology.setup.utils.ts

### applyIncomingEdgeCounts

`(buildContext: import("src/architecture/network/network.types").TopologyBuildContext) => void`

Apply in-degree increments from non-self connections.

Parameters:
- `buildContext` - Mutable build context.

Returns: Void.

### asTopologyProps

`(network: import("src/architecture/network").default) => import("src/architecture/network/network.types").TopologyNetworkProps`

Cast network to internal topology props view.

Parameters:
- `network` - Network instance.

Returns: Internal topology props view.

### clearCachedTopoOrder

`(internalTopologyProps: import("src/architecture/network/network.types").TopologyNetworkProps) => void`

Clear cached topological order state.

Parameters:
- `internalTopologyProps` - Internal topology props view.

Returns: Void.

### createTopologyBuildContext

`(network: import("src/architecture/network").default, internalTopologyProps: import("src/architecture/network/network.types").TopologyNetworkProps) => import("src/architecture/network/network.types").TopologyBuildContext`

Create mutable build context for Kahn traversal.

Parameters:
- `network` - Network instance.
- `internalTopologyProps` - Internal topology props view.

Returns: Initialized build context.

### finalizeTopoOrder

`(buildContext: import("src/architecture/network/network.types").TopologyBuildContext) => void`

Finalize cached order, falling back to raw node order on cycle detection.

Parameters:
- `buildContext` - Mutable build context.

Returns: Void.

### incrementNodeInDegree

`(buildContext: import("src/architecture/network/network.types").TopologyBuildContext, node: import("src/architecture/node").default) => void`

Increment in-degree for a node in the tally map.

Parameters:
- `buildContext` - Mutable build context.
- `node` - Target node.

Returns: Void.

### initializeAllNodeInDegreeCounts

`(buildContext: import("src/architecture/network/network.types").TopologyBuildContext) => void`

Initialize all nodes with zero in-degree.

Parameters:
- `buildContext` - Mutable build context.

Returns: Void.

### isSelfConnection

`(from: import("src/architecture/node").default, to: import("src/architecture/node").default) => boolean`

Test whether a connection is a self-loop.

Parameters:
- `from` - Source node.
- `to` - Target node.

Returns: True when source and target are the same node.

### resolveFinalOrder

`(buildContext: import("src/architecture/network/network.types").TopologyBuildContext) => import("src/architecture/node").default[]`

Resolve final topological order with cycle fallback.

Parameters:
- `buildContext` - Mutable build context.

Returns: Fully valid topological order or raw node order fallback.

### shouldUseRawNodeOrder

`(internalTopologyProps: import("src/architecture/network/network.types").TopologyNetworkProps) => boolean`

Determine whether topological order should be bypassed.

Parameters:
- `internalTopologyProps` - Internal topology props view.

Returns: True when acyclic mode is disabled.

## architecture/network/topology/network.topology.factory.utils.ts

### addOutgoingConnectionsToSet

`(outgoingConnections: import("src/architecture/connection").default[], allConnections: Set<import("src/architecture/connection").default>) => void`

Add all outgoing connections to a deduplication set.

Parameters:
- `outgoingConnections` - Outgoing connections from one node.
- `allConnections` - Deduplication set for network connections.

### assignNetworkNodes

`(networkInstance: import("src/architecture/network").default, mlpNodeLayers: MlpNodeLayers) => void`

Assign ordered nodes to the network instance.

Parameters:
- `networkInstance` - Target network.
- `mlpNodeLayers` - Grouped node layers for this MLP.

### collectUniqueOutgoingConnections

`(networkInstance: import("src/architecture/network").default) => Set<import("src/architecture/connection").default>`

Collect unique outgoing connections across all network nodes.

Parameters:
- `networkInstance` - Target network.

Returns: Set of unique outgoing connections.

### connectLayerPair

`(sourceLayer: import("src/architecture/node").default[], targetLayer: import("src/architecture/node").default[]) => void`

Fully connect every source node to every target node.

Parameters:
- `sourceLayer` - Source layer.
- `targetLayer` - Target layer.

### connectMlpLayers

`(mlpNodeLayers: MlpNodeLayers) => void`

Fully connect each adjacent layer in MLP order.

Parameters:
- `mlpNodeLayers` - Grouped node layers for this MLP.

### convertConnectionSetToArray

`(uniqueConnections: Set<import("src/architecture/connection").default>) => import("src/architecture/connection").default[]`

Convert a connection set into the canonical array format.

Parameters:
- `uniqueConnections` - Unique network connections.

Returns: Array of network connections.

### createHiddenLayers

`(hiddenCounts: number[]) => import("src/architecture/node").default[][]`

Create all hidden layers for an MLP topology.

Parameters:
- `hiddenCounts` - Hidden-layer node counts.

Returns: Hidden layers in forward order.

### createMLP

`(inputCount: number, hiddenCounts: number[], outputCount: number) => import("src/architecture/network").default`

Build a strictly layered and fully connected MLP network.

Parameters:
- `this` - Network constructor.
- `inputCount` - Number of input nodes.
- `hiddenCounts` - Hidden-layer node counts.
- `outputCount` - Number of output nodes.

Returns: Newly created MLP network.

### createMlpNodeLayers

`(inputCount: number, hiddenCounts: number[], outputCount: number) => MlpNodeLayers`

Build input, hidden, and output node layers for an MLP topology.

Parameters:
- `inputCount` - Number of input nodes.
- `hiddenCounts` - Hidden-layer node counts.
- `outputCount` - Number of output nodes.

Returns: Grouped node layers for MLP assembly.

### createNodesOfType

`(nodeCount: number, nodeType: "input" | "output" | "hidden") => import("src/architecture/node").default[]`

Create all nodes for a single fixed node type.

Parameters:
- `nodeCount` - Number of nodes to create.
- `nodeType` - Node type identifier.

Returns: Node list of the requested type.

### createOrderedNodeList

`(mlpNodeLayers: MlpNodeLayers) => import("src/architecture/node").default[]`

Build the canonical ordered node list used by the network.

Parameters:
- `mlpNodeLayers` - Grouped node layers for this MLP.

Returns: Ordered node list: input, hidden, then output.

### flattenNodeLayers

`(nodeLayers: import("src/architecture/node").default[][]) => import("src/architecture/node").default[]`

Flatten layered node collections into a single ordered list.

Parameters:
- `nodeLayers` - Layered node collections.

Returns: Flattened node list.

### instantiateNetwork

`(networkFactory: NetworkConstructor, inputCount: number, outputCount: number) => import("src/architecture/network").default`

Instantiate a new network using the runtime constructor.

Parameters:
- `networkFactory` - Network constructor function.
- `inputCount` - Number of input nodes.
- `outputCount` - Number of output nodes.

Returns: Newly instantiated network.

### markTopologyDirty

`(networkInstance: import("src/architecture/network").default) => void`

Mark a network topology as dirty after structural edits.

Parameters:
- `networkInstance` - Network instance to mark.

### rebuildConnections

`(networkInstance: import("src/architecture/network").default) => void`

Rebuild the canonical connection array from per-node outgoing lists.

Parameters:
- `networkInstance` - Target network.

## architecture/network/topology/network.topology.architecture.utils.ts

### createArchitectureDescriptor

`(hiddenLayerSizes: number[], hasCycles: boolean, source: import("src/architecture/network/network.types").NetworkArchitectureSource, totalNodes: number, totalConnections: number) => import("src/architecture/network/network.types").NetworkArchitectureDescriptor`

Creates the final immutable descriptor shape used by telemetry and UI code.

Keeping descriptor assembly in one place ensures every resolution strategy
returns the same payload contract and avoids accidental field drift.

Parameters:
- `hiddenLayerSizes` - - Hidden-layer widths.
- `hasCycles` - - Whether cycles were detected.
- `source` - - Descriptor provenance.
- `totalNodes` - - Node count.
- `totalConnections` - - Connection count.

Returns: Descriptor object.

### createDirectedEdgeList

`(runtimeConnections: RuntimeConnectionLike[], nodeByIndex: Map<number, RuntimeNodeLike>) => { fromIndex: number; toIndex: number; }[]`

Produces a validated list of enabled directed edges.

Invalid references, disabled connections, and self-loops are removed so the
remaining edge list can be consumed safely by cycle and depth algorithms.

Parameters:
- `runtimeConnections` - - Runtime connections.
- `nodeByIndex` - - Indexed nodes.

Returns: Valid directed edges.

### createNodeIndexMap

`(runtimeNodes: RuntimeNodeLike[]) => Map<number, RuntimeNodeLike>`

Builds a node lookup table keyed by stable index.

Runtime objects may omit `index`; in that case the current array position is
used as a deterministic fallback to keep downstream graph logic total.

Parameters:
- `runtimeNodes` - - Runtime nodes.

Returns: Node map keyed by stable node index.

### describeArchitecture

`(network: import("src/architecture/network").default) => import("src/architecture/network/network.types").NetworkArchitectureDescriptor`

Describes network architecture for diagnostics, telemetry, and UI rendering.

This function prefers factual sources over heuristics so downstream tooling
can rely on the descriptor while still receiving useful output for partially
specified runtime graphs.

Resolution priority is intentionally explicit:
1) node `layer` metadata (factual when present)
2) graph-derived feed-forward depth layering (factual for acyclic graphs)
3) hidden-node count fallback (heuristic inference)

Parameters:
- `network` - - Runtime network instance.

Returns: Stable architecture descriptor.

### isHiddenNode

`(runtimeNode: RuntimeNodeLike) => boolean`

Identifies whether a runtime node should be treated as hidden for topology
reconstruction and fallback inference.

Parameters:
- `runtimeNode` - - Candidate node.

Returns: True when node type is hidden.

### resolveCycleStateAndTopoOrder

`(nodeByIndex: Map<number, RuntimeNodeLike>, directedEdges: { fromIndex: number; toIndex: number; }[]) => { topologicalOrder: number[]; hasCycles: boolean; }`

Resolves cycle presence and, when possible, returns a topological order
using Kahn's algorithm.

A complete topological ordering implies an acyclic graph. If some nodes
remain unprocessed, at least one cycle exists.

Parameters:
- `nodeByIndex` - - Indexed nodes.
- `directedEdges` - - Directed edges.

Returns: Topological order and cycle status.

### resolveHiddenCountsByDepth

`(nodeByIndex: Map<number, RuntimeNodeLike>, depthByNodeIndex: Map<number, number>) => Map<number, number>`

Aggregates hidden-node counts per derived depth.

This is the final transformation before emitting architecture widths:
hidden nodes are grouped by depth and counted in insertion-safe maps.

Parameters:
- `nodeByIndex` - - Indexed nodes.
- `depthByNodeIndex` - - Derived depths.

Returns: Hidden-node counts by depth.

### resolveHiddenLayerSizesFromGraphTopology

`(runtimeNodes: RuntimeNodeLike[], runtimeConnections: RuntimeConnectionLike[]) => { hiddenLayerSizes: number[]; hasCycles: boolean; }`

Derives hidden-layer widths from graph topology when no explicit layer
metadata is available.

The method computes a topological depth model for acyclic graphs; cyclic
graphs are flagged and intentionally return no width inference because depth
is not well-defined in recurrent loops.

Parameters:
- `runtimeNodes` - - Runtime nodes.
- `runtimeConnections` - - Runtime connections.

Returns: Hidden-layer widths derived from acyclic topology and cycle flag.

### resolveHiddenLayerSizesFromLayerMetadata

`(runtimeNodes: RuntimeNodeLike[]) => number[]`

Resolves hidden-layer widths from explicit `node.layer` metadata.

This is treated as the most trustworthy source because layer assignment is
usually produced by architecture-aware builders and does not depend on
topological reconstruction.

Parameters:
- `runtimeNodes` - - Runtime nodes.

Returns: Hidden-layer widths from explicit node.layer metadata.

### resolveNodeDepthByIndex

`(nodeByIndex: Map<number, RuntimeNodeLike>, directedEdges: { fromIndex: number; toIndex: number; }[], topologicalOrder: number[]) => Map<number, number>`

Computes node depth (feed-forward distance from inputs) for an acyclic graph.

Depth assignment is parent-driven: each node depth is one plus the maximum
resolved parent depth. Nodes with no resolved parents are skipped.

Parameters:
- `nodeByIndex` - - Indexed nodes.
- `directedEdges` - - Directed edges.
- `topologicalOrder` - - Acyclic topological order.

Returns: Derived depth by node index.
