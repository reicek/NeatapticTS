# architecture/network/topology

Topology utilities.

Provides:
 - computeTopoOrder: Kahn-style topological sorting with graceful fallback when cycles detected.
 - hasPath: depth-first reachability query (used to prevent cycle introduction when acyclicity enforced).
 - topology contract helpers: public intent accessors that keep semantic API state aligned with low-level runtime flags.

Design Notes:
 - We deliberately tolerate cycles by falling back to raw node ordering instead of throwing; this
   allows callers performing interim structural mutations to proceed (e.g. during evolve phases)
   while signaling that the fast acyclic optimizations should not be used.
 - Input nodes are seeded into the queue immediately regardless of in-degree to keep them early in
   the ordering even if an unusual inbound edge was added (defensive redundancy).
 - Self loops are ignored for in-degree accounting and queue progression (they neither unlock new
   nodes nor should they block ordering completion).

## architecture/network/topology/network.topology.utils.types.ts

### ActivationSchedule

Deterministic activation schedule type produced by topology helpers and consumed by the activation chapter.

### ActivationScheduleStep

One ordered step in the deterministic activation schedule produced by the topology sort and consumed at inference time.

### ActivationSchedulingDiagnostics

Human-readable activation scheduling diagnostics type carrying node counts, depth, and coverage metadata for inspection.

### IN_DEGREE_DECREMENT

Unit step value applied when decrementing or incrementing in-degree tally entries during Kahn queue processing.

### INPUT_NODE_TYPE

Input node-type discriminator used to seed the Kahn topological sort with source nodes that have no predecessors.

### PathSearchContext

Mutable scratch context allocated and carried while running iterative depth-first reachability checks across the graph.

### TopologyBuildContext

Mutable scratch context allocated and carried while building the Kahn-algorithm topological activation order.

### TopologyNetwork

Network instance type alias used by topology helpers to avoid direct runtime imports at the utility boundary.

### TopologyNetworkProps

Internal topology state view carrying network node and connection lists across all topology helper groups consistently.

### TopologyNode

Node instance type alias used by topology helpers to avoid direct runtime imports at the utility boundary.

### ZERO_COUNT

Zero baseline value used to initialize degree counters and empty-size comparisons during topological scheduling.

## architecture/network/topology/network.topology.utils.ts

### computeTopoOrder

```ts
computeTopoOrder(): void
```

Compute a deterministic activation schedule for the current topology mode.

Acyclic mode uses Kahn traversal with stable waves and still flattens those
waves back into the legacy `_topoOrder` cache for callers that depend on one
ordered list. Recurrent mode uses the SCC condensation graph to emit
deterministic recurrent-component boundaries while leaving the legacy acyclic
cache empty until the activation path adopts the richer schedule directly.

Parameters:
- `this` - Network instance bound by method call.

### createMLP

```ts
createMLP(
  inputCount: number,
  hiddenCounts: number[],
  outputCount: number,
): default
```

Contract for createMLP.

### getTopologyIntent

```ts
getTopologyIntent(): NetworkTopologyIntent
```

Read the public topology intent preserved on a network instance.

This accessor keeps the semantic contract visible to callers even though the
lower-level runtime ultimately enforces acyclicity through booleans and cache
invalidation.

Parameters:
- `this` - Target network instance.

Returns: Current topology intent.

### hasFeedForwardTopologyContract

```ts
hasFeedForwardTopologyContract(
  carrier: FeedForwardTopologyContractCarrier,
): boolean
```

Check whether a runtime shape currently carries the feed-forward contract.

The helper is intentionally conservative when callers are in a mismatched
transitional state: either an explicit `feed-forward` intent or a truthy
`_enforceAcyclic` flag is treated as a feed-forward contract. That keeps
mutation and crossover helpers from introducing recurrent structure into a
genome that still advertises acyclic semantics anywhere on its runtime seam.

Parameters:
- `carrier` - Narrow runtime shape or full network instance.

Returns: True when feed-forward semantics are currently enforced.

### hasPath

```ts
hasPath(
  from: default,
  to: default,
): boolean
```

Depth-first reachability test that avoids infinite loops using a visited set.

Parameters:
- `this` - Network instance bound by method call.
- `from` - Source node from which reachability is tested.
- `to` - Target node to which reachability is tested.

Returns: True when a path exists from `from` to `to`, false otherwise.

### networkTopologyUtils

Default export bundle for the topology utilities chapter.

Bundles the core topology helpers so the network facade can bind them as methods
without importing each function individually.

### rebuildConnections

```ts
rebuildConnections(
  networkInstance: default,
): void
```

Rebuild the canonical connection array from all per-node outgoing lists.

Parameters:
- `networkInstance` - Target network.

### setEnforceAcyclic

```ts
setEnforceAcyclic(
  flag: boolean,
): void
```

Toggle low-level acyclic enforcement while preserving a coherent public contract.

This exists for backward compatibility with callers that still use the legacy
boolean API instead of the semantic `topologyIntent` field.

Parameters:
- `this` - Target network instance.
- `flag` - Whether to enforce acyclic connectivity.

Returns: Nothing.

### setTopologyIntent

```ts
setTopologyIntent(
  topologyIntent: NetworkTopologyIntent,
): void
```

Set the public topology intent and synchronize low-level runtime flags.

Updating the semantic contract also updates acyclic enforcement and marks the
topological cache dirty so later activation paths rebuild consistent state.

Parameters:
- `this` - Target network instance.
- `topologyIntent` - Desired topology intent.

Returns: Nothing.

## architecture/network/topology/network.topology.loop.utils.ts

### appendActivationStep

```ts
appendActivationStep(
  activationSteps: number[][],
  activationStep: number[],
): void
```

Append one completed activation wave to the cached schedule.

Parameters:
- `activationSteps` - Accumulated activation waves.
- `activationStep` - Current activation wave.

Returns: Void.

### appendActivationStepNode

```ts
appendActivationStepNode(
  activationStep: number[],
  node: default,
): void
```

Append one stable node id to the current activation wave.

Parameters:
- `activationStep` - Current activation wave.
- `node` - Node to append.

Returns: Void.

### appendTopoNode

```ts
appendTopoNode(
  topoOrder: default[],
  node: default,
): void
```

Append one node to topological order output.

Parameters:
- `topoOrder` - Accumulated topological order.
- `node` - Node to append.

Returns: Void.

### compareNodesByStableTieBreak

```ts
compareNodesByStableTieBreak(
  leftNode: default,
  rightNode: default,
): number
```

Compare two nodes using a stable deterministic activation tie-break order.

Parameters:
- `leftNode` - First node.
- `rightNode` - Second node.

Returns: Negative when left should run first, positive when right should run first.

### decrementNodeInDegree

```ts
decrementNodeInDegree(
  buildContext: TopologyBuildContext,
  node: default,
): number
```

Decrement node in-degree and return remaining value.

Parameters:
- `buildContext` - Mutable build context.
- `node` - Target node.

Returns: Remaining in-degree after decrement.

### getInDegree

```ts
getInDegree(
  buildContext: TopologyBuildContext,
  node: default,
): number
```

Read in-degree for a node with zero fallback.

Parameters:
- `buildContext` - Mutable build context.
- `node` - Candidate node.

Returns: In-degree value.

### isInputNode

```ts
isInputNode(
  node: default,
): boolean
```

Test whether a node is an input node.

Parameters:
- `node` - Candidate node.

Returns: True when node type is input.

### isQueueSeedNode

```ts
isQueueSeedNode(
  node: default,
  buildContext: TopologyBuildContext,
): boolean
```

Determine whether a node belongs in the initial queue.

Parameters:
- `node` - Candidate node.
- `buildContext` - Mutable build context.

Returns: True when node is input-type or has zero in-degree.

### isSelfConnection

```ts
isSelfConnection(
  from: default,
  to: default,
): boolean
```

Test whether a connection is a self-loop.

Parameters:
- `from` - Source node.
- `to` - Target node.

Returns: True when source and target are the same node.

### processKahnQueue

```ts
processKahnQueue(
  buildContext: TopologyBuildContext,
): void
```

Process the Kahn queue until all available topology nodes are emitted.

Parameters:
- `buildContext` - Mutable build context.

Returns: Void.

### relaxOutgoingEdges

```ts
relaxOutgoingEdges(
  buildContext: TopologyBuildContext,
  currentNode: default,
  nextProcessingQueue: default[],
): void
```

Relax outgoing edges for one processed node.

Parameters:
- `buildContext` - Mutable build context.
- `currentNode` - Processed node.

Returns: Void.

### resolveStableNodeTieBreakValue

```ts
resolveStableNodeTieBreakValue(
  node: default,
): number
```

Resolve the deterministic activation tie-break scalar for one node.

Stable gene ids are preferred. Node index remains a conservative fallback for
unusual fixtures that bypass ordinary node construction.

Parameters:
- `node` - Candidate node.

Returns: Deterministic scalar used for sorting and schedule emission.

### seedProcessingQueue

```ts
seedProcessingQueue(
  buildContext: TopologyBuildContext,
): void
```

Seed Kahn queue with input nodes and zero in-degree nodes.

Parameters:
- `buildContext` - Mutable build context.

Returns: Void.

### sortNodesByStableTieBreak

```ts
sortNodesByStableTieBreak(
  nodes: default[],
): default[]
```

Sort one node collection by the deterministic activation wave tie-break order.

Parameters:
- `nodes` - Candidate nodes.

Returns: Sorted node collection.

### takeNextQueueStep

```ts
takeNextQueueStep(
  processingQueue: default[],
): default[]
```

Take the full current Kahn wave from the processing queue.

Parameters:
- `processingQueue` - Queue of pending nodes.

Returns: Current zero-in-degree wave in deterministic order.

## architecture/network/topology/network.topology.path.utils.ts

### createPathSearchContext

```ts
createPathSearchContext(
  from: default,
  to: default,
): PathSearchContext
```

Create a depth-first search context for reachability testing between two topology nodes.

Parameters:
- `from` - Origin node.
- `to` - Target node.

Returns: Initialized path-search context.

### hasVisitedNode

```ts
hasVisitedNode(
  visitedNodes: Set<default>,
  node: default,
): boolean
```

Test whether a node has already been visited.

Parameters:
- `visitedNodes` - Visited-node set.
- `node` - Candidate node.

Returns: True when node is already visited.

### isSameNode

```ts
isSameNode(
  leftNode: default,
  rightNode: default,
): boolean
```

Compare two node references by strict identity and return true when they refer to the same node.

Parameters:
- `leftNode` - Left node.
- `rightNode` - Right node.

Returns: True when references are identical.

### isSelfConnection

```ts
isSelfConnection(
  from: default,
  to: default,
): boolean
```

Test whether a connection is a self-loop.

Parameters:
- `from` - Source node.
- `to` - Target node.

Returns: True when source and target are the same node.

### markVisited

```ts
markVisited(
  visitedNodes: Set<default>,
  node: default,
): void
```

Mark a node as visited.

Parameters:
- `visitedNodes` - Visited-node set.
- `node` - Node to mark.

Returns: Void.

### pushOutgoingTargets

```ts
pushOutgoingTargets(
  nodesToVisitStack: default[],
  currentNode: default,
): void
```

Push non-self outgoing targets to DFS stack.

Parameters:
- `nodesToVisitStack` - DFS stack.
- `currentNode` - Current expanded node.

Returns: Void.

### takeNextStackNode

```ts
takeNextStackNode(
  nodesToVisitStack: default[],
): default
```

Pop and return next DFS stack node.

Parameters:
- `nodesToVisitStack` - DFS stack.

Returns: Next node to process.

### traversePathSearch

```ts
traversePathSearch(
  searchContext: PathSearchContext,
): boolean
```

Traverse the DFS search stack and test whether the target node is reachable.

Parameters:
- `searchContext` - Mutable search context.

Returns: True when target node is reachable.

## architecture/network/topology/network.topology.setup.utils.ts

### applyIncomingEdgeCounts

```ts
applyIncomingEdgeCounts(
  buildContext: TopologyBuildContext,
): void
```

Apply in-degree increments from non-self connections so topological scheduling reflects only true inter-node dependencies.
Self loops are excluded because they do not participate in feed-forward ordering.

Parameters:
- `buildContext` - Mutable build context.

Returns: Void.

### asTopologyProps

```ts
asTopologyProps(
  network: default,
): TopologyNetworkProps
```

Cast a network instance to the internal topology props view for flag access.

Parameters:
- `network` - Network instance.

Returns: Internal topology props view.

### buildRecurrentScheduleSteps

```ts
buildRecurrentScheduleSteps(
  stronglyConnectedComponents: readonly default[][],
  condensationContext: CondensationContext,
): ActivationScheduleStep[]
```

Build deterministic recurrent schedule steps from the condensation graph.

Parameters:
- `stronglyConnectedComponents` - Stable SCC list.
- `condensationContext` - Condensation graph context.

Returns: Structured recurrent schedule steps.

### clearCachedTopoOrder

```ts
clearCachedTopoOrder(
  internalTopologyProps: TopologyNetworkProps,
): void
```

Clear cached topological order state and reset compiled scheduling diagnostics when topology changes invalidate previous results.
This keeps later activation passes from reusing stale ordering data.

Parameters:
- `internalTopologyProps` - Internal topology props view.

Returns: Void.

### collectStronglyConnectedComponents

```ts
collectStronglyConnectedComponents(
  nodes: readonly default[],
): default[][]
```

Collect strongly-connected components using Tarjan traversal.

Parameters:
- `nodes` - Candidate graph nodes.

Returns: Stable SCC list.

### createComponentIndexByNode

```ts
createComponentIndexByNode(
  stronglyConnectedComponents: readonly default[][],
): Map<default, number>
```

Build a reverse lookup from node to SCC index.

Parameters:
- `stronglyConnectedComponents` - Stable SCC list.

Returns: Node-to-component lookup map.

### createCondensationContext

```ts
createCondensationContext(
  network: default,
  stronglyConnectedComponents: readonly default[][],
  componentIndexByNode: ReadonlyMap<default, number>,
): CondensationContext
```

Build the SCC condensation graph.

Parameters:
- `network` - Network instance.
- `stronglyConnectedComponents` - Stable SCC list.
- `componentIndexByNode` - Node-to-component lookup.

Returns: Condensation graph context.

### createTopologyBuildContext

```ts
createTopologyBuildContext(
  network: default,
  internalTopologyProps: TopologyNetworkProps,
): TopologyBuildContext
```

Create mutable build context for Kahn traversal so in-degree maps, queues, and output buffers share one typed state object.
Centralizing this context keeps scheduling helpers composable and deterministic.

Parameters:
- `network` - Network instance.
- `internalTopologyProps` - Internal topology props view.

Returns: Initialized build context.

### finalizeRecurrentSchedule

```ts
finalizeRecurrentSchedule(
  network: default,
  internalTopologyProps: TopologyNetworkProps,
): void
```

Build and cache the deterministic recurrent activation schedule for the network.

Parameters:
- `network` - Network instance.
- `internalTopologyProps` - Internal topology props view.

Returns: Void.

### finalizeTopoOrder

```ts
finalizeTopoOrder(
  buildContext: TopologyBuildContext,
): void
```

Finalize cached order, falling back to raw node order on cycle detection.

Parameters:
- `buildContext` - Mutable build context.

Returns: Void.

### incrementNodeInDegree

```ts
incrementNodeInDegree(
  buildContext: TopologyBuildContext,
  node: default,
): void
```

Increment in-degree for a node in the tally map.

Parameters:
- `buildContext` - Mutable build context.
- `node` - Target node.

Returns: Void.

### initializeAllNodeInDegreeCounts

```ts
initializeAllNodeInDegreeCounts(
  buildContext: TopologyBuildContext,
): void
```

Initialize all nodes with zero in-degree before incoming-edge counting populates the mutable Kahn traversal state.
This explicit reset prevents stale counts when contexts are reused across rebuilds.

Parameters:
- `buildContext` - Mutable build context.

Returns: Void.

### isRecurrentComponent

```ts
isRecurrentComponent(
  componentNodes: readonly default[],
): boolean
```

Check whether one SCC should be treated as a recurrent execution boundary.

Parameters:
- `componentNodes` - Stable SCC node list.

Returns: True when the component is cyclic or carries a self-loop.

### isSelfConnection

```ts
isSelfConnection(
  from: default,
  to: default,
): boolean
```

Test whether a connection is a self-loop.

Parameters:
- `from` - Source node.
- `to` - Target node.

Returns: True when source and target are the same node.

### resolveCompiledSchedulingDiagnostics

```ts
resolveCompiledSchedulingDiagnostics(
  network: default,
  activationSchedule: ActivationSchedule,
): ActivationSchedulingDiagnostics
```

Resolve the standard diagnostics payload for a compiled schedule.

Parameters:
- `network` - Network instance.
- `activationSchedule` - Compiled activation schedule.

Returns: Scheduling diagnostics snapshot.

### resolveComponentTieBreakValue

```ts
resolveComponentTieBreakValue(
  componentNodes: readonly default[],
): number
```

Resolve one SCC tie-break value from its first stable node.

Parameters:
- `componentNodes` - Stable SCC node list.

Returns: Deterministic component sort scalar.

### resolveCycleNodeIds

```ts
resolveCycleNodeIds(
  buildContext: TopologyBuildContext,
): number[]
```

Resolve stable node ids that remained unscheduled after acyclic traversal.

Parameters:
- `buildContext` - Mutable build context.

Returns: Stable node ids implicated in the cycle fallback.

### resolveFinalActivationSchedule

```ts
resolveFinalActivationSchedule(
  buildContext: TopologyBuildContext,
): ActivationSchedule | null
```

Resolve the final deterministic activation schedule when the graph is acyclic.

Parameters:
- `buildContext` - Mutable build context.

Returns: Cached activation schedule or null when a complete acyclic order was not found.

### resolveFinalOrder

```ts
resolveFinalOrder(
  buildContext: TopologyBuildContext,
): default[]
```

Resolve final topological order with cycle fallback.

Parameters:
- `buildContext` - Mutable build context.

Returns: Fully valid topological order or raw node order fallback.

### resolveFinalSchedulingDiagnostics

```ts
resolveFinalSchedulingDiagnostics(
  buildContext: TopologyBuildContext,
): ActivationSchedulingDiagnostics
```

Resolve final human-friendly scheduling diagnostics for acyclic mode.

Parameters:
- `buildContext` - Mutable build context.

Returns: Scheduling diagnostics snapshot.

### resolveOutgoingNeighbors

```ts
resolveOutgoingNeighbors(
  node: default,
): default[]
```

Resolve one node's outgoing neighbors for SCC traversal.

Self-loops are excluded from traversal because they do not change SCC
membership, but singleton self-loops are still classified as recurrent later.

Parameters:
- `node` - Candidate node.

Returns: Deterministic outgoing neighbors.

### resolveRecurrentActivationSchedule

```ts
resolveRecurrentActivationSchedule(
  network: default,
): ActivationSchedule
```

Resolve the deterministic recurrent activation schedule.

The schedule is based on the SCC condensation graph so recurrent structure is
explicit before activation-path integration consumes it.

Parameters:
- `network` - Network instance.

Returns: Deterministic recurrent activation schedule.

### seedCondensationQueue

```ts
seedCondensationQueue(
  stronglyConnectedComponents: readonly default[][],
  componentInDegree: readonly number[],
  queuedComponentIndexes: Set<number>,
): number[]
```

Seed the condensation queue with zero-indegree or input-owning components.

Parameters:
- `stronglyConnectedComponents` - Stable SCC list.
- `componentInDegree` - Component indegree counts.
- `queuedComponentIndexes` - Mutable set of already queued components.

Returns: Initial deterministic queue.

### shouldBuildRecurrentSchedule

```ts
shouldBuildRecurrentSchedule(
  internalTopologyProps: TopologyNetworkProps,
): boolean
```

Determine whether recurrent scheduling should be used based on topology enforcement flags stored on internal runtime props.
This gate decides whether Kahn-style acyclic ordering or recurrent schedule compilation is executed.

Parameters:
- `internalTopologyProps` - Internal topology props view.

Returns: True when acyclic mode is disabled.

### sortComponentIndexesByTieBreak

```ts
sortComponentIndexesByTieBreak(
  componentIndexes: readonly number[],
  stronglyConnectedComponents: readonly default[][],
): number[]
```

Sort component indexes by the deterministic node tie-break of each SCC root.

Parameters:
- `componentIndexes` - Candidate component indexes.
- `stronglyConnectedComponents` - Stable SCC list.

Returns: Sorted component indexes.

## architecture/network/topology/network.topology.factory.utils.ts

Build a strictly layered, fully connected MLP network from layer sizes.

### addOutgoingConnectionsToSet

```ts
addOutgoingConnectionsToSet(
  outgoingConnections: default[],
  allConnections: Set<default>,
): void
```

Add all outgoing connections to a deduplication set.

Parameters:
- `outgoingConnections` - Outgoing connections from one node.
- `allConnections` - Deduplication set for network connections.

### assignNetworkNodes

```ts
assignNetworkNodes(
  networkInstance: default,
  mlpNodeLayers: MlpNodeLayers,
): void
```

Assign ordered nodes to the network instance.

Parameters:
- `networkInstance` - Target network.
- `mlpNodeLayers` - Grouped node layers for this MLP.

### collectUniqueOutgoingConnections

```ts
collectUniqueOutgoingConnections(
  networkInstance: default,
): Set<default>
```

Collect unique outgoing connections across all network nodes.

Parameters:
- `networkInstance` - Target network.

Returns: Set of unique outgoing connections.

### connectLayerPair

```ts
connectLayerPair(
  sourceLayer: default[],
  targetLayer: default[],
): void
```

Fully connect every source node to every target node.

Parameters:
- `sourceLayer` - Source layer.
- `targetLayer` - Target layer.

### connectMlpLayers

```ts
connectMlpLayers(
  mlpNodeLayers: MlpNodeLayers,
): void
```

Fully connect each adjacent layer in MLP order.

Parameters:
- `mlpNodeLayers` - Grouped node layers for this MLP.

### convertConnectionSetToArray

```ts
convertConnectionSetToArray(
  uniqueConnections: Set<default>,
): default[]
```

Convert a connection set into the canonical array format.

Parameters:
- `uniqueConnections` - Unique network connections.

Returns: Array of network connections.

### createHiddenLayers

```ts
createHiddenLayers(
  hiddenCounts: number[],
): default[][]
```

Create all hidden layers for an MLP topology.

Parameters:
- `hiddenCounts` - Hidden-layer node counts.

Returns: Hidden layers in forward order.

### createMLP

```ts
createMLP(
  inputCount: number,
  hiddenCounts: number[],
  outputCount: number,
): default
```

Contract for createMLP.

### createMlpNodeLayers

```ts
createMlpNodeLayers(
  inputCount: number,
  hiddenCounts: number[],
  outputCount: number,
): MlpNodeLayers
```

Build input, hidden, and output node layers for an MLP topology.

Parameters:
- `inputCount` - Number of input nodes.
- `hiddenCounts` - Hidden-layer node counts.
- `outputCount` - Number of output nodes.

Returns: Grouped node layers for MLP assembly.

### createNodesOfType

```ts
createNodesOfType(
  nodeCount: number,
  nodeType: "hidden" | "input" | "output",
): default[]
```

Create all nodes for a single fixed node type.

Parameters:
- `nodeCount` - Number of nodes to create.
- `nodeType` - Node type identifier.

Returns: Node list of the requested type.

### createOrderedNodeList

```ts
createOrderedNodeList(
  mlpNodeLayers: MlpNodeLayers,
): default[]
```

Build the canonical ordered node list used by the network.

Parameters:
- `mlpNodeLayers` - Grouped node layers for this MLP.

Returns: Ordered node list: input, hidden, then output.

### flattenNodeLayers

```ts
flattenNodeLayers(
  nodeLayers: default[][],
): default[]
```

Flatten layered node collections into a single ordered list.

Parameters:
- `nodeLayers` - Layered node collections.

Returns: Flattened node list.

### instantiateNetwork

```ts
instantiateNetwork(
  networkFactory: NetworkConstructor,
  inputCount: number,
  outputCount: number,
): default
```

Instantiate a new network using the runtime constructor.

Parameters:
- `networkFactory` - Network constructor function.
- `inputCount` - Number of input nodes.
- `outputCount` - Number of output nodes.

Returns: Newly instantiated network.

### markTopologyDirty

```ts
markTopologyDirty(
  networkInstance: default,
): void
```

Mark a network topology as dirty after structural edits.

Parameters:
- `networkInstance` - Network instance to mark.

### rebuildConnections

```ts
rebuildConnections(
  networkInstance: default,
): void
```

Rebuild the canonical connection array from all per-node outgoing lists.

Parameters:
- `networkInstance` - Target network.

## architecture/network/topology/network.topology.contract.utils.ts

### FeedForwardTopologyContractCarrier

Minimal runtime surface needed to read the active feed-forward contract.

Some callers have a full `Network` instance with `getTopologyIntent()`, while
others only hold a narrow runtime genome shape with the low-level acyclic flag.
This contract keeps both shapes usable from one small helper.

### getTopologyIntent

```ts
getTopologyIntent(): NetworkTopologyIntent
```

Read the public topology intent preserved on a network instance.

This accessor keeps the semantic contract visible to callers even though the
lower-level runtime ultimately enforces acyclicity through booleans and cache
invalidation.

Parameters:
- `this` - Target network instance.

Returns: Current topology intent.

### hasFeedForwardTopologyContract

```ts
hasFeedForwardTopologyContract(
  carrier: FeedForwardTopologyContractCarrier,
): boolean
```

Check whether a runtime shape currently carries the feed-forward contract.

The helper is intentionally conservative when callers are in a mismatched
transitional state: either an explicit `feed-forward` intent or a truthy
`_enforceAcyclic` flag is treated as a feed-forward contract. That keeps
mutation and crossover helpers from introducing recurrent structure into a
genome that still advertises acyclic semantics anywhere on its runtime seam.

Parameters:
- `carrier` - Narrow runtime shape or full network instance.

Returns: True when feed-forward semantics are currently enforced.

### setEnforceAcyclic

```ts
setEnforceAcyclic(
  flag: boolean,
): void
```

Toggle low-level acyclic enforcement while preserving a coherent public contract.

This exists for backward compatibility with callers that still use the legacy
boolean API instead of the semantic `topologyIntent` field.

Parameters:
- `this` - Target network instance.
- `flag` - Whether to enforce acyclic connectivity.

Returns: Nothing.

### setTopologyIntent

```ts
setTopologyIntent(
  topologyIntent: NetworkTopologyIntent,
): void
```

Set the public topology intent and synchronize low-level runtime flags.

Updating the semantic contract also updates acyclic enforcement and marks the
topological cache dirty so later activation paths rebuild consistent state.

Parameters:
- `this` - Target network instance.
- `topologyIntent` - Desired topology intent.

Returns: Nothing.

## architecture/network/topology/network.topology.architecture.utils.ts

### createArchitectureDescriptor

```ts
createArchitectureDescriptor(
  hiddenLayerSizes: number[],
  hasCycles: boolean,
  source: NetworkArchitectureSource,
  totalNodes: number,
  totalConnections: number,
): NetworkArchitectureDescriptor
```

Creates the final immutable descriptor shape used by telemetry and UI code.

Keeping descriptor assembly in one place ensures every resolution strategy
returns the same payload contract and avoids accidental field drift.

Parameters:
- `hiddenLayerSizes` - Hidden-layer widths.
- `hasCycles` - Whether cycles were detected.
- `source` - Descriptor provenance.
- `totalNodes` - Node count.
- `totalConnections` - Connection count.

Returns: Descriptor object.

Example:

```ts
const descriptor = createArchitectureDescriptor([6, 3], false, 'graph-topology', 14, 25);
// descriptor.totalNodes === 14
```

### createDirectedEdgeList

```ts
createDirectedEdgeList(
  runtimeConnections: RuntimeConnectionLike[],
  nodeByIndex: Map<number, RuntimeNodeLike>,
): { fromIndex: number; toIndex: number; }[]
```

Produces a validated list of enabled directed edges.

Invalid references, disabled connections, and self-loops are removed so the
remaining edge list can be consumed safely by cycle and depth algorithms.

Parameters:
- `runtimeConnections` - Runtime connections.
- `nodeByIndex` - Indexed nodes.

Returns: Valid directed edges.

Example:

```ts
const edges = createDirectedEdgeList(runtimeConnections, nodeByIndex);
// edges -> [{ fromIndex: 0, toIndex: 3 }, ...]
```

### createNodeIndexMap

```ts
createNodeIndexMap(
  runtimeNodes: RuntimeNodeLike[],
): Map<number, RuntimeNodeLike>
```

Builds a node lookup table keyed by stable index.

Runtime objects may omit `index`; in that case the current array position is
used as a deterministic fallback to keep downstream graph logic total.

Parameters:
- `runtimeNodes` - Runtime nodes.

Returns: Node map keyed by stable node index.

Example:

```ts
const nodeByIndex = createNodeIndexMap(nodes);
// nodeByIndex.get(0) -> first node or node with explicit index 0
```

### describeArchitecture

```ts
describeArchitecture(
  network: default,
): NetworkArchitectureDescriptor
```

Describes network architecture for diagnostics, telemetry, and UI rendering.

This function prefers factual sources over heuristics so downstream tooling
can rely on the descriptor while still receiving useful output for partially
specified runtime graphs.

Resolution priority is intentionally explicit:
1) node `layer` metadata (factual when present)
2) graph-derived feed-forward depth layering (factual for acyclic graphs)
3) hidden-node count fallback (heuristic inference)

Parameters:
- `network` - Runtime network instance.

Returns: Stable architecture descriptor.

Example:

```ts
const descriptor = describeArchitecture(network);
// descriptor.hiddenLayerSizes -> [8, 4]
// descriptor.source -> 'layer-metadata' | 'graph-topology' | 'inferred'
```

### isHiddenNode

```ts
isHiddenNode(
  runtimeNode: RuntimeNodeLike,
): boolean
```

Identifies whether a runtime node should be treated as hidden for topology
reconstruction and fallback inference.

Parameters:
- `runtimeNode` - Candidate node.

Returns: True when node type is hidden.

Example:

```ts
if (isHiddenNode(node)) {
  // Include in hidden-layer counting
}
```

### isHydratedDescriptorCompatible

```ts
isHydratedDescriptorCompatible(
  network: default,
  hydratedDescriptor: NetworkArchitectureDescriptor | undefined,
): boolean
```

Check whether hydrated descriptor metadata still matches the current graph shape.

Parameters:
- `network` - Runtime network instance.
- `hydratedDescriptor` - Optional hydrated descriptor candidate.

Returns: True when hydrated descriptor can safely stand in for the inferred result.

### resolveArchitectureDescriptor

```ts
resolveArchitectureDescriptor(
  network: default,
): NetworkArchitectureDescriptor
```

Resolve the public architecture descriptor, preferring live graph facts and
falling back to hydrated serialization metadata only when the live result is
still purely inferred.

This helper keeps the descriptor ownership story in one chapter: topology
owns the live analysis while serialization can optionally hydrate a cached
descriptor that remains safe to reuse when the runtime graph shape matches.

Parameters:
- `network` - Runtime network instance.

Returns: Public architecture descriptor for telemetry and UI consumers.

### resolveCycleStateAndTopoOrder

```ts
resolveCycleStateAndTopoOrder(
  nodeByIndex: Map<number, RuntimeNodeLike>,
  directedEdges: { fromIndex: number; toIndex: number; }[],
): { topologicalOrder: number[]; hasCycles: boolean; }
```

Resolves cycle presence and, when possible, returns a topological order
using Kahn's algorithm.

A complete topological ordering implies an acyclic graph. If some nodes
remain unprocessed, at least one cycle exists.

Parameters:
- `nodeByIndex` - Indexed nodes.
- `directedEdges` - Directed edges.

Returns: Topological order and cycle status.

Example:

```ts
const { topologicalOrder, hasCycles } = resolveCycleStateAndTopoOrder(nodeByIndex, edges);
```

### resolveHiddenCountsByDepth

```ts
resolveHiddenCountsByDepth(
  nodeByIndex: Map<number, RuntimeNodeLike>,
  depthByNodeIndex: Map<number, number>,
): Map<number, number>
```

Aggregates hidden-node counts per derived depth.

This is the final transformation before emitting architecture widths:
hidden nodes are grouped by depth and counted in insertion-safe maps.

Parameters:
- `nodeByIndex` - Indexed nodes.
- `depthByNodeIndex` - Derived depths.

Returns: Hidden-node counts by depth.

Example:

```ts
const hiddenCountsByDepth = resolveHiddenCountsByDepth(nodeByIndex, depthByNodeIndex);
```

### resolveHiddenLayerSizesFromGraphTopology

```ts
resolveHiddenLayerSizesFromGraphTopology(
  runtimeNodes: RuntimeNodeLike[],
  runtimeConnections: RuntimeConnectionLike[],
): { hiddenLayerSizes: number[]; hasCycles: boolean; }
```

Derives hidden-layer widths from graph topology when no explicit layer
metadata is available.

The method computes a topological depth model for acyclic graphs; cyclic
graphs are flagged and intentionally return no width inference because depth
is not well-defined in recurrent loops.

Parameters:
- `runtimeNodes` - Runtime nodes.
- `runtimeConnections` - Runtime connections.

Returns: Hidden-layer widths derived from acyclic topology and cycle flag.

Example:

```ts
const { hiddenLayerSizes, hasCycles } = resolveHiddenLayerSizesFromGraphTopology(nodes, edges);
```

### resolveHiddenLayerSizesFromLayerMetadata

```ts
resolveHiddenLayerSizesFromLayerMetadata(
  runtimeNodes: RuntimeNodeLike[],
): number[]
```

Resolves hidden-layer widths from explicit `node.layer` metadata.

This is treated as the most trustworthy source because layer assignment is
usually produced by architecture-aware builders and does not depend on
topological reconstruction.

Parameters:
- `runtimeNodes` - Runtime nodes.

Returns: Hidden-layer widths from explicit node.layer metadata.

Example:

```ts
// Hidden nodes in layers 1, 1, and 2 -> [2, 1]
const sizes = resolveHiddenLayerSizesFromLayerMetadata(nodes);
```

### resolveNodeDepthByIndex

```ts
resolveNodeDepthByIndex(
  nodeByIndex: Map<number, RuntimeNodeLike>,
  directedEdges: { fromIndex: number; toIndex: number; }[],
  topologicalOrder: number[],
): Map<number, number>
```

Computes node depth (feed-forward distance from inputs) for an acyclic graph.

Depth assignment is parent-driven: each node depth is one plus the maximum
resolved parent depth. Nodes with no resolved parents are skipped.

Parameters:
- `nodeByIndex` - Indexed nodes.
- `directedEdges` - Directed edges.
- `topologicalOrder` - Acyclic topological order.

Returns: Derived depth by node index.

Example:

```ts
const depthByNodeIndex = resolveNodeDepthByIndex(nodeByIndex, edges, topologicalOrder);
```
