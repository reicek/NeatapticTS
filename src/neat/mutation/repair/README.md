# neat/mutation/repair

Dead-end repair helpers.

This chapter owns the connectivity repair pass that makes sure input,
output, and hidden nodes are not stranded without the minimum in/out edges
expected by the surrounding mutation and evaluation flows.

## neat/mutation/repair/mutation.dead-ends.ts

### chooseRandomNodeForDeadEnds

```ts
chooseRandomNodeForDeadEnds(
  candidates: NodeWithMetadata[],
  internal: NeatControllerForMutation,
): NodeWithMetadata | null
```

Choose a random node from candidates for dead-end repair.

Parameters:
- `candidates` - - candidate nodes
- `internal` - - neat controller context

Returns: selected node or null

### collectNodeGroupsForDeadEnds

```ts
collectNodeGroupsForDeadEnds(
  networkToInspect: GenomeWithMetadata,
): { inputNodes: NodeWithMetadata[]; outputNodes: NodeWithMetadata[]; hiddenNodes: NodeWithMetadata[]; }
```

Collect categorized node arrays for dead-end repair.

Parameters:
- `networkToInspect` - - network to inspect

Returns: grouped node arrays

### connectIfCandidatesExistForDeadEnds

```ts
connectIfCandidatesExistForDeadEnds(
  networkToEdit: GenomeWithMetadata,
  anchorNode: NodeWithMetadata,
  candidates: NodeWithMetadata[],
  reverse: boolean,
  internal: NeatControllerForMutation,
): void
```

Connect a node to a random candidate if candidates exist.

Parameters:
- `networkToEdit` - - network to edit
- `anchorNode` - - node to connect from/to
- `candidates` - - candidate nodes for connection
- `reverse` - - whether to connect candidate -> anchor
- `internal` - - neat controller context

Returns: void

### ensureHiddenConnectivityForDeadEnds

```ts
ensureHiddenConnectivityForDeadEnds(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToUse: { inputNodes: NodeWithMetadata[]; outputNodes: NodeWithMetadata[]; hiddenNodes: NodeWithMetadata[]; },
  internal: NeatControllerForMutation,
): void
```

Ensure hidden nodes have both incoming and outgoing connections.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `internal` - - neat controller context

Returns: void

### ensureInputConnectivityForDeadEnds

```ts
ensureInputConnectivityForDeadEnds(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToUse: { inputNodes: NodeWithMetadata[]; outputNodes: NodeWithMetadata[]; hiddenNodes: NodeWithMetadata[]; },
  internal: NeatControllerForMutation,
): void
```

Ensure all input nodes have at least one outgoing connection.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `internal` - - neat controller context

Returns: void

### ensureOutputConnectivityForDeadEnds

```ts
ensureOutputConnectivityForDeadEnds(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToUse: { inputNodes: NodeWithMetadata[]; outputNodes: NodeWithMetadata[]; hiddenNodes: NodeWithMetadata[]; },
  internal: NeatControllerForMutation,
): void
```

Ensure all output nodes have at least one incoming connection.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `internal` - - neat controller context

Returns: void

### hasIncomingForDeadEnds

```ts
hasIncomingForDeadEnds(
  node: NodeWithMetadata,
): boolean
```

Check whether a node has any incoming connections.

Parameters:
- `node` - - node to inspect

Returns: true when incoming connections exist

### hasOutgoingForDeadEnds

```ts
hasOutgoingForDeadEnds(
  node: NodeWithMetadata,
): boolean
```

Check whether a node has any outgoing connections.

Parameters:
- `node` - - node to inspect

Returns: true when outgoing connections exist

## neat/mutation/repair/mutation.min-hidden.ts

Minimum-hidden repair helpers.

This chapter owns the small maintenance pass that enforces a minimum hidden
node budget and rewires newly created hidden nodes so they remain connected
enough for later mutation and evaluation passes.

### chooseRandomNodeForMinHidden

```ts
chooseRandomNodeForMinHidden(
  candidates: NodeWithMetadata[],
  internal: NeatControllerForMutation,
): NodeWithMetadata | null
```

Choose a random node from a candidate list.

Parameters:
- `candidates` - - candidate nodes
- `internal` - - neat controller context

Returns: selected node or null

### collectNodeGroupsForMinHidden

```ts
collectNodeGroupsForMinHidden(
  networkToInspect: GenomeWithMetadata,
): { inputNodes: NodeWithMetadata[]; outputNodes: NodeWithMetadata[]; hiddenNodes: NodeWithMetadata[]; }
```

Collect categorized node arrays for the network.

Parameters:
- `networkToInspect` - - network to inspect

Returns: grouped node arrays

### computeMinimumHiddenSize

```ts
computeMinimumHiddenSize(
  inputCount: number,
  outputCount: number,
  explicitMinimumHidden: number | undefined,
  hiddenMultiplier: number | undefined,
): number
```

Compute the minimum hidden node count using explicit or multiplier-based settings.

Parameters:
- `inputCount` - - Number of input nodes in the network.
- `outputCount` - - Number of output nodes in the network.
- `explicitMinimumHidden` - - Optional explicit minimum hidden count.
- `hiddenMultiplier` - - Optional multiplier used when explicit minimum is absent.

Returns: Minimum hidden node requirement.

### ensureHiddenConnectivityForMinHidden

```ts
ensureHiddenConnectivityForMinHidden(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToUse: { inputNodes: NodeWithMetadata[]; outputNodes: NodeWithMetadata[]; hiddenNodes: NodeWithMetadata[]; },
  internal: NeatControllerForMutation,
): void
```

Ensure hidden nodes have both incoming and outgoing connections.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `internal` - - neat controller context

Returns: void

### ensureHiddenNodeCountForMinHidden

```ts
ensureHiddenNodeCountForMinHidden(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToEdit: { hiddenNodes: NodeWithMetadata[]; },
  minimumHidden: number,
  maxNodesLimit: number,
): Promise<void>
```

Ensure the network has at least the minimum number of hidden nodes.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToEdit` - - grouped node arrays
- `minimumHidden` - - minimum hidden nodes required
- `maxNodesLimit` - - maximum allowed nodes

Returns: Promise resolving when nodes are created

### ensureIncomingConnectionForMinHidden

```ts
ensureIncomingConnectionForMinHidden(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToUse: { inputNodes: NodeWithMetadata[]; hiddenNodes: NodeWithMetadata[]; },
  hiddenNode: NodeWithMetadata,
  internal: NeatControllerForMutation,
): void
```

Ensure a hidden node has at least one incoming connection.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `hiddenNode` - - hidden node to connect
- `internal` - - neat controller context

Returns: void

### ensureOutgoingConnectionForMinHidden

```ts
ensureOutgoingConnectionForMinHidden(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToUse: { outputNodes: NodeWithMetadata[]; hiddenNodes: NodeWithMetadata[]; },
  hiddenNode: NodeWithMetadata,
  internal: NeatControllerForMutation,
): void
```

Ensure a hidden node has at least one outgoing connection.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `hiddenNode` - - hidden node to connect
- `internal` - - neat controller context

Returns: void

### hasRequiredEndpointsForMinHidden

```ts
hasRequiredEndpointsForMinHidden(
  nodeGroupsToCheck: { inputNodes: NodeWithMetadata[]; outputNodes: NodeWithMetadata[]; },
): boolean
```

Check whether the network has at least one input and output node.

Parameters:
- `nodeGroupsToCheck` - - grouped node arrays

Returns: true when inputs and outputs are present

### MINIMUM_HIDDEN_BASELINE

Baseline minimum hidden nodes when no configuration is provided.

### rebuildNetworkConnectionsForMinHidden

```ts
rebuildNetworkConnectionsForMinHidden(
  networkToEdit: GenomeWithMetadata,
): Promise<void>
```

Rebuild connection caches after structural edits.

Parameters:
- `networkToEdit` - - network to rebuild

Returns: Promise resolving after rebuild completes

### resolveMaxNodesForMinHidden

```ts
resolveMaxNodesForMinHidden(
  internal: NeatControllerForMutation,
): number
```

Resolve the maximum node limit for the network.

Parameters:
- `internal` - - neat controller context

Returns: maximum node limit

### resolveMinHiddenForMinHidden

```ts
resolveMinHiddenForMinHidden(
  networkToInspect: GenomeWithMetadata,
  maxNodesLimit: number,
  multiplier: number | undefined,
  internal: NeatControllerForMutation,
): number
```

Resolve the minimum hidden node requirement for the network.

Parameters:
- `networkToInspect` - - network to inspect
- `maxNodesLimit` - - maximum allowed nodes
- `multiplier` - - optional size multiplier
- `internal` - - neat controller context

Returns: minimum hidden node count

### warnMissingEndpointsForMinHidden

```ts
warnMissingEndpointsForMinHidden(): void
```

Emit a warning when the network lacks input or output nodes.

Returns: void
