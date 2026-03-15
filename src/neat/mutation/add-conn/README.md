# neat/mutation/add-conn

Add-connection mutation helpers.

This chapter owns candidate-pair discovery, cycle guarding, and innovation-id
reuse for newly added structural connections.

## neat/mutation/add-conn/mutation.add-conn.ts

### assignInnovationForConnection

```ts
assignInnovationForConnection(
  connection: ConnectionWithMetadata,
  pairNodes: { symmetricKey: string; legacyForwardKey: string; legacyReverseKey: string; },
  internal: NeatControllerForMutation,
): void
```

Assign an innovation id for a new connection, reusing when possible.

Parameters:
- `connection` - - newly created connection
- `pairNodes` - - resolved pair metadata
- `internal` - - neat controller context

Returns: void

### buildLegacyKeyForConn

```ts
buildLegacyKeyForConn(
  sourceNode: NodeWithMetadata,
  targetNode: NodeWithMetadata,
): string
```

Build a legacy directional innovation key.

Parameters:
- `sourceNode` - - source node
- `targetNode` - - target node

Returns: directional innovation key

### buildSymmetricKeyForConn

```ts
buildSymmetricKeyForConn(
  sourceNode: NodeWithMetadata,
  targetNode: NodeWithMetadata,
): string
```

Build a symmetric innovation key for an unordered node pair.

Parameters:
- `sourceNode` - - source node
- `targetNode` - - target node

Returns: symmetric innovation key

### choosePairForConn

```ts
choosePairForConn(
  pairs: [NodeWithMetadata, NodeWithMetadata][],
  internal: NeatControllerForMutation,
): [NodeWithMetadata, NodeWithMetadata] | null
```

Choose a pair deterministically when only one candidate exists.

Parameters:
- `pairs` - - selection pool
- `internal` - - neat controller context

Returns: chosen pair or null

### collectCandidatePairsForConn

```ts
collectCandidatePairsForConn(
  genomeToInspect: GenomeWithMetadata,
): [NodeWithMetadata, NodeWithMetadata][]
```

Collect legal (from,to) node pairs not already connected.

Parameters:
- `genomeToInspect` - - genome to scan

Returns: candidate node pairs

### connectChosenPair

```ts
connectChosenPair(
  genomeToEdit: GenomeWithMetadata,
  pairNodes: { sourceNode: NodeWithMetadata; targetNode: NodeWithMetadata; },
): ConnectionWithMetadata | undefined
```

Create the connection for the chosen pair.

Parameters:
- `genomeToEdit` - - genome to edit
- `pairNodes` - - resolved pair nodes

Returns: created connection or undefined

### createsCycle

```ts
createsCycle(
  sourceNode: NodeWithMetadata,
  targetNode: NodeWithMetadata,
): boolean
```

Detect whether adding a connection would create a cycle.

Parameters:
- `sourceNode` - - source node of the new connection
- `targetNode` - - target node of the new connection

Returns: true when a cycle is detected

### filterPairsWithInnovations

```ts
filterPairsWithInnovations(
  pairs: [NodeWithMetadata, NodeWithMetadata][],
  internal: NeatControllerForMutation,
): [NodeWithMetadata, NodeWithMetadata][]
```

Filter candidate pairs that already have innovation reuse keys.

Parameters:
- `pairs` - - candidate node pairs
- `internal` - - neat controller context

Returns: reuse candidates

### resolvePairNodes

```ts
resolvePairNodes(
  chosenPair: [NodeWithMetadata, NodeWithMetadata],
): { sourceNode: NodeWithMetadata; targetNode: NodeWithMetadata; symmetricKey: string; legacyForwardKey: string; legacyReverseKey: string; }
```

Resolve nodes and innovation key details for a chosen pair.

Parameters:
- `chosenPair` - - pair to connect

Returns: resolved pair metadata

### selectPairPool

```ts
selectPairPool(
  allPairs: [NodeWithMetadata, NodeWithMetadata][],
  reusePairs: [NodeWithMetadata, NodeWithMetadata][],
): [NodeWithMetadata, NodeWithMetadata][]
```

Build the final selection pool based on reuse and hidden-node preference.

Parameters:
- `allPairs` - - all candidate pairs
- `reusePairs` - - pairs with historical innovations

Returns: selection pool

### shouldAbortForCycle

```ts
shouldAbortForCycle(
  genomeToInspect: GenomeWithMetadata,
  pairNodes: { sourceNode: NodeWithMetadata; targetNode: NodeWithMetadata; },
): boolean
```

Determine whether adding the connection would create a cycle.

Parameters:
- `genomeToInspect` - - genome to inspect
- `pairNodes` - - resolved pair nodes

Returns: true if the connection should be aborted
