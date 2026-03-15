# neat/mutation/add-node

Add-node mutation helpers.

This chapter owns the "split one connection into two" mechanics used by the
root mutation flow when preserving node-split innovation reuse.

## neat/mutation/add-node/mutation.add-node.ts

### applySplitWithExistingRecord

```ts
applySplitWithExistingRecord(
  genomeToEdit: GenomeWithMetadata,
  connectionToSplit: ConnectionWithMetadata,
  splitDescriptor: { splitKey: string; originalWeight: number; },
  splitRecord: { newNodeGeneId: number; inInnov: number; outInnov: number; },
  NodeClass: new (type: "input" | "output" | "hidden") => unknown,
): void
```

Apply a split using an existing innovation record.

Parameters:
- `genomeToEdit` - - genome being modified
- `connectionToSplit` - - connection being split
- `splitDescriptor` - - metadata for the split
- `splitRecord` - - existing innovation record
- `NodeClass` - - node constructor

Returns: void

### applySplitWithNewRecord

```ts
applySplitWithNewRecord(
  genomeToEdit: GenomeWithMetadata,
  connectionToSplit: ConnectionWithMetadata,
  splitDescriptor: { splitKey: string; originalWeight: number; },
  NodeClass: new (type: "input" | "output" | "hidden") => unknown,
  internal: NeatControllerForMutation,
): void
```

Apply a split and create a new innovation record.

Parameters:
- `genomeToEdit` - - genome being modified
- `connectionToSplit` - - connection being split
- `splitDescriptor` - - metadata for the split
- `NodeClass` - - node constructor
- `internal` - - neat controller context

Returns: void

### assignInnovationsForNewSplit

```ts
assignInnovationsForNewSplit(
  newNode: NodeWithMetadata,
  splitConnections: { incomingConnection?: ConnectionWithMetadata | undefined; outgoingConnection?: ConnectionWithMetadata | undefined; },
  internal: NeatControllerForMutation,
): { newNodeGeneId: number; inInnov: number; outInnov: number; }
```

Assign new innovations for a split and build the innovation record.

Parameters:
- `newNode` - - newly created hidden node
- `splitConnections` - - incoming/outgoing connections
- `internal` - - neat controller context

Returns: innovation record for the split

### buildSplitDescriptor

```ts
buildSplitDescriptor(
  connectionToSplit: ConnectionWithMetadata,
): { splitKey: string; originalWeight: number; }
```

Build the split descriptor used for innovation lookup and connection creation.

Parameters:
- `connectionToSplit` - - connection being split

Returns: split descriptor

### chooseConnectionForSplit

```ts
chooseConnectionForSplit(
  enabledConnectionsList: ConnectionWithMetadata[],
  internal: NeatControllerForMutation,
): ConnectionWithMetadata | null
```

Choose a random enabled connection to split.

Parameters:
- `enabledConnectionsList` - - candidate connections
- `internal` - - neat controller context

Returns: selected connection or null

### collectEnabledConnections

```ts
collectEnabledConnections(
  genomeToInspect: GenomeWithMetadata,
): ConnectionWithMetadata[]
```

Collect all enabled connections from a genome.

Parameters:
- `genomeToInspect` - - genome to inspect

Returns: enabled connections list

### connectSplitEdges

```ts
connectSplitEdges(
  genomeToEdit: GenomeWithMetadata,
  connectionToSplit: ConnectionWithMetadata,
  newNode: NodeWithMetadata,
  originalWeight: number,
): { incomingConnection?: ConnectionWithMetadata | undefined; outgoingConnection?: ConnectionWithMetadata | undefined; }
```

Create the incoming and outgoing split connections.

Parameters:
- `genomeToEdit` - - genome being modified
- `connectionToSplit` - - connection being split
- `newNode` - - newly created hidden node
- `originalWeight` - - weight to preserve on the outgoing connection

Returns: incoming/outgoing connection handles

### disconnectOriginalConnection

```ts
disconnectOriginalConnection(
  genomeToEdit: GenomeWithMetadata,
  connectionToRemove: ConnectionWithMetadata,
): void
```

Disconnect the original connection before inserting the split node.

Parameters:
- `genomeToEdit` - - genome to edit
- `connectionToRemove` - - original connection to remove

Returns: void

### ensureBootstrapConnection

```ts
ensureBootstrapConnection(
  genomeToSeed: GenomeWithMetadata,
  internal: NeatControllerForMutation,
): void
```

Ensure the genome has at least one connection by linking input to output.

Parameters:
- `genomeToSeed` - - genome that may need a bootstrap connection
- `internal` - - neat controller context retained for compatibility with existing callers

Returns: void

### findFirstNodeByType

```ts
findFirstNodeByType(
  genomeToSearch: GenomeWithMetadata,
  nodeType: "input" | "output" | "hidden",
): NodeWithMetadata | undefined
```

Find the first node of a given type.

Parameters:
- `genomeToSearch` - - genome whose nodes are searched
- `nodeType` - - node type to match

Returns: the first matching node or undefined

### resolveInsertIndex

```ts
resolveInsertIndex(
  genomeToEdit: GenomeWithMetadata,
  targetNode: NodeWithMetadata,
): number
```

Resolve the insertion index for a new node, keeping outputs at the end.

Parameters:
- `genomeToEdit` - - genome whose node list is updated
- `targetNode` - - original target node of the split connection

Returns: insertion index
