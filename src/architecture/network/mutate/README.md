# architecture/network/mutate

Mutation orchestration entrypoint for network-level structural and parametric edits.

This module intentionally stays lightweight:
- It resolves the incoming mutation request into a dispatch key.
- It selects a concrete handler from the dispatch table.
- It delegates execution and marks topology caches dirty after successful handling.

Handler-specific logic lives in dedicated helper files so this module remains a stable,
high-level control surface for mutation flow.

## architecture/network/mutate/network.mutate.utils.types.ts

### NODE_TYPE_INPUT

Canonical node-type literal for input nodes.

### NODE_TYPE_OUTPUT

Canonical node-type literal for output nodes.

### NODE_TYPE_HIDDEN

Canonical node-type literal for hidden nodes.

### RECURRENT_BLOCK_LSTM

Canonical recurrent block literal for LSTM expansion.

### RECURRENT_BLOCK_GRU

Canonical recurrent block literal for GRU expansion.

### SINGLE_UNIT_RECURRENT_BLOCK_WIDTH

Width used when creating a minimal recurrent block.

### SUB_NODE_STABILITY_WEIGHT_DELTA

Weight delta used to keep mutation side effects numerically observable.

### GATE_REASSIGN_THRESHOLD

Threshold used for random 50/50 gating decisions.

### DEFAULT_MUTATION_MIN

Default minimum mutation value when no method override is provided.

### DEFAULT_MUTATION_MAX

Default maximum mutation value when no method override is provided.

### MIN_REDUNDANT_CONNECTION_COUNT

Minimum redundant in/out degree required before removing a connection.

### MIN_SWAPPABLE_NODE_COUNT

Minimum node count required to perform swap-node mutation.

### WARNING_NO_HIDDEN_NODES_TO_REMOVE

Message emitted when no hidden node can be removed.

### WARNING_NO_ACTIVATION_MUTATION_TARGETS

Message emitted when activation mutation has no eligible nodes.

### WARNING_SELF_CONNECTIONS_ALREADY_PRESENT

Message emitted when all self-connection candidates are already occupied.

### WARNING_NO_SELF_CONNECTIONS_TO_REMOVE

Message emitted when no self-connections are available to remove.

### WARNING_ALL_CONNECTIONS_GATED

Message emitted when gating cannot be added because all are already gated.

### WARNING_NO_GATED_CONNECTIONS_TO_REMOVE

Message emitted when no gate exists to remove.

### UNKNOWN_MUTATION_WARNING_PREFIX

Prefix for unknown-mutation warning logs.

### ERROR_NO_MUTATE_METHOD

Error emitted when mutate is called without a valid method.

### BATCH_NORM_FLAG_KEY

Internal node field used to enable batch normalization.

## architecture/network/mutate/network.mutate.utils.ts

### mutateImpl

```ts
mutateImpl(
  method: MutationMethod | undefined,
): void
```

Public entry point: apply a single mutation operator to the network.

Runtime flow:
1. Validate mutation input.
2. Resolve the mutation key from string/object/reference forms.
3. Resolve a concrete handler from the dispatch table.
4. Delegate execution and mark topology-derived caches dirty.

Error and warning behavior:
- Throws when no method is provided.
- Emits a warning and no-ops when an unknown method key is received.

Parameters:
- `this` - - Network instance.
- `method` - - Mutation enum value or descriptor object.

Returns: Nothing.

Example:

```ts
network.mutate('ADD_NODE');
network.mutate({ name: 'MOD_WEIGHT', min: -0.1, max: 0.1 });
```

### MutationMethod

Mutation method descriptor shape.

## architecture/network/mutate/network.mutate.public.utils.ts

Public structural-mutation helpers that stay outside the mutation dispatch table.

This file owns small graph-editing methods that are exposed directly on
`Network` for callers who want one specific structural operation without
going through the broader mutation-method dispatch flow.

### addNodeBetweenImpl

```ts
addNodeBetweenImpl(): void
```

Split one randomly selected connection by inserting a hidden node.

This preserves the long-standing public `addNodeBetween()` behavior:
- it does not opt into `ADD_NODE` deterministic-chain policy,
- it preserves the original source-edge weight on the first new connection,
- it uses `1` for the hidden-to-target edge to keep the split easy to reason about.

Parameters:
- `this` - Target network instance.

Returns: Nothing.

## architecture/network/mutate/network.mutate.dispatch.utils.ts

Mutation-key normalization and warning helpers used by the mutate orchestrator.

Responsibilities:
- Normalize string/object/reference mutation inputs into a dispatch key.
- Emit unknown-mutation warnings when warning mode is enabled.

The helpers in this module are intentionally side-effect-light, except for optional
warning emission, so orchestration code can remain deterministic and easy to inspect.

### resolveMutationKey

```ts
resolveMutationKey(
  method: MutationMethod,
): string | undefined
```

Resolves a mutation dispatch key from user-provided method input.

Resolution order:
1. Use string input directly.
2. Use direct object identity fields (`name`, `type`, `identity`).
3. Fall back to identity-reference comparison against known mutation objects.

Parameters:
- `method` - - Mutation method input.

Returns: Dispatch key or undefined.

Example:

```ts
const key = resolveMutationKey('ADD_NODE');
```

### warnUnknownMutation

```ts
warnUnknownMutation(
  mutationKey: string | undefined,
): void
```

Emits unknown-mutation warning when configured.

This helper intentionally no-ops when warnings are disabled so callers can invoke it
without repeating feature-flag checks.

Parameters:
- `mutationKey` - - Resolved mutation key.

Returns: Nothing.

### isMutationMethodKeyString

```ts
isMutationMethodKeyString(
  method: MutationMethod,
): boolean
```

Checks whether mutation input is already a direct key string.

Parameters:
- `method` - - Mutation method input.

Returns: True when method is a key string.

### resolveMutationKeyFromObject

```ts
resolveMutationKeyFromObject(
  methodObject: { [key: string]: unknown; name?: string | undefined; type?: string | undefined; identity?: string | undefined; max?: number | undefined; min?: number | undefined; mutateOutput?: boolean | undefined; },
): string | undefined
```

Resolves mutation key from object-form descriptor.

Parameters:
- `methodObject` - - Mutation method object.

Returns: Dispatch key or undefined.

### resolveDirectMutationKey

```ts
resolveDirectMutationKey(
  methodObject: { [key: string]: unknown; name?: string | undefined; type?: string | undefined; identity?: string | undefined; max?: number | undefined; min?: number | undefined; mutateOutput?: boolean | undefined; },
): string | undefined
```

Resolves direct object fields that can represent a mutation key.

Parameters:
- `methodObject` - - Mutation method object.

Returns: Direct key or undefined.

### findMutationKeyByIdentityReference

```ts
findMutationKeyByIdentityReference(
  method: MutationMethod,
): string | undefined
```

Resolves a mutation key by direct identity-reference comparison.

Parameters:
- `method` - - Mutation object reference.

Returns: Matching mutation key or undefined.

## architecture/network/mutate/network.mutate.handlers.utils.ts

Concrete mutation handler implementations used by the network mutate orchestrator.

Organization:
- Exported functions represent public mutation operations mapped by dispatch key.
- Internal helpers encapsulate candidate collection, validation, and graph rewiring.
- Shared constants and warning strings are imported from `network.mutate.utils.types.ts`
  to keep cross-file contracts explicit and avoid circular dependencies.

Behavioral notes:
- Handlers preserve fail-soft semantics where possible (return early when no candidate exists).
- Acyclic mode checks are enforced in handlers that could introduce recurrence.
- Randomness is sourced from network mutation internals for reproducible deterministic flows.

### addNode

```ts
addNode(): void
```

Adds one hidden node by splitting an existing connection.

Execution modes:
- Deterministic chain mode grows a linear input→...→output chain.
- Standard mode splits a randomly selected forward connection.

Parameters:
- `this` - - Bound network.

Returns: Nothing.

### subNode

```ts
subNode(): void
```

Removes one hidden node and applies a tiny weight nudge for numerical continuity.

The stability nudge helps keep downstream mutation effects observable in edge cases
where node removal substantially changes effective signal flow.

Parameters:
- `this` - - Bound network.

Returns: Nothing.

### addConn

```ts
addConn(): void
```

Adds one forward connection between currently unconnected eligible node pairs.

Candidate generation respects node ordering so the added edge is feed-forward.

Parameters:
- `this` - - Bound network.

Returns: Nothing.

### subConn

```ts
subConn(): void
```

Removes one forward connection when structural redundancy constraints are satisfied.

Constraints require endpoint redundancy and avoid disconnecting peer-layer groups.

Parameters:
- `this` - - Bound network.

Returns: Nothing.

### modWeight

```ts
modWeight(
  method: MutationMethod | undefined,
): void
```

Perturbs one connection weight using a uniform delta sampled from configured bounds.

The candidate pool includes standard and self-connections.

Parameters:
- `this` - - Bound network.
- `method` - - Optional method descriptor.

Returns: Nothing.

### modBias

```ts
modBias(
  method: MutationMethod | undefined,
): void
```

Mutates bias parameters on one random non-input node.

Output nodes remain eligible for this operator.

Parameters:
- `this` - - Bound network.
- `method` - - Optional method descriptor.

Returns: Nothing.

### modActivation

```ts
modActivation(
  method: MutationMethod | undefined,
): void
```

Mutates activation function on one random non-input node.

Output-node eligibility is controlled by `method.mutateOutput` when provided.

Parameters:
- `this` - - Bound network.
- `method` - - Optional method descriptor.

Returns: Nothing.

### addSelfConn

```ts
addSelfConn(): void
```

Adds one self-connection on an eligible node that does not already have one.

This operation is skipped in acyclic mode.

Parameters:
- `this` - - Bound network.

Returns: Nothing.

### subSelfConn

```ts
subSelfConn(): void
```

Removes one existing self-connection chosen at random.

Parameters:
- `this` - - Bound network.

Returns: Nothing.

### addGate

```ts
addGate(): void
```

Assigns a random eligible node as gater for a random ungated connection.

Candidate pool includes normal and self-connections.

Parameters:
- `this` - - Bound network.

Returns: Nothing.

### subGate

```ts
subGate(): void
```

Removes gating from one randomly selected gated connection.

Parameters:
- `this` - - Bound network.

Returns: Nothing.

### addBackConn

```ts
addBackConn(): void
```

Adds one backward (recurrent) connection between eligible node pairs.

This operation is skipped in acyclic mode.

Parameters:
- `this` - - Bound network.

Returns: Nothing.

### subBackConn

```ts
subBackConn(): void
```

Removes one backward connection that satisfies redundancy constraints.

Parameters:
- `this` - - Bound network.

Returns: Nothing.

### swapNodes

```ts
swapNodes(
  method: MutationMethod | undefined,
): void
```

Swaps bias and activation squash functions between two distinct mutable nodes.

This provides a lightweight structural-parameter recombination without changing
graph connectivity.

Parameters:
- `this` - - Bound network.
- `method` - - Optional method descriptor.

Returns: Nothing.

### addLSTMNode

```ts
addLSTMNode(): void
```

Replaces one connection by inserting a minimal LSTM recurrent block.

Parameters:
- `this` - - Bound network.

Returns: Nothing.

### addGRUNode

```ts
addGRUNode(): void
```

Replaces one connection by inserting a minimal GRU recurrent block.

Parameters:
- `this` - - Bound network.

Returns: Nothing.

### reinitWeight

```ts
reinitWeight(
  method: MutationMethod | undefined,
): void
```

Reinitializes incoming, outgoing, and self-connection weights for one target node.

Weight sampling bounds come from method overrides or default mutation bounds.

Parameters:
- `this` - - Bound network.
- `method` - - Optional method descriptor.

Returns: Nothing.

### batchNorm

```ts
batchNorm(): void
```

Enables the internal batch-normalization flag on one random hidden node.

Parameters:
- `this` - - Bound network.

Returns: Nothing.

### asMutationProps

```ts
asMutationProps(
  network: default,
): NetworkMutationProps
```

Converts a network to its internal mutation runtime shape.

Parameters:
- `network` - - Network to convert.

Returns: Runtime mutation props.

### markTopoDirtyIfAcyclic

```ts
markTopoDirtyIfAcyclic(
  mutationProps: NetworkMutationProps,
): void
```

Marks topology caches dirty when acyclic mode is enforced.

Parameters:
- `mutationProps` - - Runtime mutation props.

Returns: Nothing.

### findFirstNodeByType

```ts
findFirstNodeByType(
  network: default,
  nodeType: string,
): default | undefined
```

Returns the first node by type.

Parameters:
- `network` - - Target network.
- `nodeType` - - Node type to match.

Returns: Matching node or undefined.

### pickRandomEntry

```ts
pickRandomEntry(
  entries: T[],
  randomValue: () => number,
): T | undefined
```

Selects a random array element.

Parameters:
- `entries` - - Source entries.
- `randomValue` - - Random generator.

Returns: Random entry or undefined when empty.

### createHiddenNode

```ts
createHiddenNode(
  randomValue: () => number,
): default
```

Creates a hidden node with random activation mutation.

Parameters:
- `randomValue` - - Random generator.

Returns: Hidden node.

### insertNodeBeforeOutputTail

```ts
insertNodeBeforeOutputTail(
  network: default,
  nodeToInsert: default,
  targetNode: default,
  mutationProps: NetworkMutationProps,
): void
```

Inserts a node before output tail while preserving output block ordering.

Parameters:
- `network` - - Target network.
- `nodeToInsert` - - Node to insert.
- `targetNode` - - Target node for insertion alignment.
- `mutationProps` - - Runtime mutation props.

Returns: Nothing.

### findConnection

```ts
findConnection(
  network: default,
  fromNode: default,
  toNode: default,
): default | undefined
```

Gets a connection between two nodes when it exists.

Parameters:
- `network` - - Target network.
- `fromNode` - - Source node.
- `toNode` - - Target node.

Returns: Matching connection or undefined.

### ensureConnection

```ts
ensureConnection(
  network: default,
  fromNode: default,
  toNode: default,
): default | undefined
```

Ensures a connection exists and returns it.

Parameters:
- `network` - - Target network.
- `fromNode` - - Source node.
- `toNode` - - Target node.

Returns: Existing or created connection.

### tryDisconnectConnection

```ts
tryDisconnectConnection(
  network: default,
  connection: default,
): void
```

Disconnects a connection pair while suppressing errors.

Parameters:
- `network` - - Target network.
- `connection` - - Connection to remove.

Returns: Nothing.

### addNodeDeterministicChain

```ts
addNodeDeterministicChain(
  network: default,
  mutationProps: NetworkMutationProps,
): void
```

Applies deterministic chain-growth ADD_NODE mutation.

Parameters:
- `network` - - Target network.
- `mutationProps` - - Runtime mutation props.

Returns: Nothing.

### resolveDeterministicChainMutationContext

```ts
resolveDeterministicChainMutationContext(
  network: default,
  mutationProps: NetworkMutationProps,
): DeterministicChainMutationContext | undefined
```

Resolves all deterministic add-node prerequisites into one context object.

Parameters:
- `network` - - Target network.
- `mutationProps` - - Runtime mutation props.

Returns: Deterministic context or undefined when one or more prerequisites fail.

### resolveInputOutputEndpoints

```ts
resolveInputOutputEndpoints(
  network: default,
): InputOutputEndpoints | undefined
```

Resolves input/output endpoints required for seed and deterministic flows.

Parameters:
- `network` - - Target network.

Returns: Endpoint nodes or undefined when missing.

### initializeDeterministicChain

```ts
initializeDeterministicChain(
  network: default,
  mutationProps: NetworkMutationProps,
  inputNode: default,
  outputNode: default,
): void
```

Initializes deterministic chain storage and seed edge.

Parameters:
- `network` - - Target network.
- `mutationProps` - - Runtime mutation props.
- `inputNode` - - Input node.
- `outputNode` - - Output node.

Returns: Nothing.

### pruneDeterministicChainExtraEdges

```ts
pruneDeterministicChainExtraEdges(
  network: default,
  deterministicChain: default[],
  outputNode: default,
): void
```

Prunes side edges from chain nodes to preserve linear deterministic depth.

Parameters:
- `network` - - Target network.
- `deterministicChain` - - Chain node list.
- `outputNode` - - Output node.

Returns: Nothing.

### resolveExpectedChainTarget

```ts
resolveExpectedChainTarget(
  deterministicChain: default[],
  chainNodeIndex: number,
  outputNode: default,
): default
```

Resolves the expected outgoing target for a chain node position.

Parameters:
- `deterministicChain` - - Chain node list.
- `chainNodeIndex` - - Current chain index.
- `outputNode` - - Terminal output node.

Returns: Expected successor target.

### disconnectUnexpectedOutgoingConnections

```ts
disconnectUnexpectedOutgoingConnections(
  network: default,
  chainNode: default,
  expectedTargetNode: default,
): void
```

Removes outgoing connections that do not match the expected chain target.

Parameters:
- `network` - - Target network.
- `chainNode` - - Node whose outgoing edges are validated.
- `expectedTargetNode` - - Allowed outgoing target.

Returns: Nothing.

### addNodeRandomSplit

```ts
addNodeRandomSplit(
  network: default,
  mutationProps: NetworkMutationProps,
): void
```

Applies non-deterministic ADD_NODE by splitting a random connection.

Parameters:
- `network` - - Target network.
- `mutationProps` - - Runtime mutation props.

Returns: Nothing.

### splitConnectionThroughHiddenNode

```ts
splitConnectionThroughHiddenNode(
  network: default,
  mutationProps: NetworkMutationProps,
  connectionToSplit: default,
): ConnectionSplitResult
```

Replaces one connection by inserting a hidden node and reconnecting edges.

Parameters:
- `network` - - Target network.
- `mutationProps` - - Runtime mutation props.
- `connectionToSplit` - - Connection to split.

Returns: Split result values.

### tryReassignGateAfterSplit

```ts
tryReassignGateAfterSplit(
  network: default,
  randomValue: () => number,
  splitResult: ConnectionSplitResult,
): void
```

Reassigns prior gater to one of the new split connections when possible.

Parameters:
- `network` - - Target network.
- `randomValue` - - Random generator.
- `splitResult` - - Split result values.

Returns: Nothing.

### ensureSeedForwardConnectionWhenEmpty

```ts
ensureSeedForwardConnectionWhenEmpty(
  network: default,
): boolean
```

Ensures a seed input->output connection exists when connection list is empty.

Parameters:
- `network` - - Target network.

Returns: True when mutation may continue.

### selectHiddenNodeForRemoval

```ts
selectHiddenNodeForRemoval(
  network: default,
  hiddenNodes: default[],
): default | undefined
```

Selects a hidden node candidate for SUB_NODE mutation.

Parameters:
- `network` - - Target network.
- `hiddenNodes` - - Hidden nodes.

Returns: Selected hidden node.

### removeHiddenNodeAndApplyStabilityNudge

```ts
removeHiddenNodeAndApplyStabilityNudge(
  network: default,
  hiddenNode: default,
): void
```

Removes selected hidden node and applies stability nudge.

Parameters:
- `network` - - Target network.
- `hiddenNode` - - Hidden node to remove.

Returns: Nothing.

### applyFirstConnectionStabilityNudge

```ts
applyFirstConnectionStabilityNudge(
  network: default,
): void
```

Applies tiny stability nudge to the first remaining connection.

Parameters:
- `network` - - Target network.

Returns: Nothing.

### collectNodesByType

```ts
collectNodesByType(
  network: default,
  nodeType: string,
): default[]
```

Collects nodes by type.

Parameters:
- `network` - - Source network.
- `nodeType` - - Desired node type.

Returns: Matching nodes.

### warnWhenEnabled

```ts
warnWhenEnabled(
  message: string,
): void
```

Emits a warning when warning mode is enabled.

Parameters:
- `message` - - Warning message.

Returns: Nothing.

### resolveSelectedForwardConnectionPair

```ts
resolveSelectedForwardConnectionPair(
  network: default,
  randomValue: () => number,
): NodePair | undefined
```

Resolves random selected forward connection pair.

Parameters:
- `network` - - Target network.
- `randomValue` - - Random generator.

Returns: Selected source/target pair.

### collectForwardConnectionCandidates

```ts
collectForwardConnectionCandidates(
  network: default,
): NodePair[]
```

Collects forward connection candidates.

Parameters:
- `network` - - Target network.

Returns: Candidate source/target pairs.

### collectForwardTraversalContexts

```ts
collectForwardTraversalContexts(
  network: default,
): ForwardCandidateTraversalContext[]
```

Collects forward traversal contexts for all eligible source nodes.

Parameters:
- `network` - - Target network.

Returns: Forward traversal contexts.

### collectForwardCandidatesFromContext

```ts
collectForwardCandidatesFromContext(
  forwardConnectionCandidates: NodePair[],
  traversalContext: ForwardCandidateTraversalContext,
): NodePair[]
```

Reduces one forward traversal context into candidate connection pairs.

Parameters:
- `forwardConnectionCandidates` - - Existing candidate pairs.
- `traversalContext` - - Source traversal context.

Returns: Updated candidate pairs.

### createForwardCandidateTraversalContext

```ts
createForwardCandidateTraversalContext(
  network: default,
  sourceNodeIndex: number,
): ForwardCandidateTraversalContext
```

Creates context for one forward-candidate source traversal pass.

Parameters:
- `network` - - Target network.
- `sourceNodeIndex` - - Current source index.

Returns: Immutable traversal context.

### collectForwardCandidatesForSource

```ts
collectForwardCandidatesForSource(
  traversalContext: ForwardCandidateTraversalContext,
): NodePair[]
```

Collects all forward candidates for one source traversal context.

Parameters:
- `traversalContext` - - Source traversal context.

Returns: Candidate source/target pairs.

### isForwardCandidateTargetAvailable

```ts
isForwardCandidateTargetAvailable(
  sourceNode: default,
  targetNode: default,
): boolean
```

Checks whether a forward candidate target is not already projected.

Parameters:
- `sourceNode` - - Candidate source node.
- `targetNode` - - Candidate target node.

Returns: True when connection may be added.

### disconnectConnectionPair

```ts
disconnectConnectionPair(
  network: default,
  selectedConnection: default,
): void
```

Disconnects selected connection by endpoints.

Parameters:
- `network` - - Target network.
- `selectedConnection` - - Connection to disconnect.

Returns: Nothing.

### collectRemovableForwardConnections

```ts
collectRemovableForwardConnections(
  network: default,
): default[]
```

Collects removable forward connections using redundancy constraints.

Parameters:
- `network` - - Target network.

Returns: Removable forward connections.

### isRemovableForwardConnection

```ts
isRemovableForwardConnection(
  network: default,
  candidateConnection: default,
): boolean
```

Evaluates whether a forward connection is safe to remove.

Parameters:
- `network` - - Target network.
- `candidateConnection` - - Connection under evaluation.

Returns: True when removable.

### isForwardConnectionStructurallyRemovable

```ts
isForwardConnectionStructurallyRemovable(
  network: default,
  candidateConnection: default,
): boolean
```

Checks structural preconditions for removable forward connections.

Parameters:
- `network` - - Target network.
- `candidateConnection` - - Candidate connection.

Returns: True when the connection is a forward edge with redundant endpoints.

### createDirectionalConnectionContext

```ts
createDirectionalConnectionContext(
  network: default,
  candidateConnection: default,
): DirectionalConnectionContext
```

Creates indexed directional context for a connection candidate.

Parameters:
- `network` - - Target network.
- `candidateConnection` - - Candidate connection.

Returns: Directional context.

### hasRedundantEndpoints

```ts
hasRedundantEndpoints(
  candidateConnection: default,
): boolean
```

Checks whether both endpoints maintain at least one redundant edge.

Parameters:
- `candidateConnection` - - Candidate connection.

Returns: True when endpoint redundancy exists.

### isForwardDirectionalContext

```ts
isForwardDirectionalContext(
  directionContext: DirectionalConnectionContext,
): boolean
```

Checks whether a directional context represents a forward edge.

Parameters:
- `directionContext` - - Directional context.

Returns: True when forward.

### wouldDisconnectTargetPeerLayerGroup

```ts
wouldDisconnectTargetPeerLayerGroup(
  network: default,
  candidateConnection: default,
): boolean
```

Determines whether removal would disconnect a target peer-layer group.

Parameters:
- `network` - - Target network.
- `candidateConnection` - - Connection under evaluation.

Returns: True when peer group would be disconnected.

### countSourceConnectionsIntoPeerSet

```ts
countSourceConnectionsIntoPeerSet(
  network: default,
  candidateConnection: default,
  targetLayerPeers: default[],
): number
```

Counts source-originated connections that end inside the target peer set.

Parameters:
- `network` - - Target network.
- `candidateConnection` - - Candidate connection.
- `targetLayerPeers` - - Peer-set nodes.

Returns: Number of source-to-peer connections.

### createSourcePeerConnectionCountContext

```ts
createSourcePeerConnectionCountContext(
  candidateConnection: default,
  targetLayerPeers: default[],
): SourcePeerConnectionCountContext
```

Creates immutable context for source-to-peer connection counting.

Parameters:
- `candidateConnection` - - Candidate connection.
- `targetLayerPeers` - - Peer-set nodes.

Returns: Count context.

### countConnectionWhenSourceTargetsPeer

```ts
countConnectionWhenSourceTargetsPeer(
  peerConnectionsFromSource: number,
  existingConnection: default,
  countContext: SourcePeerConnectionCountContext,
): number
```

Counts one connection when it originates from source and targets a peer.

Parameters:
- `peerConnectionsFromSource` - - Current count.
- `existingConnection` - - Existing network connection.
- `countContext` - - Count context.

Returns: Updated count.

### collectTargetLayerPeers

```ts
collectTargetLayerPeers(
  network: default,
  targetNode: default,
): default[]
```

Collects peers around a target node in the same type/layer neighborhood.

Parameters:
- `network` - - Target network.
- `targetNode` - - Node whose peers are collected.

Returns: Peer nodes.

### isTargetLayerPeer

```ts
isTargetLayerPeer(
  candidateNode: default,
  candidateNodeIndex: number,
  peerContext: TargetLayerPeerContext,
): boolean
```

Checks whether candidate node belongs to the target peer-layer set.

Parameters:
- `candidateNode` - - Candidate node.
- `candidateNodeIndex` - - Candidate node index.
- `peerContext` - - Peer traversal context.

Returns: True when candidate is an eligible peer.

### createTargetLayerPeerContext

```ts
createTargetLayerPeerContext(
  network: default,
  targetNode: default,
): TargetLayerPeerContext
```

Creates immutable context for peer-layer collection.

Parameters:
- `network` - - Target network.
- `targetNode` - - Peer anchor node.

Returns: Peer traversal context.

### isTargetLayerPeerTypeMatch

```ts
isTargetLayerPeerTypeMatch(
  candidateNode: default,
  peerContext: TargetLayerPeerContext,
): boolean
```

Checks whether node type matches the target-layer peer type.

Parameters:
- `candidateNode` - - Candidate node.
- `peerContext` - - Peer traversal context.

Returns: True when type matches.

### isTargetLayerPeerWithinDistance

```ts
isTargetLayerPeerWithinDistance(
  candidateNodeIndex: number,
  peerContext: TargetLayerPeerContext,
): boolean
```

Checks whether candidate index lies within allowed peer distance.

Parameters:
- `candidateNodeIndex` - - Candidate node index.
- `peerContext` - - Peer traversal context.

Returns: True when within distance.

### containsNode

```ts
containsNode(
  nodes: default[],
  node: default,
): boolean
```

Checks whether a node list contains a node reference.

Parameters:
- `nodes` - - Node list.
- `node` - - Node reference.

Returns: True when contained.

### collectAllConnections

```ts
collectAllConnections(
  network: default,
): default[]
```

Collects normal and self connections.

Parameters:
- `network` - - Target network.

Returns: Combined connections.

### resolveMethodObject

```ts
resolveMethodObject(
  method: MutationMethod | undefined,
): { [key: string]: unknown; name?: string | undefined; type?: string | undefined; identity?: string | undefined; max?: number | undefined; min?: number | undefined; mutateOutput?: boolean | undefined; }
```

Extracts method-object form when provided.

Parameters:
- `method` - - Optional mutation method.

Returns: Method object view.

### sampleUniform

```ts
sampleUniform(
  randomValue: () => number,
  minValue: number,
  maxValue: number,
): number
```

Samples a uniform value from [minValue, maxValue].

Parameters:
- `randomValue` - - Random generator.
- `minValue` - - Minimum value.
- `maxValue` - - Maximum value.

Returns: Sampled value.

### pickRandomNonInputNode

```ts
pickRandomNonInputNode(
  network: default,
  excludeOutputNodes: boolean,
  randomValue: () => number,
): default | undefined
```

Selects a random mutable non-input node.

Parameters:
- `network` - - Target network.
- `excludeOutputNodes` - - True to exclude outputs.
- `randomValue` - - Random generator.

Returns: Selected mutable node.

### collectMutableNonInputNodes

```ts
collectMutableNonInputNodes(
  network: default,
  excludeOutputNodes: boolean,
): default[]
```

Collects mutable non-input nodes.

Parameters:
- `network` - - Target network.
- `excludeOutputNodes` - - True to exclude output nodes.

Returns: Mutable nodes.

### resolveSelfConnectionTargetNode

```ts
resolveSelfConnectionTargetNode(
  network: default,
  randomValue: () => number,
): default | undefined
```

Resolves random node eligible for self-connection creation.

Parameters:
- `network` - - Target network.
- `randomValue` - - Random generator.

Returns: Selected node.

### collectNodesWithoutSelfLoop

```ts
collectNodesWithoutSelfLoop(
  network: default,
): default[]
```

Collects non-input nodes that do not have self loops.

Parameters:
- `network` - - Target network.

Returns: Eligible nodes.

### isNodeWithoutSelfLoop

```ts
isNodeWithoutSelfLoop(
  candidateNode: default,
): boolean
```

Checks whether a node currently has no self-loop connections.

Parameters:
- `candidateNode` - - Node under evaluation.

Returns: True when the node has no self-loop.

### collectUngatedConnections

```ts
collectUngatedConnections(
  network: default,
): default[]
```

Collects ungated connections including self-connections.

Parameters:
- `network` - - Target network.

Returns: Ungated connections.

### isUngatedConnection

```ts
isUngatedConnection(
  candidateConnection: default,
): boolean
```

Checks whether a connection has no gater attached.

Parameters:
- `candidateConnection` - - Connection under evaluation.

Returns: True when ungated.

### resolveSelectedBackwardConnectionPair

```ts
resolveSelectedBackwardConnectionPair(
  network: default,
  randomValue: () => number,
): NodePair | undefined
```

Resolves random selected backward connection pair.

Parameters:
- `network` - - Target network.
- `randomValue` - - Random generator.

Returns: Selected source/target pair.

### collectBackwardConnectionCandidates

```ts
collectBackwardConnectionCandidates(
  network: default,
): NodePair[]
```

Collects backward (recurrent) connection candidates.

Parameters:
- `network` - - Target network.

Returns: Candidate source/target pairs.

### collectBackwardTraversalContexts

```ts
collectBackwardTraversalContexts(
  network: default,
): BackwardCandidateTraversalContext[]
```

Collects backward traversal contexts for all eligible later nodes.

Parameters:
- `network` - - Target network.

Returns: Backward traversal contexts.

### collectBackwardCandidatesFromContext

```ts
collectBackwardCandidatesFromContext(
  backwardConnectionCandidates: NodePair[],
  traversalContext: BackwardCandidateTraversalContext,
): NodePair[]
```

Reduces one backward traversal context into candidate connection pairs.

Parameters:
- `backwardConnectionCandidates` - - Existing candidate pairs.
- `traversalContext` - - Later-node traversal context.

Returns: Updated candidate pairs.

### createBackwardCandidateTraversalContext

```ts
createBackwardCandidateTraversalContext(
  network: default,
  laterNodeIndex: number,
): BackwardCandidateTraversalContext
```

Creates context for one backward-candidate traversal pass.

Parameters:
- `network` - - Target network.
- `laterNodeIndex` - - Current later-node index.

Returns: Immutable traversal context.

### collectBackwardCandidatesForLaterNode

```ts
collectBackwardCandidatesForLaterNode(
  traversalContext: BackwardCandidateTraversalContext,
): NodePair[]
```

Collects all backward candidates for one later-node traversal context.

Parameters:
- `traversalContext` - - Later-node traversal context.

Returns: Candidate source/target pairs.

### isBackwardCandidateTargetAvailable

```ts
isBackwardCandidateTargetAvailable(
  laterNode: default,
  earlierNode: default,
): boolean
```

Checks whether a backward candidate target is not already projected.

Parameters:
- `laterNode` - - Candidate source node.
- `earlierNode` - - Candidate target node.

Returns: True when connection may be added.

### connectPair

```ts
connectPair(
  network: default,
  selectedConnectionPair: NodePair,
): void
```

Connects source/target node pair.

Parameters:
- `network` - - Target network.
- `selectedConnectionPair` - - Source/target pair.

Returns: Nothing.

### collectRemovableBackwardConnections

```ts
collectRemovableBackwardConnections(
  network: default,
): default[]
```

Collects removable backward connections using redundancy constraints.

Parameters:
- `network` - - Target network.

Returns: Removable backward connections.

### isRemovableBackwardConnection

```ts
isRemovableBackwardConnection(
  network: default,
  candidateConnection: default,
): boolean
```

Evaluates whether a backward connection is safe to remove.

Parameters:
- `network` - - Target network.
- `candidateConnection` - - Connection under evaluation.

Returns: True when removable.

### isBackwardDirectionalContext

```ts
isBackwardDirectionalContext(
  directionContext: DirectionalConnectionContext,
): boolean
```

Checks whether a directional context represents a backward edge.

Parameters:
- `directionContext` - - Directional context.

Returns: True when backward.

### collectSwappableNodesForMutation

```ts
collectSwappableNodesForMutation(
  network: default,
  method: MutationMethod | undefined,
): default[]
```

Collects swap-eligible nodes based on mutation configuration.

Parameters:
- `network` - - Target network.
- `method` - - Optional method descriptor.

Returns: Swap-eligible nodes.

### pickDistinctNodePair

```ts
pickDistinctNodePair(
  swappableNodes: default[],
  randomValue: () => number,
): DistinctNodePair | undefined
```

Picks two distinct nodes from a candidate set.

Parameters:
- `swappableNodes` - - Swap candidate nodes.
- `randomValue` - - Random generator.

Returns: Distinct pair or undefined.

### resolveDistinctPairWithKnownFirstNode

```ts
resolveDistinctPairWithKnownFirstNode(
  swappableNodes: default[],
  firstNode: default,
  randomValue: () => number,
): DistinctNodePair | undefined
```

Resolves a distinct pair when first node is already known.

Parameters:
- `swappableNodes` - - Swap candidate nodes.
- `firstNode` - - Chosen first node.
- `randomValue` - - Random generator.

Returns: Distinct pair or undefined.

### pickDistinctRandomNode

```ts
pickDistinctRandomNode(
  nodeCandidates: default[],
  excludedNode: default,
  randomValue: () => number,
): default | undefined
```

Picks a random node distinct from a given reference.

Parameters:
- `nodeCandidates` - - Candidate nodes.
- `excludedNode` - - Node to exclude.
- `randomValue` - - Random generator.

Returns: Distinct node or undefined.

### collectDistinctNodeCandidates

```ts
collectDistinctNodeCandidates(
  nodeCandidates: default[],
  excludedNode: default,
): default[]
```

Collects candidates that are distinct from an excluded node.

Parameters:
- `nodeCandidates` - - Candidate nodes.
- `excludedNode` - - Node to exclude.

Returns: Distinct candidates.

### swapNodeBiasAndSquash

```ts
swapNodeBiasAndSquash(
  firstNode: default,
  secondNode: default,
): void
```

Swaps bias and squash values between two nodes.

Parameters:
- `firstNode` - - First node.
- `secondNode` - - Second node.

Returns: Nothing.

### addRecurrentNode

```ts
addRecurrentNode(
  network: default,
  blockType: "lstm" | "gru",
): void
```

Shared orchestrator for recurrent-node mutation variants.

Parameters:
- `network` - - Target network.
- `blockType` - - Recurrent block type.

Returns: Nothing.

### expandConnectionWithRecurrentBlock

```ts
expandConnectionWithRecurrentBlock(
  network: default,
  connectionToExpand: default,
  blockType: "lstm" | "gru",
): void
```

Replaces one connection with a minimal recurrent block.

Parameters:
- `network` - - Target network.
- `connectionToExpand` - - Connection to replace.
- `blockType` - - Recurrent block type.

Returns: Nothing.

### disconnectConnectionAndGetGater

```ts
disconnectConnectionAndGetGater(
  network: default,
  connectionToExpand: default,
): default | null
```

Disconnects a connection and returns its previous gater.

Parameters:
- `network` - - Target network.
- `connectionToExpand` - - Connection being expanded.

Returns: Previous gater reference.

### reconnectThroughRecurrentLayer

```ts
reconnectThroughRecurrentLayer(
  network: default,
  connectionToExpand: default,
  recurrentLayer: RecurrentLayerShape,
): default | undefined
```

Reconnects a source/target pair through a recurrent layer.

Parameters:
- `network` - - Target network.
- `connectionToExpand` - - Original connection.
- `recurrentLayer` - - Recurrent-layer shape.

Returns: Latest newly created connection or undefined.

### tryGateLatestConnection

```ts
tryGateLatestConnection(
  network: default,
  previousGater: default | null,
  latestConnection: default | undefined,
): void
```

Gates the latest connection when both previous gater and target exist.

Parameters:
- `network` - - Target network.
- `previousGater` - - Previously assigned gater.
- `latestConnection` - - Connection to receive the gater.

Returns: Nothing.

### createRecurrentLayer

```ts
createRecurrentLayer(
  blockType: "lstm" | "gru",
): RecurrentLayerShape
```

Creates recurrent layer by type.

Parameters:
- `blockType` - - Recurrent block type.

Returns: Created recurrent layer.

### appendRecurrentLayerNodes

```ts
appendRecurrentLayerNodes(
  network: default,
  layerNodes: default[],
): void
```

Appends recurrent layer nodes as hidden nodes.

Parameters:
- `network` - - Target network.
- `layerNodes` - - Layer nodes.

Returns: Nothing.

### createConnectionGroupReinitContext

```ts
createConnectionGroupReinitContext(
  randomValue: () => number,
  methodObject: { [key: string]: unknown; name?: string | undefined; type?: string | undefined; identity?: string | undefined; max?: number | undefined; min?: number | undefined; mutateOutput?: boolean | undefined; },
): ConnectionGroupReinitContext
```

Creates immutable context for connection-group reinitialization.

Parameters:
- `randomValue` - - Random generator.
- `methodObject` - - Method override object.

Returns: Reinitialization context.

### collectConnectionGroupsForReinit

```ts
collectConnectionGroupsForReinit(
  targetNode: default,
): default[][]
```

Collects all connection groups affected by REINIT_WEIGHT.

Parameters:
- `targetNode` - - Node receiving the reinitialization.

Returns: Mutable connection groups.

### reinitializeConnectionGroupWeights

```ts
reinitializeConnectionGroupWeights(
  connections: default[],
  reinitContext: ConnectionGroupReinitContext,
): void
```

Reinitializes all weights in a connection group.

Parameters:
- `connections` - - Connection group.
- `randomValue` - - Random generator.
- `minWeight` - - Minimum sampled weight.
- `maxWeight` - - Maximum sampled weight.

Returns: Nothing.

### createWeightSamplingRangeContext

```ts
createWeightSamplingRangeContext(
  reinitContext: ConnectionGroupReinitContext,
): WeightSamplingRangeContext
```

Creates immutable sampling range context.

Parameters:
- `reinitContext` - - Reinitialization context.

Returns: Sampling range context.

### sampleUniformFromContext

```ts
sampleUniformFromContext(
  samplingContext: WeightSamplingRangeContext,
): number
```

Samples one weight using a prebuilt range context.

Parameters:
- `samplingContext` - - Sampling range context.

Returns: Sampled weight.

### enableNodeBatchNorm

```ts
enableNodeBatchNorm(
  node: default,
): void
```

Enables internal batch-norm flag on a node.

Parameters:
- `node` - - Node to flag.

Returns: Nothing.
