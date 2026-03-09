# architecture/network/mutate

## architecture/network/mutate/network.mutate.utils.types.ts

### BATCH_NORM_FLAG_KEY

### DEFAULT_MUTATION_MAX

### DEFAULT_MUTATION_MIN

### ERROR_NO_MUTATE_METHOD

### GATE_REASSIGN_THRESHOLD

### LAYER_MODULE_PATH

### MIN_REDUNDANT_CONNECTION_COUNT

### MIN_SWAPPABLE_NODE_COUNT

### NODE_TYPE_HIDDEN

### NODE_TYPE_INPUT

### NODE_TYPE_OUTPUT

### RECURRENT_BLOCK_GRU

### RECURRENT_BLOCK_LSTM

### SINGLE_UNIT_RECURRENT_BLOCK_WIDTH

### SUB_NODE_STABILITY_WEIGHT_DELTA

### UNKNOWN_MUTATION_WARNING_PREFIX

### WARNING_ALL_CONNECTIONS_GATED

### WARNING_NO_ACTIVATION_MUTATION_TARGETS

### WARNING_NO_GATED_CONNECTIONS_TO_REMOVE

### WARNING_NO_HIDDEN_NODES_TO_REMOVE

### WARNING_NO_SELF_CONNECTIONS_TO_REMOVE

### WARNING_SELF_CONNECTIONS_ALREADY_PRESENT

## architecture/network/mutate/network.mutate.utils.ts

### mutateImpl

`(method: import("src/architecture/network/network.types").MutationMethod | undefined) => void`

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

### MutationMethod

Mutation method descriptor shape.

## architecture/network/mutate/network.mutate.dispatch.utils.ts

### findMutationKeyByIdentityReference

`(method: import("src/architecture/network/network.types").MutationMethod) => string | undefined`

Resolves a mutation key by direct identity-reference comparison.

Parameters:
- `method` - - Mutation object reference.

Returns: Matching mutation key or undefined.

### isMutationMethodKeyString

`(method: import("src/architecture/network/network.types").MutationMethod) => boolean`

Checks whether mutation input is already a direct key string.

Parameters:
- `method` - - Mutation method input.

Returns: True when method is a key string.

### resolveDirectMutationKey

`(methodObject: { [key: string]: unknown; name?: string | undefined; type?: string | undefined; identity?: string | undefined; max?: number | undefined; min?: number | undefined; mutateOutput?: boolean | undefined; }) => string | undefined`

Resolves direct object fields that can represent a mutation key.

Parameters:
- `methodObject` - - Mutation method object.

Returns: Direct key or undefined.

### resolveMutationKey

`(method: import("src/architecture/network/network.types").MutationMethod) => string | undefined`

Mutation-key normalization and warning helpers used by the mutate orchestrator.

Responsibilities:
- Normalize string/object/reference mutation inputs into a dispatch key.
- Emit unknown-mutation warnings when warning mode is enabled.

The helpers in this module are intentionally side-effect-light, except for optional
warning emission, so orchestration code can remain deterministic and easy to inspect.

### resolveMutationKeyFromObject

`(methodObject: { [key: string]: unknown; name?: string | undefined; type?: string | undefined; identity?: string | undefined; max?: number | undefined; min?: number | undefined; mutateOutput?: boolean | undefined; }) => string | undefined`

Resolves mutation key from object-form descriptor.

Parameters:
- `methodObject` - - Mutation method object.

Returns: Dispatch key or undefined.

### warnUnknownMutation

`(mutationKey: string | undefined) => void`

Emits unknown-mutation warning when configured.

This helper intentionally no-ops when warnings are disabled so callers can invoke it
without repeating feature-flag checks.

Parameters:
- `mutationKey` - - Resolved mutation key.

Returns: Nothing.

## architecture/network/mutate/network.mutate.handlers.utils.ts

### addBackConn

`() => void`

Adds one backward (recurrent) connection between eligible node pairs.

This operation is skipped in acyclic mode.

Parameters:
- `this` - - Bound network.

Returns: Nothing.

### addConn

`() => void`

Adds one forward connection between currently unconnected eligible node pairs.

Candidate generation respects node ordering so the added edge is feed-forward.

Parameters:
- `this` - - Bound network.

Returns: Nothing.

### addGate

`() => void`

Assigns a random eligible node as gater for a random ungated connection.

Candidate pool includes normal and self-connections.

Parameters:
- `this` - - Bound network.

Returns: Nothing.

### addGRUNode

`() => void`

Replaces one connection by inserting a minimal GRU recurrent block.

Parameters:
- `this` - - Bound network.

Returns: Nothing.

### addLSTMNode

`() => void`

Replaces one connection by inserting a minimal LSTM recurrent block.

Parameters:
- `this` - - Bound network.

Returns: Nothing.

### addNode

`() => void`

Adds one hidden node by splitting an existing connection.

Execution modes:
- Deterministic chain mode grows a linear input→...→output chain.
- Standard mode splits a randomly selected forward connection.

Parameters:
- `this` - - Bound network.

Returns: Nothing.

### addNodeDeterministicChain

`(network: import("src/architecture/network").default, mutationProps: import("src/architecture/network/network.types").NetworkMutationProps) => void`

Applies deterministic chain-growth ADD_NODE mutation.

Parameters:
- `network` - - Target network.
- `mutationProps` - - Runtime mutation props.

Returns: Nothing.

### addNodeRandomSplit

`(network: import("src/architecture/network").default, mutationProps: import("src/architecture/network/network.types").NetworkMutationProps) => void`

Applies non-deterministic ADD_NODE by splitting a random connection.

Parameters:
- `network` - - Target network.
- `mutationProps` - - Runtime mutation props.

Returns: Nothing.

### addRecurrentNode

`(network: import("src/architecture/network").default, blockType: "lstm" | "gru") => void`

Shared orchestrator for recurrent-node mutation variants.

Parameters:
- `network` - - Target network.
- `blockType` - - Recurrent block type.

Returns: Nothing.

### addSelfConn

`() => void`

Adds one self-connection on an eligible node that does not already have one.

This operation is skipped in acyclic mode.

Parameters:
- `this` - - Bound network.

Returns: Nothing.

### appendBackwardCandidatesForLaterNode

`(traversalContext: import("src/architecture/network/network.types").BackwardCandidateTraversalContext, backwardConnectionCandidates: import("src/architecture/network/network.types").NodePair[]) => void`

Appends backward candidates for one later-node traversal context.

Parameters:
- `traversalContext` - - Later-node traversal context.
- `backwardConnectionCandidates` - - Collector array.

Returns: Nothing.

### appendForwardCandidatesForSource

`(traversalContext: import("src/architecture/network/network.types").ForwardCandidateTraversalContext, forwardConnectionCandidates: import("src/architecture/network/network.types").NodePair[]) => void`

Appends forward candidates for a single source-node traversal context.

Parameters:
- `traversalContext` - - Source traversal context.
- `forwardConnectionCandidates` - - Collector array.

Returns: Nothing.

### appendRecurrentLayerNodes

`(network: import("src/architecture/network").default, layerNodes: import("src/architecture/node").default[]) => void`

Appends recurrent layer nodes as hidden nodes.

Parameters:
- `network` - - Target network.
- `layerNodes` - - Layer nodes.

Returns: Nothing.

### applyFirstConnectionStabilityNudge

`(network: import("src/architecture/network").default) => void`

Applies tiny stability nudge to the first remaining connection.

Parameters:
- `network` - - Target network.

Returns: Nothing.

### asMutationProps

`(network: import("src/architecture/network").default) => import("src/architecture/network/network.types").NetworkMutationProps`

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

### batchNorm

`() => void`

Enables the internal batch-normalization flag on one random hidden node.

Parameters:
- `this` - - Bound network.

Returns: Nothing.

### collectAllConnections

`(network: import("src/architecture/network").default) => import("src/architecture/connection").default[]`

Collects normal and self connections.

Parameters:
- `network` - - Target network.

Returns: Combined connections.

### collectBackwardCandidatesForLaterNode

`(traversalContext: import("src/architecture/network/network.types").BackwardCandidateTraversalContext) => import("src/architecture/network/network.types").NodePair[]`

Collects all backward candidates for one later-node traversal context.

Parameters:
- `traversalContext` - - Later-node traversal context.

Returns: Candidate source/target pairs.

### collectBackwardCandidatesFromContext

`(backwardConnectionCandidates: import("src/architecture/network/network.types").NodePair[], traversalContext: import("src/architecture/network/network.types").BackwardCandidateTraversalContext) => import("src/architecture/network/network.types").NodePair[]`

Reduces one backward traversal context into candidate connection pairs.

Parameters:
- `backwardConnectionCandidates` - - Existing candidate pairs.
- `traversalContext` - - Later-node traversal context.

Returns: Updated candidate pairs.

### collectBackwardConnectionCandidates

`(network: import("src/architecture/network").default) => import("src/architecture/network/network.types").NodePair[]`

Collects backward (recurrent) connection candidates.

Parameters:
- `network` - - Target network.

Returns: Candidate source/target pairs.

### collectBackwardTraversalContexts

`(network: import("src/architecture/network").default) => import("src/architecture/network/network.types").BackwardCandidateTraversalContext[]`

Collects backward traversal contexts for all eligible later nodes.

Parameters:
- `network` - - Target network.

Returns: Backward traversal contexts.

### collectConnectionGroupsForReinit

`(targetNode: import("src/architecture/node").default) => import("src/architecture/connection").default[][]`

Collects all connection groups affected by REINIT_WEIGHT.

Parameters:
- `targetNode` - - Node receiving the reinitialization.

Returns: Mutable connection groups.

### collectDistinctNodeCandidates

`(nodeCandidates: import("src/architecture/node").default[], excludedNode: import("src/architecture/node").default) => import("src/architecture/node").default[]`

Collects candidates that are distinct from an excluded node.

Parameters:
- `nodeCandidates` - - Candidate nodes.
- `excludedNode` - - Node to exclude.

Returns: Distinct candidates.

### collectForwardCandidatesForSource

`(traversalContext: import("src/architecture/network/network.types").ForwardCandidateTraversalContext) => import("src/architecture/network/network.types").NodePair[]`

Collects all forward candidates for one source traversal context.

Parameters:
- `traversalContext` - - Source traversal context.

Returns: Candidate source/target pairs.

### collectForwardCandidatesFromContext

`(forwardConnectionCandidates: import("src/architecture/network/network.types").NodePair[], traversalContext: import("src/architecture/network/network.types").ForwardCandidateTraversalContext) => import("src/architecture/network/network.types").NodePair[]`

Reduces one forward traversal context into candidate connection pairs.

Parameters:
- `forwardConnectionCandidates` - - Existing candidate pairs.
- `traversalContext` - - Source traversal context.

Returns: Updated candidate pairs.

### collectForwardConnectionCandidates

`(network: import("src/architecture/network").default) => import("src/architecture/network/network.types").NodePair[]`

Collects forward connection candidates.

Parameters:
- `network` - - Target network.

Returns: Candidate source/target pairs.

### collectForwardTraversalContexts

`(network: import("src/architecture/network").default) => import("src/architecture/network/network.types").ForwardCandidateTraversalContext[]`

Collects forward traversal contexts for all eligible source nodes.

Parameters:
- `network` - - Target network.

Returns: Forward traversal contexts.

### collectMutableNonInputNodes

`(network: import("src/architecture/network").default, excludeOutputNodes: boolean) => import("src/architecture/node").default[]`

Collects mutable non-input nodes.

Parameters:
- `network` - - Target network.
- `excludeOutputNodes` - - True to exclude output nodes.

Returns: Mutable nodes.

### collectNodesByType

`(network: import("src/architecture/network").default, nodeType: string) => import("src/architecture/node").default[]`

Collects nodes by type.

Parameters:
- `network` - - Source network.
- `nodeType` - - Desired node type.

Returns: Matching nodes.

### collectNodesWithoutSelfLoop

`(network: import("src/architecture/network").default) => import("src/architecture/node").default[]`

Collects non-input nodes that do not have self loops.

Parameters:
- `network` - - Target network.

Returns: Eligible nodes.

### collectRemovableBackwardConnections

`(network: import("src/architecture/network").default) => import("src/architecture/connection").default[]`

Collects removable backward connections using redundancy constraints.

Parameters:
- `network` - - Target network.

Returns: Removable backward connections.

### collectRemovableForwardConnections

`(network: import("src/architecture/network").default) => import("src/architecture/connection").default[]`

Collects removable forward connections using redundancy constraints.

Parameters:
- `network` - - Target network.

Returns: Removable forward connections.

### collectSwappableNodesForMutation

`(network: import("src/architecture/network").default, method: import("src/architecture/network/network.types").MutationMethod | undefined) => import("src/architecture/node").default[]`

Collects swap-eligible nodes based on mutation configuration.

Parameters:
- `network` - - Target network.
- `method` - - Optional method descriptor.

Returns: Swap-eligible nodes.

### collectTargetLayerPeers

`(network: import("src/architecture/network").default, targetNode: import("src/architecture/node").default) => import("src/architecture/node").default[]`

Collects peers around a target node in the same type/layer neighborhood.

Parameters:
- `network` - - Target network.
- `targetNode` - - Node whose peers are collected.

Returns: Peer nodes.

### collectUngatedConnections

`(network: import("src/architecture/network").default) => import("src/architecture/connection").default[]`

Collects ungated connections including self-connections.

Parameters:
- `network` - - Target network.

Returns: Ungated connections.

### connectPair

`(network: import("src/architecture/network").default, selectedConnectionPair: import("src/architecture/network/network.types").NodePair) => void`

Connects source/target node pair.

Parameters:
- `network` - - Target network.
- `selectedConnectionPair` - - Source/target pair.

Returns: Nothing.

### containsNode

`(nodes: import("src/architecture/node").default[], node: import("src/architecture/node").default) => boolean`

Checks whether a node list contains a node reference.

Parameters:
- `nodes` - - Node list.
- `node` - - Node reference.

Returns: True when contained.

### countConnectionWhenSourceTargetsPeer

`(peerConnectionsFromSource: number, existingConnection: import("src/architecture/connection").default, countContext: import("src/architecture/network/network.types").SourcePeerConnectionCountContext) => number`

Counts one connection when it originates from source and targets a peer.

Parameters:
- `peerConnectionsFromSource` - - Current count.
- `existingConnection` - - Existing network connection.
- `countContext` - - Count context.

Returns: Updated count.

### countSourceConnectionsIntoPeerSet

`(network: import("src/architecture/network").default, candidateConnection: import("src/architecture/connection").default, targetLayerPeers: import("src/architecture/node").default[]) => number`

Counts source-originated connections that end inside the target peer set.

Parameters:
- `network` - - Target network.
- `candidateConnection` - - Candidate connection.
- `targetLayerPeers` - - Peer-set nodes.

Returns: Number of source-to-peer connections.

### createBackwardCandidateTraversalContext

`(network: import("src/architecture/network").default, laterNodeIndex: number) => import("src/architecture/network/network.types").BackwardCandidateTraversalContext`

Creates context for one backward-candidate traversal pass.

Parameters:
- `network` - - Target network.
- `laterNodeIndex` - - Current later-node index.

Returns: Immutable traversal context.

### createConnectionGroupReinitContext

`(randomValue: () => number, methodObject: { [key: string]: unknown; name?: string | undefined; type?: string | undefined; identity?: string | undefined; max?: number | undefined; min?: number | undefined; mutateOutput?: boolean | undefined; }) => import("src/architecture/network/network.types").ConnectionGroupReinitContext`

Creates immutable context for connection-group reinitialization.

Parameters:
- `randomValue` - - Random generator.
- `methodObject` - - Method override object.

Returns: Reinitialization context.

### createDirectionalConnectionContext

`(network: import("src/architecture/network").default, candidateConnection: import("src/architecture/connection").default) => import("src/architecture/network/network.types").DirectionalConnectionContext`

Creates indexed directional context for a connection candidate.

Parameters:
- `network` - - Target network.
- `candidateConnection` - - Candidate connection.

Returns: Directional context.

### createForwardCandidateTraversalContext

`(network: import("src/architecture/network").default, sourceNodeIndex: number) => import("src/architecture/network/network.types").ForwardCandidateTraversalContext`

Creates context for one forward-candidate source traversal pass.

Parameters:
- `network` - - Target network.
- `sourceNodeIndex` - - Current source index.

Returns: Immutable traversal context.

### createHiddenNode

`(randomValue: () => number) => import("src/architecture/node").default`

Creates a hidden node with random activation mutation.

Parameters:
- `randomValue` - - Random generator.

Returns: Hidden node.

### createRecurrentLayer

`(blockType: "lstm" | "gru") => import("src/architecture/network/network.types").RecurrentLayerShape`

Creates recurrent layer by type.

Parameters:
- `blockType` - - Recurrent block type.

Returns: Created recurrent layer.

### createSourcePeerConnectionCountContext

`(candidateConnection: import("src/architecture/connection").default, targetLayerPeers: import("src/architecture/node").default[]) => import("src/architecture/network/network.types").SourcePeerConnectionCountContext`

Creates immutable context for source-to-peer connection counting.

Parameters:
- `candidateConnection` - - Candidate connection.
- `targetLayerPeers` - - Peer-set nodes.

Returns: Count context.

### createTargetLayerPeerContext

`(network: import("src/architecture/network").default, targetNode: import("src/architecture/node").default) => import("src/architecture/network/network.types").TargetLayerPeerContext`

Creates immutable context for peer-layer collection.

Parameters:
- `network` - - Target network.
- `targetNode` - - Peer anchor node.

Returns: Peer traversal context.

### createWeightSamplingRangeContext

`(reinitContext: import("src/architecture/network/network.types").ConnectionGroupReinitContext) => import("src/architecture/network/network.types").WeightSamplingRangeContext`

Creates immutable sampling range context.

Parameters:
- `reinitContext` - - Reinitialization context.

Returns: Sampling range context.

### disconnectConnectionAndGetGater

`(network: import("src/architecture/network").default, connectionToExpand: import("src/architecture/connection").default) => import("src/architecture/node").default | null`

Disconnects a connection and returns its previous gater.

Parameters:
- `network` - - Target network.
- `connectionToExpand` - - Connection being expanded.

Returns: Previous gater reference.

### disconnectConnectionPair

`(network: import("src/architecture/network").default, selectedConnection: import("src/architecture/connection").default) => void`

Disconnects selected connection by endpoints.

Parameters:
- `network` - - Target network.
- `selectedConnection` - - Connection to disconnect.

Returns: Nothing.

### disconnectUnexpectedOutgoingConnections

`(network: import("src/architecture/network").default, chainNode: import("src/architecture/node").default, expectedTargetNode: import("src/architecture/node").default) => void`

Removes outgoing connections that do not match the expected chain target.

Parameters:
- `network` - - Target network.
- `chainNode` - - Node whose outgoing edges are validated.
- `expectedTargetNode` - - Allowed outgoing target.

Returns: Nothing.

### enableNodeBatchNorm

`(node: import("src/architecture/node").default) => void`

Enables internal batch-norm flag on a node.

Parameters:
- `node` - - Node to flag.

Returns: Nothing.

### ensureConnection

`(network: import("src/architecture/network").default, fromNode: import("src/architecture/node").default, toNode: import("src/architecture/node").default) => import("src/architecture/connection").default | undefined`

Ensures a connection exists and returns it.

Parameters:
- `network` - - Target network.
- `fromNode` - - Source node.
- `toNode` - - Target node.

Returns: Existing or created connection.

### ensureSeedForwardConnectionWhenEmpty

`(network: import("src/architecture/network").default) => boolean`

Ensures a seed input->output connection exists when connection list is empty.

Parameters:
- `network` - - Target network.

Returns: True when mutation may continue.

### expandConnectionWithRecurrentBlock

`(network: import("src/architecture/network").default, connectionToExpand: import("src/architecture/connection").default, blockType: "lstm" | "gru") => void`

Replaces one connection with a minimal recurrent block.

Parameters:
- `network` - - Target network.
- `connectionToExpand` - - Connection to replace.
- `blockType` - - Recurrent block type.

Returns: Nothing.

### findConnection

`(network: import("src/architecture/network").default, fromNode: import("src/architecture/node").default, toNode: import("src/architecture/node").default) => import("src/architecture/connection").default | undefined`

Gets a connection between two nodes when it exists.

Parameters:
- `network` - - Target network.
- `fromNode` - - Source node.
- `toNode` - - Target node.

Returns: Matching connection or undefined.

### findFirstNodeByType

`(network: import("src/architecture/network").default, nodeType: string) => import("src/architecture/node").default | undefined`

Returns the first node by type.

Parameters:
- `network` - - Target network.
- `nodeType` - - Node type to match.

Returns: Matching node or undefined.

### hasRedundantEndpoints

`(candidateConnection: import("src/architecture/connection").default) => boolean`

Checks whether both endpoints maintain at least one redundant edge.

Parameters:
- `candidateConnection` - - Candidate connection.

Returns: True when endpoint redundancy exists.

### initializeDeterministicChain

`(network: import("src/architecture/network").default, mutationProps: import("src/architecture/network/network.types").NetworkMutationProps, inputNode: import("src/architecture/node").default, outputNode: import("src/architecture/node").default) => void`

Initializes deterministic chain storage and seed edge.

Parameters:
- `network` - - Target network.
- `mutationProps` - - Runtime mutation props.
- `inputNode` - - Input node.
- `outputNode` - - Output node.

Returns: Nothing.

### insertNodeBeforeOutputTail

`(network: import("src/architecture/network").default, nodeToInsert: import("src/architecture/node").default, targetNode: import("src/architecture/node").default, mutationProps: import("src/architecture/network/network.types").NetworkMutationProps) => void`

Inserts a node before output tail while preserving output block ordering.

Parameters:
- `network` - - Target network.
- `nodeToInsert` - - Node to insert.
- `targetNode` - - Target node for insertion alignment.
- `mutationProps` - - Runtime mutation props.

Returns: Nothing.

### isBackwardCandidateTargetAvailable

`(laterNode: import("src/architecture/node").default, earlierNode: import("src/architecture/node").default) => boolean`

Checks whether a backward candidate target is not already projected.

Parameters:
- `laterNode` - - Candidate source node.
- `earlierNode` - - Candidate target node.

Returns: True when connection may be added.

### isBackwardDirectionalContext

`(directionContext: import("src/architecture/network/network.types").DirectionalConnectionContext) => boolean`

Checks whether a directional context represents a backward edge.

Parameters:
- `directionContext` - - Directional context.

Returns: True when backward.

### isForwardCandidateTargetAvailable

`(sourceNode: import("src/architecture/node").default, targetNode: import("src/architecture/node").default) => boolean`

Checks whether a forward candidate target is not already projected.

Parameters:
- `sourceNode` - - Candidate source node.
- `targetNode` - - Candidate target node.

Returns: True when connection may be added.

### isForwardConnectionStructurallyRemovable

`(network: import("src/architecture/network").default, candidateConnection: import("src/architecture/connection").default) => boolean`

Checks structural preconditions for removable forward connections.

Parameters:
- `network` - - Target network.
- `candidateConnection` - - Candidate connection.

Returns: True when the connection is a forward edge with redundant endpoints.

### isForwardDirectionalContext

`(directionContext: import("src/architecture/network/network.types").DirectionalConnectionContext) => boolean`

Checks whether a directional context represents a forward edge.

Parameters:
- `directionContext` - - Directional context.

Returns: True when forward.

### isNodeWithoutSelfLoop

`(candidateNode: import("src/architecture/node").default) => boolean`

Checks whether a node currently has no self-loop connections.

Parameters:
- `candidateNode` - - Node under evaluation.

Returns: True when the node has no self-loop.

### isRemovableBackwardConnection

`(network: import("src/architecture/network").default, candidateConnection: import("src/architecture/connection").default) => boolean`

Evaluates whether a backward connection is safe to remove.

Parameters:
- `network` - - Target network.
- `candidateConnection` - - Connection under evaluation.

Returns: True when removable.

### isRemovableForwardConnection

`(network: import("src/architecture/network").default, candidateConnection: import("src/architecture/connection").default) => boolean`

Evaluates whether a forward connection is safe to remove.

Parameters:
- `network` - - Target network.
- `candidateConnection` - - Connection under evaluation.

Returns: True when removable.

### isTargetLayerPeer

`(candidateNode: import("src/architecture/node").default, candidateNodeIndex: number, peerContext: import("src/architecture/network/network.types").TargetLayerPeerContext) => boolean`

Checks whether candidate node belongs to the target peer-layer set.

Parameters:
- `candidateNode` - - Candidate node.
- `candidateNodeIndex` - - Candidate node index.
- `peerContext` - - Peer traversal context.

Returns: True when candidate is an eligible peer.

### isTargetLayerPeerTypeMatch

`(candidateNode: import("src/architecture/node").default, peerContext: import("src/architecture/network/network.types").TargetLayerPeerContext) => boolean`

Checks whether node type matches the target-layer peer type.

Parameters:
- `candidateNode` - - Candidate node.
- `peerContext` - - Peer traversal context.

Returns: True when type matches.

### isTargetLayerPeerWithinDistance

`(candidateNodeIndex: number, peerContext: import("src/architecture/network/network.types").TargetLayerPeerContext) => boolean`

Checks whether candidate index lies within allowed peer distance.

Parameters:
- `candidateNodeIndex` - - Candidate node index.
- `peerContext` - - Peer traversal context.

Returns: True when within distance.

### isUngatedConnection

`(candidateConnection: import("src/architecture/connection").default) => boolean`

Checks whether a connection has no gater attached.

Parameters:
- `candidateConnection` - - Connection under evaluation.

Returns: True when ungated.

### markTopoDirtyIfAcyclic

`(mutationProps: import("src/architecture/network/network.types").NetworkMutationProps) => void`

Marks topology caches dirty when acyclic mode is enforced.

Parameters:
- `mutationProps` - - Runtime mutation props.

Returns: Nothing.

### modActivation

`(method: import("src/architecture/network/network.types").MutationMethod | undefined) => void`

Mutates activation function on one random non-input node.

Output-node eligibility is controlled by `method.mutateOutput` when provided.

Parameters:
- `this` - - Bound network.
- `method` - - Optional method descriptor.

Returns: Nothing.

### modBias

`(method: import("src/architecture/network/network.types").MutationMethod | undefined) => void`

Mutates bias parameters on one random non-input node.

Output nodes remain eligible for this operator.

Parameters:
- `this` - - Bound network.
- `method` - - Optional method descriptor.

Returns: Nothing.

### modWeight

`(method: import("src/architecture/network/network.types").MutationMethod | undefined) => void`

Perturbs one connection weight using a uniform delta sampled from configured bounds.

The candidate pool includes standard and self-connections.

Parameters:
- `this` - - Bound network.
- `method` - - Optional method descriptor.

Returns: Nothing.

### pickDistinctNodePair

`(swappableNodes: import("src/architecture/node").default[], randomValue: () => number) => import("src/architecture/network/network.types").DistinctNodePair | undefined`

Picks two distinct nodes from a candidate set.

Parameters:
- `swappableNodes` - - Swap candidate nodes.
- `randomValue` - - Random generator.

Returns: Distinct pair or undefined.

### pickDistinctRandomNode

`(nodeCandidates: import("src/architecture/node").default[], excludedNode: import("src/architecture/node").default, randomValue: () => number) => import("src/architecture/node").default | undefined`

Picks a random node distinct from a given reference.

Parameters:
- `nodeCandidates` - - Candidate nodes.
- `excludedNode` - - Node to exclude.
- `randomValue` - - Random generator.

Returns: Distinct node or undefined.

### pickRandomEntry

`(entries: T[], randomValue: () => number) => T | undefined`

Selects a random array element.

Parameters:
- `entries` - - Source entries.
- `randomValue` - - Random generator.

Returns: Random entry or undefined when empty.

### pickRandomNonInputNode

`(network: import("src/architecture/network").default, excludeOutputNodes: boolean, randomValue: () => number) => import("src/architecture/node").default | undefined`

Selects a random mutable non-input node.

Parameters:
- `network` - - Target network.
- `excludeOutputNodes` - - True to exclude outputs.
- `randomValue` - - Random generator.

Returns: Selected mutable node.

### pruneDeterministicChainExtraEdges

`(network: import("src/architecture/network").default, deterministicChain: import("src/architecture/node").default[], outputNode: import("src/architecture/node").default) => void`

Prunes side edges from chain nodes to preserve linear deterministic depth.

Parameters:
- `network` - - Target network.
- `deterministicChain` - - Chain node list.
- `outputNode` - - Output node.

Returns: Nothing.

### reconnectThroughRecurrentLayer

`(network: import("src/architecture/network").default, connectionToExpand: import("src/architecture/connection").default, recurrentLayer: import("src/architecture/network/network.types").RecurrentLayerShape) => import("src/architecture/connection").default | undefined`

Reconnects a source/target pair through a recurrent layer.

Parameters:
- `network` - - Target network.
- `connectionToExpand` - - Original connection.
- `recurrentLayer` - - Recurrent-layer shape.

Returns: Latest newly created connection or undefined.

### reinitializeConnectionGroupWeights

`(connections: import("src/architecture/connection").default[], reinitContext: import("src/architecture/network/network.types").ConnectionGroupReinitContext) => void`

Reinitializes all weights in a connection group.

Parameters:
- `connections` - - Connection group.
- `randomValue` - - Random generator.
- `minWeight` - - Minimum sampled weight.
- `maxWeight` - - Maximum sampled weight.

Returns: Nothing.

### reinitWeight

`(method: import("src/architecture/network/network.types").MutationMethod | undefined) => void`

Reinitializes incoming, outgoing, and self-connection weights for one target node.

Weight sampling bounds come from method overrides or default mutation bounds.

Parameters:
- `this` - - Bound network.
- `method` - - Optional method descriptor.

Returns: Nothing.

### removeHiddenNodeAndApplyStabilityNudge

`(network: import("src/architecture/network").default, hiddenNode: import("src/architecture/node").default) => void`

Removes selected hidden node and applies stability nudge.

Parameters:
- `network` - - Target network.
- `hiddenNode` - - Hidden node to remove.

Returns: Nothing.

### resolveDeterministicChainMutationContext

`(network: import("src/architecture/network").default, mutationProps: import("src/architecture/network/network.types").NetworkMutationProps) => import("src/architecture/network/network.types").DeterministicChainMutationContext | undefined`

Resolves all deterministic add-node prerequisites into one context object.

Parameters:
- `network` - - Target network.
- `mutationProps` - - Runtime mutation props.

Returns: Deterministic context or undefined when any prerequisite fails.

### resolveDistinctPairWithKnownFirstNode

`(swappableNodes: import("src/architecture/node").default[], firstNode: import("src/architecture/node").default, randomValue: () => number) => import("src/architecture/network/network.types").DistinctNodePair | undefined`

Resolves a distinct pair when first node is already known.

Parameters:
- `swappableNodes` - - Swap candidate nodes.
- `firstNode` - - Chosen first node.
- `randomValue` - - Random generator.

Returns: Distinct pair or undefined.

### resolveExpectedChainTarget

`(deterministicChain: import("src/architecture/node").default[], chainNodeIndex: number, outputNode: import("src/architecture/node").default) => import("src/architecture/node").default`

Resolves the expected outgoing target for a chain node position.

Parameters:
- `deterministicChain` - - Chain node list.
- `chainNodeIndex` - - Current chain index.
- `outputNode` - - Terminal output node.

Returns: Expected successor target.

### resolveInputOutputEndpoints

`(network: import("src/architecture/network").default) => import("src/architecture/network/network.types").InputOutputEndpoints | undefined`

Resolves input/output endpoints required for seed and deterministic flows.

Parameters:
- `network` - - Target network.

Returns: Endpoint nodes or undefined when missing.

### resolveMethodObject

`(method: import("src/architecture/network/network.types").MutationMethod | undefined) => { [key: string]: unknown; name?: string | undefined; type?: string | undefined; identity?: string | undefined; max?: number | undefined; min?: number | undefined; mutateOutput?: boolean | undefined; }`

Extracts method-object form when provided.

Parameters:
- `method` - - Optional mutation method.

Returns: Method object view.

### resolveSelectedBackwardConnectionPair

`(network: import("src/architecture/network").default, randomValue: () => number) => import("src/architecture/network/network.types").NodePair | undefined`

Resolves random selected backward connection pair.

Parameters:
- `network` - - Target network.
- `randomValue` - - Random generator.

Returns: Selected source/target pair.

### resolveSelectedForwardConnectionPair

`(network: import("src/architecture/network").default, randomValue: () => number) => import("src/architecture/network/network.types").NodePair | undefined`

Resolves random selected forward connection pair.

Parameters:
- `network` - - Target network.
- `randomValue` - - Random generator.

Returns: Selected source/target pair.

### resolveSelfConnectionTargetNode

`(network: import("src/architecture/network").default, randomValue: () => number) => import("src/architecture/node").default | undefined`

Resolves random node eligible for self-connection creation.

Parameters:
- `network` - - Target network.
- `randomValue` - - Random generator.

Returns: Selected node.

### sampleUniform

`(randomValue: () => number, minValue: number, maxValue: number) => number`

Samples a uniform value from [minValue, maxValue].

Parameters:
- `randomValue` - - Random generator.
- `minValue` - - Minimum value.
- `maxValue` - - Maximum value.

Returns: Sampled value.

### sampleUniformFromContext

`(samplingContext: import("src/architecture/network/network.types").WeightSamplingRangeContext) => number`

Samples one weight using a prebuilt range context.

Parameters:
- `samplingContext` - - Sampling range context.

Returns: Sampled weight.

### selectHiddenNodeForRemoval

`(network: import("src/architecture/network").default, hiddenNodes: import("src/architecture/node").default[]) => import("src/architecture/node").default | undefined`

Selects a hidden node candidate for SUB_NODE mutation.

Parameters:
- `network` - - Target network.
- `hiddenNodes` - - Hidden nodes.

Returns: Selected hidden node.

### splitConnectionThroughHiddenNode

`(network: import("src/architecture/network").default, mutationProps: import("src/architecture/network/network.types").NetworkMutationProps, connectionToSplit: import("src/architecture/connection").default) => import("src/architecture/network/network.types").ConnectionSplitResult`

Replaces one connection by inserting a hidden node and reconnecting edges.

Parameters:
- `network` - - Target network.
- `mutationProps` - - Runtime mutation props.
- `connectionToSplit` - - Connection to split.

Returns: Split result values.

### subBackConn

`() => void`

Removes one backward connection that satisfies redundancy constraints.

Parameters:
- `this` - - Bound network.

Returns: Nothing.

### subConn

`() => void`

Removes one forward connection when structural redundancy constraints are satisfied.

Constraints require endpoint redundancy and avoid disconnecting peer-layer groups.

Parameters:
- `this` - - Bound network.

Returns: Nothing.

### subGate

`() => void`

Removes gating from one randomly selected gated connection.

Parameters:
- `this` - - Bound network.

Returns: Nothing.

### subNode

`() => void`

Removes one hidden node and applies a tiny weight nudge for numerical continuity.

The stability nudge helps keep downstream mutation effects observable in edge cases
where node removal substantially changes effective signal flow.

Parameters:
- `this` - - Bound network.

Returns: Nothing.

### subSelfConn

`() => void`

Removes one existing self-connection chosen at random.

Parameters:
- `this` - - Bound network.

Returns: Nothing.

### swapNodeBiasAndSquash

`(firstNode: import("src/architecture/node").default, secondNode: import("src/architecture/node").default) => void`

Swaps bias and squash values between two nodes.

Parameters:
- `firstNode` - - First node.
- `secondNode` - - Second node.

Returns: Nothing.

### swapNodes

`(method: import("src/architecture/network/network.types").MutationMethod | undefined) => void`

Swaps bias and activation squash functions between two distinct mutable nodes.

This provides a lightweight structural-parameter recombination without changing
graph connectivity.

Parameters:
- `this` - - Bound network.
- `method` - - Optional method descriptor.

Returns: Nothing.

### tryDisconnectConnection

`(network: import("src/architecture/network").default, connection: import("src/architecture/connection").default) => void`

Disconnects a connection pair while suppressing errors.

Parameters:
- `network` - - Target network.
- `connection` - - Connection to remove.

Returns: Nothing.

### tryGateLatestConnection

`(network: import("src/architecture/network").default, previousGater: import("src/architecture/node").default | null, latestConnection: import("src/architecture/connection").default | undefined) => void`

Gates the latest connection when both previous gater and target exist.

Parameters:
- `network` - - Target network.
- `previousGater` - - Previously assigned gater.
- `latestConnection` - - Connection to receive the gater.

Returns: Nothing.

### tryReassignGateAfterSplit

`(network: import("src/architecture/network").default, randomValue: () => number, splitResult: import("src/architecture/network/network.types").ConnectionSplitResult) => void`

Reassigns prior gater to one of the new split connections when possible.

Parameters:
- `network` - - Target network.
- `randomValue` - - Random generator.
- `splitResult` - - Split result values.

Returns: Nothing.

### warnWhenEnabled

`(message: string) => void`

Emits a warning when warning mode is enabled.

Parameters:
- `message` - - Warning message.

Returns: Nothing.

### wouldDisconnectTargetPeerLayerGroup

`(network: import("src/architecture/network").default, candidateConnection: import("src/architecture/connection").default) => boolean`

Determines whether removal would disconnect a target peer-layer group.

Parameters:
- `network` - - Target network.
- `candidateConnection` - - Connection under evaluation.

Returns: True when peer group would be disconnected.
