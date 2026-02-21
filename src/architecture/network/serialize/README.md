# architecture/network/serialize

## architecture/network/serialize/network.serialize.utils.ts

### appendJsonForwardConnections

`(networkInternals: import("C:/NeatapticTS/src/architecture/network/network.types").SerializeNetworkInternals, networkJson: import("C:/NeatapticTS/src/architecture/network/network.types").NetworkJSON) => void`

Appends JSON entries for forward connection list.

Parameters:
- `networkInternals` - - Runtime internals.
- `networkJson` - - JSON accumulator.

Returns: Nothing.

### appendJsonNodesAndSelfConnections

`(networkInternals: import("C:/NeatapticTS/src/architecture/network/network.types").SerializeNetworkInternals, networkJson: import("C:/NeatapticTS/src/architecture/network/network.types").NetworkJSON) => void`

Appends JSON nodes and their optional self-connections.

Parameters:
- `networkInternals` - - Runtime internals.
- `networkJson` - - Target JSON accumulator.

Returns: Nothing.

### appendJsonSelfConnectionWhenPresent

`(nodeInternals: import("C:/NeatapticTS/src/architecture/network/network.types").SerializeNodeInternals, nodeIndex: number, networkJson: import("C:/NeatapticTS/src/architecture/network/network.types").NetworkJSON) => void`

Appends JSON self-connection when node has one.

Parameters:
- `nodeInternals` - - Node internals.
- `nodeIndex` - - Node index.
- `networkJson` - - JSON accumulator.

Returns: Nothing.

### asNetworkInternals

`(network: import("C:/NeatapticTS/src/architecture/network").default) => import("C:/NeatapticTS/src/architecture/network/network.types").SerializeNetworkInternals`

Casts network instance to internal runtime shape.

Parameters:
- `network` - - Network instance.

Returns: Runtime internals.

### asNetworkInternalsWithDropout

`(network: import("C:/NeatapticTS/src/architecture/network").default) => import("C:/NeatapticTS/src/architecture/network/network.types").NetworkInternalsWithDropout`

Casts network instance to internals with optional dropout.

Parameters:
- `network` - - Network instance.

Returns: Runtime internals with optional dropout.

### asNodeInternals

`(node: import("C:/NeatapticTS/src/architecture/node").default) => import("C:/NeatapticTS/src/architecture/network/network.types").SerializeNodeInternals`

Casts node to internal runtime shape.

Parameters:
- `node` - - Node instance.

Returns: Node internals.

### assignCompactGaterWhenValid

`(networkInternals: import("C:/NeatapticTS/src/architecture/network/network.types").SerializeNetworkInternals, gaterIndex: number | null, createdConnection: import("C:/NeatapticTS/src/architecture/connection").default | undefined) => void`

Assigns compact gater when both connection and gater index are valid.

Parameters:
- `networkInternals` - - Runtime internals.
- `gaterIndex` - - Optional gater index.
- `createdConnection` - - Created connection.

Returns: Nothing.

### assignJsonEnabledFlagWhenProvided

`(createdConnection: import("C:/NeatapticTS/src/architecture/connection").default | undefined, enabled: boolean) => void`

Assigns enabled flag when value is provided.

Parameters:
- `createdConnection` - - Created connection.
- `enabled` - - Optional enabled value.

Returns: Nothing.

### assignJsonGaterWhenValid

`(networkInternals: import("C:/NeatapticTS/src/architecture/network/network.types").SerializeNetworkInternals, gaterIndex: number | null, createdConnection: import("C:/NeatapticTS/src/architecture/connection").default | undefined) => void`

Assigns JSON gater when connection and gater index are valid.

Parameters:
- `networkInternals` - - Runtime internals.
- `gaterIndex` - - Optional gater index.
- `createdConnection` - - Created connection.

Returns: Nothing.

### collectAllConnections

`(networkInternals: import("C:/NeatapticTS/src/architecture/network/network.types").SerializeNetworkInternals) => import("C:/NeatapticTS/src/architecture/connection").default[]`

Collects all runtime connections into a single list.

Parameters:
- `networkInternals` - - Runtime internals.

Returns: Combined connections.

### collectNodeActivations

`(nodes: import("C:/NeatapticTS/src/architecture/node").default[]) => number[]`

Collects node activation values in positional order.

Parameters:
- `nodes` - - Node list.

Returns: Activation list.

### collectNodeSquashKeys

`(nodes: import("C:/NeatapticTS/src/architecture/node").default[]) => string[]`

Collects node squash keys in positional order.

Parameters:
- `nodes` - - Node list.

Returns: Squash-key list.

### collectNodeStates

`(nodes: import("C:/NeatapticTS/src/architecture/node").default[]) => number[]`

Collects node state values in positional order.

Parameters:
- `nodes` - - Node list.

Returns: State list.

### collectSerializedConnections

`(networkInternals: import("C:/NeatapticTS/src/architecture/network/network.types").SerializeNetworkInternals) => import("C:/NeatapticTS/src/architecture/network/network.types").SerializedConnection[]`

Collects serialized connections from forward and self groups.

Parameters:
- `networkInternals` - - Runtime internals.

Returns: Serialized connection list.

### createCompactPayloadContext

`(data: import("C:/NeatapticTS/src/architecture/network/network.types").CompactSerializedNetworkTuple) => import("C:/NeatapticTS/src/architecture/network/network.types").CompactPayloadContext`

Creates compact payload context from tuple input.

Parameters:
- `data` - - Compact tuple payload.

Returns: Normalized payload context.

### createConnection

`(networkInternals: import("C:/NeatapticTS/src/architecture/network/network.types").SerializeNetworkInternals, sourceNode: import("C:/NeatapticTS/src/architecture/node").default, targetNode: import("C:/NeatapticTS/src/architecture/node").default, weight: number) => import("C:/NeatapticTS/src/architecture/connection").default | undefined`

Creates one connection and returns first created instance.

Parameters:
- `networkInternals` - - Runtime internals.
- `sourceNode` - - Source node.
- `targetNode` - - Target node.
- `weight` - - Connection weight.

Returns: Created connection or undefined.

### createEmptyNetworkJson

`(networkInternals: import("C:/NeatapticTS/src/architecture/network/network.types").NetworkInternalsWithDropout) => import("C:/NeatapticTS/src/architecture/network/network.types").NetworkJSON`

Creates empty JSON shell from runtime internals.

Parameters:
- `networkInternals` - - Runtime internals with optional dropout.

Returns: Empty JSON shell.

### createJsonConnection

`(from: number, to: number, weight: number, gater: number | null, enabled: boolean) => import("C:/NeatapticTS/src/architecture/network/network.types").NetworkJSONConnection`

Creates one JSON connection entry.

Parameters:
- `from` - - Source index.
- `to` - - Target index.
- `weight` - - Connection weight.
- `gater` - - Optional gater index.
- `enabled` - - Enabled status.

Returns: JSON connection entry.

### createJsonNode

`(node: import("C:/NeatapticTS/src/architecture/node").default, nodeInternals: import("C:/NeatapticTS/src/architecture/network/network.types").SerializeNodeInternals, nodeIndex: number) => import("C:/NeatapticTS/src/architecture/network/network.types").NetworkJSONNode`

Creates one JSON node entry.

Parameters:
- `node` - - Runtime node.
- `nodeInternals` - - Node internals.
- `nodeIndex` - - Canonical index.

Returns: JSON node entry.

### createNetworkInstance

`(input: number, output: number) => import("C:/NeatapticTS/src/architecture/network").default`

Creates a network instance using runtime constructor loading.

Parameters:
- `input` - - Input size.
- `output` - - Output size.

Returns: New network instance.

### createNodeWithType

`(nodeType: string) => import("C:/NeatapticTS/src/architecture/node").default`

Creates one node with provided type.

Parameters:
- `nodeType` - - Node type.

Returns: New node.

### deserialize

`(data: import("C:/NeatapticTS/src/architecture/network/network.types").CompactSerializedNetworkTuple, inputSize: number | undefined, outputSize: number | undefined) => import("C:/NeatapticTS/src/architecture/network").default`

### findActivationByFunctionName

`(squashName: string | undefined) => import("C:/NeatapticTS/src/methods/activation.utils").ActivationFunction | undefined`

Resolves activation by matching function.name.

Parameters:
- `squashName` - - Activation function name.

Returns: Activation function or undefined.

### findActivationByKey

`(squashName: string | undefined) => import("C:/NeatapticTS/src/methods/activation.utils").ActivationFunction | undefined`

Resolves activation by direct key lookup.

Parameters:
- `squashName` - - Activation key.

Returns: Activation function or undefined.

### findActivationEntryByReference

`(squashFunction: import("C:/NeatapticTS/src/methods/activation.utils").ActivationFunction) => [string, import("C:/NeatapticTS/src/methods/activation.utils").ActivationFunction] | undefined`

Finds activation entry by function reference.

Parameters:
- `squashFunction` - - Activation function instance.

Returns: Activation entry or undefined.

### fromJSONImpl

`(json: import("C:/NeatapticTS/src/architecture/network/network.types").NetworkJSON) => import("C:/NeatapticTS/src/architecture/network").default`

### hydrateNodeFromJsonEntry

`(rebuiltNode: import("C:/NeatapticTS/src/architecture/node").default, nodeJsonEntry: import("C:/NeatapticTS/src/architecture/network/network.types").NetworkJSONNode, nodeIndex: number) => void`

Hydrates one node from JSON node entry.

Parameters:
- `rebuiltNode` - - Node to hydrate.
- `nodeJsonEntry` - - JSON node entry.
- `nodeIndex` - - Canonical node index.

Returns: Nothing.

### hydrateNodeStateFromCompactPayload

`(rebuiltNode: import("C:/NeatapticTS/src/architecture/node").default, activation: number, state: number, squashName: string | undefined, nodeIndex: number) => void`

Hydrates node runtime state from compact tuple values.

Parameters:
- `rebuiltNode` - - Node to hydrate.
- `activation` - - Activation value.
- `state` - - State value.
- `squashName` - - Activation key.
- `nodeIndex` - - Canonical node index.

Returns: Nothing.

### isConnectionEnabled

`(connectionInstance: import("C:/NeatapticTS/src/architecture/connection").default) => boolean`

Resolves enabled status from optional connection flag.

Parameters:
- `connectionInstance` - - Connection instance.

Returns: True when connection is enabled.

### isFiniteIndex

`(index: number) => boolean`

Checks whether value is a finite index.

Parameters:
- `index` - - Candidate index value.

Returns: True when finite number.

### isJsonConnectionInNodeBounds

`(nodes: import("C:/NeatapticTS/src/architecture/node").default[], connectionJsonEntry: import("C:/NeatapticTS/src/architecture/network/network.types").NetworkJSONConnection) => boolean`

Checks JSON connection indices against node list bounds.

Parameters:
- `nodes` - - Node list.
- `connectionJsonEntry` - - JSON connection entry.

Returns: True when both indices are valid.

### isJsonConnectionShapeValid

`(connectionJsonEntry: import("C:/NeatapticTS/src/architecture/network/network.types").NetworkJSONConnection) => boolean`

Checks that JSON connection has numeric endpoint fields.

Parameters:
- `connectionJsonEntry` - - JSON connection entry.

Returns: True when endpoint fields are numbers.

### isNodeIndexInBounds

`(nodes: import("C:/NeatapticTS/src/architecture/node").default[], index: number) => boolean`

Checks whether index is within array bounds.

Parameters:
- `nodes` - - Node list.
- `index` - - Candidate index.

Returns: True when index is valid.

### isSerializedConnectionInNodeBounds

`(networkInternals: import("C:/NeatapticTS/src/architecture/network/network.types").SerializeNetworkInternals, serializedConnection: import("C:/NeatapticTS/src/architecture/network/network.types").SerializedConnection) => boolean`

Checks compact connection bounds against current node list.

Parameters:
- `networkInternals` - - Runtime internals.
- `serializedConnection` - - Serialized connection record.

Returns: True when endpoints are valid.

### network.serialize.utils

### rebuildConnectionsFromCompactPayload

`(compactConnectionContext: import("C:/NeatapticTS/src/architecture/network/network.types").CompactConnectionRebuildContext) => void`

Rebuilds connections from compact payload records.

Parameters:
- `compactConnectionContext` - - Compact connection rebuild context.

Returns: Nothing.

### rebuildConnectionsFromJsonPayload

`(jsonConnectionContext: import("C:/NeatapticTS/src/architecture/network/network.types").JsonConnectionRebuildContext) => void`

Rebuilds connections from verbose JSON payload.

Parameters:
- `jsonConnectionContext` - - JSON connection rebuild context.

Returns: Nothing.

### rebuildNodesFromCompactPayload

`(networkInternals: import("C:/NeatapticTS/src/architecture/network/network.types").SerializeNetworkInternals, compactNodeContext: import("C:/NeatapticTS/src/architecture/network/network.types").CompactNodeRebuildContext) => void`

Rebuilds nodes from compact payload.

Parameters:
- `networkInternals` - - Runtime internals.
- `compactNodeContext` - - Compact node rebuild context.

Returns: Nothing.

### rebuildNodesFromJsonPayload

`(jsonNodeContext: import("C:/NeatapticTS/src/architecture/network/network.types").JsonNodeRebuildContext) => void`

Rebuilds nodes from verbose JSON payload.

Parameters:
- `jsonNodeContext` - - JSON node rebuild context.

Returns: Nothing.

### rebuildOneCompactConnection

`(networkInternals: import("C:/NeatapticTS/src/architecture/network/network.types").SerializeNetworkInternals, serializedConnection: import("C:/NeatapticTS/src/architecture/network/network.types").SerializedConnection) => void`

Rebuilds one compact serialized connection.

Parameters:
- `networkInternals` - - Runtime internals.
- `serializedConnection` - - Serialized connection record.

Returns: Nothing.

### rebuildOneJsonConnection

`(networkInternals: import("C:/NeatapticTS/src/architecture/network/network.types").SerializeNetworkInternals, connectionJsonEntry: import("C:/NeatapticTS/src/architecture/network/network.types").NetworkJSONConnection) => void`

Rebuilds one verbose JSON connection entry.

Parameters:
- `networkInternals` - - Runtime internals.
- `connectionJsonEntry` - - JSON connection entry.

Returns: Nothing.

### refreshNodeIndices

`(nodes: import("C:/NeatapticTS/src/architecture/node").default[]) => void`

Refreshes node.index for every node in list.

Parameters:
- `nodes` - - Node list.

Returns: Nothing.

### resetMutableRuntimeCollections

`(networkInternals: import("C:/NeatapticTS/src/architecture/network/network.types").SerializeNetworkInternals) => void`

Clears mutable runtime collections before reconstruction.

Parameters:
- `networkInternals` - - Runtime internals.

Returns: Nothing.

### resolveActivationFunction

`(squashName: string | undefined) => import("C:/NeatapticTS/src/methods/activation.utils").ActivationFunction`

Resolves activation function from stored key or function name.

Parameters:
- `squashName` - - Activation key or function name.

Returns: Activation function.

### resolveActivationKey

`(squashFunction: import("C:/NeatapticTS/src/methods/activation.utils").ActivationFunction) => string`

Resolves canonical activation key from function reference.

Parameters:
- `squashFunction` - - Activation function instance.

Returns: Activation key.

### resolveDropout

`(dropout: number | undefined) => number`

Resolves dropout value with numeric fallback.

Parameters:
- `dropout` - - Optional dropout value.

Returns: Effective dropout.

### resolveGaterIndex

`(gaterNode: import("C:/NeatapticTS/src/architecture/node").default | null) => number | null`

Resolves gater node index from gater reference.

Parameters:
- `gaterNode` - - Optional gater node.

Returns: Gater index or null.

### resolveNamedActivationFromFunction

`(squashFunction: import("C:/NeatapticTS/src/methods/activation.utils").ActivationFunction) => string | undefined`

Resolves activation name from function.name when non-empty.

Parameters:
- `squashFunction` - - Activation function instance.

Returns: Activation name or undefined.

### resolveNetworkSize

`(compactPayload: import("C:/NeatapticTS/src/architecture/network/network.types").CompactPayloadContext, inputSizeOverride: number | undefined, outputSizeOverride: number | undefined) => import("C:/NeatapticTS/src/architecture/network/network.types").ResolvedNetworkSizeContext`

Resolves effective input/output dimensions with optional overrides.

Parameters:
- `compactPayload` - - Compact payload context.
- `inputSizeOverride` - - Optional input override.
- `outputSizeOverride` - - Optional output override.

Returns: Resolved network size context.

### resolveNodeTypeFromCompactIndex

`(nodeIndex: number, totalNodeCount: number, input: number, output: number) => string`

Resolves node type from compact tuple position.

Parameters:
- `nodeIndex` - - Node index.
- `totalNodeCount` - - Total node count.
- `input` - - Input size.
- `output` - - Output size.

Returns: Node type string.

### resolveSizeOverride

`(overrideValue: number | undefined, serializedValue: number) => number`

Resolves one size value preferring explicit override.

Parameters:
- `overrideValue` - - Optional explicit override.
- `serializedValue` - - Serialized fallback value.

Returns: Effective size.

### serialize

`() => import("C:/NeatapticTS/src/architecture/network/network.types").CompactSerializedNetworkTuple`

Instance-level lightweight serializer used primarily for fast inter-thread transfer.

Parameters:
- `this` - - Bound network instance.

Returns: Compact serialized tuple payload.

### SerializedConnection

Serialized connection representation.

### serializeOneConnection

`(connectionInstance: import("C:/NeatapticTS/src/architecture/connection").default) => import("C:/NeatapticTS/src/architecture/network/network.types").SerializedConnection`

Serializes one connection into compact indexed form.

Parameters:
- `connectionInstance` - - Runtime connection.

Returns: Serialized connection record.

### toJSONImpl

`() => import("C:/NeatapticTS/src/architecture/network/network.types").NetworkJSON`

Verbose JSON export (stable formatVersion).

Parameters:
- `this` - - Bound network instance.

Returns: Verbose structural JSON payload.

### validateNetworkJsonOrThrow

`(json: import("C:/NeatapticTS/src/architecture/network/network.types").NetworkJSON) => void`

Validates verbose JSON payload and throws on invalid root shape.

Parameters:
- `json` - - Payload candidate.

Returns: Nothing.

### warnUnknownSquashName

`(squashName: string | undefined) => void`

Warns about unknown activation and fallback to identity.

Parameters:
- `squashName` - - Unknown activation name.

Returns: Nothing.

### warnWhenJsonFormatVersionIsUnknown

`(formatVersion: number) => void`

Warns when verbose format version differs from expected.

Parameters:
- `formatVersion` - - Incoming format version.

Returns: Nothing.

### default

#### _flags

Packed state flags (private for future-proofing hidden class):
bit0 => enabled gene expression (1 = active)
bit1 => DropConnect active mask (1 = not dropped this forward pass)
bit2 => hasGater (1 = symbol field present)
bit3 => plastic (plasticityRate > 0)
bits4+ reserved.

#### acquire

`(from: import("C:/NeatapticTS/src/architecture/node").default, to: import("C:/NeatapticTS/src/architecture/node").default, weight: number | undefined) => import("C:/NeatapticTS/src/architecture/connection").default`

Acquire a `Connection` from the pool (or construct new). Fields are fully reset & given
a fresh sequential `innovation` id. Prefer this in evolutionary algorithms that mutate
topology frequently to reduce GC pressure.

Parameters:
- `from` - Source node.
- `to` - Target node.
- `weight` - Optional initial weight.

Returns: Reinitialized connection instance.

#### dcMask

DropConnect active mask: 1 = not dropped (active), 0 = dropped for this stochastic pass.

#### dropConnectActiveMask

Convenience alias for DropConnect mask with clearer naming.

#### eligibility

Standard eligibility trace (e.g., for RTRL / policy gradient credit assignment).

#### enabled

Whether the gene (connection) is currently expressed (participates in forward pass).

#### firstMoment

First moment estimate (Adam / AdamW) (was opt_m).

#### from

The source (pre-synaptic) node supplying activation.

#### gain

Multiplicative modulation applied *after* weight. Default is `1` (neutral). We only store an
internal symbol-keyed property when the gain is non-neutral, reducing memory usage across
large populations where most connections are ungated.

#### gater

Optional gating node whose activation can modulate effective weight (symbol-backed).

#### gradientAccumulator

Generic gradient accumulator (RMSProp / AdaGrad) (was opt_cache).

#### hasGater

Whether a gater node is assigned (modulates gain); true if the gater symbol field is present.

#### infinityNorm

Adamax: Exponential moving infinity norm (was opt_u).

#### innovation

Unique historical marking (auto-increment) for evolutionary alignment.

#### innovationID

`(sourceNodeId: number, targetNodeId: number) => number`

Deterministic Cantor pairing function for a (sourceNodeId, targetNodeId) pair.
Useful when you want a stable innovation id without relying on global mutable counters
(e.g., for hashing or reproducible experiments).

NOTE: For large indices this can overflow 53-bit safe integer space; keep node indices reasonable.

Parameters:
- `sourceNodeId` - Source node integer id / index.
- `targetNodeId` - Target node integer id / index.

Returns: Unique non-negative integer derived from the ordered pair.

#### lookaheadShadowWeight

Lookahead: shadow (slow) weight parameter (was _la_shadowWeight).

#### maxSecondMoment

AMSGrad: Maximum of past second moment (was opt_vhat).

#### plastic

Whether this connection participates in plastic adaptation (rate > 0).

#### plasticityRate

Per-connection plasticity / learning rate (0 means non-plastic). Setting >0 marks plastic flag.

#### previousDeltaWeight

Last applied delta weight (used by classic momentum).

#### release

`(conn: import("C:/NeatapticTS/src/architecture/connection").default) => void`

Return a `Connection` to the internal pool for later reuse. Do NOT use the instance again
afterward unless re-acquired (treat as surrendered). Optimizer / trace fields are not
scrubbed here (they're overwritten during `acquire`).

Parameters:
- `conn` - The connection instance to recycle.

#### resetInnovationCounter

`(value: number) => void`

Reset the monotonic auto-increment innovation counter (used for newly constructed / pooled instances).
You normally only call this at the start of an experiment or when deserializing a full population.

Parameters:
- `value` - New starting value (default 1).

#### secondMoment

Second raw moment estimate (Adam family) (was opt_v).

#### secondMomentum

Secondary momentum (Lion variant) (was opt_m2).

#### to

The target (post-synaptic) node receiving activation.

#### toJSON

`() => { from: number | undefined; to: number | undefined; weight: number; gain: number; innovation: number; enabled: boolean; gater?: number | undefined; }`

Serialize to a minimal JSON-friendly shape (used for saving genomes / networks).
Undefined indices are preserved as `undefined` to allow later resolution / remapping.

Returns: Object with node indices, weight, gain, gater index (if any), innovation id & enabled flag.

#### totalDeltaWeight

Accumulated (batched) delta weight awaiting an apply step.

#### weight

Scalar multiplier applied to the source activation (prior to gain modulation).

#### xtrace

Extended trace structure for modulatory / eligibility propagation algorithms. Parallel arrays for cache-friendly iteration.
