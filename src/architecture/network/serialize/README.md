# architecture/network/serialize

## architecture/network/serialize/network.serialize.utils.types.ts

### CompactConnectionRebuildContext

Context for compact-connection reconstruction.

Connection rows are processed independently so malformed entries can be skipped without aborting import.

### CompactNodeRebuildContext

Context for compact-node reconstruction.

Arrays are expected to be index-aligned so each node can be hydrated deterministically.

### CompactPayloadContext

Context carrying compact payload fields.

This named-object form replaces tuple index access in internal orchestration code.

### CompactSerializedNetworkTuple

Compact tuple payload used by `serialize` output.

Tuple slots are intentionally positional to reduce payload size:
0) activations, 1) states, 2) squash keys, 3) connections, 4) input size, 5) output size.

Example:

```ts
const compactTuple: CompactSerializedNetworkTuple = [
  [0.1, 0.2],
  [0, 0],
  ['identity', 'tanh'],
  [{ from: 0, to: 1, weight: 0.5, gater: null }],
  1,
  1,
];
```

### ConnectionInternalsWithEnabled

Connection view with optional enabled flag.

Some serialized formats preserve per-edge enablement, while others treat missing values
as implicitly enabled.

### DEFAULT_NUMERIC_VALUE

Default numeric fallback for absent scalar values.

This keeps import logic deterministic when optional numeric fields are missing.

### ERROR_INVALID_NETWORK_JSON

Error text emitted for invalid verbose JSON payload roots.

### FALLBACK_ACTIVATION_KEY

Fallback activation key used when no mapping is found.

Identity is selected as the safest non-disruptive activation fallback.

### FIRST_CONNECTION_INDEX

Index of the first created connection returned by `connect()`.

The runtime API returns an array, and serializer logic consistently reads index `0`.

### JsonConnectionRebuildContext

Context for JSON-connection reconstruction.

Connection rows may include optional gater and enabled metadata.

### JsonNodeRebuildContext

Context for JSON-node reconstruction.

Node entries are rebuilt in order and pushed into mutable runtime internals.

### NETWORK_JSON_FORMAT_VERSION

Default format version for verbose serialization payloads.

Consumers can use this value to identify the JSON schema revision.

### NetworkInternalsWithDropout

Serialize internals with optional dropout field.

Verbose JSON snapshots normalize this value so readers can treat dropout as numeric data.

### NetworkJSON

Verbose JSON payload representation used by `toJSONImpl` and `fromJSONImpl`.

`formatVersion` enables compatibility checks and migration handling.

Example:

```ts
const payload: NetworkJSON = {
  formatVersion: 2,
  input: 2,
  output: 1,
  dropout: 0,
  nodes: [{ type: 'input', bias: 0, squash: 'identity', index: 0 }],
  connections: [],
};
```

### NetworkJSONConnection

Verbose JSON connection representation.

Includes optional gater and explicit enabled state for portability.

### NetworkJSONNode

Verbose JSON node representation.

Node entries are self-describing and intended for readable, versioned snapshots.

### NODE_TYPE_HIDDEN

Node type literal used for hidden layer nodes.

### NODE_TYPE_INPUT

Node type literal used for input layer nodes.

### NODE_TYPE_OUTPUT

Node type literal used for output layer nodes.

### ResolvedNetworkSizeContext

Resolved input/output sizes for rebuild.

Values reflect override-first resolution semantics used during deserialization.

### SerializedConnection

Serialized connection representation used by compact and JSON formats.

Endpoints are canonical node indices, which keeps payloads deterministic and language-agnostic.

### SerializeNetworkInternals

Runtime interface for accessing network internals during serialization.

This is an internal bridge type used by serializer helpers to read and rebuild
topology without exposing private implementation details in public APIs.

### SerializeNodeInternals

Runtime node internals needed for serialization workflows.

These fields are the minimal node state required to round-trip compact and JSON payloads.

### WARNING_INVALID_CONNECTION_DURING_DESERIALIZE

Warning emitted when compact deserialize sees invalid edge endpoints.

### WARNING_INVALID_CONNECTION_DURING_FROM_JSON

Warning emitted when JSON deserialize sees invalid edge endpoints.

### WARNING_INVALID_GATER_DURING_DESERIALIZE

Warning emitted when compact deserialize sees an invalid gater index.

### WARNING_INVALID_GATER_DURING_FROM_JSON

Warning emitted when JSON deserialize sees an invalid gater index.

### WARNING_UNKNOWN_FORMAT_VERSION

Warning emitted for unknown verbose format versions.

### WARNING_UNKNOWN_SQUASH_PREFIX

Prefix used when warning about unknown activation keys.

### WARNING_UNKNOWN_SQUASH_SUFFIX

Suffix used when warning about unknown activation keys.

## architecture/network/serialize/network.serialize.utils.ts

### network.serialize.utils

### applyHydratedArchitectureDescriptor

`(network: import("src/architecture/network").default, architectureDescriptor: import("src/architecture/network/network.types").NetworkArchitectureDescriptor | undefined) => void`

Applies hydrated architecture metadata to runtime network when shape is valid.

Parameters:
- `network` - - Rebuilt network instance.
- `architectureDescriptor` - - Optional serialized descriptor.

Returns: Nothing.

### deserialize

`(data: import("src/architecture/network/network.types").CompactSerializedNetworkTuple, inputSize: number | undefined, outputSize: number | undefined) => import("src/architecture/network").default`

Rebuilds a network instance from compact tuple form.

Use this importer for compact payloads produced by `serialize`.
Optional `inputSize` and `outputSize` let callers enforce shape overrides at import time.

Parameters:
- `data` - - Compact tuple payload.
- `inputSize` - - Optional input-size override that takes precedence over serialized input.
- `outputSize` - - Optional output-size override that takes precedence over serialized output.

Returns: Reconstructed network instance.

Example:

```ts
import { deserialize } from './network.serialize.utils';

const rebuiltNetwork = deserialize(compactTuple, 2, 1);
```

### fromJSONImpl

`(json: import("src/architecture/network/network.types").NetworkJSON) => import("src/architecture/network").default`

Reconstructs a network instance from the verbose JSON payload.

This importer validates payload shape, restores dropout and topology, and then rebuilds
connections, gating relationships, and optional enabled flags.

Parameters:
- `json` - - Verbose JSON payload.

Returns: Reconstructed network instance.

Example:

```ts
import { fromJSONImpl } from './network.serialize.utils';

const rebuiltNetwork = fromJSONImpl(snapshotJson);
```

### isArchitectureDescriptorShapeValid

`(architectureDescriptor: import("src/architecture/network/network.types").NetworkArchitectureDescriptor | undefined) => boolean`

Parameters:
- `architectureDescriptor` - - Optional descriptor candidate.

Returns: True when minimal descriptor shape is valid.

### serialize

`() => import("src/architecture/network/network.types").CompactSerializedNetworkTuple`

Serializes a network instance into the compact tuple format.

Use this format when payload size and serialization speed matter more than readability.
The tuple layout is positional and optimized for transport/storage efficiency.

Parameters:
- `this` - - Bound network instance.

Returns: Compact tuple payload containing activations, states, squash keys, connections, and input/output sizes.

Example:

```ts
import Network from '../../network';
import { deserialize, serialize } from './network.serialize.utils';

const sourceNetwork = new Network(2, 1);
const compactTuple = serialize.call(sourceNetwork);
const rebuiltNetwork = deserialize(compactTuple);
```

### SerializedConnection

Serialized connection representation used by compact and JSON formats.

Endpoints are canonical node indices, which keeps payloads deterministic and language-agnostic.

### toJSONImpl

`() => import("src/architecture/network/network.types").NetworkJSON`

Serializes a network instance into the verbose JSON format.

Use this format when you need human-readable snapshots, explicit schema versioning,
and better forward/backward compatibility handling.

Parameters:
- `this` - - Bound network instance.

Returns: Versioned JSON payload with shape metadata, nodes, and connections.

Example:

```ts
import Network from '../../network';
import { fromJSONImpl, toJSONImpl } from './network.serialize.utils';

const sourceNetwork = new Network(3, 1);
const snapshotJson = toJSONImpl.call(sourceNetwork);
const rebuiltNetwork = fromJSONImpl(snapshotJson);
```

### default

#### _flags

Packed state flags (private for future-proofing hidden class):
bit0 => enabled gene expression (1 = active)
bit1 => DropConnect active mask (1 = not dropped this forward pass)
bit2 => hasGater (1 = symbol field present)
bit3 => plastic (plasticityRate > 0)
bits4+ reserved.

#### acquire

`(from: import("src/architecture/node").default, to: import("src/architecture/node").default, weight: number | undefined) => import("src/architecture/connection").default`

Acquire a `Connection` from the pool (or construct new). Fields are fully reset & given
a fresh sequential `innovation` id. Prefer this in evolutionary algorithms that mutate
topology frequently to reduce GC pressure.

Parameters:
- `from` - Source node.
- `to` - Target node.
- `weight` - Optional initial weight.

Returns: Reinitialized connection instance.

Example:

const conn = Connection.acquire(a, b);
// ... use conn ...
Connection.release(conn); // when permanently removed

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

Example:

const id = Connection.innovationID(2, 5); // deterministic

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

`(conn: import("src/architecture/connection").default) => void`

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

Example:

Connection.resetInnovationCounter();     // back to 1
Connection.resetInnovationCounter(1000); // start counting from 1000

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

Example:

const json = connection.toJSON();
// => { from: 0, to: 3, weight: 0.12, gain: 1, innovation: 57, enabled: true }

#### totalDeltaWeight

Accumulated (batched) delta weight awaiting an apply step.

#### weight

Scalar multiplier applied to the source activation (prior to gain modulation).

#### xtrace

Extended trace structure for modulatory / eligibility propagation algorithms. Parallel arrays for cache-friendly iteration.

## architecture/network/serialize/network.serialize.json.utils.ts

### appendJsonForwardConnections

`(networkInternals: import("src/architecture/network/network.types").SerializeNetworkInternals, networkJson: import("src/architecture/network/network.types").NetworkJSON) => void`

Appends JSON entries for forward connections.

Non-finite endpoint indices are ignored to prevent malformed output records.

Parameters:
- `networkInternals` - - Runtime internals.
- `networkJson` - - JSON accumulator.

Returns: Nothing.

### appendJsonNodesAndSelfConnections

`(networkInternals: import("src/architecture/network/network.types").SerializeNetworkInternals, networkJson: import("src/architecture/network/network.types").NetworkJSON) => void`

Appends JSON node entries and optional self-connections.

Node indices are refreshed during this step so connection records can reference
stable numeric endpoints.

Parameters:
- `networkInternals` - - Runtime internals.
- `networkJson` - - Target JSON accumulator.

Returns: Nothing.

### appendJsonSelfConnectionWhenPresent

`(nodeInternals: import("src/architecture/network/network.types").SerializeNodeInternals, nodeIndex: number, networkJson: import("src/architecture/network/network.types").NetworkJSON) => void`

Appends JSON self-connection when node has one.

Parameters:
- `nodeInternals` - - Node internals.
- `nodeIndex` - - Node index.
- `networkJson` - - JSON accumulator.

Returns: Nothing.

### assignJsonEnabledFlagWhenProvided

`(createdConnection: import("src/architecture/connection").default | undefined, enabled: boolean) => void`

Assigns enabled flag when value is provided.

Parameters:
- `createdConnection` - - Created connection.
- `enabled` - - Optional enabled value.

Returns: Nothing.

### assignJsonGaterWhenValid

`(networkInternals: import("src/architecture/network/network.types").SerializeNetworkInternals, gaterIndex: number | null, createdConnection: import("src/architecture/connection").default | undefined) => void`

Assigns JSON gater when connection and gater index are valid.

Parameters:
- `networkInternals` - - Runtime internals.
- `gaterIndex` - - Optional gater index.
- `createdConnection` - - Created connection.

Returns: Nothing.

### createConnection

`(networkInternals: import("src/architecture/network/network.types").SerializeNetworkInternals, sourceNode: import("src/architecture/node").default, targetNode: import("src/architecture/node").default, weight: number) => import("src/architecture/connection").default | undefined`

Creates one connection and returns first created instance.

Parameters:
- `networkInternals` - - Runtime internals.
- `sourceNode` - - Source node.
- `targetNode` - - Target node.
- `weight` - - Connection weight.

Returns: Created connection or undefined.

### createEmptyNetworkJson

`(networkInternals: import("src/architecture/network/network.types").NetworkInternalsWithDropout) => import("src/architecture/network/network.types").NetworkJSON`

Creates an empty verbose JSON shell from runtime internals.

The shell includes format and shape metadata and is filled in by subsequent
node and connection append steps.

Parameters:
- `networkInternals` - - Runtime internals with optional dropout.

Returns: Empty JSON shell with `formatVersion` and scalar metadata initialized.

Example:

```ts
const networkJson = createEmptyNetworkJson(networkInternals);
```

### createJsonConnection

`(from: number, to: number, weight: number, gater: number | null, enabled: boolean) => import("src/architecture/network/network.types").NetworkJSONConnection`

Creates one JSON connection entry.

Parameters:
- `from` - - Source index.
- `to` - - Target index.
- `weight` - - Connection weight.
- `gater` - - Optional gater index.
- `enabled` - - Enabled status.

Returns: JSON connection entry.

### createJsonNode

`(node: import("src/architecture/node").default, nodeInternals: import("src/architecture/network/network.types").SerializeNodeInternals, nodeIndex: number) => import("src/architecture/network/network.types").NetworkJSONNode`

Creates one JSON node entry.

Parameters:
- `node` - - Runtime node.
- `nodeInternals` - - Node internals.
- `nodeIndex` - - Canonical index.

Returns: JSON node entry.

### createNodeWithType

`(nodeType: string) => import("src/architecture/node").default`

Creates one node with provided type.

Parameters:
- `nodeType` - - Node type.

Returns: New node.

### hydrateNodeFromJsonEntry

`(rebuiltNode: import("src/architecture/node").default, nodeJsonEntry: import("src/architecture/network/network.types").NetworkJSONNode, nodeIndex: number) => void`

Hydrates one node from JSON node entry.

Parameters:
- `rebuiltNode` - - Node to hydrate.
- `nodeJsonEntry` - - JSON node entry.
- `nodeIndex` - - Canonical node index.

Returns: Nothing.

### isConnectionEnabled

`(connectionInstance: import("src/architecture/connection").default) => boolean`

Resolves enabled status from optional connection flag.

Parameters:
- `connectionInstance` - - Connection instance.

Returns: True when connection is enabled.

### isJsonConnectionInNodeBounds

`(nodes: import("src/architecture/node").default[], connectionJsonEntry: import("src/architecture/network/network.types").NetworkJSONConnection) => boolean`

Checks JSON connection indices against node list bounds.

Parameters:
- `nodes` - - Node list.
- `connectionJsonEntry` - - JSON connection entry.

Returns: True when both indices are valid.

### isJsonConnectionShapeValid

`(connectionJsonEntry: import("src/architecture/network/network.types").NetworkJSONConnection) => boolean`

Checks that JSON connection has numeric endpoint fields.

Parameters:
- `connectionJsonEntry` - - JSON connection entry.

Returns: True when endpoint fields are numbers.

### rebuildConnectionsFromJsonPayload

`(jsonConnectionContext: import("src/architecture/network/network.types").JsonConnectionRebuildContext) => void`

Rebuilds runtime connections from verbose JSON entries.

Invalid entries are skipped with warnings so import continues for valid records.

Parameters:
- `jsonConnectionContext` - - JSON connection rebuild context.

Returns: Nothing.

Example:

```ts
rebuildConnectionsFromJsonPayload({
  networkInternals,
  connectionJsonEntries,
});
```

### rebuildNodesFromJsonPayload

`(jsonNodeContext: import("src/architecture/network/network.types").JsonNodeRebuildContext) => void`

Rebuilds runtime nodes from verbose JSON entries.

Parameters:
- `jsonNodeContext` - - JSON node rebuild context.

Returns: Nothing.

Example:

```ts
rebuildNodesFromJsonPayload({ networkInternals, nodeJsonEntries });
```

### rebuildOneJsonConnection

`(networkInternals: import("src/architecture/network/network.types").SerializeNetworkInternals, connectionJsonEntry: import("src/architecture/network/network.types").NetworkJSONConnection) => void`

Rebuilds one verbose JSON connection entry.

Parameters:
- `networkInternals` - - Runtime internals.
- `connectionJsonEntry` - - JSON connection entry.

Returns: Nothing.

### resolveDropout

`(dropout: number | undefined) => number`

Resolves dropout with a numeric fallback when the value is absent.

Parameters:
- `dropout` - - Optional dropout value.

Returns: Effective dropout.

### resolveGaterIndex

`(gaterNode: import("src/architecture/node").default | null) => number | null`

Resolves gater node index from gater reference.

Parameters:
- `gaterNode` - - Optional gater node.

Returns: Gater index or null.

### validateNetworkJsonOrThrow

`(json: import("src/architecture/network/network.types").NetworkJSON) => void`

Validates the verbose JSON payload root shape.

Parameters:
- `json` - - Payload candidate.

Returns: Nothing.

### warnWhenJsonFormatVersionIsUnknown

`(formatVersion: number) => void`

Warns when incoming verbose format version differs from the expected one.

Parameters:
- `formatVersion` - - Incoming format version.

Returns: Nothing.

## architecture/network/serialize/network.serialize.compact.utils.ts

### assignCompactGaterWhenValid

`(networkInternals: import("src/architecture/network/network.types").SerializeNetworkInternals, gaterIndex: number | null, createdConnection: import("src/architecture/connection").default | undefined) => void`

Assigns compact gater when both connection and gater index are valid.

Parameters:
- `networkInternals` - - Runtime internals.
- `gaterIndex` - - Optional gater index.
- `createdConnection` - - Created connection.

Returns: Nothing.

### collectAllConnections

`(networkInternals: import("src/architecture/network/network.types").SerializeNetworkInternals) => import("src/architecture/connection").default[]`

Collects all runtime connections into a single list.

Parameters:
- `networkInternals` - - Runtime internals.

Returns: Combined connections.

### collectNodeActivations

`(nodes: import("src/architecture/node").default[]) => number[]`

Collects node activation values in positional order.

Parameters:
- `nodes` - - Node list.

Returns: Activation list aligned to node indices.

### collectNodeSquashKeys

`(nodes: import("src/architecture/node").default[]) => string[]`

Collects node activation keys in positional order.

Each function reference is normalized to a string key for compact transfer.

Parameters:
- `nodes` - - Node list.

Returns: Squash-key list aligned to node indices.

### collectNodeStates

`(nodes: import("src/architecture/node").default[]) => number[]`

Collects node state values in positional order.

Parameters:
- `nodes` - - Node list.

Returns: State list aligned to node indices.

### collectSerializedConnections

`(networkInternals: import("src/architecture/network/network.types").SerializeNetworkInternals) => import("src/architecture/network/network.types").SerializedConnection[]`

Collects compact connection records from forward and self connection groups.

Parameters:
- `networkInternals` - - Runtime internals.

Returns: Serialized connection list.

### createConnection

`(networkInternals: import("src/architecture/network/network.types").SerializeNetworkInternals, sourceNode: import("src/architecture/node").default, targetNode: import("src/architecture/node").default, weight: number) => import("src/architecture/connection").default | undefined`

Creates one connection and returns first created instance.

Parameters:
- `networkInternals` - - Runtime internals.
- `sourceNode` - - Source node.
- `targetNode` - - Target node.
- `weight` - - Connection weight.

Returns: Created connection or undefined.

### createNodeWithType

`(nodeType: string) => import("src/architecture/node").default`

Creates one node with provided type.

Parameters:
- `nodeType` - - Node type.

Returns: New node.

### hydrateNodeStateFromCompactPayload

`(rebuiltNode: import("src/architecture/node").default, activation: number, state: number, squashName: string | undefined, nodeIndex: number) => void`

Hydrates node runtime state from compact tuple values.

Parameters:
- `rebuiltNode` - - Node to hydrate.
- `activation` - - Activation value.
- `state` - - State value.
- `squashName` - - Activation key.
- `nodeIndex` - - Canonical node index.

Returns: Nothing.

### isSerializedConnectionInNodeBounds

`(networkInternals: import("src/architecture/network/network.types").SerializeNetworkInternals, serializedConnection: import("src/architecture/network/network.types").SerializedConnection) => boolean`

Checks compact connection bounds against current node list.

Parameters:
- `networkInternals` - - Runtime internals.
- `serializedConnection` - - Serialized connection record.

Returns: True when endpoints are valid.

### rebuildConnectionsFromCompactPayload

`(compactConnectionContext: import("src/architecture/network/network.types").CompactConnectionRebuildContext) => void`

Rebuilds runtime connections from compact connection records.

Invalid endpoint or gater indices are skipped with warnings to preserve import flow.

Parameters:
- `compactConnectionContext` - - Compact connection rebuild context.

Returns: Nothing.

Example:

```ts
import { rebuildConnectionsFromCompactPayload } from './network.serialize.compact.utils';

rebuildConnectionsFromCompactPayload({
  networkInternals,
  serializedConnections,
});
```

### rebuildNodesFromCompactPayload

`(networkInternals: import("src/architecture/network/network.types").SerializeNetworkInternals, compactNodeContext: import("src/architecture/network/network.types").CompactNodeRebuildContext) => void`

Rebuilds runtime nodes from compact payload arrays.

Node type is inferred from index position relative to input/output boundaries.

Parameters:
- `networkInternals` - - Runtime internals.
- `compactNodeContext` - - Compact node rebuild context.

Returns: Nothing.

Example:

```ts
import { rebuildNodesFromCompactPayload } from './network.serialize.compact.utils';

rebuildNodesFromCompactPayload(networkInternals, compactNodeContext);
```

### rebuildOneCompactConnection

`(networkInternals: import("src/architecture/network/network.types").SerializeNetworkInternals, serializedConnection: import("src/architecture/network/network.types").SerializedConnection) => void`

Rebuilds one compact serialized connection.

Parameters:
- `networkInternals` - - Runtime internals.
- `serializedConnection` - - Serialized connection record.

Returns: Nothing.

### refreshNodeIndices

`(nodes: import("src/architecture/node").default[]) => void`

Refreshes `node.index` for each node in list order.

Canonical indices are required so compact connection records can store endpoints
and gaters as stable numeric positions.

Parameters:
- `nodes` - - Node list.

Returns: Nothing.

### resolveNodeTypeFromCompactIndex

`(nodeIndex: number, totalNodeCount: number, input: number, output: number) => string`

Resolves node type from compact tuple position.

Parameters:
- `nodeIndex` - - Node index.
- `totalNodeCount` - - Total node count.
- `input` - - Input size.
- `output` - - Output size.

Returns: Node type string.

### serializeOneConnection

`(connectionInstance: import("src/architecture/connection").default) => import("src/architecture/network/network.types").SerializedConnection`

Serializes one connection into compact indexed form.

Parameters:
- `connectionInstance` - - Runtime connection.

Returns: Serialized connection record.

## architecture/network/serialize/network.serialize.runtime.utils.ts

### asNetworkInternals

`(network: import("src/architecture/network").default) => import("src/architecture/network/network.types").SerializeNetworkInternals`

Casts a network instance to the internal runtime shape used by serializer helpers.

Parameters:
- `network` - - Network instance.

Returns: Runtime internals.

### asNetworkInternalsWithDropout

`(network: import("src/architecture/network").default) => import("src/architecture/network/network.types").NetworkInternalsWithDropout`

Casts a network instance to internals that include optional dropout metadata.

Parameters:
- `network` - - Network instance.

Returns: Runtime internals with optional dropout.

### asNodeInternals

`(node: import("src/architecture/node").default) => import("src/architecture/network/network.types").SerializeNodeInternals`

Casts a node instance to its internal runtime representation.

Parameters:
- `node` - - Node instance.

Returns: Node internals.

### createCompactPayloadContext

`(data: import("src/architecture/network/network.types").CompactSerializedNetworkTuple) => import("src/architecture/network/network.types").CompactPayloadContext`

Normalizes a compact tuple payload into a named object context.

This improves readability in orchestration code by replacing positional tuple access
with semantically named fields.

Parameters:
- `data` - - Compact tuple payload.

Returns: Normalized payload context.

Example:

```ts
const compactPayload = createCompactPayloadContext(compactTuple);
```

### createNetworkInstance

`(input: number, output: number) => import("src/architecture/network").default`

Creates a new network instance for deserialize workflows.

Parameters:
- `input` - - Input size.
- `output` - - Output size.

Returns: New network instance.

### isFiniteIndex

`(index: number) => boolean`

Checks whether a candidate index value is a finite number.

Parameters:
- `index` - - Candidate index value.

Returns: True when finite number.

### isNodeIndexInBounds

`(nodes: import("src/architecture/node").default[], index: number) => boolean`

Checks whether an index is inside the bounds of a node array.

Parameters:
- `nodes` - - Node list.
- `index` - - Candidate index.

Returns: True when index is valid.

### resetMutableRuntimeCollections

`(networkInternals: import("src/architecture/network/network.types").SerializeNetworkInternals) => void`

Clears mutable runtime collections before reconstruction.

Parameters:
- `networkInternals` - - Runtime internals.

Returns: Nothing.

### resolveNetworkSize

`(compactPayload: import("src/architecture/network/network.types").CompactPayloadContext, inputSizeOverride: number | undefined, outputSizeOverride: number | undefined) => import("src/architecture/network/network.types").ResolvedNetworkSizeContext`

Resolves effective input/output dimensions using optional explicit overrides.

When an override is provided, it takes precedence over serialized values.

Parameters:
- `compactPayload` - - Compact payload context.
- `inputSizeOverride` - - Optional input override.
- `outputSizeOverride` - - Optional output override.

Returns: Resolved network size context.

### resolveSizeOverride

`(overrideValue: number | undefined, serializedValue: number) => number`

Resolves one size value with override-first semantics.

Parameters:
- `overrideValue` - - Optional explicit override.
- `serializedValue` - - Serialized fallback value.

Returns: Effective size.

## architecture/network/serialize/network.serialize.activation.utils.ts

### findActivationByFunctionName

`(squashName: string | undefined) => import("src/methods/activation.utils").ActivationFunction | undefined`

Resolves activation by matching function.name.

Parameters:
- `squashName` - - Activation function name.

Returns: Activation function or undefined.

### findActivationByKey

`(squashName: string | undefined) => import("src/methods/activation.utils").ActivationFunction | undefined`

Resolves activation by direct key lookup.

Parameters:
- `squashName` - - Activation key.

Returns: Activation function or undefined.

### findActivationEntryByReference

`(squashFunction: import("src/methods/activation.utils").ActivationFunction) => [string, import("src/methods/activation.utils").ActivationFunction] | undefined`

Finds activation entry by function reference.

Parameters:
- `squashFunction` - - Activation function instance.

Returns: Activation entry or undefined.

### resolveActivationFunction

`(squashName: string | undefined) => import("src/methods/activation.utils").ActivationFunction`

Resolves an activation function from a stored key or function name.

Unknown values produce a warning and return the identity activation to keep
deserialization deterministic and non-throwing.

Parameters:
- `squashName` - - Activation key or function name.

Returns: Activation function.

Example:

```ts
const squashFunction = resolveActivationFunction('relu');
```

### resolveActivationKey

`(squashFunction: import("src/methods/activation.utils").ActivationFunction) => string`

Resolves a canonical activation key from a runtime activation function reference.

Resolution order is: direct registry reference match, then the function name,
then a stable identity fallback key.

Parameters:
- `squashFunction` - - Activation function instance.

Returns: Activation key.

Example:

```ts
import * as methods from '../../../methods/methods';

const key = resolveActivationKey(methods.Activation.tanh);
```

### resolveNamedActivationFromFunction

`(squashFunction: import("src/methods/activation.utils").ActivationFunction) => string | undefined`

Resolves activation name from function.name when non-empty.

Parameters:
- `squashFunction` - - Activation function instance.

Returns: Activation name or undefined.

### warnUnknownSquashName

`(squashName: string | undefined) => void`

Warns about unknown activation and fallback to identity.

Parameters:
- `squashName` - - Unknown activation name.

Returns: Nothing.
