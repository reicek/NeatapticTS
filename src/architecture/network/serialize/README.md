# architecture/network/serialize

Raised when network JSON serialization helpers receive an invalid root payload.

## architecture/network/serialize/network.serialize.errors.ts

### NetworkSerializeInvalidJsonError

Raised when network JSON serialization helpers receive an invalid root payload.

## architecture/network/serialize/network.serialize.utils.types.ts

### CompactConnectionRebuildContext

Context for compact-connection reconstruction.

Connection rows are processed independently so malformed entries can be skipped without aborting import.

### CompactNodeRebuildContext

Context for compact-node reconstruction.

Arrays are expected to be index-aligned so each node can be hydrated deterministically.

This is a serialization/hydration constraint only: the compact format stores node fields
(activation, state, squash, and optional gene id) as parallel arrays.

Do not read this as guidance for genetic alignment. In NEAT-style crossover and speciation,
homologous structure is matched by historical markings (innovation ids), not by array indices.

### CompactPayloadContext

Context carrying compact payload fields.

This named-object form replaces tuple index access in internal orchestration code.

### CompactSerializedNetworkTuple

Compact tuple payload used by `serialize` output.

Tuple slots are intentionally positional to reduce payload size:
0) activations, 1) states, 2) squash keys, 3) connections, 4) input size,
5) output size, 6) optional node gene ids, 7) optional topology intent.

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

Endpoints stay index-based for deterministic reconstruction, while the optional
historical fields preserve NEAT identity across clone, export, and restore flows.

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

Connection (Synapse / Edge)
===========================
Directed weighted link between two nodes. The connection keeps the everyday
graph fields (`from`, `to`, `weight`, `innovation`) directly on the instance,
then pushes rarer capabilities behind symbol-backed accessors so large
populations do not pay object-shape costs for features they are not using.

This makes the boundary useful in three different modes:

- ordinary feed-forward links that only need endpoints and weight,
- gated or plastic links that gradually opt into extra runtime state,
- optimizer-heavy training paths that need moment buffers without turning
  every connection into a bloated record.

Example:

```ts
const source = new Node('input');
const target = new Node('output');
const edge = new Connection(source, target, 0.42);

edge.gain = 1.5;
edge.enabled = true;
```

### applyHydratedArchitectureDescriptor

```ts
applyHydratedArchitectureDescriptor(
  network: default,
  architectureDescriptor: NetworkArchitectureDescriptor | undefined,
): void
```

Applies hydrated architecture metadata to runtime network when shape is valid.

Parameters:
- `network` - Rebuilt network instance.
- `architectureDescriptor` - Optional serialized descriptor.

Returns: Nothing.

### applyHydratedExtensionBag

```ts
applyHydratedExtensionBag(
  runtimeNetwork: RuntimeNetworkWithSerializedExtensions,
  extensions: NetworkJSONExtensions | undefined,
): void
```

Hydrates the generic extension bag onto the runtime network when shape is valid.

Parameters:
- `runtimeNetwork` - Runtime network receiving the extension bag.
- `extensions` - Optional serialized extension bag.

Returns: Nothing.

### applySerializedExtensionBag

```ts
applySerializedExtensionBag(
  runtimeNetwork: RuntimeNetworkWithSerializedExtensions,
  networkJson: NetworkJSON,
): void
```

Re-emits the hydrated generic extension bag on verbose JSON snapshots.

Parameters:
- `runtimeNetwork` - Runtime network that may carry hydrated extensions.
- `networkJson` - Target JSON payload.

Returns: Nothing.

### cloneNetworkJsonExtensions

```ts
cloneNetworkJsonExtensions(
  extensions: NetworkJSONExtensions | undefined,
): NetworkJSONExtensions | undefined
```

Clones one generic network extension bag when the top-level shape is valid.

Parameters:
- `extensions` - Optional serialized extension bag.

Returns: Cloned extension bag or `undefined` when shape is invalid.

### deserialize

```ts
deserialize(
  data: CompactSerializedNetworkTuple,
  inputSize: number | undefined,
  outputSize: number | undefined,
): default
```

Rebuilds a network instance from compact tuple form.

Use this importer for compact payloads produced by `serialize`.
Optional `inputSize` and `outputSize` let callers enforce shape overrides at import time.

Parameters:
- `data` - Compact tuple payload.
- `inputSize` - Optional input-size override that takes precedence over serialized input.
- `outputSize` - Optional output-size override that takes precedence over serialized output.

Returns: Reconstructed network instance.

Example:

```ts
import { deserialize } from './network.serialize.utils';

const rebuiltNetwork = deserialize(compactTuple, 2, 1);
```

### fromJSONImpl

```ts
fromJSONImpl(
  json: NetworkJSON,
): default
```

Reconstructs a network instance from the verbose JSON payload.

This importer validates payload shape, restores dropout and topology, and then rebuilds
connections, gating relationships, and optional enabled flags.

Parameters:
- `json` - Verbose JSON payload.

Returns: Reconstructed network instance.

Example:

```ts
import { fromJSONImpl } from './network.serialize.utils';

const rebuiltNetwork = fromJSONImpl(snapshotJson);
```

### isArchitectureDescriptorShapeValid

```ts
isArchitectureDescriptorShapeValid(
  architectureDescriptor: NetworkArchitectureDescriptor | undefined,
): boolean
```

Parameters:
- `architectureDescriptor` - Optional descriptor candidate.

Returns: True when minimal descriptor shape is valid.

### serialize

```ts
serialize(): CompactSerializedNetworkTuple
```

Serializes a network instance into the compact tuple format.

Use this format when payload size and serialization speed matter more than readability.
The tuple layout is positional and optimized for transport/storage efficiency.

Parameters:
- `this` - Bound network instance.

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

Endpoints stay index-based for deterministic reconstruction, while the optional
historical fields preserve NEAT identity across clone, export, and restore flows.

### toJSONImpl

```ts
toJSONImpl(): NetworkJSON
```

Serializes a network instance into the verbose JSON format.

Use this format when you need human-readable snapshots, explicit schema versioning,
and better forward/backward compatibility handling.

Parameters:
- `this` - Bound network instance.

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

```ts
acquire(
  from: default,
  to: default,
  weight: number | undefined,
): default
```

Acquire a connection from the internal pool, or construct a fresh one when the pool is empty.
This is the low-allocation path used by topology mutation and other edge-churn heavy flows.

Parameters:
- `from` - Source node.
- `to` - Target node.
- `weight` - Optional initial weight.

Returns: Reinitialized connection instance.

#### dcMask

DropConnect active mask: `1` means active for this stochastic pass, `0` means dropped.

#### dropConnectActiveMask

Convenience alias for DropConnect mask with clearer naming.

#### eligibility

Standard eligibility trace (e.g., for RTRL / policy gradient credit assignment).

#### enabled

Whether the gene is currently expressed and participates in the forward pass.

#### firstMoment

First moment estimate used by Adam-family optimizers.

#### from

The source (pre-synaptic) node supplying activation.

#### gain

Multiplicative modulation applied after weight. Neutral gain `1` is omitted from storage.

#### gater

Optional gating node whose activation modulates effective weight.

#### gradientAccumulator

Generic gradient accumulator used by RMSProp and AdaGrad.

#### hasGater

Whether a gater node is assigned to modulate this connection's effective weight.

#### infinityNorm

Adamax infinity norm accumulator.

#### innovation

Unique historical marking (auto-increment) for evolutionary alignment.

#### innovationID

```ts
innovationID(
  sourceNodeId: number,
  targetNodeId: number,
): number
```

Deterministic Cantor pairing function for a `(sourceNodeId, targetNodeId)` pair.
Use it when you need a stable edge identifier without relying on the mutable
auto-increment counter.

Parameters:
- `sourceNodeId` - Source node integer id or index.
- `targetNodeId` - Target node integer id or index.

Returns: Unique non-negative integer derived from the ordered pair.

Example:

```ts
const id = Connection.innovationID(2, 5);
```

#### lookaheadShadowWeight

Lookahead slow-weight snapshot.

#### maxSecondMoment

AMSGrad maximum of past second-moment estimates.

#### nextInnovation

Read the current next-innovation cursor without advancing it.

Use this to seed an external innovation tracker so its counter never
overlaps with innovation IDs already assigned by the Connection constructor.

Returns: Current value of the monotonic connection innovation counter.

#### plastic

Whether this connection participates in plastic adaptation.

#### plasticityRate

Per-connection plasticity rate. `0` means the connection is not plastic.

#### previousDeltaWeight

Last applied delta weight (used by classic momentum).

#### release

```ts
release(
  conn: default,
): void
```

Return a connection instance to the internal pool for later reuse.
Treat the instance as surrendered after calling this method.

Parameters:
- `conn` - The connection instance to recycle.

Returns: Nothing.

#### resetInnovationCounter

```ts
resetInnovationCounter(
  value: number,
): void
```

Reset the monotonic innovation counter used for newly constructed or pooled connections.
You usually call this at the start of an experiment or before rebuilding a whole population.

Parameters:
- `value` - New starting value.

Returns: Nothing.

#### secondMoment

Second raw moment estimate used by Adam-family optimizers.

#### secondMomentum

Secondary momentum buffer used by Lion-style updates.

#### syncInnovationCounter

```ts
syncInnovationCounter(
  maxObservedInnovation: number,
): void
```

Advances the innovation cursor past a restored maximum.

This keeps import and clone paths monotonic: once a payload brings in a high
innovation id, newly created edges continue from above that value.

Parameters:
- `maxObservedInnovation` - Highest restored innovation id currently in memory.

Returns: Nothing.

#### to

The target (post-synaptic) node receiving activation.

#### toJSON

```ts
toJSON(): { from: number | undefined; to: number | undefined; weight: number; gain: number; innovation: number; enabled: boolean; gater?: number | undefined; }
```

Serialize to a minimal JSON-friendly shape used by genome and network save flows.
Undefined node indices are preserved so callers can resolve or remap them later.

Returns: Object with node indices, weight, gain, innovation id, enabled flag, and gater index when one exists.

Example:

```ts
const json = connection.toJSON();
// => { from: 0, to: 3, weight: 0.12, gain: 1, innovation: 57, enabled: true }
```

#### totalDeltaWeight

Accumulated (batched) delta weight awaiting an apply step.

#### weight

Scalar multiplier applied to the source activation (prior to gain modulation).

#### xtrace

Extended trace structure for modulatory / eligibility propagation algorithms. Parallel arrays for cache-friendly iteration.

## architecture/network/serialize/network.serialize.json.utils.ts

### appendJsonForwardConnections

```ts
appendJsonForwardConnections(
  networkInternals: SerializeNetworkInternals,
  networkJson: NetworkJSON,
): void
```

Appends JSON entries for forward connections.

Non-finite endpoint indices are ignored to prevent malformed output records.

Parameters:
- `networkInternals` - Runtime internals.
- `networkJson` - JSON accumulator.

Returns: Nothing.

### appendJsonNodesAndSelfConnections

```ts
appendJsonNodesAndSelfConnections(
  networkInternals: SerializeNetworkInternals,
  networkJson: NetworkJSON,
): void
```

Appends JSON node entries and optional self-connections.

Node indices are refreshed during this step so connection records can reference
stable numeric endpoints.

Parameters:
- `networkInternals` - Runtime internals.
- `networkJson` - Target JSON accumulator.

Returns: Nothing.

### appendJsonSelfConnectionWhenPresent

```ts
appendJsonSelfConnectionWhenPresent(
  nodeInternals: SerializeNodeInternals,
  nodeIndex: number,
  networkJson: NetworkJSON,
): void
```

Appends JSON self-connection when node has one.

Parameters:
- `nodeInternals` - Node internals.
- `nodeIndex` - Node index.
- `networkJson` - JSON accumulator.

Returns: Nothing.

### assignJsonEnabledFlagWhenProvided

```ts
assignJsonEnabledFlagWhenProvided(
  createdConnection: default | undefined,
  enabled: boolean,
): void
```

Assigns enabled flag when value is provided.

Parameters:
- `createdConnection` - Created connection.
- `enabled` - Optional enabled value.

Returns: Nothing.

### assignJsonGainWhenProvided

```ts
assignJsonGainWhenProvided(
  createdConnection: default | undefined,
  gain: number | undefined,
): void
```

Assigns a restored connection gain when one was serialized explicitly.

Parameters:
- `createdConnection` - Created connection.
- `gain` - Optional serialized gain.

Returns: Nothing.

### assignJsonGaterWhenValid

```ts
assignJsonGaterWhenValid(
  networkInternals: SerializeNetworkInternals,
  gaterIndex: number | null,
  createdConnection: default | undefined,
): void
```

Assigns JSON gater when connection and gater index are valid.

Parameters:
- `networkInternals` - Runtime internals.
- `gaterIndex` - Optional gater index.
- `createdConnection` - Created connection.

Returns: Nothing.

### createConnection

```ts
createConnection(
  networkInternals: SerializeNetworkInternals,
  sourceNode: default,
  targetNode: default,
  weight: number,
): default | undefined
```

Creates one connection and returns first created instance.

Parameters:
- `networkInternals` - Runtime internals.
- `sourceNode` - Source node.
- `targetNode` - Target node.
- `weight` - Connection weight.

Returns: Created connection or undefined.

### createEmptyNetworkJson

```ts
createEmptyNetworkJson(
  networkInternals: NetworkInternalsWithDropout,
): NetworkJSON
```

Creates an empty verbose JSON shell from runtime internals.

The shell includes format and shape metadata and is filled in by subsequent
node and connection append steps.

Parameters:
- `networkInternals` - Runtime internals with optional dropout.

Returns: Empty JSON shell with `formatVersion` and scalar metadata initialized.

Example:

```ts
const networkJson = createEmptyNetworkJson(networkInternals);
```

### createJsonConnection

```ts
createJsonConnection(
  connectionInstance: default,
  from: number,
  to: number,
): NetworkJSONConnection
```

Creates one JSON connection entry.

Parameters:
- `connectionInstance` - Runtime connection carrying the historical identity to persist.
- `from` - Source index.
- `to` - Target index.

Returns: JSON connection entry.

### createJsonNode

```ts
createJsonNode(
  node: default,
  nodeInternals: SerializeNodeInternals,
  nodeIndex: number,
): NetworkJSONNode
```

Creates one JSON node entry.

Parameters:
- `node` - Runtime node.
- `nodeInternals` - Node internals.
- `nodeIndex` - Canonical index.

Returns: JSON node entry.

### createNodeWithType

```ts
createNodeWithType(
  nodeType: string,
): default
```

Creates one node with provided type.

Parameters:
- `nodeType` - Node type.

Returns: New node.

### hydrateNodeFromJsonEntry

```ts
hydrateNodeFromJsonEntry(
  rebuiltNode: default,
  nodeJsonEntry: NetworkJSONNode,
  nodeIndex: number,
): void
```

Hydrates one node from JSON node entry.

Parameters:
- `rebuiltNode` - Node to hydrate.
- `nodeJsonEntry` - JSON node entry.
- `nodeIndex` - Canonical node index.

Returns: Nothing.

### isConnectionEnabled

```ts
isConnectionEnabled(
  connectionInstance: default,
): boolean
```

Resolves enabled status from optional connection flag.

Parameters:
- `connectionInstance` - Connection instance.

Returns: True when connection is enabled.

### isCurrentJsonEndpoint

```ts
isCurrentJsonEndpoint(
  nodes: default[],
  endpointNode: default,
  endpointIndex: number,
): boolean
```

Validates that one serialized endpoint still points at the canonical node table.

Parameters:
- `nodes` - Canonical node table being serialized.
- `endpointNode` - Connection endpoint node reference.
- `endpointIndex` - Endpoint index stored on the node internals.

Returns: True when the endpoint still belongs to the canonical node table.

### isJsonConnectionInNodeBounds

```ts
isJsonConnectionInNodeBounds(
  nodes: default[],
  connectionJsonEntry: NetworkJSONConnection,
): boolean
```

Checks JSON connection indices against node list bounds.

Parameters:
- `nodes` - Node list.
- `connectionJsonEntry` - JSON connection entry.

Returns: True when both indices are valid.

### isJsonConnectionShapeValid

```ts
isJsonConnectionShapeValid(
  connectionJsonEntry: NetworkJSONConnection,
): boolean
```

Checks that JSON connection has numeric endpoint fields.

Parameters:
- `connectionJsonEntry` - JSON connection entry.

Returns: True when endpoint fields are numbers.

### rebuildConnectionsFromJsonPayload

```ts
rebuildConnectionsFromJsonPayload(
  jsonConnectionContext: JsonConnectionRebuildContext,
): void
```

Rebuilds runtime connections from verbose JSON entries.

Invalid entries are skipped with warnings so import continues for valid records.

Parameters:
- `jsonConnectionContext` - JSON connection rebuild context.

Returns: Nothing.

Example:

```ts
rebuildConnectionsFromJsonPayload({
  networkInternals,
  connectionJsonEntries,
});
```

### rebuildNodesFromJsonPayload

```ts
rebuildNodesFromJsonPayload(
  jsonNodeContext: JsonNodeRebuildContext,
): void
```

Rebuilds runtime nodes from verbose JSON entries.

Parameters:
- `jsonNodeContext` - JSON node rebuild context.

Returns: Nothing.

Example:

```ts
rebuildNodesFromJsonPayload({ networkInternals, nodeJsonEntries });
```

### rebuildOneJsonConnection

```ts
rebuildOneJsonConnection(
  networkInternals: SerializeNetworkInternals,
  connectionJsonEntry: NetworkJSONConnection,
): void
```

Rebuilds one verbose JSON connection entry.

Parameters:
- `networkInternals` - Runtime internals.
- `connectionJsonEntry` - JSON connection entry.

Returns: Nothing.

### resolveDropout

```ts
resolveDropout(
  dropout: number | undefined,
): number
```

Resolves dropout with a numeric fallback when the value is absent.

Parameters:
- `dropout` - Optional dropout value.

Returns: Effective dropout.

### resolveGaterIndex

```ts
resolveGaterIndex(
  gaterNode: default | null,
): number | null
```

Resolves gater node index from gater reference.

Parameters:
- `gaterNode` - Optional gater node.

Returns: Gater index or null.

### validateNetworkJsonOrThrow

```ts
validateNetworkJsonOrThrow(
  json: NetworkJSON,
): void
```

Validates the verbose JSON payload root shape.

Parameters:
- `json` - Payload candidate.

Returns: Nothing.

### warnWhenJsonFormatVersionIsUnknown

```ts
warnWhenJsonFormatVersionIsUnknown(
  formatVersion: number,
): void
```

Warns when incoming verbose format version differs from the expected one.

Parameters:
- `formatVersion` - Incoming format version.

Returns: Nothing.

## architecture/network/serialize/network.serialize.public.utils.ts

Public serialization-facing helpers that stay above the lower-level payload builders.

This file owns small convenience methods that callers expect on `Network`
itself, while delegating the real persistence work to the serialize chapter.

### cloneImpl

```ts
cloneImpl(): default
```

Create a deep copy of one network through the verbose JSON round-trip.

This keeps cloning behavior aligned with the same versioned payload contract
used by `toJSON()` and `fromJSON()`, so clone semantics stay stable as the
serialization chapter evolves.

Parameters:
- `this` - Target network instance.

Returns: Deep-cloned network instance.

## architecture/network/serialize/network.serialize.compact.utils.ts

### assignCompactGaterGeneIdWhenProvided

```ts
assignCompactGaterGeneIdWhenProvided(
  networkInternals: SerializeNetworkInternals,
  gaterIndex: number | null,
  gaterGeneId: number | null | undefined,
): void
```

Restores a gater node's historical gene id when compact metadata provides it.

Parameters:
- `networkInternals` - Runtime internals.
- `gaterIndex` - Optional compact gater index.
- `gaterGeneId` - Optional persisted gater gene id.

Returns: Nothing.

### assignCompactGaterWhenValid

```ts
assignCompactGaterWhenValid(
  networkInternals: SerializeNetworkInternals,
  gaterIndex: number | null,
  createdConnection: default | undefined,
): void
```

Assigns compact gater when both connection and gater index are valid.

Parameters:
- `networkInternals` - Runtime internals.
- `gaterIndex` - Optional gater index.
- `createdConnection` - Created connection.

Returns: Nothing.

### collectAllConnections

```ts
collectAllConnections(
  networkInternals: SerializeNetworkInternals,
): default[]
```

Collects all runtime connections into a single list.

Parameters:
- `networkInternals` - Runtime internals.

Returns: Combined connections.

### collectNodeActivations

```ts
collectNodeActivations(
  nodes: default[],
): number[]
```

Collects node activation values in positional order.

Parameters:
- `nodes` - Node list.

Returns: Activation list aligned to node indices.

### collectNodeGeneIds

```ts
collectNodeGeneIds(
  nodes: default[],
): (number | null)[]
```

Collects stable node gene ids aligned to compact node order.

This optional compact payload slot closes the identity gap that previously forced
restored nodes to receive fresh constructor-time ids.

Parameters:
- `nodes` - Node list in compact export order.

Returns: Gene-id list aligned to node indices.

### collectNodeSquashKeys

```ts
collectNodeSquashKeys(
  nodes: default[],
): string[]
```

Collects node activation keys in positional order.

Each function reference is normalized to a string key for compact transfer.

Parameters:
- `nodes` - Node list.

Returns: Squash-key list aligned to node indices.

### collectNodeStates

```ts
collectNodeStates(
  nodes: default[],
): number[]
```

Collects node state values in positional order.

Parameters:
- `nodes` - Node list.

Returns: State list aligned to node indices.

### collectSerializedConnections

```ts
collectSerializedConnections(
  networkInternals: SerializeNetworkInternals,
): SerializedConnection[]
```

Collects compact connection records from forward and self connection groups.

Parameters:
- `networkInternals` - Runtime internals.

Returns: Serialized connection list.

### createConnection

```ts
createConnection(
  networkInternals: SerializeNetworkInternals,
  sourceNode: default,
  targetNode: default,
  weight: number,
): default | undefined
```

Creates one connection and returns first created instance.

Parameters:
- `networkInternals` - Runtime internals.
- `sourceNode` - Source node.
- `targetNode` - Target node.
- `weight` - Connection weight.

Returns: Created connection or undefined.

### createNodeWithType

```ts
createNodeWithType(
  nodeType: string,
): default
```

Creates one node with provided type.

Parameters:
- `nodeType` - Node type.

Returns: New node.

### hydrateNodeStateFromCompactPayload

```ts
hydrateNodeStateFromCompactPayload(
  rebuiltNode: default,
  activation: number,
  state: number,
  squashName: string | undefined,
  nodeIndex: number,
): void
```

Hydrates node runtime state from compact tuple values.

Parameters:
- `rebuiltNode` - Node to hydrate.
- `activation` - Activation value.
- `state` - State value.
- `squashName` - Activation key.
- `nodeIndex` - Canonical node index.

Returns: Nothing.

### isSerializedConnectionInNodeBounds

```ts
isSerializedConnectionInNodeBounds(
  networkInternals: SerializeNetworkInternals,
  serializedConnection: SerializedConnection,
): boolean
```

Checks compact connection bounds against current node list.

Parameters:
- `networkInternals` - Runtime internals.
- `serializedConnection` - Serialized connection record.

Returns: True when endpoints are valid.

### rebuildConnectionsFromCompactPayload

```ts
rebuildConnectionsFromCompactPayload(
  compactConnectionContext: CompactConnectionRebuildContext,
): void
```

Rebuilds runtime connections from compact connection records.

Invalid endpoint or gater indices are skipped with warnings to preserve import flow.

Parameters:
- `compactConnectionContext` - Compact connection rebuild context.

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

```ts
rebuildNodesFromCompactPayload(
  networkInternals: SerializeNetworkInternals,
  compactNodeContext: CompactNodeRebuildContext,
): void
```

Rebuilds runtime nodes from compact payload arrays.

Node type is inferred from index position relative to input/output boundaries.

Parameters:
- `networkInternals` - Runtime internals.
- `compactNodeContext` - Compact node rebuild context.

Returns: Nothing.

Example:

```ts
import { rebuildNodesFromCompactPayload } from './network.serialize.compact.utils';

rebuildNodesFromCompactPayload(networkInternals, compactNodeContext);
```

### rebuildOneCompactConnection

```ts
rebuildOneCompactConnection(
  networkInternals: SerializeNetworkInternals,
  serializedConnection: SerializedConnection,
): void
```

Rebuilds one compact serialized connection.

Parameters:
- `networkInternals` - Runtime internals.
- `serializedConnection` - Serialized connection record.

Returns: Nothing.

### refreshNodeIndices

```ts
refreshNodeIndices(
  nodes: default[],
): void
```

Refreshes `node.index` for each node in list order.

Canonical indices are required so compact connection records can store endpoints
and gaters as stable numeric positions.

Parameters:
- `nodes` - Node list.

Returns: Nothing.

### resolveNodeTypeFromCompactIndex

```ts
resolveNodeTypeFromCompactIndex(
  nodeIndex: number,
  totalNodeCount: number,
  input: number,
  output: number,
): string
```

Resolves node type from compact tuple position.

Parameters:
- `nodeIndex` - Node index.
- `totalNodeCount` - Total node count.
- `input` - Input size.
- `output` - Output size.

Returns: Node type string.

### serializeOneConnection

```ts
serializeOneConnection(
  connectionInstance: default,
): SerializedConnection
```

Serializes one connection into compact indexed form.

Parameters:
- `connectionInstance` - Runtime connection.

Returns: Serialized connection record.

## architecture/network/serialize/network.serialize.runtime.utils.ts

### applyRestoredConnectionIdentity

```ts
applyRestoredConnectionIdentity(
  createdConnection: default | undefined,
  identity: { innovation?: number | undefined; enabled?: boolean | undefined; },
): void
```

Restores persisted connection identity onto a freshly created runtime connection.

Import paths still build edges through `connect()` so graph bookkeeping stays
centralized. This helper then reapplies the persisted innovation and enabled state.

Parameters:
- `createdConnection` - Newly created runtime connection.
- `identity` - Persisted identity metadata.

Returns: Nothing.

### asNetworkInternals

```ts
asNetworkInternals(
  network: default,
): SerializeNetworkInternals
```

Casts a network instance to the internal runtime shape used by serializer helpers.

Parameters:
- `network` - Network instance.

Returns: Runtime internals.

### asNetworkInternalsWithDropout

```ts
asNetworkInternalsWithDropout(
  network: default,
): NetworkInternalsWithDropout
```

Casts a network instance to internals that include optional dropout metadata.

Parameters:
- `network` - Network instance.

Returns: Runtime internals with optional dropout.

### asNodeInternals

```ts
asNodeInternals(
  node: default,
): SerializeNodeInternals
```

Casts a node instance to its internal runtime representation.

Parameters:
- `node` - Node instance.

Returns: Node internals.

### createCompactPayloadContext

```ts
createCompactPayloadContext(
  data: CompactSerializedNetworkTuple,
): CompactPayloadContext
```

Normalizes a compact tuple payload into a named object context.

This improves readability in orchestration code by replacing positional tuple access
with semantically named fields.

Parameters:
- `data` - Compact tuple payload.

Returns: Normalized payload context.

Example:

```ts
const compactPayload = createCompactPayloadContext(compactTuple);
```

### createNetworkInstance

```ts
createNetworkInstance(
  input: number,
  output: number,
): default
```

Creates a new network instance for deserialize workflows.

Parameters:
- `input` - Input size.
- `output` - Output size.

Returns: New network instance.

### hydrateNodeGeneIdWhenProvided

```ts
hydrateNodeGeneIdWhenProvided(
  node: default,
  geneId: number | null | undefined,
): void
```

Writes a restored node gene id when serialized identity data is available.

Compact restore paths construct fresh runtime nodes first, then overwrite the
temporary constructor-assigned ids with persisted historical ids.

Parameters:
- `node` - Restored runtime node.
- `geneId` - Persisted stable gene id.

Returns: Nothing.

### isFiniteIndex

```ts
isFiniteIndex(
  index: number,
): boolean
```

Checks whether a candidate index value is a finite number.

Parameters:
- `index` - Candidate index value.

Returns: True when finite number.

### isNodeIndexInBounds

```ts
isNodeIndexInBounds(
  nodes: default[],
  index: number,
): boolean
```

Checks whether an index is inside the bounds of a node array.

Parameters:
- `nodes` - Node list.
- `index` - Candidate index.

Returns: True when index is valid.

### resetMutableRuntimeCollections

```ts
resetMutableRuntimeCollections(
  networkInternals: SerializeNetworkInternals,
): void
```

Clears mutable runtime collections before reconstruction.

Parameters:
- `networkInternals` - Runtime internals.

Returns: Nothing.

### resolveNetworkSize

```ts
resolveNetworkSize(
  compactPayload: CompactPayloadContext,
  inputSizeOverride: number | undefined,
  outputSizeOverride: number | undefined,
): ResolvedNetworkSizeContext
```

Resolves effective input/output dimensions using optional explicit overrides.

When an override is provided, it takes precedence over serialized values.

Parameters:
- `compactPayload` - Compact payload context.
- `inputSizeOverride` - Optional input override.
- `outputSizeOverride` - Optional output override.

Returns: Resolved network size context.

### resolveSizeOverride

```ts
resolveSizeOverride(
  overrideValue: number | undefined,
  serializedValue: number,
): number
```

Resolves one size value with override-first semantics.

Parameters:
- `overrideValue` - Optional explicit override.
- `serializedValue` - Serialized fallback value.

Returns: Effective size.

### syncRestoredHistoricalCounters

```ts
syncRestoredHistoricalCounters(
  networkInternals: SerializeNetworkInternals,
): void
```

Advances static node and connection counters past all restored historical ids.

Without this step, a fresh process could deserialize a high-id genome and then
allocate colliding `geneId` or `innovation` values on the next mutation.

Parameters:
- `networkInternals` - Restored mutable network internals.

Returns: Nothing.

## architecture/network/serialize/network.serialize.activation.utils.ts

### findActivationByFunctionName

```ts
findActivationByFunctionName(
  squashName: string | undefined,
): ActivationFunction | undefined
```

Resolves activation by matching function.name.

Parameters:
- `squashName` - Activation function name.

Returns: Activation function or undefined.

### findActivationByKey

```ts
findActivationByKey(
  squashName: string | undefined,
): ActivationFunction | undefined
```

Resolves activation by direct key lookup.

Parameters:
- `squashName` - Activation key.

Returns: Activation function or undefined.

### findActivationEntryByReference

```ts
findActivationEntryByReference(
  squashFunction: ActivationFunction,
): [string, ActivationFunction] | undefined
```

Finds activation entry by function reference.

Parameters:
- `squashFunction` - Activation function instance.

Returns: Activation entry or undefined.

### resolveActivationFunction

```ts
resolveActivationFunction(
  squashName: string | undefined,
): ActivationFunction
```

Resolves an activation function from a stored key or function name.

Unknown values produce a warning and return the identity activation to keep
deserialization deterministic and non-throwing.

Parameters:
- `squashName` - Activation key or function name.

Returns: Activation function.

Example:

```ts
const squashFunction = resolveActivationFunction('relu');
```

### resolveActivationKey

```ts
resolveActivationKey(
  squashFunction: ActivationFunction,
): string
```

Resolves a canonical activation key from a runtime activation function reference.

Resolution order is: attached stable key, direct registry reference match,
then the function name, then a stable identity fallback key.

Parameters:
- `squashFunction` - Activation function instance.

Returns: Activation key.

Example:

```ts
import * as methods from '../../../methods/methods';

const key = resolveActivationKey(methods.Activation.tanh);
```

### resolveAttachedActivationKey

```ts
resolveAttachedActivationKey(
  squashFunction: ActivationFunction,
): string | undefined
```

Resolves a stable activation key attached directly to the function object.

Parameters:
- `squashFunction` - Activation function instance.

Returns: Attached activation key or undefined.

### resolveNamedActivationFromFunction

```ts
resolveNamedActivationFromFunction(
  squashFunction: ActivationFunction,
): string | undefined
```

Resolves activation name from function.name when non-empty.

Parameters:
- `squashFunction` - Activation function instance.

Returns: Activation name or undefined.

### warnUnknownSquashName

```ts
warnUnknownSquashName(
  squashName: string | undefined,
): void
```

Warns about unknown activation and fallback to identity.

Parameters:
- `squashName` - Unknown activation name.

Returns: Nothing.
