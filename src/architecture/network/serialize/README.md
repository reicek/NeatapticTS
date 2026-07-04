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

### CompressedSerializedConnectionBlock

Array-oriented compressed connection payload for compact serialization.

This keeps the compact serializer lossless while removing per-connection key
repetition and object allocation overhead from the transport payload.

### CompressedSerializedConnectionWeights

Lossless weight-word payload for compressed compact serialization.

The encoding stores each non-zero float64 weight as four signed 16-bit words
and then delta-encodes those words across the non-zero connection sequence.
Exact positive-zero spans are represented separately as run metadata.

### CompressedSerializedIndexRun

Index-aligned run metadata used by compressed connection payloads.

A run starts at `startIndex` and covers `length` contiguous connection rows.

### CompressedSerializedNetwork

Compressed compact serialization payload.

This format is additive to the legacy compact tuple API: it keeps the same
runtime reconstruction semantics while using array-oriented connection data
to reduce UTF-8 payload size for storage or transport.

### CompressedSerializedNetworkArchive

Node-side archive wrapper around a compressed compact serialization payload.

The wrapped `payload` string stores the UTF-8 JSON form of
`CompressedSerializedNetwork` after gzip or zstd compression, encoded as
base64 for portable storage.

### CompressedSerializedNetworkArchiveCompression

Supported Node-side compression codecs for writing compressed serialized network archive payloads.

### CompressedSerializedNetworkArchiveOptions

Optional codec settings for archiving one compressed network payload in binary form.

### ConnectionInternalsWithEnabled

Connection view with optional enabled flag.

Some serialized formats preserve per-edge enablement, while others treat missing values
as implicitly enabled.

### DEFAULT_NUMERIC_VALUE

Neutral numeric fallback applied whenever optional scalar fields are absent in serialized payloads during rebuild normalization.
Using a single constant keeps rebuild defaults predictable across compact and verbose import paths.

### ERROR_INVALID_NETWORK_JSON

Stable error text thrown when verbose JSON root validation fails before any field-level reconstruction logic executes.
Keeping this message deterministic helps tests and diagnostics tooling match invalid-payload failures.

### FALLBACK_ACTIVATION_KEY

Activation key used when a serialized squash name cannot be resolved to any registered runtime activation function safely.
Identity preserves import continuity while warning paths surface the unknown key to diagnostics.

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

Current verbose `NetworkJSON` schema version used by serializer and deserializer boundaries to coordinate compatibility checks and migration warnings.
Readers should treat mismatches as migration signals rather than silently assuming field parity.

### NetworkInternalsWithDropout

Serialize internals with optional dropout field.

Verbose JSON snapshots normalize this value so readers can treat dropout as numeric data.

### NetworkJSON

Canonical verbose network JSON contract used across serializer and deserializer entry points in persistence, migration, and diagnostics workflows.
This local alias keeps payload-shape ownership discoverable from the serialize boundary without requiring deep type imports.

### NetworkJSONConnection

Verbose JSON connection representation.

Includes optional gater and explicit enabled state for portability.

### NetworkJSONNode

Verbose JSON node representation.

Node entries are self-describing and intended for readable, versioned snapshots.

### NODE_TYPE_HIDDEN

Node type literal for hidden layer nodes used during JSON serialization and deserialization rebuilds.

### NODE_TYPE_INPUT

Node type literal for input layer nodes used during JSON serialization and deserialization rebuilds.

### NODE_TYPE_OUTPUT

Node type literal for output layer nodes used during JSON serialization and deserialization rebuilds.

### ParameterLayoutEntry

One ordered descriptor inside parameter-layout version `1`.

Bias entries use the stable node gene id in `nodeId`.
Weight entries prefer the live connection `innovation` when it exists and
otherwise fall back to the stable endpoint gene ids in `from` and `to`.
Duplicate innovation ids or duplicate fallback endpoint pairs are rejected
as ambiguous instead of inheriting incidental container order.

### ParameterLayoutV1

Deterministic parameter-layout descriptor owned by the network serialize boundary.

Version `1` keeps one stable fold order: all bias entries first, then all
weight entries. For a fixed topology with stable historical ids, this gives
ordered determinism on the same runtime.

### ParameterVector

Versioned parameter payload for same-runtime vector roundtrips.

Layout metadata and scalar values travel together so imports can reject
incompatible payloads before mutating a live network.

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

Warning emitted when compact deserialize encounters invalid edge endpoints and skips the connection reconstruction silently.

### WARNING_INVALID_CONNECTION_DURING_FROM_JSON

Warning emitted when JSON verbose deserialize encounters invalid edge endpoints and skips the connection reconstruction silently.

### WARNING_INVALID_GATER_DURING_DESERIALIZE

Warning emitted when compact deserialize encounters an invalid gater index and skips the gater assignment silently.

### WARNING_INVALID_GATER_DURING_FROM_JSON

Warning emitted when JSON verbose deserialize encounters an invalid gater index and skips the gater assignment silently.

### WARNING_UNKNOWN_FORMAT_VERSION

Warning emitted for unknown verbose format versions encountered during JSON import boundaries with best-effort fallback semantics.
Keep this message stable so diagnostics tooling can classify unknown-format imports consistently.

### WARNING_UNKNOWN_SQUASH_PREFIX

Prefix used when emitting a warning about an unknown activation squash key encountered during deserialization.

### WARNING_UNKNOWN_SQUASH_SUFFIX

Suffix appended when emitting a warning about an unknown activation squash key encountered during deserialization.

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

### applySerializedRuntimeState

```ts
applySerializedRuntimeState(
  rebuiltNetwork: default,
  activations: number[],
  states: number[],
): void
```

Restore live activation and recurrent-state scalars after structural import.

Parameters:
- `rebuiltNetwork` - Reconstructed runtime network.
- `activations` - Activation values aligned to node order.
- `states` - Recurrent state values aligned to node order.

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

### createParameterLayoutV1

```ts
createParameterLayoutV1(
  network: default,
): ParameterLayoutV1
```

Build the deterministic descriptor order for parameter-layout version `1`.

Version `1` folds all bias entries before all weight entries. Biases are
ordered by stable node gene id. Weights prefer connection innovation when it
exists and fall back to stable endpoint gene ids when it does not.
Ambiguous or missing ordering identity fails explicitly instead of inheriting
incidental runtime array order. For a fixed topology with stable historical
identities, the result is ordered deterministic on the same runtime.
This helper describes layout order only; it is not a cross-runtime exact
replay promise.

Parameters:
- `network` - Live network instance to inspect.

Returns: Versioned ordered layout descriptors for the current topology.

Example:

```ts
const layout = createParameterLayoutV1(network);
console.log(layout.entries[0]);
```

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

### deserializeCompressed

```ts
deserializeCompressed(
  data: CompressedSerializedNetwork,
  inputSize: number | undefined,
  outputSize: number | undefined,
): default
```

Rebuild a network instance from the compressed compact payload and restore runtime activation/state vectors after structure import.
The function validates payload format first so incompatible compressed data fails before partial reconstruction occurs.

Parameters:
- `data` - Compressed compact payload.
- `inputSize` - Optional input-size override.
- `outputSize` - Optional output-size override.

Returns: Reconstructed network instance.

### deserializeCompressedArchive

```ts
deserializeCompressedArchive(
  data: CompressedSerializedNetworkArchive,
  inputSize: number | undefined,
  outputSize: number | undefined,
): default
```

Rebuild a network instance from the compressed archive wrapper by inflating archive bytes and delegating to compressed deserialization.
This keeps archive-specific decode concerns separate from structural reconstruction and runtime-state restoration.

Parameters:
- `data` - Archived compressed payload.
- `inputSize` - Optional input-size override.
- `outputSize` - Optional output-size override.

Returns: Reconstructed network instance.

### deserializeCompressedArchiveAsync

```ts
deserializeCompressedArchiveAsync(
  data: CompressedSerializedNetworkArchive,
  inputSize: number | undefined,
  outputSize: number | undefined,
  options: CompressedArchiveDecodeOptions,
): Promise<default>
```

Rebuilds a network instance from the compressed archive wrapper with async runtime codecs.

Browser runtimes prefer the archive stream path so payload hydration can stay
off the synchronous main-thread lane, while Node falls back to the existing
archive owner when browser streams are unavailable.

Parameters:
- `data` - Archived compressed payload.
- `inputSize` - Optional input-size override.
- `outputSize` - Optional output-size override.

Returns: Reconstructed network instance.

### deserializeCompressedArchiveAsyncWithMetrics

```ts
deserializeCompressedArchiveAsyncWithMetrics(
  data: CompressedSerializedNetworkArchive,
  inputSize: number | undefined,
  outputSize: number | undefined,
  options: CompressedArchiveDecodeOptions,
): Promise<CompressedArchiveDecodeResult<default>>
```

Rebuild a network archive with async codecs and report size plus decode-time metrics for streaming or browser import paths.
This helper preserves deterministic reconstruction while exposing decode telemetry for responsiveness tuning.

Parameters:
- `data` - Archived compressed payload.
- `inputSize` - Optional input-size override.
- `outputSize` - Optional output-size override.
- `options` - Optional incremental decode callbacks.

Returns: Rebuilt network plus decode metrics.

### deserializeCompressedArchiveWithMetrics

```ts
deserializeCompressedArchiveWithMetrics(
  data: CompressedSerializedNetworkArchive,
  inputSize: number | undefined,
  outputSize: number | undefined,
): CompressedArchiveDecodeResult<default>
```

Rebuild a network archive and report size plus decode-time metrics for import performance and payload diagnostics.
The returned metrics quantify archive inflation and reconstruction overhead alongside the rebuilt runtime.

Parameters:
- `data` - Archived compressed payload.
- `inputSize` - Optional input-size override.
- `outputSize` - Optional output-size override.

Returns: Rebuilt network plus decode metrics.

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

### fromParameterVector

```ts
fromParameterVector(
  network: default,
  parameterVector: ParameterVector,
): void
```

Import one versioned parameter vector into a compatible live network runtime.

The target layout is rebuilt fresh and validated against the incoming vector
before any bias or weight mutation occurs. Version `1` applies only live
node biases and live forward-connection or self-connection weights, so
disabled connections still consume slots whenever they still exist in the
target runtime graph. Ordered descriptor compatibility stays innovation-first
and falls back to stable endpoint gene ids when an innovation id is absent,
which keeps same-runtime imports aligned with the export layout contract.
Import rejects non-neutral `node.response` and `connection.gain` explicitly
and also rejects version, entry-count, values-length, or descriptor mismatch
before mutation.

Parameters:
- `network` - Live target network instance.
- `parameterVector` - Versioned payload to apply.

Returns: Nothing. The target runtime mutates only after compatibility checks pass.

### isArchitectureDescriptorShapeValid

```ts
isArchitectureDescriptorShapeValid(
  architectureDescriptor: NetworkArchitectureDescriptor | undefined,
): boolean
```

Parameters:
- `architectureDescriptor` - Optional descriptor candidate.

Returns: True when minimal descriptor shape is valid.

### ParameterLayoutEntry

One ordered descriptor inside parameter-layout version `1`.

Bias entries use the stable node gene id in `nodeId`.
Weight entries prefer the live connection `innovation` when it exists and
otherwise fall back to the stable endpoint gene ids in `from` and `to`.
Duplicate innovation ids or duplicate fallback endpoint pairs are rejected
as ambiguous instead of inheriting incidental container order.

### ParameterLayoutV1

Deterministic parameter-layout descriptor owned by the network serialize boundary.

Version `1` keeps one stable fold order: all bias entries first, then all
weight entries. For a fixed topology with stable historical ids, this gives
ordered determinism on the same runtime.

### ParameterVector

Versioned parameter payload for same-runtime vector roundtrips.

Layout metadata and scalar values travel together so imports can reject
incompatible payloads before mutating a live network.

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

### serializeCompressed

```ts
serializeCompressed(): CompressedSerializedNetwork
```

Serializes a network instance into the compressed compact format.

This path keeps round-trip semantics identical to `serialize()` while
replacing the object-per-connection payload with one array-oriented block.

Parameters:
- `this` - Bound network instance.

Returns: Compressed compact payload.

### serializeCompressedArchive

```ts
serializeCompressedArchive(
  options: CompressedSerializedNetworkArchiveOptions | undefined,
): CompressedSerializedNetworkArchive
```

Serializes a network instance into the compressed archive wrapper.

This is the Node-side storage path for `serializeCompressed()`: it first
builds the exact compressed JSON payload, then applies gzip or zstd above
that payload without changing replay semantics.

Parameters:
- `this` - Bound network instance.
- `options` - Optional archive compression settings.

Returns: Archived compressed payload.

### serializeCompressedArchiveAsync

```ts
serializeCompressedArchiveAsync(
  options: CompressedSerializedNetworkArchiveOptions | undefined,
): Promise<CompressedSerializedNetworkArchive>
```

Serializes a network instance into the compressed archive wrapper with async runtime codecs.

Browser runtimes prefer the archive stream path so payload compression can stay
off the synchronous main-thread lane, while Node falls back to the existing
archive owner when browser streams are unavailable.

Parameters:
- `this` - Bound network instance.
- `options` - Optional archive compression settings.

Returns: Archived compressed payload.

### serializeCompressedArchiveAsyncWithMetrics

```ts
serializeCompressedArchiveAsyncWithMetrics(
  options: CompressedSerializedNetworkArchiveOptions | undefined,
): Promise<CompressedArchiveEncodeResult<CompressedSerializedNetworkArchive>>
```

Serialize a network archive with async codecs and report size plus encode-time metrics for responsive environments.
This variant keeps the same archive semantics while allowing non-blocking compression paths in browser runtimes.

Parameters:
- `this` - Bound network instance.
- `options` - Optional archive compression settings.

Returns: Archived payload plus encode metrics.

### serializeCompressedArchiveWithMetrics

```ts
serializeCompressedArchiveWithMetrics(
  options: CompressedSerializedNetworkArchiveOptions | undefined,
): CompressedArchiveEncodeResult<CompressedSerializedNetworkArchive>
```

Serialize a network archive and report size plus encode-time metrics for deterministic storage and transport audits.
The metrics payload helps compare archive codecs without changing the underlying compressed network contract.

Parameters:
- `this` - Bound network instance.
- `options` - Optional archive compression settings.

Returns: Archived payload plus encode metrics.

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

### toParameterVector

```ts
toParameterVector(
  network: default,
): ParameterVector
```

Export one versioned parameter vector from a live network runtime.

Version `1` keeps ordered `ParameterLayoutV1` metadata and the aligned bias
and weight scalars together so same-runtime imports can verify compatibility
before mutating another network. The payload includes every live node bias
and every live forward-connection or self-connection weight in layout order,
including disabled connections when they still exist in the runtime graph.
Weight slots still prefer innovation-backed descriptors and fall back to
stable endpoint gene ids when a live connection has no innovation id.
Export rejects non-neutral `node.response` and `connection.gain` explicitly
instead of flattening those deferred families into the weights-and-biases v1
payload.

Parameters:
- `network` - Live network instance to export.

Returns: Ordered layout metadata plus aligned scalar values for same-runtime deterministic roundtrips.

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

Append JSON entries for forward connections after validating that each endpoint still maps to canonical node indices.
Invalid or stale endpoints are skipped to prevent malformed rows from contaminating serialized payloads.

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

Rebuild runtime connections from verbose JSON entries while preserving best-effort import behavior for mixed-quality payloads.
Invalid rows are skipped with warnings so valid edge records still reconstruct successfully.

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

Rebuild runtime nodes from verbose JSON entries while restoring scalar fields and optional historical identifiers.
Unknown squash names are normalized through activation fallback logic to keep import paths resilient.

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

Resolve dropout with a numeric fallback when the serialized value is absent or undefined.
Normalizing this value at one boundary keeps verbose JSON payloads stable for downstream consumers.

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

Validate the verbose JSON payload root shape before node and connection rebuild routines process individual fields.
This early guard prevents non-object payloads from entering best-effort rebuild paths.

Parameters:
- `json` - Payload candidate.

Returns: Nothing.

### warnWhenJsonFormatVersionIsUnknown

```ts
warnWhenJsonFormatVersionIsUnknown(
  formatVersion: number,
): void
```

Warn when incoming verbose format version differs from the expected serializer schema version.
The warning-only policy allows controlled migration attempts without forcing immediate import failure.

Parameters:
- `formatVersion` - Incoming format version.

Returns: Nothing.

## architecture/network/serialize/network.serialize.genome.utils.ts

### COMPRESSED_GENOME_ARCHIVE_FORMAT

Stable archive wrapper format tag used for identifying strict-genome archives.

### COMPRESSED_GENOME_FORMAT

Stable payload format tag used for reliably identifying strict-genome archives.

### CompressedSerializedGenomeArchive

JSON-safe archive wrapper for one strict genome serialization checkpoint contract.

### CompressedSerializedGenomeArchiveOptions

Archive options for strict-genome compression that combine binary codec choices with capture-time genome field toggles.
These options let callers tune payload size and fidelity without introducing phenotype-only runtime state.

### deserializeCompressedGenomeArchive

```ts
deserializeCompressedGenomeArchive(
  compressedArchive: CompressedSerializedGenomeArchive,
  runtimeHints: GenomeMaterializationRuntimeHints,
): default
```

Materialize one runnable phenotype from a compressed strict-genome archive using validated contract data.
Runtime hints are applied only after strict-genome restoration so deterministic genotype state stays authoritative.

Parameters:
- `compressedArchive` - Base64-wrapped strict-genome archive payload.
- `runtimeHints` - Optional phenotype-only metadata to restore.

Returns: Rebuilt executable runtime phenotype.

### deserializeCompressedGenomeArchiveAsync

```ts
deserializeCompressedGenomeArchiveAsync(
  compressedArchive: CompressedSerializedGenomeArchive,
  runtimeHints: GenomeMaterializationRuntimeHints,
  options: CompressedArchiveDecodeOptions,
): Promise<default>
```

Materialize one runnable phenotype from a compressed strict-genome archive with async decode progress callbacks.
This path is suited for browser or streaming contexts where large payload inflation should stay responsive.

Parameters:
- `compressedArchive` - Base64-wrapped strict-genome archive payload.
- `runtimeHints` - Optional phenotype-only metadata to restore.
- `options` - Optional incremental decode callbacks.

Returns: Rebuilt executable runtime phenotype.

### deserializeCompressedGenomeArchiveAsyncWithMetrics

```ts
deserializeCompressedGenomeArchiveAsyncWithMetrics(
  compressedArchive: CompressedSerializedGenomeArchive,
  runtimeHints: GenomeMaterializationRuntimeHints,
  options: CompressedArchiveDecodeOptions,
): Promise<CompressedArchiveDecodeResult<default>>
```

Materialize one runnable phenotype from a compressed strict-genome archive with async decode metrics and progress support.
The returned telemetry helps compare streaming decode strategies while preserving the same strict reconstruction contract.

Parameters:
- `compressedArchive` - Base64-wrapped strict-genome archive payload.
- `runtimeHints` - Optional phenotype-only metadata to restore.
- `options` - Optional incremental decode callbacks.

Returns: Rebuilt network plus decode metrics.

### deserializeCompressedGenomeArchiveWithMetrics

```ts
deserializeCompressedGenomeArchiveWithMetrics(
  compressedArchive: CompressedSerializedGenomeArchive,
  runtimeHints: GenomeMaterializationRuntimeHints,
): CompressedArchiveDecodeResult<default>
```

Materialize one runnable phenotype from a compressed strict-genome archive and report decode metrics for runtime analysis.
This variant keeps decode telemetry alongside the rebuilt network for reproducibility and performance audits.

Parameters:
- `compressedArchive` - Base64-wrapped strict-genome archive payload.
- `runtimeHints` - Optional phenotype-only metadata to restore.

Returns: Rebuilt network plus decode metrics.

### parseCompressedGenomeArchive

```ts
parseCompressedGenomeArchive(
  compressedArchive: CompressedSerializedGenomeArchive,
): NeatGenome
```

Parse one archived strict genome contract back into validated JSON state before any runtime materialization begins.
This boundary enforces archive tags and schema validity so malformed payloads fail with clear diagnostics.

Parameters:
- `compressedArchive` - Base64-wrapped strict-genome archive payload.

Returns: Restored strict genome contract.

### parseCompressedGenomeArchiveAsync

```ts
parseCompressedGenomeArchiveAsync(
  compressedArchive: CompressedSerializedGenomeArchive,
  options: CompressedArchiveDecodeOptions,
): Promise<NeatGenome>
```

Parse one archived strict genome contract with async runtime codecs and progress callbacks.

Browser runtimes prefer the streaming decode path so large archives can emit
incremental progress while `DecompressionStream` inflates the wrapped UTF-8
JSON bytes. Node falls back to one completed snapshot.

Parameters:
- `compressedArchive` - Base64-wrapped strict-genome archive payload.
- `options` - Optional incremental decode callbacks.

Returns: Restored strict genome contract.

### serializeCompressedGenomeArchive

```ts
serializeCompressedGenomeArchive(
  options: CompressedSerializedGenomeArchiveOptions,
): CompressedSerializedGenomeArchive
```

Archive one runtime phenotype through the strict `NeatGenome` contract.

This keeps the payload structural and replay-safe: runtime activation traces,
slab allocations, and other phenotype-only state remain outside the archive.

Parameters:
- `options` - Optional archive codec and genome-capture settings.

Returns: Base64-wrapped compressed strict-genome archive.

Example:

```ts
const archive = serializeCompressedGenomeArchive.call(network);
const rebuiltNetwork = deserializeCompressedGenomeArchive(archive);
```

### serializeCompressedGenomeArchiveWithMetrics

```ts
serializeCompressedGenomeArchiveWithMetrics(
  options: CompressedSerializedGenomeArchiveOptions,
): CompressedArchiveEncodeResult<CompressedSerializedGenomeArchive>
```

Archive one runtime phenotype through the strict genome contract and report deterministic encode metrics for observability.
The metrics payload helps compare codec and capture-option tradeoffs without changing archive semantics.

Parameters:
- `options` - Optional archive codec and genome-capture settings.

Returns: Archived strict-genome payload plus encode metrics.

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

Collect node activation values in positional order so compact payload slot alignment stays deterministic across export and import paths.
The returned vector is index-stable relative to the current node list order.

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

Collect node activation keys in positional order by normalizing function references to stable string identifiers.
This keeps compact payloads portable across runtimes where function identity values cannot be serialized directly.

Parameters:
- `nodes` - Node list.

Returns: Squash-key list aligned to node indices.

### collectNodeStates

```ts
collectNodeStates(
  nodes: default[],
): number[]
```

Collect node state values in positional order so recurrent runtime state can be restored exactly after compact deserialization.
This preserves alignment with activation and squash-key arrays in the compact tuple.

Parameters:
- `nodes` - Node list.

Returns: State list aligned to node indices.

### collectSerializedConnections

```ts
collectSerializedConnections(
  networkInternals: SerializeNetworkInternals,
): SerializedConnection[]
```

Collect compact connection records from forward and self-connection groups while preserving historical identity metadata.
Callers should refresh node indices first so endpoint references remain canonical.

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

Rebuild runtime nodes from compact payload arrays by restoring type, scalar state, activation key, and optional gene identifiers.
Node type is inferred from positional input/output boundaries to keep payload shape compact.

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

Cast a public network instance to the internal serializer runtime shape so low-level deserialization helpers can access mutable fields without duplicating bridge-cast logic.

Parameters:
- `network` - Network instance.

Returns: Runtime internals.

### asNetworkInternalsWithDropout

```ts
asNetworkInternalsWithDropout(
  network: default,
): NetworkInternalsWithDropout
```

Cast a public network instance to serializer internals that include optional dropout metadata so verbose restore and export paths can read dropout fields consistently.

Parameters:
- `network` - Network instance.

Returns: Runtime internals with optional dropout.

### asNodeInternals

```ts
asNodeInternals(
  node: default,
): SerializeNodeInternals
```

Cast a node instance to its internal runtime representation so serializer helpers can read and write persisted node metadata through one shared bridge.

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

Create a fresh network instance for deserialize workflows so restoration code can hydrate graph state onto a clean runtime object.

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

Check whether a candidate index value is finite so endpoint validation can reject NaN and infinite references before bounds checks execute.

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

Check whether a candidate node index falls inside the valid array bounds so restore logic can reject malformed serialized endpoint references.

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

Clear mutable runtime collections before reconstruction so node, connection, self-connection, and gate arrays are reset to a predictable empty baseline.

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

Resolve effective input and output dimensions using optional explicit overrides so compact payload restore paths can apply caller-provided sizes deterministically.

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

Resolve one size value with override-first semantics so deserialization can prioritize explicit caller intent while preserving serialized fallbacks when overrides are absent.

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

## architecture/network/serialize/network.serialize.compression.utils.ts

Archive one compressed network payload with the best available async runtime codec.

Browser runtimes prefer `CompressionStream` with gzip so large payload work can
stay off the synchronous main-thread path. Node falls back to the existing zlib
owner when browser streams are unavailable.

### collectDecodedArchivePayloadBytes

```ts
collectDecodedArchivePayloadBytes(
  decodedStream: ReadableStream<Uint8Array<ArrayBufferLike>>,
  encodedByteLength: number,
  options: CompressedArchiveDecodeOptions,
): Promise<Uint8Array<ArrayBufferLike>>
```

Collect one decoded archive stream into bytes while surfacing chunk progress.

The implementation buffers one chunk ahead so the final emitted snapshot can
mark `done: true` on the last real decoded chunk rather than on a synthetic
zero-byte completion event.

Parameters:
- `decodedStream` - Stream of decoded UTF-8 payload chunks.
- `encodedByteLength` - Total compressed archive byte length.
- `options` - Optional incremental progress callbacks.

Returns: Concatenated decoded payload bytes.

### compressArchivePayloadBytes

```ts
compressArchivePayloadBytes(
  payloadBytes: Uint8Array<ArrayBufferLike>,
  compression: CompressedSerializedNetworkArchiveCompression,
): Uint8Array<ArrayBufferLike>
```

Compress UTF-8 payload bytes with one supported Node-side archive codec so binary wrappers stay compact while preserving exact JSON payload semantics.

Parameters:
- `payloadBytes` - UTF-8 payload bytes.
- `compression` - Archive compression codec.

Returns: Compressed payload bytes.

### compressArchivePayloadBytesAsync

```ts
compressArchivePayloadBytesAsync(
  payloadBytes: Uint8Array<ArrayBufferLike>,
  compression: CompressedSerializedNetworkArchiveCompression,
): Promise<Uint8Array<ArrayBufferLike>>
```

Compress UTF-8 payload bytes with the best available async archive codec.

Parameters:
- `payloadBytes` - UTF-8 payload bytes.
- `compression` - Archive compression codec.

Returns: Compressed payload bytes.

### COMPRESSED_NETWORK_ARCHIVE_ENCODING

Base64 string encoding applied to archived compressed bytes for safe JSON transport and storage of binary payloads.

### COMPRESSED_NETWORK_ARCHIVE_FORMAT

Stable format tag identifying the compressed archive wrapper payload version used by archive encode and decode utilities.

### COMPRESSED_NETWORK_FORMAT

Stable format tag identifying the compressed compact serialization payload version consumed by decompression utilities.

### COMPRESSED_WEIGHT_ENCODING

Stable format tag identifying the IEEE-754 float64 signed-int16 delta encoding used for compact weight storage.

### CompressedArchiveDecodeMetrics

Decode metrics extending the shared archive metrics with elapsed decode time in milliseconds.

### CompressedArchiveDecodeOptions

Optional progress and lifecycle callbacks supplied by the caller while an archive payload is being decoded.

### CompressedArchiveDecodeProgress

Progress snapshot emitted incrementally while an archive payload is being decoded so callers can surface decode progress.

### CompressedArchiveDecodeResult

Typed result wrapper pairing the decoded value with byte-size and timing metrics from the decode operation.

### CompressedArchiveEncodeMetrics

Encode metrics extending the shared archive metrics with elapsed encode time in milliseconds.

### CompressedArchiveEncodeResult

Typed result wrapper pairing the encoded archive payload with its byte-size and timing metrics.

### CompressedArchiveMetrics

Shared byte-size and compression-ratio metrics recorded for a single archive encode or decode operation.

### compressMatchingRuns

```ts
compressMatchingRuns(
  values: Value[],
  shouldCompressValue: (value: Value) => boolean,
): CompressedSerializedIndexRun[] | undefined
```

Collapse contiguous matching entries into run-length metadata.

Parameters:
- `values` - Ordered values to scan.
- `shouldCompressValue` - Predicate deciding whether one value belongs to a run.

Returns: Run metadata or `undefined` when no matching span exists.

### compressOptionalNumericSeries

```ts
compressOptionalNumericSeries(
  values: (number | null | undefined)[],
): (number | null)[] | undefined
```

Collapse optional numeric fields to `undefined` when no entries are present.

Parameters:
- `values` - Optional numeric series.

Returns: Normalized nullable series or `undefined` when empty of numeric content.

### compressSerializedConnections

```ts
compressSerializedConnections(
  serializedConnections: NetworkJSONConnection[],
): CompressedSerializedConnectionBlock
```

Exported contract for compressSerializedConnections.

### concatenateArchiveByteChunks

```ts
concatenateArchiveByteChunks(
  byteChunks: Uint8Array<ArrayBufferLike>[],
): Uint8Array<ArrayBufferLike>
```

Concatenate one ordered set of archive byte chunks into a single buffer.

Parameters:
- `byteChunks` - Ordered archive byte chunks.

Returns: One contiguous byte buffer.

### countRunEntries

```ts
countRunEntries(
  runs: CompressedSerializedIndexRun[] | undefined,
): number
```

Count how many connection rows are covered by one run list.

Parameters:
- `runs` - Optional run metadata.

Returns: Total covered row count.

### createCompressedArchiveDecodeMetrics

```ts
createCompressedArchiveDecodeMetrics(
  uncompressedByteLength: number,
  compressedByteLength: number,
  decodeTimeMs: number,
): CompressedArchiveDecodeMetrics
```

Create one metrics snapshot for an archive decode operation so callers can inspect decompression efficiency and elapsed decoding cost consistently.

Parameters:
- `uncompressedByteLength` - UTF-8 byte length after archive inflation.
- `compressedByteLength` - Binary byte length before archive inflation.
- `decodeTimeMs` - Elapsed decode time in milliseconds.

Returns: Archive decode metrics.

### createCompressedArchiveEncodeMetrics

```ts
createCompressedArchiveEncodeMetrics(
  uncompressedByteLength: number,
  compressedByteLength: number,
  encodeTimeMs: number,
): CompressedArchiveEncodeMetrics
```

Create one metrics snapshot for an archive encode operation so callers can log compression efficiency and elapsed encoding cost consistently.

Parameters:
- `uncompressedByteLength` - UTF-8 byte length before archive compression.
- `compressedByteLength` - Binary byte length after archive compression.
- `encodeTimeMs` - Elapsed encode time in milliseconds.

Returns: Archive encode metrics.

### createCompressedNetworkArchive

```ts
createCompressedNetworkArchive(
  compressedPayload: CompressedSerializedNetwork,
  options: CompressedSerializedNetworkArchiveOptions,
): CompressedSerializedNetworkArchive
```

Archive one compressed network payload with a Node-side binary codec.

This is intentionally additive: the wrapped payload stays the exact JSON form
returned by `serializeCompressed`, then gzip or zstd is applied above it.

Parameters:
- `compressedPayload` - Existing compressed network payload.
- `options` - Optional archive compression settings.

Returns: Base64-wrapped compressed archive payload.

### createCompressedNetworkArchiveAsync

```ts
createCompressedNetworkArchiveAsync(
  compressedPayload: CompressedSerializedNetwork,
  options: CompressedSerializedNetworkArchiveOptions,
): Promise<CompressedSerializedNetworkArchive>
```

Exported contract for createCompressedNetworkArchiveAsync.

### decodeArchivePayloadBase64

```ts
decodeArchivePayloadBase64(
  payload: string,
): Uint8Array<ArrayBufferLike>
```

Decode archive payload bytes from base64 without assuming a specific runtime so compressed archives can hydrate reliably in browser and Node.

Parameters:
- `payload` - Base64-encoded payload text.

Returns: Binary archive payload bytes.

### decodeExactConnectionWeights

```ts
decodeExactConnectionWeights(
  weightWords: CompressedSerializedConnectionWeights,
  connectionCount: number,
): number[]
```

Decode one exact signed-16-bit delta stream back to float64 weights.

Parameters:
- `weightWords` - Encoded weight-word payload.
- `connectionCount` - Expected number of weights.

Returns: Exact reconstructed weights.

### decodeSignedInt16WordsToFloat64

```ts
decodeSignedInt16WordsToFloat64(
  words: number[],
): number
```

Decode four signed 16-bit words back into one float64 number.

Parameters:
- `words` - Signed 16-bit little-endian words.

Returns: Decoded numeric value.

### decompressArchivePayloadBytes

```ts
decompressArchivePayloadBytes(
  payloadBytes: Uint8Array<ArrayBufferLike>,
  compression: CompressedSerializedNetworkArchiveCompression,
): Uint8Array<ArrayBufferLike>
```

Decompress archive payload bytes with one supported Node-side codec so compressed archive wrappers recover deterministic UTF-8 serialization payload bytes.

Parameters:
- `payloadBytes` - Compressed archive payload bytes.
- `compression` - Archive compression codec.

Returns: Inflated UTF-8 payload bytes.

### decompressArchivePayloadBytesAsync

```ts
decompressArchivePayloadBytesAsync(
  payloadBytes: Uint8Array<ArrayBufferLike>,
  compression: CompressedSerializedNetworkArchiveCompression,
  options: CompressedArchiveDecodeOptions,
): Promise<Uint8Array<ArrayBufferLike>>
```

Decompress archive payload bytes with the best available async archive codec so browser and Node runtimes can share one non-blocking restore flow.

Parameters:
- `payloadBytes` - Compressed archive payload bytes.
- `compression` - Archive compression codec.

Returns: Inflated UTF-8 payload bytes.

### decompressArchivePayloadBytesWithBrowserStream

```ts
decompressArchivePayloadBytesWithBrowserStream(
  payloadBytes: Uint8Array<ArrayBufferLike>,
  streamConstructor: BrowserDecompressionStreamConstructor,
  options: CompressedArchiveDecodeOptions,
): Promise<Uint8Array<ArrayBufferLike>>
```

Decompress archive payload bytes through one browser stream while reporting progress.

Parameters:
- `payloadBytes` - Compressed archive payload bytes.
- `streamConstructor` - Browser decompression stream constructor.
- `options` - Optional incremental progress callbacks.

Returns: Inflated UTF-8 payload bytes.

### decompressSerializedConnections

```ts
decompressSerializedConnections(
  compressedConnections: CompressedSerializedConnectionBlock,
): NetworkJSONConnection[]
```

Decompress one array-oriented connection block back into compact rows so archived connection payloads regain legacy-friendly per-edge field records.

Parameters:
- `compressedConnections` - Compressed connection block.

Returns: Reconstructed serialized connection rows.

### emitArchiveDecodeProgress

```ts
emitArchiveDecodeProgress(
  options: CompressedArchiveDecodeOptions,
  progress: CompressedArchiveDecodeProgress,
): Promise<void>
```

Notify callers that one decoded archive chunk has been observed.

Parameters:
- `options` - Optional incremental progress callbacks.
- `progress` - Progress payload for the decoded chunk.

Returns: Nothing.

### encodeArchivePayloadBase64

```ts
encodeArchivePayloadBase64(
  payloadBytes: Uint8Array<ArrayBufferLike>,
): string
```

Encode archive payload bytes to base64 without assuming a specific runtime so archive wrappers remain portable across browser and Node environments.

Parameters:
- `payloadBytes` - Binary archive payload bytes.

Returns: Base64-encoded payload text.

### encodeExactConnectionWeights

```ts
encodeExactConnectionWeights(
  weights: number[],
): CompressedSerializedConnectionWeights
```

Encode float64 weights into one exact signed-16-bit delta stream.

Parameters:
- `weights` - Connection weights.

Returns: Exact weight-word delta payload.

### encodeFloat64ToSignedInt16Words

```ts
encodeFloat64ToSignedInt16Words(
  value: number,
): number[]
```

Encode one float64 number into four signed 16-bit words.

Parameters:
- `value` - Numeric value to encode.

Returns: Signed 16-bit little-endian words.

### estimateSerializedByteLength

```ts
estimateSerializedByteLength(
  payload: unknown,
): number
```

Estimate the UTF-8 byte length of one serialization payload so archive ratio metrics can compare compressed and uncompressed storage cost.

Parameters:
- `payload` - Payload to measure.

Returns: UTF-8 byte length of the JSON string form.

### expandDecodedNonZeroWeights

```ts
expandDecodedNonZeroWeights(
  decodedNonZeroWeights: number[],
  zeroWeightMask: boolean[] | undefined,
  connectionCount: number,
): number[]
```

Expand the decoded non-zero weight sequence back across zero-weight spans.

Parameters:
- `decodedNonZeroWeights` - Exact non-zero weights in encoded order.
- `zeroWeightMask` - Optional zero-run membership mask.
- `connectionCount` - Total number of serialized connections.

Returns: Connection-aligned exact weight values.

### expandRunsToMask

```ts
expandRunsToMask(
  runs: CompressedSerializedIndexRun[] | undefined,
  entryCount: number,
): boolean[] | undefined
```

Expand run-length metadata into one boolean mask aligned to connection order.

Parameters:
- `runs` - Optional run metadata.
- `entryCount` - Total number of connection rows.

Returns: Boolean run-membership mask or `undefined` when no runs exist.

### hasNodeCompressionRuntime

```ts
hasNodeCompressionRuntime(): boolean
```

Determine whether the current runtime can resolve the Node compression owner.

Returns: Whether Node zlib support is available.

### isPositiveZeroWeightWords

```ts
isPositiveZeroWeightWords(
  encodedWeightWords: number[],
): boolean
```

Determine whether one encoded float64 word sequence represents exact positive zero.

Parameters:
- `encodedWeightWords` - Signed 16-bit float64 words.

Returns: Whether the encoded value is exact positive zero.

### normalizeSignedInt16

```ts
normalizeSignedInt16(
  value: number,
): number
```

Normalize one integer through wrapped signed-16-bit arithmetic.

Parameters:
- `value` - Integer value to normalize.

Returns: Wrapped signed 16-bit integer.

### parseCompressedNetworkArchive

```ts
parseCompressedNetworkArchive(
  compressedArchive: CompressedSerializedNetworkArchive,
): CompressedSerializedNetwork
```

Rebuild one compressed network payload from its Node-side archive wrapper so persisted archives can be restored into deterministic compact serialization objects.

Parameters:
- `compressedArchive` - Base64-wrapped compressed archive payload.

Returns: Restored compressed network payload.

### parseCompressedNetworkArchiveAsync

```ts
parseCompressedNetworkArchiveAsync(
  compressedArchive: CompressedSerializedNetworkArchive,
  options: CompressedArchiveDecodeOptions,
): Promise<CompressedSerializedNetwork>
```

Exported contract for parseCompressedNetworkArchiveAsync.

### resolveBrowserCompressionStreamConstructor

```ts
resolveBrowserCompressionStreamConstructor(): BrowserCompressionStreamConstructor | undefined
```

Resolve the browser compression-stream constructor when one is available.

Returns: Browser compression-stream constructor.

### resolveBrowserDecompressionStreamConstructor

```ts
resolveBrowserDecompressionStreamConstructor(): BrowserDecompressionStreamConstructor | undefined
```

Resolve the browser decompression-stream constructor when one is available.

Returns: Browser decompression-stream constructor.

### resolveNodeCompressionModule

```ts
resolveNodeCompressionModule(): NodeCompressionModule
```

Resolve the Node builtin zlib module without introducing a static browser import.

Returns: Node compression module.

### resolveNodeCompressionModuleOrUndefined

```ts
resolveNodeCompressionModuleOrUndefined(): NodeCompressionModule | undefined
```

Resolve the Node builtin zlib module when the runtime exposes it.

Returns: Node compression module when available.

### transformArchivePayloadBytesWithStream

```ts
transformArchivePayloadBytesWithStream(
  payloadBytes: Uint8Array<ArrayBufferLike>,
  streamConstructor: BrowserDecompressionStreamConstructor | BrowserCompressionStreamConstructor,
): Promise<Uint8Array<ArrayBufferLike>>
```

Transform archive payload bytes through one browser compression stream.

Parameters:
- `payloadBytes` - Source payload bytes.
- `streamConstructor` - Browser stream constructor.

Returns: Transformed payload bytes.

### validateCompressedOptionalVectorLength

```ts
validateCompressedOptionalVectorLength(
  fieldName: string,
  values: unknown[] | undefined,
  expectedLength: number,
): void
```

Validate one optional vector width when the vector is present.

Parameters:
- `fieldName` - Logical field name.
- `values` - Optional vector.
- `expectedLength` - Required vector length.

Returns: Nothing.

### validateCompressedRuns

```ts
validateCompressedRuns(
  fieldName: string,
  runs: CompressedSerializedIndexRun[] | undefined,
  entryCount: number,
): void
```

Validate run metadata for bounds, ordering, and overlap.

Parameters:
- `fieldName` - Logical field name.
- `runs` - Optional run metadata.
- `entryCount` - Total number of connection rows.

Returns: Nothing.

### validateCompressedVectorLength

```ts
validateCompressedVectorLength(
  fieldName: string,
  values: unknown[],
  expectedLength: number,
): void
```

Validate one required vector width inside the compressed connection block.

Parameters:
- `fieldName` - Logical field name.
- `values` - Vector to validate.
- `expectedLength` - Required vector length.

Returns: Nothing.
