# architecture/network/construct

Raised when construct-from-parts cannot resolve an explicit input or output node id.

## architecture/network/construct/network.construct.utils.ts

### constructNetwork

```ts
constructNetwork(
  parts: readonly ConstructPart[],
  options: ConstructOptions,
): ConstructResult
```

Construct one runnable `Network` from mixed primitive parts.

The builder flattens `Node`, `Group`, and `Layer` inputs into one canonical
runtime graph, validates that every referenced edge remains inside the
provided part boundary, preserves explicit input/output ordering, and then
compiles the existing network scheduling cache in either acyclic or recurrent
mode.

Parameters:

- `this` - Network constructor used to instantiate the runtime graph.
- `parts` - Mixed architecture parts to flatten.
- `options` - Optional construct-time validation, ordering, and runtime flags.

Returns: Materialized runtime plus lightweight diagnostics and a detached graph snapshot.

Example:

```ts
const leftSensor = new Node('input');
const rightSensor = new Node('input');
const hidden = new Group(2);
const readout = Layer.dense(1, 'output');

leftSensor.describe({ label: 'leftSensor' });
rightSensor.describe({ label: 'rightSensor' });
readout.nodes[0].describe({ label: 'readout' });

leftSensor.connect(hidden);
rightSensor.connect(hidden);
hidden.connect(readout);

const { network } = Network.construct(
  [hidden, rightSensor, readout, leftSensor],
  {
    inputNodes: ['leftSensor', 'rightSensor'],
    outputNodes: ['readout'],
  },
);
```

## architecture/network/construct/network.construct.utils.types.ts

### ConstructDiagnostics

Lightweight diagnostics returned alongside one constructed network.

`activationOrder` resolves into network node indices so downstream tooling can
inspect one canonical traversal order without re-reading private scheduling
fields.

### ConstructGraphConnectionSummary

One detached, JSON-serializable edge row in the construct graph snapshot.

Carries stable innovation id, source/target identity, current weight and
enabled state, gater identity when present, and self-loop classification.

### ConstructGraphNodeSummary

One detached, JSON-serializable node row in the construct graph snapshot.

Carries stable gene identity, semantic role, and public I/O ordering metadata
so developer tooling can inspect the materialized graph without accessing
mutable `Network` internals directly.

### ConstructGraphSnapshot

Detached construct graph snapshot for developer tooling.

The snapshot is intentionally JSON-friendly so callers can log it directly,
persist it to diagnostics artifacts, or feed it into visualization tooling
without re-reading mutable `Network` internals.

### ConstructNodeId

Stable node identifiers accepted by the construct-from-parts API.

Numeric ids resolve against `node.geneId`. String ids resolve against
`node.label` when one was attached through `describe({ label })`.

### ConstructOptions

Public options for `Network.construct(...)`.

This surface deliberately reuses the existing `Network` runtime instead of
introducing a second execution engine. Callers provide graph parts, choose
acyclic versus recurrent compilation, and optionally pin the public input and
output vector ordering explicitly. Public input nodes are always validated as
pure sources, and output nodes default to pure sinks unless validation opts
into outward feedback or gating explicitly.

### ConstructPart

Mixed primitive inputs accepted by `Network.construct(...)`.

The builder flattens nodes out of each composite part while preserving
deterministic ordering and validating that referenced edges stay inside the
provided part set.

### ConstructResult

Return payload emitted by `Network.construct(...)` bundling the network, diagnostics, and graph.

### ConstructValidationOptions

Extra structural validation switches for the construct-from-parts network materialization pipeline.

## architecture/network/construct/network.construct.summary.utils.ts

### formatConstructSummary

```ts
formatConstructSummary(
  constructResult: ConstructResult,
): string
```

Build a compact human-readable summary for one construct-from-parts result.

The summary is layered on top of the detached `ConstructResult.graph`
snapshot so tooling can log or display one stable explanation of the built
graph without reading mutable `Network` internals.

Parameters:

- `constructResult` - Construct result returned by `Network.construct(...)`.

Returns: Multi-line summary string suitable for logs, diagnostics panels, or snapshots.

Example:

```ts
const construction = Network.construct([sensor, hidden, readout]);
const summary = formatConstructSummary(construction);
```

## architecture/network/construct/network.construct.errors.ts

### NetworkConstructAmbiguousNodeIdError

Raised when a single explicit string node id matches more than one labeled node in the provided parts list during network construction.

### NetworkConstructCycleModeError

Raised when a cyclic graph is compiled while acyclic mode is required.

### NetworkConstructDuplicateEdgeError

Raised when the same source-to-target node pair appears more than once while duplicate edges are explicitly disallowed by the construction policy.

### NetworkConstructInputNodeIncomingEdgeError

Raised when an input-role node is wired as a connection target while the active construction policy forbids incoming edges on input nodes.

### NetworkConstructIsolatedHiddenNodeError

Raised when the construction pass finds hidden nodes with no connections while the active construction policy forbids isolated hidden neurons in the graph.

### NetworkConstructMissingReferencedNodeError

Raised when collected connections reference nodes that were not included in the provided parts.

### NetworkConstructNodeIdResolutionError

Raised when construct-from-parts cannot resolve an explicit input or output node id.

### NetworkConstructNoInputNodesError

Raised when the network construction pass cannot find any nodes classified with an input role in the provided node list.

### NetworkConstructNoOutputNodesError

Raised when the network construction pass cannot find any nodes classified with an output role in the provided node list.

### NetworkConstructOutputNodeGatedConnectionError

Raised when a public output node gates one or more connections while sink-only output validation is enabled.

### NetworkConstructOutputNodeOutgoingEdgeError

Raised when a public output node emits one or more outgoing edges while sink-only output validation is enabled.

### NetworkConstructSelfEdgeError

Raised when a connection loops from a node back to itself while self-edges are explicitly forbidden by the active construction policy.
