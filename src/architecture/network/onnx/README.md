# architecture/network/onnx

NeatapticTS ONNX-like serialization for networks.

This module provides the two public entry points:
- `exportToONNX()` turns a runtime `Network` into a plain JSON object (`OnnxModel`).
- `importFromONNX()` reconstructs a `Network` from that JSON object.

What this format is (and is not):
- It is **JSON-first** and intentionally resembles ONNX’s model/graph concepts.
- It is **not** a full ONNX protobuf implementation and is not guaranteed to run on
  general ONNX runtimes.
- The compatibility promise is primarily **within this repo**: models produced by
  `exportToONNX()` should be accepted by `importFromONNX()` (same version family).

Trust boundary:
- Treat imported models as **untrusted input**. The importer validates structure, but
  you should still apply the same care you would for any JSON payload.

Example (export → persist → import):

```ts
import { exportToONNX, importFromONNX } from './network.onnx';

const model = exportToONNX(network, { includeMetadata: true });
const jsonText = JSON.stringify(model);

const modelRoundTrip = JSON.parse(jsonText);
const restored = importFromONNX(modelRoundTrip);
```

## architecture/network/onnx/network.onnx.ts

### exportToONNX

```ts
exportToONNX(
  network: default,
  options: OnnxExportOptions,
): OnnxModel
```

Export a NeatapticTS network to an ONNX-like **JSON object** (`OnnxModel`).

What you get:
- A plain object that you can persist with `JSON.stringify()`.
- A minimal ONNX-ish graph (`model.graph`) plus optional metadata (`model.metadata_props`).

When to use this:
- You want a portable snapshot that can be inspected/diffed as JSON.
- You want to reconstruct the network later via `importFromONNX()`.

Tradeoffs:
- The output is ONNX-like, but **not** intended to be universally compatible with all ONNX
  runtimes.
- Some advanced features (partial connectivity, mixed activations, recurrent heuristics)
  may produce graphs that are primarily meant for this library’s importer.

High-level algorithm:
 1) Normalize/rebuild local connection state for deterministic traversal.
 2) Infer an ordered layer view and validate export constraints.
 3) Materialize graph nodes/tensors and (optionally) attach metadata.

Example (export → JSON text):

```ts
const model = exportToONNX(network, { includeMetadata: true });
const jsonText = JSON.stringify(model);
```

Parameters:
- `network` - Source network instance to serialize.
- `options` - Export controls (validation strictness and metadata behavior).

Returns: ONNX-like model object suitable for persistence or re-import.

### importFromONNX

```ts
importFromONNX(
  onnx: OnnxModel,
): default
```

Reconstruct a NeatapticTS network from an exported `OnnxModel`.

Expected input:
- A model produced by `exportToONNX()` (same repo/version family).

Trust boundary:
- Do not import untrusted blobs. A malformed model can be extremely large or internally
  inconsistent and may cause errors or high memory usage.

High-level behavior:
 1) Build a perceptron-shaped scaffold from the payload layer sizes.
 2) Assign weights/biases and activation functions.
 3) Re-apply recurrent and pooling metadata when present.

Example (JSON text → restore):

```ts
const model = JSON.parse(jsonText) as OnnxModel;
const restored = importFromONNX(model);
const output = restored.activate([0.1, 0.9]);
```

Parameters:
- `onnx` - ONNX-like model to reconstruct.

Returns: Reconstructed network ready for inference/evolution workflows.

### Conv2DMapping

Mapping declaration for treating a fully-connected layer as a 2D convolution during export.

This does **not** magically turn an MLP into a convolutional network at runtime.
It annotates a particular export-layer index with a conv interpretation so that:
- The exported graph uses conv-shaped tensors/operators, and
- Import can re-attach pooling/flatten metadata appropriately.

Pitfall: mappings must match the actual layer sizes. If `inHeight * inWidth * inChannels`
does not correspond to the prior layer width (and similarly for outputs), export or import
may reject the model.

### OnnxExportOptions

Options controlling ONNX-like export.

These options trade off strictness, portability, and fidelity:

- **Strict (default-ish)** export tries to keep the graph easy to interpret:
  layered topology, homogeneous activations per layer, and fully-connected layers.

- **Relaxed** export (`allowPartialConnectivity` / `allowMixedActivations`) can represent
  more networks, but it may generate graphs that are primarily meant for NeatapticTS’s
  importer (and may be less friendly to external ONNX tooling).

- **Recurrent export** (`allowRecurrent`) is intentionally conservative and currently
  focuses on a constrained single-step representation and optional fused heuristics.

Key fields (high-level):
- `includeMetadata`: includes `metadata_props` with architecture hints.
- `opset`: numeric opset version stored in the exported model metadata (default is
  resolved by the exporter; commonly 18 in this codebase).
- `legacyNodeOrdering`: keeps older node ordering for backward compatibility.
- `conv2dMappings` / `pool2dMappings`: encode conv/pool semantics for fully-connected
  layers via explicit mapping declarations.

### OnnxModel

ONNX-like model container (JSON-serializable).

This is the main “wire format” object in this folder. Persist it as JSON text:

```ts
const jsonText = JSON.stringify(model);
const restoredModel = JSON.parse(jsonText) as OnnxModel;
```

Notes:
- `metadata_props` contains NeatapticTS-specific keys (layer sizes, recurrent flags,
  conv/pool mappings, etc.). This is where most round-trip hints live.
- Initializers currently store floating-point weights in `float_data`.

Security/trust boundary:
- Treat this as untrusted input if it comes from outside your process.

### Pool2DMapping

Mapping describing a pooling operation inserted after a given export-layer index.

This is represented as metadata and optional graph nodes during export.
Import uses it to attach pooling-related runtime metadata back onto the reconstructed
network (when supported).

## architecture/network/onnx/network.onnx.utils.types.ts

Types for NeatapticTS’s ONNX-like JSON export/import.

The exporter produces an `OnnxModel` (a JSON-serializable object) and the importer
reconstructs a `Network` from that object.

Practical notes:
- These types intentionally resemble ONNX’s `ModelProto`/`GraphProto` concepts, but they
  are *not* a full ONNX protobuf implementation.
- `opset` and `ir_version` are recorded as metadata for inspection/compat bookkeeping.
  They are not a promise of universal ONNX-runtime compatibility.

Stability & compatibility expectations:
- This repo’s importer is only guaranteed to accept models produced by this repo’s
  exporter.
- The schema is JSON-first and may evolve; prefer re-exporting/importing through the
  library rather than hand-editing blobs.

### OnnxRuntimePerceptronFactory

```ts
OnnxRuntimePerceptronFactory(
  sizes: number[],
): default
```

Runtime perceptron factory signature used by ONNX import orchestration.

This factory is injected so the ONNX import path can rebuild an MLP without taking a
hard dependency on a specific constructor shape.

### OnnxRuntimeLayerFactory

```ts
OnnxRuntimeLayerFactory(
  size: number,
): default
```

Runtime layer-constructor signature used for recurrent layer reconstruction.

ONNX import can optionally reconstruct higher-level recurrent layers (like LSTM/GRU)
from exported metadata. This factory provides the concrete layer implementation.

### OnnxRuntimeLayerModule

Runtime layer module shape consumed by ONNX import orchestration.

This is the minimal set of recurrent factories needed by the importer.

### OnnxRuntimeFactories

Runtime factories consumed during ONNX import network reconstruction.

These factories let the importer reconstruct runtime objects (network + layers)
while keeping the ONNX parser itself mostly pure.

### OnnxPerceptronSizeValidationContext

Validation context for perceptron size-list checks during ONNX import.

### OnnxPerceptronBuildContext

Build context for mapping ONNX layer sizes into a Neataptic MLP factory call.

### NodeInternals

Runtime interface for accessing node internal properties.

This is intentionally "internal": it exposes mutable fields that the ONNX exporter/importer
needs (connections, bias, squash). Regular library users should generally interact with
the public `Node` API instead.

### NodeInternalsWithExportIndex

Runtime node internals augmented with optional export index metadata.

### ActivationFunction

```ts
ActivationFunction(
  x: number,
  derivate: boolean | undefined,
): number
```

Runtime activation function signature used by ONNX activation import/export paths.

Neataptic-style activations support a dual-purpose call pattern:
- `derivate === false | undefined`: return activation output $f(x)$
- `derivate === true`: return derivative $f'(x)$

This matches historical Neataptic semantics and keeps ONNX import/export compatible.

Example:

```ts
const y = activation(x);
const dy = activation(x, true);
```

### LayerOrderingNodeGroups

Node partitions used by ONNX layered-ordering inference traversal.

### LayerOrderingResolutionContext

Mutable traversal state while resolving hidden-layer ordering.

### LayerValidationTraversalContext

Layer-wise validation context for activation and connectivity checks.

### LayerActivationValidationContext

Activation-homogeneity decision context for one current layer.

### LayerConnectivityValidationContext

Connectivity decision context for one source-target node pair.

### OnnxActivationOperation

Supported ONNX activation operators recognized during activation import.

### OnnxActivationLayerOperations

Layer-indexed activation operator lookup extracted from ONNX graph nodes.

### OnnxActivationParseResult

Parsed ONNX activation-node naming payload.

### OnnxActivationAssignmentContext

Shared activation-assignment context for hidden and output traversal.

### HiddenLayerActivationTraversalContext

Hidden-layer traversal context for assigning imported activation functions.

### OutputLayerActivationContext

Output-layer activation assignment context.

### OnnxActivationOperationResolutionContext

Activation operation resolution context for one neuron or layer default.

### ExportNodeIndexAssignmentContext

Context for assigning a stable export index to one node.

### LstmPatternStub

Heuristic LSTM pattern stub for metadata output.

### LstmLayerTraversalContext

Traversal context for one hidden layer during LSTM stub collection.

### LstmCandidateContext

Candidate context for validating one LSTM-like hidden layer pattern.

### ConvInferenceTraversalContext

Traversal context for one hidden layer during Conv inference.

### ConvInferenceEvaluationContext

Width and shape evaluation context used by Conv inference helpers.

### ConvInferenceKernelEvaluationContext

Kernel candidate context for one Conv inference evaluation pass.

### ConvInferenceResult

Collected inferred Conv metadata payload.

### OnnxExportOptions

Options controlling ONNX-like export.

These options trade off strictness, portability, and fidelity:

- **Strict (default-ish)** export tries to keep the graph easy to interpret:
  layered topology, homogeneous activations per layer, and fully-connected layers.

- **Relaxed** export (`allowPartialConnectivity` / `allowMixedActivations`) can represent
  more networks, but it may generate graphs that are primarily meant for NeatapticTS’s
  importer (and may be less friendly to external ONNX tooling).

- **Recurrent export** (`allowRecurrent`) is intentionally conservative and currently
  focuses on a constrained single-step representation and optional fused heuristics.

Key fields (high-level):
- `includeMetadata`: includes `metadata_props` with architecture hints.
- `opset`: numeric opset version stored in the exported model metadata (default is
  resolved by the exporter; commonly 18 in this codebase).
- `legacyNodeOrdering`: keeps older node ordering for backward compatibility.
- `conv2dMappings` / `pool2dMappings`: encode conv/pool semantics for fully-connected
  layers via explicit mapping declarations.

### OnnxBuildResolvedOptions

Resolved options used by ONNX model build orchestration.

### OnnxGraphDimensionBuildContext

Context for constructing input/output ONNX graph dimensions.

### OnnxGraphDimensions

Output dimensions used by ONNX graph input/output value info payloads.

### OnnxBaseModelBuildContext

Context for constructing a base ONNX model shell.

### OnnxModelMetadataContext

Context for applying optional ONNX model metadata.

### OnnxRecurrentCollectionContext

Context for collecting recurrent layer indices during model build.

### OnnxRecurrentLayerTraversalContext

Traversal context for one hidden layer during recurrent-input collection.

### OnnxRecurrentInputValueInfoContext

Context for constructing one recurrent previous-state graph input payload.

### OnnxRecurrentLayerProcessingContext

Execution context for processing one hidden recurrent layer.

### OnnxLayerEmissionResult

Result of emitting non-input export layers.

### OnnxLayerEmissionContext

Context for emitting non-input layers during model build.

### LayerBuildContext

Layer build context used while emitting one ONNX graph layer segment.

### LayerTraversalContext

Layer traversal context with adjacent layers and output classification.

### LayerActivationContext

Activation analysis context for one layer.

### LayerRecurrentDecisionContext

Context used to decide recurrent emission branch usage.

### OnnxPostProcessingContext

Context for post-processing and export metadata finalization.

### RecurrentHeuristicEmissionContext

Context for heuristic recurrent operator emission traversal.

### HiddenLayerHeuristicContext

Context for one hidden layer during heuristic recurrent emission.

### LstmEmissionContext

Context for heuristic LSTM emission when a layer matches expected shape.

### GruEmissionContext

Context for heuristic GRU emission when a layer matches expected shape.

### RecurrentGateRowCollectionContext

Context for collecting one recurrent gate row (one neuron).

### RecurrentGateRow

One recurrent gate row payload before flatten fold.

### RecurrentGateParameterCollectionResult

Flattened recurrent gate parameter vectors for one fused operator.

### RecurrentGateBlockCollectionContext

Context for collecting one gate parameter block.

### FusedRecurrentInitializerNames

Context for ONNX fused recurrent initializer names.

### FusedRecurrentGraphNames

Context for ONNX fused recurrent node payload names.

### FusedRecurrentEmissionExecutionContext

Shared execution context for emitting one fused recurrent layer payload.

### RecurrentLayerEmissionParams

Parameters for single-step recurrent layer emission.

### RecurrentLayerEmissionContext

Derived execution context for single-step recurrent layer emission.

### RecurrentInitializerNames

Initializer tensor names for one single-step recurrent layer.

### RecurrentInitializerValues

Collected initializer vectors for one single-step recurrent layer.

### RecurrentInitializerEmissionContext

Context for pushing recurrent initializers into ONNX graph state.

### RecurrentGemmEmissionContext

Context for emitting one Gemm node for recurrent single-step export.

### RecurrentGraphNames

Derived graph names for one recurrent single-step layer payload.

### RecurrentActivationEmissionContext

Context for selecting and emitting recurrent activation node payload.

### ConvSharingValidationResult

Result of Conv sharing validation across declared mappings.

### ConvSharingValidationContext

Context for validating Conv sharing across all declared mappings.

### ConvLayerPairContext

Context for one resolved Conv mapping layer pair.

### ConvOutputCoordinate

Coordinate for one Conv output neuron position.

### ConvRepresentativeKernelContext

Context for representative Conv kernel collection per output channel.

### ConvKernelConsistencyContext

Context for kernel-coordinate consistency checks at one output position.

### WeightToleranceComparisonContext

Context for comparing two scalar weights with numeric tolerance.

### OnnxConvEmissionParams

Parameters accepted by Conv layer emission.

### OnnxConvEmissionContext

Context used after resolving Conv mapping for one layer.

### OnnxConvParameters

Flattened Conv parameters for ONNX initializers.

### OnnxConvTensorNames

Tensor names generated for Conv parameters.

### OnnxConvKernelCoordinate

Coordinate for one Conv kernel weight lookup.

### Conv2DMapping

Mapping declaration for treating a fully-connected layer as a 2D convolution during export.

This does **not** magically turn an MLP into a convolutional network at runtime.
It annotates a particular export-layer index with a conv interpretation so that:
- The exported graph uses conv-shaped tensors/operators, and
- Import can re-attach pooling/flatten metadata appropriately.

Pitfall: mappings must match the actual layer sizes. If `inHeight * inWidth * inChannels`
does not correspond to the prior layer width (and similarly for outputs), export or import
may reject the model.

### Pool2DMapping

Mapping describing a pooling operation inserted after a given export-layer index.

This is represented as metadata and optional graph nodes during export.
Import uses it to attach pooling-related runtime metadata back onto the reconstructed
network (when supported).

### OnnxDimension

ONNX tensor type shape dimension.

### OnnxShape

ONNX tensor type shape.

### OnnxTensorType

ONNX tensor type.

### OnnxValueInfo

ONNX value info (input/output description).

### OnnxAttribute

ONNX node attribute.

### OnnxModel

ONNX-like model container (JSON-serializable).

This is the main “wire format” object in this folder. Persist it as JSON text:

```ts
const jsonText = JSON.stringify(model);
const restoredModel = JSON.parse(jsonText) as OnnxModel;
```

Notes:
- `metadata_props` contains NeatapticTS-specific keys (layer sizes, recurrent flags,
  conv/pool mappings, etc.). This is where most round-trip hints live.
- Initializers currently store floating-point weights in `float_data`.

Security/trust boundary:
- Treat this as untrusted input if it comes from outside your process.

### OnnxGraph

### OnnxTensor

### OnnxNode

### ActivationSquashFunction

```ts
ActivationSquashFunction(
  x: number,
  derivate: boolean | undefined,
): number
```

Activation function signature used by ONNX layer emission helpers.

### SharedGemmNodeBuildParams

Shared parameters for constructing a Gemm node payload.

### SharedActivationNodeBuildParams

Shared parameters for constructing an activation node payload.

### OptionalLayerOutputParams

Shared parameters for optional pooling/flatten output emission.

### DenseWeightBuildContext

Context for building dense layer initializers from two adjacent layers.

### DenseWeightRow

One collected dense row before fold to flattened initializers.

### DenseWeightBuildResult

Dense layer initializer fold output.

### DenseWeightRowCollectionContext

Context for collecting one dense row.

### DiagonalRecurrentBuildContext

Context for building a diagonal recurrent matrix from self-connections.

### RecurrentRowCollectionContext

Context for collecting one recurrent matrix row.

### OptionalPoolingAndFlattenParams

Parameters for optional pooling + flatten emission after a layer output.

### PoolingEmissionContext

Pooling emission context resolved for one layer output.

### FlattenAfterPoolingContext

Flatten emission context after optional pooling.

### PoolingAttributes

Pooling tensor attributes for ONNX node payloads.

### IndexedMetadataAppendContext

Append-an-index metadata context for JSON-array metadata keys.

### SpecMetadataAppendContext

Append-a-spec metadata context for JSON-array metadata keys.

### OnnxMetadataProperty

Canonical metadata key-value pair used in ONNX model metadata_props.

### DenseLayerParams

Parameters for dense layer emission.

### DenseLayerContext

Dense layer context enriched with resolved activation function.

### DenseTensorNames

Dense initializer tensor names.

### DenseInitializerValues

Dense initializer value arrays.

### DenseGraphNames

Dense graph tensor names.

### DenseActivationContext

Dense activation emission context.

### DenseGemmNodePayload

Strongly typed Gemm node payload used by dense export helpers.

### DenseActivationNodePayload

Strongly typed activation node payload used by dense export helpers.

### DenseOrderedNodePayload

Dense node payload union used by ordered append helpers.

### PerNeuronLayerParams

Parameters for per-neuron layer emission.

### PerNeuronLayerContext

Per-neuron layer context alias.

### PerNeuronSubgraphContext

Per-neuron subgraph emission context.

### PerNeuronNodeContext

Per-neuron normalized node context.

### PerNeuronTensorNames

Per-neuron initializer tensor names.

### PerNeuronGraphNames

Per-neuron graph tensor names.

### PerNeuronConcatNodePayload

Per-neuron concat node payload.

### OnnxLayerFactory

Runtime factory map used to construct dynamic recurrent layer modules.

### OnnxRuntimeLayerFactoryMap

Runtime layer module shape widened for fused-recurrent reconstruction wiring.

### OnnxFusedRecurrentKind

Supported fused recurrent operator families recognized during ONNX import.

### OnnxFusedLayerRuntime

Runtime interface of a reconstructed fused recurrent layer instance.

### OnnxFusedRecurrentSpec

Fused recurrent family specification used during import reconstruction.

### OnnxFusedLayerNeighborhood

Hidden-layer neighborhood slices around a reconstructed fused layer.

### OnnxFusedTensorPayload

Fused recurrent tensor payload read from ONNX initializers.

### OnnxFusedLayerReconstructionContext

Execution context for one fused recurrent layer reconstruction.

### OnnxFusedGateApplicationContext

Gate-weight application context for one reconstructed fused layer.

### OnnxFusedGateRowAssignmentContext

Context for assigning one gate-neuron row from flattened ONNX tensors.

### OnnxIncomingWeightAssignmentContext

Context for assigning dense incoming weights for one gate-neuron row.

### OnnxImportLayerWeightBucket

Bucketed ONNX dense/per-neuron tensors for one exported layer index.

### OnnxImportHiddenSizeDerivationContext

Context for deriving hidden layer sizes from initializer tensors and metadata.

### OnnxImportWeightAssignmentContext

Shared weight-assignment context built once per ONNX import.

### OnnxImportWeightAssignmentBuildParams

Build params for creating shared ONNX import weight-assignment context.

### OnnxImportLayerNodePair

Node slices for one sequential imported layer assignment pass.

### OnnxImportLayerNodePairBuildParams

Build params for one sequential layer node-pair slice operation.

### OnnxImportLayerTensorNames

Weight tensor names for one imported layer index.

### OnnxImportAggregatedLayerAssignmentContext

Context for assigning aggregated dense tensors for one layer.

### OnnxImportPerNeuronLayerAssignmentContext

Context for assigning per-neuron tensors for one layer.

### OnnxImportAggregatedNeuronAssignmentContext

Context for assigning one aggregated dense target neuron row.

### OnnxImportPerNeuronAssignmentContext

Context for assigning one per-neuron imported target node.

### OnnxImportConvMetadata

Parsed Conv metadata payload used for optional reconstruction pass.

### OnnxImportConvLayerContext

Context for reconstructing one Conv layer's imported connectivity.

### OnnxImportConvLayerContextBuildParams

Build params for creating one Conv reconstruction layer context.

### OnnxImportConvTensorContext

Resolved Conv initializer tensors and dimensions for one layer.

### OnnxImportConvNodeSlices

Layer node slices used while applying Conv reconstruction assignments.

### OnnxImportConvOutputCoordinate

Coordinate for one Conv output neuron traversal position.

### OnnxImportConvCoordinateAssignmentContext

Context for applying Conv weights/bias at one output coordinate.

### OnnxImportInboundConnectionMap

Inbound connection lookup map keyed by source node for one target neuron.

### OnnxImportConvKernelAssignmentContext

Context for assigning one concrete Conv kernel connection weight.

### OnnxImportArchitectureResult

Parsed architecture dimensions extracted from ONNX import graph payloads.

### OnnxImportArchitectureContext

Shared architecture extraction context with resolved graph dimensions.

### OnnxImportDimensionRecord

Loose ONNX shape-dimension record used by legacy import payload access.

### OnnxImportRecurrentRestorationContext

Context for recurrent self-connection restoration from ONNX metadata and tensors.

### OnnxImportHiddenLayerSpan

Hidden-layer span payload with one-based layer numbering and global offset.

### OnnxImportLayerConnectionContext

Execution context for assigning one hidden-layer recurrent diagonal tensor.

### OnnxImportSelfConnectionUpsertContext

Context for upserting one hidden node self-connection from recurrent weight.

### OnnxImportPoolingMetadata

Parsed pooling metadata payload attached to imported network instances.

### NetworkWithOnnxImportPooling

Network instance augmented with optional imported ONNX pooling metadata.

## architecture/network/onnx/network.onnx.utils.ts

ONNX export/import utilities for a constrained, documented subset of networks.

Phase Coverage (incremental roadmap implemented so far):
 - Phase 1: Deterministic layered MLP export (Gemm + Activation pairs) with basic metadata.
 - Phase 2: Optional partial connectivity (missing edges -> 0 weight) and mixed per-neuron activations
             (decomposed into per-neuron Gemm + Activation + Concat) via `allowPartialConnectivity` /
             `allowMixedActivations`.
 - Phase 3 (baseline): Multi-layer self‑recurrence single‑step representation (`allowRecurrent` +
             `recurrentSingleStep`) adding per-recurrent-layer previous state inputs and diagonal R matrices.
 - Phase 3 (experimental extension): Heuristic detection + emission of simplified LSTM / GRU fused nodes
             (no sequence axis, simplified bias & recurrence handling) while retaining original Gemm path.

Scope & Assumptions (current):
 - Network must be strictly layered and acyclic (feed‑forward between layers; optional self recurrence within
   hidden layers when enabled).
 - Homogeneous activation per layer unless `allowMixedActivations` is true (then per-neuron decomposition used).
 - Only a minimal ONNX tensor / node subset is emitted (no external ONNX proto dependency; pure JSON shape).
 - Recurrent support limited to: (a) self-connections mapped to diagonal Rk matrices (single step),
   (b) experimental fused LSTM/GRU heuristics relying on equal partition patterns (not spec-complete).
 - LSTM / GRU biases currently single segment (Wb only) and recurrent bias (Rb) implicitly zero; ordering of
   gates documented in code comments (may differ from canonical ONNX gate ordering and will be normalized later).

Metadata Keys (may appear in `model.metadata_props` when `includeMetadata` true):
 - `layer_sizes`: JSON array of hidden layer sizes.
 - `recurrent_single_step`: JSON array of 1-based hidden layer indices with exported self recurrence.
 - `lstm_groups_stub`: Heuristic grouping stubs for prospective LSTM layers (pre-emission discovery data).
 - `lstm_emitted_layers` / `gru_emitted_layers`: Arrays of export-layer indices where fused nodes were emitted.
 - `rnn_pattern_fallback`: Records near-miss pattern sizes for diagnostic purposes.

Design Goals:
 - Zero heavy runtime dependencies; the structure is intentionally lightweight & serializable.
 - Early, explicit structural validation with actionable error messages.
 - Transparent, stepwise transform for testability and deterministic round-tripping.

Limitations / TODO (tracked for later phases):
 - Proper ONNX-compliant LSTM/GRU biases (split Wb/Rb) & complete gate ordering alignment.
 - Pruning or replacing redundant Gemm graph segments when fused recurrent ops are emitted (currently both kept).
 - Multi-time-step sequence handling (currently single-step recurrent representation only).
 - Richer recurrence (off-diagonal intra-layer connectivity) and gating reconstruction fidelity.

NOTE: Import is only guaranteed to work for models produced by `exportToONNX()`; arbitrary ONNX graphs are
NOT supported. Experimental fused recurrent nodes are best-effort and may silently degrade if shapes mismatch.

### buildOnnxModel

```ts
buildOnnxModel(
  network: default,
  layers: default[][],
  options: OnnxExportOptions,
): OnnxModel
```

Build an ONNX-like model from a validated layered network view.

Role in the ONNX pipeline:
- This function is a thin, stable orchestration boundary used by higher-level exporters.
- It forwards to the implementation module while preserving a predictable public API
  for callers that import from this compatibility barrel.
- Keeping this wrapper explicit helps isolate call sites from internal file splits
  and phased refactors in export internals.

Expected preconditions:
- `layers` has already been inferred from the same `network` instance.
- Structural validation (layer homogeneity/connectivity and option gates) is complete.
- Export options are normalized by the caller according to project defaults.

High-level behavior:
 1. Receive network, ordered layer matrix, and export options.
 2. Delegate model construction to the concrete builder implementation.
 3. Return the resulting ONNX-like JSON graph container unchanged.

Parameters:
- `network` - - Source network to serialize.
- `layers` - - Ordered layer matrix produced by layer inference utilities.
- `options` - - Export options controlling metadata/recurrent/partial-connectivity behavior.

Returns: ONNX-like model object representing graph nodes, tensors, and metadata.

Example:

```ts
const layers = inferLayerOrdering(network);
const model = buildOnnxModel(network, layers, { includeMetadata: true });
```

### OnnxModel

ONNX-like model container (JSON-serializable).

This is the main “wire format” object in this folder. Persist it as JSON text:

```ts
const jsonText = JSON.stringify(model);
const restoredModel = JSON.parse(jsonText) as OnnxModel;
```

Notes:
- `metadata_props` contains NeatapticTS-specific keys (layer sizes, recurrent flags,
  conv/pool mappings, etc.). This is where most round-trip hints live.
- Initializers currently store floating-point weights in `float_data`.

Security/trust boundary:
- Treat this as untrusted input if it comes from outside your process.

### inferLayerOrdering

```ts
inferLayerOrdering(
  network: default,
): default[][]
```

Infer strictly layered ordering from a network.

Parameters:
- `network` - Source network.

Returns: Ordered layers: input, hidden..., output.

### rebuildConnectionsLocal

```ts
rebuildConnectionsLocal(
  networkLike: default,
): void
```

Rebuild the network's flat connections array from each node's outgoing list.

Parameters:
- `networkLike` - Network-like instance to mutate.

Returns: Nothing.

### validateLayerHomogeneityAndConnectivity

```ts
validateLayerHomogeneityAndConnectivity(
  layers: default[][],
  network: default,
  options: OnnxExportOptions,
): void
```

Validate connectivity and activation homogeneity constraints per layer.

Parameters:
- `layers` - Layered node arrays.
- `network` - Source network (reserved for compatibility).
- `options` - Export options.

Returns: Nothing.

### runOnnxExportFlow

```ts
runOnnxExportFlow(
  network: default,
  options: OnnxExportOptions,
): OnnxModel
```

Execute the complete ONNX export flow for one network instance.

High-level behavior:
 1. Rebuild runtime connection caches and assign stable export indices.
 2. Infer layered ordering and collect recurrent-pattern stubs.
 3. Validate structural constraints for the requested export options.
 4. Build ONNX graph payload and append inference-oriented metadata.

Parameters:
- `network` - Source network to serialize.
- `options` - Optional ONNX export controls.

Returns: ONNX-like model payload.

### runOnnxImportFlow

```ts
runOnnxImportFlow(
  onnx: OnnxModel,
): default
```

Execute the complete ONNX import flow and reconstruct a runtime network.

High-level behavior:
 1. Extract architecture dimensions and build a perceptron scaffold.
 2. Restore dense parameters and activation functions.
 3. Reconstruct recurrent/pooling metadata and rebuild connection caches.

Parameters:
- `onnx` - ONNX-like model payload to reconstruct.

Returns: Reconstructed network instance.

### assignActivationFunctions

```ts
assignActivationFunctions(
  network: default,
  onnx: OnnxModel,
  hiddenLayerSizes: number[],
): void
```

Assign node activation functions from ONNX activation nodes.

Parameters:
- `network` - Target network to mutate.
- `onnx` - Source ONNX model.
- `hiddenLayerSizes` - Hidden layer size list.

Returns: Nothing.

### assignWeightsAndBiases

```ts
assignWeightsAndBiases(
  network: default,
  onnx: OnnxModel,
  hiddenLayerSizes: number[],
  metadataProps: OnnxMetadataProperty[] | undefined,
): void
```

Assign weights and biases from ONNX initializers to a newly created network.

Parameters:
- `network` - Target network to mutate.
- `onnx` - Source ONNX model.
- `hiddenLayerSizes` - Hidden layer sizes.
- `metadataProps` - Optional ONNX metadata properties.

Returns: Nothing.

### deriveHiddenLayerSizes

```ts
deriveHiddenLayerSizes(
  initializers: OnnxTensor[],
  metadataProps: OnnxMetadataProperty[] | undefined,
): number[]
```

Extract hidden layer sizes from ONNX initializers (weight tensors).

Parameters:
- `initializers` - ONNX initializer tensors.
- `metadataProps` - Optional ONNX metadata properties.

Returns: Hidden layer sizes in order.

### applyModelMetadata

```ts
applyModelMetadata(
  context: OnnxModelMetadataContext,
): void
```

Attach producer and opset metadata to a model when metadata emission is enabled.

Parameters:
- `context` - Metadata application context.

Returns: Nothing.

### collectRecurrentLayerIndices

```ts
collectRecurrentLayerIndices(
  context: OnnxRecurrentCollectionContext,
): number[]
```

Detect hidden layers with self-recurrence and add matching previous-state graph inputs.

Parameters:
- `context` - Recurrent collection context.

Returns: Export-layer indices with recurrent self-connections.

### createBaseModel

```ts
createBaseModel(
  context: OnnxBaseModelBuildContext,
): OnnxModel
```

Create the base ONNX model shell with graph input/output declarations.

Parameters:
- `context` - Base model build context.

Returns: Initialized ONNX model with empty initializer/node lists.

### createGraphDimensions

```ts
createGraphDimensions(
  context: OnnxGraphDimensionBuildContext,
): OnnxGraphDimensions
```

Build tensor dimensions for model input and output, optionally with symbolic batch dimension.

Parameters:
- `context` - Dimension construction context.

Returns: Input and output dimension arrays for ONNX value info.

### emitLayerGraph

```ts
emitLayerGraph(
  context: LayerBuildContext,
): string
```

Emit one export layer graph segment and return the produced output tensor name.

Parameters:
- `context` - Layer build context.

Returns: Output tensor name produced by this layer.

### emitFusedRecurrentHeuristics

```ts
emitFusedRecurrentHeuristics(
  model: OnnxModel,
  layers: default[][],
  allowRecurrent: boolean | undefined,
  previousOutputName: string,
): void
```

Emit heuristic fused recurrent operators (LSTM/GRU) when recurrent export is enabled.

Parameters:
- `model` - Target ONNX model.
- `layers` - Layered network nodes.
- `allowRecurrent` - Whether recurrent export is enabled.
- `previousOutputName` - Current graph output name (kept for backward-compatible emission semantics).

Returns: Nothing.

### finalizeExportMetadata

```ts
finalizeExportMetadata(
  model: OnnxModel,
  layers: default[][],
  options: OnnxExportOptions,
  includeMetadata: boolean,
  hiddenSizesMetadata: number[],
  recurrentLayerIndices: number[],
): void
```

Finalize export metadata and optional conv-sharing validation.

Parameters:
- `model` - Target ONNX model.
- `layers` - Layered network nodes.
- `options` - Export options.
- `includeMetadata` - Whether metadata emission is enabled.
- `hiddenSizesMetadata` - Hidden-layer sizes collected during emission.
- `recurrentLayerIndices` - Recurrent layer indices.

Returns: Nothing.

## architecture/network/onnx/network.onnx.export-conv.utils.ts

### tryEmitConvLayer

```ts
tryEmitConvLayer(
  params: OnnxConvEmissionParams,
): string | undefined
```

Try to emit a conv-mapped layer.

Parameters:
- `params` - Conv emission parameters.

Returns: New output tensor name when handled, otherwise undefined.

## architecture/network/onnx/network.onnx.export-flow.utils.ts

### runOnnxExportFlow

```ts
runOnnxExportFlow(
  network: default,
  options: OnnxExportOptions,
): OnnxModel
```

Execute the complete ONNX export flow for one network instance.

High-level behavior:
 1. Rebuild runtime connection caches and assign stable export indices.
 2. Infer layered ordering and collect recurrent-pattern stubs.
 3. Validate structural constraints for the requested export options.
 4. Build ONNX graph payload and append inference-oriented metadata.

Parameters:
- `network` - Source network to serialize.
- `options` - Optional ONNX export controls.

Returns: ONNX-like model payload.

## architecture/network/onnx/network.onnx.import-flow.utils.ts

### runOnnxImportFlow

```ts
runOnnxImportFlow(
  onnx: OnnxModel,
): default
```

Execute the complete ONNX import flow and reconstruct a runtime network.

High-level behavior:
 1. Extract architecture dimensions and build a perceptron scaffold.
 2. Restore dense parameters and activation functions.
 3. Reconstruct recurrent/pooling metadata and rebuild connection caches.

Parameters:
- `onnx` - ONNX-like model payload to reconstruct.

Returns: Reconstructed network instance.

## architecture/network/onnx/network.onnx.export-build.utils.ts

### buildOnnxModel

```ts
buildOnnxModel(
  network: default,
  layers: default[][],
  options: OnnxExportOptions,
): OnnxModel
```

Construct ONNX graph (initializers + nodes) from validated layered network structure.

Parameters:
- `network` - Source network (retained for API compatibility).
- `layers` - Layered nodes including input and output layers.
- `options` - Export options.

Returns: ONNX model.

## architecture/network/onnx/network.onnx.export-dense.utils.ts

### emitDenseLayer

```ts
emitDenseLayer(
  params: DenseLayerParams,
): string
```

Emit dense layer representation.

Parameters:
- `params` - Dense emission parameters.

Returns: Output tensor name.

### emitPerNeuronLayer

```ts
emitPerNeuronLayer(
  params: PerNeuronLayerParams,
): string
```

Emit per-neuron decomposition layer representation.

Parameters:
- `params` - Per-neuron emission parameters.

Returns: Output tensor name.

### emitDenseInitializers

```ts
emitDenseInitializers(
  layerContext: DenseLayerContext,
): DenseTensorNames
```

Emit dense initializers and return tensor names.

Parameters:
- `layerContext` - Dense layer context.

Returns: Tensor names.

### collectDenseInitializerValues

```ts
collectDenseInitializerValues(
  layerContext: DenseLayerContext,
): DenseInitializerValues
```

Collect dense weight matrix and bias vector values.

Parameters:
- `layerContext` - Dense layer context.

Returns: Dense initializer values.

### createDenseTensorNames

```ts
createDenseTensorNames(
  layerIndex: number,
): DenseTensorNames
```

Build dense tensor names for initializer emission.

Parameters:
- `layerIndex` - Layer index.

Returns: Dense tensor names.

### appendDenseWeightInitializer

```ts
appendDenseWeightInitializer(
  layerContext: DenseLayerContext,
  weightTensorName: string,
  weightMatrixValues: number[],
): void
```

Append dense weight initializer.

Parameters:
- `layerContext` - Dense layer context.
- `weightTensorName` - Weight tensor name.
- `weightMatrixValues` - Weight values.

Returns: Nothing.

### appendDenseBiasInitializer

```ts
appendDenseBiasInitializer(
  layerContext: DenseLayerContext,
  biasTensorName: string,
  biasVector: number[],
): void
```

Append dense bias initializer.

Parameters:
- `layerContext` - Dense layer context.
- `biasTensorName` - Bias tensor name.
- `biasVector` - Bias vector values.

Returns: Nothing.

### emitDenseActivationSubgraph

```ts
emitDenseActivationSubgraph(
  model: OnnxModel,
  denseActivationContext: DenseActivationContext,
): void
```

Emit Gemm and activation nodes using requested ordering.

Parameters:
- `model` - Target ONNX model.
- `denseActivationContext` - Dense activation context.

Returns: Nothing.

### resolveDenseNodeOrder

```ts
resolveDenseNodeOrder(
  gemmNode: DenseGemmNodePayload,
  activationNode: DenseActivationNodePayload,
  legacyNodeOrdering: boolean,
): DenseOrderedNodePayload[]
```

Resolve dense node order for legacy and current exports.

Parameters:
- `gemmNode` - Gemm node.
- `activationNode` - Activation node.
- `legacyNodeOrdering` - Whether legacy ordering is required.

Returns: Ordered node list.

### appendDenseNodes

```ts
appendDenseNodes(
  model: OnnxModel,
  orderedNodes: DenseOrderedNodePayload[],
): void
```

Append ordered dense nodes to the model graph.

Parameters:
- `model` - Target model.
- `orderedNodes` - Ordered dense nodes.

Returns: Nothing.

### createGemmNode

```ts
createGemmNode(
  denseActivationContext: DenseActivationContext,
): DenseGemmNodePayload
```

Create dense Gemm node definition.

Parameters:
- `denseActivationContext` - Dense activation context.

Returns: ONNX Gemm node payload.

### createActivationNode

```ts
createActivationNode(
  denseActivationContext: DenseActivationContext,
): DenseActivationNodePayload
```

Create dense activation node definition.

Parameters:
- `denseActivationContext` - Dense activation context.

Returns: ONNX activation node payload.

### emitPerNeuronSubgraph

```ts
emitPerNeuronSubgraph(
  perNeuronSubgraphContext: PerNeuronSubgraphContext,
): string
```

Emit per-neuron Gemm + activation subgraph.

Parameters:
- `perNeuronSubgraphContext` - Per-neuron subgraph context.

Returns: Per-neuron activation output name.

### buildSingleNeuronWeightRow

```ts
buildSingleNeuronWeightRow(
  targetNodeInternal: NodeInternals,
  previousLayerNodes: default[],
): number[]
```

Build one neuron's incoming weight row against previous layer.

Parameters:
- `targetNodeInternal` - Target node internals.
- `previousLayerNodes` - Previous layer nodes.

Returns: Weight row values.

### resolveSingleNeuronInboundWeight

```ts
resolveSingleNeuronInboundWeight(
  targetNodeInternal: NodeInternals,
  sourceNode: default,
): number
```

Resolve one inbound connection weight for a source node.

Parameters:
- `targetNodeInternal` - Target node internals.
- `sourceNode` - Source node.

Returns: Inbound weight or zero when missing.

### createSharedGemmNodePayload

```ts
createSharedGemmNodePayload(
  params: SharedGemmNodeBuildParams,
): DenseGemmNodePayload
```

Build a shared Gemm node payload.

Parameters:
- `params` - Shared Gemm build parameters.

Returns: Gemm node payload.

### createSharedActivationNodePayload

```ts
createSharedActivationNodePayload(
  params: SharedActivationNodeBuildParams,
): DenseActivationNodePayload
```

Build a shared activation node payload.

Parameters:
- `params` - Shared activation build parameters.

Returns: Activation node payload.

### createDefaultGemmAttributes

```ts
createDefaultGemmAttributes(): { name: string; type: string; f?: number | undefined; i?: number | undefined; }[]
```

Build default Gemm attributes for ONNX export.

Returns: Default Gemm attribute list.

### emitOptionalLayerOutput

```ts
emitOptionalLayerOutput(
  params: OptionalLayerOutputParams,
): string
```

Emit optional pooling and flatten output fold.

Parameters:
- `params` - Optional output parameters.

Returns: Output tensor name.

## architecture/network/onnx/network.onnx.export-setup.utils.ts

### createGraphDimensions

```ts
createGraphDimensions(
  context: OnnxGraphDimensionBuildContext,
): OnnxGraphDimensions
```

Build tensor dimensions for model input and output, optionally with symbolic batch dimension.

Parameters:
- `context` - Dimension construction context.

Returns: Input and output dimension arrays for ONNX value info.

### createBaseModel

```ts
createBaseModel(
  context: OnnxBaseModelBuildContext,
): OnnxModel
```

Create the base ONNX model shell with graph input/output declarations.

Parameters:
- `context` - Base model build context.

Returns: Initialized ONNX model with empty initializer/node lists.

### applyModelMetadata

```ts
applyModelMetadata(
  context: OnnxModelMetadataContext,
): void
```

Attach producer and opset metadata to a model when metadata emission is enabled.

Parameters:
- `context` - Metadata application context.

Returns: Nothing.

### collectRecurrentLayerIndices

```ts
collectRecurrentLayerIndices(
  context: OnnxRecurrentCollectionContext,
): number[]
```

Detect hidden layers with self-recurrence and add matching previous-state graph inputs.

Parameters:
- `context` - Recurrent collection context.

Returns: Export-layer indices with recurrent self-connections.

### createTensorDimensions

```ts
createTensorDimensions(
  width: number,
  batchDimension: boolean,
): OnnxDimension[]
```

Build one tensor shape dimension payload for dense vectors.

Parameters:
- `width` - Vector width.
- `batchDimension` - Whether symbolic batch dimension is enabled.

Returns: ONNX dimensions for the vector payload.

### createGraphValueInfo

```ts
createGraphValueInfo(
  valueName: string,
  dimensions: OnnxDimension[],
): OnnxValueInfo
```

Create ONNX value info payload for one graph boundary tensor.

Parameters:
- `valueName` - Tensor value name.
- `dimensions` - Tensor dimensions.

Returns: ONNX value info payload.

### isRecurrentCollectionEnabled

```ts
isRecurrentCollectionEnabled(
  context: OnnxRecurrentCollectionContext,
): boolean
```

Determine whether recurrent layer collection should execute.

Parameters:
- `context` - Recurrent collection context.

Returns: True when recurrent collection is enabled.

### createHiddenLayerTraversalContexts

```ts
createHiddenLayerTraversalContexts(
  context: OnnxRecurrentCollectionContext,
): OnnxRecurrentLayerTraversalContext[]
```

Build traversal contexts for all hidden layers.

Parameters:
- `context` - Recurrent collection context.

Returns: Hidden layer traversal contexts.

### createHiddenLayerIndices

```ts
createHiddenLayerIndices(
  totalLayerCount: number,
): number[]
```

Build hidden layer indices excluding input and output layers.

Parameters:
- `totalLayerCount` - Total number of network layers.

Returns: Hidden layer indices.

### processHiddenLayerRecurrence

```ts
processHiddenLayerRecurrence(
  context: OnnxRecurrentLayerProcessingContext,
): void
```

Process one hidden layer for recurrent self-connections.

Parameters:
- `context` - Hidden layer recurrent processing context.

Returns: Nothing.

### appendRecurrentLayerIndex

```ts
appendRecurrentLayerIndex(
  recurrentLayerIndices: number[],
  traversalContext: OnnxRecurrentLayerTraversalContext,
): void
```

Append one recurrent layer index to the collected index list.

Parameters:
- `recurrentLayerIndices` - Collected recurrent layer indices.
- `traversalContext` - Hidden layer traversal context.

Returns: Nothing.

### appendRecurrentGraphInput

```ts
appendRecurrentGraphInput(
  model: OnnxModel,
  traversalContext: OnnxRecurrentLayerTraversalContext,
): void
```

Append one recurrent previous-state graph input for a hidden layer.

Parameters:
- `model` - Target ONNX model.
- `traversalContext` - Hidden layer traversal context.

Returns: Nothing.

### hasLayerSelfRecurrence

```ts
hasLayerSelfRecurrence(
  hiddenLayerNodes: default[],
): boolean
```

Detect whether a hidden layer contains at least one self-recurrent node.

Parameters:
- `hiddenLayerNodes` - Hidden layer nodes.

Returns: True when any node has a self-connection.

### createRecurrentInputValueInfoContext

```ts
createRecurrentInputValueInfoContext(
  traversalContext: OnnxRecurrentLayerTraversalContext,
): OnnxRecurrentInputValueInfoContext
```

Build recurrent input context for one hidden recurrent layer.

Parameters:
- `traversalContext` - Hidden layer traversal context.

Returns: Recurrent input value-info context.

### createRecurrentInputValueInfo

```ts
createRecurrentInputValueInfo(
  context: OnnxRecurrentInputValueInfoContext,
): OnnxValueInfo
```

Build one recurrent previous-state graph input payload.

Parameters:
- `context` - Recurrent input value-info context.

Returns: ONNX value info payload for recurrent state input.

## architecture/network/onnx/network.onnx.runtime-load.utils.ts

### loadRuntimeFactories

```ts
loadRuntimeFactories(): OnnxRuntimeFactories
```

Resolve runtime factories used by ONNX import orchestration.

Returns: Perceptron factory and layer module object.

### createPerceptronFactory

```ts
createPerceptronFactory(): OnnxRuntimePerceptronFactory
```

Create an ONNX import network factory from modern static constructors.

Returns: Perceptron-compatible factory function.

### createRuntimeLayerModule

```ts
createRuntimeLayerModule(): OnnxRuntimeLayerModule
```

Create the runtime layer-module wiring used by ONNX import orchestrators.

Returns: Runtime recurrent-layer module object.

### foldRuntimeFactories

```ts
foldRuntimeFactories(
  perceptronFactory: OnnxRuntimePerceptronFactory,
  layerModule: OnnxRuntimeLayerModule,
): OnnxRuntimeFactories
```

Fold runtime perceptron and layer module into a transport payload.

Parameters:
- `perceptronFactory` - Perceptron factory function.
- `layerModule` - Runtime recurrent-layer constructors.

Returns: Runtime factories payload.

### foldRuntimeLayerModule

```ts
foldRuntimeLayerModule(
  lstmFactory: OnnxRuntimeLayerFactory,
  gruFactory: OnnxRuntimeLayerFactory,
): OnnxRuntimeLayerModule
```

Fold LSTM/GRU factories into a runtime layer module payload.

Parameters:
- `lstmFactory` - Runtime LSTM layer factory.
- `gruFactory` - Runtime GRU layer factory.

Returns: Runtime layer module.

### resolveLayerFactory

```ts
resolveLayerFactory(
  layerKey: keyof OnnxRuntimeLayerModule,
): OnnxRuntimeLayerFactory
```

Resolve one runtime layer factory by module key.

Parameters:
- `layerKey` - Runtime layer key.

Returns: Matching layer factory.

### createPerceptronSizeValidationContext

```ts
createPerceptronSizeValidationContext(
  sizes: number[],
): OnnxPerceptronSizeValidationContext
```

Build perceptron-size validation context.

Parameters:
- `sizes` - Layer-size payload.

Returns: Validation context.

### validatePerceptronSizes

```ts
validatePerceptronSizes(
  validationContext: OnnxPerceptronSizeValidationContext,
): void
```

Validate perceptron size-list constraints.

Parameters:
- `validationContext` - Validation context.

Returns: Nothing. Throws on invalid size-list.

### createPerceptronBuildContext

```ts
createPerceptronBuildContext(
  sizes: number[],
): OnnxPerceptronBuildContext
```

Build perceptron-network construction context.

Parameters:
- `sizes` - Layer-size payload.

Returns: Build context.

### buildPerceptronNetwork

```ts
buildPerceptronNetwork(
  buildContext: OnnxPerceptronBuildContext,
): default
```

Build a perceptron network from size-extraction context.

Parameters:
- `buildContext` - Perceptron build context.

Returns: Reconstructed network instance.

## architecture/network/onnx/network.onnx.import-weights.utils.ts

### deriveHiddenLayerSizes

```ts
deriveHiddenLayerSizes(
  initializers: OnnxTensor[],
  metadataProps: OnnxMetadataProperty[] | undefined,
): number[]
```

Extract hidden layer sizes from ONNX initializers (weight tensors).

Parameters:
- `initializers` - ONNX initializer tensors.
- `metadataProps` - Optional ONNX metadata properties.

Returns: Hidden layer sizes in order.

### assignWeightsAndBiases

```ts
assignWeightsAndBiases(
  network: default,
  onnx: OnnxModel,
  hiddenLayerSizes: number[],
  metadataProps: OnnxMetadataProperty[] | undefined,
): void
```

Assign weights and biases from ONNX initializers to a newly created network.

Parameters:
- `network` - Target network to mutate.
- `onnx` - Source ONNX model.
- `hiddenLayerSizes` - Hidden layer sizes.
- `metadataProps` - Optional ONNX metadata properties.

Returns: Nothing.

### parseMetadataLayerSizes

```ts
parseMetadataLayerSizes(
  metadataProps: OnnxMetadataProperty[],
): number[] | null
```

Parse explicit metadata-driven hidden layer sizes.

Parameters:
- `metadataProps` - ONNX metadata properties.

Returns: Parsed hidden layer sizes when available.

### collectLayerWeightBuckets

```ts
collectLayerWeightBuckets(
  initializers: OnnxTensor[],
): Record<string, OnnxImportLayerWeightBucket>
```

Collect ONNX weight tensor buckets grouped by export layer index.

Parameters:
- `initializers` - ONNX initializer tensors.

Returns: Layer-weight buckets keyed by export layer index.

### collectSortedLayerIndices

```ts
collectSortedLayerIndices(
  layerWeightBuckets: Record<string, OnnxImportLayerWeightBucket>,
): number[]
```

Collect sorted layer indices from weight buckets.

Parameters:
- `layerWeightBuckets` - Layer-weight buckets.

Returns: Ascending export layer indices.

### buildHiddenLayerSizesFromBuckets

```ts
buildHiddenLayerSizesFromBuckets(
  layerWeightBuckets: Record<string, OnnxImportLayerWeightBucket>,
  sortedLayerIndices: number[],
): number[]
```

Build hidden-layer sizes from weight buckets while excluding output layer.

Parameters:
- `layerWeightBuckets` - Layer-weight buckets.
- `sortedLayerIndices` - Ascending layer indices.

Returns: Hidden-layer sizes.

### resolveLayerHiddenSize

```ts
resolveLayerHiddenSize(
  layerWeightBuckets: Record<string, OnnxImportLayerWeightBucket>,
  layerIndex: number,
): number
```

Resolve one hidden-layer size from its weight bucket.

Parameters:
- `layerWeightBuckets` - Layer-weight buckets.
- `layerIndex` - Export layer index.

Returns: Hidden-layer size.

### buildWeightAssignmentContext

```ts
buildWeightAssignmentContext(
  params: OnnxImportWeightAssignmentBuildParams,
): OnnxImportWeightAssignmentContext
```

Build the shared assignment context for import weight restoration.

Parameters:
- `params` - Assignment context input params.

Returns: Shared assignment context.

### parseLayerIndexFromWeightTensor

```ts
parseLayerIndexFromWeightTensor(
  tensorName: string,
): number | null
```

Parse layer index from dense/per-neuron weight tensor name.

Parameters:
- `tensorName` - Tensor name.

Returns: Parsed layer index or null.

### parseWeightTensorName

```ts
parseWeightTensorName(
  tensorName: string,
): { layerIndex: string; neuronIndex: number | null; } | null
```

Parse layer/neuron components from a weight tensor name.

Parameters:
- `tensorName` - Tensor name.

Returns: Parsed layer+neuron components when matched.

### buildInitializerMap

```ts
buildInitializerMap(
  initializers: OnnxTensor[],
): Record<string, OnnxTensor>
```

Build ONNX initializer map keyed by tensor name.

Parameters:
- `initializers` - ONNX initializer list.

Returns: Tensor map by name.

### collectSortedUniqueLayerIndices

```ts
collectSortedUniqueLayerIndices(
  initializerMap: Record<string, OnnxTensor>,
): number[]
```

Collect unique sorted layer indices from initializer weight tensors.

Parameters:
- `initializerMap` - Initializer map keyed by tensor name.

Returns: Unique sorted layer indices.

### collectNodesByType

```ts
collectNodesByType(
  nodes: default[],
  nodeType: "input" | "output" | "hidden",
): default[]
```

Collect nodes by runtime node type discriminator.

Parameters:
- `nodes` - Network nodes.
- `nodeType` - Runtime node type.

Returns: Filtered nodes.

### applyDenseWeightAssignments

```ts
applyDenseWeightAssignments(
  assignmentContext: OnnxImportWeightAssignmentContext,
): void
```

Apply dense/per-neuron assignments for all sorted layer indices.

Parameters:
- `assignmentContext` - Shared assignment context.

Returns: Nothing.

### buildLayerNodePair

```ts
buildLayerNodePair(
  assignmentContext: OnnxImportWeightAssignmentContext,
  params: OnnxImportLayerNodePairBuildParams,
): OnnxImportLayerNodePair
```

Build current/previous node slices for one sequential import layer pass.

Parameters:
- `assignmentContext` - Shared assignment context.
- `params` - Sequential traversal params.

Returns: Layer node pair.

### resolveCurrentLayerNodes

```ts
resolveCurrentLayerNodes(
  assignmentContext: OnnxImportWeightAssignmentContext,
  params: { sequentialIndex: number; },
): default[]
```

Resolve current layer nodes for one sequential layer assignment pass.

Parameters:
- `assignmentContext` - Shared assignment context.
- `params` - Sequential traversal params.

Returns: Current layer nodes.

### resolvePreviousLayerNodes

```ts
resolvePreviousLayerNodes(
  assignmentContext: OnnxImportWeightAssignmentContext,
  params: { sequentialIndex: number; },
): default[]
```

Resolve previous layer nodes for one sequential layer assignment pass.

Parameters:
- `assignmentContext` - Shared assignment context.
- `params` - Sequential traversal params.

Returns: Previous layer nodes.

### sumHiddenSizesToIndex

```ts
sumHiddenSizesToIndex(
  hiddenLayerSizes: number[],
  exclusiveEndIndex: number,
): number
```

Sum hidden-layer sizes from index `0` to `exclusiveEndIndex`.

Parameters:
- `hiddenLayerSizes` - Hidden-layer size list.
- `exclusiveEndIndex` - Exclusive end index.

Returns: Prefix sum.

### assignLayerWeights

```ts
assignLayerWeights(
  initializerMap: Record<string, OnnxTensor>,
  nodePair: OnnxImportLayerNodePair,
): void
```

Assign one layer's weights using aggregated or per-neuron tensors.

Parameters:
- `initializerMap` - ONNX initializer map.
- `nodePair` - Current/previous node slices.

Returns: Nothing.

### hasAggregatedLayerWeights

```ts
hasAggregatedLayerWeights(
  aggregatedContext: OnnxImportAggregatedLayerAssignmentContext,
): boolean
```

Determine whether the layer has aggregated weight tensor data.

Parameters:
- `aggregatedContext` - Aggregated assignment context.

Returns: True when aggregated tensor exists.

### buildLayerTensorNames

```ts
buildLayerTensorNames(
  layerIndex: number,
): OnnxImportLayerTensorNames
```

Build dense weight/bias tensor names for one layer index.

Parameters:
- `layerIndex` - Export layer index.

Returns: Layer tensor names.

### applyAggregatedLayerWeights

```ts
applyAggregatedLayerWeights(
  aggregatedContext: OnnxImportAggregatedLayerAssignmentContext,
): void
```

Apply aggregated dense tensor assignments for one layer.

Parameters:
- `aggregatedContext` - Aggregated assignment context.

Returns: Nothing.

### applyAggregatedNeuronAssignment

```ts
applyAggregatedNeuronAssignment(
  neuronContext: OnnxImportAggregatedNeuronAssignmentContext,
): void
```

Apply aggregated dense row weights and bias for one target neuron.

Parameters:
- `neuronContext` - Aggregated neuron assignment context.

Returns: Nothing.

### applyPerNeuronLayerWeights

```ts
applyPerNeuronLayerWeights(
  perNeuronContext: OnnxImportPerNeuronLayerAssignmentContext,
): void
```

Apply per-neuron tensor assignments for one layer.

Parameters:
- `perNeuronContext` - Per-neuron layer assignment context.

Returns: Nothing.

### buildPerNeuronTensorNames

```ts
buildPerNeuronTensorNames(
  layerIndex: number,
  neuronIndex: number,
): OnnxImportLayerTensorNames
```

Build per-neuron tensor names for one layer and neuron index.

Parameters:
- `layerIndex` - Export layer index.
- `neuronIndex` - Neuron index in layer.

Returns: Per-neuron tensor names.

### applyPerNeuronAssignment

```ts
applyPerNeuronAssignment(
  perNeuronAssignmentContext: OnnxImportPerNeuronAssignmentContext,
): void
```

Apply one per-neuron weight vector and bias assignment.

Parameters:
- `perNeuronAssignmentContext` - Per-neuron assignment context.

Returns: Nothing.

### applyOptionalConvReconstruction

```ts
applyOptionalConvReconstruction(
  assignmentContext: OnnxImportWeightAssignmentContext,
): void
```

Apply optional Conv2D reconstruction pass from metadata payloads.

Parameters:
- `assignmentContext` - Shared assignment context.

Returns: Nothing.

### parseConvMetadata

```ts
parseConvMetadata(
  metadataProps: OnnxMetadataProperty[],
): OnnxImportConvMetadata | null
```

Parse Conv reconstruction metadata payload.

Parameters:
- `metadataProps` - ONNX metadata properties.

Returns: Parsed Conv metadata.

### buildConvLayerContext

```ts
buildConvLayerContext(
  params: OnnxImportConvLayerContextBuildParams,
): OnnxImportConvLayerContext | null
```

Build one Conv layer reconstruction context.

Parameters:
- `params` - Conv context input params.

Returns: Conv layer context when valid.

### applyConvLayerReconstruction

```ts
applyConvLayerReconstruction(
  layerContext: OnnxImportConvLayerContext,
): void
```

Apply Conv reconstruction for one validated Conv layer context.

Parameters:
- `layerContext` - Conv layer context.

Returns: Nothing.

### buildConvNodeSlices

```ts
buildConvNodeSlices(
  layerContext: OnnxImportConvLayerContext,
): OnnxImportConvNodeSlices
```

Build Conv current/previous node slices for one layer context.

Parameters:
- `layerContext` - Conv layer context.

Returns: Node slice payload.

### buildConvTensorContext

```ts
buildConvTensorContext(
  layerContext: OnnxImportConvLayerContext,
): OnnxImportConvTensorContext | null
```

Build validated Conv tensor context for one layer.

Parameters:
- `layerContext` - Conv layer context.

Returns: Conv tensor context when valid.

### collectConvOutputCoordinates

```ts
collectConvOutputCoordinates(
  convSpec: Conv2DMapping,
  outChannels: number,
): OnnxImportConvOutputCoordinate[]
```

Collect all output traversal coordinates for one Conv layer.

Parameters:
- `convSpec` - Conv mapping spec.
- `outChannels` - Output channel count.

Returns: Output traversal coordinates.

### applyConvCoordinateAssignment

```ts
applyConvCoordinateAssignment(
  coordinateContext: OnnxImportConvCoordinateAssignmentContext,
): void
```

Apply Conv bias and kernel weights for one output coordinate.

Parameters:
- `coordinateContext` - Conv coordinate assignment context.

Returns: Nothing.

### buildConvNeuronLinearIndex

```ts
buildConvNeuronLinearIndex(
  coordinate: OnnxImportConvOutputCoordinate,
  convSpec: Conv2DMapping,
): number
```

Build flattened linear index for one Conv output coordinate.

Parameters:
- `coordinate` - Conv output coordinate.
- `convSpec` - Conv mapping spec.

Returns: Linear neuron index.

### collectConvKernelCoordinates

```ts
collectConvKernelCoordinates(
  inChannels: number,
  kernelHeight: number,
  kernelWidth: number,
): OnnxConvKernelCoordinate[]
```

Collect all kernel traversal coordinates for one Conv output position.

Parameters:
- `inChannels` - Input channel count.
- `kernelHeight` - Kernel height.
- `kernelWidth` - Kernel width.

Returns: Kernel traversal coordinates.

### assignConvKernelWeight

```ts
assignConvKernelWeight(
  kernelAssignmentContext: OnnxImportConvKernelAssignmentContext,
): void
```

Assign one Conv kernel weight to the matching inbound neuron connection.

Parameters:
- `kernelAssignmentContext` - Conv kernel assignment context.

Returns: Nothing.

### buildInputCoordinate

```ts
buildInputCoordinate(
  kernelAssignmentContext: OnnxImportConvKernelAssignmentContext,
): { inputRow: number; inputColumn: number; } | null
```

Build input-space coordinate for one Conv kernel element.

Parameters:
- `kernelAssignmentContext` - Conv kernel assignment context.

Returns: Input coordinate when in bounds.

### buildInputFeatureLinearIndex

```ts
buildInputFeatureLinearIndex(
  convSpec: Conv2DMapping,
  inChannelIndex: number,
  inputRow: number,
  inputColumn: number,
): number
```

Build linear feature index in input feature space.

Parameters:
- `convSpec` - Conv mapping spec.
- `inChannelIndex` - Input channel index.
- `inputRow` - Input row index.
- `inputColumn` - Input column index.

Returns: Linear input feature index.

### buildInboundConnectionMap

```ts
buildInboundConnectionMap(
  neuronInternal: NodeInternals,
): OnnxImportInboundConnectionMap
```

Build inbound connection lookup map for one neuron.

Parameters:
- `neuronInternal` - Neuron internals.

Returns: Inbound connection map keyed by source node.

### readConvKernelWeight

```ts
readConvKernelWeight(
  kernelAssignmentContext: OnnxImportConvKernelAssignmentContext,
): number
```

Read one Conv kernel weight from flattened ONNX tensor payload.

Parameters:
- `kernelAssignmentContext` - Conv kernel assignment context.

Returns: Kernel weight.

## architecture/network/onnx/network.onnx.layer-analysis.utils.ts

### rebuildConnectionsLocal

```ts
rebuildConnectionsLocal(
  networkLike: default,
): void
```

Rebuild the network's flat connections array from each node's outgoing list.

Parameters:
- `networkLike` - Network-like instance to mutate.

Returns: Nothing.

### mapActivationToOnnx

```ts
mapActivationToOnnx(
  squash: ((x: number, derivate?: boolean | undefined) => number) & { name?: string | undefined; },
): OnnxActivationOperation
```

Map an internal activation function (squash) to an ONNX op_type.

Parameters:
- `squash` - Activation function reference.

Returns: ONNX activation operator name.

### inferLayerOrdering

```ts
inferLayerOrdering(
  network: default,
): default[][]
```

Infer strictly layered ordering from a network.

Parameters:
- `network` - Source network.

Returns: Ordered layers: input, hidden..., output.

### validateLayerHomogeneityAndConnectivity

```ts
validateLayerHomogeneityAndConnectivity(
  layers: default[][],
  network: default,
  options: OnnxExportOptions,
): void
```

Validate connectivity and activation homogeneity constraints per layer.

Parameters:
- `layers` - Layered node arrays.
- `network` - Source network (reserved for compatibility).
- `options` - Export options.

Returns: Nothing.

### collectUniqueOutgoingConnections

```ts
collectUniqueOutgoingConnections(
  nodes: default[],
): default[]
```

Collect unique outgoing connections across a node list.

Parameters:
- `nodes` - Nodes to traverse.

Returns: Stable array of unique connections.

### normalizeActivationName

```ts
normalizeActivationName(
  squash: ((x: number, derivate?: boolean | undefined) => number) & { name?: string | undefined; },
): string
```

Normalize activation function name to uppercase for token matching.

Parameters:
- `squash` - Runtime activation function reference.

Returns: Uppercased activation name or empty string.

### resolveOnnxActivationOperation

```ts
resolveOnnxActivationOperation(
  normalizedActivationName: string,
): OnnxActivationOperation
```

Resolve ONNX activation op from a normalized activation name token.

Parameters:
- `normalizedActivationName` - Uppercased activation name.

Returns: ONNX activation operation.

### warnWhenActivationFallbackIsUsed

```ts
warnWhenActivationFallbackIsUsed(
  context: { squash: ((x: number, derivate?: boolean | undefined) => number) & { name?: string | undefined; }; resolvedActivationOperation: OnnxActivationOperation; },
): void
```

Emit a warning when activation export falls back to Identity.

Parameters:
- `context` - Activation fallback evaluation context.

Returns: Nothing.

### collectLayerOrderingNodeGroups

```ts
collectLayerOrderingNodeGroups(
  network: default,
): LayerOrderingNodeGroups
```

Partition all network nodes into input/hidden/output groups.

Parameters:
- `network` - Source network.

Returns: Node groups used by layered-ordering inference.

### filterNodesByType

```ts
filterNodesByType(
  nodes: default[],
  nodeType: string,
): default[]
```

Filter nodes by one expected node type.

Parameters:
- `nodes` - Candidate node list.
- `nodeType` - Expected node type.

Returns: Matching nodes.

### hasNoHiddenNodes

```ts
hasNoHiddenNodes(
  nodeGroups: LayerOrderingNodeGroups,
): boolean
```

Check whether the layer groups contain no hidden nodes.

Parameters:
- `nodeGroups` - Partitioned node groups.

Returns: True when hidden layer traversal can be skipped.

### finalizeOrderingWithoutHiddenNodes

```ts
finalizeOrderingWithoutHiddenNodes(
  nodeGroups: LayerOrderingNodeGroups,
): default[][]
```

Finalize ordering for networks without hidden layers.

Parameters:
- `nodeGroups` - Partitioned node groups.

Returns: Input and output layers only.

### initializeLayerOrderingResolutionContext

```ts
initializeLayerOrderingResolutionContext(
  nodeGroups: LayerOrderingNodeGroups,
): LayerOrderingResolutionContext
```

Create initial hidden-layer resolution context.

Parameters:
- `nodeGroups` - Partitioned node groups.

Returns: Initial mutable state for hidden-layer resolution.

### resolveAllHiddenLayers

```ts
resolveAllHiddenLayers(
  initialContext: LayerOrderingResolutionContext,
): LayerOrderingResolutionContext
```

Resolve all hidden layers in dependency order.

Parameters:
- `initialContext` - Starting hidden-layer resolution context.

Returns: Final resolved layer-ordering context.

### resolveNextHiddenLayer

```ts
resolveNextHiddenLayer(
  resolutionContext: LayerOrderingResolutionContext,
): LayerOrderingResolutionContext
```

Resolve the next hidden layer from unresolved candidates.

Parameters:
- `resolutionContext` - Current resolution state.

Returns: Updated resolution state.

### collectCurrentResolvableHiddenLayer

```ts
collectCurrentResolvableHiddenLayer(
  resolutionContext: LayerOrderingResolutionContext,
): default[]
```

Collect unresolved hidden nodes that can be placed in the next layer.

Parameters:
- `resolutionContext` - Current hidden-layer resolution context.

Returns: Hidden nodes that are resolvable in this pass.

### hasAllIncomingConnectionsFromPreviousLayer

```ts
hasAllIncomingConnectionsFromPreviousLayer(
  context: { hiddenNode: default; previousLayerNodes: default[]; },
): boolean
```

Check whether a hidden node receives all inputs from the previous layer.

Parameters:
- `context` - Hidden-node connectivity check context.

Returns: True when the hidden node is layer-resolvable.

### ensureLayerWasResolved

```ts
ensureLayerWasResolved(
  currentLayerNodes: default[],
): void
```

Ensure current hidden-layer resolution pass produced at least one node.

Parameters:
- `currentLayerNodes` - Nodes resolved for current layer.

Returns: Nothing.

### filterUnresolvedHiddenNodes

```ts
filterUnresolvedHiddenNodes(
  context: { remainingHiddenNodes: default[]; currentLayerNodes: default[]; },
): default[]
```

Remove just-resolved hidden nodes from unresolved candidates.

Parameters:
- `context` - Remaining/just-resolved hidden node context.

Returns: Hidden nodes still unresolved.

### appendLastResolvedLayer

```ts
appendLastResolvedLayer(
  resolutionContext: LayerOrderingResolutionContext,
): LayerOrderingResolutionContext
```

Append the final resolved hidden layer into ordered layer output.

Parameters:
- `resolutionContext` - Final traversal state before append.

Returns: Traversal state with last hidden layer persisted.

### finalizeOrderingWithOutputLayer

```ts
finalizeOrderingWithOutputLayer(
  context: { orderedLayers: default[][]; outputNodes: default[]; },
): default[][]
```

Append output layer to resolved input/hidden ordering.

Parameters:
- `context` - Final ordering context.

Returns: Full layer ordering including output layer.

### buildLayerValidationContexts

```ts
buildLayerValidationContexts(
  layers: default[][],
  options: OnnxExportOptions,
): LayerValidationTraversalContext[]
```

Build per-layer validation contexts for all non-input layers.

Parameters:
- `layers` - Ordered network layers.
- `options` - ONNX export options.

Returns: Traversal contexts used by layer validators.

### validateSingleLayer

```ts
validateSingleLayer(
  layerValidationContext: LayerValidationTraversalContext,
): void
```

Validate one current layer against activation/connectivity constraints.

Parameters:
- `layerValidationContext` - Layer validation context.

Returns: Nothing.

### createLayerActivationValidationContext

```ts
createLayerActivationValidationContext(
  layerValidationContext: LayerValidationTraversalContext,
): LayerActivationValidationContext
```

Create activation validation context from one layer traversal context.

Parameters:
- `layerValidationContext` - Layer validation context.

Returns: Activation validation context.

### validateLayerActivationHomogeneity

```ts
validateLayerActivationHomogeneity(
  activationValidationContext: LayerActivationValidationContext,
): void
```

Validate that a layer has homogeneous activation unless explicitly allowed.

Parameters:
- `activationValidationContext` - Activation validation context.

Returns: Nothing.

### validateLayerConnectivity

```ts
validateLayerConnectivity(
  layerValidationContext: LayerValidationTraversalContext,
): void
```

Validate that each current-layer node has required incoming connectivity.

Parameters:
- `layerValidationContext` - Layer connectivity traversal context.

Returns: Nothing.

### validateTargetNodeConnectivity

```ts
validateTargetNodeConnectivity(
  context: { targetNode: default; previousLayerNodes: default[]; layerIndex: number; allowPartialConnectivity: boolean; },
): void
```

Validate full source coverage for one target node.

Parameters:
- `context` - Target-node connectivity context.

Returns: Nothing.

### validateSourceToTargetConnectivity

```ts
validateSourceToTargetConnectivity(
  connectivityValidationContext: LayerConnectivityValidationContext,
): void
```

Validate one source->target connection pair under export constraints.

Parameters:
- `connectivityValidationContext` - Source/target connectivity context.

Returns: Nothing.

## architecture/network/onnx/network.onnx.export-recurrent.utils.ts

### emitRecurrentLayer

```ts
emitRecurrentLayer(
  params: RecurrentLayerEmissionParams,
): string
```

Emit recurrent single-step layer representation.

Parameters:
- `params` - Recurrent emission parameters.

Returns: Output tensor name.

### buildRecurrentLayerEmissionContext

```ts
buildRecurrentLayerEmissionContext(
  params: RecurrentLayerEmissionParams,
): RecurrentLayerEmissionContext
```

Build derived recurrent-layer context from input params.

Parameters:
- `params` - User-provided recurrent layer params.

Returns: Derived context with cached dimensions and layer slot.

### buildRecurrentInitializerNames

```ts
buildRecurrentInitializerNames(
  context: RecurrentLayerEmissionContext,
): RecurrentInitializerNames
```

Build deterministic tensor names for recurrent initializer emission.

Parameters:
- `context` - Recurrent layer execution context.

Returns: Tensor-name group for initializer emission.

### buildRecurrentGraphNames

```ts
buildRecurrentGraphNames(
  context: RecurrentLayerEmissionContext,
): RecurrentGraphNames
```

Build deterministic graph names for recurrent-node emission.

Parameters:
- `context` - Recurrent layer execution context.

Returns: Graph-name group for branch and activation nodes.

### collectRecurrentInitializerValues

```ts
collectRecurrentInitializerValues(
  context: RecurrentLayerEmissionContext,
): RecurrentInitializerValues
```

Collect recurrent initializer vectors for one layer.

Parameters:
- `context` - Recurrent layer execution context.

Returns: Dense and recurrent initializer vectors.

### emitRecurrentInitializers

```ts
emitRecurrentInitializers(
  context: RecurrentInitializerEmissionContext,
): void
```

Emit dense and recurrent initializer tensors.

Parameters:
- `context` - Initializer emission context.

Returns: Nothing.

### buildInputBranchGemmEmissionContext

```ts
buildInputBranchGemmEmissionContext(
  context: RecurrentLayerEmissionContext,
  initializerNames: RecurrentInitializerNames,
  graphNames: RecurrentGraphNames,
): RecurrentGemmEmissionContext
```

Build Gemm emission context for the feed-forward branch.

Parameters:
- `context` - Recurrent layer execution context.
- `initializerNames` - Recurrent initializer names.
- `graphNames` - Recurrent graph names.

Returns: Gemm emission context.

### buildRecurrentBranchGemmEmissionContext

```ts
buildRecurrentBranchGemmEmissionContext(
  context: RecurrentLayerEmissionContext,
  initializerNames: RecurrentInitializerNames,
  graphNames: RecurrentGraphNames,
): RecurrentGemmEmissionContext
```

Build Gemm emission context for the recurrent hidden-state branch.

Parameters:
- `context` - Recurrent layer execution context.
- `initializerNames` - Recurrent initializer names.
- `graphNames` - Recurrent graph names.

Returns: Gemm emission context.

### resolvePreviousHiddenInputName

```ts
resolvePreviousHiddenInputName(
  layerIndex: number,
): string
```

Resolve recurrent branch hidden-state input for one layer.

Parameters:
- `layerIndex` - Current recurrent layer index.

Returns: Hidden-state tensor input name.

### emitRecurrentGemmNode

```ts
emitRecurrentGemmNode(
  context: RecurrentGemmEmissionContext,
): void
```

Emit one recurrent Gemm node with shared ONNX attributes.

Parameters:
- `context` - Gemm emission context.

Returns: Nothing.

### emitRecurrentAddNode

```ts
emitRecurrentAddNode(
  model: OnnxModel,
  graphNames: RecurrentGraphNames,
): void
```

Emit Add node that fuses feed-forward and recurrent branch outputs.

Parameters:
- `model` - Target ONNX model.
- `graphNames` - Deterministic graph names for this layer.

Returns: Nothing.

### emitRecurrentActivationNode

```ts
emitRecurrentActivationNode(
  context: RecurrentActivationEmissionContext,
): void
```

Emit activation node for recurrent branch sum output.

Parameters:
- `context` - Activation emission context.

Returns: Nothing.

### resolveRecurrentActivationType

```ts
resolveRecurrentActivationType(
  currentLayerNodes: default[],
): string
```

Resolve ONNX activation type from first node in recurrent layer.

Parameters:
- `currentLayerNodes` - Current recurrent layer nodes.

Returns: ONNX activation op type.

### readNodeInternals

```ts
readNodeInternals(
  node: default,
): NodeInternals
```

Normalize runtime node shape to recurrent-export internals contract.

Parameters:
- `node` - Runtime node instance.

Returns: Node internals used by ONNX emission helpers.

### buildDefaultGemmAttributes

```ts
buildDefaultGemmAttributes(): { name: string; type: string; f?: number | undefined; i?: number | undefined; }[]
```

Build the shared attribute list for ONNX Gemm node payloads.

Returns: Gemm attribute payload list.

## architecture/network/onnx/network.onnx.export-layer-graph.utils.ts

### emitLayerGraph

```ts
emitLayerGraph(
  context: LayerBuildContext,
): string
```

Emit one export layer graph segment and return the produced output tensor name.

Parameters:
- `context` - Layer build context.

Returns: Output tensor name produced by this layer.

## architecture/network/onnx/network.onnx.export-postprocess.utils.ts

### emitFusedRecurrentHeuristics

```ts
emitFusedRecurrentHeuristics(
  model: OnnxModel,
  layers: default[][],
  allowRecurrent: boolean | undefined,
  previousOutputName: string,
): void
```

Emit heuristic fused recurrent operators (LSTM/GRU) when recurrent export is enabled.

Parameters:
- `model` - Target ONNX model.
- `layers` - Layered network nodes.
- `allowRecurrent` - Whether recurrent export is enabled.
- `previousOutputName` - Current graph output name (kept for backward-compatible emission semantics).

Returns: Nothing.

### finalizeExportMetadata

```ts
finalizeExportMetadata(
  model: OnnxModel,
  layers: default[][],
  options: OnnxExportOptions,
  includeMetadata: boolean,
  hiddenSizesMetadata: number[],
  recurrentLayerIndices: number[],
): void
```

Finalize export metadata and optional conv-sharing validation.

Parameters:
- `model` - Target ONNX model.
- `layers` - Layered network nodes.
- `options` - Export options.
- `includeMetadata` - Whether metadata emission is enabled.
- `hiddenSizesMetadata` - Hidden-layer sizes collected during emission.
- `recurrentLayerIndices` - Recurrent layer indices.

Returns: Nothing.

### tryEmitFusedLstm

```ts
tryEmitFusedLstm(
  context: HiddenLayerHeuristicContext,
): void
```

Try emitting heuristic fused LSTM node and metadata.

### buildFusedLstmExecutionContext

```ts
buildFusedLstmExecutionContext(
  context: LstmEmissionContext,
): FusedRecurrentEmissionExecutionContext
```

Build shared fused-recurrent execution context for LSTM.

### tryEmitFusedGru

```ts
tryEmitFusedGru(
  context: HiddenLayerHeuristicContext,
): void
```

Try emitting heuristic fused GRU node and metadata.

### buildFusedGruExecutionContext

```ts
buildFusedGruExecutionContext(
  context: GruEmissionContext,
): FusedRecurrentEmissionExecutionContext
```

Build shared fused-recurrent execution context for GRU.

### emitFusedRecurrentLayer

```ts
emitFusedRecurrentLayer(
  context: FusedRecurrentEmissionExecutionContext,
): void
```

Emit shared fused recurrent payload (initializers, node, metadata).

### appendIndexMetadata

```ts
appendIndexMetadata(
  model: OnnxModel,
  key: string,
  layerIndex: number,
): void
```

Append a unique layer index to metadata array key.

### findMetadataPropertyIndex

```ts
findMetadataPropertyIndex(
  metadataProperties: OnnxMetadataProperty[],
  key: string,
): number
```

Find metadata property index by key.

### upsertLayerIndexMetadataValue

```ts
upsertLayerIndexMetadataValue(
  metadataProperties: OnnxMetadataProperty[],
  metadataIndex: number,
  layerIndex: number,
): void
```

Upsert one layer index into metadata array-like JSON value.

### parseMetadataLayerIndices

```ts
parseMetadataLayerIndices(
  metadataValue: string,
): number[]
```

Parse metadata JSON value into a numeric layer-index array.

### buildRecurrentHeuristicEmissionContext

```ts
buildRecurrentHeuristicEmissionContext(
  model: OnnxModel,
  layers: default[][],
  previousOutputName: string,
): RecurrentHeuristicEmissionContext
```

Build reusable context for recurrent heuristic traversal.

### collectHiddenLayerIndices

```ts
collectHiddenLayerIndices(
  layers: default[][],
): number[]
```

Collect hidden-layer indices for recurrent traversal.

### buildHiddenLayerHeuristicContext

```ts
buildHiddenLayerHeuristicContext(
  context: RecurrentHeuristicEmissionContext,
  layerIndex: number,
): HiddenLayerHeuristicContext
```

Build one hidden-layer traversal context.

### emitFallbackRecurrentPatternMetadata

```ts
emitFallbackRecurrentPatternMetadata(
  context: HiddenLayerHeuristicContext,
): void
```

Emit fallback metadata for recurrent-size ambiguity.

### isFallbackRecurrentPatternSize

```ts
isFallbackRecurrentPatternSize(
  currentSize: number,
): boolean
```

Check whether hidden size should emit recurrent fallback metadata.

### isEligibleForLstmHeuristic

```ts
isEligibleForLstmHeuristic(
  currentSize: number,
): boolean
```

Check LSTM heuristic eligibility by size and gate divisibility.

### buildLstmEmissionContext

```ts
buildLstmEmissionContext(
  context: HiddenLayerHeuristicContext,
): LstmEmissionContext
```

Build LSTM emission context from one hidden-layer traversal record.

### collectLstmGateNodeGroups

```ts
collectLstmGateNodeGroups(
  context: LstmEmissionContext,
): default[][]
```

Collect LSTM gate node groups in canonical export order.

### isEligibleForGruHeuristic

```ts
isEligibleForGruHeuristic(
  currentSize: number,
): boolean
```

Check GRU heuristic eligibility by size and gate divisibility.

### buildGruEmissionContext

```ts
buildGruEmissionContext(
  context: HiddenLayerHeuristicContext,
): GruEmissionContext
```

Build GRU emission context from one hidden-layer traversal record.

### collectGruGateNodeGroups

```ts
collectGruGateNodeGroups(
  context: GruEmissionContext,
): default[][]
```

Collect GRU gate node groups in canonical export order.

### collectRecurrentGateBlockParameters

```ts
collectRecurrentGateBlockParameters(
  context: RecurrentGateBlockCollectionContext,
): RecurrentGateParameterCollectionResult
```

Collect flattened parameter vectors for one gate node block.

### collectRecurrentGateRow

```ts
collectRecurrentGateRow(
  context: RecurrentGateRowCollectionContext,
): RecurrentGateRow
```

Collect one recurrent gate row payload (inputs, recurrent slice, and bias).

### resolveRecurrentRowWeight

```ts
resolveRecurrentRowWeight(
  context: RecurrentGateRowCollectionContext,
  columnIndex: number,
): number
```

Resolve one recurrent row value at the requested column.

### foldRecurrentGateRows

```ts
foldRecurrentGateRows(
  gateRows: RecurrentGateRow[],
): RecurrentGateParameterCollectionResult
```

Fold recurrent gate rows into flattened ONNX initializer vectors.

### foldRecurrentGateBlocks

```ts
foldRecurrentGateBlocks(
  gateParameterBlocks: RecurrentGateParameterCollectionResult[],
): RecurrentGateParameterCollectionResult
```

Fold gate blocks into a single fused parameter payload.

### buildFusedRecurrentInitializerNames

```ts
buildFusedRecurrentInitializerNames(
  operatorType: "LSTM" | "GRU",
  layerIndex: number,
): FusedRecurrentInitializerNames
```

Build fused recurrent initializer names for the current layer.

### buildFusedRecurrentGraphNames

```ts
buildFusedRecurrentGraphNames(
  nodePrefix: string,
  outputSuffix: string,
  layerIndex: number,
): FusedRecurrentGraphNames
```

Build fused recurrent graph names for node and output.

### appendFusedRecurrentInitializers

```ts
appendFusedRecurrentInitializers(
  model: OnnxModel,
  initializerNames: FusedRecurrentInitializerNames,
  parameters: RecurrentGateParameterCollectionResult,
  gateCount: number,
  unitSize: number,
  previousSize: number,
): void
```

Append fused recurrent initializer tensors to the ONNX graph.

### appendFusedRecurrentNode

```ts
appendFusedRecurrentNode(
  graph: OnnxGraph,
  operatorType: "LSTM" | "GRU",
  previousOutputName: string,
  initializerNames: FusedRecurrentInitializerNames,
  graphNames: FusedRecurrentGraphNames,
  unitSize: number,
): void
```

Append fused recurrent operator node to the ONNX graph.

### resolveGruPreviousOutputName

```ts
resolveGruPreviousOutputName(
  layerIndex: number,
): string
```

Resolve previous output naming semantics for GRU heuristic emission.

### appendRecurrentSingleStepMetadata

```ts
appendRecurrentSingleStepMetadata(
  model: OnnxModel,
  recurrentLayerIndices: number[],
): void
```

Append recurrent single-step metadata when recurrent layers exist.

### shouldValidateConvSharing

```ts
shouldValidateConvSharing(
  options: OnnxExportOptions,
): boolean
```

Determine whether Conv2D sharing validation is enabled and configured.

### validateConvSharingAcrossMappings

```ts
validateConvSharingAcrossMappings(
  context: ConvSharingValidationContext,
): ConvSharingValidationResult
```

Validate Conv2D sharing across all declared Conv mappings.

### resolveConvLayerPairContext

```ts
resolveConvLayerPairContext(
  layers: default[][],
  layerIndex: number,
  convSpec: Conv2DMapping,
): ConvLayerPairContext | undefined
```

Resolve one Conv mapping layer pair or return undefined for invalid layout.

### isConvLayerPairConsistent

```ts
isConvLayerPairConsistent(
  context: ConvLayerPairContext,
): boolean
```

Validate one Conv layer pair against representative kernel sharing.

### appendConvLayerValidationResult

```ts
appendConvLayerValidationResult(
  result: ConvSharingValidationResult,
  layerIndex: number,
  isConsistent: boolean,
): void
```

Append one Conv-layer validation outcome and optional warning.

### appendConvSharingMetadata

```ts
appendConvSharingMetadata(
  model: OnnxModel,
  result: ConvSharingValidationResult,
): void
```

Append Conv-sharing validation metadata arrays.

### collectRepresentativeKernels

```ts
collectRepresentativeKernels(
  context: ConvLayerPairContext,
): number[][]
```

Collect representative kernels for each output channel.

### collectRepresentativeKernelForChannel

```ts
collectRepresentativeKernelForChannel(
  context: ConvRepresentativeKernelContext,
): number[]
```

Collect one representative kernel by reading the first output position for a channel.

### collectConvOutputCoordinates

```ts
collectConvOutputCoordinates(
  convSpec: Conv2DMapping,
): ConvOutputCoordinate[]
```

Collect output coordinates for full Conv traversal.

### collectConvKernelCoordinates

```ts
collectConvKernelCoordinates(
  convSpec: Conv2DMapping,
): OnnxConvKernelCoordinate[]
```

Collect kernel coordinates for one Conv kernel traversal.

### isOutputCoordinateConsistent

```ts
isOutputCoordinateConsistent(
  context: ConvLayerPairContext,
  outputCoordinate: ConvOutputCoordinate,
  representativeKernels: number[][],
  tolerance: number,
): boolean
```

Validate one output coordinate against channel representative kernel weights.

### resolveNeuronInternalAtOutputCoordinate

```ts
resolveNeuronInternalAtOutputCoordinate(
  context: ConvLayerPairContext,
  outputCoordinate: ConvOutputCoordinate,
): NodeInternals | undefined
```

Resolve runtime internals for output coordinate neuron, if present.

### isKernelCoordinateConsistent

```ts
isKernelCoordinateConsistent(
  context: ConvKernelConsistencyContext,
): boolean
```

Validate one kernel coordinate against its representative channel value.

### resolveInputPosition

```ts
resolveInputPosition(
  context: ConvKernelConsistencyContext,
): { inputRow: number; inputColumn: number; }
```

Resolve input row/column projected by output and kernel coordinates.

### isInputPositionInsideBounds

```ts
isInputPositionInsideBounds(
  convSpec: Conv2DMapping,
  inputRow: number,
  inputColumn: number,
): boolean
```

Check whether input row/column falls inside Conv input bounds.

### resolveSourceNodeAtInputPosition

```ts
resolveSourceNodeAtInputPosition(
  convSpec: Conv2DMapping,
  previousLayerNodes: default[],
  inChannelIndex: number,
  inputRow: number,
  inputColumn: number,
): default | undefined
```

Resolve source node by Conv input position coordinates.

### collectRepresentativeKernelWeight

```ts
collectRepresentativeKernelWeight(
  convSpec: Conv2DMapping,
  previousLayerNodes: default[],
  representativeInternal: NodeInternals,
  kernelCoordinate: OnnxConvKernelCoordinate,
): number
```

Collect representative kernel value using top-left receptive field indexing.

### areWeightsWithinTolerance

```ts
areWeightsWithinTolerance(
  context: WeightToleranceComparisonContext,
): boolean
```

Compare two scalar weights using configured tolerance.

### asNodeInternals

```ts
asNodeInternals(
  node: default,
): NodeInternals
```

Resolve runtime node internals in one typed helper.

### resolveIncomingWeight

```ts
resolveIncomingWeight(
  targetNodeInternal: NodeInternals,
  sourceNode: default,
): number
```

Resolve incoming connection weight from a specific source node.

### resolveSelfConnectionWeight

```ts
resolveSelfConnectionWeight(
  targetNodeInternal: NodeInternals,
): number
```

Resolve self-connection weight for diagonal recurrent matrix entries.

### buildMetadataProperty

```ts
buildMetadataProperty(
  key: string,
  value: unknown,
): OnnxMetadataProperty
```

Build a metadata key/value property with JSON string serialization.

### appendMetadataProperty

```ts
appendMetadataProperty(
  model: OnnxModel,
  metadataProperty: OnnxMetadataProperty,
): void
```

Append metadata property to model metadata_props list.

### ensureMetadataProps

```ts
ensureMetadataProps(
  model: OnnxModel,
): OnnxMetadataProperty[]
```

Ensure metadata_props array exists and return it.

## architecture/network/onnx/network.onnx.import-activations.utils.ts

### assignActivationFunctions

```ts
assignActivationFunctions(
  network: default,
  onnx: OnnxModel,
  hiddenLayerSizes: number[],
): void
```

Assign node activation functions from ONNX activation nodes.

Parameters:
- `network` - Target network to mutate.
- `onnx` - Source ONNX model.
- `hiddenLayerSizes` - Hidden layer size list.

Returns: Nothing.

## architecture/network/onnx/network.onnx.export-layer-common.utils.ts

### buildDenseWeightsAndBiases

```ts
buildDenseWeightsAndBiases(
  previousLayerNodes: default[],
  currentLayerNodes: default[],
): DenseWeightBuildResult
```

Build dense-layer weight matrix and bias vector.

Parameters:
- `previousLayerNodes` - Source layer nodes.
- `currentLayerNodes` - Destination layer nodes.

Returns: Flattened row-major weight matrix and bias vector.

### buildDiagonalRecurrentWeights

```ts
buildDiagonalRecurrentWeights(
  currentLayerNodes: default[],
): number[]
```

Build a diagonal recurrent matrix from self-connections.

Parameters:
- `currentLayerNodes` - Layer nodes.

Returns: Flattened row-major recurrent matrix.

### emitOptionalPoolingAndFlatten

```ts
emitOptionalPoolingAndFlatten(
  params: OptionalPoolingAndFlattenParams,
): string
```

Emit optional pooling and flatten nodes after a layer output.

Parameters:
- `params` - Pooling parameters.

Returns: Final output tensor name after optional pooling/flatten.

### appendIndexedMetadata

```ts
appendIndexedMetadata(
  model: OnnxModel,
  key: string,
  layerIndex: number,
): void
```

Append an integer index to JSON-array metadata key.

Parameters:
- `model` - Target model.
- `key` - Metadata key.
- `layerIndex` - Layer index to append.

Returns: Nothing.

### appendMetadataSpec

```ts
appendMetadataSpec(
  model: OnnxModel,
  key: string,
  spec: Conv2DMapping | Pool2DMapping,
): void
```

Append a JSON object to JSON-array metadata key.

Parameters:
- `model` - Target model.
- `key` - Metadata key.
- `spec` - Metadata object.

Returns: Nothing.

### collectDenseRows

```ts
collectDenseRows(
  context: DenseWeightBuildContext,
): DenseWeightRow[]
```

Collect dense rows for each target node in current layer.

Parameters:
- `context` - Dense row collection context.

Returns: Dense rows containing per-target weights and bias.

### foldDenseRowsToInitializers

```ts
foldDenseRowsToInitializers(
  denseRows: DenseWeightRow[],
): DenseWeightBuildResult
```

Fold dense rows into flattened ONNX initializer arrays.

Parameters:
- `denseRows` - Dense rows.

Returns: Flattened dense initializer result.

### collectDenseRowWeights

```ts
collectDenseRowWeights(
  context: DenseWeightRowCollectionContext,
): number[]
```

Collect source-to-target weights for one dense row.

Parameters:
- `context` - Dense row collection context.

Returns: Row weights in source-node order.

### resolveInboundWeight

```ts
resolveInboundWeight(
  targetNodeInternal: NodeInternals,
  sourceNode: default,
): number
```

Resolve source-to-target inbound connection weight.

Parameters:
- `targetNodeInternal` - Target node internals.
- `sourceNode` - Source node.

Returns: Inbound weight or zero for disconnected edges.

### asNodeInternals

```ts
asNodeInternals(
  node: default,
): NodeInternals
```

Normalize a public node instance into ONNX export internals.

Parameters:
- `node` - Source node.

Returns: Internal runtime-facing node representation.

### collectRecurrentRows

```ts
collectRecurrentRows(
  context: DiagonalRecurrentBuildContext,
): number[][]
```

Collect recurrent matrix rows for one layer.

Parameters:
- `context` - Recurrent matrix build context.

Returns: Recurrent row collection.

### collectRecurrentRow

```ts
collectRecurrentRow(
  context: RecurrentRowCollectionContext,
): number[]
```

Collect one recurrent matrix row.

Parameters:
- `context` - Row collection context.

Returns: Recurrent row values.

### resolveDiagonalRecurrentWeight

```ts
resolveDiagonalRecurrentWeight(
  context: RecurrentRowCollectionContext,
  columnIndex: number,
): number
```

Resolve recurrent weight value for one matrix coordinate.

Parameters:
- `context` - Row collection context.
- `columnIndex` - Column index in row.

Returns: Recurrent weight for diagonal entries, otherwise zero.

### toPoolingEmissionContext

```ts
toPoolingEmissionContext(
  params: OptionalPoolingAndFlattenParams,
): PoolingEmissionContext
```

Resolve pooling emission context from optional pooling parameters.

Parameters:
- `params` - Optional pooling and flatten parameters.

Returns: Pooling emission context.

### emitPoolingNode

```ts
emitPoolingNode(
  context: PoolingEmissionContext,
): string
```

Emit one pooling node and return its output tensor name.

Parameters:
- `context` - Pooling emission context.

Returns: Pooling output tensor name.

### collectPoolingAttributes

```ts
collectPoolingAttributes(
  poolSpec: Pool2DMapping,
): PoolingAttributes
```

Collect ONNX pooling attributes from one pooling spec.

Parameters:
- `poolSpec` - Pooling spec.

Returns: Pooling attributes for ONNX node payload.

### emitOptionalFlattenAfterPooling

```ts
emitOptionalFlattenAfterPooling(
  context: FlattenAfterPoolingContext,
): string
```

Conditionally emit flatten node after pooling.

Parameters:
- `context` - Flatten emission context.

Returns: Output tensor name after optional flatten.

### appendPoolingMetadata

```ts
appendPoolingMetadata(
  context: PoolingEmissionContext,
): void
```

Append pooling metadata for one emitted pooling layer.

Parameters:
- `context` - Pooling emission context.

Returns: Nothing.

### ensureMetadataRegistry

```ts
ensureMetadataRegistry(
  model: OnnxModel,
): OnnxMetadataProperty[]
```

Ensure model metadata registry exists.

Parameters:
- `model` - Target model.

Returns: Mutable metadata registry.

### findMetadataProperty

```ts
findMetadataProperty(
  metadataRegistry: OnnxMetadataProperty[],
  key: string,
): OnnxMetadataProperty | undefined
```

Find a metadata property by key.

Parameters:
- `metadataRegistry` - Metadata registry.
- `key` - Metadata key.

Returns: Matching metadata property if present.

### buildIndexedMetadataProperty

```ts
buildIndexedMetadataProperty(
  key: string,
  layerIndex: number,
): OnnxMetadataProperty
```

Build a new index-array metadata property.

Parameters:
- `key` - Metadata key.
- `layerIndex` - Layer index.

Returns: Metadata property.

### buildSpecMetadataProperty

```ts
buildSpecMetadataProperty(
  key: string,
  spec: Conv2DMapping | Pool2DMapping,
): OnnxMetadataProperty
```

Build a new spec-array metadata property.

Parameters:
- `key` - Metadata key.
- `spec` - Mapping spec.

Returns: Metadata property.

### serializeIndexedMetadataValue

```ts
serializeIndexedMetadataValue(
  currentValue: string,
  layerIndex: number,
): string
```

Serialize index metadata after appending one unique index.

Parameters:
- `currentValue` - Existing JSON value.
- `layerIndex` - Layer index.

Returns: Serialized JSON value.

### serializeSpecMetadataValue

```ts
serializeSpecMetadataValue(
  currentValue: string,
  spec: Conv2DMapping | Pool2DMapping,
): string
```

Serialize spec metadata after appending one spec object.

Parameters:
- `currentValue` - Existing JSON value.
- `spec` - Mapping spec.

Returns: Serialized JSON value.

### parseMetadataArray

```ts
parseMetadataArray(
  metadataValue: string,
): ItemType[] | undefined
```

Parse a metadata JSON array value safely.

Parameters:
- `metadataValue` - Metadata JSON string.

Returns: Parsed array when valid, otherwise undefined.

## architecture/network/onnx/network.onnx.export-orchestrators.utils.ts

### assignExportNodeIndices

```ts
assignExportNodeIndices(
  network: default,
): void
```

Assign stable index values to nodes for export diagnostics.

Parameters:
- `network` - Source network.

Returns: Nothing.

### collectLstmPatternStubs

```ts
collectLstmPatternStubs(
  layers: default[][],
  allowRecurrent: boolean | undefined,
): LstmPatternStub[]
```

Collect heuristic LSTM grouping stubs from hidden layers.

Parameters:
- `layers` - Layered network nodes.
- `allowRecurrent` - Whether recurrent export heuristics are enabled.

Returns: Candidate LSTM pattern stubs.

### appendConvInferenceMetadata

```ts
appendConvInferenceMetadata(
  model: OnnxModel,
  layers: default[][],
  options: OnnxExportOptions,
): void
```

Append heuristic conv inference metadata when requested.

Parameters:
- `model` - Target ONNX model.
- `layers` - Layered network nodes.
- `options` - Export options.

Returns: Nothing.

### appendLstmPatternStubMetadata

```ts
appendLstmPatternStubMetadata(
  model: OnnxModel,
  lstmPatternStubs: LstmPatternStub[],
): void
```

Append LSTM pattern stub metadata.

Parameters:
- `model` - Target ONNX model.
- `lstmPatternStubs` - Pattern stubs.

Returns: Nothing.

### createExportNodeIndexAssignmentContexts

```ts
createExportNodeIndexAssignmentContexts(
  network: default,
): ExportNodeIndexAssignmentContext[]
```

Create node/index assignment contexts for export diagnostics.

Parameters:
- `network` - Source network.

Returns: Assignment contexts.

### applyExportNodeIndexAssignments

```ts
applyExportNodeIndexAssignments(
  assignmentContexts: ExportNodeIndexAssignmentContext[],
): void
```

Apply prepared node/index assignment contexts.

Parameters:
- `assignmentContexts` - Prepared contexts.

Returns: Nothing.

### applySingleExportNodeIndexAssignment

```ts
applySingleExportNodeIndexAssignment(
  assignmentContext: ExportNodeIndexAssignmentContext,
): void
```

Apply one export index assignment.

Parameters:
- `assignmentContext` - Assignment context.

Returns: Nothing.

### safelyCollectLstmPatternStubs

```ts
safelyCollectLstmPatternStubs(
  layers: default[][],
): LstmPatternStub[]
```

Collect LSTM pattern stubs with heuristic error isolation.

Parameters:
- `layers` - Layered network nodes.

Returns: LSTM pattern stubs.

### collectLstmPatternStubsFromLayers

```ts
collectLstmPatternStubsFromLayers(
  layers: default[][],
): LstmPatternStub[]
```

Collect LSTM pattern stubs from hidden layers.

Parameters:
- `layers` - Layered network nodes.

Returns: LSTM pattern stubs.

### createHiddenLayerTraversalContexts

```ts
createHiddenLayerTraversalContexts(
  layers: default[][],
): LstmLayerTraversalContext[]
```

Create traversal contexts for hidden layers only.

Parameters:
- `layers` - Layered network nodes.

Returns: Hidden layer contexts.

### createLstmCandidateContext

```ts
createLstmCandidateContext(
  hiddenLayerContext: LstmLayerTraversalContext,
): LstmCandidateContext
```

Build LSTM candidate context for one hidden layer.

Parameters:
- `hiddenLayerContext` - Hidden layer context.

Returns: LSTM candidate context.

### isValidLstmCandidateContext

```ts
isValidLstmCandidateContext(
  candidateContext: LstmCandidateContext,
): boolean
```

Determine whether a candidate context satisfies heuristic LSTM conditions.

Parameters:
- `candidateContext` - Candidate context.

Returns: True when the candidate is a valid LSTM stub.

### mapLstmCandidateToStub

```ts
mapLstmCandidateToStub(
  candidateContext: LstmCandidateContext,
): LstmPatternStub
```

Map a valid candidate context to metadata stub.

Parameters:
- `candidateContext` - Valid candidate context.

Returns: LSTM pattern stub.

### hasRequiredSelfConnectionCount

```ts
hasRequiredSelfConnectionCount(
  nodeItem: default,
): boolean
```

Check whether one node has the required self-connection count.

Parameters:
- `nodeItem` - Node to inspect.

Returns: True when self-connection count matches requirement.

### collectInferredConvMetadata

```ts
collectInferredConvMetadata(
  context: { layers: default[][]; declaredMappings: Conv2DMapping[] | undefined; },
): ConvInferenceResult
```

Collect inferred Conv metadata from hidden-layer traversals.

Parameters:
- `context` - Conv traversal context.

Returns: Inferred Conv metadata result.

### createConvTraversalContexts

```ts
createConvTraversalContexts(
  context: { layers: default[][]; declaredMappings: Conv2DMapping[] | undefined; },
): ConvInferenceTraversalContext[]
```

Create Conv traversal contexts for hidden layers.

Parameters:
- `context` - Conv traversal source context.

Returns: Conv traversal contexts.

### resolveConvInferenceForLayer

```ts
resolveConvInferenceForLayer(
  traversalContext: ConvInferenceTraversalContext,
): (Conv2DMapping & { note?: string | undefined; }) | undefined
```

Resolve inferred Conv specification for one hidden layer.

Parameters:
- `traversalContext` - Conv traversal context.

Returns: Inferred Conv specification when matched.

### createConvInferenceEvaluationContext

```ts
createConvInferenceEvaluationContext(
  traversalContext: ConvInferenceTraversalContext,
): ConvInferenceEvaluationContext
```

Create width/square-evaluation context for Conv inference.

Parameters:
- `traversalContext` - Conv traversal context.

Returns: Conv evaluation context.

### resolveConvSpecFromKernelCandidates

```ts
resolveConvSpecFromKernelCandidates(
  evaluationContext: ConvInferenceEvaluationContext,
): (Conv2DMapping & { note?: string | undefined; }) | undefined
```

Resolve Conv specification using ordered kernel candidates.

Parameters:
- `evaluationContext` - Conv evaluation context.

Returns: Inferred Conv specification when matched.

### resolveConvSpecForKernel

```ts
resolveConvSpecForKernel(
  kernelContext: ConvInferenceKernelEvaluationContext,
): (Conv2DMapping & { note?: string | undefined; }) | undefined
```

Resolve Conv specification for one kernel candidate.

Parameters:
- `kernelContext` - Kernel-evaluation context.

Returns: Inferred Conv specification when matched.

### isDeclaredConvLayer

```ts
isDeclaredConvLayer(
  traversalContext: ConvInferenceTraversalContext,
): boolean
```

Check whether a traversal layer already has declared Conv mapping.

Parameters:
- `traversalContext` - Conv traversal context.

Returns: True when mapping is already declared.

### isInferredConvSpec

```ts
isInferredConvSpec(
  specification: (Conv2DMapping & { note?: string | undefined; }) | undefined,
): boolean
```

Type guard for inferred Conv specifications.

Parameters:
- `specification` - Conv specification candidate.

Returns: True when specification is defined.

### hasInferredConvMetadata

```ts
hasInferredConvMetadata(
  inferenceResult: ConvInferenceResult,
): boolean
```

Check whether inferred Conv metadata exists.

Parameters:
- `inferenceResult` - Inferred Conv result.

Returns: True when inferred metadata exists.

### appendMetadataProperties

```ts
appendMetadataProperties(
  model: OnnxModel,
  metadataProperties: OnnxMetadataProperty[],
): void
```

Append metadata properties in a single, normalized path.

Parameters:
- `model` - Target ONNX model.
- `metadataProperties` - Metadata properties to append.

Returns: Nothing.

## architecture/network/onnx/network.onnx.import-orchestrators.utils.ts

### extractOnnxArchitecture

```ts
extractOnnxArchitecture(
  onnx: OnnxModel,
): OnnxImportArchitectureResult
```

Extract input/output counts and hidden layer sizes from ONNX model.

Parameters:
- `onnx` - Source ONNX model.

Returns: Parsed architecture dimensions.

### pruneSingleLayerHiddenPlaceholders

```ts
pruneSingleLayerHiddenPlaceholders(
  network: default,
  hiddenLayerSizes: number[],
): void
```

Remove placeholder hidden nodes for single-layer perceptron imports.

Parameters:
- `network` - Target network.
- `hiddenLayerSizes` - Hidden layer sizes.

Returns: Nothing.

### restoreRecurrentSelfConnections

```ts
restoreRecurrentSelfConnections(
  network: default,
  onnx: OnnxModel,
  hiddenLayerSizes: number[],
  metadata: OnnxMetadataProperty[],
): void
```

Restore recurrent self-connections from recurrent metadata and R tensors.

Parameters:
- `network` - Target network.
- `onnx` - Source ONNX model.
- `hiddenLayerSizes` - Hidden layer sizes.
- `metadata` - Parsed metadata properties.

Returns: Nothing.

### attachOnnxPoolingMetadata

```ts
attachOnnxPoolingMetadata(
  network: default,
  metadata: OnnxMetadataProperty[],
): void
```

Attach optional pooling metadata from ONNX model to network instance.

Parameters:
- `network` - Target network.
- `metadata` - ONNX metadata.

Returns: Nothing.

### reconstructFusedRecurrentLayers

```ts
reconstructFusedRecurrentLayers(
  network: default,
  onnx: OnnxModel,
  hiddenLayerSizes: number[],
  layerFactory: OnnxLayerFactory,
  metadata: OnnxMetadataProperty[],
): void
```

Reconstruct emitted fused LSTM/GRU layers from ONNX metadata and initializers.

Parameters:
- `network` - Target network.
- `onnx` - Source ONNX model.
- `hiddenLayerSizes` - Hidden layer sizes.
- `layerFactory` - Dynamic layer module.
- `metadata` - ONNX metadata properties.

Returns: Nothing.

### parseRecurrentLayerIndices

```ts
parseRecurrentLayerIndices(
  rawMetadataValue: string,
): number[]
```

Parse recurrent layer indices metadata.

Parameters:
- `rawMetadataValue` - Raw metadata JSON string.

Returns: Normalized recurrent layer indices.

### applyLayerSelfConnections

```ts
applyLayerSelfConnections(
  layerConnectionContext: OnnxImportLayerConnectionContext,
): void
```

Apply one hidden layer diagonal recurrent self-weights.

Parameters:
- `layerConnectionContext` - Layer connection context.

Returns: Nothing.

### buildArchitectureContext

```ts
buildArchitectureContext(
  onnx: OnnxModel,
): OnnxImportArchitectureContext
```

Build architecture extraction context from ONNX graph state.

Parameters:
- `onnx` - Source ONNX model.

Returns: Normalized architecture extraction context.

### readLastDimensionValue

```ts
readLastDimensionValue(
  dimensions: { dim_value?: number | undefined; }[],
): number
```

Read the terminal ONNX shape dimension value from one shape array.

Parameters:
- `dimensions` - ONNX shape dimensions.

Returns: Terminal `dim_value` payload.

### isSingleLayerPerceptronImport

```ts
isSingleLayerPerceptronImport(
  hiddenLayerSizes: number[],
): boolean
```

Determine whether import shape corresponds to a single-layer perceptron.

Parameters:
- `hiddenLayerSizes` - Hidden-layer size list.

Returns: True when no hidden layers exist.

### collectPerceptronBoundaryNodes

```ts
collectPerceptronBoundaryNodes(
  nodes: default[],
): default[]
```

Collect input and output boundary nodes for perceptron imports.

Parameters:
- `nodes` - Full network node list.

Returns: Input/output-only node list.

### collectNodesByType

```ts
collectNodesByType(
  nodes: default[],
  nodeType: "input" | "output" | "hidden",
): default[]
```

Collect nodes matching one runtime node-type discriminator.

Parameters:
- `nodes` - Node list.
- `nodeType` - Runtime node type.

Returns: Filtered node list.

### collectRecurrentLayerSpans

```ts
collectRecurrentLayerSpans(
  restorationContext: OnnxImportRecurrentRestorationContext,
): OnnxImportHiddenLayerSpan[]
```

Resolve recurrent-target hidden-layer spans from metadata + hidden sizes.

Parameters:
- `restorationContext` - Recurrent restoration context.

Returns: Hidden-layer spans requiring recurrent restoration.

### resolveRecurrentLayerIndices

```ts
resolveRecurrentLayerIndices(
  metadata: OnnxMetadataProperty[],
): number[]
```

Resolve recurrent layer indices from ONNX metadata.

Parameters:
- `metadata` - ONNX metadata payload.

Returns: Parsed recurrent layer indices.

### findMetadataProperty

```ts
findMetadataProperty(
  metadata: OnnxMetadataProperty[],
  metadataKey: string,
): OnnxMetadataProperty | undefined
```

Find one ONNX metadata property by key.

Parameters:
- `metadata` - ONNX metadata array.
- `metadataKey` - Metadata key.

Returns: Matching metadata property when present.

### buildHiddenLayerSpans

```ts
buildHiddenLayerSpans(
  hiddenLayerSizes: number[],
): OnnxImportHiddenLayerSpan[]
```

Build hidden-layer spans with one-based layer numbering and global offsets.

Parameters:
- `hiddenLayerSizes` - Hidden-layer size list.

Returns: Hidden-layer span payload list.

### normalizeRecurrentLayerIndices

```ts
normalizeRecurrentLayerIndices(
  parsedMetadataValue: string | number | boolean | number[] | Record<string, number> | null,
): number[]
```

Normalize recurrent layer indices parsed from metadata JSON.

Parameters:
- `parsedMetadataValue` - Parsed metadata JSON value.

Returns: Recurrent layer indices.

### findRecurrentInitializer

```ts
findRecurrentInitializer(
  layerConnectionContext: OnnxImportLayerConnectionContext,
): { name: string; float_data: number[]; } | undefined
```

Resolve recurrent initializer tensor for one hidden-layer span.

Parameters:
- `layerConnectionContext` - Layer connection context.

Returns: Recurrent initializer tensor when available.

### sliceLayerHiddenNodes

```ts
sliceLayerHiddenNodes(
  layerConnectionContext: OnnxImportLayerConnectionContext,
): default[]
```

Slice hidden nodes for one hidden-layer span.

Parameters:
- `layerConnectionContext` - Layer connection context.

Returns: Hidden nodes belonging to the span.

### collectDiagonalRecurrentWeights

```ts
collectDiagonalRecurrentWeights(
  recurrentTensorWeights: number[],
  hiddenLayerSize: number,
): number[]
```

Collect diagonal recurrent weights from flattened layer tensor data.

Parameters:
- `recurrentTensorWeights` - Flattened recurrent tensor weights.
- `hiddenLayerSize` - Hidden-layer width.

Returns: Diagonal recurrent self-weights.

### upsertSelfConnection

```ts
upsertSelfConnection(
  selfConnectionContext: OnnxImportSelfConnectionUpsertContext,
): void
```

Upsert one node self-connection for recurrent import restoration.

Parameters:
- `selfConnectionContext` - Self-connection upsert context.

Returns: Nothing.

### parsePoolingMetadata

```ts
parsePoolingMetadata(
  metadata: OnnxMetadataProperty[],
): OnnxImportPoolingMetadata | null
```

Parse pooling metadata payload from ONNX metadata.

Parameters:
- `metadata` - ONNX metadata entries.

Returns: Parsed pooling metadata payload.

### attachParsedPoolingMetadata

```ts
attachParsedPoolingMetadata(
  network: default,
  poolingMetadata: OnnxImportPoolingMetadata,
): void
```

Attach parsed pooling metadata to imported network instance.

Parameters:
- `network` - Target network.
- `poolingMetadata` - Parsed pooling metadata payload.

Returns: Nothing.

## architecture/network/onnx/network.onnx.import-fused-recurrent.utils.ts

### reconstructFusedRecurrentLayers

```ts
reconstructFusedRecurrentLayers(
  network: default,
  onnx: OnnxModel,
  hiddenLayerSizes: number[],
  layerFactory: OnnxLayerFactory,
  metadata: OnnxMetadataProperty[],
): void
```

Reconstruct emitted fused LSTM/GRU layers from ONNX metadata and initializers.

Parameters:
- `network` - Target network.
- `onnx` - Source ONNX model.
- `hiddenLayerSizes` - Hidden layer sizes.
- `layerFactory` - Dynamic layer module.
- `metadata` - ONNX metadata properties.

Returns: Nothing.
