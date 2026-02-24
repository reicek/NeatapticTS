# architecture/network/onnx

## architecture/network/onnx/network.onnx.ts

### network.onnx

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

### Conv2DMapping

Mapping declaration for treating a fully-connected layer as a 2D convolution during export.

This does **not** magically turn an MLP into a convolutional network at runtime.
It annotates a particular export-layer index with a conv interpretation so that:
- The exported graph uses conv-shaped tensors/operators, and
- Import can re-attach pooling/flatten metadata appropriately.

Pitfall: mappings must match the actual layer sizes. If `inHeight * inWidth * inChannels`
does not correspond to the prior layer width (and similarly for outputs), export or import
may reject the model.

### exportToONNX

`(network: import("C:/NeatapticTS/src/architecture/network").default, options: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxExportOptions) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel`

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

`(onnx: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel) => import("C:/NeatapticTS/src/architecture/network").default`

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

### network.onnx.utils.types

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

### ActivationFunction

`(x: number, derivate: boolean | undefined) => number`

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

### ActivationSquashFunction

`(x: number, derivate: boolean | undefined) => number`

Activation function signature used by ONNX layer emission helpers.

### Conv2DMapping

Mapping declaration for treating a fully-connected layer as a 2D convolution during export.

This does **not** magically turn an MLP into a convolutional network at runtime.
It annotates a particular export-layer index with a conv interpretation so that:
- The exported graph uses conv-shaped tensors/operators, and
- Import can re-attach pooling/flatten metadata appropriately.

Pitfall: mappings must match the actual layer sizes. If `inHeight * inWidth * inChannels`
does not correspond to the prior layer width (and similarly for outputs), export or import
may reject the model.

### ConvInferenceEvaluationContext

Width and shape evaluation context used by Conv inference helpers.

### ConvInferenceKernelEvaluationContext

Kernel candidate context for one Conv inference evaluation pass.

### ConvInferenceResult

Collected inferred Conv metadata payload.

### ConvInferenceTraversalContext

Traversal context for one hidden layer during Conv inference.

### ConvKernelConsistencyContext

Context for kernel-coordinate consistency checks at one output position.

### ConvLayerPairContext

Context for one resolved Conv mapping layer pair.

### ConvOutputCoordinate

Coordinate for one Conv output neuron position.

### ConvRepresentativeKernelContext

Context for representative Conv kernel collection per output channel.

### ConvSharingValidationContext

Context for validating Conv sharing across all declared mappings.

### ConvSharingValidationResult

Result of Conv sharing validation across declared mappings.

### DenseActivationContext

Dense activation emission context.

### DenseActivationNodePayload

Strongly typed activation node payload used by dense export helpers.

### DenseGemmNodePayload

Strongly typed Gemm node payload used by dense export helpers.

### DenseGraphNames

Dense graph tensor names.

### DenseInitializerValues

Dense initializer value arrays.

### DenseLayerContext

Dense layer context enriched with resolved activation function.

### DenseLayerParams

Parameters for dense layer emission.

### DenseOrderedNodePayload

Dense node payload union used by ordered append helpers.

### DenseTensorNames

Dense initializer tensor names.

### DenseWeightBuildContext

Context for building dense layer initializers from two adjacent layers.

### DenseWeightBuildResult

Dense layer initializer fold output.

### DenseWeightRow

One collected dense row before fold to flattened initializers.

### DenseWeightRowCollectionContext

Context for collecting one dense row.

### DiagonalRecurrentBuildContext

Context for building a diagonal recurrent matrix from self-connections.

### ExportNodeIndexAssignmentContext

Context for assigning a stable export index to one node.

### FlattenAfterPoolingContext

Flatten emission context after optional pooling.

### FusedRecurrentEmissionExecutionContext

Shared execution context for emitting one fused recurrent layer payload.

### FusedRecurrentGraphNames

Context for ONNX fused recurrent node payload names.

### FusedRecurrentInitializerNames

Context for ONNX fused recurrent initializer names.

### GruEmissionContext

Context for heuristic GRU emission when a layer matches expected shape.

### HiddenLayerActivationTraversalContext

Hidden-layer traversal context for assigning imported activation functions.

### HiddenLayerHeuristicContext

Context for one hidden layer during heuristic recurrent emission.

### IndexedMetadataAppendContext

Append-an-index metadata context for JSON-array metadata keys.

### LayerActivationContext

Activation analysis context for one layer.

### LayerActivationValidationContext

Activation-homogeneity decision context for one current layer.

### LayerBuildContext

Layer build context used while emitting one ONNX graph layer segment.

### LayerConnectivityValidationContext

Connectivity decision context for one source-target node pair.

### LayerOrderingNodeGroups

Node partitions used by ONNX layered-ordering inference traversal.

### LayerOrderingResolutionContext

Mutable traversal state while resolving hidden-layer ordering.

### LayerRecurrentDecisionContext

Context used to decide recurrent emission branch usage.

### LayerTraversalContext

Layer traversal context with adjacent layers and output classification.

### LayerValidationTraversalContext

Layer-wise validation context for activation and connectivity checks.

### LstmCandidateContext

Candidate context for validating one LSTM-like hidden layer pattern.

### LstmEmissionContext

Context for heuristic LSTM emission when a layer matches expected shape.

### LstmLayerTraversalContext

Traversal context for one hidden layer during LSTM stub collection.

### LstmPatternStub

Heuristic LSTM pattern stub for metadata output.

### NetworkWithOnnxImportPooling

Network instance augmented with optional imported ONNX pooling metadata.

### NodeInternals

Runtime interface for accessing node internal properties.

This is intentionally "internal": it exposes mutable fields that the ONNX exporter/importer
needs (connections, bias, squash). Regular library users should generally interact with
the public `Node` API instead.

### NodeInternalsWithExportIndex

Runtime node internals augmented with optional export index metadata.

### OnnxActivationAssignmentContext

Shared activation-assignment context for hidden and output traversal.

### OnnxActivationLayerOperations

Layer-indexed activation operator lookup extracted from ONNX graph nodes.

### OnnxActivationOperation

Supported ONNX activation operators recognized during activation import.

### OnnxActivationOperationResolutionContext

Activation operation resolution context for one neuron or layer default.

### OnnxActivationParseResult

Parsed ONNX activation-node naming payload.

### OnnxAttribute

ONNX node attribute.

### OnnxBaseModelBuildContext

Context for constructing a base ONNX model shell.

### OnnxBuildResolvedOptions

Resolved options used by ONNX model build orchestration.

### OnnxConvEmissionContext

Context used after resolving Conv mapping for one layer.

### OnnxConvEmissionParams

Parameters accepted by Conv layer emission.

### OnnxConvKernelCoordinate

Coordinate for one Conv kernel weight lookup.

### OnnxConvParameters

Flattened Conv parameters for ONNX initializers.

### OnnxConvTensorNames

Tensor names generated for Conv parameters.

### OnnxDimension

ONNX tensor type shape dimension.

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

### OnnxFusedGateApplicationContext

Gate-weight application context for one reconstructed fused layer.

### OnnxFusedGateRowAssignmentContext

Context for assigning one gate-neuron row from flattened ONNX tensors.

### OnnxFusedLayerNeighborhood

Hidden-layer neighborhood slices around a reconstructed fused layer.

### OnnxFusedLayerReconstructionContext

Execution context for one fused recurrent layer reconstruction.

### OnnxFusedLayerRuntime

Runtime interface of a reconstructed fused recurrent layer instance.

### OnnxFusedRecurrentKind

Supported fused recurrent operator families recognized during ONNX import.

### OnnxFusedRecurrentSpec

Fused recurrent family specification used during import reconstruction.

### OnnxFusedTensorPayload

Fused recurrent tensor payload read from ONNX initializers.

### OnnxGraph

### OnnxGraphDimensionBuildContext

Context for constructing input/output ONNX graph dimensions.

### OnnxGraphDimensions

Output dimensions used by ONNX graph input/output value info payloads.

### OnnxImportAggregatedLayerAssignmentContext

Context for assigning aggregated dense tensors for one layer.

### OnnxImportAggregatedNeuronAssignmentContext

Context for assigning one aggregated dense target neuron row.

### OnnxImportArchitectureContext

Shared architecture extraction context with resolved graph dimensions.

### OnnxImportArchitectureResult

Parsed architecture dimensions extracted from ONNX import graph payloads.

### OnnxImportConvCoordinateAssignmentContext

Context for applying Conv weights/bias at one output coordinate.

### OnnxImportConvKernelAssignmentContext

Context for assigning one concrete Conv kernel connection weight.

### OnnxImportConvLayerContext

Context for reconstructing one Conv layer's imported connectivity.

### OnnxImportConvLayerContextBuildParams

Build params for creating one Conv reconstruction layer context.

### OnnxImportConvMetadata

Parsed Conv metadata payload used for optional reconstruction pass.

### OnnxImportConvNodeSlices

Layer node slices used while applying Conv reconstruction assignments.

### OnnxImportConvOutputCoordinate

Coordinate for one Conv output neuron traversal position.

### OnnxImportConvTensorContext

Resolved Conv initializer tensors and dimensions for one layer.

### OnnxImportDimensionRecord

Loose ONNX shape-dimension record used by legacy import payload access.

### OnnxImportHiddenLayerSpan

Hidden-layer span payload with one-based layer numbering and global offset.

### OnnxImportHiddenSizeDerivationContext

Context for deriving hidden layer sizes from initializer tensors and metadata.

### OnnxImportInboundConnectionMap

Inbound connection lookup map keyed by source node for one target neuron.

### OnnxImportLayerConnectionContext

Execution context for assigning one hidden-layer recurrent diagonal tensor.

### OnnxImportLayerNodePair

Node slices for one sequential imported layer assignment pass.

### OnnxImportLayerNodePairBuildParams

Build params for one sequential layer node-pair slice operation.

### OnnxImportLayerTensorNames

Weight tensor names for one imported layer index.

### OnnxImportLayerWeightBucket

Bucketed ONNX dense/per-neuron tensors for one exported layer index.

### OnnxImportPerNeuronAssignmentContext

Context for assigning one per-neuron imported target node.

### OnnxImportPerNeuronLayerAssignmentContext

Context for assigning per-neuron tensors for one layer.

### OnnxImportPoolingMetadata

Parsed pooling metadata payload attached to imported network instances.

### OnnxImportRecurrentRestorationContext

Context for recurrent self-connection restoration from ONNX metadata and tensors.

### OnnxImportSelfConnectionUpsertContext

Context for upserting one hidden node self-connection from recurrent weight.

### OnnxImportWeightAssignmentBuildParams

Build params for creating shared ONNX import weight-assignment context.

### OnnxImportWeightAssignmentContext

Shared weight-assignment context built once per ONNX import.

### OnnxIncomingWeightAssignmentContext

Context for assigning dense incoming weights for one gate-neuron row.

### OnnxLayerEmissionContext

Context for emitting non-input layers during model build.

### OnnxLayerEmissionResult

Result of emitting non-input export layers.

### OnnxLayerFactory

Runtime factory map used to construct dynamic recurrent layer modules.

### OnnxMetadataProperty

Canonical metadata key-value pair used in ONNX model metadata_props.

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

### OnnxModelMetadataContext

Context for applying optional ONNX model metadata.

### OnnxNode

### OnnxPerceptronBuildContext

Build context for mapping ONNX layer sizes into a Neataptic MLP factory call.

### OnnxPerceptronSizeValidationContext

Validation context for perceptron size-list checks during ONNX import.

### OnnxPostProcessingContext

Context for post-processing and export metadata finalization.

### OnnxRecurrentCollectionContext

Context for collecting recurrent layer indices during model build.

### OnnxRecurrentInputValueInfoContext

Context for constructing one recurrent previous-state graph input payload.

### OnnxRecurrentLayerProcessingContext

Execution context for processing one hidden recurrent layer.

### OnnxRecurrentLayerTraversalContext

Traversal context for one hidden layer during recurrent-input collection.

### OnnxRuntimeFactories

Runtime factories consumed during ONNX import network reconstruction.

These factories let the importer reconstruct runtime objects (network + layers)
while keeping the ONNX parser itself mostly pure.

### OnnxRuntimeLayerFactory

`(size: number) => import("C:/NeatapticTS/src/architecture/layer").default`

Runtime layer-constructor signature used for recurrent layer reconstruction.

ONNX import can optionally reconstruct higher-level recurrent layers (like LSTM/GRU)
from exported metadata. This factory provides the concrete layer implementation.

### OnnxRuntimeLayerFactoryMap

Runtime layer module shape widened for fused-recurrent reconstruction wiring.

### OnnxRuntimeLayerModule

Runtime layer module shape consumed by ONNX import orchestration.

This is the minimal set of recurrent factories needed by the importer.

### OnnxRuntimePerceptronFactory

`(sizes: number[]) => import("C:/NeatapticTS/src/architecture/network").default`

Runtime perceptron factory signature used by ONNX import orchestration.

This factory is injected so the ONNX import path can rebuild an MLP without taking a
hard dependency on a specific constructor shape.

### OnnxShape

ONNX tensor type shape.

### OnnxTensor

### OnnxTensorType

ONNX tensor type.

### OnnxValueInfo

ONNX value info (input/output description).

### OptionalLayerOutputParams

Shared parameters for optional pooling/flatten output emission.

### OptionalPoolingAndFlattenParams

Parameters for optional pooling + flatten emission after a layer output.

### OutputLayerActivationContext

Output-layer activation assignment context.

### PerNeuronConcatNodePayload

Per-neuron concat node payload.

### PerNeuronGraphNames

Per-neuron graph tensor names.

### PerNeuronLayerContext

Per-neuron layer context alias.

### PerNeuronLayerParams

Parameters for per-neuron layer emission.

### PerNeuronNodeContext

Per-neuron normalized node context.

### PerNeuronSubgraphContext

Per-neuron subgraph emission context.

### PerNeuronTensorNames

Per-neuron initializer tensor names.

### Pool2DMapping

Mapping describing a pooling operation inserted after a given export-layer index.

This is represented as metadata and optional graph nodes during export.
Import uses it to attach pooling-related runtime metadata back onto the reconstructed
network (when supported).

### PoolingAttributes

Pooling tensor attributes for ONNX node payloads.

### PoolingEmissionContext

Pooling emission context resolved for one layer output.

### RecurrentActivationEmissionContext

Context for selecting and emitting recurrent activation node payload.

### RecurrentGateBlockCollectionContext

Context for collecting one gate parameter block.

### RecurrentGateParameterCollectionResult

Flattened recurrent gate parameter vectors for one fused operator.

### RecurrentGateRow

One recurrent gate row payload before flatten fold.

### RecurrentGateRowCollectionContext

Context for collecting one recurrent gate row (one neuron).

### RecurrentGemmEmissionContext

Context for emitting one Gemm node for recurrent single-step export.

### RecurrentGraphNames

Derived graph names for one recurrent single-step layer payload.

### RecurrentHeuristicEmissionContext

Context for heuristic recurrent operator emission traversal.

### RecurrentInitializerEmissionContext

Context for pushing recurrent initializers into ONNX graph state.

### RecurrentInitializerNames

Initializer tensor names for one single-step recurrent layer.

### RecurrentInitializerValues

Collected initializer vectors for one single-step recurrent layer.

### RecurrentLayerEmissionContext

Derived execution context for single-step recurrent layer emission.

### RecurrentLayerEmissionParams

Parameters for single-step recurrent layer emission.

### RecurrentRowCollectionContext

Context for collecting one recurrent matrix row.

### SharedActivationNodeBuildParams

Shared parameters for constructing an activation node payload.

### SharedGemmNodeBuildParams

Shared parameters for constructing a Gemm node payload.

### SpecMetadataAppendContext

Append-a-spec metadata context for JSON-array metadata keys.

### WeightToleranceComparisonContext

Context for comparing two scalar weights with numeric tolerance.

## architecture/network/onnx/network.onnx.utils.ts

### network.onnx.utils

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

### applyModelMetadata

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModelMetadataContext) => void`

Attach producer and opset metadata to a model when metadata emission is enabled.

Parameters:
- `context` - Metadata application context.

Returns: Nothing.

### assignActivationFunctions

`(network: import("C:/NeatapticTS/src/architecture/network").default, onnx: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel, hiddenLayerSizes: number[]) => void`

Assign node activation functions from ONNX activation nodes.

Parameters:
- `network` - Target network to mutate.
- `onnx` - Source ONNX model.
- `hiddenLayerSizes` - Hidden layer size list.

Returns: Nothing.

### assignWeightsAndBiases

`(network: import("C:/NeatapticTS/src/architecture/network").default, onnx: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel, hiddenLayerSizes: number[], metadataProps: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxMetadataProperty[] | undefined) => void`

Assign weights and biases from ONNX initializers to a newly created network.

Parameters:
- `network` - Target network to mutate.
- `onnx` - Source ONNX model.
- `hiddenLayerSizes` - Hidden layer sizes.
- `metadataProps` - Optional ONNX metadata properties.

Returns: Nothing.

### buildOnnxModel

`(network: import("C:/NeatapticTS/src/architecture/network").default, layers: import("C:/NeatapticTS/src/architecture/node").default[][], options: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxExportOptions) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel`

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

### collectRecurrentLayerIndices

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxRecurrentCollectionContext) => number[]`

Detect hidden layers with self-recurrence and add matching previous-state graph inputs.

Parameters:
- `context` - Recurrent collection context.

Returns: Export-layer indices with recurrent self-connections.

### createBaseModel

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxBaseModelBuildContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel`

Create the base ONNX model shell with graph input/output declarations.

Parameters:
- `context` - Base model build context.

Returns: Initialized ONNX model with empty initializer/node lists.

### createGraphDimensions

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxGraphDimensionBuildContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxGraphDimensions`

Build tensor dimensions for model input and output, optionally with symbolic batch dimension.

Parameters:
- `context` - Dimension construction context.

Returns: Input and output dimension arrays for ONNX value info.

### deriveHiddenLayerSizes

`(initializers: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxTensor[], metadataProps: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxMetadataProperty[] | undefined) => number[]`

Extract hidden layer sizes from ONNX initializers (weight tensors).

Parameters:
- `initializers` - ONNX initializer tensors.
- `metadataProps` - Optional ONNX metadata properties.

Returns: Hidden layer sizes in order.

### emitFusedRecurrentHeuristics

`(model: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel, layers: import("C:/NeatapticTS/src/architecture/node").default[][], allowRecurrent: boolean | undefined, previousOutputName: string) => void`

Emit heuristic fused recurrent operators (LSTM/GRU) when recurrent export is enabled.

Parameters:
- `model` - Target ONNX model.
- `layers` - Layered network nodes.
- `allowRecurrent` - Whether recurrent export is enabled.
- `previousOutputName` - Current graph output name (kept for backward-compatible emission semantics).

Returns: Nothing.

### emitLayerGraph

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LayerBuildContext) => string`

Emit one export layer graph segment and return the produced output tensor name.

Parameters:
- `context` - Layer build context.

Returns: Output tensor name produced by this layer.

### finalizeExportMetadata

`(model: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel, layers: import("C:/NeatapticTS/src/architecture/node").default[][], options: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxExportOptions, includeMetadata: boolean, hiddenSizesMetadata: number[], recurrentLayerIndices: number[]) => void`

Finalize export metadata and optional conv-sharing validation.

Parameters:
- `model` - Target ONNX model.
- `layers` - Layered network nodes.
- `options` - Export options.
- `includeMetadata` - Whether metadata emission is enabled.
- `hiddenSizesMetadata` - Hidden-layer sizes collected during emission.
- `recurrentLayerIndices` - Recurrent layer indices.

Returns: Nothing.

### inferLayerOrdering

`(network: import("C:/NeatapticTS/src/architecture/network").default) => import("C:/NeatapticTS/src/architecture/node").default[][]`

Infer strictly layered ordering from a network.

Parameters:
- `network` - Source network.

Returns: Ordered layers: input, hidden..., output.

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

### rebuildConnectionsLocal

`(networkLike: import("C:/NeatapticTS/src/architecture/network").default) => void`

Rebuild the network's flat connections array from each node's outgoing list.

Parameters:
- `networkLike` - Network-like instance to mutate.

Returns: Nothing.

### runOnnxExportFlow

`(network: import("C:/NeatapticTS/src/architecture/network").default, options: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxExportOptions) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel`

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

`(onnx: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel) => import("C:/NeatapticTS/src/architecture/network").default`

Execute the complete ONNX import flow and reconstruct a runtime network.

High-level behavior:
 1. Extract architecture dimensions and build a perceptron scaffold.
 2. Restore dense parameters and activation functions.
 3. Reconstruct recurrent/pooling metadata and rebuild connection caches.

Parameters:
- `onnx` - ONNX-like model payload to reconstruct.

Returns: Reconstructed network instance.

### validateLayerHomogeneityAndConnectivity

`(layers: import("C:/NeatapticTS/src/architecture/node").default[][], network: import("C:/NeatapticTS/src/architecture/network").default, options: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxExportOptions) => void`

Validate connectivity and activation homogeneity constraints per layer.

Parameters:
- `layers` - Layered node arrays.
- `network` - Source network (reserved for compatibility).
- `options` - Export options.

Returns: Nothing.

## architecture/network/onnx/network.onnx.export-conv.utils.ts

### tryEmitConvLayer

`(params: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxConvEmissionParams) => string | undefined`

Try to emit a conv-mapped layer.

Parameters:
- `params` - Conv emission parameters.

Returns: New output tensor name when handled, otherwise undefined.

## architecture/network/onnx/network.onnx.export-flow.utils.ts

### runOnnxExportFlow

`(network: import("C:/NeatapticTS/src/architecture/network").default, options: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxExportOptions) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel`

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

`(onnx: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel) => import("C:/NeatapticTS/src/architecture/network").default`

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

`(network: import("C:/NeatapticTS/src/architecture/network").default, layers: import("C:/NeatapticTS/src/architecture/node").default[][], options: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxExportOptions) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel`

Construct ONNX graph (initializers + nodes) from validated layered network structure.

Parameters:
- `network` - Source network (retained for API compatibility).
- `layers` - Layered nodes including input and output layers.
- `options` - Export options.

Returns: ONNX model.

## architecture/network/onnx/network.onnx.export-dense.utils.ts

### appendDenseBiasInitializer

`(layerContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").DenseLayerContext, biasTensorName: string, biasVector: number[]) => void`

Append dense bias initializer.

Parameters:
- `layerContext` - Dense layer context.
- `biasTensorName` - Bias tensor name.
- `biasVector` - Bias vector values.

Returns: Nothing.

### appendDenseNodes

`(model: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel, orderedNodes: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").DenseOrderedNodePayload[]) => void`

Append ordered dense nodes to the model graph.

Parameters:
- `model` - Target model.
- `orderedNodes` - Ordered dense nodes.

Returns: Nothing.

### appendDenseWeightInitializer

`(layerContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").DenseLayerContext, weightTensorName: string, weightMatrixValues: number[]) => void`

Append dense weight initializer.

Parameters:
- `layerContext` - Dense layer context.
- `weightTensorName` - Weight tensor name.
- `weightMatrixValues` - Weight values.

Returns: Nothing.

### buildSingleNeuronWeightRow

`(targetNodeInternal: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").NodeInternals, previousLayerNodes: import("C:/NeatapticTS/src/architecture/node").default[]) => number[]`

Build one neuron's incoming weight row against previous layer.

Parameters:
- `targetNodeInternal` - Target node internals.
- `previousLayerNodes` - Previous layer nodes.

Returns: Weight row values.

### collectDenseInitializerValues

`(layerContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").DenseLayerContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").DenseInitializerValues`

Collect dense weight matrix and bias vector values.

Parameters:
- `layerContext` - Dense layer context.

Returns: Dense initializer values.

### createActivationNode

`(denseActivationContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").DenseActivationContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").DenseActivationNodePayload`

Create dense activation node definition.

Parameters:
- `denseActivationContext` - Dense activation context.

Returns: ONNX activation node payload.

### createDefaultGemmAttributes

`() => { name: string; type: string; f?: number | undefined; i?: number | undefined; }[]`

Build default Gemm attributes for ONNX export.

Returns: Default Gemm attribute list.

### createDenseTensorNames

`(layerIndex: number) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").DenseTensorNames`

Build dense tensor names for initializer emission.

Parameters:
- `layerIndex` - Layer index.

Returns: Dense tensor names.

### createGemmNode

`(denseActivationContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").DenseActivationContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").DenseGemmNodePayload`

Create dense Gemm node definition.

Parameters:
- `denseActivationContext` - Dense activation context.

Returns: ONNX Gemm node payload.

### createSharedActivationNodePayload

`(params: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").SharedActivationNodeBuildParams) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").DenseActivationNodePayload`

Build a shared activation node payload.

Parameters:
- `params` - Shared activation build parameters.

Returns: Activation node payload.

### createSharedGemmNodePayload

`(params: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").SharedGemmNodeBuildParams) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").DenseGemmNodePayload`

Build a shared Gemm node payload.

Parameters:
- `params` - Shared Gemm build parameters.

Returns: Gemm node payload.

### emitDenseActivationSubgraph

`(model: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel, denseActivationContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").DenseActivationContext) => void`

Emit Gemm and activation nodes using requested ordering.

Parameters:
- `model` - Target ONNX model.
- `denseActivationContext` - Dense activation context.

Returns: Nothing.

### emitDenseInitializers

`(layerContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").DenseLayerContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").DenseTensorNames`

Emit dense initializers and return tensor names.

Parameters:
- `layerContext` - Dense layer context.

Returns: Tensor names.

### emitDenseLayer

`(params: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").DenseLayerParams) => string`

Emit dense layer representation.

Parameters:
- `params` - Dense emission parameters.

Returns: Output tensor name.

### emitOptionalLayerOutput

`(params: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OptionalLayerOutputParams) => string`

Emit optional pooling and flatten output fold.

Parameters:
- `params` - Optional output parameters.

Returns: Output tensor name.

### emitPerNeuronLayer

`(params: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").PerNeuronLayerParams) => string`

Emit per-neuron decomposition layer representation.

Parameters:
- `params` - Per-neuron emission parameters.

Returns: Output tensor name.

### emitPerNeuronSubgraph

`(perNeuronSubgraphContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").PerNeuronSubgraphContext) => string`

Emit per-neuron Gemm + activation subgraph.

Parameters:
- `perNeuronSubgraphContext` - Per-neuron subgraph context.

Returns: Per-neuron activation output name.

### resolveDenseNodeOrder

`(gemmNode: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").DenseGemmNodePayload, activationNode: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").DenseActivationNodePayload, legacyNodeOrdering: boolean) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").DenseOrderedNodePayload[]`

Resolve dense node order for legacy and current exports.

Parameters:
- `gemmNode` - Gemm node.
- `activationNode` - Activation node.
- `legacyNodeOrdering` - Whether legacy ordering is required.

Returns: Ordered node list.

### resolveSingleNeuronInboundWeight

`(targetNodeInternal: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").NodeInternals, sourceNode: import("C:/NeatapticTS/src/architecture/node").default) => number`

Resolve one inbound connection weight for a source node.

Parameters:
- `targetNodeInternal` - Target node internals.
- `sourceNode` - Source node.

Returns: Inbound weight or zero when missing.

## architecture/network/onnx/network.onnx.export-setup.utils.ts

### appendRecurrentGraphInput

`(model: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel, traversalContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxRecurrentLayerTraversalContext) => void`

Append one recurrent previous-state graph input for a hidden layer.

Parameters:
- `model` - Target ONNX model.
- `traversalContext` - Hidden layer traversal context.

Returns: Nothing.

### appendRecurrentLayerIndex

`(recurrentLayerIndices: number[], traversalContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxRecurrentLayerTraversalContext) => void`

Append one recurrent layer index to the collected index list.

Parameters:
- `recurrentLayerIndices` - Collected recurrent layer indices.
- `traversalContext` - Hidden layer traversal context.

Returns: Nothing.

### applyModelMetadata

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModelMetadataContext) => void`

Attach producer and opset metadata to a model when metadata emission is enabled.

Parameters:
- `context` - Metadata application context.

Returns: Nothing.

### collectRecurrentLayerIndices

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxRecurrentCollectionContext) => number[]`

Detect hidden layers with self-recurrence and add matching previous-state graph inputs.

Parameters:
- `context` - Recurrent collection context.

Returns: Export-layer indices with recurrent self-connections.

### createBaseModel

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxBaseModelBuildContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel`

Create the base ONNX model shell with graph input/output declarations.

Parameters:
- `context` - Base model build context.

Returns: Initialized ONNX model with empty initializer/node lists.

### createGraphDimensions

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxGraphDimensionBuildContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxGraphDimensions`

Build tensor dimensions for model input and output, optionally with symbolic batch dimension.

Parameters:
- `context` - Dimension construction context.

Returns: Input and output dimension arrays for ONNX value info.

### createGraphValueInfo

`(valueName: string, dimensions: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxDimension[]) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxValueInfo`

Create ONNX value info payload for one graph boundary tensor.

Parameters:
- `valueName` - Tensor value name.
- `dimensions` - Tensor dimensions.

Returns: ONNX value info payload.

### createHiddenLayerIndices

`(totalLayerCount: number) => number[]`

Build hidden layer indices excluding input and output layers.

Parameters:
- `totalLayerCount` - Total number of network layers.

Returns: Hidden layer indices.

### createHiddenLayerTraversalContexts

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxRecurrentCollectionContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxRecurrentLayerTraversalContext[]`

Build traversal contexts for all hidden layers.

Parameters:
- `context` - Recurrent collection context.

Returns: Hidden layer traversal contexts.

### createRecurrentInputValueInfo

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxRecurrentInputValueInfoContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxValueInfo`

Build one recurrent previous-state graph input payload.

Parameters:
- `context` - Recurrent input value-info context.

Returns: ONNX value info payload for recurrent state input.

### createRecurrentInputValueInfoContext

`(traversalContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxRecurrentLayerTraversalContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxRecurrentInputValueInfoContext`

Build recurrent input context for one hidden recurrent layer.

Parameters:
- `traversalContext` - Hidden layer traversal context.

Returns: Recurrent input value-info context.

### createTensorDimensions

`(width: number, batchDimension: boolean) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxDimension[]`

Build one tensor shape dimension payload for dense vectors.

Parameters:
- `width` - Vector width.
- `batchDimension` - Whether symbolic batch dimension is enabled.

Returns: ONNX dimensions for the vector payload.

### hasLayerSelfRecurrence

`(hiddenLayerNodes: import("C:/NeatapticTS/src/architecture/node").default[]) => boolean`

Detect whether a hidden layer contains at least one self-recurrent node.

Parameters:
- `hiddenLayerNodes` - Hidden layer nodes.

Returns: True when any node has a self-connection.

### isRecurrentCollectionEnabled

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxRecurrentCollectionContext) => boolean`

Determine whether recurrent layer collection should execute.

Parameters:
- `context` - Recurrent collection context.

Returns: True when recurrent collection is enabled.

### processHiddenLayerRecurrence

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxRecurrentLayerProcessingContext) => void`

Process one hidden layer for recurrent self-connections.

Parameters:
- `context` - Hidden layer recurrent processing context.

Returns: Nothing.

## architecture/network/onnx/network.onnx.runtime-load.utils.ts

### buildPerceptronNetwork

`(buildContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxPerceptronBuildContext) => import("C:/NeatapticTS/src/architecture/network").default`

Build a perceptron network from size-extraction context.

Parameters:
- `buildContext` - Perceptron build context.

Returns: Reconstructed network instance.

### createPerceptronBuildContext

`(sizes: number[]) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxPerceptronBuildContext`

Build perceptron-network construction context.

Parameters:
- `sizes` - Layer-size payload.

Returns: Build context.

### createPerceptronFactory

`() => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxRuntimePerceptronFactory`

Create an ONNX import network factory from modern static constructors.

Returns: Perceptron-compatible factory function.

### createPerceptronSizeValidationContext

`(sizes: number[]) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxPerceptronSizeValidationContext`

Build perceptron-size validation context.

Parameters:
- `sizes` - Layer-size payload.

Returns: Validation context.

### createRuntimeLayerModule

`() => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxRuntimeLayerModule`

Create the runtime layer-module wiring used by ONNX import orchestrators.

Returns: Runtime recurrent-layer module object.

### foldRuntimeFactories

`(perceptronFactory: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxRuntimePerceptronFactory, layerModule: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxRuntimeLayerModule) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxRuntimeFactories`

Fold runtime perceptron and layer module into a transport payload.

Parameters:
- `perceptronFactory` - Perceptron factory function.
- `layerModule` - Runtime recurrent-layer constructors.

Returns: Runtime factories payload.

### foldRuntimeLayerModule

`(lstmFactory: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxRuntimeLayerFactory, gruFactory: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxRuntimeLayerFactory) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxRuntimeLayerModule`

Fold LSTM/GRU factories into a runtime layer module payload.

Parameters:
- `lstmFactory` - Runtime LSTM layer factory.
- `gruFactory` - Runtime GRU layer factory.

Returns: Runtime layer module.

### loadRuntimeFactories

`() => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxRuntimeFactories`

Resolve runtime factories used by ONNX import orchestration.

Returns: Perceptron factory and layer module object.

### resolveLayerFactory

`(layerKey: keyof import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxRuntimeLayerModule) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxRuntimeLayerFactory`

Resolve one runtime layer factory by module key.

Parameters:
- `layerKey` - Runtime layer key.

Returns: Matching layer factory.

### validatePerceptronSizes

`(validationContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxPerceptronSizeValidationContext) => void`

Validate perceptron size-list constraints.

Parameters:
- `validationContext` - Validation context.

Returns: Nothing. Throws on invalid size-list.

## architecture/network/onnx/network.onnx.import-weights.utils.ts

### applyAggregatedLayerWeights

`(aggregatedContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportAggregatedLayerAssignmentContext) => void`

Apply aggregated dense tensor assignments for one layer.

Parameters:
- `aggregatedContext` - Aggregated assignment context.

Returns: Nothing.

### applyAggregatedNeuronAssignment

`(neuronContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportAggregatedNeuronAssignmentContext) => void`

Apply aggregated dense row weights and bias for one target neuron.

Parameters:
- `neuronContext` - Aggregated neuron assignment context.

Returns: Nothing.

### applyConvCoordinateAssignment

`(coordinateContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportConvCoordinateAssignmentContext) => void`

Apply Conv bias and kernel weights for one output coordinate.

Parameters:
- `coordinateContext` - Conv coordinate assignment context.

Returns: Nothing.

### applyConvLayerReconstruction

`(layerContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportConvLayerContext) => void`

Apply Conv reconstruction for one validated Conv layer context.

Parameters:
- `layerContext` - Conv layer context.

Returns: Nothing.

### applyDenseWeightAssignments

`(assignmentContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportWeightAssignmentContext) => void`

Apply dense/per-neuron assignments for all sorted layer indices.

Parameters:
- `assignmentContext` - Shared assignment context.

Returns: Nothing.

### applyOptionalConvReconstruction

`(assignmentContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportWeightAssignmentContext) => void`

Apply optional Conv2D reconstruction pass from metadata payloads.

Parameters:
- `assignmentContext` - Shared assignment context.

Returns: Nothing.

### applyPerNeuronAssignment

`(perNeuronAssignmentContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportPerNeuronAssignmentContext) => void`

Apply one per-neuron weight vector and bias assignment.

Parameters:
- `perNeuronAssignmentContext` - Per-neuron assignment context.

Returns: Nothing.

### applyPerNeuronLayerWeights

`(perNeuronContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportPerNeuronLayerAssignmentContext) => void`

Apply per-neuron tensor assignments for one layer.

Parameters:
- `perNeuronContext` - Per-neuron layer assignment context.

Returns: Nothing.

### assignConvKernelWeight

`(kernelAssignmentContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportConvKernelAssignmentContext) => void`

Assign one Conv kernel weight to the matching inbound neuron connection.

Parameters:
- `kernelAssignmentContext` - Conv kernel assignment context.

Returns: Nothing.

### assignLayerWeights

`(initializerMap: Record<string, import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxTensor>, nodePair: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportLayerNodePair) => void`

Assign one layer's weights using aggregated or per-neuron tensors.

Parameters:
- `initializerMap` - ONNX initializer map.
- `nodePair` - Current/previous node slices.

Returns: Nothing.

### assignWeightsAndBiases

`(network: import("C:/NeatapticTS/src/architecture/network").default, onnx: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel, hiddenLayerSizes: number[], metadataProps: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxMetadataProperty[] | undefined) => void`

Assign weights and biases from ONNX initializers to a newly created network.

Parameters:
- `network` - Target network to mutate.
- `onnx` - Source ONNX model.
- `hiddenLayerSizes` - Hidden layer sizes.
- `metadataProps` - Optional ONNX metadata properties.

Returns: Nothing.

### buildConvLayerContext

`(params: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportConvLayerContextBuildParams) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportConvLayerContext | null`

Build one Conv layer reconstruction context.

Parameters:
- `params` - Conv context input params.

Returns: Conv layer context when valid.

### buildConvNeuronLinearIndex

`(coordinate: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportConvOutputCoordinate, convSpec: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").Conv2DMapping) => number`

Build flattened linear index for one Conv output coordinate.

Parameters:
- `coordinate` - Conv output coordinate.
- `convSpec` - Conv mapping spec.

Returns: Linear neuron index.

### buildConvNodeSlices

`(layerContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportConvLayerContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportConvNodeSlices`

Build Conv current/previous node slices for one layer context.

Parameters:
- `layerContext` - Conv layer context.

Returns: Node slice payload.

### buildConvTensorContext

`(layerContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportConvLayerContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportConvTensorContext | null`

Build validated Conv tensor context for one layer.

Parameters:
- `layerContext` - Conv layer context.

Returns: Conv tensor context when valid.

### buildHiddenLayerSizesFromBuckets

`(layerWeightBuckets: Record<string, import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportLayerWeightBucket>, sortedLayerIndices: number[]) => number[]`

Build hidden-layer sizes from weight buckets while excluding output layer.

Parameters:
- `layerWeightBuckets` - Layer-weight buckets.
- `sortedLayerIndices` - Ascending layer indices.

Returns: Hidden-layer sizes.

### buildInboundConnectionMap

`(neuronInternal: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").NodeInternals) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportInboundConnectionMap`

Build inbound connection lookup map for one neuron.

Parameters:
- `neuronInternal` - Neuron internals.

Returns: Inbound connection map keyed by source node.

### buildInitializerMap

`(initializers: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxTensor[]) => Record<string, import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxTensor>`

Build ONNX initializer map keyed by tensor name.

Parameters:
- `initializers` - ONNX initializer list.

Returns: Tensor map by name.

### buildInputCoordinate

`(kernelAssignmentContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportConvKernelAssignmentContext) => { inputRow: number; inputColumn: number; } | null`

Build input-space coordinate for one Conv kernel element.

Parameters:
- `kernelAssignmentContext` - Conv kernel assignment context.

Returns: Input coordinate when in bounds.

### buildInputFeatureLinearIndex

`(convSpec: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").Conv2DMapping, inChannelIndex: number, inputRow: number, inputColumn: number) => number`

Build linear feature index in input feature space.

Parameters:
- `convSpec` - Conv mapping spec.
- `inChannelIndex` - Input channel index.
- `inputRow` - Input row index.
- `inputColumn` - Input column index.

Returns: Linear input feature index.

### buildLayerNodePair

`(assignmentContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportWeightAssignmentContext, params: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportLayerNodePairBuildParams) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportLayerNodePair`

Build current/previous node slices for one sequential import layer pass.

Parameters:
- `assignmentContext` - Shared assignment context.
- `params` - Sequential traversal params.

Returns: Layer node pair.

### buildLayerTensorNames

`(layerIndex: number) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportLayerTensorNames`

Build dense weight/bias tensor names for one layer index.

Parameters:
- `layerIndex` - Export layer index.

Returns: Layer tensor names.

### buildPerNeuronTensorNames

`(layerIndex: number, neuronIndex: number) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportLayerTensorNames`

Build per-neuron tensor names for one layer and neuron index.

Parameters:
- `layerIndex` - Export layer index.
- `neuronIndex` - Neuron index in layer.

Returns: Per-neuron tensor names.

### buildWeightAssignmentContext

`(params: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportWeightAssignmentBuildParams) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportWeightAssignmentContext`

Build the shared assignment context for import weight restoration.

Parameters:
- `params` - Assignment context input params.

Returns: Shared assignment context.

### collectConvKernelCoordinates

`(inChannels: number, kernelHeight: number, kernelWidth: number) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxConvKernelCoordinate[]`

Collect all kernel traversal coordinates for one Conv output position.

Parameters:
- `inChannels` - Input channel count.
- `kernelHeight` - Kernel height.
- `kernelWidth` - Kernel width.

Returns: Kernel traversal coordinates.

### collectConvOutputCoordinates

`(convSpec: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").Conv2DMapping, outChannels: number) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportConvOutputCoordinate[]`

Collect all output traversal coordinates for one Conv layer.

Parameters:
- `convSpec` - Conv mapping spec.
- `outChannels` - Output channel count.

Returns: Output traversal coordinates.

### collectLayerWeightBuckets

`(initializers: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxTensor[]) => Record<string, import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportLayerWeightBucket>`

Collect ONNX weight tensor buckets grouped by export layer index.

Parameters:
- `initializers` - ONNX initializer tensors.

Returns: Layer-weight buckets keyed by export layer index.

### collectNodesByType

`(nodes: import("C:/NeatapticTS/src/architecture/node").default[], nodeType: "input" | "output" | "hidden") => import("C:/NeatapticTS/src/architecture/node").default[]`

Collect nodes by runtime node type discriminator.

Parameters:
- `nodes` - Network nodes.
- `nodeType` - Runtime node type.

Returns: Filtered nodes.

### collectSortedLayerIndices

`(layerWeightBuckets: Record<string, import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportLayerWeightBucket>) => number[]`

Collect sorted layer indices from weight buckets.

Parameters:
- `layerWeightBuckets` - Layer-weight buckets.

Returns: Ascending export layer indices.

### collectSortedUniqueLayerIndices

`(initializerMap: Record<string, import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxTensor>) => number[]`

Collect unique sorted layer indices from initializer weight tensors.

Parameters:
- `initializerMap` - Initializer map keyed by tensor name.

Returns: Unique sorted layer indices.

### deriveHiddenLayerSizes

`(initializers: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxTensor[], metadataProps: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxMetadataProperty[] | undefined) => number[]`

Extract hidden layer sizes from ONNX initializers (weight tensors).

Parameters:
- `initializers` - ONNX initializer tensors.
- `metadataProps` - Optional ONNX metadata properties.

Returns: Hidden layer sizes in order.

### hasAggregatedLayerWeights

`(aggregatedContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportAggregatedLayerAssignmentContext) => boolean`

Determine whether the layer has aggregated weight tensor data.

Parameters:
- `aggregatedContext` - Aggregated assignment context.

Returns: True when aggregated tensor exists.

### parseConvMetadata

`(metadataProps: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxMetadataProperty[]) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportConvMetadata | null`

Parse Conv reconstruction metadata payload.

Parameters:
- `metadataProps` - ONNX metadata properties.

Returns: Parsed Conv metadata.

### parseLayerIndexFromWeightTensor

`(tensorName: string) => number | null`

Parse layer index from dense/per-neuron weight tensor name.

Parameters:
- `tensorName` - Tensor name.

Returns: Parsed layer index or null.

### parseMetadataLayerSizes

`(metadataProps: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxMetadataProperty[]) => number[] | null`

Parse explicit metadata-driven hidden layer sizes.

Parameters:
- `metadataProps` - ONNX metadata properties.

Returns: Parsed hidden layer sizes when available.

### parseWeightTensorName

`(tensorName: string) => { layerIndex: string; neuronIndex: number | null; } | null`

Parse layer/neuron components from a weight tensor name.

Parameters:
- `tensorName` - Tensor name.

Returns: Parsed layer+neuron components when matched.

### readConvKernelWeight

`(kernelAssignmentContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportConvKernelAssignmentContext) => number`

Read one Conv kernel weight from flattened ONNX tensor payload.

Parameters:
- `kernelAssignmentContext` - Conv kernel assignment context.

Returns: Kernel weight.

### resolveCurrentLayerNodes

`(assignmentContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportWeightAssignmentContext, params: { sequentialIndex: number; }) => import("C:/NeatapticTS/src/architecture/node").default[]`

Resolve current layer nodes for one sequential layer assignment pass.

Parameters:
- `assignmentContext` - Shared assignment context.
- `params` - Sequential traversal params.

Returns: Current layer nodes.

### resolveLayerHiddenSize

`(layerWeightBuckets: Record<string, import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportLayerWeightBucket>, layerIndex: number) => number`

Resolve one hidden-layer size from its weight bucket.

Parameters:
- `layerWeightBuckets` - Layer-weight buckets.
- `layerIndex` - Export layer index.

Returns: Hidden-layer size.

### resolvePreviousLayerNodes

`(assignmentContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportWeightAssignmentContext, params: { sequentialIndex: number; }) => import("C:/NeatapticTS/src/architecture/node").default[]`

Resolve previous layer nodes for one sequential layer assignment pass.

Parameters:
- `assignmentContext` - Shared assignment context.
- `params` - Sequential traversal params.

Returns: Previous layer nodes.

### sumHiddenSizesToIndex

`(hiddenLayerSizes: number[], exclusiveEndIndex: number) => number`

Sum hidden-layer sizes from index `0` to `exclusiveEndIndex`.

Parameters:
- `hiddenLayerSizes` - Hidden-layer size list.
- `exclusiveEndIndex` - Exclusive end index.

Returns: Prefix sum.

## architecture/network/onnx/network.onnx.layer-analysis.utils.ts

### appendLastResolvedLayer

`(resolutionContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LayerOrderingResolutionContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LayerOrderingResolutionContext`

Append the final resolved hidden layer into ordered layer output.

Parameters:
- `resolutionContext` - Final traversal state before append.

Returns: Traversal state with last hidden layer persisted.

### buildLayerValidationContexts

`(layers: import("C:/NeatapticTS/src/architecture/node").default[][], options: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxExportOptions) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LayerValidationTraversalContext[]`

Build per-layer validation contexts for all non-input layers.

Parameters:
- `layers` - Ordered network layers.
- `options` - ONNX export options.

Returns: Traversal contexts used by layer validators.

### collectCurrentResolvableHiddenLayer

`(resolutionContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LayerOrderingResolutionContext) => import("C:/NeatapticTS/src/architecture/node").default[]`

Collect unresolved hidden nodes that can be placed in the next layer.

Parameters:
- `resolutionContext` - Current hidden-layer resolution context.

Returns: Hidden nodes that are resolvable in this pass.

### collectLayerOrderingNodeGroups

`(network: import("C:/NeatapticTS/src/architecture/network").default) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LayerOrderingNodeGroups`

Partition all network nodes into input/hidden/output groups.

Parameters:
- `network` - Source network.

Returns: Node groups used by layered-ordering inference.

### collectUniqueOutgoingConnections

`(nodes: import("C:/NeatapticTS/src/architecture/node").default[]) => import("C:/NeatapticTS/src/architecture/connection").default[]`

Collect unique outgoing connections across a node list.

Parameters:
- `nodes` - Nodes to traverse.

Returns: Stable array of unique connections.

### createLayerActivationValidationContext

`(layerValidationContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LayerValidationTraversalContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LayerActivationValidationContext`

Create activation validation context from one layer traversal context.

Parameters:
- `layerValidationContext` - Layer validation context.

Returns: Activation validation context.

### ensureLayerWasResolved

`(currentLayerNodes: import("C:/NeatapticTS/src/architecture/node").default[]) => void`

Ensure current hidden-layer resolution pass produced at least one node.

Parameters:
- `currentLayerNodes` - Nodes resolved for current layer.

Returns: Nothing.

### filterNodesByType

`(nodes: import("C:/NeatapticTS/src/architecture/node").default[], nodeType: string) => import("C:/NeatapticTS/src/architecture/node").default[]`

Filter nodes by one expected node type.

Parameters:
- `nodes` - Candidate node list.
- `nodeType` - Expected node type.

Returns: Matching nodes.

### filterUnresolvedHiddenNodes

`(context: { remainingHiddenNodes: import("C:/NeatapticTS/src/architecture/node").default[]; currentLayerNodes: import("C:/NeatapticTS/src/architecture/node").default[]; }) => import("C:/NeatapticTS/src/architecture/node").default[]`

Remove just-resolved hidden nodes from unresolved candidates.

Parameters:
- `context` - Remaining/just-resolved hidden node context.

Returns: Hidden nodes still unresolved.

### finalizeOrderingWithoutHiddenNodes

`(nodeGroups: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LayerOrderingNodeGroups) => import("C:/NeatapticTS/src/architecture/node").default[][]`

Finalize ordering for networks without hidden layers.

Parameters:
- `nodeGroups` - Partitioned node groups.

Returns: Input and output layers only.

### finalizeOrderingWithOutputLayer

`(context: { orderedLayers: import("C:/NeatapticTS/src/architecture/node").default[][]; outputNodes: import("C:/NeatapticTS/src/architecture/node").default[]; }) => import("C:/NeatapticTS/src/architecture/node").default[][]`

Append output layer to resolved input/hidden ordering.

Parameters:
- `context` - Final ordering context.

Returns: Full layer ordering including output layer.

### hasAllIncomingConnectionsFromPreviousLayer

`(context: { hiddenNode: import("C:/NeatapticTS/src/architecture/node").default; previousLayerNodes: import("C:/NeatapticTS/src/architecture/node").default[]; }) => boolean`

Check whether a hidden node receives all inputs from the previous layer.

Parameters:
- `context` - Hidden-node connectivity check context.

Returns: True when the hidden node is layer-resolvable.

### hasNoHiddenNodes

`(nodeGroups: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LayerOrderingNodeGroups) => boolean`

Check whether the layer groups contain no hidden nodes.

Parameters:
- `nodeGroups` - Partitioned node groups.

Returns: True when hidden layer traversal can be skipped.

### inferLayerOrdering

`(network: import("C:/NeatapticTS/src/architecture/network").default) => import("C:/NeatapticTS/src/architecture/node").default[][]`

Infer strictly layered ordering from a network.

Parameters:
- `network` - Source network.

Returns: Ordered layers: input, hidden..., output.

### initializeLayerOrderingResolutionContext

`(nodeGroups: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LayerOrderingNodeGroups) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LayerOrderingResolutionContext`

Create initial hidden-layer resolution context.

Parameters:
- `nodeGroups` - Partitioned node groups.

Returns: Initial mutable state for hidden-layer resolution.

### mapActivationToOnnx

`(squash: ((x: number, derivate?: boolean | undefined) => number) & { name?: string | undefined; }) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxActivationOperation`

Map an internal activation function (squash) to an ONNX op_type.

Parameters:
- `squash` - Activation function reference.

Returns: ONNX activation operator name.

### normalizeActivationName

`(squash: ((x: number, derivate?: boolean | undefined) => number) & { name?: string | undefined; }) => string`

Normalize activation function name to uppercase for token matching.

Parameters:
- `squash` - Runtime activation function reference.

Returns: Uppercased activation name or empty string.

### rebuildConnectionsLocal

`(networkLike: import("C:/NeatapticTS/src/architecture/network").default) => void`

Rebuild the network's flat connections array from each node's outgoing list.

Parameters:
- `networkLike` - Network-like instance to mutate.

Returns: Nothing.

### resolveAllHiddenLayers

`(initialContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LayerOrderingResolutionContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LayerOrderingResolutionContext`

Resolve all hidden layers in dependency order.

Parameters:
- `initialContext` - Starting hidden-layer resolution context.

Returns: Final resolved layer-ordering context.

### resolveNextHiddenLayer

`(resolutionContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LayerOrderingResolutionContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LayerOrderingResolutionContext`

Resolve the next hidden layer from unresolved candidates.

Parameters:
- `resolutionContext` - Current resolution state.

Returns: Updated resolution state.

### resolveOnnxActivationOperation

`(normalizedActivationName: string) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxActivationOperation`

Resolve ONNX activation op from a normalized activation name token.

Parameters:
- `normalizedActivationName` - Uppercased activation name.

Returns: ONNX activation operation.

### validateLayerActivationHomogeneity

`(activationValidationContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LayerActivationValidationContext) => void`

Validate that a layer has homogeneous activation unless explicitly allowed.

Parameters:
- `activationValidationContext` - Activation validation context.

Returns: Nothing.

### validateLayerConnectivity

`(layerValidationContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LayerValidationTraversalContext) => void`

Validate that each current-layer node has required incoming connectivity.

Parameters:
- `layerValidationContext` - Layer connectivity traversal context.

Returns: Nothing.

### validateLayerHomogeneityAndConnectivity

`(layers: import("C:/NeatapticTS/src/architecture/node").default[][], network: import("C:/NeatapticTS/src/architecture/network").default, options: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxExportOptions) => void`

Validate connectivity and activation homogeneity constraints per layer.

Parameters:
- `layers` - Layered node arrays.
- `network` - Source network (reserved for compatibility).
- `options` - Export options.

Returns: Nothing.

### validateSingleLayer

`(layerValidationContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LayerValidationTraversalContext) => void`

Validate one current layer against activation/connectivity constraints.

Parameters:
- `layerValidationContext` - Layer validation context.

Returns: Nothing.

### validateSourceToTargetConnectivity

`(connectivityValidationContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LayerConnectivityValidationContext) => void`

Validate one source->target connection pair under export constraints.

Parameters:
- `connectivityValidationContext` - Source/target connectivity context.

Returns: Nothing.

### validateTargetNodeConnectivity

`(context: { targetNode: import("C:/NeatapticTS/src/architecture/node").default; previousLayerNodes: import("C:/NeatapticTS/src/architecture/node").default[]; layerIndex: number; allowPartialConnectivity: boolean; }) => void`

Validate full source coverage for one target node.

Parameters:
- `context` - Target-node connectivity context.

Returns: Nothing.

### warnWhenActivationFallbackIsUsed

`(context: { squash: ((x: number, derivate?: boolean | undefined) => number) & { name?: string | undefined; }; resolvedActivationOperation: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxActivationOperation; }) => void`

Emit a warning when activation export falls back to Identity.

Parameters:
- `context` - Activation fallback evaluation context.

Returns: Nothing.

## architecture/network/onnx/network.onnx.export-recurrent.utils.ts

### buildDefaultGemmAttributes

`() => { name: string; type: string; f?: number | undefined; i?: number | undefined; }[]`

Build the shared attribute list for ONNX Gemm node payloads.

Returns: Gemm attribute payload list.

### buildInputBranchGemmEmissionContext

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentLayerEmissionContext, initializerNames: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentInitializerNames, graphNames: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentGraphNames) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentGemmEmissionContext`

Build Gemm emission context for the feed-forward branch.

Parameters:
- `context` - Recurrent layer execution context.
- `initializerNames` - Recurrent initializer names.
- `graphNames` - Recurrent graph names.

Returns: Gemm emission context.

### buildRecurrentBranchGemmEmissionContext

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentLayerEmissionContext, initializerNames: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentInitializerNames, graphNames: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentGraphNames) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentGemmEmissionContext`

Build Gemm emission context for the recurrent hidden-state branch.

Parameters:
- `context` - Recurrent layer execution context.
- `initializerNames` - Recurrent initializer names.
- `graphNames` - Recurrent graph names.

Returns: Gemm emission context.

### buildRecurrentGraphNames

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentLayerEmissionContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentGraphNames`

Build deterministic graph names for recurrent-node emission.

Parameters:
- `context` - Recurrent layer execution context.

Returns: Graph-name group for branch and activation nodes.

### buildRecurrentInitializerNames

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentLayerEmissionContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentInitializerNames`

Build deterministic tensor names for recurrent initializer emission.

Parameters:
- `context` - Recurrent layer execution context.

Returns: Tensor-name group for initializer emission.

### buildRecurrentLayerEmissionContext

`(params: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentLayerEmissionParams) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentLayerEmissionContext`

Build derived recurrent-layer context from input params.

Parameters:
- `params` - User-provided recurrent layer params.

Returns: Derived context with cached dimensions and layer slot.

### collectRecurrentInitializerValues

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentLayerEmissionContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentInitializerValues`

Collect recurrent initializer vectors for one layer.

Parameters:
- `context` - Recurrent layer execution context.

Returns: Dense and recurrent initializer vectors.

### emitRecurrentActivationNode

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentActivationEmissionContext) => void`

Emit activation node for recurrent branch sum output.

Parameters:
- `context` - Activation emission context.

Returns: Nothing.

### emitRecurrentAddNode

`(model: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel, graphNames: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentGraphNames) => void`

Emit Add node that fuses feed-forward and recurrent branch outputs.

Parameters:
- `model` - Target ONNX model.
- `graphNames` - Deterministic graph names for this layer.

Returns: Nothing.

### emitRecurrentGemmNode

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentGemmEmissionContext) => void`

Emit one recurrent Gemm node with shared ONNX attributes.

Parameters:
- `context` - Gemm emission context.

Returns: Nothing.

### emitRecurrentInitializers

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentInitializerEmissionContext) => void`

Emit dense and recurrent initializer tensors.

Parameters:
- `context` - Initializer emission context.

Returns: Nothing.

### emitRecurrentLayer

`(params: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentLayerEmissionParams) => string`

Emit recurrent single-step layer representation.

Parameters:
- `params` - Recurrent emission parameters.

Returns: Output tensor name.

### readNodeInternals

`(node: import("C:/NeatapticTS/src/architecture/node").default) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").NodeInternals`

Normalize runtime node shape to recurrent-export internals contract.

Parameters:
- `node` - Runtime node instance.

Returns: Node internals used by ONNX emission helpers.

### resolvePreviousHiddenInputName

`(layerIndex: number) => string`

Resolve recurrent branch hidden-state input for one layer.

Parameters:
- `layerIndex` - Current recurrent layer index.

Returns: Hidden-state tensor input name.

### resolveRecurrentActivationType

`(currentLayerNodes: import("C:/NeatapticTS/src/architecture/node").default[]) => string`

Resolve ONNX activation type from first node in recurrent layer.

Parameters:
- `currentLayerNodes` - Current recurrent layer nodes.

Returns: ONNX activation op type.

## architecture/network/onnx/network.onnx.export-layer-graph.utils.ts

### emitLayerGraph

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LayerBuildContext) => string`

Emit one export layer graph segment and return the produced output tensor name.

Parameters:
- `context` - Layer build context.

Returns: Output tensor name produced by this layer.

## architecture/network/onnx/network.onnx.export-postprocess.utils.ts

### appendConvLayerValidationResult

`(result: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").ConvSharingValidationResult, layerIndex: number, isConsistent: boolean) => void`

Append one Conv-layer validation outcome and optional warning.

### appendConvSharingMetadata

`(model: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel, result: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").ConvSharingValidationResult) => void`

Append Conv-sharing validation metadata arrays.

### appendFusedRecurrentInitializers

`(model: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel, initializerNames: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").FusedRecurrentInitializerNames, parameters: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentGateParameterCollectionResult, gateCount: number, unitSize: number, previousSize: number) => void`

Append fused recurrent initializer tensors to the ONNX graph.

### appendFusedRecurrentNode

`(graph: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxGraph, operatorType: "LSTM" | "GRU", previousOutputName: string, initializerNames: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").FusedRecurrentInitializerNames, graphNames: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").FusedRecurrentGraphNames, unitSize: number) => void`

Append fused recurrent operator node to the ONNX graph.

### appendIndexMetadata

`(model: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel, key: string, layerIndex: number) => void`

Append a unique layer index to metadata array key.

### appendMetadataProperty

`(model: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel, metadataProperty: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxMetadataProperty) => void`

Append metadata property to model metadata_props list.

### appendRecurrentSingleStepMetadata

`(model: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel, recurrentLayerIndices: number[]) => void`

Append recurrent single-step metadata when recurrent layers exist.

### areWeightsWithinTolerance

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").WeightToleranceComparisonContext) => boolean`

Compare two scalar weights using configured tolerance.

### asNodeInternals

`(node: import("C:/NeatapticTS/src/architecture/node").default) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").NodeInternals`

Resolve runtime node internals in one typed helper.

### buildFusedGruExecutionContext

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").GruEmissionContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").FusedRecurrentEmissionExecutionContext`

Build shared fused-recurrent execution context for GRU.

### buildFusedLstmExecutionContext

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LstmEmissionContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").FusedRecurrentEmissionExecutionContext`

Build shared fused-recurrent execution context for LSTM.

### buildFusedRecurrentGraphNames

`(nodePrefix: string, outputSuffix: string, layerIndex: number) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").FusedRecurrentGraphNames`

Build fused recurrent graph names for node and output.

### buildFusedRecurrentInitializerNames

`(operatorType: "LSTM" | "GRU", layerIndex: number) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").FusedRecurrentInitializerNames`

Build fused recurrent initializer names for the current layer.

### buildGruEmissionContext

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").HiddenLayerHeuristicContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").GruEmissionContext`

Build GRU emission context from one hidden-layer traversal record.

### buildHiddenLayerHeuristicContext

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentHeuristicEmissionContext, layerIndex: number) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").HiddenLayerHeuristicContext`

Build one hidden-layer traversal context.

### buildLstmEmissionContext

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").HiddenLayerHeuristicContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LstmEmissionContext`

Build LSTM emission context from one hidden-layer traversal record.

### buildMetadataProperty

`(key: string, value: unknown) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxMetadataProperty`

Build a metadata key/value property with JSON string serialization.

### buildRecurrentHeuristicEmissionContext

`(model: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel, layers: import("C:/NeatapticTS/src/architecture/node").default[][], previousOutputName: string) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentHeuristicEmissionContext`

Build reusable context for recurrent heuristic traversal.

### collectConvKernelCoordinates

`(convSpec: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").Conv2DMapping) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxConvKernelCoordinate[]`

Collect kernel coordinates for one Conv kernel traversal.

### collectConvOutputCoordinates

`(convSpec: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").Conv2DMapping) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").ConvOutputCoordinate[]`

Collect output coordinates for full Conv traversal.

### collectGruGateNodeGroups

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").GruEmissionContext) => import("C:/NeatapticTS/src/architecture/node").default[][]`

Collect GRU gate node groups in canonical export order.

### collectHiddenLayerIndices

`(layers: import("C:/NeatapticTS/src/architecture/node").default[][]) => number[]`

Collect hidden-layer indices for recurrent traversal.

### collectLstmGateNodeGroups

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LstmEmissionContext) => import("C:/NeatapticTS/src/architecture/node").default[][]`

Collect LSTM gate node groups in canonical export order.

### collectRecurrentGateBlockParameters

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentGateBlockCollectionContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentGateParameterCollectionResult`

Collect flattened parameter vectors for one gate node block.

### collectRecurrentGateRow

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentGateRowCollectionContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentGateRow`

Collect one recurrent gate row payload (inputs, recurrent slice, and bias).

### collectRepresentativeKernelForChannel

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").ConvRepresentativeKernelContext) => number[]`

Collect one representative kernel by reading the first output position for a channel.

### collectRepresentativeKernels

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").ConvLayerPairContext) => number[][]`

Collect representative kernels for each output channel.

### collectRepresentativeKernelWeight

`(convSpec: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").Conv2DMapping, previousLayerNodes: import("C:/NeatapticTS/src/architecture/node").default[], representativeInternal: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").NodeInternals, kernelCoordinate: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxConvKernelCoordinate) => number`

Collect representative kernel value using top-left receptive field indexing.

### emitFallbackRecurrentPatternMetadata

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").HiddenLayerHeuristicContext) => void`

Emit fallback metadata for recurrent-size ambiguity.

### emitFusedRecurrentHeuristics

`(model: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel, layers: import("C:/NeatapticTS/src/architecture/node").default[][], allowRecurrent: boolean | undefined, previousOutputName: string) => void`

Emit heuristic fused recurrent operators (LSTM/GRU) when recurrent export is enabled.

Parameters:
- `model` - Target ONNX model.
- `layers` - Layered network nodes.
- `allowRecurrent` - Whether recurrent export is enabled.
- `previousOutputName` - Current graph output name (kept for backward-compatible emission semantics).

Returns: Nothing.

### emitFusedRecurrentLayer

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").FusedRecurrentEmissionExecutionContext) => void`

Emit shared fused recurrent payload (initializers, node, metadata).

### ensureMetadataProps

`(model: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxMetadataProperty[]`

Ensure metadata_props array exists and return it.

### finalizeExportMetadata

`(model: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel, layers: import("C:/NeatapticTS/src/architecture/node").default[][], options: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxExportOptions, includeMetadata: boolean, hiddenSizesMetadata: number[], recurrentLayerIndices: number[]) => void`

Finalize export metadata and optional conv-sharing validation.

Parameters:
- `model` - Target ONNX model.
- `layers` - Layered network nodes.
- `options` - Export options.
- `includeMetadata` - Whether metadata emission is enabled.
- `hiddenSizesMetadata` - Hidden-layer sizes collected during emission.
- `recurrentLayerIndices` - Recurrent layer indices.

Returns: Nothing.

### findMetadataPropertyIndex

`(metadataProperties: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxMetadataProperty[], key: string) => number`

Find metadata property index by key.

### foldRecurrentGateBlocks

`(gateParameterBlocks: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentGateParameterCollectionResult[]) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentGateParameterCollectionResult`

Fold gate blocks into a single fused parameter payload.

### foldRecurrentGateRows

`(gateRows: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentGateRow[]) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentGateParameterCollectionResult`

Fold recurrent gate rows into flattened ONNX initializer vectors.

### isConvLayerPairConsistent

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").ConvLayerPairContext) => boolean`

Validate one Conv layer pair against representative kernel sharing.

### isEligibleForGruHeuristic

`(currentSize: number) => boolean`

Check GRU heuristic eligibility by size and gate divisibility.

### isEligibleForLstmHeuristic

`(currentSize: number) => boolean`

Check LSTM heuristic eligibility by size and gate divisibility.

### isFallbackRecurrentPatternSize

`(currentSize: number) => boolean`

Check whether hidden size should emit recurrent fallback metadata.

### isInputPositionInsideBounds

`(convSpec: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").Conv2DMapping, inputRow: number, inputColumn: number) => boolean`

Check whether input row/column falls inside Conv input bounds.

### isKernelCoordinateConsistent

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").ConvKernelConsistencyContext) => boolean`

Validate one kernel coordinate against its representative channel value.

### isOutputCoordinateConsistent

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").ConvLayerPairContext, outputCoordinate: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").ConvOutputCoordinate, representativeKernels: number[][], tolerance: number) => boolean`

Validate one output coordinate against channel representative kernel weights.

### parseMetadataLayerIndices

`(metadataValue: string) => number[]`

Parse metadata JSON value into a numeric layer-index array.

### resolveConvLayerPairContext

`(layers: import("C:/NeatapticTS/src/architecture/node").default[][], layerIndex: number, convSpec: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").Conv2DMapping) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").ConvLayerPairContext | undefined`

Resolve one Conv mapping layer pair or return undefined for invalid layout.

### resolveGruPreviousOutputName

`(layerIndex: number) => string`

Resolve previous output naming semantics for GRU heuristic emission.

### resolveIncomingWeight

`(targetNodeInternal: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").NodeInternals, sourceNode: import("C:/NeatapticTS/src/architecture/node").default) => number`

Resolve incoming connection weight from a specific source node.

### resolveInputPosition

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").ConvKernelConsistencyContext) => { inputRow: number; inputColumn: number; }`

Resolve input row/column projected by output and kernel coordinates.

### resolveNeuronInternalAtOutputCoordinate

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").ConvLayerPairContext, outputCoordinate: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").ConvOutputCoordinate) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").NodeInternals | undefined`

Resolve runtime internals for output coordinate neuron, if present.

### resolveRecurrentRowWeight

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentGateRowCollectionContext, columnIndex: number) => number`

Resolve one recurrent row value at the requested column.

### resolveSelfConnectionWeight

`(targetNodeInternal: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").NodeInternals) => number`

Resolve self-connection weight for diagonal recurrent matrix entries.

### resolveSourceNodeAtInputPosition

`(convSpec: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").Conv2DMapping, previousLayerNodes: import("C:/NeatapticTS/src/architecture/node").default[], inChannelIndex: number, inputRow: number, inputColumn: number) => import("C:/NeatapticTS/src/architecture/node").default | undefined`

Resolve source node by Conv input position coordinates.

### shouldValidateConvSharing

`(options: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxExportOptions) => boolean`

Determine whether Conv2D sharing validation is enabled and configured.

### tryEmitFusedGru

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").HiddenLayerHeuristicContext) => void`

Try emitting heuristic fused GRU node and metadata.

### tryEmitFusedLstm

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").HiddenLayerHeuristicContext) => void`

Try emitting heuristic fused LSTM node and metadata.

### upsertLayerIndexMetadataValue

`(metadataProperties: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxMetadataProperty[], metadataIndex: number, layerIndex: number) => void`

Upsert one layer index into metadata array-like JSON value.

### validateConvSharingAcrossMappings

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").ConvSharingValidationContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").ConvSharingValidationResult`

Validate Conv2D sharing across all declared Conv mappings.

## architecture/network/onnx/network.onnx.import-activations.utils.ts

### assignActivationFunctions

`(network: import("C:/NeatapticTS/src/architecture/network").default, onnx: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel, hiddenLayerSizes: number[]) => void`

Assign node activation functions from ONNX activation nodes.

Parameters:
- `network` - Target network to mutate.
- `onnx` - Source ONNX model.
- `hiddenLayerSizes` - Hidden layer size list.

Returns: Nothing.

## architecture/network/onnx/network.onnx.export-layer-common.utils.ts

### appendIndexedMetadata

`(model: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel, key: string, layerIndex: number) => void`

Append an integer index to JSON-array metadata key.

Parameters:
- `model` - Target model.
- `key` - Metadata key.
- `layerIndex` - Layer index to append.

Returns: Nothing.

### appendMetadataSpec

`(model: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel, key: string, spec: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").Conv2DMapping | import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").Pool2DMapping) => void`

Append a JSON object to JSON-array metadata key.

Parameters:
- `model` - Target model.
- `key` - Metadata key.
- `spec` - Metadata object.

Returns: Nothing.

### appendPoolingMetadata

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").PoolingEmissionContext) => void`

Append pooling metadata for one emitted pooling layer.

Parameters:
- `context` - Pooling emission context.

Returns: Nothing.

### asNodeInternals

`(node: import("C:/NeatapticTS/src/architecture/node").default) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").NodeInternals`

Normalize a public node instance into ONNX export internals.

Parameters:
- `node` - Source node.

Returns: Internal runtime-facing node representation.

### buildDenseWeightsAndBiases

`(previousLayerNodes: import("C:/NeatapticTS/src/architecture/node").default[], currentLayerNodes: import("C:/NeatapticTS/src/architecture/node").default[]) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").DenseWeightBuildResult`

Build dense-layer weight matrix and bias vector.

Parameters:
- `previousLayerNodes` - Source layer nodes.
- `currentLayerNodes` - Destination layer nodes.

Returns: Flattened row-major weight matrix and bias vector.

### buildDiagonalRecurrentWeights

`(currentLayerNodes: import("C:/NeatapticTS/src/architecture/node").default[]) => number[]`

Build a diagonal recurrent matrix from self-connections.

Parameters:
- `currentLayerNodes` - Layer nodes.

Returns: Flattened row-major recurrent matrix.

### buildIndexedMetadataProperty

`(key: string, layerIndex: number) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxMetadataProperty`

Build a new index-array metadata property.

Parameters:
- `key` - Metadata key.
- `layerIndex` - Layer index.

Returns: Metadata property.

### buildSpecMetadataProperty

`(key: string, spec: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").Conv2DMapping | import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").Pool2DMapping) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxMetadataProperty`

Build a new spec-array metadata property.

Parameters:
- `key` - Metadata key.
- `spec` - Mapping spec.

Returns: Metadata property.

### collectDenseRows

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").DenseWeightBuildContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").DenseWeightRow[]`

Collect dense rows for each target node in current layer.

Parameters:
- `context` - Dense row collection context.

Returns: Dense rows containing per-target weights and bias.

### collectDenseRowWeights

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").DenseWeightRowCollectionContext) => number[]`

Collect source-to-target weights for one dense row.

Parameters:
- `context` - Dense row collection context.

Returns: Row weights in source-node order.

### collectPoolingAttributes

`(poolSpec: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").Pool2DMapping) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").PoolingAttributes`

Collect ONNX pooling attributes from one pooling spec.

Parameters:
- `poolSpec` - Pooling spec.

Returns: Pooling attributes for ONNX node payload.

### collectRecurrentRow

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentRowCollectionContext) => number[]`

Collect one recurrent matrix row.

Parameters:
- `context` - Row collection context.

Returns: Recurrent row values.

### collectRecurrentRows

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").DiagonalRecurrentBuildContext) => number[][]`

Collect recurrent matrix rows for one layer.

Parameters:
- `context` - Recurrent matrix build context.

Returns: Recurrent row collection.

### emitOptionalFlattenAfterPooling

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").FlattenAfterPoolingContext) => string`

Conditionally emit flatten node after pooling.

Parameters:
- `context` - Flatten emission context.

Returns: Output tensor name after optional flatten.

### emitOptionalPoolingAndFlatten

`(params: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OptionalPoolingAndFlattenParams) => string`

Emit optional pooling and flatten nodes after a layer output.

Parameters:
- `params` - Pooling parameters.

Returns: Final output tensor name after optional pooling/flatten.

### emitPoolingNode

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").PoolingEmissionContext) => string`

Emit one pooling node and return its output tensor name.

Parameters:
- `context` - Pooling emission context.

Returns: Pooling output tensor name.

### ensureMetadataRegistry

`(model: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxMetadataProperty[]`

Ensure model metadata registry exists.

Parameters:
- `model` - Target model.

Returns: Mutable metadata registry.

### findMetadataProperty

`(metadataRegistry: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxMetadataProperty[], key: string) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxMetadataProperty | undefined`

Find a metadata property by key.

Parameters:
- `metadataRegistry` - Metadata registry.
- `key` - Metadata key.

Returns: Matching metadata property if present.

### foldDenseRowsToInitializers

`(denseRows: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").DenseWeightRow[]) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").DenseWeightBuildResult`

Fold dense rows into flattened ONNX initializer arrays.

Parameters:
- `denseRows` - Dense rows.

Returns: Flattened dense initializer result.

### parseMetadataArray

`(metadataValue: string) => ItemType[] | undefined`

Parse a metadata JSON array value safely.

Parameters:
- `metadataValue` - Metadata JSON string.

Returns: Parsed array when valid, otherwise undefined.

### resolveDiagonalRecurrentWeight

`(context: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").RecurrentRowCollectionContext, columnIndex: number) => number`

Resolve recurrent weight value for one matrix coordinate.

Parameters:
- `context` - Row collection context.
- `columnIndex` - Column index in row.

Returns: Recurrent weight for diagonal entries, otherwise zero.

### resolveInboundWeight

`(targetNodeInternal: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").NodeInternals, sourceNode: import("C:/NeatapticTS/src/architecture/node").default) => number`

Resolve source-to-target inbound connection weight.

Parameters:
- `targetNodeInternal` - Target node internals.
- `sourceNode` - Source node.

Returns: Inbound weight or zero for disconnected edges.

### serializeIndexedMetadataValue

`(currentValue: string, layerIndex: number) => string`

Serialize index metadata after appending one unique index.

Parameters:
- `currentValue` - Existing JSON value.
- `layerIndex` - Layer index.

Returns: Serialized JSON value.

### serializeSpecMetadataValue

`(currentValue: string, spec: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").Conv2DMapping | import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").Pool2DMapping) => string`

Serialize spec metadata after appending one spec object.

Parameters:
- `currentValue` - Existing JSON value.
- `spec` - Mapping spec.

Returns: Serialized JSON value.

### toPoolingEmissionContext

`(params: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OptionalPoolingAndFlattenParams) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").PoolingEmissionContext`

Resolve pooling emission context from optional pooling parameters.

Parameters:
- `params` - Optional pooling and flatten parameters.

Returns: Pooling emission context.

## architecture/network/onnx/network.onnx.export-orchestrators.utils.ts

### appendConvInferenceMetadata

`(model: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel, layers: import("C:/NeatapticTS/src/architecture/node").default[][], options: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxExportOptions) => void`

Append heuristic conv inference metadata when requested.

Parameters:
- `model` - Target ONNX model.
- `layers` - Layered network nodes.
- `options` - Export options.

Returns: Nothing.

### appendLstmPatternStubMetadata

`(model: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel, lstmPatternStubs: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LstmPatternStub[]) => void`

Append LSTM pattern stub metadata.

Parameters:
- `model` - Target ONNX model.
- `lstmPatternStubs` - Pattern stubs.

Returns: Nothing.

### appendMetadataProperties

`(model: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel, metadataProperties: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxMetadataProperty[]) => void`

Append metadata properties in a single, normalized path.

Parameters:
- `model` - Target ONNX model.
- `metadataProperties` - Metadata properties to append.

Returns: Nothing.

### applyExportNodeIndexAssignments

`(assignmentContexts: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").ExportNodeIndexAssignmentContext[]) => void`

Apply prepared node/index assignment contexts.

Parameters:
- `assignmentContexts` - Prepared contexts.

Returns: Nothing.

### applySingleExportNodeIndexAssignment

`(assignmentContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").ExportNodeIndexAssignmentContext) => void`

Apply one export index assignment.

Parameters:
- `assignmentContext` - Assignment context.

Returns: Nothing.

### assignExportNodeIndices

`(network: import("C:/NeatapticTS/src/architecture/network").default) => void`

Assign stable index values to nodes for export diagnostics.

Parameters:
- `network` - Source network.

Returns: Nothing.

### collectInferredConvMetadata

`(context: { layers: import("C:/NeatapticTS/src/architecture/node").default[][]; declaredMappings: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").Conv2DMapping[] | undefined; }) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").ConvInferenceResult`

Collect inferred Conv metadata from hidden-layer traversals.

Parameters:
- `context` - Conv traversal context.

Returns: Inferred Conv metadata result.

### collectLstmPatternStubs

`(layers: import("C:/NeatapticTS/src/architecture/node").default[][], allowRecurrent: boolean | undefined) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LstmPatternStub[]`

Collect heuristic LSTM grouping stubs from hidden layers.

Parameters:
- `layers` - Layered network nodes.
- `allowRecurrent` - Whether recurrent export heuristics are enabled.

Returns: Candidate LSTM pattern stubs.

### collectLstmPatternStubsFromLayers

`(layers: import("C:/NeatapticTS/src/architecture/node").default[][]) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LstmPatternStub[]`

Collect LSTM pattern stubs from hidden layers.

Parameters:
- `layers` - Layered network nodes.

Returns: LSTM pattern stubs.

### createConvInferenceEvaluationContext

`(traversalContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").ConvInferenceTraversalContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").ConvInferenceEvaluationContext`

Create width/square-evaluation context for Conv inference.

Parameters:
- `traversalContext` - Conv traversal context.

Returns: Conv evaluation context.

### createConvTraversalContexts

`(context: { layers: import("C:/NeatapticTS/src/architecture/node").default[][]; declaredMappings: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").Conv2DMapping[] | undefined; }) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").ConvInferenceTraversalContext[]`

Create Conv traversal contexts for hidden layers.

Parameters:
- `context` - Conv traversal source context.

Returns: Conv traversal contexts.

### createExportNodeIndexAssignmentContexts

`(network: import("C:/NeatapticTS/src/architecture/network").default) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").ExportNodeIndexAssignmentContext[]`

Create node/index assignment contexts for export diagnostics.

Parameters:
- `network` - Source network.

Returns: Assignment contexts.

### createHiddenLayerTraversalContexts

`(layers: import("C:/NeatapticTS/src/architecture/node").default[][]) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LstmLayerTraversalContext[]`

Create traversal contexts for hidden layers only.

Parameters:
- `layers` - Layered network nodes.

Returns: Hidden layer contexts.

### createLstmCandidateContext

`(hiddenLayerContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LstmLayerTraversalContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LstmCandidateContext`

Build LSTM candidate context for one hidden layer.

Parameters:
- `hiddenLayerContext` - Hidden layer context.

Returns: LSTM candidate context.

### hasInferredConvMetadata

`(inferenceResult: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").ConvInferenceResult) => boolean`

Check whether inferred Conv metadata exists.

Parameters:
- `inferenceResult` - Inferred Conv result.

Returns: True when inferred metadata exists.

### hasRequiredSelfConnectionCount

`(nodeItem: import("C:/NeatapticTS/src/architecture/node").default) => boolean`

Check whether one node has the required self-connection count.

Parameters:
- `nodeItem` - Node to inspect.

Returns: True when self-connection count matches requirement.

### isDeclaredConvLayer

`(traversalContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").ConvInferenceTraversalContext) => boolean`

Check whether a traversal layer already has declared Conv mapping.

Parameters:
- `traversalContext` - Conv traversal context.

Returns: True when mapping is already declared.

### isInferredConvSpec

`(specification: (import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").Conv2DMapping & { note?: string | undefined; }) | undefined) => boolean`

Type guard for inferred Conv specifications.

Parameters:
- `specification` - Conv specification candidate.

Returns: True when specification is defined.

### isValidLstmCandidateContext

`(candidateContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LstmCandidateContext) => boolean`

Determine whether a candidate context satisfies heuristic LSTM conditions.

Parameters:
- `candidateContext` - Candidate context.

Returns: True when the candidate is a valid LSTM stub.

### mapLstmCandidateToStub

`(candidateContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LstmCandidateContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LstmPatternStub`

Map a valid candidate context to metadata stub.

Parameters:
- `candidateContext` - Valid candidate context.

Returns: LSTM pattern stub.

### resolveConvInferenceForLayer

`(traversalContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").ConvInferenceTraversalContext) => (import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").Conv2DMapping & { note?: string | undefined; }) | undefined`

Resolve inferred Conv specification for one hidden layer.

Parameters:
- `traversalContext` - Conv traversal context.

Returns: Inferred Conv specification when matched.

### resolveConvSpecForKernel

`(kernelContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").ConvInferenceKernelEvaluationContext) => (import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").Conv2DMapping & { note?: string | undefined; }) | undefined`

Resolve Conv specification for one kernel candidate.

Parameters:
- `kernelContext` - Kernel-evaluation context.

Returns: Inferred Conv specification when matched.

### resolveConvSpecFromKernelCandidates

`(evaluationContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").ConvInferenceEvaluationContext) => (import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").Conv2DMapping & { note?: string | undefined; }) | undefined`

Resolve Conv specification using ordered kernel candidates.

Parameters:
- `evaluationContext` - Conv evaluation context.

Returns: Inferred Conv specification when matched.

### safelyCollectLstmPatternStubs

`(layers: import("C:/NeatapticTS/src/architecture/node").default[][]) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").LstmPatternStub[]`

Collect LSTM pattern stubs with heuristic error isolation.

Parameters:
- `layers` - Layered network nodes.

Returns: LSTM pattern stubs.

## architecture/network/onnx/network.onnx.import-orchestrators.utils.ts

### applyLayerSelfConnections

`(layerConnectionContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportLayerConnectionContext) => void`

Apply one hidden layer diagonal recurrent self-weights.

Parameters:
- `layerConnectionContext` - Layer connection context.

Returns: Nothing.

### attachOnnxPoolingMetadata

`(network: import("C:/NeatapticTS/src/architecture/network").default, metadata: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxMetadataProperty[]) => void`

Attach optional pooling metadata from ONNX model to network instance.

Parameters:
- `network` - Target network.
- `metadata` - ONNX metadata.

Returns: Nothing.

### attachParsedPoolingMetadata

`(network: import("C:/NeatapticTS/src/architecture/network").default, poolingMetadata: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportPoolingMetadata) => void`

Attach parsed pooling metadata to imported network instance.

Parameters:
- `network` - Target network.
- `poolingMetadata` - Parsed pooling metadata payload.

Returns: Nothing.

### buildArchitectureContext

`(onnx: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportArchitectureContext`

Build architecture extraction context from ONNX graph state.

Parameters:
- `onnx` - Source ONNX model.

Returns: Normalized architecture extraction context.

### buildHiddenLayerSpans

`(hiddenLayerSizes: number[]) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportHiddenLayerSpan[]`

Build hidden-layer spans with one-based layer numbering and global offsets.

Parameters:
- `hiddenLayerSizes` - Hidden-layer size list.

Returns: Hidden-layer span payload list.

### collectDiagonalRecurrentWeights

`(recurrentTensorWeights: number[], hiddenLayerSize: number) => number[]`

Collect diagonal recurrent weights from flattened layer tensor data.

Parameters:
- `recurrentTensorWeights` - Flattened recurrent tensor weights.
- `hiddenLayerSize` - Hidden-layer width.

Returns: Diagonal recurrent self-weights.

### collectNodesByType

`(nodes: import("C:/NeatapticTS/src/architecture/node").default[], nodeType: "input" | "output" | "hidden") => import("C:/NeatapticTS/src/architecture/node").default[]`

Collect nodes matching one runtime node-type discriminator.

Parameters:
- `nodes` - Node list.
- `nodeType` - Runtime node type.

Returns: Filtered node list.

### collectPerceptronBoundaryNodes

`(nodes: import("C:/NeatapticTS/src/architecture/node").default[]) => import("C:/NeatapticTS/src/architecture/node").default[]`

Collect input and output boundary nodes for perceptron imports.

Parameters:
- `nodes` - Full network node list.

Returns: Input/output-only node list.

### collectRecurrentLayerSpans

`(restorationContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportRecurrentRestorationContext) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportHiddenLayerSpan[]`

Resolve recurrent-target hidden-layer spans from metadata + hidden sizes.

Parameters:
- `restorationContext` - Recurrent restoration context.

Returns: Hidden-layer spans requiring recurrent restoration.

### extractOnnxArchitecture

`(onnx: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportArchitectureResult`

Extract input/output counts and hidden layer sizes from ONNX model.

Parameters:
- `onnx` - Source ONNX model.

Returns: Parsed architecture dimensions.

### findMetadataProperty

`(metadata: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxMetadataProperty[], metadataKey: string) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxMetadataProperty | undefined`

Find one ONNX metadata property by key.

Parameters:
- `metadata` - ONNX metadata array.
- `metadataKey` - Metadata key.

Returns: Matching metadata property when present.

### findRecurrentInitializer

`(layerConnectionContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportLayerConnectionContext) => { name: string; float_data: number[]; } | undefined`

Resolve recurrent initializer tensor for one hidden-layer span.

Parameters:
- `layerConnectionContext` - Layer connection context.

Returns: Recurrent initializer tensor when available.

### isSingleLayerPerceptronImport

`(hiddenLayerSizes: number[]) => boolean`

Determine whether import shape corresponds to a single-layer perceptron.

Parameters:
- `hiddenLayerSizes` - Hidden-layer size list.

Returns: True when no hidden layers exist.

### normalizeRecurrentLayerIndices

`(parsedMetadataValue: string | number | boolean | number[] | Record<string, number> | null) => number[]`

Normalize recurrent layer indices parsed from metadata JSON.

Parameters:
- `parsedMetadataValue` - Parsed metadata JSON value.

Returns: Recurrent layer indices.

### parsePoolingMetadata

`(metadata: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxMetadataProperty[]) => import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportPoolingMetadata | null`

Parse pooling metadata payload from ONNX metadata.

Parameters:
- `metadata` - ONNX metadata entries.

Returns: Parsed pooling metadata payload.

### parseRecurrentLayerIndices

`(rawMetadataValue: string) => number[]`

Parse recurrent layer indices metadata.

Parameters:
- `rawMetadataValue` - Raw metadata JSON string.

Returns: Normalized recurrent layer indices.

### pruneSingleLayerHiddenPlaceholders

`(network: import("C:/NeatapticTS/src/architecture/network").default, hiddenLayerSizes: number[]) => void`

Remove placeholder hidden nodes for single-layer perceptron imports.

Parameters:
- `network` - Target network.
- `hiddenLayerSizes` - Hidden layer sizes.

Returns: Nothing.

### readLastDimensionValue

`(dimensions: { dim_value?: number | undefined; }[]) => number`

Read the terminal ONNX shape dimension value from one shape array.

Parameters:
- `dimensions` - ONNX shape dimensions.

Returns: Terminal `dim_value` payload.

### reconstructFusedRecurrentLayers

`(network: import("C:/NeatapticTS/src/architecture/network").default, onnx: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel, hiddenLayerSizes: number[], layerFactory: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxLayerFactory, metadata: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxMetadataProperty[]) => void`

Reconstruct emitted fused LSTM/GRU layers from ONNX metadata and initializers.

Parameters:
- `network` - Target network.
- `onnx` - Source ONNX model.
- `hiddenLayerSizes` - Hidden layer sizes.
- `layerFactory` - Dynamic layer module.
- `metadata` - ONNX metadata properties.

Returns: Nothing.

### resolveRecurrentLayerIndices

`(metadata: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxMetadataProperty[]) => number[]`

Resolve recurrent layer indices from ONNX metadata.

Parameters:
- `metadata` - ONNX metadata payload.

Returns: Parsed recurrent layer indices.

### restoreRecurrentSelfConnections

`(network: import("C:/NeatapticTS/src/architecture/network").default, onnx: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel, hiddenLayerSizes: number[], metadata: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxMetadataProperty[]) => void`

Restore recurrent self-connections from recurrent metadata and R tensors.

Parameters:
- `network` - Target network.
- `onnx` - Source ONNX model.
- `hiddenLayerSizes` - Hidden layer sizes.
- `metadata` - Parsed metadata properties.

Returns: Nothing.

### sliceLayerHiddenNodes

`(layerConnectionContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportLayerConnectionContext) => import("C:/NeatapticTS/src/architecture/node").default[]`

Slice hidden nodes for one hidden-layer span.

Parameters:
- `layerConnectionContext` - Layer connection context.

Returns: Hidden nodes belonging to the span.

### upsertSelfConnection

`(selfConnectionContext: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxImportSelfConnectionUpsertContext) => void`

Upsert one node self-connection for recurrent import restoration.

Parameters:
- `selfConnectionContext` - Self-connection upsert context.

Returns: Nothing.

## architecture/network/onnx/network.onnx.import-fused-recurrent.utils.ts

### reconstructFusedRecurrentLayers

`(network: import("C:/NeatapticTS/src/architecture/network").default, onnx: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxModel, hiddenLayerSizes: number[], layerFactory: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxLayerFactory, metadata: import("C:/NeatapticTS/src/architecture/network/onnx/network.onnx.utils.types").OnnxMetadataProperty[]) => void`

Reconstruct emitted fused LSTM/GRU layers from ONNX metadata and initializers.

Parameters:
- `network` - Target network.
- `onnx` - Source ONNX model.
- `hiddenLayerSizes` - Hidden layer sizes.
- `layerFactory` - Dynamic layer module.
- `metadata` - ONNX metadata properties.

Returns: Nothing.
