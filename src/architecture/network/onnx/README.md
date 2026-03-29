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

How to read this chapter:
- Start here for the public round-trip API and the trust boundary.
- Continue into `export/` to see how layered networks become JSON graph payloads.
- Continue into `import/` to see how that payload becomes a runtime network again.
- Continue into `schema/` for the persisted wire-format shapes.
- Use `network.onnx.utils.ts` and `network.onnx.utils.types.ts` as compatibility and
  bridge surfaces rather than the first place to learn the pipeline.

Why the folder is split this way:
- The root file keeps the stable entry points and the promise of the format.
- The `export/` and `import/` chapters carry the heavier execution details.
- The `schema/` chapter keeps the persisted document model separate from runtime logic.
- The root utility barrels exist so public ergonomics stay stable while the implementation
  can keep moving toward smaller, teachable chapters.

Trust boundary:
- Treat imported models as **untrusted input**. The importer validates structure, but
  you should still apply the same care you would for a generic JSON payload.

Example (export → persist → import):

```ts
import { exportToONNX, importFromONNX } from './network.onnx';

const model = exportToONNX(network, { includeMetadata: true });
const jsonText = JSON.stringify(model);

const modelRoundTrip = JSON.parse(jsonText);
const restored = importFromONNX(modelRoundTrip);
```

## architecture/network/onnx/network.onnx.ts

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

## architecture/network/onnx/network.onnx.utils.ts

ONNX export/import utilities for a constrained, documented subset of networks.

This file is the root compatibility barrel for the ONNX execution helpers.
It exists so callers can keep a stable import surface while the heavier
exporter and importer logic lives in the narrower `export/` and `import/`
chapters.

How to read this file:
- Start here if you want the thin orchestration-facing helpers that still
  bridge the public root API to the split implementation files.
- Continue into `export/` when you want the full graph-emission pipeline.
- Continue into `import/` when you want the reconstruction pipeline.
- Treat this file as a compatibility and forwarding surface, not the main
  home of ONNX execution details.

What still belongs here:
- Re-exports that are intentionally stable for root ONNX callers.
- Thin wrappers such as `buildOnnxModel()` that preserve a predictable
  orchestration surface while delegating the real work into smaller chapters.
- A small amount of roadmap context that helps readers understand why some
  recurrent and mixed-activation helpers still look transitional.

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

```ts
applyModelMetadata(
  context: OnnxModelMetadataContext,
): void
```

Attach producer and opset metadata to a model when metadata emission is enabled.

Parameters:
- `context` - Metadata application context.

Returns: Nothing.

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

### emitLayerGraph

```ts
emitLayerGraph(
  context: LayerBuildContext,
): string
```

Emit one export layer graph segment by routing the layer through the correct
ONNX emission strategy.

Dispatch order matters:
- explicit Conv mappings win first,
- recurrent single-step export is considered only for hidden layers with
  self-connections,
- non-recurrent layers fall back to compact dense emission or mixed-activation
  per-neuron decomposition.

Important invariants:
- recurrent mixed activations are rejected elsewhere rather than silently
  decomposed here,
- `allowMixedActivations` only affects the dense-family fallback path,
- the returned tensor name is the canonical input for the next layer.

Parameters:
- `context` - Layer build context.

Returns: Output tensor name produced by this layer.

Example:

```ts
const outputName = emitLayerGraph({
  model,
  layers,
  layerIndex: 2,
  previousOutputName: 'Layer_1',
  options: { allowMixedActivations: true },
  recurrentLayerIndices: [],
  batchDimension: false,
  legacyNodeOrdering: false,
});
```

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

```ts
rebuildConnectionsLocal(
  networkLike: default,
): void
```

Rebuild the network's flat connections array from each node's outgoing list.

Parameters:
- `networkLike` - Network-like instance to mutate.

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

## architecture/network/onnx/network.onnx.utils.types.ts

Types for NeatapticTS’s ONNX-like JSON export/import.

This file is now the root compatibility barrel for shared ONNX type surfaces.
Most exporter-owned, importer-owned, and schema-owned type families have
already moved into their chapter-local files. What remains here is the
narrow bridge layer that still needs to be visible from the root ONNX API.

How to read this type surface:
- Start with `schema/` if you want the persisted wire-format document model.
- Continue into `export/` for graph-emission contexts and layer payloads.
- Continue into `import/` for reconstruction-only contexts.
- Use this root file when you specifically need shared runtime bridge types,
  compatibility re-exports, or the small set of contracts that still span
  multiple ONNX chapters.

What still belongs here:
- Root re-exports that preserve public ergonomics while the underlying
  ownership lives in `schema/`, `export/`, or `import/`.
- Shared bridge types such as `NodeInternals`, activation-assignment
  contracts, and the runtime layer-factory widening used across chapters.
- Transitional compatibility groupings that are still safer to keep at the
  root until a later cleanup proves they can move without widening seams.

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

### ActivationSquashFunction

```ts
ActivationSquashFunction(
  x: number,
  derivate: boolean | undefined,
): number
```

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

### LstmEmissionContext

Context for heuristic LSTM emission when a layer matches expected shape.

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

ONNX node attribute payload.

This simplified JSON-first shape is enough for the operators emitted by the
current exporter. It intentionally avoids protobuf-level complexity while
still preserving the attribute variants needed by the importer.

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

One dimension inside an ONNX tensor shape.

Use `dim_value` for fixed numeric widths and `dim_param` for symbolic names
such as a batch dimension.

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

The importer only relies on a narrow runtime contract: access to the
reconstructed nodes, an input wiring hook, and an optional output group that
can be reconnected to the next restored layer.

### OnnxFusedRecurrentKind

Supported fused recurrent operator families recognized during ONNX import.

### OnnxFusedRecurrentSpec

Fused recurrent family specification used during import reconstruction.

This tells the importer how to interpret one emitted ONNX recurrent family:
how many gates to expect, what order those gates were serialized in, and
which gate owns the self-recurrent diagonal replay.

### OnnxFusedTensorPayload

Fused recurrent tensor payload read from ONNX initializers.

The importer resolves the three recurrent tensor families up front so the
reconstruction pass can focus on wiring and row assignment instead of
repeatedly re-looking up initializers.

### OnnxGraph

Graph body of an ONNX-like model.

The exporter writes three main collections here:
- `inputs` and `outputs` describe graph boundaries,
- `initializer` stores constant tensors such as weights and biases,
- `node` stores the ordered operator payloads that consume those tensors.

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

Context for applying Conv weights and bias at one output coordinate.

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

One ONNX operator invocation inside the graph.

Nodes connect named tensors rather than object references, which keeps the
exported payload easy to serialize, inspect, and diff as plain JSON.

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

### OnnxRuntimeLayerFactory

```ts
OnnxRuntimeLayerFactory(
  size: number,
): default
```

Runtime layer-constructor signature used for recurrent layer reconstruction.

### OnnxRuntimeLayerFactoryMap

Runtime layer module shape widened for fused-recurrent reconstruction wiring.

### OnnxRuntimeLayerModule

Runtime layer module shape consumed by ONNX import orchestration.

### OnnxRuntimePerceptronFactory

```ts
OnnxRuntimePerceptronFactory(
  sizes: number[],
): default
```

Runtime perceptron factory signature used by ONNX import orchestration.

### OnnxShape

ONNX tensor type shape.

### OnnxTensor

Serialized tensor payload stored inside graph initializers.

NeatapticTS currently writes floating-point parameter vectors and matrices to
`float_data`, along with the tensor name, element type, and logical shape.

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

## architecture/network/onnx/network.onnx.errors.ts

Raised when ONNX export cannot resolve a valid layered ordering.

### NetworkOnnxLayerOrderingUnresolvableError

Raised when ONNX export cannot resolve a valid layered ordering.

### NetworkOnnxMixedActivationsUnsupportedError

Raised when ONNX export encounters mixed activations without mixed-activation support enabled.

### NetworkOnnxPartialConnectivityUnsupportedError

Raised when ONNX export requires a connection that is missing.

### NetworkOnnxPerceptronSizeValidationError

Raised when ONNX import perceptron metadata omits required input/output sizes.

### NetworkOnnxRecurrentMixedActivationsUnsupportedError

Raised when recurrent ONNX export encounters unsupported mixed activations.

## architecture/network/onnx/network.onnx.layer-analysis.utils.ts

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
