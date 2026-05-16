/**
 * NeatapticTS ONNX-like serialization for networks.
 *
 * This module provides the two public entry points:
 * - `exportToONNX()` turns a runtime `Network` into a plain JSON object (`OnnxModel`).
 * - `importFromONNX()` reconstructs a `Network` from that JSON object.
 *
 * What this format is (and is not):
 * - It is **JSON-first** and intentionally resembles ONNX’s model/graph concepts.
 * - It is **not** a full ONNX protobuf implementation and is not guaranteed to run on
 *   general ONNX runtimes.
 * - The compatibility promise is primarily **within this repo**: models produced by
 *   `exportToONNX()` should be accepted by `importFromONNX()` (same version family).
 *
 * How to read this chapter:
 * - Start here for the public round-trip API and the trust boundary.
 * - Continue into `export/` to see how layered networks become JSON graph payloads.
 * - Continue into `import/` to see how that payload becomes a runtime network again.
 * - Continue into `schema/` for the persisted wire-format shapes.
 * - Use `network.onnx.utils.ts` and `network.onnx.utils.types.ts` as compatibility and
 *   bridge surfaces rather than the first place to learn the pipeline.
 *
 * Why the folder is split this way:
 * - The root file keeps the stable entry points and the promise of the format.
 * - The `export/` and `import/` chapters carry the heavier execution details.
 * - The `schema/` chapter keeps the persisted document model separate from runtime logic.
 * - The root utility barrels exist so public ergonomics stay stable while the implementation
 *   can keep moving toward smaller, teachable chapters.
 *
 * Trust boundary:
 * - Treat imported models as **untrusted input**. The importer validates structure, but
 *   you should still apply the same care you would for a generic JSON payload.
 *
 * Example (export → persist → import):
 *
 * ```ts
 * import { exportToONNX, importFromONNX } from './network.onnx';
 *
 * const model = exportToONNX(network, { includeMetadata: true });
 * const jsonText = JSON.stringify(model);
 *
 * const modelRoundTrip = JSON.parse(jsonText);
 * const restored = importFromONNX(modelRoundTrip);
 * ```
 */

import type Network from '../../network/network';
import { runOnnxExportFlow, runOnnxImportFlow } from './network.onnx.utils';
import type {
  AttentionMapping,
  ConcatMapping,
  OnnxExportOptions,
} from './network.onnx.utils.types';
import type {
  Conv2DMapping,
  OnnxModel,
  Pool2DMapping,
} from './schema/network.onnx.schema.types';

export type {
  AttentionMapping,
  ConcatMapping,
  Conv2DMapping,
  OnnxExportOptions,
  OnnxModel,
  Pool2DMapping,
};

/**
 * Export a NeatapticTS network to an ONNX-like **JSON object** (`OnnxModel`).
 *
 * What you get:
 * - A plain object that you can persist with `JSON.stringify()`.
 * - A minimal ONNX-ish graph (`model.graph`) plus optional metadata (`model.metadata_props`).
 *
 * When to use this:
 * - You want a portable snapshot that can be inspected/diffed as JSON.
 * - You want to reconstruct the network later via `importFromONNX()`.
 *
 * Tradeoffs:
 * - The output is ONNX-like, but **not** intended to be universally compatible with all ONNX
 *   runtimes.
 * - Some advanced features (partial connectivity, mixed activations, recurrent heuristics)
 *   may produce graphs that are primarily meant for this library’s importer.
 * - Phase 6 closes the first optimization wave conservatively: exact unary activation
 *   emission now covers `Softplus`, `Softsign`, `Selu`, `Mish` (opset >= 18), and
 *   `Gelu` with explicit `approximate='tanh'` (opset >= 20), while opset-incompatible
 *   or unsupported activations stay on the honest Identity baseline.
 * - Phase 7 is now active on the first reduced-precision lane: `precision.mode =
 *   'storage-fp16'` packs eligible same-family dense and Conv weight or bias
 *   initializers as float16 payloads and prepends deterministic `Cast -> float32`
 *   bridges so `Gemm` and `Conv` inputs stay type-consistent. Recurrent,
 *   advanced-graph, mixed-activation, and partial-connectivity requests stay on
 *   float32 with explicit metadata fallback reasons instead of silently widening
 *   the supported subset.
 * - Quantization packets are now exporter-owned and calibration-backed. Static
 *   8-bit requests can carry explicit layer-target calibration ranges and emit
 *   deterministic scale or zero-point initializers plus metadata for the
 *   supported same-family dense and explicit Conv subset. The current Phase 7D
 *   dense slice is now landed: explicitly targeted same-family dense layers
 *   can lower into `QuantizeLinear -> QLinearMatMul -> DequantizeLinear`,
 *   reattach nonzero bias through an explicit float-domain `Add` bridge,
 *   preserve the exporter-owned unary activation node, and emit a
 *   deterministic quantized weight tensor with
 *   `effective_quantization_mode = static-8bit`.
 *   Spatial qlinear lowering, dynamic quantization, and quantized import
 *   remain later Phase 7 work, so unsupported requests still record explicit
 *   float32 fallback reasons instead of widening the supported subset
 *   implicitly.
 * - The current spatial subset is still conservative: explicit Conv mappings round-trip,
 *   pooling/flatten import remains metadata-driven, and heuristic Conv inference stays
 *   metadata-only unless `autoPromoteInferredConv` is enabled and the inferred dense layer
 *   passes the shared-kernel safety gate for the current proven subset, including
 *   conservative multi-channel layouts, unpooled stacked Conv-like chains, deeper
 *   single-channel post-pool chains whose pooled tensor shape can be derived
 *   sequentially, and deeper pooled multi-channel chains whose pooled tensor shapes
 *   can be derived sequentially while export keeps the pooled source compact per
 *   channel. The only proven flatten-after-pool promotion path is the narrow final
 *   hidden-stage reshape-bridge subset, where export restores the derived pooled
 *   `[C,H,W]` shape before the later Conv. Earlier flattened pooled consumers,
 *   repeated flatten-bridge chains, or downstream dense layers that still depend on
 *   extra non-pooled inputs keep later inferred stages on the honest fallback path.
 * - Export now validates a conservative internal tensor-shape ledger before model
 *   finalization and prunes exporter-owned Identity activation scaffolding only when the
 *   graph stays semantically equivalent for the already-supported same-family subset.
 *
 * High-level algorithm:
 *  1) Normalize/rebuild local connection state for deterministic traversal.
 *  2) Infer an ordered layer view and validate export constraints.
 *  3) Materialize graph nodes/tensors and (optionally) attach metadata.
 *
 * Example (export → JSON text):
 *
 * ```ts
 * const model = exportToONNX(network, { includeMetadata: true });
 * const jsonText = JSON.stringify(model);
 * ```
 *
 * @param network Source network instance to serialize.
 * @param options Export controls (validation strictness and metadata behavior).
 * @returns ONNX-like model object suitable for persistence or re-import.
 * @throws If the network cannot be represented safely under the selected options.
 */
export function exportToONNX(
  network: Network,
  options: OnnxExportOptions = {},
): OnnxModel {
  // Step 1: Delegate complete export orchestration to the centralized flow helper.
  return runOnnxExportFlow(network, options);
}

/**
 * Reconstruct a NeatapticTS network from an exported `OnnxModel`.
 *
 * Expected input:
 * - A model produced by `exportToONNX()` (same repo/version family), including the
 *   current storage-fp16 subset where eligible weight and bias initializers are
 *   packed as float16 payloads and decoded back into the native runtime during import.
 * - Quantized Phase 7 exports remain export-only for now. The importer does not
 *   yet reconstruct `QLinearMatMul` or other quantized operators back into the
 *   native runtime, so quantized ONNX payloads are outside the current import contract.
 *
 * Trust boundary:
 * - Do not import untrusted blobs. A malformed model can be extremely large or internally
 *   inconsistent and may cause errors or high memory usage.
 *
 * High-level behavior:
 *  1) Build a perceptron-shaped scaffold from the payload layer sizes.
 *  2) Assign weights/biases and activation functions.
 *  3) Re-apply recurrent and pooling metadata when present.
 *
 * Example (JSON text → restore):
 *
 * ```ts
 * const model = JSON.parse(jsonText) as OnnxModel;
 * const restored = importFromONNX(model);
 * const output = restored.activate([0.1, 0.9]);
 * ```
 *
 * @param onnx ONNX-like model to reconstruct.
 * @returns Reconstructed network ready for inference/evolution workflows.
 * @throws If the model schema is incompatible or cannot be reconstructed safely.
 */
export function importFromONNX(onnx: OnnxModel): Network {
  // Step 1: Delegate complete import orchestration to the centralized flow helper.
  return runOnnxImportFlow(onnx);
}

/**
 * ONNX (JSON) serialization for NeatapticTS networks.
 *
 * NeatapticTS provides an **ONNX-like, JSON-first interchange format** for exporting and
 * reconstructing a constrained subset of `Network` instances.
 *
 * This is primarily meant for:
 * - Saving a trained network snapshot in a portable representation.
 * - Debugging / inspecting network structure as a graph of tensors and nodes.
 * - Interop with tooling that can consume graph-shaped metadata.
 *
 * Formats that exist
 * ------------------
 * In this folder, the “format” is really two layers:
 *
 * 1) In-memory model object (`OnnxModel`)
 *    - A plain JavaScript object shaped like an ONNX `ModelProto`, but represented as JSON.
 *    - This is what `exportToONNX()` returns and what `importFromONNX()` consumes.
 *
 * 2) Serialized JSON text
 *    - Persist via `JSON.stringify(model)`.
 *    - Human-readable and diffable.
 *
 * There is no binary/protobuf serializer here. For compactness, apply compression
 * (gzip/brotli) to the JSON string at the application layer.
 *
 * Guarantees & stability
 * ----------------------
 * - Round-trip intent: `importFromONNX(exportToONNX(network))` aims to reconstruct a
 *   functionally equivalent network for the supported subset.
 * - Compatibility: the importer is only guaranteed to accept models produced by this
 *   repo’s exporter.
 * - Determinism: export is designed to be deterministic given the same network state and
 *   export options.
 *
 * Supported recurrent subset (current)
 * ------------------------------------
 * - Single-step self-recurrent hidden layers emitted by this repo’s exporter.
 * - Heuristic LSTM/GRU export/import roundtrips from the same version family when the
 *   exported `W`, `R`, and `B` tensors are complete and shape-compatible.
 *
 * Supported spatial subset (current)
 * ----------------------------------
 * - Explicit `conv2dMappings` export real `Conv` nodes and import back into dense runtime
 *   connections for the supported internal roundtrip family.
 * - `pool2dMappings` and `flattenAfterPooling` export real graph nodes, but import currently
 *   re-attaches those semantics as `_onnxPooling` metadata for inspection rather than changing
 *   runtime inference.
 * - `flattenConsistency` is intentionally audit-only for now: it records whether flattened pool
 *   width matches the next dense consumer width without warning, rejecting, or rewriting weights.
 * - Heuristic `conv2d_inferred_specs` stays metadata-only by default. With
 *   `autoPromoteInferredConv`, supported Conv-like patterns can upgrade into real `Conv`
 *   emission only when the dense weights already satisfy the same shared-kernel consistency
 *   rules used by explicit Conv validation. The currently proven subset includes
 *   single-channel layouts, conservative multi-channel layouts, unpooled stacked
 *   Conv-like chains, deeper single-channel post-pool chains when the exporter can
 *   derive each pooled tensor shape sequentially and preserve the compact per-channel
 *   pooled source slice, and deeper pooled multi-channel chains when the exporter
 *   can do the same across repeated pooled stages. The only proven flatten-after-pool
 *   promotion path is the final hidden-stage reshape-bridge subset. Earlier flattened
 *   pooled consumers, repeated flatten-bridge chains, and downstream dense layers that
 *   still carry non-zero weights from extra non-pooled inputs remain metadata-only.
 *
 * Honest fallback boundary
 * ------------------------
 * - Arbitrary external ONNX recurrent graphs are not supported.
 * - One-hop dense-family residual adds now roundtrip for the same-family subset:
 *   export emits an explicit `Add` merge plus `advanced_graph_residual_adds`
 *   metadata, and import rebuilds the skipped feed-forward edges from the
 *   residual branch tensor together with the recorded cross-layer audit edges.
 * - Explicit same-family concat mappings now roundtrip for the narrow Phase 5
 *   subset: export emits a deterministic `Concat -> Gemm` path together with
 *   `advanced_graph_concat_merges` metadata, and import validates that merge,
 *   rebuilds the skipped source-layer fan-in from the widened dense tensor
 *   tail, and preserves the audit payload as `_onnxAdvancedGraph.concatMerges`.
 * - Fixed-width same-family self-attention mappings can now emit a deterministic
 *   shadow subgraph (`Q/K/V`, head split, score `MatMul`, optional scaling,
 *   `Softmax`, value aggregation, head merge, and output projection) together
 *   with `advanced_graph_attention_blocks` metadata. Import validates that
 *   shadow structure and preserves it as `_onnxAdvancedGraph.attentionBlocks`
 *   audit data while runtime inference stays on the dense fallback scaffold.
 * - Other non-adjacent feed-forward edges are still preserved as
 *   `advanced_graph_cross_layer_connections` metadata and re-attached on import
 *   as `_onnxAdvancedGraph` audit data, but merge families outside the explicit
 *   residual and concat subset remain on the honest fallback path.
 * - Exact dense/per-neuron initializer aliases can now reuse one canonical
 *   tensor name when metadata is enabled, but that subset is still same-family
 *   only: near-equal, cross-family, transposed, or otherwise non-exact tensors
 *   remain duplicated rather than being silently tied together.
 * - Export-time optimization remains subset-gated: opset-incompatible activations,
 *   unresolved shape arithmetic, and invalid broadcast or axis combinations fail early
 *   or stay on baseline emission rather than widening support claims implicitly.
 * - If fused recurrent metadata is malformed or required recurrent tensors are missing,
 *   the importer falls back to the base layered reconstruction rather than claiming a
 *   generic recurrent import success.
 *
 * Trust boundary (security)
 * -------------------------
 * Treat an `OnnxModel` like a generic JSON payload: do not import untrusted blobs.
 *
 * Common pitfalls
 * ---------------
 * - Layer shape mismatches: inconsistent tensor shapes/metadata can make import fail.
 * - Relaxed export options can produce graphs that are harder to interpret outside
 *   NeatapticTS.
 * - Conv/pool mappings must match actual layer sizes.
 *
 * Minimal round-trip example
 * --------------------------
 *
 * ```ts
 * import Architect from '../../architect';
 * import { exportToONNX, importFromONNX } from './network.onnx';
 *
 * const network = Architect.perceptron(2, 3, 1);
 *
 * const model = exportToONNX(network, { includeMetadata: true });
 * const jsonText = JSON.stringify(model);
 *
 * const restored = importFromONNX(JSON.parse(jsonText) as typeof model);
 * const output = restored.activate([0.2, 0.8]);
 * ```
 */
export default {
  exportToONNX,
  importFromONNX,
};
