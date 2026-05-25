/**
 * ONNX export/import utilities for a constrained, documented subset of networks.
 *
 * This file is the root compatibility barrel for the ONNX execution helpers.
 * It exists so callers can keep a stable import surface while the heavier
 * exporter and importer logic lives in the narrower `export/` and `import/`
 * chapters.
 *
 * How to read this file:
 * - Start here if you want the thin orchestration-facing helpers that still
 *   bridge the public root API to the split implementation files.
 * - Continue into `export/` when you want the full graph-emission pipeline.
 * - Continue into `import/` when you want the reconstruction pipeline.
 * - Treat this file as a compatibility and forwarding surface, not the main
 *   home of ONNX execution details.
 *
 * What still belongs here:
 * - Re-exports that are intentionally stable for root ONNX callers.
 * - Thin wrappers such as `buildOnnxModel()` that preserve a predictable
 *   orchestration surface while delegating the real work into smaller chapters.
 *
 * Current capability set:
 *  - Deterministic layered MLP export (Gemm + Activation pairs) with basic metadata.
 *  - Optional partial connectivity (missing edges -> 0 weight) and mixed per-neuron activations
 *    (decomposed into per-neuron Gemm + Activation + Concat) via `allowPartialConnectivity` /
 *    `allowMixedActivations`.
 *  - Multi-layer self-recurrence single-step representation (`allowRecurrent` + `recurrentSingleStep`)
 *    adding per-recurrent-layer previous state inputs and diagonal R matrices.
 *  - Experimental: heuristic detection and emission of simplified LSTM / GRU fused nodes
 *    (no sequence axis, simplified bias and recurrence handling) while retaining original Gemm path.
 *  - Phase 5 seam preparation: non-adjacent feed-forward edges are preserved as
 *    `advanced_graph_cross_layer_connections` metadata and re-attached on import
 *    as `_onnxAdvancedGraph` audit payloads, without yet promoting those paths
 *    into explicit residual, concat, or attention reconstruction.
 *  - Phase 5 alias reuse subset: exact dense and per-neuron initializer duplicates
 *    can reuse one canonical tensor name when `includeMetadata` is enabled,
 *    recorded as `shared_initializer_aliases` and re-attached on import as
 *    `_onnxAdvancedGraph.sharedInitializerAliases`, while near-equal or
 *    unsupported-family tensors remain distinct.
 *  - Phase 5 residual subset: dense-family one-hop skip branches now emit an
 *    explicit `Add` merge when exactly one skipped source layer feeds the
 *    target layer, recorded as `advanced_graph_residual_adds` and re-attached
 *    on import as `_onnxAdvancedGraph.residualAdds`, while longer or ambiguous
 *    non-adjacent merges stay on the audit-only fallback path.
 *   - Conservative spatial subset: explicit Conv mappings round-trip through Conv initializers,
 *     while pooling/flatten import stays metadata-only (`_onnxPooling` audit payloads) and
 *     heuristic Conv inference stays metadata-only by default unless
 *     `autoPromoteInferredConv` is enabled and the inferred layer passes the shared-kernel
 *     safety gate for the current proven subset, including conservative
 *     multi-channel layouts, unpooled stacked Conv-like chains, deeper
 *     single-channel post-pool chains whose pooled tensor shapes can be derived
 *     sequentially, and deeper pooled multi-channel chains whose pooled tensor
 *     shapes can be derived sequentially while the exporter keeps the pooled
 *     source compact per channel. The only proven flatten-after-pool promotion
 *     path is the final hidden-stage reshape-bridge subset. Earlier flattened
 *     pooled consumers, repeated flatten-bridge chains, and downstream dense
 *     stages that still depend on extra non-pooled inputs keep the later
 *     inferred stage on the honest fallback path.
 *
 * Scope & Assumptions (current):
 *  - Network must be strictly layered and acyclic (feed‑forward between layers; optional self recurrence within
 *    hidden layers when enabled).
 *  - Homogeneous activation per layer unless `allowMixedActivations` is true (then per-neuron decomposition used).
 *  - Only a minimal ONNX tensor / node subset is emitted (no external ONNX proto dependency; pure JSON shape).
 *  - Cross-layer feed-forward edges are currently audit-only: export records them in metadata,
 *    and import preserves that audit payload while keeping the layered fallback scaffold,
 *    except for the current one-hop residual-add subset.
 *  - Shared initializer alias reuse is currently metadata-gated and limited to
 *    exact dense/per-neuron tensor matches from the same exporter version family.
 *  - Explicit residual-add support is currently limited to homogeneous dense-family
 *    layers with exactly one skipped source layer and same-family import/export.
 *  - Recurrent support limited to: (a) self-connections mapped to diagonal Rk matrices (single step),
 *    (b) experimental fused LSTM/GRU heuristics relying on equal partition patterns (not spec-complete).
 *  - LSTM / GRU biases currently single segment (Wb only) and recurrent bias (Rb) implicitly zero; ordering of
 *    gates documented in code comments (may differ from canonical ONNX gate ordering and will be normalized later).
 *
 * Supported recurrent subset (current):
 *  - Models exported by this repo that use single-step self recurrence on hidden layers.
 *  - Same-family export/import of heuristic LSTM and GRU layers when the emitted `W`, `R`, and `B`
 *    tensors are all present and shape-compatible with the importer.
 *
 * Fallback and rejection boundary:
 *  - Arbitrary external ONNX recurrent graphs are not a supported import target.
 *  - If fused recurrent metadata is malformed, or one of the required `W`, `R`, or `B` tensors is
 *    missing or incompatible, fused reconstruction is skipped and the importer keeps the base layered
 *    reconstruction instead of claiming generic recurrent-graph support.
 *  - Near-miss recurrent shapes can emit `rnn_pattern_fallback` metadata for diagnostics, but that
 *    metadata is not a promise that the graph is an accepted fused recurrent family.
 *
 * Metadata Keys (may appear in `model.metadata_props` when `includeMetadata` true):
 *  - `layer_sizes`: JSON array of hidden layer sizes.
 *  - `recurrent_single_step`: JSON array of 1-based hidden layer indices with exported self recurrence.
 *  - `lstm_groups_stub`: Heuristic grouping stubs for prospective LSTM layers (pre-emission discovery data).
 *  - `lstm_emitted_layers` / `gru_emitted_layers`: Arrays of export-layer indices where fused nodes were emitted.
 *  - `rnn_pattern_fallback`: Records near-miss pattern sizes for diagnostic purposes.
 *  - `conv2d_layers` / `conv2d_specs`: Explicit Conv export mappings for the current spatial subset,
 *    including safety-gated auto-promoted heuristic Conv layers when enabled.
 *  - `conv2d_inferred_layers` / `conv2d_inferred_specs`: Heuristic Conv-like spatial metadata that
 *    remains advisory unless the auto-promotion gate upgrades a layer into real Conv emission.
 *  - `pool2d_layers` / `pool2d_specs` / `flatten_layers`: Pooling and flatten bridge metadata consumed as
 *    import-side audit hints rather than runtime graph rewrites.
 *  - `advanced_graph_cross_layer_connections`: Audit-only records for non-adjacent
 *    feed-forward edges that the current exporter either promotes into the narrow
 *    residual subset or keeps on the fallback path for later concat/attention work.
 *  - `advanced_graph_residual_adds`: Explicit one-hop residual merge records for the
 *    supported dense-family subset. Import uses these together with the residual
 *    branch tensors and cross-layer audit edges to rebuild the skipped connections.
 *  - `shared_initializer_aliases`: Audit-only records mapping reused dense-family
 *    initializer names back to their canonical tensors so import can preserve
 *    exact roundtrip fidelity while unsupported alias families stay duplicated.
 *
 * Design Goals:
 *  - Zero heavy runtime dependencies; the structure is intentionally lightweight & serializable.
 *  - Early, explicit structural validation with actionable error messages.
 *  - Transparent, stepwise transform for testability and deterministic round-tripping.
 *
 * Known limitations:
 *  - LSTM/GRU biases use single-segment Wb only; Rb is implicitly zero and gate ordering may diverge from canonical ONNX.
 *  - Redundant Gemm segments are retained alongside fused recurrent ops rather than pruned.
 *  - Only single-step recurrent representation is supported; multi-time-step sequences are not yet handled.
 *  - Richer recurrence (off-diagonal intra-layer connectivity) and gating reconstruction fidelity.
 *
 * NOTE: Import is only guaranteed to work for models produced by `exportToONNX()`; arbitrary ONNX graphs are
 * NOT supported. The recurrent import promise is intentionally narrow: same-family export/import for the supported
 * subset above, with explicit fallback to the layered baseline when fused recurrent tensors are incomplete.
 */

import type Network from '../../network/network';
import type NeatapticNode from '../../node';
import type { OnnxExportOptions } from './network.onnx.utils.types';
import type { OnnxModel } from './schema/network.onnx.schema.types';
export type { OnnxModel } from './schema/network.onnx.schema.types';

import {
  inferLayerOrdering,
  rebuildConnectionsLocal as rebuildConnectionsLocalImpl,
  validateLayerHomogeneityAndConnectivity as validateLayerHomogeneityAndConnectivityImpl,
} from './network.onnx.layer-analysis.utils';
export { inferLayerOrdering };
export { runOnnxExportFlow } from './export/network.onnx.export-flow.utils';
export { runOnnxImportFlow } from './import/network.onnx.import-flow.utils';
import { buildOnnxModel as buildOnnxModelImpl } from './export/network.onnx.export-build.utils';
import { assignActivationFunctions as assignActivationFunctionsImpl } from './import/network.onnx.import-activations.utils';
import {
  assignWeightsAndBiases as assignWeightsAndBiasesImpl,
  deriveHiddenLayerSizes as deriveHiddenLayerSizesImpl,
} from './import/network.onnx.import-weights.utils';
import {
  applyModelMetadata as applyModelMetadataImpl,
  collectRecurrentLayerIndices as collectRecurrentLayerIndicesImpl,
  createBaseModel as createBaseModelImpl,
  createGraphDimensions as createGraphDimensionsImpl,
} from './export/network.onnx.export-setup.utils';
export { emitLayerGraph } from './export/layers/network.onnx.export-layer-graph.utils';
import {
  emitFusedRecurrentHeuristics as emitFusedRecurrentHeuristicsImpl,
  finalizeExportMetadata as finalizeExportMetadataImpl,
} from './export/network.onnx.export-postprocess.utils';

/**
 * Rebuild local connections from layered ONNX-like data so import flows can restore deterministic adjacency wiring before activation or serialization passes.
 */
export const rebuildConnectionsLocal = rebuildConnectionsLocalImpl;

/**
 * Validate layer homogeneity and connectivity so export and import paths fail early when topology assumptions required by the supported ONNX subset are broken.
 */
export const validateLayerHomogeneityAndConnectivity =
  validateLayerHomogeneityAndConnectivityImpl;

/**
 * Assign activation functions during import reconstruction so each rebuilt layer preserves nonlinear behavior captured by export metadata.
 * This forwarding seam keeps root ONNX callers stable while the concrete activation mapping logic evolves in import internals.
 */
export const assignActivationFunctions = assignActivationFunctionsImpl;

/**
 * Assign weights and biases onto imported layers so reconstructed parameters match serialized ONNX tensor values from export.
 * Keeping this alias documented at the compatibility barrel helps users discover parameter hydration behavior without reading nested modules first.
 */
export const assignWeightsAndBiases = assignWeightsAndBiasesImpl;

/**
 * Derive hidden layer sizes from exported graph structures so import routines allocate correctly shaped intermediate containers.
 * The helper also centralizes size inference assumptions used by reconstruction and compatibility diagnostics.
 */
export const deriveHiddenLayerSizes = deriveHiddenLayerSizesImpl;

/**
 * Apply model metadata to exported artifacts so downstream tools can inspect capability flags and advanced graph hints.
 * This alias preserves a stable public seam for metadata population while implementation details stay split by concern.
 */
export const applyModelMetadata = applyModelMetadataImpl;

/**
 * Collect recurrent layer indices so post-processing can identify hidden stages that use supported single-step recurrence.
 * Export metadata and import diagnostics both depend on this deterministic recurrent-stage inventory.
 */
export const collectRecurrentLayerIndices = collectRecurrentLayerIndicesImpl;

/**
 * Create the base ONNX-like model scaffold so later export stages can append graph nodes, initializers, and metadata in deterministic order.
 */
export const createBaseModel = createBaseModelImpl;

/**
 * Create graph dimension metadata so emitted tensor shapes stay explicit and consistent across exporter and importer paths.
 * Clear dimension records also improve debugging when validating compatibility between serialized tensors and rebuilt layers.
 */
export const createGraphDimensions = createGraphDimensionsImpl;

/**
 * Emit fused recurrent heuristic nodes so eligible LSTM and GRU patterns can be represented compactly within the current conservative subset.
 */
export const emitFusedRecurrentHeuristics = emitFusedRecurrentHeuristicsImpl;

/**
 * Finalize export metadata so generated models include complete capability records and audit hints for compatibility diagnostics.
 * The finalization step normalizes emitted annotations before artifacts are returned to callers or saved.
 */
export const finalizeExportMetadata = finalizeExportMetadataImpl;

// ---------------------------------------------------------------------------
// Helper functions consumed by network.onnx.ts
// ---------------------------------------------------------------------------

/**
 * Build an ONNX-like model from a validated layered network view.
 *
 * Role in the ONNX pipeline:
 * - This function is a thin, stable orchestration boundary used by higher-level exporters.
 * - It forwards to the implementation module while preserving a predictable public API
 *   for callers that import from this compatibility barrel.
 * - Keeping this wrapper explicit helps isolate call sites from internal file splits
 *   and phased refactors in export internals.
 *
 * Expected preconditions:
 * - `layers` has already been inferred from the same `network` instance.
 * - Structural validation (layer homogeneity/connectivity and option gates) is complete.
 * - Export options are normalized by the caller according to project defaults.
 *
 * High-level behavior:
 *  1. Receive network, ordered layer matrix, and export options.
 *  2. Delegate model construction to the concrete builder implementation.
 *  3. Return the resulting ONNX-like JSON graph container unchanged.
 *
 * @param network - Source network to serialize.
 * @param layers - Ordered layer matrix produced by layer inference utilities.
 * @param options - Export options controlling metadata/recurrent/partial-connectivity behavior.
 * @returns ONNX-like model object representing graph nodes, tensors, and metadata.
 *
 * @example
 * ```ts
 * const layers = inferLayerOrdering(network);
 * const model = buildOnnxModel(network, layers, { includeMetadata: true });
 * ```
 */
export function buildOnnxModel(
  network: Network,
  layers: NeatapticNode[][],
  options: OnnxExportOptions = {},
): OnnxModel {
  return buildOnnxModelImpl(network, layers, options);
}

// This module intentionally serves as a compatibility barrel + thin forwarders.
