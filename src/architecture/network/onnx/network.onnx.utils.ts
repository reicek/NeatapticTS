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
 *
 * Scope & Assumptions (current):
 *  - Network must be strictly layered and acyclic (feed‑forward between layers; optional self recurrence within
 *    hidden layers when enabled).
 *  - Homogeneous activation per layer unless `allowMixedActivations` is true (then per-neuron decomposition used).
 *  - Only a minimal ONNX tensor / node subset is emitted (no external ONNX proto dependency; pure JSON shape).
 *  - Recurrent support limited to: (a) self-connections mapped to diagonal Rk matrices (single step),
 *    (b) experimental fused LSTM/GRU heuristics relying on equal partition patterns (not spec-complete).
 *  - LSTM / GRU biases currently single segment (Wb only) and recurrent bias (Rb) implicitly zero; ordering of
 *    gates documented in code comments (may differ from canonical ONNX gate ordering and will be normalized later).
 *
 * Metadata Keys (may appear in `model.metadata_props` when `includeMetadata` true):
 *  - `layer_sizes`: JSON array of hidden layer sizes.
 *  - `recurrent_single_step`: JSON array of 1-based hidden layer indices with exported self recurrence.
 *  - `lstm_groups_stub`: Heuristic grouping stubs for prospective LSTM layers (pre-emission discovery data).
 *  - `lstm_emitted_layers` / `gru_emitted_layers`: Arrays of export-layer indices where fused nodes were emitted.
 *  - `rnn_pattern_fallback`: Records near-miss pattern sizes for diagnostic purposes.
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
 * NOT supported. Experimental fused recurrent nodes are best-effort and may silently degrade if shapes mismatch.
 */

import type Network from '../../network/network';
import type NeatapticNode from '../../node';
import type { OnnxExportOptions } from './network.onnx.utils.types';
import type { OnnxModel } from './schema/network.onnx.schema.types';
export type { OnnxModel } from './schema/network.onnx.schema.types';

export {
  inferLayerOrdering,
  rebuildConnectionsLocal,
  validateLayerHomogeneityAndConnectivity,
} from './network.onnx.layer-analysis.utils';
export { runOnnxExportFlow } from './export/network.onnx.export-flow.utils';
export { runOnnxImportFlow } from './import/network.onnx.import-flow.utils';
import { buildOnnxModel as buildOnnxModelImpl } from './export/network.onnx.export-build.utils';
export { assignActivationFunctions } from './import/network.onnx.import-activations.utils';
export {
  assignWeightsAndBiases,
  deriveHiddenLayerSizes,
} from './import/network.onnx.import-weights.utils';
export {
  applyModelMetadata,
  collectRecurrentLayerIndices,
  createBaseModel,
  createGraphDimensions,
} from './export/network.onnx.export-setup.utils';
export { emitLayerGraph } from './export/layers/network.onnx.export-layer-graph.utils';
export {
  emitFusedRecurrentHeuristics,
  finalizeExportMetadata,
} from './export/network.onnx.export-postprocess.utils';

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
