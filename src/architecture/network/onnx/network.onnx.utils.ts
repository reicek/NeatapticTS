/**
 * ONNX export/import utilities for a constrained, documented subset of networks.
 *
 * Phase Coverage (incremental roadmap implemented so far):
 *  - Phase 1: Deterministic layered MLP export (Gemm + Activation pairs) with basic metadata.
 *  - Phase 2: Optional partial connectivity (missing edges -> 0 weight) and mixed per-neuron activations
 *              (decomposed into per-neuron Gemm + Activation + Concat) via `allowPartialConnectivity` /
 *              `allowMixedActivations`.
 *  - Phase 3 (baseline): Multi-layer self‑recurrence single‑step representation (`allowRecurrent` +
 *              `recurrentSingleStep`) adding per-recurrent-layer previous state inputs and diagonal R matrices.
 *  - Phase 3 (experimental extension): Heuristic detection + emission of simplified LSTM / GRU fused nodes
 *              (no sequence axis, simplified bias & recurrence handling) while retaining original Gemm path.
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
 * Limitations / TODO (tracked for later phases):
 *  - Proper ONNX-compliant LSTM/GRU biases (split Wb/Rb) & complete gate ordering alignment.
 *  - Pruning or replacing redundant Gemm graph segments when fused recurrent ops are emitted (currently both kept).
 *  - Multi-time-step sequence handling (currently single-step recurrent representation only).
 *  - Richer recurrence (off-diagonal intra-layer connectivity) and gating reconstruction fidelity.
 *
 * NOTE: Import is only guaranteed to work for models produced by {@link exportToONNX}; arbitrary ONNX graphs are
 * NOT supported. Experimental fused recurrent nodes are best-effort and may silently degrade if shapes mismatch.
 */

import type Network from '../../network';
import type NeatapticNode from '../../node';
import type { OnnxExportOptions, OnnxModel } from './network.onnx.utils.types';
export type { OnnxModel } from './network.onnx.utils.types';

export {
  inferLayerOrdering,
  rebuildConnectionsLocal,
  validateLayerHomogeneityAndConnectivity,
} from './network.onnx.layer-analysis.utils';
import { buildOnnxModel as buildOnnxModelImpl } from './network.onnx.export-build.utils';
export { assignActivationFunctions } from './network.onnx.import-activations.utils';
export {
  assignWeightsAndBiases,
  deriveHiddenLayerSizes,
} from './network.onnx.import-weights.utils';
export {
  applyModelMetadata,
  collectRecurrentLayerIndices,
  createBaseModel,
  createGraphDimensions,
} from './network.onnx.export-setup.utils';
export { emitLayerGraph } from './network.onnx.export-layer-graph.utils';
export {
  emitFusedRecurrentHeuristics,
  finalizeExportMetadata,
} from './network.onnx.export-postprocess.utils';

// ---------------------------------------------------------------------------
// Helper functions consumed by network.onnx.ts
// ---------------------------------------------------------------------------

/**
 * Build ONNX graph for validated layered network (barrel forwarder).
 *
 * @param network Source network.
 * @param layers Layered nodes.
 * @param options Export options.
 * @returns ONNX model.
 */
export function buildOnnxModel(
  network: Network,
  layers: NeatapticNode[][],
  options: OnnxExportOptions = {},
): OnnxModel {
  return buildOnnxModelImpl(network, layers, options);
}

// This module intentionally serves as a compatibility barrel + thin forwarders.
