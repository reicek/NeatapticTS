import {
  activate as activateImpl,
  gaussianRand as gaussianRandImpl,
} from './activate/network.activate.utils';
import * as activateUtilsImpl from './activate/network.activate.utils';
import { canUseFastSlab as canUseFastSlabImpl } from './slab/network.slab.utils';
import { createMLP as createMLPImpl } from './topology/network.topology.utils';
import {
  getCurrentSparsity as getCurrentSparsityImpl,
  getSparsityBudgetSnapshot as getSparsityBudgetSnapshotImpl,
} from './prune/network.prune.utils';
import { testNetwork as testNetworkImpl } from './stats/network.stats.utils';
import {
  propagate as propagateImpl,
  trainImpl as trainImplImpl,
} from './training/network.training.utils';

/**
 * Activate a network with one input vector and return the resulting output vector while preserving the standard activation semantics used by compatibility-facing runtime callers.
 *
 * @param this - Network instance bound by method call.
 * @param inputs - Input activation vector.
 * @returns Output activation vector.
 */
export const activate = activateImpl;

/**
 * Draw one Gaussian-distributed random sample used by activation noise and related stochastic helpers so randomized routines can share one deterministic distribution utility surface.
 *
 * @returns Random value sampled from a normal-like distribution.
 */
export const gaussianRand = gaussianRandImpl;

/**
 * Report whether the network can safely use the slab fast path under current topology and runtime constraints before callers choose between typed-array and node-traversal execution.
 *
 * @param this - Network instance bound by method call.
 * @returns True when slab fast-path activation is valid.
 */
export const canUseFastSlab = canUseFastSlabImpl;

/**
 * Build a feed-forward multilayer perceptron with the supplied layer-size sequence so callers can quickly bootstrap a deterministic baseline topology without manual wiring.
 *
 * @param this - Network constructor context.
 * @param layerSizes - Ordered input, hidden, and output widths.
 * @returns Constructed feed-forward network.
 */
export const createMLP = createMLPImpl;

/**
 * Return the current runtime sparsity ratio for the active connection graph using the pruning baseline so diagnostics and adaptive policies see consistent density measurements.
 *
 * @param this - Network instance bound by method call.
 * @returns Current sparsity ratio in the closed interval [0, 1].
 */
export const getCurrentSparsity = getCurrentSparsityImpl;

/**
 * Snapshot the configured sparsity budget controls used by pruning and growth guardrails so external telemetry can inspect limits without mutating internal configuration state.
 *
 * @param this - Network instance bound by method call.
 * @returns Immutable view of current sparsity budget settings.
 */
export const getSparsityBudgetSnapshot = getSparsityBudgetSnapshotImpl;

/**
 * Evaluate a network on test samples and return aggregate diagnostics for error-style reporting, including loss metrics needed by validation and benchmarking workflows.
 *
 * @param this - Network instance bound by method call.
 * @param set - Evaluation dataset.
 * @param cost - Optional cost function.
 * @returns Aggregate test diagnostics.
 */
export const testNetwork = testNetworkImpl;

/**
 * Run one backward-pass propagation step for the current network state so gradients, weight updates, and optional momentum behavior are applied through one shared training primitive.
 *
 * @param this - Network instance bound by method call.
 * @param rate - Learning rate.
 * @param momentum - Optional momentum scalar.
 * @param update - Whether to apply updates immediately.
 * @param target - Optional target vector.
 * @returns Nothing.
 */
export const propagate = propagateImpl;

/**
 * Train the network over a dataset with configured iteration, batching, and callback controls while returning summary metrics used by callers for stop-condition and progress decisions.
 *
 * @param this - Network instance bound by method call.
 * @param set - Training dataset.
 * @param options - Training options.
 * @returns Training summary metrics.
 */
export const trainImpl = trainImplImpl;

/** No-trace fast-path activators and raw or batch fallbacks from the activation chapter for high-throughput inference. */
export {
  noTraceActivate,
  activateRaw,
  activateBatch,
} from './activate/network.activate.utils';
/** Forward-windowed activation helpers for sliding-window temporal inference over variable-length input sequences. */
export {
  forwardWindowed,
  forwardWindowedAsync,
} from './window/network.window.utils';
/** Connection creation, batch connection, and disconnection helpers used to build and modify the connection graph. */
export {
  connect,
  connectBatch,
  disconnect,
} from './connect/network.connect.utils';
/** RNG seed, snapshot, restore, and accessor helpers for deterministic replay. */
export {
  setSeed,
  snapshotRNG,
  restoreRNG,
  getRNGState,
  setRNGState,
  getRandomFn,
} from './deterministic/network.deterministic.utils';
/** Evolutionary generation step used by Neat to advance one genome. */
export { evolveNetwork } from './evolve/network.evolve.utils';
/** Connection gating (add gate) and ungating (remove gate) helpers. */
export { gate, ungate } from './gating/network.gating.utils';
/** Crossover operator: splice two parent genomes into a child network. */
export { crossOver } from './genetic/network.genetic.utils';
/** Public implementation of the NEAT add-node-between structural mutation. */
export { addNodeBetweenImpl } from './mutate/network.mutate.public.utils';
/** Core mutation dispatcher: routes a mutation operator request to the correct handler. */
export { mutateImpl } from './mutate/network.mutate.utils';
/** Sparsity budget configuration, opportunistic pruning, and immediate pruning helpers. */
export {
  configureSparsityBudget,
  maybePrune,
  pruneToSparsity,
} from './prune/network.prune.utils';
/** Ensure that a growth budget slot is available before a structural mutation. */
export { ensureGrowthBudget } from './prune/network.prune.budget.utils';
/** Remove a node and rewire or drop its incident connections. */
export { removeNode } from './remove/network.remove.utils';
/** JSON serialize and deserialize helpers for full Network roundtrips. */
export {
  serialize,
  deserialize,
  toJSONImpl,
  fromJSONImpl,
} from './serialize/network.serialize.utils';
/** Clone a network by serializing and immediately deserializing it. */
export { cloneImpl as serializeCloneImpl } from './serialize/network.serialize.public.utils';
/** Typed-array slab management: build, rebuild, activate, inspect, and measure the connection slab. */
export {
  rebuildConnectionSlab,
  rebuildConnectionSlabAsync,
  fastSlabActivate,
  getConnectionSlab,
  getSlabAllocationStats,
} from './slab/network.slab.utils';
/** Generate a self-contained inference function string from a trained network. */
export { generateStandalone } from './standalone/network.standalone.utils';
/** Compute L1/L2 regularization statistics for observability and training callbacks. */
export { getRegularizationStats } from './stats/network.stats.utils';
/** Topological sort, path-existence check, and connection-map rebuild helpers. */
export {
  computeTopoOrder,
  hasPath,
  rebuildConnections,
} from './topology/network.topology.utils';
/** Architecture description and descriptor resolution helpers. */
export {
  describeArchitecture,
  resolveArchitectureDescriptor,
} from './topology/network.topology.architecture.utils';
/** Temporal recurrent structure descriptor helpers (LSTM, GRU, NARX structure detection). */
export { describeTemporalStructure } from './network.temporal.extensions.utils';

export {
  applyGradientClippingImpl,
  clearState,
  trainSetImpl,
  __trainingInternals,
} from './training/network.training.utils';
export { removeNode as gatingRemoveNode } from './gating/network.gating.utils';

/**
 * Re-export the activation helper namespace used by the Network facade for forward-pass and activation-buffer policy.
 */
export const activateUtils = activateUtilsImpl;
/** Connection creation, batching, and disconnection helpers used by the Network facade to build and modify the runtime connection graph safely. */
export * as connectUtils from './connect/network.connect.utils';
/** Deterministic RNG seed, snapshot, restore, and state-accessor helpers for reproducible multi-run and replay workflows across sessions. */
export * as deterministicUtils from './deterministic/network.deterministic.utils';
/** One-genome evolutionary step, fitness evaluation loop, and population-advance helpers consumed by the evolve chapter and Neat controller. */
export * as evolveUtils from './evolve/network.evolve.utils';
/** Connection gating, ungating, and gater-management helpers used by the Network facade to add and remove gated connections atomically. */
export * as gatingUtils from './gating/network.gating.utils';
/** Crossover operator helpers for splicing two parent genomes into a child network used by NEAT reproductive passes and heredity flows. */
export * as geneticUtils from './genetic/network.genetic.utils';
/** Public structural mutation helpers for the add-node-between NEAT operator exposed through the Network facade as a first-class surface. */
export * as mutatePublicUtils from './mutate/network.mutate.public.utils';
/** Core mutation dispatcher and batch structural mutation helpers that route every operator request to the correct per-type handler. */
export * as mutateUtils from './mutate/network.mutate.utils';
/** ONNX export build-phase helpers: tensor wiring, op-node assembly, and graph-attribute configuration for the core export step. */
export * as onnxExportBuildUtils from './onnx/export/network.onnx.export-build.utils';
/** ONNX export conv-layer helpers that emit 2-D convolution op-nodes and their supporting initializer tensors into the graph. */
export * as onnxExportConvUtils from './onnx/export/layers/network.onnx.export-conv.utils';
/** ONNX export dense-layer helpers that emit Gemm and MatMul op-nodes for fully connected and projection network sections. */
export * as onnxExportDenseUtils from './onnx/export/layers/network.onnx.export-dense.utils';
/** Shared layer-common helpers used across all ONNX export layer families for axis handling and per-op invariant enforcement. */
export * as onnxExportLayerCommonUtils from './onnx/export/layers/network.onnx.export-layer-common.utils';
/** Graph-topology helpers used during ONNX export to resolve layer ordering, value-info shapes, and inter-layer edge wiring. */
export * as onnxExportLayerGraphUtils from './onnx/export/layers/network.onnx.export-layer-graph.utils';
/** ONNX export orchestration entry-points that drive the full multi-step export pipeline from a trained network to a serialized payload. */
export * as onnxExportOrchestratorsUtils from './onnx/export/network.onnx.export-orchestrators.utils';
/** Post-processing helpers applied after the core ONNX export build step to finalize node names and output type descriptors. */
export * as onnxExportPostprocessUtils from './onnx/export/network.onnx.export-postprocess.utils';
/** ONNX recurrent-layer export helpers for LSTM and GRU op-nodes including state initialization and sequence-axis wiring contracts. */
export * as onnxExportRecurrentUtils from './onnx/export/layers/network.onnx.export-recurrent.utils';
/** ONNX export setup helpers that initialize graph metadata, model-version fields, and opset declarations before build begins. */
export * as onnxExportSetupUtils from './onnx/export/network.onnx.export-setup.utils';
/** ONNX import activation-function resolution helpers that map ONNX activation attribute strings to library squash functions consistently. */
export * as onnxImportActivationsUtils from './onnx/import/network.onnx.import-activations.utils';
/** ONNX import fused-recurrent helpers that reconstruct LSTM and GRU blocks from fused ONNX op-node payloads during import. */
export * as onnxImportFusedRecurrentUtils from './onnx/import/network.onnx.import-fused-recurrent.utils';
/** ONNX import orchestration entry-points that drive the full multi-step import pipeline from a parsed ONNX payload to a Network. */
export * as onnxImportOrchestratorsUtils from './onnx/import/network.onnx.import-orchestrators.utils';
/** ONNX import weight-extraction helpers that map initializer tensors to network connection weight and bias targets precisely. */
export * as onnxImportWeightsUtils from './onnx/import/network.onnx.import-weights.utils';
/** ONNX layer-analysis helpers that classify op-node patterns and report supported-subset membership for guided import decisions. */
export * as onnxLayerAnalysisUtils from './onnx/network.onnx.layer-analysis.utils';
/** ONNX runtime-load helpers that detect and lazily initialize the optional onnxruntime-node or onnxruntime-web inference backend. */
export * as onnxRuntimeLoadUtils from './onnx/import/network.onnx.runtime-load.utils';
/** Top-level ONNX utilities including opset declarations, format version tags, and shared ONNX type constants used across import and export. */
export * as onnxUtils from './onnx/network.onnx.utils';
/** Opportunistic and threshold-based pruning helpers that enforce configured sparsity budgets on the active connection graph incrementally. */
export * as pruneUtils from './prune/network.prune.utils';
/** Growth-budget tracking helpers that gate structural mutations when the current topology exceeds its configured capacity ceiling. */
export * as pruneBudgetUtils from './prune/network.prune.budget.utils';
/** Node-removal helpers that unwire a node from the graph and optionally rewire or drop its bridged incident connection paths. */
export * as removeUtils from './remove/network.remove.utils';
/** Public clone and serialization helpers providing the user-facing Network.clone, toJSON, and JSON roundtrip API surface. */
export * as serializePublicUtils from './serialize/network.serialize.public.utils';
/** Core JSON-based serialize and deserialize helpers for full Network roundtrip persistence, migration, and compact format paths. */
export * as serializeUtils from './serialize/network.serialize.utils';
/** Slab adjacency helpers that rebuild the compact adjacency index used by the fast-path activation scheduling pass. */
export * as slabAdjacencyHelpersUtils from './slab/network.slab.adjacency.helpers.utils';
/** Slab fast-path helpers that execute typed-array activation sweeps through the connection slab without object-traversal overhead. */
export * as slabFastPathHelpersUtils from './slab/network.slab.fast-path.helpers.utils';
/** Slab pool helpers that manage typed-array buffer allocation and reclamation lifecycle for the connection slab layer. */
export * as slabPoolUtils from './slab/network.slab.pool.utils';
/** Slab rebuild helpers that wire typed-array weight buffers from the current connection graph into the flat slab layout. */
export * as slabRebuildHelpersUtils from './slab/network.slab.rebuild.helpers.utils';
/** Shared slab helpers used across adjacency, fast-path, and rebuild layers for consistent buffer-sizing and stride calculations. */
export * as slabSharedHelpersUtils from './slab/network.slab.shared.helpers.utils';
/** Public slab management surface: build, rebuild, activate, inspect, and measure the typed-array connection slab end-to-end. */
export * as slabUtils from './slab/network.slab.utils';
/** Standalone inference-function code-generation helpers that emit a self-contained JS string from a trained and frozen network. */
export * as standaloneUtils from './standalone/network.standalone.utils';
/** Network evaluation statistics helpers that compute aggregate loss and diagnostic metrics for test and benchmark workflows. */
export * as statsUtils from './stats/network.stats.utils';
/** Topological sort, path-existence check, and connection-map rebuild helpers used by activation scheduling and mutation guards. */
export * as topologyUtils from './topology/network.topology.utils';
/** Gradient-descent training helpers including propagation, learning-rate scheduling, momentum, and dataset-batch iteration. */
export * as trainingUtils from './training/network.training.utils';
