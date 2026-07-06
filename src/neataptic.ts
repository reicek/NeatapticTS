/**
 * Root public entry point for the NeatapticTS library.
 *
 * Import everything you need for NEAT-based neuroevolution from this single
 * surface. The library organizes its exports into five cooperating layers:
 *
 * - **`Neat`** — the NEAT evolutionary controller: population management,
 *   speciation, selection, mutation, and crossover.
 * - **`Network`** — the mutable graph: activation, training, structural
 *   editing, serialization, and ONNX export.
 * - **Primitives** — `Node`, `Connection`, `Group`, `Layer`, `Architect` for
 *   hand-assembling custom architectures.
 * - **Namespaces** — `methods` (activation functions, cost functions, mutation
 *   and selection operators), `config` (global library settings), `multi`
 *   (worker-thread parallel evaluation).
 * - **`nge`** — experimental Genesis EvoDevo (NGE) lifecycle namespace. A narrow,
 *   unstable preview of adult staging, juvenile growth, lifecycle
 *   orchestration, and assimilation write-back. Not covered by the stable
 *   public API contract.
 *
 * The NEAT algorithm behind the `Neat` controller was introduced by Stanley
 * and Miikkulainen,
 * [Evolving Neural Networks through Augmenting Topologies](https://nn.cs.utexas.edu/?stanley:ec02).
 *
 * ```mermaid
 * flowchart LR
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *   classDef entry fill:#0f1f10,stroke:#39d353,color:#d4fcd7,stroke-width:1.5px;
 *   classDef experimental fill:#1a0f1a,stroke:#ff6b9d,color:#ffd6e5,stroke-width:1.5px;
 *
 *   neataptic[neataptic.ts root entry]:::entry
 *   neataptic --> Neat[Neat evolutionary controller]:::accent
 *   neataptic --> Network[Network graph facade]:::accent
 *   neataptic --> Primitives[Node Connection Group Layer Architect]:::base
 *   neataptic --> Namespaces[methods config multi namespaces]:::base
 *   neataptic --> nge[nge experimental namespace]:::experimental
 *   Neat --> Network
 *   Network --> Workers[worker inference transport]:::base
 *   Network --> ONNX[ONNX export and import]:::base
 *   nge --> adult[adult lifecycle]:::experimental
 *   nge --> juvenile[juvenile growth]:::experimental
 *   nge --> lifecycle[lifecycle runner]:::experimental
 *   nge --> assimilation[assimilation write-back]:::experimental
 * ```
 *
 * @example
 * ```ts
 * import { Neat, Network, methods, config } from 'neataptic';
 *
 * // Configure global settings
 * config.backend = 'float32';
 *
 * // Build a feed-forward network directly
 * const network = Network.createMLP(2, [4], 1);
 * const output = network.activate([0.5, 0.8]);
 *
 * // Run NEAT evolution on a population
 * const neat = new Neat(2, 1, fitnessFunction, { popsize: 50 });
 * await neat.evolve();
 * ```
 *
 * @module neataptic
 */
export { default as Neat } from './neat';
export { default as Network } from './architecture/network';
export { formatConstructSummary } from './architecture/network';
export { exportVisualizationGraph, toDot } from './architecture/network';
export {
  createNeatParallelPopulationEvaluator,
  detectInferenceWorkerCapabilities,
  evaluateInWorkers,
  getTransferList,
  INFERENCE_ACTIVATION_TABLE,
  openInferenceChannel,
  openSharedInferenceWorker,
  ParallelInferencePool,
  resolveAutoInferenceTransport,
  resolveBrowserWorkerAssetUrl,
  SHARED_INFERENCE_REQUIRES_CROSS_ORIGIN_ISOLATION,
} from './architecture/network';
import {
  createInferencePredictor as createInferencePredictorImpl,
  extractNetworkInferenceIR as extractNetworkInferenceIRImpl,
  exportPortableInferencePayload as exportPortableInferencePayloadImpl,
  exportTransferableInferencePayload as exportTransferableInferencePayloadImpl,
} from './architecture/network';
/**
 * Create an inference predictor adapter so callers can execute compiled network inference repeatedly with transport-aware worker or local execution backends.
 */
export const createInferencePredictor = createInferencePredictorImpl;
/**
 * Extract network inference intermediate representation so worker transport helpers and portable payload exporters can serialize deterministic execution structures without requiring the original mutable network instance at runtime.
 */
export const extractNetworkInferenceIR = extractNetworkInferenceIRImpl;
/**
 * Export a portable inference payload so non-transferable runtimes can hydrate prediction graphs from plain structured-clone-safe objects while preserving stable node-edge ordering for reproducible inference behavior.
 */
export const exportPortableInferencePayload =
  exportPortableInferencePayloadImpl;
/**
 * Export a transferable inference payload so worker channels can move typed-array-heavy inference data efficiently between threads with explicit ownership transfer and minimal serialization overhead.
 */
export const exportTransferableInferencePayload =
  exportTransferableInferencePayloadImpl;
export type {
  AutoInferenceTransport,
  BatchEvaluationResult,
  BrowserWorkerAssetUrlOptions,
  EvaluateInWorkersOptions,
  ExportVisualizationOptions,
  NeatParallelPopulationEvaluatorOptions,
  InferenceChannel,
  InferencePredictor,
  InferenceWorkerCapabilities,
  InferenceWorkerCapabilityOptions,
  NetworkInferenceIR,
  NetworkInferenceIRNode,
  ParallelInferencePoolOptions,
  ParallelInferenceWorkerLike,
  PortableInferencePayload,
  PortableInferencePayloadNode,
  SharedInferenceWorker,
  SharedInferenceWorkerOptions,
  TransferableInferencePayload,
  VisualizationGraphV1,
  VisualizationIOV1,
  VisualizationMetadataV1,
  VisualizationNodeV1,
} from './architecture/network';
/**
 * Configure inference channel behavior so request multiplexing, worker lifecycle transitions, and batching semantics remain explicit for asynchronous prediction clients across browser and node worker transports.
 */
export type InferenceChannelOptions =
  import('./architecture/network').InferenceChannelOptions;
/**
 * Describe one inference-graph edge in the network intermediate representation so transport, replay, and debugging tools can reconstruct connectivity deterministically across payload and channel boundaries.
 */
export type NetworkInferenceIREdge =
  import('./architecture/network').NetworkInferenceIREdge;
/**
 * Describe one portable payload edge so serialized inference artifacts can preserve weighted topology connections and directional metadata in runtime-agnostic JSON form.
 */
export type PortableInferencePayloadEdge =
  import('./architecture/network').PortableInferencePayloadEdge;
/**
 * Configure transferable payload export so callers can choose cloning and ownership strategies for typed buffers crossing worker boundaries in high-throughput prediction scenarios.
 */
export type TransferableInferencePayloadOptions =
  import('./architecture/network').TransferableInferencePayloadOptions;
/**
 * Describe one visualization edge record so renderers can draw weighted directed links with consistent metadata across interactive, static, and documentation-oriented UI surfaces.
 */
export type VisualizationEdgeV1 =
  import('./architecture/network').VisualizationEdgeV1;
export {
  fromParameterVector,
  toParameterVector,
} from './architecture/network/serialize/network.serialize.utils';
export type {
  ParameterLayoutEntry,
  ParameterLayoutV1,
  ParameterVector,
} from './architecture/network/serialize/network.serialize.utils.types';
export { fineTuneVector } from './architecture/network/training/network.training.isolate.utils';
export type {
  FineTuneOptions,
  FineTuneResult,
} from './architecture/network/training/network.training.isolate.utils';
export { evaluateCandidate } from './neat/hybrid/neat.hybrid';
export type {
  EvaluateCandidateOptions,
  HybridEvaluationPolicy,
  HybridEvaluationResult,
  HybridFineTuneMode,
  HybridScoreNetwork,
} from './neat/hybrid/neat.hybrid.types';
export { default as Node } from './architecture/node';
export { default as Layer } from './architecture/layer';
export { default as Group } from './architecture/group';
export { default as Connection } from './architecture/connection';
export { default as Architect } from './architecture/architect';
/** Activation, cost, crossover, mutation, and selection method objects. Stateless algorithm namespaces for use with {@link Network} and {@link Neat}. */
export * as methods from './methods/methods';
/** Global library configuration namespace. Controls backend precision, debug flags, and runtime behavior. */
export * as config from './config';
/** Worker-thread parallel genome population evaluation utilities for Node.js runtime environments. */
export * as multi from './multithreading/multi';
/**
 * Experimental NGE lifecycle namespace.
 *
 * Re-exports `adult`, `juvenile`, `lifecycle`, and `assimilation` sub-namespaces
 * from a single unstable entrypoint. This surface may change or be removed
 * without a major version bump; prefer `Neat` and `Network` for stable work.
 *
 * @experimental
 */
export * as nge from './neat/nge-experimental';
export {
  renderNetworkView,
  positionNetworkNodes,
  centerPositionedNodesInDrawableArea,
  resolveNetworkVisualizationTopologyPlan,
} from './visualization/visualization';
import { resolveNetworkVisualizationLayers as resolveNetworkVisualizationLayersImpl } from './visualization/visualization';
export type {
  PositionedNetworkNode,
  VisualNetworkConnection,
  NetworkNodeDimensions,
  NetworkVisualizationColorScales,
  NetworkVisualizationResolvedFrame,
  OverlayFactoryHooks,
  RenderNetworkViewOptions,
  VisualNetworkNode,
  NetworkLayerAnnotation,
  NetworkVisualizationTopologyPlan,
} from './visualization/visualization';
/**
 * Define drawable edge padding constraints so visualization layout helpers can reserve safe margins around graph links, labels, and annotation overlays across canvas and SVG render targets.
 */
export type EdgePadding = import('./visualization/visualization').EdgePadding;

/**
 * Resolve network visualization layers so renderers receive stable, ordered groups suitable for node placement, edge routing passes, and annotation alignment in deterministic graph layout workflows.
 */
export const resolveNetworkVisualizationLayers =
  resolveNetworkVisualizationLayersImpl;
