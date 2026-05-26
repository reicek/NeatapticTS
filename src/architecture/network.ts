/**
 * Architecture root — the graph substrate every network is built from.
 *
 * ## A Neural Network as a Directed Graph
 *
 * At its mathematical core, a neural network is a **weighted directed graph**.
 * Each *node* (neuron) receives a weighted sum of its incoming signals, applies
 * a non-linear activation function, and sends its output along outgoing
 * connections to the next layer of nodes. The forward-propagation formula for
 * a single node is:
 *
 * ```
 * output = activation( Σᵢ wᵢ · xᵢ  +  bias )
 * ```
 *
 * where xᵢ are the inputs arriving through connections, wᵢ are the
 * connection weights, and `activation` is a non-linear function (ReLU, tanh,
 * etc.) that gives the network its representational power.
 *
 * In NEAT, this graph is not fixed in advance. Nodes and connections are
 * added by structural mutations over many generations, so the architecture
 * layer must support both *fixed-topology training* and *dynamic structural
 * growth*. See Wikipedia contributors,
 * [Artificial neural network](https://en.wikipedia.org/wiki/Artificial_neural_network),
 * for an introduction to the graph model, and Wikipedia contributors,
 * [Topological sorting](https://en.wikipedia.org/wiki/Topological_sorting),
 * for the scheduling problem that arises when nodes must be activated in the
 * correct dependency order.
 *
 * ## The Building Blocks
 *
 * The architecture layer is organized around four shelves:
 *
 * - **Graph primitives** (`node`, `connection`) — the atoms of every network.
 *   A `Node` holds activation state, bias, and a reference to its activation
 *   function. A `Connection` holds the weight, innovation number, and gate flag.
 * - **Composition helpers** (`group`, `layer`, `architect`) — tools for
 *   building structured networks from groups of nodes, without wiring each
 *   connection by hand.
 * - **Orchestration** (`network`) — the facade that owns activation, training,
 *   mutation, serialization, ONNX export, and slab-optimized forward passes.
 * - **Allocation helpers** (`nodePool`, `activationArrayPool`) — typed-array
 *   pools that keep hot evaluation paths free of per-step object allocation.
 *
 * The flat `src/architecture/*.ts` files are compatibility facades. They keep
 * public imports stable while the real implementation and longer explanations
 * live in the chaptered subfolders.
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#001522,stroke:#0fb5ff,color:#9fdcff,stroke-width:1.5px;
 *   classDef accent fill:#0f1f33,stroke:#00e5ff,color:#d8f6ff,stroke-width:2px;
 *
 *   Architecture["architecture root"]:::accent --> Primitives["Node · Connection\ngraph atoms"]:::base
 *   Architecture --> Composition["Group · Layer · Architect\ncomposition helpers"]:::base
 *   Architecture --> Orchestration["Network\nactivate · train · mutate · serialize · ONNX"]:::base
 *   Architecture --> Pools["NodePool · ActivationArrayPool\nallocation helpers"]:::base
 *   Orchestration --> Slab["slab/\ntyped-array fast path"]:::base
 * ```
 *
 * ## Practical Reading Order
 *
 * 1. `network/` — start here for the orchestration surface and its subchapters.
 * 2. `node/` and `connection/` — graph building blocks and their state semantics.
 * 3. `group/`, `layer/`, `architect/` — composition and preset builder helpers.
 * 4. `nodePool/` and `activationArrayPool/` — allocation-efficient runtime paths.
 *
 * Example: build a preset feed-forward network and run one forward pass.
 *
 * ```ts
 * const perceptron = new Architect.Perceptron(2, 3, 1);
 * const outputValues = perceptron.activate([0, 1]);
 * ```
 *
 * Example: start from `Network` directly when you want to inspect or mutate
 * the graph yourself.
 *
 * ```ts
 * const network = new Network(2, 1);
 * const outputValues = network.activate([0, 1]);
 * ```
 */
import {
  createInferencePredictor as createInferencePredictorImpl,
  exportPortableInferencePayload as exportPortableInferencePayloadImpl,
  exportTransferableInferencePayload as exportTransferableInferencePayloadImpl,
  extractNetworkInferenceIR as extractNetworkInferenceIRImpl,
} from './network/worker-payload/network.worker-payload';
import type { ConstructResult as ConstructResultType } from './network/construct/network.construct.utils.types';
import type {
  InferenceChannelOptions as InferenceChannelOptionsType,
  NetworkInferenceIREdge as NetworkInferenceIREdgeType,
  PortableInferencePayloadEdge as PortableInferencePayloadEdgeType,
  TransferableInferencePayloadOptions as TransferableInferencePayloadOptionsType,
} from './network/worker-payload/network.worker-payload';
import type { VisualizationEdgeV1 as VisualizationEdgeV1Type } from './network/visualization/network.visualization.types';

/**
 * Create a lightweight inference predictor from an exported payload so runtime scoring can run without reconstructing a full mutable network instance, especially in worker and browser inference contexts.
 *
 * @param payload - Portable or transferable inference payload.
 * @returns Predictor object that exposes forward inference helpers.
 */
export const createInferencePredictor = createInferencePredictorImpl;

/**
 * Export a portable inference payload that can be serialized, persisted, and reused outside the live network instance while preserving deterministic node-edge execution semantics for later prediction.
 *
 * @param network - Source network.
 * @param options - Optional export controls.
 * @returns JSON-friendly portable payload.
 */
export const exportPortableInferencePayload =
  exportPortableInferencePayloadImpl;

/**
 * Export a transferable inference payload optimized for worker-message transport and typed-array transfer lists so large numeric buffers can move across threads with minimal copying overhead.
 *
 * @param network - Source network.
 * @param options - Transfer export options.
 * @returns Transfer-oriented payload with buffers.
 */
export const exportTransferableInferencePayload =
  exportTransferableInferencePayloadImpl;

/**
 * Extract the intermediate inference graph representation used by payload export and worker runtime prediction paths, providing a stable edge-node IR contract for downstream transport helpers.
 *
 * @param network - Source network.
 * @returns Inference graph intermediate representation.
 */
export const extractNetworkInferenceIR = extractNetworkInferenceIRImpl;

/**
 * Structured result returned from network construction helpers, including constructed graph artifacts and metadata used by diagnostics, tests, and architecture-inspection workflows.
 */
export type ConstructResult = ConstructResultType;

/**
 * Options for opening an inference channel that streams request batches to a worker-backed predictor, including transport and lifecycle controls needed for stable long-running sessions.
 */
export type InferenceChannelOptions = InferenceChannelOptionsType;

/**
 * Directed edge entry in the inference intermediate representation graph that links source and target node identifiers with weight metadata required by predictor execution.
 */
export type NetworkInferenceIREdge = NetworkInferenceIREdgeType;

/**
 * Edge record inside a portable inference payload that preserves connection identity, direction, and numeric parameters across serialization and runtime reconstruction boundaries.
 */
export type PortableInferencePayloadEdge = PortableInferencePayloadEdgeType;

/**
 * Options that control transferable payload packing and transfer-list shaping so callers can tune memory ownership, buffer transfer behavior, and channel compatibility.
 */
export type TransferableInferencePayloadOptions =
  TransferableInferencePayloadOptionsType;

/**
 * Edge schema used by visualization graph exports so rendering tools can consume connection direction, style, and metadata consistently across browser and offline documentation views.
 */
export type VisualizationEdgeV1 = VisualizationEdgeV1Type;

export { default } from './network/network';
export { formatConstructSummary } from './network/construct/network.construct.summary.utils';
export {
  exportVisualizationGraph,
  toDot,
} from './network/visualization/network.visualization';
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
} from './network/worker-payload/network.worker-payload';
export type {
  AutoInferenceTransport,
  BatchEvaluationResult,
  BrowserWorkerAssetUrlOptions,
  EvaluateInWorkersOptions,
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
} from './network/worker-payload/network.worker-payload';
export type {
  ExportVisualizationOptions,
  VisualizationGraphV1,
  VisualizationIOV1,
  VisualizationMetadataV1,
  VisualizationNodeV1,
} from './network/visualization/network.visualization.types';
export type {
  ConstructDiagnostics,
  ConstructGraphConnectionSummary,
  ConstructGraphNodeSummary,
  ConstructGraphSnapshot,
  ConstructOptions,
  ConstructPart,
} from './network/construct/network.construct.utils.types';
