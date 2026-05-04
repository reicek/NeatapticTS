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
export { default } from './network/network';
export { formatConstructSummary } from './network/construct/network.construct.summary.utils';
export {
  exportVisualizationGraph,
  toDot,
} from './network/visualization/network.visualization';
export type {
  ExportVisualizationOptions,
  VisualizationEdgeV1,
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
  ConstructResult,
} from './network/construct/network.construct.utils.types';
