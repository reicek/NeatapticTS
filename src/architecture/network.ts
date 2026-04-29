/**
 * Architecture root chapter map and compatibility surface.
 *
 * This root answers the first structural question behind the whole library:
 * what are the durable building blocks a NeatapticTS network is made from
 * before NEAT, training, workers, or ONNX translation start adding policy
 * around it? The answer is a small graph vocabulary. Nodes and connections hold
 * local signal semantics. Groups and layers organize repeated structure.
 * `Network` orchestrates the whole graph. `Architect` offers common presets.
 * The pool chapters keep allocation-heavy hot paths from turning runtime work
 * into needless object churn.
 *
 * Read this folder as a map between two kinds of knowledge. One kind is graph
 * anatomy: what a node, connection, group, or layer means. The other kind is
 * graph orchestration: how those parts become a runnable, mutable, trainable,
 * serializable network. The architecture root exists so a reader can see both
 * at once before diving into the heavier subchapters.
 *
 * That boundary matters because the rest of the repo reuses architecture in
 * several different ways. The NEAT controller mutates and evaluates graphs.
 * The methods chapter defines reusable activation and structural vocabulary.
 * Browser examples step and visualize networks. ONNX import and export
 * translate networks across ecosystems. If the architecture layer is unclear,
 * everything above it feels like policy without a substrate. This root chapter
 * makes the substrate legible first.
 *
 * One helpful mental model is to split the folder into four shelves: graph
 * primitives (`node`, `connection`), composition helpers (`group`, `layer`,
 * `architect`), orchestration (`network`), and runtime-efficiency helpers
 * (`nodePool`, `activationArrayPool`). The flat `src/architecture/*.ts` files
 * are compatibility facades that keep imports stable while the real ownership
 * and longer explanations live in the chaptered subfolders.
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   Architecture[architecture root]:::accent --> Primitives[Node and Connection<br/>graph primitives]:::base
 *   Architecture --> Composition[Group Layer Architect<br/>composition helpers]:::base
 *   Architecture --> Orchestration[Network<br/>runtime orchestration]:::base
 *   Architecture --> Pools[NodePool and ActivationArrayPool<br/>allocation helpers]:::base
 * ```
 *
 * The second useful lens is ownership versus compatibility. This root folder
 * intentionally still exposes flat compatibility files because public imports,
 * tests, and examples rely on them. But those files are no longer the best
 * place to learn the subsystem. They are the stable front desk. The chaptered
 * subfolders are where the real guided tour now lives.
 *
 * ```mermaid
 * flowchart LR
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   Caller[Caller import]:::base --> Facade[Flat compatibility facade]:::accent
 *   Facade --> Chapter[Chapter-owned implementation folder]:::base
 *   Chapter --> Details[Runtime training mutation serialize ONNX details]:::base
 * ```
 *
 * For background reading, the most relevant mathematical ideas are the directed
 * graph itself and the ordering questions that appear once a graph has to be
 * executed. See Wikipedia contributors,
 * [Directed graph](https://en.wikipedia.org/wiki/Directed_graph), and
 * Wikipedia contributors,
 * [Topological sorting](https://en.wikipedia.org/wiki/Topological_sorting),
 * for compact background on the graph and scheduling ideas that sit underneath
 * this chapter's public surface.
 *
 * Example: start from a preset builder when you want architecture first and
 * wiring details second.
 *
 * ```ts
 * const perceptron = new Architect.Perceptron(2, 3, 1);
 * const outputValues = perceptron.activate([0, 1]);
 * ```
 *
 * Example: start from `Network` when you want the orchestration surface and
 * plan to inspect or mutate the graph more directly.
 *
 * ```ts
 * const network = new Network(2, 1);
 * const outputValues = network.activate([0, 1]);
 * ```
 *
 * Practical reading order:
 *
 * 1. Start here for the high-level map of the architecture package.
 * 2. Continue into `network/` for the main orchestration surface and its
 *    runtime, training, mutation, serialization, and ONNX subchapters.
 * 3. Continue into `node/`, `connection/`, `group/`, and `layer/` for the
 *    graph building blocks.
 * 4. Continue into `architect/` for preset builders and into `nodePool/` and
 *    `activationArrayPool/` for allocation-oriented helpers.
 * 5. Treat the flat `src/architecture/*.ts` files as compatibility facades that
 *    keep public imports stable while the implementation lives in narrower,
 *    teachable chapters.
 */
export { default } from './network/network';
export { formatConstructSummary } from './network/construct/network.construct.summary.utils';
export type {
  ConstructDiagnostics,
  ConstructGraphConnectionSummary,
  ConstructGraphNodeSummary,
  ConstructGraphSnapshot,
  ConstructOptions,
  ConstructPart,
  ConstructResult,
} from './network/construct/network.construct.utils.types';
