import type { ActivationPrecision, PrecisionConfig } from '../../config';
import type Network from './network';
import type Node from '../node';
import type Connection from '../connection/connection';
import type { TestWorkerInstance } from '../../multithreading/types';
import type { ConstructResult as ConstructResultContract } from './construct/network.construct.utils.types';
export type {
  ConstructDiagnostics,
  ConstructGraphConnectionSummary,
  ConstructGraphNodeSummary,
  ConstructGraphSnapshot,
  ConstructNodeId,
  ConstructOptions,
  ConstructPart,
  ConstructValidationOptions,
} from './construct/network.construct.utils.types';

/**
 * Structured result contract returned by construct utilities after graph assembly, validation, and diagnostics collation complete.
 * This alias keeps the construct outcome type discoverable from the network root type surface.
 */
export type ConstructResult = ConstructResultContract;

// Keep construct types at the top because they are the most common import path
// for graph-build diagnostics used by tests, docs, and runtime helpers.
export * from './onnx/network.onnx.utils.types';
// Re-export slab types here so network consumers can import allocation contracts
// without reaching into the slab chapter directly.
export * from './slab/network.slab.utils.types';

/**
 * Diagnostic runtime properties optionally attached to Network instances.
 *
 * These underscored fields carry side-channel state for observability,
 * architecture reconstruction, and cross-boundary tooling. They are not
 * required for normal activation or training, but their presence enables
 * richer diagnostics, UI rendering, and checkpoint fidelity.
 */
export interface NetworkRuntimeProps {
  /** Indices of hidden layers skipped by stochastic-depth during the most recent activation. Useful for dropout diagnostics and variance analysis. */
  _lastSkippedLayers?: number[];
  /** Most recent aggregated statistics payload cached after the last `testNetwork` or training iteration, available for quick diagnostics access. */
  _lastStats?: unknown;
  /** Runtime layer cache used by tools that traverse a network as an ordered list of node groups rather than a flat array. */
  layers?: unknown[];
  /** Architecture descriptor deserialized from checkpoint metadata. Allows UI and telemetry tools to reconstruct hidden-layer shape without re-analyzing the live graph. */
  _serializedArchitectureDescriptor?: NetworkArchitectureDescriptor;
  /** Arbitrary extension bag loaded from serialized checkpoints. Keeps forward-compatible metadata accessible without requiring schema changes to the core JSON shape. */
  _serializedExtensions?: NetworkJSONExtensions;
  /** Public topology intent preserved across serialization round-trips so feed-forward enforcement survives checkpointing. */
  _topologyIntent?: NetworkTopologyIntent;
}

/**
 * Provenance of hidden-layer architecture information.
 *
 * - `'layer-metadata'`: sizes were read directly from stored layer objects.
 * - `'graph-topology'`: sizes were inferred by traversing the live graph.
 * - `'inferred'`: sizes were estimated when no authoritative source was available.
 */
export type NetworkArchitectureSource =
  'layer-metadata' | 'graph-topology' | 'inferred';

/**
 * Stable architecture descriptor for UI and telemetry consumers.
 *
 * Hidden-layer sizes are ordered from input-side to output-side. Visualizers
 * and loggers can rely on this snapshot without re-traversing the live graph.
 */
export interface NetworkArchitectureDescriptor {
  /** Hidden-layer widths in forward-pass order (input side to output side). */
  hiddenLayerSizes: number[];
  /** True when the graph contains at least one directed cycle, indicating recurrent or gated structure. */
  hasCycles: boolean;
  /** How hidden-layer sizing was determined â€” from stored layer metadata, graph traversal, or estimation. */
  source: NetworkArchitectureSource;
  /** Total node count across all layers (input, hidden, output). */
  totalNodes: number;
  /** Total connection count in the live graph, including gates and self-connections. */
  totalConnections: number;
}

/** Explicit recurrent-module family tags carried by temporal descriptor snapshots. Each tag identifies the gated-cell variant built by the corresponding `Architect` preset. */
export type NetworkTemporalRecurrentModuleKind = 'lstm' | 'gru' | 'narx-memory';

/**
 * Public snapshot of one validated recurrent module on a runtime network.
 *
 * The role map keeps architecture-aware tooling honest: a visualizer can label
 * LSTM gates or NARX delay shelves directly instead of reverse-engineering the
 * meaning of each hidden node from raw graph topology alone.
 */
export interface NetworkTemporalRecurrentModuleDescriptor {
  /** Stable identity string that survives runtime synchronization and graph edits. Used to correlate descriptors across diagnostic snapshots. */
  moduleId: string;
  /** Recurrent-module family tag (e.g. `'lstm'`, `'gru'`, `'narx-memory'`). Allows visualizers to label gate nodes without re-deriving topology. */
  kind: NetworkTemporalRecurrentModuleKind;
  /** Ordered node gene ids grouped by semantic role (e.g. `inputGate`, `forgetGate`, `cell`). Keys are defined by the builder that created the module. */
  nodeGeneIdsByRole: Record<string, number[]>;
  /** Innovation numbers of connections that define the live module boundary. Stale innovations are pruned during synchronization. */
  connectionInnovations: number[];
  /** Optional human-readable sub-label. Currently used by NARX delay shelves to identify their time-step offset. */
  moduleLabel?: string;
}

/**
 * Public snapshot of one validated gated block on a runtime network.
 *
 * This keeps recurrent-aware tooling free to highlight which gates own which
 * structural edges without exposing the raw private extension bag directly.
 */
export interface NetworkTemporalGatedBlockDescriptor {
  /** Stable block identity that persists across synchronization so tooling can track the same gated structure across diagnostic snapshots. */
  blockId: string;
  /** Gene ids of the nodes acting as gate owners for this block, in the order they were registered. */
  gaterGeneIds: number[];
  /** Innovation numbers of connections gated by this block. Pruned when the live graph no longer includes the corresponding connection. */
  connectionInnovations: number[];
}

/**
 * Public temporal-structure descriptor for one runtime network.
 *
 * Consumers should treat this as a read-only teaching and diagnostics surface.
 * It summarizes validated recurrent modules and gated blocks after stale
 * extension records have been synchronized against the live graph.
 */
export interface NetworkTemporalStructureDescriptor {
  /** Recurrent modules (LSTM, GRU, NARX) validated against the live graph. Stale modules whose innovations no longer exist are removed during synchronization. */
  recurrentModules: NetworkTemporalRecurrentModuleDescriptor[];
  /** Gated blocks validated against the live graph. Used by visualizers and diagnostics to highlight which connections are modulated by gate nodes. */
  gatedBlocks: NetworkTemporalGatedBlockDescriptor[];
}

/**
 * Public topology intent exposed by the network API.
 *
 * Use `feed-forward` when the caller wants the library to preserve an acyclic,
 * forward-only contract. Use `unconstrained` when recurrent, gated, or other
 * cyclic structures may be introduced.
 */
export type NetworkTopologyIntent = 'feed-forward' | 'unconstrained';

/**
 * Explicit ordered node-role metadata stored on a runtime network.
 *
 * The order of each array defines the public input and output vector semantics.
 */
export interface ExplicitIORoles {
  /** Stable gene ids for input-role nodes in public input-vector order. Position in this array defines which slot of the activation input each node reads. */
  inputNodeIds: number[];
  /** Stable gene ids for output-role nodes in public output-vector order. Position in this array defines which slot of the result array each node writes. */
  outputNodeIds: number[];
}

/** Execution mode used by the activation scheduler. `'acyclic'` uses a deterministic Kahn wave schedule; `'recurrent'` permits cycles and uses fixed-iteration SCC unrolling. */
export type ActivationMode = 'acyclic' | 'recurrent';

/**
 * Which execution path the most recent scheduling decision chose.
 *
 * - `'compiled-schedule'`: a deterministic Kahn-ordered schedule was built and is in use.
 * - `'cycle-fallback-order'`: a cycle was detected; the runtime falls back to node-array iteration.
 * - `'raw-node-order'`: no schedule exists; nodes are activated in their raw storage order.
 */
export type ActivationSchedulingExecutionPath =
  'compiled-schedule' | 'cycle-fallback-order' | 'raw-node-order';

/** High-level issue attached to the latest scheduling decision. `'cycle-detected'` means acyclic enforcement found a back-edge; `'schedule-missing'` means topology was dirty and recompilation was needed. `null` means scheduling succeeded cleanly. */
export type ActivationSchedulingIssue =
  'cycle-detected' | 'schedule-missing' | null;

/** Execution step shape inside a compiled activation schedule. `'wave'` steps are plain feed-forward Kahn waves; `'recurrent-component'` steps unroll one strongly-connected component for a fixed iteration count. */
export type ActivationScheduleStepKind = 'wave' | 'recurrent-component';

/** State-handling rule for recurrent schedule execution. `'carry'` means previous activation state is retained across calls, which is the expected semantic for sequence-processing networks. */
export type RecurrentStateSemantics = 'carry';

/** One deterministic activation step inside a compiled Kahn-ordered network activation schedule. */
export interface ActivationScheduleStep {
  /** Whether this step is a plain feed-forward wave or a recurrent SCC boundary that requires fixed-iteration unrolling. */
  kind: ActivationScheduleStepKind;
  /** Stable node gene ids to activate in this step, in deterministic topological order. */
  nodeIds: ReadonlyArray<number>;
  /** Fixed iteration count for recurrent-component steps. Absent for plain wave steps. */
  iterations?: number;
}

/**
 * Deterministic execution schedule for a network graph.
 *
 * Steps preserve deterministic order while distinguishing ordinary waves from
 * recurrent strongly-connected components.
 */
export interface ActivationSchedule {
  /** Execution mode (acyclic or recurrent) used to derive this schedule. */
  mode: ActivationMode;
  /** Ordered execution steps, each covering a Kahn wave or a recurrent SCC. */
  steps: ReadonlyArray<ActivationScheduleStep>;
  /** Stable output-role node gene ids in public output-vector order. */
  outputNodeIds: number[];
  /** How recurrent state is carried between activation calls. Present only when `mode` is `'recurrent'`. */
  stateSemantics?: RecurrentStateSemantics;
}

/**
 * Human-friendly snapshot of the current activation-ordering contract.
 *
 * Activation ordering is the resolved execution story for one network: which
 * mode is active, whether execution is using a compiled schedule or a fallback
 * path, what recurrent state semantics apply, and what callers should do next
 * when a cycle or stale cache prevents the preferred path.
 */
export interface ActivationSchedulingDiagnostics {
  /** Semantic topology contract advertised by the network (`'feed-forward'` or `'unconstrained'`). */
  topologyIntent: NetworkTopologyIntent;
  /** Scheduling mode requested by the current topology contract. */
  requestedMode: ActivationMode;
  /** True when a structural edit has dirtied the scheduling cache and recompilation is pending. */
  topologyDirty: boolean;
  /** Execution path currently selected for activation traversal. */
  executionPath: ActivationSchedulingExecutionPath;
  /** Scheduling issue detected in the last recompilation attempt, or `null` when none. */
  issue: ActivationSchedulingIssue;
  /** Human-readable summary of the scheduling outcome for logging and UI display. */
  message: string;
  /** Stable input-role node ids defining public input-vector semantics. */
  inputNodeIds: number[];
  /** Stable output-role node ids defining public output-vector semantics. */
  outputNodeIds: number[];
  /** Number of steps in the compiled schedule. Zero when no compiled schedule exists. */
  stepCount: number;
  /** Number of recurrent-component steps in the compiled schedule. */
  recurrentComponentCount: number;
  /** Recurrent state policy for the current schedule, or `null` when the schedule is acyclic. */
  stateSemantics: RecurrentStateSemantics | null;
  /** Node ids implicated in a detected cycle, populated only when `issue === 'cycle-detected'`. */
  cycleNodeIds: number[];
  /** Actionable suggestions for callers who want to change the current scheduling outcome. */
  suggestions: string[];
}

/**
 * Public constructor options for `Network`.
 *
 * `topologyIntent` is the semantic, DX-first contract. `enforceAcyclic`
 * remains available for backward compatibility and must not contradict the
 * declared topology intent.
 */
export interface NetworkConstructorOptions {
  /** Minimum number of hidden nodes to pre-populate by node-splitting during construction. Useful when an evolutionary search should start from a non-trivial topology. */
  minHidden?: number;
  /** Deterministic RNG seed. Produces reproducible initial weights and mutation decisions when set. */
  seed?: number;
  /** Low-level acyclic enforcement flag. Deprecated in favor of `topologyIntent`; provided for backward compatibility. Must not contradict a declared topology intent. */
  enforceAcyclic?: boolean;
  /** Semantic topology contract for the network. `'feed-forward'` enforces acyclic structure; `'unconstrained'` allows recurrent and gated connections. */
  topologyIntent?: NetworkTopologyIntent;
  /** Typed-array precision for compiled activation paths and pooled activation buffers. Node runtime state and training traces remain standard JS number storage. */
  activationPrecision?: ActivationPrecision;
  /** When true, pooled activation arrays are reused across activation calls to reduce GC pressure in tight inference loops. */
  reuseActivationArrays?: boolean;
  /** When true, plain activation output arrays rotate through a small reusable sequence ring, reducing allocations in windowed sequence scenarios. */
  reuseSequenceBuffers?: boolean;
  /** When true, pooled typed activation arrays may be returned directly rather than copied into plain JS arrays, trading aliasing risk for zero-copy throughput. */
  returnTypedActivations?: boolean;
}

/** One emitted chunk from bounded sequence activation through the windowed forward-pass API. */
export interface NetworkForwardWindowChunk {
  /** True when this chunk closes the requested sequence window, signalling that no further chunks will be emitted. */
  done: boolean;
  /** Exclusive end index (row number in the input sequence) covered by this chunk. */
  endIndexExclusive: number;
  /** Output rows emitted for input rows in `[startIndex, endIndexExclusive)`. One row per activation call. */
  outputs: number[][];
  /** Inclusive start index (row number in the input sequence) covered by this chunk. */
  startIndex: number;
  /** Zero-based chunk index within the overall windowed pass. */
  windowIndex: number;
}

/** Optional settings for bounded sequence activation through the `forwardWindowed()` streaming API. */
export interface NetworkForwardWindowOptions {
  /** When true, all outputs are collected into the returned result matrix. When false, callers rely on the `onWindow` callback instead. */
  collectOutputs?: boolean;
  /** Optional synchronous callback invoked after each emitted window chunk, allowing streaming consumers to process partial results immediately. */
  onWindow?: (chunk: NetworkForwardWindowChunk) => void;
  /** When true, each activation call keeps training traces for gradient computation. Set to false for inference-only windowed passes. */
  training?: boolean;
  /** Number of input rows processed per chunk before emitting a window. Controls the granularity of streaming output. */
  windowSize?: number;
}

/** Optional settings for async bounded sequence activation. Extends the synchronous variant with async windowing and a configurable yield cadence. */
export interface NetworkForwardWindowAsyncOptions extends Omit<
  NetworkForwardWindowOptions,
  'onWindow'
> {
  /** Optional async callback invoked after each emitted window chunk. May `await` inside the handler without blocking the scheduler. */
  onWindow?: (chunk: NetworkForwardWindowChunk) => void | Promise<void>;
  /** Number of completed windows between yield calls. Higher values reduce scheduler overhead; lower values improve UI responsiveness. */
  yieldAfterWindows?: number;
  /** Optional custom yield hook used instead of the default `setTimeout(0)` scheduler. Useful in environments where microtask or RAF-based yielding is preferred. */
  yieldControl?: () => Promise<void>;
}

/**
 * One ordered edge request consumed by {@link Network.connectBatch} when callers add multiple connections with deterministic endpoint sequencing.
 */
export interface NetworkConnectionRequest {
  /** Source node that emits the signal. */
  from: Node;
  /** Target node that receives the weighted signal. */
  to: Node;
  /** Optional explicit initial weight. When omitted the network's RNG provides a random default. */
  weight?: number;
}

/** Internal constructor-time surface used by bootstrap helpers to assemble the initial graph state before the public Network facade is returned. */
export interface NetworkBootstrapInternals {
  /** Number of input-role nodes allocated during construction. */
  input: number;
  /** Number of output-role nodes allocated during construction. */
  output: number;
  /** Flat node collection shared by activation, mutation, and serialization. */
  nodes: Node[];
  /** All non-self, non-gate connections in the graph. */
  connections: Connection[];
  /** Gate connections that modulate other connection weights. */
  gates: Connection[];
  /** Self-connections (recurrent loops) on individual nodes. */
  selfconns: Connection[];
  /** Dropout probability applied during training forward passes. */
  dropout: number;
  /** Topology intent (`'feed-forward'` or `'unconstrained'`) set at construction time. */
  _topologyIntent: NetworkTopologyIntent;
  /** Whether acyclic connectivity is enforced at the low level. Deprecated in favor of `_topologyIntent`. */
  _enforceAcyclic: boolean;
  /** Active random number generator. Replaced by `setSeed` for deterministic runs. */
  _rand: () => number;
  /** Mixed-precision configuration for typed-array activation buffers. */
  _precisionConfig: PrecisionConfig;
  /** Requested activation precision label (`'float32'`, `'float64'`, etc.). */
  _activationPrecision: ActivationPrecision;
  /** When true, pooled activation arrays are reused across calls to reduce GC pressure. */
  _reuseActivationArrays: boolean;
  /** When true, plain activation outputs rotate through a small reusable sequence ring. */
  _reuseSequenceBuffers: boolean;
  /** When true, pooled typed activation arrays may be returned directly without copying. */
  _returnTypedActivations: boolean;
  /** Rebuild the cached ordered input/output role id arrays from the current node list. */
  refreshExplicitIORoles: () => void;
  /** Seed the internal deterministic RNG. Called by the constructor when `seed` is provided. */
  setSeed: (seed: number) => void;
  /** Add one directed edge between two nodes and return the new Connection objects. */
  connect: (from: Node, to: Node, weight?: number) => Connection[];
  /** Add multiple directed edges in one deterministic pass. Returns all new Connection objects. */
  connectBatch: (requests: readonly NetworkConnectionRequest[]) => Connection[];
  /** Split a randomly chosen connection by inserting one new hidden node. */
  addNodeBetween: () => void;
}

/** Runtime weight-noise and DropConnect scratch props attached to Connection instances by the regularization layer. */
export interface ConnectionWeightNoiseProps {
  /** Stashed original weight before noise was applied. Used to restore the base weight after each activation step. */
  _origWeightNoise?: number;
  /** Most recently sampled noise perturbation. Kept for diagnostics and schedule-aware noise policies. */
  _wnLast?: number;
  /** Stashed original weight before DropConnect masking. Restored when the mask clears. */
  _origWeight?: number;
  /** Current DropConnect binary mask (0 or 1). Cached so the same mask applies across forward and backward passes in one training step. */
  dcMask?: number;
}

/** Internal network surface projected by activation helpers to access scheduling, slab, and traversal state without depending on the full Network class. */
export interface ActivateNetworkInternals {
  /** When true, acyclic enforcement is active and cycles will raise an error rather than falling back to raw-node-order traversal. */
  _enforceAcyclic?: boolean;
  /** Dirty flag set after structural edits. When true, the activation helper must recompute the topological order before the next activation. */
  _topoDirty: boolean;
  /** Hook that triggers a topological-order recomputation when `_topoDirty` is set. */
  _computeTopoOrder: () => void;
  /** Predicate that returns true when the slab fast path is safe to use under current topology and training mode. */
  _canUseFastSlab: (training: boolean) => boolean;
  /** Typed-array fast path that bypasses node-object traversal when eligible. */
  _fastSlabActivate: (input: number[]) => number[];
  /** When true, pooled activation arrays are reused to reduce allocations. */
  _reuseActivationArrays?: boolean;
  /** When true, plain activation outputs rotate through a reusable sequence ring. */
  _reuseSequenceBuffers?: boolean;
  /** Standard `activate` method used when the slab fast path is not eligible. */
  activate: (
    input: number[] | Float32Array,
    training?: boolean,
    maxActivationDepth?: number,
  ) => number[];
}

/** Internal network surface projected by connection helpers to access enforcement flags and dirty markers. */
export interface ConnectNetworkInternals {
  /** When true, new connections that would create a cycle are rejected. */
  _enforceAcyclic?: boolean;
  /** Network-owned RNG used for deterministic default connection weights. */
  _rand?: () => number;
  /** Dirty flag set after structural edits to invalidate the cached topological order. */
  _topoDirty: boolean;
  /** Dirty flag set after structural edits to invalidate the cached connection slab. */
  _slabDirty: boolean;
}

/** Internal network surface projected by deterministic helpers for RNG snapshot and restore operations. */
export interface DeterministicNetworkInternals {
  /** Raw xorshift RNG state word. `undefined` when no seed has been set. */
  _rngState: number | undefined;
  /** Active random function, either the seeded xorshift or `Math.random`. */
  _rand: (() => number) | undefined;
  /** Current training step counter used by snapshot/restore for exact-resume workflows. */
  _trainingStep: number | undefined;
}

/** Point-in-time snapshot for RNG state restore. Captures both the xorshift state word and the training step so an exact-resume restore can replay from the same position. */
export interface RNGSnapshot {
  /** Training step at the time of the snapshot. Used by exact-resume workflows. */
  step: number | undefined;
  /** Xorshift RNG state word at the time of the snapshot. */
  state: number | undefined;
}

/** Internal network properties accessed by runtime-control helpers for dropout, noise, and iteration state. */
export interface NetworkRuntimeControlInternals {
  /** Connection list */
  connections: Connection[];
  /** Optional layered network view */
  layers?: { nodes: Node[] }[];
  /** Active random generator */
  _rand: () => number;
  /** Current training step */
  _trainingStep: number;
  /** Optional forced-overflow test hook */
  _forceNextOverflow?: boolean;
  /** Scheduled pruning configuration */
  _pruningConfig?: NetworkPruningProps['_pruningConfig'];
  /** Scheduled-pruning baseline connection count */
  _initialConnectionCount?: number;
  /** Global weight-noise standard deviation */
  _weightNoiseStd: number;
  /** Per-hidden-layer weight-noise standard deviations */
  _weightNoisePerHidden: number[];
  /** Optional dynamic weight-noise schedule */
  _weightNoiseSchedule?: (step: number) => number;
  /** Active stochastic-depth survival probabilities */
  _stochasticDepth: number[];
  /** Optional dynamic stochastic-depth schedule */
  _stochasticDepthSchedule?: (step: number, current: number[]) => number[];
  /** Last skipped hidden-layer indices */
  _lastSkippedLayers?: number[];
}

/** Internal network properties accessed by runtime diagnostics helpers for topology and optimizer state. */
export interface NetworkRuntimeDiagnosticsInternals {
  /** Optional layered network view */
  layers?: { nodes: Node[] }[];
  /** Flat network node collection */
  nodes: Node[];
  /** Ordered input-role node ids */
  inputNodeIds: number[];
  /** Ordered output-role node ids */
  outputNodeIds: number[];
  /** Public topology intent preserved on the runtime seam */
  _topologyIntent?: NetworkTopologyIntent;
  /** Cached compiled activation schedule */
  _activationSchedule?: ActivationSchedule | null;
  /** Cached scheduling diagnostics snapshot */
  _activationSchedulingDiagnostics?: ActivationSchedulingDiagnostics | null;
  /** Topology dirty marker */
  _topoDirty?: boolean;
  /** Acyclic mode enforcement flag */
  _enforceAcyclic?: boolean;
  /** Active DropConnect probability */
  _dropConnectProb: number;
  /** Last recorded gradient norm */
  _lastGradNorm?: number;
  /** Optimizer step counter */
  _optimizerStep: number;
  /** Mixed-precision runtime configuration */
  _mixedPrecision: { enabled: boolean; lossScale: number };
  /** Mixed-precision state counters */
  _mixedPrecisionState: {
    goodSteps: number;
    badSteps: number;
    minLossScale: number;
    maxLossScale: number;
    overflowCount?: number;
    underflowCount?: number;
    lastUnderflowStep?: number;
    scaleUpEvents?: number;
    scaleDownEvents?: number;
  };
  /** Last recorded raw gradient norm */
  _lastRawGradNorm: number;
  /** Last gradient-clipping group count */
  _lastGradClipGroupCount: number;
  /** Last overflow training step index */
  _lastOverflowStep: number;
}

/** Internal network properties accessed during gating operations for node-index dirty flag management. */
export interface GatingNetworkProps {
  /** Node-index dirty marker */
  _nodeIndexDirty?: boolean;
}

/** Mutation keep-gates option surface used by sub-node removal logic and gate reassignment. */
export interface SubNodeMutationConfig {
  /** Preserve and reassign gated connections during node removal */
  keep_gates?: boolean;
}

/** Internal network properties used by stats operations to cache the last aggregated metrics payload. */
export interface StatsNetworkProps {
  /** Last aggregated stats payload */
  _lastStats?: Record<string, unknown>;
}

/** Internal topology state carrier for acyclic enforcement, cached schedule, and dirty markers. */
export interface TopologyNetworkProps {
  /** Acyclic mode enforcement flag */
  _enforceAcyclic?: boolean;
  /** Cached deterministic activation schedule for acyclic execution */
  _activationSchedule?: ActivationSchedule | null;
  /** Cached human-friendly scheduling diagnostics snapshot */
  _activationSchedulingDiagnostics?: ActivationSchedulingDiagnostics | null;
  /** Cached topological order */
  _topoOrder: Node[] | null;
  /** Topology dirty marker */
  _topoDirty?: boolean;
}

/** Mutable context assembled when building Kahn-ordered topological node activation waves. */
export interface TopologyBuildContext {
  /** Target network */
  network: Network;
  /** Internal topology state holder */
  internalTopologyProps: TopologyNetworkProps;
  /** In-degree tracking table */
  inDegreeByNode: Map<Node, number>;
  /** Pending processing queue */
  processingQueue: Node[];
  /** Deterministic wave groups captured during Kahn traversal */
  activationSteps: number[][];
  /** Built topological order */
  topoOrder: Node[];
}

/** Mutable context used when running iterative DFS-style path search for reachability checks. */
export interface PathSearchContext {
  /** Target node for reachability check */
  targetNode: Node;
  /** Already visited nodes */
  visitedNodes: Set<Node>;
  /** DFS stack of nodes pending visitation */
  nodesToVisitStack: Node[];
}

/** Internal standalone generation network view exposing node count and precision for code emission. */
export interface NetworkStandaloneProps {
  /** Network node collection */
  nodes: Node[];
  /** Input count */
  input: number;
  /** Output count */
  output: number;
  /** Optional activation precision flag */
  _precisionConfig?: PrecisionConfig;
  /** Optional activation precision flag */
  _activationPrecision?: ActivationPrecision;
}

/** Node with a generated contiguous index used during standalone function source emission. */
export interface NodeWithIndex extends Node {
  /** Assigned contiguous index for code generation */
  index: number;
}

/** Shared mutable state assembled for one standalone network function source generation pass. */
export interface StandaloneGenerationContext {
  /** Standalone network projection */
  standaloneProps: NetworkStandaloneProps;
  /** Resolved activation precision for generated standalone storage */
  resolvedActivationPrecision?: ActivationPrecision;
  /** Input node indexes in public input-vector order, used for emitting the read loop in the generated function. */
  inputNodeIndexes: number[];
  /** Activation-only node indexes in runtime execution order (excludes inputs), used for emitting the forward-pass body. */
  activationNodeIndexes: number[];
  /** Output node indexes in public output-vector order, used for emitting the result-collection loop. */
  outputNodeIndexes: number[];
  /** Map of activation-function name to already-emitted source snippet, preventing duplicate function declarations. */
  emittedActivationSource: Record<string, string>;
  /** Ordered activation-function source snippets collected for the generated function preamble. */
  activationFunctionSources: string[];
  /** Lookup from activation-function name to its assigned array index in the generated function. */
  activationFunctionIndexMap: Record<string, number>;
  /** Counter for assigning unique array indices to activation functions in the generated source. */
  nextActivationFunctionIndex: number;
  /** Initial activation values seeded from the live network state at generation time. */
  initialActivations: number[];
  /** Initial recurrent state values seeded from the live network state at generation time. */
  initialStates: number[];
  /** Accumulated forward-pass body lines, assembled in execution order before being joined into the final source string. */
  bodyLines: string[];
}

/** Internal network dirty-flag surface used by node-removal helpers to invalidate affected caches after graph surgery. */
export interface NetworkRemoveProps {
  /** Set to true after node removal to invalidate the cached topological order. */
  _topoDirty?: boolean;
  /** Set to true after node removal to invalidate the node-index lookup table. */
  _nodeIndexDirty?: boolean;
  /** Set to true after node removal to invalidate the connection slab. */
  _slabDirty?: boolean;
  /** Set to true after node removal to invalidate the adjacency index. */
  _adjDirty?: boolean;
}

/** Validated, immutable context assembled before node removal begins. Passed through the removal pipeline to avoid re-deriving the target node and its index at each step. */
export interface NodeRemovalContext {
  /** Owning network instance. */
  network: Network;
  /** Internal mutable dirty-flag surface for cache invalidation. */
  internalNetwork: NetworkRemoveProps;
  /** The node to be removed. */
  targetNode: Node;
  /** Position of the target node in `network.nodes` at the time the context was created. */
  targetNodeIndex: number;
}

/** Adjacency snapshot captured before a node is removed. Passed to reconnection helpers so inbound and outbound paths can be bridged without re-inspecting the live (partially mutated) graph. */
export interface NodeConnectionSnapshotContext {
  /** Connections pointing into the removed node from other nodes. */
  inboundConnections: Connection[];
  /** Connections pointing out of the removed node to other nodes. */
  outboundConnections: Connection[];
  /** Number of self-connections that were removed along with the node. */
  selfConnectionCount: number;
}

/** One candidate source-target pair for reconnecting paths across a removed node. Used by the bridging helper to reconstruct connectivity without the removed intermediary. */
export interface ReconnectEndpointPairContext {
  /** Source node of the candidate reconnect edge. */
  sourceNode: Node;
  /** Target node of the candidate reconnect edge. */
  targetNode: Node;
}

/** Pruning strategy identifiers specifying available connection removal approaches such as magnitude and SNIP. */
export type PruningMethod = 'magnitude' | 'snip';

/** Growth-budget decision categories controlling structural mutation allow, deny, and prune-then-allow paths. */
export type SparsityBudgetDecision = 'allow' | 'prune-then-allow' | 'deny';

/** Read-only snapshot describing the latest growth-budget decision and connection count metrics. */
export interface NetworkSparsityBudgetSnapshot {
  /** Effective total-connection cap after grace is applied */
  allowedConnectionLimit: number;
  /** Total forward-plus-self connection count when the budget check started. */
  connectionCountBeforeDecision: number;
  /** Total forward-plus-self connection count immediately before growth may run. */
  connectionCountBeforeGrowth: number;
  /** Final decision emitted by the budget helper */
  decision: SparsityBudgetDecision;
  /** Desired total-connection count before the pending growth write */
  desiredConnectionCountBeforeGrowth: number;
  /** Number of forward or self connections the helper planned to prune. */
  plannedPruneCount: number;
  /** Projected total-connection count after the pending growth write */
  projectedConnectionCount: number;
  /** Remaining total-connection headroom after the decision */
  remainingHeadroom: number;
  /** Net total-connection increase requested by the caller */
  requiredAdditionalConnections: number;
  /** Runtime environment whose soft memory target tightened the effective cap. */
  softBudgetEnvironment?: 'browser' | 'node';
  /** Whether a Node/browser soft memory target tightened the effective cap. */
  softBudgetTriggered: boolean;
}

/** Internal network properties accessed during sparsity-budget enforcement and last snapshot storage. */
export interface NetworkSparsityBudgetProps {
  /** Optional active total-connection growth budget configuration */
  _sparsityBudgetConfig?: {
    /** Hard total-connection cap before grace headroom is applied */
    maxConnections: number;
    /** Optional proportional total-connection growth headroom */
    growthGraceFraction: number;
    /** Pruning heuristic used when space must be freed */
    method: PruningMethod;
  };
  /** Last recorded read-only budget decision snapshot */
  _lastSparsityBudgetSnapshot?: NetworkSparsityBudgetSnapshot;
}

/** Internal network properties accessed during pruning operations including scheduled configuration and baseline. */
export interface NetworkPruningProps {
  /** Optional active pruning config */
  _pruningConfig?: {
    /** Start iteration for pruning window */
    start: number;
    /** End iteration for pruning window */
    end: number;
    /** Pruning frequency in iterations */
    frequency: number;
    /** Target sparsity at end of schedule */
    targetSparsity: number;
    /** Ranking method for connection removal */
    method: PruningMethod;
    /** Fraction of removed edges to regrow */
    regrowFraction: number;
    /** Last iteration where pruning was performed */
    lastPruneIter?: number;
  };
  /** Baseline connection count captured for scheduled pruning */
  _initialConnectionCount?: number;
  /** Baseline connection count for evolutionary pruning */
  _evoInitialConnCount?: number;
  /** Active random generator */
  _rand: () => number;
  /** Acyclic mode enforcement flag */
  _enforceAcyclic?: boolean;
  /** Topology dirty marker */
  _topoDirty?: boolean;
}

/** Context for scheduled-pruning target computation providing iteration position and schedule bounds. */
export interface ScheduledTargetContext {
  /** Current iteration */
  iteration: number;
  /** Start of schedule window */
  scheduleStart: number;
  /** End of schedule window */
  scheduleEnd: number;
  /** Final target sparsity */
  targetSparsity: number;
  /** Baseline connection count */
  baselineConnectionCount: number;
}

/** Result of scheduled-pruning target computation for the number of connections to remove. */
export interface ScheduledTargetResult {
  /** Desired number of connections to keep */
  desiredRemainingConnections: number;
  /** Number of connections to prune now */
  excessConnectionCount: number;
}

/** Context for selecting prune candidate connections by magnitude or SNIP scoring. */
export interface PruneSelectionContext {
  /** Candidate connection pool */
  connections: Connection[];
  /** Requested number of removals */
  removalCount: number;
  /** Ranking method used for selection */
  method: PruningMethod;
}

/** Result of prune candidate selection listing the connections scheduled for removal. */
export interface PruneSelectionResult {
  /** Selected connections for removal */
  connectionsToPrune: Connection[];
}

/** Context for deriving regrowth plan from prune count, fraction, and remaining target. */
export interface RegrowthPlanContext {
  /** Number of pruned connections */
  prunedConnectionCount: number;
  /** Fraction requested for regrowth */
  regrowFraction: number;
  /** Desired remaining connection count target */
  desiredRemainingConnections: number;
}

/** Derived regrowth execution plan specifying the max regrowth attempts and connection target. */
export interface RegrowthPlan {
  /** Desired remaining connection count target */
  desiredRemainingConnections: number;
  /** Maximum random regrowth attempts */
  maxAttempts: number;
}

/** Context for regrowth execution routine specifying target, network, and maximum attempts. */
export interface RegrowthExecutionContext {
  /** Target network for regrowth */
  network: Network;
  /** Desired remaining connection count target */
  desiredRemainingConnections: number;
  /** Maximum random regrowth attempts */
  maxAttempts: number;
}

/** Context for evolutionary sparsity target computation during pruning callbacks in evolve. */
export interface EvolutionaryTargetContext {
  /** Target sparsity ratio */
  targetSparsity: number;
  /** Baseline connection count */
  baselineConnectionCount: number;
}

/** Result of evolutionary sparsity target computation for evolution-driven connection pruning. */
export interface EvolutionaryTargetResult {
  /** Desired number of connections to keep */
  desiredRemainingConnections: number;
  /** Number of connections to prune now */
  excessConnectionCount: number;
}

/**
 * Runtime interface for accessing network internals during serialization.
 *
 * This is an internal bridge type used by serializer helpers to read and rebuild
 * topology without exposing private implementation details in public APIs.
 */
export interface SerializeNetworkInternals {
  /** Network node list */
  nodes: Node[];
  /** Directed connection list */
  connections: Connection[];
  /** Self-connection list */
  selfconns: Connection[];
  /** Gated connection list */
  gates: Connection[];
  /** Input count */
  input: number;
  /** Output count */
  output: number;
  /** Connect API used during reconstruction */
  connect: (from: Node, to: Node, weight: number) => Connection[];
  /** Gate API used during reconstruction */
  gate: (gater: Node, connection: Connection) => void;
  /** Optional public topology intent contract */
  _topologyIntent?: NetworkTopologyIntent;
}

/**
 * Serialize internals with optional dropout field.
 *
 * Verbose JSON snapshots normalize this value so readers can treat dropout as numeric data.
 */
export interface NetworkInternalsWithDropout extends SerializeNetworkInternals {
  /** Optional dropout probability */
  dropout?: number;
}

/**
 * Runtime node internals needed for serialization workflows.
 *
 * These fields are the minimal node state required to round-trip compact and JSON payloads.
 */
export interface SerializeNodeInternals {
  /** Node index in network ordering */
  index: number;
  /** Current activation value */
  activation: number;
  /** Current recurrent state value */
  state: number;
  /** Node bias value */
  bias: number;
  /** Node response multiplier applied before squashing */
  response: number;
  /** Optional stable gene identifier */
  geneId?: number;
  /** Node self-connection holder */
  connections: { self: Connection[] };
  /** Activation function with exposed name */
  squash: ((x: number, derivate?: boolean) => number) & { name: string };
}

/**
 * Connection view with optional enabled flag.
 *
 * Some serialized formats preserve per-edge enablement, while others treat missing values
 * as implicitly enabled.
 */
export type ConnectionInternalsWithEnabled = Connection & {
  /** Optional enabled marker used by some formats */
  enabled?: boolean;
};

/**
 * Stable historical identity fields for a connection gene.
 *
 * Runtime node indices are useful for fast reconstruction, but NEAT alignment,
 * checkpoint migration, and future genotype-first work all depend on the
 * historical identifiers that survive reindexing.
 */
export interface ConnectionHistoricalIdentity {
  /** Stable innovation number for this connection gene */
  innovation?: number;
  /** Stable gene id of the source node */
  fromGeneId?: number;
  /** Stable gene id of the target node */
  toGeneId?: number;
  /** Stable gene id of the gater node when one exists. */
  gaterGeneId?: number | null;
}

/**
 * Serialized connection representation used by compact and JSON formats.
 *
 * Endpoints stay index-based for deterministic reconstruction, while the optional
 * historical fields preserve NEAT identity across clone, export, and restore flows.
 */
export interface SerializedConnection extends ConnectionHistoricalIdentity {
  /** Source node index */
  from: number;
  /** Target node index */
  to: number;
  /** Connection weight */
  weight: number;
  /** Optional gater node index */
  gater: number | null;
  /** Optional explicit enabled state for compact historical payloads */
  enabled?: boolean;
}

/**
 * Index-aligned run metadata used by compressed connection payloads.
 *
 * A run starts at `startIndex` and covers `length` contiguous connection rows.
 */
export interface CompressedSerializedIndexRun {
  /** First connection-row index covered by the run */
  startIndex: number;
  /** Number of contiguous rows covered by the run */
  length: number;
}

/**
 * Lossless weight-word payload for compressed compact serialization.
 *
 * The encoding stores each non-zero float64 weight as four signed 16-bit words
 * and then delta-encodes those words across the non-zero connection sequence.
 * Exact positive-zero spans are represented separately as run metadata.
 */
export interface CompressedSerializedConnectionWeights {
  /** Stable encoding identifier for exact float64 reconstruction */
  encoding: 'ieee754-f64-int16-delta-v1';
  /** Raw signed 16-bit words for the first encoded non-zero weight. */
  firstWeightWords: number[];
  /** Flattened signed 16-bit word deltas for the remaining encoded non-zero weights. */
  deltaWords: number[];
  /** Optional exact positive-zero spans aligned to connection order */
  zeroWeightRuns?: CompressedSerializedIndexRun[];
}

/**
 * Array-oriented compressed connection payload for compact serialization.
 *
 * This keeps the compact serializer lossless while removing per-connection key
 * repetition and object allocation overhead from the transport payload.
 */
export interface CompressedSerializedConnectionBlock {
  /** Total serialized connection row count */
  connectionCount: number;
  /** Source node indices aligned by connection order */
  fromIndices: number[];
  /** Target node indices aligned by connection order */
  toIndices: number[];
  /** Exact compressed weight payload */
  weightWords: CompressedSerializedConnectionWeights;
  /** Optional gater node indices using `-1` as the null sentinel. */
  gaterIndices?: number[];
  /** Legacy enabled-state vector retained for backward-compatible decode */
  enabledStates?: boolean[];
  /** Optional disabled connection spans aligned to connection order */
  disabledRuns?: CompressedSerializedIndexRun[];
  /** Optional innovation identifiers aligned by connection order */
  innovationIds?: Array<number | null>;
  /** Optional non-neutral gain values aligned by connection order */
  gainValues?: Array<number | null>;
  /** Optional source node gene ids aligned by connection order. */
  fromGeneIds?: Array<number | null>;
  /** Optional target node gene ids aligned by connection order. */
  toGeneIds?: Array<number | null>;
  /** Optional gater node gene ids aligned by connection order. */
  gaterGeneIds?: Array<number | null>;
}

/** Supported Node-side compression codecs for writing compressed serialized network archive payloads. */
export type CompressedSerializedNetworkArchiveCompression = 'gzip' | 'zstd';

/** Optional codec settings for archiving one compressed network payload in binary form. */
export interface CompressedSerializedNetworkArchiveOptions {
  /** Compression codec used for the archive wrapper */
  compression?: CompressedSerializedNetworkArchiveCompression;
}

/**
 * Compressed compact serialization payload.
 *
 * This format is additive to the legacy compact tuple API: it keeps the same
 * runtime reconstruction semantics while using array-oriented connection data
 * to reduce UTF-8 payload size for storage or transport.
 */
export interface CompressedSerializedNetwork {
  /** Stable format tag for the compressed compact payload */
  format: 'compact-compressed-v1';
  /** Serialization format version inherited from the verbose JSON payload. */
  formatVersion: number;
  /** Compressed connection payload */
  connections: CompressedSerializedConnectionBlock;
  /** Serialized input width */
  input: number;
  /** Serialized output width */
  output: number;
  /** Serialized dropout value */
  dropout: number;
  /** Verbose JSON node records preserved without compression in Action 1. */
  nodes: NetworkJSONNode[];
  /** Runtime activation values aligned to the serialized node order. */
  activations: number[];
  /** Runtime recurrent state values aligned to the serialized node order. */
  states: number[];
  /** Optional topology intent preserved from the runtime network */
  topologyIntent?: NetworkTopologyIntent;
  /** Optional additive extension bag mirrored from the verbose JSON payload. */
  extensions?: NetworkJSONExtensions;
  /** Optional architecture metadata mirrored from the verbose JSON payload. */
  architecture?: NetworkArchitectureDescriptor;
}

/**
 * Node-side archive wrapper around a compressed compact serialization payload.
 *
 * The wrapped `payload` string stores the UTF-8 JSON form of
 * `CompressedSerializedNetwork` after gzip or zstd compression, encoded as
 * base64 for portable storage.
 */
export interface CompressedSerializedNetworkArchive {
  /** Stable format tag for the archive wrapper */
  format: 'compact-compressed-archive-v1';
  /** Compression codec used for the base64 payload */
  compression: CompressedSerializedNetworkArchiveCompression;
  /** Wrapped compressed payload format tag */
  compressedFormat: CompressedSerializedNetwork['format'];
  /** String encoding applied to the binary archive payload */
  payloadEncoding: 'base64';
  /** Base64-encoded compressed JSON payload bytes */
  payload: string;
}

/**
 * Compact tuple payload used by `serialize` output.
 *
 * Tuple slots are intentionally positional to reduce payload size:
 * 0) activations, 1) states, 2) squash keys, 3) connections, 4) input size,
 * 5) output size, 6) optional node gene ids, 7) optional topology intent.
 *
 * @remarks
 * This format is efficient but less self-describing than JSON.
 * Prefer `NetworkJSON` for long-lived persistence and manual inspection.
 * @example
 * ```ts
 * const compactTuple: CompactSerializedNetworkTuple = [
 *   [0.1, 0.2],
 *   [0, 0],
 *   ['identity', 'tanh'],
 *   [{ from: 0, to: 1, weight: 0.5, gater: null }],
 *   1,
 *   1,
 * ];
 * ```
 */
export type CompactSerializedNetworkTuple = [
  number[],
  number[],
  string[],
  SerializedConnection[],
  number,
  number,
  Array<number | null>?,
  NetworkTopologyIntent?,
];

/**
 * Verbose JSON node representation.
 *
 * Node entries are self-describing and intended for readable, versioned snapshots.
 */
export interface NetworkJSONNode {
  /** Node type discriminator */
  type: string;
  /** Node bias value */
  bias: number;
  /** Optional non-neutral response multiplier. Missing means the neutral response value `1`. */
  response?: number;
  /** Squash function name */
  squash: string;
  /** Node index in topology */
  index: number;
  /** Optional stable gene identifier */
  geneId?: number;
}

/**
 * Verbose JSON connection representation.
 *
 * Includes optional gater and explicit enabled state for portability.
 */
export interface NetworkJSONConnection extends ConnectionHistoricalIdentity {
  /** Source node index */
  from: number;
  /** Target node index */
  to: number;
  /** Connection weight */
  weight: number;
  /** Optional non-neutral gain. Missing means the neutral gain value `1`. */
  gain?: number;
  /** Optional gater node index */
  gater: number | null;
  /** Explicit enabled state */
  enabled: boolean;
}

/**
 * Optional extension bag carried by versioned network JSON payloads.
 *
 * The runtime serializer keeps this generic so stricter boundaries such as the
 * NEAT genome adapter can attach additive, versioned metadata without forcing
 * the network layer to understand each feature-specific field.
 */
export interface NetworkJSONExtensions {
  /** Monotonic extension-bag version */
  version: number;
  /** Plain-object extension payload */
  values: Record<string, unknown>;
}

/**
 * Verbose JSON payload representation used by `toJSONImpl` and `fromJSONImpl`.
 *
 * `formatVersion` enables compatibility checks and migration handling.
 * @example
 * ```ts
 * const payload: NetworkJSON = {
 *   formatVersion: 2,
 *   input: 2,
 *   output: 1,
 *   dropout: 0,
 *   nodes: [{ type: 'input', bias: 0, squash: 'identity', index: 0 }],
 *   connections: [],
 * };
 * ```
 */
/**
 * Verbose JSON payload contract used as the canonical long-lived snapshot for network persistence, migration checkpoints, diagnostics export, and worker/runtime handoff workflows where explicit, inspectable node and connection rows are required.
 * The schema preserves explicit node and connection rows so payloads remain inspectable, versioned, and safely replayable in educational and production contexts.
 */
export interface NetworkJSON {
  /** Serialization format version */
  formatVersion: number;
  /** Input count */
  input: number;
  /** Output count */
  output: number;
  /** Dropout value */
  dropout: number;
  /** Optional public topology intent contract */
  topologyIntent?: NetworkTopologyIntent;
  /** Serialized nodes */
  nodes: NetworkJSONNode[];
  /** Serialized connections */
  connections: NetworkJSONConnection[];
  /** Optional additive extension bag preserved for higher-level bridges */
  extensions?: NetworkJSONExtensions;
  /** Optional architecture metadata for diagnostics/UI consumers */
  architecture?: NetworkArchitectureDescriptor;
}

/**
 * Context carrying compact payload fields.
 *
 * This named-object form replaces tuple index access in internal orchestration code.
 */
export interface CompactPayloadContext {
  /** Serialized node activations */
  activations: number[];
  /** Serialized node states */
  states: number[];
  /** Serialized squash names */
  squashes: string[];
  /** Serialized connections */
  connections: SerializedConnection[];
  /** Serialized input size */
  serializedInput: number;
  /** Serialized output size */
  serializedOutput: number;
  /** Optional stable node gene ids aligned to node order. */
  nodeGeneIds?: Array<number | null>;
  /** Optional topology intent contract for compact restore */
  topologyIntent?: NetworkTopologyIntent;
}

/**
 * Resolved input/output sizes for rebuild.
 *
 * Values reflect override-first resolution semantics used during deserialization.
 */
export interface ResolvedNetworkSizeContext {
  /** Input count */
  input: number;
  /** Output count */
  output: number;
}

/**
 * Context for compact-node reconstruction.
 *
 * Arrays are expected to be index-aligned so each node can be hydrated deterministically.
 *
 * This is a serialization/hydration constraint only: the compact format stores node fields
 * (activation, state, squash, and optional gene id) as parallel arrays.
 *
 * Do not read this as guidance for genetic alignment. In NEAT-style crossover and speciation,
 * homologous structure is matched by historical markings (innovation ids), not by array indices.
 */
export interface CompactNodeRebuildContext {
  /** Activation values */
  activations: number[];
  /** State values */
  states: number[];
  /** Squash function names */
  squashes: string[];
  /** Optional stable node gene ids aligned to node order. */
  nodeGeneIds?: Array<number | null>;
  /** Input size */
  input: number;
  /** Output size */
  output: number;
}

/**
 * Context for compact-connection reconstruction.
 *
 * Connection rows are processed independently so malformed entries can be skipped without aborting import.
 */
export interface CompactConnectionRebuildContext {
  /** Internal mutable network view */
  networkInternals: SerializeNetworkInternals;
  /** Serialized connection rows */
  serializedConnections: SerializedConnection[];
}

/**
 * Context for JSON-node reconstruction.
 *
 * Node entries are rebuilt in order and pushed into mutable runtime internals.
 */
export interface JsonNodeRebuildContext {
  /** Internal mutable network view */
  networkInternals: SerializeNetworkInternals;
  /** JSON node entries */
  nodeJsonEntries: NetworkJSONNode[];
}

/**
 * Context for JSON-connection reconstruction.
 *
 * Connection rows may include optional gater and enabled metadata.
 */
export interface JsonConnectionRebuildContext {
  /** Internal mutable network view */
  networkInternals: SerializeNetworkInternals;
  /** JSON connection entries */
  connectionJsonEntries: NetworkJSONConnection[];
}

/**
 * Cost / loss function used during supervised training.
 *
 * A cost function compares an expected `target` vector with the network's produced `output`
 * vector, returning a scalar error where **lower is better**.
 *
 * Design notes:
 * - This is called frequently (often once per training sample), so implementations should be
 *   **pure** and **allocation-light**.
 * - Most built-in training loops assume the returned value is non-negative.
 *
 * Example (mean squared error):
 *
 * ```ts
 * export const mse: CostFunction = (target, output) => {
 *   const sum = target.reduce((acc, targetValue, index) => {
 *     const diff = targetValue - (output[index] ?? 0);
 *     return acc + diff * diff;
 *   }, 0);
 *   return sum / Math.max(1, target.length);
 * };
 * ```
 */
export type CostFunction = (target: number[], output: number[]) => number;

/**
 * Gradient clipping configuration.
 *
 * Clipping prevents rare large gradients from causing unstable weight updates.
 * It is most useful for recurrent networks and noisy datasets.
 *
 * Conceptual modes:
 * - `norm`: clip by a global $L_2$ norm threshold.
 * - `percentile`: clip using a running percentile estimate (robust to outliers).
 * - `layerwise*`: apply the same idea per-layer (useful when layers have very different scales).
 */
export interface GradientClipConfig {
  /** Clipping strategy mode */
  mode?: 'norm' | 'percentile' | 'layerwiseNorm' | 'layerwisePercentile';
  /** Maximum norm for norm-based clipping */
  maxNorm?: number;
  /** Percentile for percentile-based clipping */
  percentile?: number;
  /** Optional bias-handling hint */
  separateBias?: boolean;
}

/**
 * Dynamic mixed-precision configuration.
 *
 * When enabled, training uses a loss-scaling heuristic that attempts to keep gradients
 * in a numerically stable range. Overflow pressure scales the loss down, while
 * persistent tiny gradients can scale it back up.
 */
export interface MixedPrecisionDynamicConfig {
  /** Minimum dynamic loss scale */
  minScale?: number;
  /** Maximum dynamic loss scale */
  maxScale?: number;
  /** Steps before automatic scale increase */
  increaseEvery?: number;
  /** Legacy alias for stable-step threshold */
  stableStepsForIncrease?: number;
}

/**
 * Mixed-precision configuration.
 *
 * Mixed precision can improve throughput by running some math in lower precision while
 * keeping a stable FP32 master copy of parameters when needed.
 */
export interface MixedPrecisionConfig {
  /** Initial loss scale */
  lossScale?: number;
  /** Optional dynamic-scaling options */
  dynamic?: MixedPrecisionDynamicConfig;
}

/**
 * Base optimizer configuration.
 *
 * Training accepts either an optimizer name (`"adam"`, `"sgd"`, ...) or an object.
 * This object form is useful when you want to pin numeric hyperparameters or wrap a base
 * optimizer (e.g. lookahead).
 *
 * Example:
 *
 * ```ts
 * net.train(set, {
 *   iterations: 1_000,
 *   rate: 0.001,
 *   optimizer: { type: 'adamw', beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.01 },
 * });
 * ```
 *
 * Notes:
 * - Exact supported `type` values are validated by training utilities.
 * - Unspecified fields fall back to sensible defaults per optimizer.
 */
export interface OptimizerConfigBase {
  /** Optimizer identifier */
  type: string;
  /** Base optimizer when wrapping (e.g., lookahead). */
  baseType?: string;
  /** Adam/RMS first-moment coefficient */
  beta1?: number;
  /** Adam/RMS second-moment coefficient */
  beta2?: number;
  /** Numeric epsilon for stability */
  eps?: number;
  /** Weight decay factor */
  weightDecay?: number;
  /** Momentum coefficient */
  momentum?: number;
  /** Lookahead sync interval */
  la_k?: number;
  /** Lookahead interpolation factor */
  la_alpha?: number;
}

/**
 * Serialized network payload used in checkpoint callbacks.
 *
 * This is intentionally loose: serialization formats evolve and may include nested
 * structures. Treat this as an opaque snapshot blob.
 */
export type SerializedNetwork = Record<string, unknown>;

/**
 * Checkpoint callback configuration.
 *
 * Training can periodically call `save(...)` with a serialized network snapshot.
 * You can persist these snapshots to disk, upload them, or keep them in-memory.
 */
export interface CheckpointConfig {
  /** Save latest state flag */
  last?: boolean;
  /** Save best state flag */
  best?: boolean;
  /** Callback invoked with checkpoint payload */
  save: (payload: {
    /** Checkpoint kind */
    type: 'last' | 'best';
    /** Iteration number */
    iteration: number;
    /** Training error at checkpoint time */
    error: number;
    /** Serialized network payload */
    network: SerializedNetwork;
  }) => void;
}

/**
 * Schedule callback configuration.
 *
 * A schedule callback is a simple "tick hook" that runs every N iterations.
 * Typical uses include logging, custom learning-rate schedules, or diagnostics.
 */
export interface ScheduleConfig {
  /** Callback frequency in iterations */
  iterations: number;
  /** Callback invoked on schedule ticks */
  function: (info: { error: number; iteration: number }) => void;
}

/**
 * Metrics hook signature.
 *
 * If provided, this callback receives summarized metrics after each iteration.
 * It is designed for lightweight telemetry, not heavy data export.
 */
export type MetricsHook = (m: {
  /** Iteration number */
  iteration: number;
  /** Current monitored error */
  error: number;
  /** Optional plateau-smoothed error */
  plateauError?: number;
  /** Gradient norm after clipping */
  gradNorm: number;
}) => void;

/**
 * Moving-average strategy identifier.
 *
 * These strategies are used to smooth the monitored error curve during training.
 * Smoothing can make early stopping and progress logging less noisy.
 */
export type MovingAverageType =
  'sma' | 'ema' | 'adaptive-ema' | 'median' | 'gaussian' | 'trimmed' | 'wma';

/**
 * Public training options accepted by the high-level training orchestration.
 *
 * Training in this codebase is conceptually:
 * 1) forward activation
 * 2) backward propagation
 * 3) optimizer update
 * repeated until a stopping condition is met.
 *
 * Minimal example:
 *
 * ```ts
 * net.train(set, {
 *   iterations: 500,
 *   rate: 0.3,
 *   batchSize: 16,
 *   gradientClip: { mode: 'norm', maxNorm: 1 },
 * });
 * ```
 *
 * Stopping conditions:
 * - Provide at least one of `iterations` or `error`.
 * - `earlyStopPatience` adds an additional "stop when no improvement" guard.
 */
export interface TrainingOptions {
  /** Max iterations stopping condition */
  iterations?: number;
  /** Target error stopping condition */
  error?: number;
  /** Learning rate */
  rate?: number;
  /** SGD momentum */
  momentum?: number;
  /** Optimizer selection/config */
  optimizer?: string | OptimizerConfigBase;
  /** Dropout probability */
  dropout?: number;
  /** Mini-batch size */
  batchSize?: number;
  /** Gradient accumulation steps */
  accumulationSteps?: number;
  /** Reduction strategy for accumulation */
  accumulationReduction?: 'average' | 'sum';
  /** Gradient clipping configuration */
  gradientClip?: GradientClipConfig;
  /** Mixed precision toggle/config */
  mixedPrecision?: boolean | MixedPrecisionConfig;
  /** Cost function selector */
  cost?: CostFunction | { fn?: CostFunction; calculate?: CostFunction };
  /** Monitoring moving-average window size */
  movingAverageWindow?: number;
  /** Monitoring moving-average strategy */
  movingAverageType?: MovingAverageType;
  /** EMA alpha override */
  emaAlpha?: number;
  /** Adaptive EMA base alpha hint */
  adaptiveEmaBaseAlpha?: number;
  /** Trimmed-mean trim ratio */
  trimmedRatio?: number;
  /** Plateau moving-average window size */
  plateauMovingAverageWindow?: number;
  /** Plateau moving-average strategy */
  plateauMovingAverageType?: MovingAverageType;
  /** Plateau EMA alpha override */
  plateauEmaAlpha?: number;
  /** Early-stop patience iterations */
  earlyStopPatience?: number;
  /** Early-stop minimum improvement delta */
  earlyStopMinDelta?: number;
  /** Checkpoint configuration */
  checkpoint?: CheckpointConfig;
  /** Periodic callback configuration */
  schedule?: ScheduleConfig;
  /** Optional metrics callback */
  metricsHook?: MetricsHook;
}

/** Mutable smoothing state for monitored error updated each supervised training iteration. */
export interface PrimarySmoothingState {
  /** EMA value */
  emaValue?: number;
  /** Base adaptive EMA value */
  adaptiveBaseEmaValue?: number;
  /** Fast adaptive EMA value */
  adaptiveEmaValue?: number;
}

/** Mutable smoothing state for plateau error metric tracked during supervised training. */
export interface PlateauSmoothingState {
  /** Plateau EMA value */
  plateauEmaValue?: number;
}

/** Config for monitored error smoothing computation driving early stopping and progress tracking. */
export interface MonitoredSmoothingConfig {
  /** Moving-average strategy */
  type: MovingAverageType;
  /** Window size */
  window: number;
  /** Optional EMA alpha override */
  emaAlpha?: number;
  /** Optional trim ratio for trimmed mean */
  trimmedRatio?: number;
}

/** Config for plateau error smoothing computation during supervised training progress monitoring. */
export interface PlateauSmoothingConfig {
  /** Moving-average strategy */
  type: MovingAverageType;
  /** Window size */
  window: number;
  /** Optional EMA alpha override */
  emaAlpha?: number;
}

/** Runtime connection view used by training internals for delta-weight accumulation and optimizer updates. */
export interface TrainingConnectionInternals {
  /** Accumulated delta weight */
  totalDeltaWeight: number;
  /** Previous step delta weight */
  previousDeltaWeight: number;
  /** Current weight value */
  weight: number;
  /** Source node runtime reference */
  from: unknown;
  /** Target node runtime reference */
  to: unknown;
  /** Optional gater runtime reference */
  gater: unknown | null;
  /** Optional FP32 master weight in mixed precision */
  _fp32Weight?: number;
}

/** Runtime node view used by training internals for bias delta accumulation and optimizer application. */
export interface TrainingNodeInternals {
  /** Node connection groups */
  connections: {
    /** Incoming connections */
    in: TrainingConnectionInternals[];
    /** Outgoing connections */
    out: TrainingConnectionInternals[];
    /** Self connections */
    self: TrainingConnectionInternals[];
    /** Gated connections */
    gated: TrainingConnectionInternals[];
  };
  /** Optional FP32 master bias in mixed precision */
  _fp32Bias?: number;
  /** Current bias value */
  bias: number;
  /** Accumulated delta bias */
  totalDeltaBias: number;
  /** Previous step delta bias */
  previousDeltaBias: number;
  /** Node type discriminator */
  type: string;
  /** Batch optimizer application hook */
  applyBatchUpdatesWithOptimizer: (config: {
    type: string;
    baseType?: string;
    beta1?: number;
    beta2?: number;
    eps?: number;
    weightDecay?: number;
    momentum?: number;
    lrScale: number;
    t: number;
    la_k?: number;
    la_alpha?: number;
  }) => void;
  /** Backprop propagation hook */
  propagate: (
    rate: number,
    momentum: number,
    update: boolean,
    regularization: RegularizationConfig,
    target?: number,
  ) => void;
}

/** Runtime network view used by training internals for optimizer step tracking and mixed-precision state. */
export interface TrainingNetworkInternals {
  /** Node collection */
  nodes: TrainingNodeInternals[];
  /** Optional grouped layers */
  layers?: { nodes: TrainingNodeInternals[] }[];
  /** Mixed-precision status */
  _mixedPrecision: {
    /** Mixed precision enabled flag */
    enabled: boolean;
    /** Active loss scale */
    lossScale: number;
  };
  /** Forced-overflow test hook */
  _forceNextOverflow?: boolean;
  /** Dynamic mixed-precision counters */
  _mixedPrecisionState: {
    /** Stable step counter */
    goodSteps: number;
    /** Overflow step counter */
    badSteps: number;
    /** Minimum loss scale bound */
    minLossScale: number;
    /** Maximum loss scale bound */
    maxLossScale: number;
    /** Optional overflow event count */
    overflowCount?: number;
    /** Optional underflow event count */
    underflowCount?: number;
    /** Optional last underflow step index */
    lastUnderflowStep?: number;
    /** Optional scale-up event count */
    scaleUpEvents?: number;
    /** Optional scale-down event count */
    scaleDownEvents?: number;
  };
  /** Scale increase cadence */
  _mpIncreaseEvery?: number;
  /** Optimizer step counter */
  _optimizerStep: number;
  /** Last overflow step index */
  _lastOverflowStep?: number;
  /** Micro-batches accumulated for gradients */
  _gradAccumMicroBatches: number;
  /** Last gradient norm */
  _lastGradNorm: number | null;
  /** Last clip-group count */
  _lastGradClipGroupCount?: number;
  /** Optional global epoch counter */
  _globalEpoch?: number;
  /** Best checkpointed error value */
  _checkpointBestError?: number;
  /** Last gradient clip configuration */
  _currentGradClip?: {
    /** Clip mode */
    mode: 'norm' | 'percentile' | 'layerwiseNorm' | 'layerwisePercentile';
    /** Max norm threshold */
    maxNorm?: number;
    /** Percentile threshold */
    percentile?: number;
  };
  /** Gradient accumulation reduction mode */
  _accumulationReduction?: 'average' | 'sum';
  /** Separate-bias clipping flag */
  _gradClipSeparateBias?: boolean;
  /** Activation hook */
  activate: (input: number[] | Float32Array, training?: boolean) => number[];
  /** Optional pruning callback hook */
  _maybePrune?: (epoch: number) => void;
}

/** L1/L2 regularization configuration for applying per-weight decay penalties during backpropagation. */
export interface RegularizationConfig {
  /** L1 regularization factor */
  l1?: number;
  /** L2 regularization factor */
  l2?: number;
}

/** Cost function object compatibility shape bridging legacy and modern cost function interfaces. */
export interface CostFunctionOrObject {
  /** Optional cost function entry point */
  fn?: (target: number[], output: number[]) => number;
  /** Optional legacy cost function entry point */
  calculate?: (target: number[], output: number[]) => number;
}

/** A single supervised training sample used in network evolution fitness scoring. */
export interface TrainingSample {
  /** Input vector */
  input: number[];
  /** Expected output vector */
  output: number[];
}

/** Evolve-side cost function signature comparing target and output vectors for fitness scoring. */
export type EvolveCostFunction = (target: number[], output: number[]) => number;

/** Evolve-side serializable cost-function reference accepting either an inline function or a named string. */
export type CostFunctionOrRef = EvolveCostFunction | { name: string };

/** Internal normalized evolution config built from user-supplied EvolveOptions for orchestration. */
export interface EvolutionConfig {
  /** Error target */
  targetError: number;
  /** Complexity growth penalty factor */
  growth: number;
  /** Cost function selector */
  cost: CostFunctionOrRef;
  /** Evaluation repetitions per genome */
  amount: number;
  /** Logging frequency */
  log: number;
  /** Optional schedule callback config */
  schedule: {
    /** Callback frequency */
    iterations: number;
    /** Callback function */
    function: (stats: {
      /** Fitness value */
      fitness: number;
      /** Error value */
      error: number;
      /** Iteration value */
      iteration: number;
    }) => void;
  };
  /** Whether to clear network traces per evaluation */
  clear: boolean;
  /** Worker thread count */
  threads: number;
}

/** Scalar evolution settings extracted from EvolveOptions and used by orchestration helpers. */
export interface EvolutionSettings {
  /** Error target */
  targetError: number;
  /** Complexity growth penalty factor */
  growth: number;
  /** Cost function selector */
  cost: CostFunctionOrRef;
  /** Evaluation repetitions per genome */
  amount: number;
  /** Logging frequency */
  log: number;
  /** Optional schedule callback config */
  schedule: EvolveOptions['schedule'];
  /** Whether to clear network traces per evaluation */
  clear: boolean;
  /** Worker thread count */
  threads: number;
}

/** Effective evolution stopping conditions extracted from raw evolve options for orchestration use. */
export interface EvolutionStopConditions {
  /** Error target */
  targetError: number;
}

/** Mutable state tracked across iterations during the evolve main loop execution. */
export interface EvolutionLoopState {
  /** Current monitored error */
  currentError: number;
  /** Current best fitness */
  bestFitness: number;
  /** Current best genome */
  bestGenome: Network | undefined;
  /** Number of consecutive invalid-error iterations */
  consecutiveInvalidErrorCount: number;
}

/** Evolve options bag controlling iteration budget, fitness callback, cost function, and stopping conditions. */
export interface EvolveOptions extends Record<string, unknown> {
  /** Target error */
  error?: number;
  /** Maximum iterations */
  iterations?: number;
  /** Complexity growth factor */
  growth?: number;
  /** Cost function selector */
  cost?: CostFunctionOrRef;
  /** Evaluation repetitions per genome */
  amount?: number;
  /** Logging frequency */
  log?: number;
  /** Optional schedule callback config */
  schedule?: {
    /** Callback frequency */
    iterations: number;
    /** Callback function */
    function: (stats: {
      /** Fitness value */
      fitness: number;
      /** Error value */
      error: number;
      /** Iteration value */
      iteration: number;
    }) => void;
  };
  /** Whether to clear traces per evaluation */
  clear?: boolean;
  /** Worker thread count */
  threads?: number;
  /** Population-level fitness callback flag */
  fitnessPopulation?: boolean;
  /** Optional seed network */
  network?: Network;
  /** Population size alias */
  populationSize?: number;
  /** Legacy population size alias */
  popsize?: number;
  /** Enable speciation flag */
  speciation?: boolean;
  /** Optional worker terminator callback */
  _workerTerminators?: () => void;
}

/** Fitness signature evaluating one genome and returning a scalar fitness score. */
export type SingleGenomeFitnessFunction = (genome: Network) => number;

/** Fitness signature evaluating the full population asynchronously and storing results in-place. */
export type PopulationFitnessFunction = (
  population: Network[],
) => Promise<void>;

/** Unified evolution fitness callback shape accepting either a single-genome or population callback. */
export type EvolutionFitnessFunction =
  SingleGenomeFitnessFunction | PopulationFitnessFunction;

/** Result of fitness-strategy setup describing the resolved callback and worker thread count. */
export interface FitnessSetup {
  /** Fitness callback reference */
  fitnessFunction: EvolutionFitnessFunction;
  /** Worker thread count */
  threads: number;
}

/** Shared mutable context coordinating one parallel population worker evaluation run. */
export interface PopulationWorkerEvaluationContext {
  /** Worker instances */
  workers: TestWorkerInstance[];
  /** Population list */
  population: Network[];
  /** Next genome index pointer */
  nextGenomeIndex: number;
  /** Active worker count */
  activeWorkerCount: number;
  /** Complexity growth penalty factor */
  growth: number;
  /** Completion callback */
  resolve: () => void;
}

/** Worker-local traversal context pairing one worker instance with its shared evaluation context. */
export interface WorkerTraversalContext {
  /** Shared evaluation context */
  evaluationContext: PopulationWorkerEvaluationContext;
  /** Current worker instance */
  worker: TestWorkerInstance;
}

/** Minimal runtime contract consumed from the NEAT controller within evolve orchestration utilities. */
export interface NeatRuntime {
  /** Current generation index */
  generation: number;
  /** Mutable options bag */
  options: {
    /** Mutation rate */
    mutationRate?: number;
    /** Mutation amount */
    mutationAmount?: number;
  };
  /** Async evolve function */
  evolve: () => Promise<Network>;
  /** Optional warning hook */
  _warnIfNoBestGenome?: () => void;
}

/** Runtime properties projected from Network for genetic operations such as crossover scoring. */
export interface NetworkGeneticProps {
  /** Directed connection list */
  connections: Connection[];
  /** Node list */
  nodes: Node[];
  /** Self-connection list */
  selfconns: Connection[];
  /** Gated connection list */
  gates: Connection[];
  /** Optional fitness score */
  score?: number;
  /** Optional re-enable probability for disabled genes */
  _reenableProb?: number;
}

/**
 * Runtime materialization descriptor for one inherited connection gene.
 *
 * The runtime gene shelf is intentionally narrower than the old crossover gene
 * shape. The phenotype materializer consumes only stable heredity identity
 * plus weight and enabled state. Runtime node indexes are intentionally
 * excluded because endpoints and gaters are resolved later by `geneId` after
 * the offspring node set is rebuilt.
 */
export interface ConnectionGene extends ConnectionHistoricalIdentity {
  /** Weight value */
  weight: number;
  /** Stable innovation number used for historical alignment */
  innovation: number;
  /** Stable gene id for the source node */
  fromGeneId: number;
  /** Stable gene id for the target node */
  toGeneId: number;
  /** Stable gene id for the gater node when one exists. */
  gaterGeneId: number | null;
  /** Enabled state */
  enabled: boolean;
}

/** Extended connection shape carrying the enabled state used during NEAT genetic crossover. */
export interface ConnectionGeneticProps {
  /** Optional enabled state */
  enabled?: boolean;
}

/** Runtime network shape intersecting Network with genetic properties for crossover helper access. */
export type GeneticNetwork = Network & NetworkGeneticProps;

/** Immutable context for NEAT offspring materialization during gene-aligned crossover reconstruction. */
export interface OffspringMaterializationContext {
  /** Mutable offspring reference */
  offspring: GeneticNetwork;
  /** Public topology intent guiding recurrent/self-gene pruning */
  topologyIntent: NetworkTopologyIntent;
  /** Gene-id lookup for resolving inherited endpoints after node reindexing. */
  offspringNodesByGeneId: Map<number, Node>;
  /** Source nodes keyed by gene id for interface-resolution fallback. */
  sourceNodesByGeneId: Map<number, Node>;
  /** Input/output ordinals keyed by source gene id for interface fallback. */
  sourceNodeInterfaceOrdinalsByGeneId: Map<number, number>;
}

/** Traversal context for one connection gene during crossover offspring gene-aligned materialization. */
export interface GeneTraversalContext {
  /** Shared materialization context */
  materializationContext: OffspringMaterializationContext;
  /** Current connection gene */
  connectionGene: ConnectionGene;
}

/** Resolved endpoint pair for one gene traversal step during crossover offspring materialization. */
export interface GeneEndpointsContext {
  /** Gene traversal context */
  traversalContext: GeneTraversalContext;
  /** Resolved source node */
  fromNode: Node;
  /** Resolved target node */
  toNode: Node;
}

/** Immutable context for selecting inherited genes during NEAT crossover gene alignment. */
export interface ConnectionGeneSelectionContext {
  /** Parent 1 runtime view */
  parent1: GeneticNetwork;
  /** Parent 2 runtime view */
  parent2: GeneticNetwork;
  /** Parent metrics summary */
  parentMetrics: ParentMetrics;
  /** Equal-treatment mode flag */
  equal: boolean;
  /** Random generator */
  randomGenerator: () => number;
  /** Parent 1 gene map by innovation id */
  parent1Genes: Record<string, ConnectionGene>;
  /** Parent 2 gene map by innovation id */
  parent2Genes: Record<string, ConnectionGene>;
}

/** Traversal state for one parent-1 innovation during the NEAT crossover gene walk. */
export interface Parent1GeneTraversalContext {
  /** Selection context */
  selectionContext: ConnectionGeneSelectionContext;
  /** Innovation identifier */
  innovationId: string;
  /** Parent-1 gene entry */
  parent1Gene: ConnectionGene;
  /** Optional parent-2 matching gene */
  parent2Gene: ConnectionGene | undefined;
}

/** Fold result from parent-1 traversal selection during NEAT crossover gene alignment. */
export interface Parent1TraversalSelectionResult {
  /** Chosen genes in traversal order */
  selectedGenes: ConnectionGene[];
  /** Parent-2 innovation ids consumed during overlap handling */
  consumedParent2InnovationIds: string[];
}

/** Immutable baseline context for one NEAT crossover run assembling parents and offspring references. */
export interface CrossoverContext {
  /** Parent network 1 */
  parentNetwork1: Network;
  /** Parent network 2 */
  parentNetwork2: Network;
  /** Equal-treatment mode flag */
  equal: boolean;
  /** Parent 1 runtime view */
  parent1: GeneticNetwork;
  /** Parent 2 runtime view */
  parent2: GeneticNetwork;
  /** Offspring runtime view */
  offspring: GeneticNetwork;
  /** Parent metrics summary */
  parentMetrics: ParentMetrics;
  /** Random generator */
  randomGenerator: () => number;
}

/** Node-build context derived from crossover baseline used during offspring node pool construction. */
export interface CrossoverNodeBuildContext {
  /** Crossover baseline context */
  crossoverContext: CrossoverContext;
  /** Chosen offspring node count */
  offspringNodeCount: number;
}

/** Compact parent metrics summary comparing fitness scores and node counts across both parents. */
export interface ParentMetrics {
  /** Parent-1 score */
  score1: number;
  /** Parent-2 score */
  score2: number;
  /** Parent-1 node count */
  nodeCount1: number;
  /** Parent-2 node count */
  nodeCount2: number;
  /** Shared output size */
  outputSize: number;
}

/** Constructor signature for runtime Network import used during crossover offspring instantiation. */
export interface NetworkConstructor {
  /** Construct a network with input/output dimensions */
  new (input: number, output: number): Network;
}

/** Mutation method descriptor shape used across all mutation strategy dispatch and planning logic. */
export type MutationMethod =
  | string
  | {
      /** Optional method name */
      name?: string;
      /** Optional method type */
      type?: string;
      /** Optional method identity token */
      identity?: string;
      /** Optional max value override */
      max?: number;
      /** Optional min value override */
      min?: number;
      /** Optional mutate-output flag */
      mutateOutput?: boolean;
      /** Additional method-specific fields */
      [key: string]: unknown;
    };

/** Object-only form of the mutation method descriptor excluding string-shorthand aliases. */
export type MutationMethodObject = Exclude<MutationMethod, string>;

/** Internal network properties accessed by mutation helpers for topology enforcement and dirty flags. */
export interface NetworkMutationProps {
  /** Acyclic mode enforcement flag */
  _enforceAcyclic?: boolean;
  /** Topology dirty marker */
  _topoDirty?: boolean;
  /** Optional deterministic-chain cache */
  _detChain?: Node[];
  /** Active random function */
  _rand: () => number;
  /** Node-index dirty marker */
  _nodeIndexDirty?: boolean;
  /** Preferred chain edge cache */
  _preferredChainEdge?: unknown;
}

/** Mutation handler function contract binding a network method to apply one mutation type. */
export interface MutationHandler {
  /** Apply one mutation method to bound network */
  (this: Network, method?: MutationMethod): void;
}

/** Immutable context for forward candidate connection traversal in acyclic mutation helpers. */
export interface ForwardCandidateTraversalContext {
  /** Target network */
  network: Network;
  /** Source index */
  sourceNodeIndex: number;
  /** Source node */
  sourceNode: Node;
  /** Target traversal start index */
  targetStartIndex: number;
}

/** Immutable context for backward candidate traversal in recurrent connection mutation helpers. */
export interface BackwardCandidateTraversalContext {
  /** Target network */
  network: Network;
  /** Later node index */
  laterNodeIndex: number;
  /** Later node reference */
  laterNode: Node;
}

/** Indexed context for directional connection metadata used during acyclic mutation candidate checks. */
export interface DirectionalConnectionContext {
  /** Target network */
  network: Network;
  /** Candidate connection */
  candidateConnection: Connection;
  /** Source node index */
  fromNodeIndex: number;
  /** Target node index */
  toNodeIndex: number;
}

/** Required input and output endpoint pair used when seeding initial feed-forward edge connections. */
export interface InputOutputEndpoints {
  /** Input anchor node */
  inputNode: Node;
  /** Output anchor node */
  outputNode: Node;
}

/** Result of replacing a connection with a newly inserted split hidden node. */
export interface ConnectionSplitResult {
  /** Inserted hidden node */
  hiddenNode: Node;
  /** Previous gater assigned to original edge */
  previousGater: Connection['gater'];
  /** New source-to-hidden edge */
  sourceToHiddenConnection: Connection | undefined;
  /** New hidden-to-target edge */
  hiddenToTargetConnection: Connection | undefined;
}

/** Minimal recurrent-layer shape consumed by mutation expanders when adding recurrent hidden nodes. */
export interface RecurrentLayerShape {
  /** Layer nodes */
  nodes: Node[];
  /** Output node wrapper */
  output: { nodes: Node[] };
}

/** Context for deterministic-chain add-node mutation targeting a terminal connection for splitting. */
export interface DeterministicChainMutationContext {
  /** Deterministic chain snapshot */
  deterministicChain: Node[];
  /** Output node in chain terminal path */
  outputNode: Node;
  /** Terminal connection used for split */
  terminalConnection: Connection;
}

/** Selected distinct node pair returned when sampling two different nodes for swap mutation. */
export interface DistinctNodePair {
  /** First sampled node */
  firstNode: Node;
  /** Second sampled node */
  secondNode: Node;
}

/** Context for target-layer peer traversal when seeding feed-forward connection candidates. */
export interface TargetLayerPeerContext {
  /** Target node type discriminator */
  targetNodeType: Node['type'];
  /** Target node index */
  targetIndex: number;
  /** Max peer distance */
  maxDistance: number;
}

/** Context for source-to-peer connection counting during feed-forward mutation candidate selection. */
export interface SourcePeerConnectionCountContext {
  /** Source node reference */
  sourceNode: Node;
  /** Candidate target peers */
  targetLayerPeers: Node[];
}

/** Context for sampling one random weight value within a configurable minimum and maximum range. */
export interface WeightSamplingRangeContext {
  /** Random function */
  randomValue: () => number;
  /** Minimum sampled value */
  minValue: number;
  /** Maximum sampled value */
  maxValue: number;
}

/** Context for reinitializing a connection group's weights during mutation weight resetting. */
export interface ConnectionGroupReinitContext {
  /** Random function */
  randomValue: () => number;
  /** Minimum sampled value */
  minWeight: number;
  /** Maximum sampled value */
  maxWeight: number;
}

/** Canonical ordered source-target node pair tuple used in connection candidate selection. */
export type NodePair = [Node, Node];
