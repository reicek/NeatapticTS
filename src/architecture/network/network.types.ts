import type { ActivationPrecision, PrecisionConfig } from '../../config';
import type Network from './network';
import type Node from '../node';
import type Connection from '../connection/connection';
import type { TestWorkerInstance } from '../../multithreading/types';
export type {
  ConstructDiagnostics,
  ConstructGraphConnectionSummary,
  ConstructGraphNodeSummary,
  ConstructGraphSnapshot,
  ConstructNodeId,
  ConstructOptions,
  ConstructPart,
  ConstructResult,
  ConstructValidationOptions,
} from './construct/network.construct.utils.types';

export * from './onnx/network.onnx.utils.types';
export * from './slab/network.slab.utils.types';

/** Internal runtime properties attached to Network instances. */
export interface NetworkRuntimeProps {
  /** Last skipped layer indices. */
  _lastSkippedLayers?: number[];
  /** Last aggregated stats payload. */
  _lastStats?: unknown;
  /** Optional runtime layers cache. */
  layers?: unknown[];
  /** Optional architecture descriptor hydrated from serialization metadata. */
  _serializedArchitectureDescriptor?: NetworkArchitectureDescriptor;
  /** Optional generic extension bag hydrated from serialization metadata. */
  _serializedExtensions?: NetworkJSONExtensions;
  /** Optional public topology intent preserved across runtime boundaries. */
  _topologyIntent?: NetworkTopologyIntent;
}

/** Provenance of hidden-layer architecture information. */
export type NetworkArchitectureSource =
  | 'layer-metadata'
  | 'graph-topology'
  | 'inferred';

/**
 * Stable architecture descriptor for UI/telemetry consumers.
 *
 * Hidden-layer sizes are ordered from input-side to output-side.
 */
export interface NetworkArchitectureDescriptor {
  /** Hidden-layer widths in forward order. */
  hiddenLayerSizes: number[];
  /** True when the graph contains at least one directed cycle. */
  hasCycles: boolean;
  /** Source used to resolve hidden-layer sizing. */
  source: NetworkArchitectureSource;
  /** Total runtime node count. */
  totalNodes: number;
  /** Total runtime connection count. */
  totalConnections: number;
}

/** Supported explicit recurrent-module descriptor kinds. */
export type NetworkTemporalRecurrentModuleKind = 'lstm' | 'gru' | 'narx-memory';

/**
 * Public snapshot of one validated recurrent module on a runtime network.
 *
 * The role map keeps architecture-aware tooling honest: a visualizer can label
 * LSTM gates or NARX delay shelves directly instead of reverse-engineering the
 * meaning of each hidden node from raw graph topology alone.
 */
export interface NetworkTemporalRecurrentModuleDescriptor {
  /** Stable module identity preserved across runtime synchronization. */
  moduleId: string;
  /** Public recurrent-module family. */
  kind: NetworkTemporalRecurrentModuleKind;
  /** Ordered node gene ids grouped by semantic role within the module. */
  nodeGeneIdsByRole: Record<string, number[]>;
  /** Connection innovations that still define the live module boundary. */
  connectionInnovations: number[];
  /** Optional user-facing sub-label, currently used by NARX delay shelves. */
  moduleLabel?: string;
}

/**
 * Public snapshot of one validated gated block on a runtime network.
 *
 * This keeps recurrent-aware tooling free to highlight which gates own which
 * structural edges without exposing the raw private extension bag directly.
 */
export interface NetworkTemporalGatedBlockDescriptor {
  /** Stable block identity preserved across runtime synchronization. */
  blockId: string;
  /** Ordered gate-owner gene ids attached to the block. */
  gaterGeneIds: number[];
  /** Connection innovations gated by this block. */
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
  /** Explicit recurrent modules that still match the runtime graph. */
  recurrentModules: NetworkTemporalRecurrentModuleDescriptor[];
  /** Explicit gated blocks that still match the runtime graph. */
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
  /** Ordered stable gene ids for input-role nodes. */
  inputNodeIds: number[];
  /** Ordered stable gene ids for output-role nodes. */
  outputNodeIds: number[];
}

/** Supported activation-schedule modes. */
export type ActivationMode = 'acyclic' | 'recurrent';

/** Execution path used by the most recent activation scheduling decision. */
export type ActivationSchedulingExecutionPath =
  | 'compiled-schedule'
  | 'cycle-fallback-order'
  | 'raw-node-order';

/** High-level issue attached to the latest scheduling decision. */
export type ActivationSchedulingIssue =
  | 'cycle-detected'
  | 'schedule-missing'
  | null;

/** Supported step kinds inside one activation schedule. */
export type ActivationScheduleStepKind = 'wave' | 'recurrent-component';

/** State-handling rule for recurrent schedule execution. */
export type RecurrentStateSemantics = 'carry';

/** One deterministic activation step inside a compiled schedule. */
export interface ActivationScheduleStep {
  /** Whether the step is a plain feed-forward wave or a recurrent SCC boundary. */
  kind: ActivationScheduleStepKind;
  /** Stable node gene ids executed by this step in deterministic order. */
  nodeIds: ReadonlyArray<number>;
  /** Fixed iteration count for recurrent components. */
  iterations?: number;
}

/**
 * Deterministic execution schedule for a network graph.
 *
 * Steps preserve deterministic order while distinguishing ordinary waves from
 * recurrent strongly-connected components.
 */
export interface ActivationSchedule {
  /** Execution mode used to derive this schedule. */
  mode: ActivationMode;
  /** Deterministic activation steps stored as stable node gene ids. */
  steps: ReadonlyArray<ActivationScheduleStep>;
  /** Ordered output-role node gene ids aligned with output vector semantics. */
  outputNodeIds: number[];
  /** Recurrent runtime state policy for recurrent schedules. */
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
  /** Semantic topology contract currently advertised by the network. */
  topologyIntent: NetworkTopologyIntent;
  /** Scheduling mode requested by the current topology contract. */
  requestedMode: ActivationMode;
  /** Whether structural edits marked the scheduling cache as stale. */
  topologyDirty: boolean;
  /** Execution path currently selected for activation traversal. */
  executionPath: ActivationSchedulingExecutionPath;
  /** High-level issue attached to the current scheduling result, when one exists. */
  issue: ActivationSchedulingIssue;
  /** Human-readable summary of the scheduling result. */
  message: string;
  /** Ordered input-role node ids that define public input-vector semantics. */
  inputNodeIds: number[];
  /** Ordered output-role node ids that define public output-vector semantics. */
  outputNodeIds: number[];
  /** Number of execution steps in the compiled schedule when one exists. */
  stepCount: number;
  /** Number of recurrent-component steps in the compiled schedule. */
  recurrentComponentCount: number;
  /** Recurrent state policy for the current schedule, when one exists. */
  stateSemantics: RecurrentStateSemantics | null;
  /** Node ids implicated in a cycle fallback, when one was detected. */
  cycleNodeIds: number[];
  /** Suggested next actions for callers who want a different scheduling result. */
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
  /** Optional minimum hidden-node count to synthesize via node splitting. */
  minHidden?: number;
  /** Optional deterministic RNG seed. */
  seed?: number;
  /** Optional legacy low-level acyclic enforcement flag. */
  enforceAcyclic?: boolean;
  /** Optional public topology intent contract. */
  topologyIntent?: NetworkTopologyIntent;
  /** Optional activation-buffer precision for compiled outputs and reusable activation arrays. Node runtime state and training traces remain normal JS-number storage. */
  activationPrecision?: ActivationPrecision;
  /** Whether pooled activation arrays should be reused. */
  reuseActivationArrays?: boolean;
  /** Whether plain activation outputs may rotate through a small reusable sequence ring. */
  reuseSequenceBuffers?: boolean;
  /** Whether pooled typed activations may be returned directly. */
  returnTypedActivations?: boolean;
}

/** One emitted chunk from bounded sequence activation. */
export interface NetworkForwardWindowChunk {
  /** Whether this chunk closes the requested sequence. */
  done: boolean;
  /** Exclusive end index covered by this chunk. */
  endIndexExclusive: number;
  /** Output rows emitted for this chunk. */
  outputs: number[][];
  /** Inclusive start index covered by this chunk. */
  startIndex: number;
  /** Zero-based emitted chunk index. */
  windowIndex: number;
}

/** Optional settings for bounded sequence activation through `forwardWindowed()`. */
export interface NetworkForwardWindowOptions {
  /** Whether all outputs should be collected into the returned result matrix. */
  collectOutputs?: boolean;
  /** Optional synchronous callback invoked after each emitted window. */
  onWindow?: (chunk: NetworkForwardWindowChunk) => void;
  /** Whether each activation step should keep training traces. */
  training?: boolean;
  /** Number of input rows processed per window before advancing the next slice. */
  windowSize?: number;
}

/** Optional settings for async bounded sequence activation. */
export interface NetworkForwardWindowAsyncOptions extends Omit<
  NetworkForwardWindowOptions,
  'onWindow'
> {
  /** Optional async callback invoked after each emitted window. */
  onWindow?: (chunk: NetworkForwardWindowChunk) => void | Promise<void>;
  /** Completed-window cadence used before yielding control back to the runtime. */
  yieldAfterWindows?: number;
  /** Optional explicit yield hook used instead of the runtime default scheduler. */
  yieldControl?: () => Promise<void>;
}

/** One ordered edge request consumed by {@link Network.connectBatch}. */
export interface NetworkConnectionRequest {
  /** Source node that emits the signal. */
  from: Node;
  /** Target node that receives the signal. */
  to: Node;
  /** Optional explicit starting weight. */
  weight?: number;
}

/** Internal constructor-time surface used by bootstrap helpers. */
export interface NetworkBootstrapInternals {
  /** Input node count. */
  input: number;
  /** Output node count. */
  output: number;
  /** Network node collection. */
  nodes: Node[];
  /** Connection list. */
  connections: Connection[];
  /** Network gates collection. */
  gates: Connection[];
  /** Self-connection list. */
  selfconns: Connection[];
  /** Dropout probability. */
  dropout: number;
  /** Public topology intent used to preserve semantic API choices. */
  _topologyIntent: NetworkTopologyIntent;
  /** Whether to enforce acyclic connectivity. */
  _enforceAcyclic: boolean;
  /** Active random number generator. */
  _rand: () => number;
  /** Typed-array precision used by compiled activation paths and pooled activation buffers. */
  _precisionConfig: PrecisionConfig;
  /** Typed-array precision used by compiled activation paths and pooled activation buffers. */
  _activationPrecision: ActivationPrecision;
  /** Whether pooled activation arrays are reused across activations. */
  _reuseActivationArrays: boolean;
  /** Whether plain activation outputs may rotate through a small reusable sequence ring. */
  _reuseSequenceBuffers: boolean;
  /** Whether pooled typed activations can be returned directly. */
  _returnTypedActivations: boolean;
  /** Refresh explicit ordered input/output role ids from the current node list. */
  refreshExplicitIORoles: () => void;
  /** Seed the internal deterministic RNG. */
  setSeed: (seed: number) => void;
  /** Connect two nodes inside the runtime graph. */
  connect: (from: Node, to: Node, weight?: number) => Connection[];
  /** Connect many node pairs inside the runtime graph. */
  connectBatch: (requests: readonly NetworkConnectionRequest[]) => Connection[];
  /** Insert a hidden node by splitting an existing connection. */
  addNodeBetween: () => void;
}

/** Internal runtime properties attached to Connection instances. */
export interface ConnectionWeightNoiseProps {
  /** Original weight before noise application. */
  _origWeightNoise?: number;
  /** Last sampled noise value. */
  _wnLast?: number;
  /** Original weight before DropConnect mask. */
  _origWeight?: number;
  /** Cached DropConnect mask. */
  dcMask?: number;
}

/** Runtime interface for activation internals. */
export interface ActivateNetworkInternals {
  /** Acyclic mode enforcement flag. */
  _enforceAcyclic?: boolean;
  /** Topology dirty marker. */
  _topoDirty: boolean;
  /** Topology recomputation hook. */
  _computeTopoOrder: () => void;
  /** Fast-slab support predicate. */
  _canUseFastSlab: (training: boolean) => boolean;
  /** Fast-slab activation hook. */
  _fastSlabActivate: (input: number[]) => number[];
  /** Activation-array reuse flag. */
  _reuseActivationArrays?: boolean;
  /** Sequence-output reuse flag for repeated plain activation calls. */
  _reuseSequenceBuffers?: boolean;
  /** Generic activate API. */
  activate: (
    input: number[],
    training?: boolean,
    maxActivationDepth?: number,
  ) => number[];
}

/** Runtime interface for connect internals. */
export interface ConnectNetworkInternals {
  /** Acyclic mode enforcement flag. */
  _enforceAcyclic?: boolean;
  /** Network-owned RNG used for deterministic default connection weights. */
  _rand?: () => number;
  /** Topology dirty marker. */
  _topoDirty: boolean;
  /** Slab dirty marker. */
  _slabDirty: boolean;
}

/** Runtime interface for deterministic internals. */
export interface DeterministicNetworkInternals {
  /** Raw RNG state word. */
  _rngState: number | undefined;
  /** Active random function. */
  _rand: (() => number) | undefined;
  /** Current training step (if tracked). */
  _trainingStep: number | undefined;
}

/** Snapshot payload for RNG state restore flows. */
export interface RNGSnapshot {
  /** Captured training step. */
  step: number | undefined;
  /** Captured RNG state word. */
  state: number | undefined;
}

/** Internal network properties accessed by runtime-control helpers. */
export interface NetworkRuntimeControlInternals {
  /** Connection list. */
  connections: Connection[];
  /** Optional layered network view. */
  layers?: { nodes: Node[] }[];
  /** Active random generator. */
  _rand: () => number;
  /** Current training step. */
  _trainingStep: number;
  /** Optional forced-overflow test hook. */
  _forceNextOverflow?: boolean;
  /** Scheduled pruning configuration. */
  _pruningConfig?: NetworkPruningProps['_pruningConfig'];
  /** Scheduled-pruning baseline connection count. */
  _initialConnectionCount?: number;
  /** Global weight-noise standard deviation. */
  _weightNoiseStd: number;
  /** Per-hidden-layer weight-noise standard deviations. */
  _weightNoisePerHidden: number[];
  /** Optional dynamic weight-noise schedule. */
  _weightNoiseSchedule?: (step: number) => number;
  /** Active stochastic-depth survival probabilities. */
  _stochasticDepth: number[];
  /** Optional dynamic stochastic-depth schedule. */
  _stochasticDepthSchedule?: (step: number, current: number[]) => number[];
  /** Last skipped hidden-layer indices. */
  _lastSkippedLayers?: number[];
}

/** Internal network properties accessed by runtime diagnostics helpers. */
export interface NetworkRuntimeDiagnosticsInternals {
  /** Optional layered network view. */
  layers?: { nodes: Node[] }[];
  /** Flat network node collection. */
  nodes: Node[];
  /** Ordered input-role node ids. */
  inputNodeIds: number[];
  /** Ordered output-role node ids. */
  outputNodeIds: number[];
  /** Public topology intent preserved on the runtime seam. */
  _topologyIntent?: NetworkTopologyIntent;
  /** Cached compiled activation schedule. */
  _activationSchedule?: ActivationSchedule | null;
  /** Cached scheduling diagnostics snapshot. */
  _activationSchedulingDiagnostics?: ActivationSchedulingDiagnostics | null;
  /** Topology dirty marker. */
  _topoDirty?: boolean;
  /** Acyclic mode enforcement flag. */
  _enforceAcyclic?: boolean;
  /** Active DropConnect probability. */
  _dropConnectProb: number;
  /** Last recorded gradient norm. */
  _lastGradNorm?: number;
  /** Optimizer step counter. */
  _optimizerStep: number;
  /** Mixed-precision runtime configuration. */
  _mixedPrecision: { enabled: boolean; lossScale: number };
  /** Mixed-precision state counters. */
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
  /** Last recorded raw gradient norm. */
  _lastRawGradNorm: number;
  /** Last gradient-clipping group count. */
  _lastGradClipGroupCount: number;
  /** Last overflow training step index. */
  _lastOverflowStep: number;
}

/** Internal network properties accessed during gating operations. */
export interface GatingNetworkProps {
  /** Node-index dirty marker. */
  _nodeIndexDirty?: boolean;
}

/** Mutation keep-gates option surface used by sub-node removal logic. */
export interface SubNodeMutationConfig {
  /** Preserve and reassign gated connections during node removal. */
  keep_gates?: boolean;
}

/** Internal network properties used by stats operations. */
export interface StatsNetworkProps {
  /** Last aggregated stats payload. */
  _lastStats?: Record<string, unknown>;
}

/** Internal topology state carrier. */
export interface TopologyNetworkProps {
  /** Acyclic mode enforcement flag. */
  _enforceAcyclic?: boolean;
  /** Cached deterministic activation schedule for acyclic execution. */
  _activationSchedule?: ActivationSchedule | null;
  /** Cached human-friendly scheduling diagnostics snapshot. */
  _activationSchedulingDiagnostics?: ActivationSchedulingDiagnostics | null;
  /** Cached topological order. */
  _topoOrder: Node[] | null;
  /** Topology dirty marker. */
  _topoDirty?: boolean;
}

/** Mutable context used while building topological ordering. */
export interface TopologyBuildContext {
  /** Target network. */
  network: Network;
  /** Internal topology state holder. */
  internalTopologyProps: TopologyNetworkProps;
  /** In-degree tracking table. */
  inDegreeByNode: Map<Node, number>;
  /** Pending processing queue. */
  processingQueue: Node[];
  /** Deterministic wave groups captured during Kahn traversal. */
  activationSteps: number[][];
  /** Built topological order. */
  topoOrder: Node[];
}

/** Mutable context used while running iterative path search. */
export interface PathSearchContext {
  /** Target node for reachability check. */
  targetNode: Node;
  /** Already visited nodes. */
  visitedNodes: Set<Node>;
  /** DFS stack of nodes pending visitation. */
  nodesToVisitStack: Node[];
}

/** Internal standalone generation network view. */
export interface NetworkStandaloneProps {
  /** Network node collection. */
  nodes: Node[];
  /** Input count. */
  input: number;
  /** Output count. */
  output: number;
  /** Optional activation precision flag. */
  _precisionConfig?: PrecisionConfig;
  /** Optional activation precision flag. */
  _activationPrecision?: ActivationPrecision;
}

/** Node with generated index for standalone-code emission. */
export interface NodeWithIndex extends Node {
  /** Assigned contiguous index for code generation. */
  index: number;
}

/** Shared mutable state for standalone source generation. */
export interface StandaloneGenerationContext {
  /** Standalone network projection. */
  standaloneProps: NetworkStandaloneProps;
  /** Resolved activation precision for generated standalone storage. */
  resolvedActivationPrecision?: ActivationPrecision;
  /** Indexed input nodes in public input-vector order. */
  inputNodeIndexes: number[];
  /** Indexed activation traversal in runtime execution order without inputs. */
  activationNodeIndexes: number[];
  /** Indexed output nodes in public output-vector order. */
  outputNodeIndexes: number[];
  /** Already emitted activation source by name. */
  emittedActivationSource: Record<string, string>;
  /** Activation source snippets in order. */
  activationFunctionSources: string[];
  /** Function-name to index lookup. */
  activationFunctionIndexMap: Record<string, number>;
  /** Next activation index counter. */
  nextActivationFunctionIndex: number;
  /** Seed activation buffer values. */
  initialActivations: number[];
  /** Seed state buffer values. */
  initialStates: number[];
  /** Output function body lines. */
  bodyLines: string[];
}

/** Internal network properties accessed during remove operations. */
export interface NetworkRemoveProps {
  /** Topology dirty marker. */
  _topoDirty?: boolean;
  /** Node-index dirty marker. */
  _nodeIndexDirty?: boolean;
  /** Slab dirty marker. */
  _slabDirty?: boolean;
  /** Adjacency dirty marker. */
  _adjDirty?: boolean;
}

/** Immutable context for validated node-removal request. */
export interface NodeRemovalContext {
  /** Owning network instance. */
  network: Network;
  /** Internal mutable network flags. */
  internalNetwork: NetworkRemoveProps;
  /** Node requested for removal. */
  targetNode: Node;
  /** Index of target node in network list. */
  targetNodeIndex: number;
}

/** Snapshot of node adjacency prior to removal. */
export interface NodeConnectionSnapshotContext {
  /** Incoming connections to removed node. */
  inboundConnections: Connection[];
  /** Outgoing connections from removed node. */
  outboundConnections: Connection[];
  /** Number of removed self-connections. */
  selfConnectionCount: number;
}

/** Endpoint pair for reconnecting bridged paths. */
export interface ReconnectEndpointPairContext {
  /** Source node of candidate reconnect edge. */
  sourceNode: Node;
  /** Target node of candidate reconnect edge. */
  targetNode: Node;
}

/** Pruning strategy identifiers. */
export type PruningMethod = 'magnitude' | 'snip';

/** Growth-budget decision categories for structural mutations. */
export type SparsityBudgetDecision = 'allow' | 'prune-then-allow' | 'deny';

/** Read-only snapshot describing the latest growth-budget decision. */
export interface NetworkSparsityBudgetSnapshot {
  /** Effective total-connection cap after grace is applied. */
  allowedConnectionLimit: number;
  /** Total forward-plus-self connection count when the budget check started. */
  connectionCountBeforeDecision: number;
  /** Total forward-plus-self connection count immediately before growth may run. */
  connectionCountBeforeGrowth: number;
  /** Final decision emitted by the budget helper. */
  decision: SparsityBudgetDecision;
  /** Desired total-connection count before the pending growth write. */
  desiredConnectionCountBeforeGrowth: number;
  /** Number of forward or self connections the helper planned to prune. */
  plannedPruneCount: number;
  /** Projected total-connection count after the pending growth write. */
  projectedConnectionCount: number;
  /** Remaining total-connection headroom after the decision. */
  remainingHeadroom: number;
  /** Net total-connection increase requested by the caller. */
  requiredAdditionalConnections: number;
  /** Runtime environment whose soft memory target tightened the effective cap. */
  softBudgetEnvironment?: 'browser' | 'node';
  /** Whether a Node/browser soft memory target tightened the effective cap. */
  softBudgetTriggered: boolean;
}

/** Internal network properties accessed during sparsity-budget enforcement. */
export interface NetworkSparsityBudgetProps {
  /** Optional active total-connection growth budget configuration. */
  _sparsityBudgetConfig?: {
    /** Hard total-connection cap before grace headroom is applied. */
    maxConnections: number;
    /** Optional proportional total-connection growth headroom. */
    growthGraceFraction: number;
    /** Pruning heuristic used when space must be freed. */
    method: PruningMethod;
  };
  /** Last recorded read-only budget decision snapshot. */
  _lastSparsityBudgetSnapshot?: NetworkSparsityBudgetSnapshot;
}

/** Internal network properties accessed during pruning operations. */
export interface NetworkPruningProps {
  /** Optional active pruning config. */
  _pruningConfig?: {
    /** Start iteration for pruning window. */
    start: number;
    /** End iteration for pruning window. */
    end: number;
    /** Pruning frequency in iterations. */
    frequency: number;
    /** Target sparsity at end of schedule. */
    targetSparsity: number;
    /** Ranking method for connection removal. */
    method: PruningMethod;
    /** Fraction of removed edges to regrow. */
    regrowFraction: number;
    /** Last iteration where pruning was performed. */
    lastPruneIter?: number;
  };
  /** Baseline connection count captured for scheduled pruning. */
  _initialConnectionCount?: number;
  /** Baseline connection count for evolutionary pruning. */
  _evoInitialConnCount?: number;
  /** Active random generator. */
  _rand: () => number;
  /** Acyclic mode enforcement flag. */
  _enforceAcyclic?: boolean;
  /** Topology dirty marker. */
  _topoDirty?: boolean;
}

/** Context for scheduled-pruning target computation. */
export interface ScheduledTargetContext {
  /** Current iteration. */
  iteration: number;
  /** Start of schedule window. */
  scheduleStart: number;
  /** End of schedule window. */
  scheduleEnd: number;
  /** Final target sparsity. */
  targetSparsity: number;
  /** Baseline connection count. */
  baselineConnectionCount: number;
}

/** Result of scheduled-pruning target computation. */
export interface ScheduledTargetResult {
  /** Desired number of connections to keep. */
  desiredRemainingConnections: number;
  /** Number of connections to prune now. */
  excessConnectionCount: number;
}

/** Context for selecting prune candidates. */
export interface PruneSelectionContext {
  /** Candidate connection pool. */
  connections: Connection[];
  /** Requested number of removals. */
  removalCount: number;
  /** Ranking method used for selection. */
  method: PruningMethod;
}

/** Result of prune candidate selection. */
export interface PruneSelectionResult {
  /** Selected connections for removal. */
  connectionsToPrune: Connection[];
}

/** Context for deriving regrowth plan. */
export interface RegrowthPlanContext {
  /** Number of pruned connections. */
  prunedConnectionCount: number;
  /** Fraction requested for regrowth. */
  regrowFraction: number;
  /** Desired remaining connection count target. */
  desiredRemainingConnections: number;
}

/** Derived regrowth execution plan. */
export interface RegrowthPlan {
  /** Desired remaining connection count target. */
  desiredRemainingConnections: number;
  /** Maximum random regrowth attempts. */
  maxAttempts: number;
}

/** Context for regrowth execution routine. */
export interface RegrowthExecutionContext {
  /** Target network for regrowth. */
  network: Network;
  /** Desired remaining connection count target. */
  desiredRemainingConnections: number;
  /** Maximum random regrowth attempts. */
  maxAttempts: number;
}

/** Context for evolutionary sparsity target computation. */
export interface EvolutionaryTargetContext {
  /** Target sparsity ratio. */
  targetSparsity: number;
  /** Baseline connection count. */
  baselineConnectionCount: number;
}

/** Result of evolutionary sparsity target computation. */
export interface EvolutionaryTargetResult {
  /** Desired number of connections to keep. */
  desiredRemainingConnections: number;
  /** Number of connections to prune now. */
  excessConnectionCount: number;
}

/**
 * Runtime interface for accessing network internals during serialization.
 *
 * This is an internal bridge type used by serializer helpers to read and rebuild
 * topology without exposing private implementation details in public APIs.
 */
export interface SerializeNetworkInternals {
  /** Network node list. */
  nodes: Node[];
  /** Directed connection list. */
  connections: Connection[];
  /** Self-connection list. */
  selfconns: Connection[];
  /** Gated connection list. */
  gates: Connection[];
  /** Input count. */
  input: number;
  /** Output count. */
  output: number;
  /** Connect API used during reconstruction. */
  connect: (from: Node, to: Node, weight: number) => Connection[];
  /** Gate API used during reconstruction. */
  gate: (gater: Node, connection: Connection) => void;
  /** Optional public topology intent contract. */
  _topologyIntent?: NetworkTopologyIntent;
}

/**
 * Serialize internals with optional dropout field.
 *
 * Verbose JSON snapshots normalize this value so readers can treat dropout as numeric data.
 */
export interface NetworkInternalsWithDropout extends SerializeNetworkInternals {
  /** Optional dropout probability. */
  dropout?: number;
}

/**
 * Runtime node internals needed for serialization workflows.
 *
 * These fields are the minimal node state required to round-trip compact and JSON payloads.
 */
export interface SerializeNodeInternals {
  /** Node index in network ordering. */
  index: number;
  /** Current activation value. */
  activation: number;
  /** Current recurrent state value. */
  state: number;
  /** Node bias value. */
  bias: number;
  /** Node response multiplier applied before squashing. */
  response: number;
  /** Optional stable gene identifier. */
  geneId?: number;
  /** Node self-connection holder. */
  connections: { self: Connection[] };
  /** Activation function with exposed name. */
  squash: ((x: number, derivate?: boolean) => number) & { name: string };
}

/**
 * Connection view with optional enabled flag.
 *
 * Some serialized formats preserve per-edge enablement, while others treat missing values
 * as implicitly enabled.
 */
export type ConnectionInternalsWithEnabled = Connection & {
  /** Optional enabled marker used by some formats. */
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
  /** Stable innovation number for this connection gene. */
  innovation?: number;
  /** Stable gene id of the source node. */
  fromGeneId?: number;
  /** Stable gene id of the target node. */
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
  /** Source node index. */
  from: number;
  /** Target node index. */
  to: number;
  /** Connection weight. */
  weight: number;
  /** Optional gater node index. */
  gater: number | null;
  /** Optional explicit enabled state for compact historical payloads. */
  enabled?: boolean;
}

/**
 * Index-aligned run metadata used by compressed connection payloads.
 *
 * A run starts at `startIndex` and covers `length` contiguous connection rows.
 */
export interface CompressedSerializedIndexRun {
  /** First connection-row index covered by the run. */
  startIndex: number;
  /** Number of contiguous rows covered by the run. */
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
  /** Stable encoding identifier for exact float64 reconstruction. */
  encoding: 'ieee754-f64-int16-delta-v1';
  /** Raw signed 16-bit words for the first encoded non-zero weight. */
  firstWeightWords: number[];
  /** Flattened signed 16-bit word deltas for the remaining encoded non-zero weights. */
  deltaWords: number[];
  /** Optional exact positive-zero spans aligned to connection order. */
  zeroWeightRuns?: CompressedSerializedIndexRun[];
}

/**
 * Array-oriented compressed connection payload for compact serialization.
 *
 * This keeps the compact serializer lossless while removing per-connection key
 * repetition and object allocation overhead from the transport payload.
 */
export interface CompressedSerializedConnectionBlock {
  /** Total serialized connection row count. */
  connectionCount: number;
  /** Source node indices aligned by connection order. */
  fromIndices: number[];
  /** Target node indices aligned by connection order. */
  toIndices: number[];
  /** Exact compressed weight payload. */
  weightWords: CompressedSerializedConnectionWeights;
  /** Optional gater node indices using `-1` as the null sentinel. */
  gaterIndices?: number[];
  /** Legacy enabled-state vector retained for backward-compatible decode. */
  enabledStates?: boolean[];
  /** Optional disabled connection spans aligned to connection order. */
  disabledRuns?: CompressedSerializedIndexRun[];
  /** Optional innovation identifiers aligned by connection order. */
  innovationIds?: Array<number | null>;
  /** Optional non-neutral gain values aligned by connection order. */
  gainValues?: Array<number | null>;
  /** Optional source node gene ids aligned by connection order. */
  fromGeneIds?: Array<number | null>;
  /** Optional target node gene ids aligned by connection order. */
  toGeneIds?: Array<number | null>;
  /** Optional gater node gene ids aligned by connection order. */
  gaterGeneIds?: Array<number | null>;
}

/** Supported Node-side archive compression codecs for compressed payloads. */
export type CompressedSerializedNetworkArchiveCompression = 'gzip' | 'zstd';

/** Optional settings for archiving one compressed network payload. */
export interface CompressedSerializedNetworkArchiveOptions {
  /** Compression codec used for the archive wrapper. */
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
  /** Stable format tag for the compressed compact payload. */
  format: 'compact-compressed-v1';
  /** Serialization format version inherited from the verbose JSON payload. */
  formatVersion: number;
  /** Compressed connection payload. */
  connections: CompressedSerializedConnectionBlock;
  /** Serialized input width. */
  input: number;
  /** Serialized output width. */
  output: number;
  /** Serialized dropout value. */
  dropout: number;
  /** Verbose JSON node records preserved without compression in Action 1. */
  nodes: NetworkJSONNode[];
  /** Runtime activation values aligned to the serialized node order. */
  activations: number[];
  /** Runtime recurrent state values aligned to the serialized node order. */
  states: number[];
  /** Optional topology intent preserved from the runtime network. */
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
  /** Stable format tag for the archive wrapper. */
  format: 'compact-compressed-archive-v1';
  /** Compression codec used for the base64 payload. */
  compression: CompressedSerializedNetworkArchiveCompression;
  /** Wrapped compressed payload format tag. */
  compressedFormat: CompressedSerializedNetwork['format'];
  /** String encoding applied to the binary archive payload. */
  payloadEncoding: 'base64';
  /** Base64-encoded compressed JSON payload bytes. */
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
  /** Node type discriminator. */
  type: string;
  /** Node bias value. */
  bias: number;
  /** Optional non-neutral response multiplier. Missing means the neutral response value `1`. */
  response?: number;
  /** Squash function name. */
  squash: string;
  /** Node index in topology. */
  index: number;
  /** Optional stable gene identifier. */
  geneId?: number;
}

/**
 * Verbose JSON connection representation.
 *
 * Includes optional gater and explicit enabled state for portability.
 */
export interface NetworkJSONConnection extends ConnectionHistoricalIdentity {
  /** Source node index. */
  from: number;
  /** Target node index. */
  to: number;
  /** Connection weight. */
  weight: number;
  /** Optional non-neutral gain. Missing means the neutral gain value `1`. */
  gain?: number;
  /** Optional gater node index. */
  gater: number | null;
  /** Explicit enabled state. */
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
  /** Monotonic extension-bag version. */
  version: number;
  /** Plain-object extension payload. */
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
export interface NetworkJSON {
  /** Serialization format version. */
  formatVersion: number;
  /** Input count. */
  input: number;
  /** Output count. */
  output: number;
  /** Dropout value. */
  dropout: number;
  /** Optional public topology intent contract. */
  topologyIntent?: NetworkTopologyIntent;
  /** Serialized nodes. */
  nodes: NetworkJSONNode[];
  /** Serialized connections. */
  connections: NetworkJSONConnection[];
  /** Optional additive extension bag preserved for higher-level bridges. */
  extensions?: NetworkJSONExtensions;
  /** Optional architecture metadata for diagnostics/UI consumers. */
  architecture?: NetworkArchitectureDescriptor;
}

/**
 * Context carrying compact payload fields.
 *
 * This named-object form replaces tuple index access in internal orchestration code.
 */
export interface CompactPayloadContext {
  /** Serialized node activations. */
  activations: number[];
  /** Serialized node states. */
  states: number[];
  /** Serialized squash names. */
  squashes: string[];
  /** Serialized connections. */
  connections: SerializedConnection[];
  /** Serialized input size. */
  serializedInput: number;
  /** Serialized output size. */
  serializedOutput: number;
  /** Optional stable node gene ids aligned to node order. */
  nodeGeneIds?: Array<number | null>;
  /** Optional topology intent contract for compact restore. */
  topologyIntent?: NetworkTopologyIntent;
}

/**
 * Resolved input/output sizes for rebuild.
 *
 * Values reflect override-first resolution semantics used during deserialization.
 */
export interface ResolvedNetworkSizeContext {
  /** Input count. */
  input: number;
  /** Output count. */
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
  /** Activation values. */
  activations: number[];
  /** State values. */
  states: number[];
  /** Squash function names. */
  squashes: string[];
  /** Optional stable node gene ids aligned to node order. */
  nodeGeneIds?: Array<number | null>;
  /** Input size. */
  input: number;
  /** Output size. */
  output: number;
}

/**
 * Context for compact-connection reconstruction.
 *
 * Connection rows are processed independently so malformed entries can be skipped without aborting import.
 */
export interface CompactConnectionRebuildContext {
  /** Internal mutable network view. */
  networkInternals: SerializeNetworkInternals;
  /** Serialized connection rows. */
  serializedConnections: SerializedConnection[];
}

/**
 * Context for JSON-node reconstruction.
 *
 * Node entries are rebuilt in order and pushed into mutable runtime internals.
 */
export interface JsonNodeRebuildContext {
  /** Internal mutable network view. */
  networkInternals: SerializeNetworkInternals;
  /** JSON node entries. */
  nodeJsonEntries: NetworkJSONNode[];
}

/**
 * Context for JSON-connection reconstruction.
 *
 * Connection rows may include optional gater and enabled metadata.
 */
export interface JsonConnectionRebuildContext {
  /** Internal mutable network view. */
  networkInternals: SerializeNetworkInternals;
  /** JSON connection entries. */
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
  /** Clipping strategy mode. */
  mode?: 'norm' | 'percentile' | 'layerwiseNorm' | 'layerwisePercentile';
  /** Maximum norm for norm-based clipping. */
  maxNorm?: number;
  /** Percentile for percentile-based clipping. */
  percentile?: number;
  /** Optional bias-handling hint. */
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
  /** Minimum dynamic loss scale. */
  minScale?: number;
  /** Maximum dynamic loss scale. */
  maxScale?: number;
  /** Steps before automatic scale increase. */
  increaseEvery?: number;
  /** Legacy alias for stable-step threshold. */
  stableStepsForIncrease?: number;
}

/**
 * Mixed-precision configuration.
 *
 * Mixed precision can improve throughput by running some math in lower precision while
 * keeping a stable FP32 master copy of parameters when needed.
 */
export interface MixedPrecisionConfig {
  /** Initial loss scale. */
  lossScale?: number;
  /** Optional dynamic-scaling options. */
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
  /** Optimizer identifier. */
  type: string;
  /** Base optimizer when wrapping (e.g., lookahead). */
  baseType?: string;
  /** Adam/RMS first-moment coefficient. */
  beta1?: number;
  /** Adam/RMS second-moment coefficient. */
  beta2?: number;
  /** Numeric epsilon for stability. */
  eps?: number;
  /** Weight decay factor. */
  weightDecay?: number;
  /** Momentum coefficient. */
  momentum?: number;
  /** Lookahead sync interval. */
  la_k?: number;
  /** Lookahead interpolation factor. */
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
  /** Save latest state flag. */
  last?: boolean;
  /** Save best state flag. */
  best?: boolean;
  /** Callback invoked with checkpoint payload. */
  save: (payload: {
    /** Checkpoint kind. */
    type: 'last' | 'best';
    /** Iteration number. */
    iteration: number;
    /** Training error at checkpoint time. */
    error: number;
    /** Serialized network payload. */
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
  /** Callback frequency in iterations. */
  iterations: number;
  /** Callback invoked on schedule ticks. */
  function: (info: { error: number; iteration: number }) => void;
}

/**
 * Metrics hook signature.
 *
 * If provided, this callback receives summarized metrics after each iteration.
 * It is designed for lightweight telemetry, not heavy data export.
 */
export type MetricsHook = (m: {
  /** Iteration number. */
  iteration: number;
  /** Current monitored error. */
  error: number;
  /** Optional plateau-smoothed error. */
  plateauError?: number;
  /** Gradient norm after clipping. */
  gradNorm: number;
}) => void;

/**
 * Moving-average strategy identifier.
 *
 * These strategies are used to smooth the monitored error curve during training.
 * Smoothing can make early stopping and progress logging less noisy.
 */
export type MovingAverageType =
  | 'sma'
  | 'ema'
  | 'adaptive-ema'
  | 'median'
  | 'gaussian'
  | 'trimmed'
  | 'wma';

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
  /** Max iterations stopping condition. */
  iterations?: number;
  /** Target error stopping condition. */
  error?: number;
  /** Learning rate. */
  rate?: number;
  /** SGD momentum. */
  momentum?: number;
  /** Optimizer selection/config. */
  optimizer?: string | OptimizerConfigBase;
  /** Dropout probability. */
  dropout?: number;
  /** Mini-batch size. */
  batchSize?: number;
  /** Gradient accumulation steps. */
  accumulationSteps?: number;
  /** Reduction strategy for accumulation. */
  accumulationReduction?: 'average' | 'sum';
  /** Gradient clipping configuration. */
  gradientClip?: GradientClipConfig;
  /** Mixed precision toggle/config. */
  mixedPrecision?: boolean | MixedPrecisionConfig;
  /** Cost function selector. */
  cost?: CostFunction | { fn?: CostFunction; calculate?: CostFunction };
  /** Monitoring moving-average window size. */
  movingAverageWindow?: number;
  /** Monitoring moving-average strategy. */
  movingAverageType?: MovingAverageType;
  /** EMA alpha override. */
  emaAlpha?: number;
  /** Adaptive EMA base alpha hint. */
  adaptiveEmaBaseAlpha?: number;
  /** Trimmed-mean trim ratio. */
  trimmedRatio?: number;
  /** Plateau moving-average window size. */
  plateauMovingAverageWindow?: number;
  /** Plateau moving-average strategy. */
  plateauMovingAverageType?: MovingAverageType;
  /** Plateau EMA alpha override. */
  plateauEmaAlpha?: number;
  /** Early-stop patience iterations. */
  earlyStopPatience?: number;
  /** Early-stop minimum improvement delta. */
  earlyStopMinDelta?: number;
  /** Checkpoint configuration. */
  checkpoint?: CheckpointConfig;
  /** Periodic callback configuration. */
  schedule?: ScheduleConfig;
  /** Optional metrics callback. */
  metricsHook?: MetricsHook;
}

/** Mutable smoothing state for monitored error. */
export interface PrimarySmoothingState {
  /** EMA value. */
  emaValue?: number;
  /** Base adaptive EMA value. */
  adaptiveBaseEmaValue?: number;
  /** Fast adaptive EMA value. */
  adaptiveEmaValue?: number;
}

/** Mutable smoothing state for plateau metric. */
export interface PlateauSmoothingState {
  /** Plateau EMA value. */
  plateauEmaValue?: number;
}

/** Config for monitored smoothing computation. */
export interface MonitoredSmoothingConfig {
  /** Moving-average strategy. */
  type: MovingAverageType;
  /** Window size. */
  window: number;
  /** Optional EMA alpha override. */
  emaAlpha?: number;
  /** Optional trim ratio for trimmed mean. */
  trimmedRatio?: number;
}

/** Config for plateau smoothing computation. */
export interface PlateauSmoothingConfig {
  /** Moving-average strategy. */
  type: MovingAverageType;
  /** Window size. */
  window: number;
  /** Optional EMA alpha override. */
  emaAlpha?: number;
}

/** Runtime connection view used by training internals. */
export interface TrainingConnectionInternals {
  /** Accumulated delta weight. */
  totalDeltaWeight: number;
  /** Previous step delta weight. */
  previousDeltaWeight: number;
  /** Current weight value. */
  weight: number;
  /** Source node runtime reference. */
  from: unknown;
  /** Target node runtime reference. */
  to: unknown;
  /** Optional gater runtime reference. */
  gater: unknown | null;
  /** Optional FP32 master weight in mixed precision. */
  _fp32Weight?: number;
}

/** Runtime node view used by training internals. */
export interface TrainingNodeInternals {
  /** Node connection groups. */
  connections: {
    /** Incoming connections. */
    in: TrainingConnectionInternals[];
    /** Outgoing connections. */
    out: TrainingConnectionInternals[];
    /** Self connections. */
    self: TrainingConnectionInternals[];
    /** Gated connections. */
    gated: TrainingConnectionInternals[];
  };
  /** Optional FP32 master bias in mixed precision. */
  _fp32Bias?: number;
  /** Current bias value. */
  bias: number;
  /** Accumulated delta bias. */
  totalDeltaBias: number;
  /** Previous step delta bias. */
  previousDeltaBias: number;
  /** Node type discriminator. */
  type: string;
  /** Batch optimizer application hook. */
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
  /** Backprop propagation hook. */
  propagate: (
    rate: number,
    momentum: number,
    update: boolean,
    regularization: RegularizationConfig,
    target?: number,
  ) => void;
}

/** Runtime network view used by training internals. */
export interface TrainingNetworkInternals {
  /** Node collection. */
  nodes: TrainingNodeInternals[];
  /** Optional grouped layers. */
  layers?: { nodes: TrainingNodeInternals[] }[];
  /** Mixed-precision status. */
  _mixedPrecision: {
    /** Mixed precision enabled flag. */
    enabled: boolean;
    /** Active loss scale. */
    lossScale: number;
  };
  /** Forced-overflow test hook. */
  _forceNextOverflow?: boolean;
  /** Dynamic mixed-precision counters. */
  _mixedPrecisionState: {
    /** Stable step counter. */
    goodSteps: number;
    /** Overflow step counter. */
    badSteps: number;
    /** Minimum loss scale bound. */
    minLossScale: number;
    /** Maximum loss scale bound. */
    maxLossScale: number;
    /** Optional overflow event count. */
    overflowCount?: number;
    /** Optional underflow event count. */
    underflowCount?: number;
    /** Optional last underflow step index. */
    lastUnderflowStep?: number;
    /** Optional scale-up event count. */
    scaleUpEvents?: number;
    /** Optional scale-down event count. */
    scaleDownEvents?: number;
  };
  /** Scale increase cadence. */
  _mpIncreaseEvery?: number;
  /** Optimizer step counter. */
  _optimizerStep: number;
  /** Last overflow step index. */
  _lastOverflowStep?: number;
  /** Micro-batches accumulated for gradients. */
  _gradAccumMicroBatches: number;
  /** Last gradient norm. */
  _lastGradNorm: number | null;
  /** Last clip-group count. */
  _lastGradClipGroupCount?: number;
  /** Optional global epoch counter. */
  _globalEpoch?: number;
  /** Best checkpointed error value. */
  _checkpointBestError?: number;
  /** Last gradient clip configuration. */
  _currentGradClip?: {
    /** Clip mode. */
    mode: 'norm' | 'percentile' | 'layerwiseNorm' | 'layerwisePercentile';
    /** Max norm threshold. */
    maxNorm?: number;
    /** Percentile threshold. */
    percentile?: number;
  };
  /** Gradient accumulation reduction mode. */
  _accumulationReduction?: 'average' | 'sum';
  /** Separate-bias clipping flag. */
  _gradClipSeparateBias?: boolean;
  /** Activation hook. */
  activate: (input: number[], training?: boolean) => number[];
  /** Optional pruning callback hook. */
  _maybePrune?: (epoch: number) => void;
}

/** L1/L2 regularization configuration. */
export interface RegularizationConfig {
  /** L1 regularization factor. */
  l1?: number;
  /** L2 regularization factor. */
  l2?: number;
}

/** Cost function object compatibility shape. */
export interface CostFunctionOrObject {
  /** Optional cost function entry point. */
  fn?: (target: number[], output: number[]) => number;
  /** Optional legacy cost function entry point. */
  calculate?: (target: number[], output: number[]) => number;
}

/** A single supervised training sample used in evolution scoring. */
export interface TrainingSample {
  /** Input vector. */
  input: number[];
  /** Expected output vector. */
  output: number[];
}

/** Evolve-side cost function signature. */
export type EvolveCostFunction = (target: number[], output: number[]) => number;

/** Evolve-side serializable cost-function reference. */
export type CostFunctionOrRef = EvolveCostFunction | { name: string };

/** Internal normalized evolution config. */
export interface EvolutionConfig {
  /** Error target. */
  targetError: number;
  /** Complexity growth penalty factor. */
  growth: number;
  /** Cost function selector. */
  cost: CostFunctionOrRef;
  /** Evaluation repetitions per genome. */
  amount: number;
  /** Logging frequency. */
  log: number;
  /** Optional schedule callback config. */
  schedule: {
    /** Callback frequency. */
    iterations: number;
    /** Callback function. */
    function: (stats: {
      /** Fitness value. */
      fitness: number;
      /** Error value. */
      error: number;
      /** Iteration value. */
      iteration: number;
    }) => void;
  };
  /** Whether to clear network traces per evaluation. */
  clear: boolean;
  /** Worker thread count. */
  threads: number;
}

/** Scalar evolution settings used by orchestration. */
export interface EvolutionSettings {
  /** Error target. */
  targetError: number;
  /** Complexity growth penalty factor. */
  growth: number;
  /** Cost function selector. */
  cost: CostFunctionOrRef;
  /** Evaluation repetitions per genome. */
  amount: number;
  /** Logging frequency. */
  log: number;
  /** Optional schedule callback config. */
  schedule: EvolveOptions['schedule'];
  /** Whether to clear network traces per evaluation. */
  clear: boolean;
  /** Worker thread count. */
  threads: number;
}

/** Effective evolution stopping conditions. */
export interface EvolutionStopConditions {
  /** Error target. */
  targetError: number;
}

/** Mutable state tracked during evolution loop. */
export interface EvolutionLoopState {
  /** Current monitored error. */
  currentError: number;
  /** Current best fitness. */
  bestFitness: number;
  /** Current best genome. */
  bestGenome: Network | undefined;
  /** Number of consecutive invalid-error iterations. */
  consecutiveInvalidErrorCount: number;
}

/** Evolve options bag. */
export interface EvolveOptions extends Record<string, unknown> {
  /** Target error. */
  error?: number;
  /** Maximum iterations. */
  iterations?: number;
  /** Complexity growth factor. */
  growth?: number;
  /** Cost function selector. */
  cost?: CostFunctionOrRef;
  /** Evaluation repetitions per genome. */
  amount?: number;
  /** Logging frequency. */
  log?: number;
  /** Optional schedule callback config. */
  schedule?: {
    /** Callback frequency. */
    iterations: number;
    /** Callback function. */
    function: (stats: {
      /** Fitness value. */
      fitness: number;
      /** Error value. */
      error: number;
      /** Iteration value. */
      iteration: number;
    }) => void;
  };
  /** Whether to clear traces per evaluation. */
  clear?: boolean;
  /** Worker thread count. */
  threads?: number;
  /** Population-level fitness callback flag. */
  fitnessPopulation?: boolean;
  /** Optional seed network. */
  network?: Network;
  /** Population size alias. */
  populationSize?: number;
  /** Legacy population size alias. */
  popsize?: number;
  /** Enable speciation flag. */
  speciation?: boolean;
  /** Optional worker terminator callback. */
  _workerTerminators?: () => void;
}

/** Fitness signature evaluating one genome. */
export type SingleGenomeFitnessFunction = (genome: Network) => number;

/** Fitness signature evaluating full population asynchronously. */
export type PopulationFitnessFunction = (
  population: Network[],
) => Promise<void>;

/** Unified evolution fitness callback shape. */
export type EvolutionFitnessFunction =
  | SingleGenomeFitnessFunction
  | PopulationFitnessFunction;

/** Result of fitness-strategy setup. */
export interface FitnessSetup {
  /** Fitness callback reference. */
  fitnessFunction: EvolutionFitnessFunction;
  /** Worker thread count. */
  threads: number;
}

/** Shared context for one population worker evaluation run. */
export interface PopulationWorkerEvaluationContext {
  /** Worker instances. */
  workers: TestWorkerInstance[];
  /** Population list. */
  population: Network[];
  /** Next genome index pointer. */
  nextGenomeIndex: number;
  /** Active worker count. */
  activeWorkerCount: number;
  /** Complexity growth penalty factor. */
  growth: number;
  /** Completion callback. */
  resolve: () => void;
}

/** Worker-local traversal context. */
export interface WorkerTraversalContext {
  /** Shared evaluation context. */
  evaluationContext: PopulationWorkerEvaluationContext;
  /** Current worker instance. */
  worker: TestWorkerInstance;
}

/** Minimal runtime contract consumed from NEAT in evolve utilities. */
export interface NeatRuntime {
  /** Current generation index. */
  generation: number;
  /** Mutable options bag. */
  options: {
    /** Mutation rate. */
    mutationRate?: number;
    /** Mutation amount. */
    mutationAmount?: number;
  };
  /** Async evolve function. */
  evolve: () => Promise<Network>;
  /** Optional warning hook. */
  _warnIfNoBestGenome?: () => void;
}

/** Runtime properties used during genetic operations. */
export interface NetworkGeneticProps {
  /** Directed connection list. */
  connections: Connection[];
  /** Node list. */
  nodes: Node[];
  /** Self-connection list. */
  selfconns: Connection[];
  /** Gated connection list. */
  gates: Connection[];
  /** Optional fitness score. */
  score?: number;
  /** Optional re-enable probability for disabled genes. */
  _reenableProb?: number;
}

/**
 * Runtime materialization descriptor for one inherited connection gene.
 *
 * Step 7.2b keeps this runtime shelf narrower than the old crossover gene
 * shape. The phenotype materializer consumes only stable heredity identity
 * plus weight and enabled state. Runtime node indexes are intentionally
 * excluded because endpoints and gaters are resolved later by `geneId` after
 * the offspring node set is rebuilt.
 */
export interface ConnectionGene extends ConnectionHistoricalIdentity {
  /** Weight value. */
  weight: number;
  /** Stable innovation number used for historical alignment. */
  innovation: number;
  /** Stable gene id for the source node. */
  fromGeneId: number;
  /** Stable gene id for the target node. */
  toGeneId: number;
  /** Stable gene id for the gater node when one exists. */
  gaterGeneId: number | null;
  /** Enabled state. */
  enabled: boolean;
}

/** Extended connection shape used during genetic crossover. */
export interface ConnectionGeneticProps {
  /** Optional enabled state. */
  enabled?: boolean;
}

/** Runtime network shape used by crossover internals. */
export type GeneticNetwork = Network & NetworkGeneticProps;

/** Immutable context for offspring materialization. */
export interface OffspringMaterializationContext {
  /** Mutable offspring reference. */
  offspring: GeneticNetwork;
  /** Public topology intent guiding recurrent/self-gene pruning. */
  topologyIntent: NetworkTopologyIntent;
  /** Gene-id lookup for resolving inherited endpoints after node reindexing. */
  offspringNodesByGeneId: Map<number, Node>;
  /** Source nodes keyed by gene id for interface-resolution fallback. */
  sourceNodesByGeneId: Map<number, Node>;
  /** Input/output ordinals keyed by source gene id for interface fallback. */
  sourceNodeInterfaceOrdinalsByGeneId: Map<number, number>;
}

/** Traversal context for one connection gene. */
export interface GeneTraversalContext {
  /** Shared materialization context. */
  materializationContext: OffspringMaterializationContext;
  /** Current connection gene. */
  connectionGene: ConnectionGene;
}

/** Endpoints for one gene traversal step. */
export interface GeneEndpointsContext {
  /** Gene traversal context. */
  traversalContext: GeneTraversalContext;
  /** Resolved source node. */
  fromNode: Node;
  /** Resolved target node. */
  toNode: Node;
}

/** Immutable context for selecting inherited genes. */
export interface ConnectionGeneSelectionContext {
  /** Parent 1 runtime view. */
  parent1: GeneticNetwork;
  /** Parent 2 runtime view. */
  parent2: GeneticNetwork;
  /** Parent metrics summary. */
  parentMetrics: ParentMetrics;
  /** Equal-treatment mode flag. */
  equal: boolean;
  /** Random generator. */
  randomGenerator: () => number;
  /** Parent 1 gene map by innovation id. */
  parent1Genes: Record<string, ConnectionGene>;
  /** Parent 2 gene map by innovation id. */
  parent2Genes: Record<string, ConnectionGene>;
}

/** Traversal state for parent-1 innovation walk. */
export interface Parent1GeneTraversalContext {
  /** Selection context. */
  selectionContext: ConnectionGeneSelectionContext;
  /** Innovation identifier. */
  innovationId: string;
  /** Parent-1 gene entry. */
  parent1Gene: ConnectionGene;
  /** Optional parent-2 matching gene. */
  parent2Gene: ConnectionGene | undefined;
}

/** Fold result for parent-1 traversal selection. */
export interface Parent1TraversalSelectionResult {
  /** Chosen genes in traversal order. */
  selectedGenes: ConnectionGene[];
  /** Parent-2 innovation ids consumed during overlap handling. */
  consumedParent2InnovationIds: string[];
}

/** Immutable baseline context for one crossover run. */
export interface CrossoverContext {
  /** Parent network 1. */
  parentNetwork1: Network;
  /** Parent network 2. */
  parentNetwork2: Network;
  /** Equal-treatment mode flag. */
  equal: boolean;
  /** Parent 1 runtime view. */
  parent1: GeneticNetwork;
  /** Parent 2 runtime view. */
  parent2: GeneticNetwork;
  /** Offspring runtime view. */
  offspring: GeneticNetwork;
  /** Parent metrics summary. */
  parentMetrics: ParentMetrics;
  /** Random generator. */
  randomGenerator: () => number;
}

/** Node-build context derived from crossover baseline. */
export interface CrossoverNodeBuildContext {
  /** Crossover baseline context. */
  crossoverContext: CrossoverContext;
  /** Chosen offspring node count. */
  offspringNodeCount: number;
}

/** Compact parent metrics summary. */
export interface ParentMetrics {
  /** Parent-1 score. */
  score1: number;
  /** Parent-2 score. */
  score2: number;
  /** Parent-1 node count. */
  nodeCount1: number;
  /** Parent-2 node count. */
  nodeCount2: number;
  /** Shared output size. */
  outputSize: number;
}

/** Constructor signature for runtime Network import. */
export interface NetworkConstructor {
  /** Construct a network with input/output dimensions. */
  new (input: number, output: number): Network;
}

/** Mutation method descriptor shape. */
export type MutationMethod =
  | string
  | {
      /** Optional method name. */
      name?: string;
      /** Optional method type. */
      type?: string;
      /** Optional method identity token. */
      identity?: string;
      /** Optional max value override. */
      max?: number;
      /** Optional min value override. */
      min?: number;
      /** Optional mutate-output flag. */
      mutateOutput?: boolean;
      /** Additional method-specific fields. */
      [key: string]: unknown;
    };

/** Object-only form of mutation method descriptor. */
export type MutationMethodObject = Exclude<MutationMethod, string>;

/** Internal network properties accessed during mutations. */
export interface NetworkMutationProps {
  /** Acyclic mode enforcement flag. */
  _enforceAcyclic?: boolean;
  /** Topology dirty marker. */
  _topoDirty?: boolean;
  /** Optional deterministic-chain cache. */
  _detChain?: Node[];
  /** Active random function. */
  _rand: () => number;
  /** Node-index dirty marker. */
  _nodeIndexDirty?: boolean;
  /** Preferred chain edge cache. */
  _preferredChainEdge?: unknown;
}

/** Mutation handler function contract. */
export interface MutationHandler {
  /** Apply one mutation method to bound network. */
  (this: Network, method?: MutationMethod): void;
}

/** Immutable context for forward candidate traversal. */
export interface ForwardCandidateTraversalContext {
  /** Target network. */
  network: Network;
  /** Source index. */
  sourceNodeIndex: number;
  /** Source node. */
  sourceNode: Node;
  /** Target traversal start index. */
  targetStartIndex: number;
}

/** Immutable context for backward candidate traversal. */
export interface BackwardCandidateTraversalContext {
  /** Target network. */
  network: Network;
  /** Later node index. */
  laterNodeIndex: number;
  /** Later node reference. */
  laterNode: Node;
}

/** Indexed context for directional connection metadata. */
export interface DirectionalConnectionContext {
  /** Target network. */
  network: Network;
  /** Candidate connection. */
  candidateConnection: Connection;
  /** Source node index. */
  fromNodeIndex: number;
  /** Target node index. */
  toNodeIndex: number;
}

/** Required endpoint pair for input/output edge seeding. */
export interface InputOutputEndpoints {
  /** Input anchor node. */
  inputNode: Node;
  /** Output anchor node. */
  outputNode: Node;
}

/** Result of replacing a connection with split hidden node. */
export interface ConnectionSplitResult {
  /** Inserted hidden node. */
  hiddenNode: Node;
  /** Previous gater assigned to original edge. */
  previousGater: Connection['gater'];
  /** New source-to-hidden edge. */
  sourceToHiddenConnection: Connection | undefined;
  /** New hidden-to-target edge. */
  hiddenToTargetConnection: Connection | undefined;
}

/** Minimal recurrent-layer shape used by mutation expanders. */
export interface RecurrentLayerShape {
  /** Layer nodes. */
  nodes: Node[];
  /** Output node wrapper. */
  output: { nodes: Node[] };
}

/** Context for deterministic-chain add-node mutation. */
export interface DeterministicChainMutationContext {
  /** Deterministic chain snapshot. */
  deterministicChain: Node[];
  /** Output node in chain terminal path. */
  outputNode: Node;
  /** Terminal connection used for split. */
  terminalConnection: Connection;
}

/** Selected distinct node pair for swap mutation. */
export interface DistinctNodePair {
  /** First sampled node. */
  firstNode: Node;
  /** Second sampled node. */
  secondNode: Node;
}

/** Context for target-layer peer traversal. */
export interface TargetLayerPeerContext {
  /** Target node type discriminator. */
  targetNodeType: Node['type'];
  /** Target node index. */
  targetIndex: number;
  /** Max peer distance. */
  maxDistance: number;
}

/** Context for source-to-peer connection counting. */
export interface SourcePeerConnectionCountContext {
  /** Source node reference. */
  sourceNode: Node;
  /** Candidate target peers. */
  targetLayerPeers: Node[];
}

/** Context for sampling one random weight value. */
export interface WeightSamplingRangeContext {
  /** Random function. */
  randomValue: () => number;
  /** Minimum sampled value. */
  minValue: number;
  /** Maximum sampled value. */
  maxValue: number;
}

/** Context for reinitializing connection group weights. */
export interface ConnectionGroupReinitContext {
  /** Random function. */
  randomValue: () => number;
  /** Minimum sampled value. */
  minWeight: number;
  /** Maximum sampled value. */
  maxWeight: number;
}

/** Canonical source-target node pair tuple. */
export type NodePair = [Node, Node];
