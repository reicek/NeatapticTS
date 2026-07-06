/**
 * Core network chapter for the architecture surface.
 *
 * This folder owns the public `Network` class: the boundary where a graph stops
 * being only nodes and connections and starts behaving like one runnable,
 * mutable, trainable system. Higher-level NEAT code can mutate or score a
 * network, but this chapter is where the graph itself learns how to activate,
 * accept structural edits, preserve deterministic state, serialize, and cross
 * the ONNX boundary.
 *
 * That boundary matters because the same instance has to serve several jobs
 * without changing shape. A caller may want ordinary inference, training-aware
 * forward passes, topology edits, reproducible stochastic behavior, sparse
 * pruning, or a portable checkpoint. Keeping those responsibilities under one
 * facade makes the public API readable while the helper chapters keep each
 * policy cluster narrow enough to teach.
 *
 * Construction can be deterministic: passing a `seed` snapshots the global
 * connection innovation counter before bootstrap and restores it afterwards,
 * so two networks built from the same seed produce identical topology and
 * innovation IDs until external mutation intervenes. This is the foundation
 * used by NGE growth checkpoints to make structural expansion replayable.
 *
 * A useful mental model is to read `network/` as four cooperating shelves.
 * `bootstrap/` explains one-time construction policy. `activate/`, `runtime/`,
 * and `training/` explain how a graph is stepped and regularized once it is
 * alive. `connect/`, `mutate/`, `remove/`, `prune/`, and `topology/` explain
 * graph surgery. `serialize/`, `standalone/`, `onnx/`, and `stats/` explain
 * portability, inspection, and reporting.
 *
 * The performance story is equally important. This chapter deliberately hides
 * storage details until they matter. Callers should be able to ask for
 * `activate()` or `train()` without first understanding slab packing, pooled
 * activation arrays, or cache invalidation. The helper folders then expose how
 * the same graph can switch between object traversal and denser typed-array
 * paths without changing the surface contract.
 *
 * ```mermaid
 * flowchart LR
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   Build[bootstrap and topology intent]:::base --> NetworkClass[Network facade]:::accent
 *   NetworkClass --> Execute[activate runtime and training]:::base
 *   Execute --> Edit[connect mutate prune remove]:::base
 *   Edit --> Persist[serialize standalone and ONNX]:::base
 * ```
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   NetworkChapter[network chapter]:::accent --> Bootstrap[bootstrap/ one-time setup]:::base
 *   NetworkChapter --> Activate[activate/ forward-pass policy]:::base
 *   NetworkChapter --> Runtime[runtime/ training-time controls]:::base
 *   NetworkChapter --> Structure[connect mutate topology prune]:::base
 *   NetworkChapter --> Interop[serialize standalone onnx stats]:::base
 * ```
 *
 * For background on the execution-order side of this chapter, see Wikipedia
 * contributors, [Topological sorting](https://en.wikipedia.org/wiki/Topological_sorting).
 * Feed-forward network execution, acyclic guards, and some of the helper
 * policies in this folder all depend on the same scheduling idea even when the
 * public API keeps that detail out of the caller's way.
 *
 * Example: create a compact layered network and use the ordinary activation
 * surface.
 *
 * ```ts
 * const network = Network.createMLP(2, [4], 1);
 * const outputValues = network.activate([0, 1]);
 * ```
 *
 * Example: checkpoint one network, then restore it for another run.
 *
 * ```ts
 * const network = new Network(2, 1, { seed: 7 });
 * const saved = network.toJSON();
 * const restored = Network.fromJSON(saved);
 * const replayed = restored.activate([1, 0]);
 * ```
 *
 * Practical reading order:
 *
 * 1. Start here for the public `Network` facade and the cross-chapter map.
 * 2. Continue into `bootstrap/` when the constructor contract is the next
 *    question.
 * 3. Continue into `activate/`, `runtime/`, and `training/` for execution and
 *    learning policy.
 * 4. Continue into `connect/`, `mutate/`, `remove/`, `prune/`, and `topology/`
 *    for structural editing.
 * 5. Finish in `serialize/`, `standalone/`, `onnx/`, and `stats/` for
 *    portability, derived reports, and export flows.
 */

import Node from '../node/node';
import Layer from '../layer/layer';
import Connection from '../connection/connection';
import type { ActivationPrecision, PrecisionConfig } from '../../config';
import type { ActivationArray } from '../activationArrayPool/activationArrayPool';
import {
  bootstrapNetwork,
  resolveAcyclicEnforcement,
  resolveTopologyIntent,
  validateTopologyIntentConfiguration,
} from './bootstrap/network.bootstrap.utils';
import { constructNetwork as _constructNetwork } from './construct/network.construct.utils';
import { NetworkConstructorDimensionRequiredError } from './network.errors';
import {
  createMLP as _createMLP,
  rebuildConnections as _rebuildConnections,
} from './topology/network.topology.factory.utils';
import {
  getTopologyIntent as _getTopologyIntent,
  setEnforceAcyclic as _setEnforceAcyclic,
  setTopologyIntent as _setTopologyIntent,
} from './topology/network.topology.contract.utils';
import {
  clearStochasticDepthSchedule as _clearStochasticDepthSchedule,
  clearWeightNoiseSchedule as _clearWeightNoiseSchedule,
  configurePruning as _configurePruning,
  disableStochasticDepth as _disableStochasticDepth,
  disableWeightNoise as _disableWeightNoise,
  enableWeightNoise as _enableWeightNoise,
  getLastSkippedLayers as _getLastSkippedLayers,
  getRuntimeRegularizationStats as _getRuntimeRegularizationStats,
  getTrainingStep as _getTrainingStep,
  setRandom as _setRandom,
  setStochasticDepth as _setStochasticDepth,
  setStochasticDepthSchedule as _setStochasticDepthSchedule,
  setWeightNoiseSchedule as _setWeightNoiseSchedule,
  testForceOverflow as _testForceOverflow,
} from './runtime/network.runtime.controls.utils';
import {
  disableDropConnect as _disableDropConnect,
  enableDropConnect as _enableDropConnect,
  getActivationSchedulingDiagnostics as _getActivationSchedulingDiagnostics,
  getLastGradClipGroupCount as _getLastGradClipGroupCount,
  getLossScale as _getLossScale,
  getRawGradientNorm as _getRawGradientNorm,
  getTrainingStats as _getTrainingStats,
  resetDropoutMasks as _resetDropoutMasks,
} from './runtime/network.runtime.diagnostics.utils';
import { testNetwork as _testNetwork } from './stats/network.stats.utils';
import { exportToONNX } from './onnx/network.onnx';
import { activateGPU } from './gpu/network.gpu.activate';
import { isGPUEligible } from './gpu/network.gpu.fallback';
import {
  activate as _activate,
  generateStandalone,
  computeTopoOrder as _computeTopoOrder,
  hasPath as _hasPath,
  rebuildConnectionSlab as _rebuildConnectionSlab,
  fastSlabActivate as _fastSlabActivate,
  canUseFastSlab as _canUseFastSlab,
  getConnectionSlab as _getConnectionSlab,
  maybePrune as _maybePrune,
  pruneToSparsity as _pruneToSparsity,
  getCurrentSparsity as _getCurrentSparsity,
  getSparsityBudgetSnapshot as _getSparsityBudgetSnapshot,
  gate as _gate,
  ungate as _ungate,
  setSeed as _setSeed,
  snapshotRNG as _snapshotRNG,
  restoreRNG as _restoreRNG,
  getRNGState as _getRNGState,
  setRNGState as _setRNGState,
  getRandomFn as _getRandomFn,
  removeNode as _removeNodeStandalone,
  connect as _connect,
  connectBatch as _connectBatch,
  disconnect as _disconnect,
  serializeCloneImpl as _cloneImpl,
  serialize as _serialize,
  deserialize as _deserialize,
  toJSONImpl as _toJSONImpl,
  fromJSONImpl as _fromJSONImpl,
  noTraceActivate as _noTraceActivate,
  activateRaw as _activateRaw,
  activateBatch as _activateBatch,
  addNodeBetweenImpl as _addNodeBetweenImpl,
  configureSparsityBudget as _configureSparsityBudget,
  mutateImpl as _mutateImpl,
  resolveArchitectureDescriptor as _resolveArchitectureDescriptor,
  applyGradientClippingImpl as _applyGradientClippingImpl,
  propagate as _propagate,
  clearState as _clearState,
  trainImpl as _trainImpl,
  crossOver as _crossOver,
  describeTemporalStructure as _describeTemporalStructure,
  forwardWindowed as _forwardWindowed,
  forwardWindowedAsync as _forwardWindowedAsync,
} from './network.utils';
import type {
  ActivationSchedule,
  ActivationSchedulingDiagnostics,
  CompactSerializedNetworkTuple,
  ConstructOptions,
  ConstructPart,
  ConstructResult,
  ExplicitIORoles,
  NetworkArchitectureDescriptor,
  MutationMethod,
  NetworkBootstrapInternals,
  NetworkConnectionRequest,
  NetworkConstructorOptions,
  NetworkForwardWindowAsyncOptions,
  NetworkForwardWindowOptions,
  NetworkSparsityBudgetSnapshot,
  NetworkTemporalStructureDescriptor,
  NetworkTopologyIntent,
  RNGSnapshot,
  TrainingOptions,
} from './network.types';
import type { NetworkView } from '../../utils/memory';

/**
 * Public graph runtime that combines execution, editing, and portability.
 *
 * `Network` is the instance callers use when one directed graph should be
 * activated, mutated, regularized, checkpointed, or exported without changing
 * the public shape of the object.
 *
 * @example
 * ```ts
 * const network = Network.createMLP(2, [4], 1);
 * const output = network.activate([0, 1]);
 * ```
 */
export default class Network implements NetworkView {
  /** @internal DropConnect probability. */
  protected _dropConnectProb: number = 0;
  /** @internal Last recorded gradient norm. */
  private _lastGradNorm?: number;
  /** @internal Optimizer step counter. */
  private _optimizerStep: number = 0;
  /** @internal Global weight-noise standard deviation. */
  private _weightNoiseStd: number = 0;
  /** @internal Per-hidden-layer weight-noise standard deviations. */
  private _weightNoisePerHidden: number[] = [];
  /** @internal Dynamic weight-noise schedule function. */
  private _weightNoiseSchedule?: (step: number) => number;
  /** @internal Stochastic depth schedule values. */
  private _stochasticDepth: number[] = [];
  /** @internal Original weights captured for weight-noise recovery. */
  private _wnOrig?: number[];
  /** @internal Training step counter. */
  private _trainingStep: number = 0;
  /** @internal Random number generator used for stochastic operations. */
  private _rand: () => number = Math.random;
  /** @internal Raw RNG state word. */
  private _rngState?: number;
  /** @internal Last recorded stats payload. */
  private _lastStats: unknown = null;
  /** @internal Dynamic stochastic depth schedule. */
  private _stochasticDepthSchedule?: (
    step: number,
    current: number[],
  ) => number[];
  /** @internal Mixed precision runtime configuration. */
  private _mixedPrecision: { enabled: boolean; lossScale: number } = {
    enabled: false,
    lossScale: 1,
  };
  /** @internal Mixed precision state counters. */
  private _mixedPrecisionState: {
    goodSteps: number;
    badSteps: number;
    minLossScale: number;
    maxLossScale: number;
    overflowCount?: number;
    underflowCount?: number;
    lastUnderflowStep?: number;
    scaleUpEvents?: number;
    scaleDownEvents?: number;
  } = {
    goodSteps: 0,
    badSteps: 0,
    minLossScale: 1,
    maxLossScale: 65536,
    overflowCount: 0,
    underflowCount: 0,
    lastUnderflowStep: -1,
    scaleUpEvents: 0,
    scaleDownEvents: 0,
  };
  /** @internal Accumulated micro-batch counter. */
  private _gradAccumMicroBatches: number = 0;
  /** @internal Gradient clip configuration for the current step. */
  private _currentGradClip?: {
    mode: 'norm' | 'percentile' | 'layerwiseNorm' | 'layerwisePercentile';
    maxNorm?: number;
    percentile?: number;
  };
  /** @internal Last recorded raw (pre-update) gradient norm. */
  private _lastRawGradNorm: number = 0;
  /** @internal Accumulation reduction mode. */
  private _accumulationReduction: 'average' | 'sum' = 'average';
  /** @internal Whether to apply separate bias clipping. */
  private _gradClipSeparateBias: boolean = false;
  /** @internal Last gradient clipping group count. */
  private _lastGradClipGroupCount: number = 0;
  /** @internal Last overflow training step index. */
  private _lastOverflowStep: number = -1;
  /** @internal Flag to force a mixed-precision overflow path. */
  private _forceNextOverflow: boolean = false;
  /** @internal Pruning configuration for scheduled pruning. */
  private _pruningConfig?: {
    start: number;
    end: number;
    targetSparsity: number;
    regrowFraction: number;
    frequency: number;
    method: 'magnitude' | 'snip';
    lastPruneIter?: number;
  };
  /** @internal Initial connection count used for pruning baselines. */
  private _initialConnectionCount?: number;
  /** @internal Whether to enforce acyclic connectivity. */
  private _enforceAcyclic: boolean = false;
  /** @internal Public topology intent used to preserve semantic API choices. */
  private _topologyIntent: NetworkTopologyIntent = 'unconstrained';
  /** @internal Ordered stable gene ids for input-role nodes. */
  private _inputNodeIds: number[] = [];
  /** @internal Ordered stable gene ids for output-role nodes. */
  private _outputNodeIds: number[] = [];
  /** @internal Cached deterministic activation schedule for acyclic graphs. */
  private _activationSchedule: ActivationSchedule | null = null;
  /** @internal Human-friendly scheduling diagnostics snapshot. */
  private _activationSchedulingDiagnostics: ActivationSchedulingDiagnostics | null =
    null;
  /** @internal Cached topological order. */
  private _topoOrder: Node[] | null = null;
  /** @internal Topology dirty marker. */
  private _topoDirty: boolean = true;
  /** @internal Global epoch counter. */
  private _globalEpoch: number = 0;
  /** @internal Baseline connection count used by evolution-time pruning. */
  private _evoInitialConnCount?: number;
  /** @internal Shared precision config resolved during bootstrap. */
  private _precisionConfig: PrecisionConfig = {
    activationPrecision: 'f64',
  };
  /** @internal Typed-array precision used by compiled activation paths. */
  private _activationPrecision: ActivationPrecision = 'f64';
  /** @internal Whether pooled activation arrays are reused across activations. */
  private _reuseActivationArrays: boolean = false;
  /** @internal Whether pooled typed activations can be returned directly. */
  private _returnTypedActivations: boolean = false;
  /** @internal Cached pooled activation output array. */
  private _activationPool?: Float32Array | Float64Array;
  /** @internal Output-start array for slab forward pass. */
  private _outStart?: Uint32Array;
  /** @internal Output-order array for slab forward pass. */
  private _outOrder?: Uint32Array;
  /** @internal Adjacency dirty marker for slab structures. */
  private _adjDirty: boolean = true;

  /** @internal Backing field for the optional WebGPU device. */
  private _gpuDevice?: GPUDevice;

  /**
   * Optional WebGPU device used by the GPU inference fast path.
   *
   * Assign a device here, then call `activate(input, { useGPU: true })` to opt
   * into the WebGPU forward pass. If the device is missing, the network is
   * ineligible, or `useGPU` is omitted, the standard CPU path is used
   * transparently. This opt-in design keeps classic NEAT behavior unchanged
   * unless a caller explicitly requests the GPU path.
   *
   * A one-shot `device.lost` listener is attached the first time a device is
   * assigned. If the device is later lost, this property is cleared so
   * subsequent activations fall back to the CPU path until a new device is
   * assigned.
   *
   * GPU output agrees with the CPU path within an absolute tolerance of `5e-1`
   * and a mean absolute error of `≤ 1e-1`. For deterministic replay or
   * cross-machine regression tests, use the CPU path as the canonical reference.
   *
   * @example
   * ```ts
   * const network = new Architect.Perceptron(2, 4, 1);
   * const adapter = await navigator.gpu.requestAdapter({
   *   powerPreference: 'high-performance',
   * });
   * network.gpuDevice = (await adapter?.requestDevice()) ?? undefined;
   * const output = await network.activate([0.5, -0.2], { useGPU: true });
   * ```
   */
  get gpuDevice(): GPUDevice | undefined {
    return this._gpuDevice;
  }

  set gpuDevice(device: GPUDevice | undefined) {
    if (this._gpuDevice === device) {
      return;
    }
    this._gpuDevice = device;
    if (device) {
      void device.lost.then(() => {
        if (this._gpuDevice === device) {
          this._gpuDevice = undefined;
        }
      });
    }
  }

  /** @internal Preferred linear-chain edge for node-split mutations. */
  private _preferredChainEdge?: Connection;

  /** Index signature for adaptive features compatibility. */
  [key: string]: unknown;
  /** Input node count. */
  input!: number;
  /** Output node count. */
  output!: number;
  /**
   * Ordered stable gene ids that define the network input vector contract.
   *
   * Returns a cloned array so callers can inspect role metadata without
   * mutating runtime state.
   *
   * @returns Ordered input node gene ids.
   */
  get inputNodeIds(): number[] {
    return [...this._inputNodeIds];
  }

  /**
   * Ordered stable gene ids that define the network output vector contract.
   *
   * Returns a cloned array so callers can inspect role metadata without
   * mutating runtime state.
   *
   * @returns Ordered output node gene ids.
   */
  get outputNodeIds(): number[] {
    return [...this._outputNodeIds];
  }
  /** Optional fitness score. */
  score?: number;
  /** Network node collection. */
  nodes!: Node[];
  /** Connection list. */
  connections!: Connection[];
  /** Network gates collection. */
  gates!: Connection[];
  /** Self-connection list. */
  selfconns!: Connection[];
  /** Dropout probability. */
  dropout: number = 0;
  /** Optional layered view cache. */
  layers?: Layer[];
  /** @internal Packed connection slab weights. */
  public _connWeights?: Float32Array | Float64Array;
  /** @internal Packed connection slab source indices. */
  public _connFrom?: Uint32Array;
  /** @internal Packed connection slab target indices. */
  public _connTo?: Uint32Array;
  /** @internal Slab dirty marker. */
  public _slabDirty: boolean = true;
  /** @internal Whether to store slab weights in float32. */
  public _useFloat32Weights: boolean = true;
  /** @internal Node index dirty marker. */
  public _nodeIndexDirty: boolean = true;
  /** @internal Cached fast activation array A. */
  public _fastA?: Float32Array | Float64Array;
  /** @internal Cached fast activation array S. */
  public _fastS?: Float32Array | Float64Array;

  /**
   * Create a network instance.
   *
   * When `options.seed` is provided, the constructor snapshots the global
   * connection innovation counter, resets it so the bootstrap emits
   * deterministic innovation IDs starting from 1, and restores the saved
   * counter (or advances it beyond any IDs assigned during bootstrap) once
   * construction completes. This makes repeated seeded constructions
   * reproducible without regressing the process-global counter used by other
   * networks or populations.
   *
   * @param input Number of input nodes.
   * @param output Number of output nodes.
   * @param options Optional constructor options.
   *
   * @example
   * ```ts
   * const network = new Network(2, 1, { seed: 42 });
   * const replayed = new Network(2, 1, { seed: 42 });
   * // network.connections[i].innovation matches replayed for every i
   * ```
   */
  constructor(
    input: number,
    output: number,
    options?: NetworkConstructorOptions,
  ) {
    // Step 1: Validate the required graph dimensions.
    if (typeof input === 'undefined' || typeof output === 'undefined') {
      throw new NetworkConstructorDimensionRequiredError(
        'No input or output size given',
      );
    }

    // Step 2: Validate the constructor topology contract before mutating runtime state.
    validateTopologyIntentConfiguration(options);

    // Step 3: Resolve the public topology contract and the low-level acyclic flag.
    const topologyIntent = resolveTopologyIntent(options);
    const enforceAcyclic = resolveAcyclicEnforcement(options, topologyIntent);

    // Step 3b: When a deterministic seed is requested, snapshot the current
    // global connection innovation counter, then reset it so the bootstrap emits
    // deterministic innovation IDs starting from 1. After bootstrap we restore
    // the cursor to the saved baseline (or beyond any IDs assigned during
    // bootstrap), which keeps repeated seeded construction reproducible
    // without regressing the process-global counter used by other networks.
    let savedInnovationCounter: number | undefined;
    if (options?.seed !== undefined) {
      savedInnovationCounter = Connection.nextInnovation;
      Connection.resetInnovationCounter(1);
    }

    // Step 4: Bootstrap the one-time runtime state and the initial graph shape.
    bootstrapNetwork(this as unknown as NetworkBootstrapInternals, {
      input,
      output,
      options,
      topologyIntent,
      enforceAcyclic,
    });

    // Step 4b: Restore the global innovation counter after seeded bootstrap so
    // other networks/populations never see a regressed or overlapping counter.
    if (savedInnovationCounter !== undefined) {
      const maxAssignedInnovation = this.connections.reduce(
        (maxInnovation, connection) =>
          Math.max(maxInnovation, connection.innovation),
        0,
      );
      Connection.syncInnovationCounter(
        Math.max(maxAssignedInnovation, savedInnovationCounter - 1),
      );
    }
  }

  /**
   * @internal
   * Check if fast-slab activation can be used.
   *
   * @param training Whether training mode is active.
   * @returns True when fast-slab activation can be used.
   */
  private _canUseFastSlab(training: boolean) {
    return _canUseFastSlab.call(this, training);
  }

  /**
   * @internal
   * Execute the fast slab activation path.
   *
   * @param input Input vector.
   * @returns Activation output.
   */
  private _fastSlabActivate(input: number[]) {
    return _fastSlabActivate.call(this, input);
  }

  /**
   * @internal
   * Recompute and cache topological node ordering.
   *
   * @returns Topological order payload from the delegate.
   */
  private _computeTopoOrder() {
    return _computeTopoOrder.call(this);
  }

  /**
   * @internal
   * Check whether a directed path exists between two nodes.
   *
   * @param from Source node.
   * @param to Target node.
   * @returns True when a path exists.
   */
  private _hasPath(from: Node, to: Node) {
    return _hasPath.call(this, from, to);
  }

  /**
   * @internal
   * Apply scheduled pruning if current iteration matches pruning policy.
   *
   * @param iteration Current training iteration.
   * @returns Delegate result for pruning attempt.
   */
  private _maybePrune(iteration: number) {
    return _maybePrune.call(this, iteration);
  }

  /**
   * @internal
   * Apply gradient clipping configuration.
   *
   * @param cfg Gradient clipping configuration.
   */
  private _applyGradientClipping(cfg: {
    mode: 'norm' | 'percentile' | 'layerwiseNorm' | 'layerwisePercentile';
    maxNorm?: number;
    percentile?: number;
  }): void {
    _applyGradientClippingImpl(this, cfg);
  }

  /**
   * @internal
   * Sample a Gaussian random value with an optional RNG.
   *
   * @param rng RNG function.
   * @returns Gaussian random value.
   */
  private static _gaussianRand(rng: () => number = Math.random): number {
    let uniformSampleOne = 0;
    let uniformSampleTwo = 0;
    while (uniformSampleOne === 0) uniformSampleOne = rng();
    while (uniformSampleTwo === 0) uniformSampleTwo = rng();
    return (
      Math.sqrt(-2.0 * Math.log(uniformSampleOne)) *
      Math.cos(2.0 * Math.PI * uniformSampleTwo)
    );
  }

  /**
   * Rebuild slab structures for fast activation.
   *
   * @param force Whether to force a rebuild.
   * @returns Slab rebuild result.
   */
  rebuildConnectionSlab(force = false) {
    return _rebuildConnectionSlab.call(this, force);
  }

  /**
   * Read slab structures for fast activation.
   *
   * @returns Slab connection structures.
   */
  getConnectionSlab() {
    return _getConnectionSlab.call(this);
  }

  /**
   * Public wrapper for fast slab forward pass.
   *
   * @param input Input vector.
   * @returns Activation output.
   */
  fastSlabActivate(input: number[]) {
    return this._fastSlabActivate(input);
  }

  /**
   * Insert a new hidden node by splitting a randomly chosen existing connection.
   *
   * The selected connection `from → to` is replaced by two new connections:
   * `from → newNode` and `newNode → to`. The new node's activation function
   * defaults to linear so the network's behavior is unchanged immediately
   * after the split — evolution pressure then shapes the new node over time.
   *
   * This is one of the canonical NEAT structural mutations. It increases
   * network depth without changing connectivity density significantly.
   * See [Stanley & Miikkulainen (2002)](https://nn.cs.utexas.edu/?stanley:ec02) for the motivating analysis.
   *
   * @example
   * ```ts
   * const network = new Network(2, 1);
   * network.connect(network.nodes[0], network.nodes[2]);
   * network.addNodeBetween(); // splits one connection, adds a hidden node
   * ```
   */
  addNodeBetween(): void {
    _addNodeBetweenImpl.call(this);
  }

  /**
   * Enable DropConnect with a probability in $[0,1)$.
   *
   * @param p DropConnect probability.
   */
  enableDropConnect(p: number) {
    _enableDropConnect.call(this, p);
  }
  /** Disable DropConnect. */
  disableDropConnect() {
    _disableDropConnect.call(this);
  }

  /**
   * Read a human-friendly snapshot of the current activation-ordering contract.
   *
   * Use this after activation or structural edits to see whether the runtime is
   * using a compiled schedule, a cycle fallback, or a raw-node-order fallback,
   * and what to do next if that result is not the one you expected.
   *
   * @returns Activation scheduling diagnostics snapshot.
   */
  getActivationSchedulingDiagnostics(): ActivationSchedulingDiagnostics {
    return _getActivationSchedulingDiagnostics.call(this);
  }

  /**
   * Returns the public topology intent for this network.
   *
   * @returns Current topology intent.
   */
  getTopologyIntent(): NetworkTopologyIntent {
    return _getTopologyIntent.call(this);
  }

  /**
   * Sets the public topology intent and keeps acyclic enforcement aligned.
   *
   * @param topologyIntent Desired topology intent.
   * @returns Nothing.
   */
  setTopologyIntent(topologyIntent: NetworkTopologyIntent): void {
    _setTopologyIntent.call(this, topologyIntent);
  }

  /**
   * Enable or disable acyclic topology enforcement.
   *
   * @param flag Whether to enforce acyclic connectivity.
   */
  setEnforceAcyclic(flag: boolean): void {
    _setEnforceAcyclic.call(this, flag);
  }

  /**
   * Refresh explicit ordered input and output role ids from the current graph.
   *
   * Builder, restore, and evolutionary materialization paths use this after
   * replacing `nodes` wholesale so the role contract stays explicit even while
   * activation semantics still rely on legacy ordering rules.
   *
   * @returns Nothing.
   */
  refreshExplicitIORoles(): void {
    const explicitIORoles = collectExplicitIORoles(this.nodes);
    this._inputNodeIds = explicitIORoles.inputNodeIds;
    this._outputNodeIds = explicitIORoles.outputNodeIds;
  }

  /**
   * Configure scheduled pruning during training.
   *
   * @param cfg Pruning schedule and strategy configuration.
   */
  configurePruning(cfg: {
    start: number;
    end: number;
    targetSparsity: number;
    regrowFraction?: number;
    frequency?: number;
    method?: 'magnitude' | 'snip';
  }) {
    _configurePruning.call(this, cfg);
  }

  /**
   * Configure a structural connection-growth budget for future mutations.
   *
   * @param cfg - Absolute connection cap plus optional grace headroom.
   */
  configureSparsityBudget(cfg: {
    maxConnections: number;
    growthGraceFraction?: number;
    method?: 'magnitude' | 'snip';
  }) {
    _configureSparsityBudget.call(this, cfg);
  }

  /**
   * Read the latest structural growth-budget decision snapshot.
   *
   * @returns Snapshot when a budgeted growth decision has already run.
   */
  getSparsityBudgetSnapshot(): NetworkSparsityBudgetSnapshot | undefined {
    return _getSparsityBudgetSnapshot(this);
  }

  /**
   * Compute the current connection sparsity ratio.
   *
   * @returns Current sparsity in $[0,1]$.
   */
  getCurrentSparsity(): number {
    return _getCurrentSparsity.call(this);
  }

  /**
   * Immediately prune connections until the graph reaches (or approaches)
   * a target sparsity fraction.
   *
   * Sparsity is defined as the fraction of connections removed relative to
   * the baseline connection count captured on the first call. A
   * `targetSparsity` of `0.8` means approximately 80% of the original
   * connections will be removed, leaving 20% intact.
   *
   * Two ranking strategies are available:
   *
   * - `'magnitude'` (default): removes the connections with the smallest
   *   absolute weight values — a fast, weight-magnitude heuristic.
   * - `'snip'`: removes connections ranked by a SNIP-style first-order
   *   gradient-magnitude saliency score.
   *
   * This method is suitable for evolutionary generation-based pruning
   * independent of a training-iteration schedule. For schedule-based
   * pruning during gradient training, use `configureSparsityBudget()`.
   *
   * @param targetSparsity - Fraction of original connections to remove,
   *   in the open interval `(0, 1)`. Values close to 1 produce very
   *   sparse networks.
   * @param method - Ranking strategy: `'magnitude'` or `'snip'`.
   *   Defaults to `'magnitude'`.
   *
   * @example
   * ```ts
   * const network = Network.createMLP(4, [16, 16], 2);
   * // Remove 70% of connections by weight magnitude:
   * network.pruneToSparsity(0.7);
   * ```
   */
  pruneToSparsity(
    targetSparsity: number,
    method: 'magnitude' | 'snip' = 'magnitude',
  ) {
    return _pruneToSparsity.call(this, targetSparsity, method);
  }

  /**
   * Enable weight noise using either a global standard deviation or per-hidden-layer values.
   *
   * @param stdDev Global standard deviation or hidden-layer schedule.
   */
  enableWeightNoise(stdDev: number | { perHiddenLayer: number[] }) {
    _enableWeightNoise.call(this, stdDev);
  }
  /** Disable all weight-noise settings. */
  disableWeightNoise() {
    _disableWeightNoise.call(this);
  }
  /**
   * Set a dynamic scheduler for global weight noise.
   *
   * @param fn Function mapping training step to noise standard deviation.
   */
  setWeightNoiseSchedule(fn: (step: number) => number) {
    _setWeightNoiseSchedule.call(this, fn);
  }
  /** Clear the dynamic global weight-noise schedule. */
  clearWeightNoiseSchedule() {
    _clearWeightNoiseSchedule.call(this);
  }
  /**
   * Replace the network random number generator.
   *
   * @param fn RNG function returning values in $[0,1)$.
   */
  setRandom(fn: () => number) {
    _setRandom.call(this, fn);
  }
  /**
   * Seed the internal deterministic RNG.
   *
   * Seeding makes every subsequent structural mutation, weight initialization,
   * and random choice reproducible for the same starting network. NGE uses this
   * in `runNgeLifecycle` to guarantee that the same DNA + seed + experience
   * stream produce identical growth checkpoints, including the same innovation
   * IDs for newly created connections. Omitting the seed leaves the network
   * using its default non-deterministic RNG.
   *
   * @param seed - Seed value.
   *
   * @see {@link runNgeLifecycle} for the seed-driven growth window.
   */
  setSeed(seed: number) {
    _setSeed.call(this, seed);
  }
  /** Force the next mixed-precision overflow path (test utility). */
  testForceOverflow() {
    _testForceOverflow.call(this);
  }
  /** Current training step counter. */
  get trainingStep() {
    return _getTrainingStep.call(this);
  }
  /** Last skipped stochastic-depth layers from activation runtime state. */
  get lastSkippedLayers(): number[] {
    return _getLastSkippedLayers.call(this);
  }
  /**
   * Snapshot deterministic RNG runtime state.
   *
   * @returns Current RNG snapshot.
   */
  snapshotRNG(): RNGSnapshot {
    return _snapshotRNG.call(this);
  }
  /**
   * Restore deterministic RNG function from a snapshot source.
   *
   * @param fn RNG function to restore.
   */
  restoreRNG(fn: () => number) {
    _restoreRNG.call(this, fn);
  }
  /**
   * Read the raw deterministic RNG state word.
   *
   * @returns RNG state value when present.
   */
  getRNGState(): number | undefined {
    return _getRNGState.call(this);
  }
  /**
   * Set the raw deterministic RNG state word.
   *
   * @param state RNG state value.
   */
  setRNGState(state: number) {
    _setRNGState.call(this, state);
  }
  /**
   * Read the active deterministic RNG function.
   *
   * @returns RNG function when deterministic state is initialized.
   */
  getRandomFn(): (() => number) | undefined {
    return _getRandomFn.call(this);
  }
  /**
   * Set stochastic-depth schedule function.
   *
   * @param fn Function mapping step and current schedule to next schedule.
   */
  setStochasticDepthSchedule(
    fn: (step: number, current: number[]) => number[],
  ) {
    _setStochasticDepthSchedule.call(this, fn);
  }
  /** Clear stochastic-depth schedule function. */
  clearStochasticDepthSchedule() {
    _clearStochasticDepthSchedule.call(this);
  }
  /**
   * Read regularization statistics collected during training.
   *
   * @returns Regularization stats payload.
   */
  getRegularizationStats() {
    return _getRuntimeRegularizationStats.call(this);
  }

  /**
   * Configure stochastic depth with survival probabilities per hidden layer.
   *
   * @param survival Survival probabilities for hidden layers.
   */
  setStochasticDepth(survival: number[]) {
    _setStochasticDepth.call(this, survival);
  }
  /** Disable stochastic depth. */
  disableStochasticDepth() {
    _disableStochasticDepth.call(this);
  }

  /**
   * Creates a deep copy of the network.
   * @returns {Network} A new Network instance that is a clone of the current network.
   */
  clone(): Network {
    return _cloneImpl.call(this);
  }

  /**
   * Resets all masks in the network to 1 (no dropout). Applies to both node-level and layer-level dropout.
   * Should be called after training to ensure inference is unaffected by previous dropout.
   */
  resetDropoutMasks(): void {
    _resetDropoutMasks.call(this);
  }

  /**
   * Generate a dependency-light standalone inference function for this network.
   *
   * Use this when you want to snapshot the current topology and weights into a
   * self-contained JavaScript function for deployment, offline benchmarking,
   * or browser embedding without the full training runtime.
   *
   * @returns Standalone JavaScript source for inference.
   */
  standalone(): string {
    return generateStandalone(this);
  }

  /**
   * GPU opt-in overload. Returns a `Promise<Float32Array>` so callers can await
   * the asynchronous readback.
   *
   * The GPU path is used only when `gpuDevice` is set and `isGPUEligible`
   * reports the network is dispatchable. Otherwise the call falls back to the
   * CPU path and returns a `Float32Array` wrapped in a resolved promise. This
   * overload therefore always resolves successfully; it only rejects when the
   * CPU path itself throws.
   *
   * GPU and CPU outputs agree within an absolute tolerance of `5e-1` and a
   * mean absolute error of `≤ 1e-1`.
   *
   * @param input - Input vector of length `this.input`.
   * @param options - Must contain `useGPU: true`.
   * @param _maxActivationDepth - Unused; kept for signature compatibility.
   * @returns A promise resolving to the output values.
   *
   * @example
   * ```ts
   * const adapter = await navigator.gpu.requestAdapter({
   *   powerPreference: 'high-performance',
   * });
   * network.gpuDevice = (await adapter?.requestDevice()) ?? undefined;
   * const output = await network.activate([0.5, -0.2], { useGPU: true });
   * ```
   */
  activate(
    input: number[] | Float32Array,
    options: { training?: boolean; useGPU: true },
    _maxActivationDepth?: number,
  ): Promise<Float32Array>;

  /**
   * Backward-compatible overload that accepts an options bag and routes to the
   * CPU path when `useGPU` is omitted or false.
   *
   * @param input - Input vector of length `this.input`.
   * @param options - Activation options. `training` keeps the CPU semantics;
   *   `useGPU` must be absent or false to match this overload.
   * @returns Output activations as a plain number[].
   */
  activate(
    input: number[] | Float32Array,
    options: { training?: boolean; useGPU?: false },
    _maxActivationDepth?: number,
  ): number[];

  /**
   * Activates the network using the given input array.
   *
   * Performs a forward pass through the network, calculating the activation of
   * each node. By default the CPU path is used and a plain `number[]` is
   * returned. Callers can opt into the WebGPU fast path by setting `gpuDevice`
   * and passing `{ useGPU: true }`; that overload returns a
   * `Promise<Float32Array>` because GPU readback is asynchronous.
   *
   * @param {number[] | Float32Array} input - An array or Float32Array of numerical values corresponding to the network's input nodes.
   * @param {boolean} [training=false] - Flag indicating if the activation is part of a training process.
   * @param {number} [_maxActivationDepth=1000] - Maximum allowed activation depth to prevent infinite loops/cycles (kept for signature compatibility).
   * @returns {number[]} An array of numerical values representing the activations of the network's output nodes.
   */
  activate(
    input: number[] | Float32Array,
    training?: boolean,
    _maxActivationDepth?: number,
  ): number[];

  /**
   * Implementation signature used by the overloads above.
   *
   * Existing callers passing a boolean `training` flag are unchanged. The GPU
   * path is used only when an options bag with `useGPU: true` is supplied,
   * `gpuDevice` is set, and `isGPUEligible` returns true. In every other case
   * the standard CPU `network.activate()` implementation runs.
   *
   * @param input - Input vector of length `this.input`.
   * @param trainingOrOptions - Boolean training flag or options bag.
   * @param _maxActivationDepth - Unused; kept for signature compatibility.
   * @returns Output values, or a promise when the GPU path is selected.
   */
  activate(
    input: number[] | Float32Array,
    trainingOrOptions:
      boolean | { training?: boolean; useGPU?: boolean } = false,
    _maxActivationDepth = 1000, // eslint-disable-line @typescript-eslint/no-unused-vars
  ): number[] | Promise<Float32Array> {
    const options =
      typeof trainingOrOptions === 'object'
        ? trainingOrOptions
        : { training: trainingOrOptions, useGPU: false };
    const training = options.training ?? false;
    const useGPU = options.useGPU ?? false;

    const device = this.gpuDevice;
    if (useGPU && isGPUEligible(this, device)) {
      return activateGPU(device, this, input);
    }

    return _activate.call(this, input as number[], training);
  }

  /**
   * Activates the network without calculating eligibility traces.
   * This is a performance optimization for scenarios where backpropagation is not needed,
   * such as during testing, evaluation, or deployment (inference).
   *
   * @param {number[] | Float32Array} input - An array or Float32Array of numerical values corresponding to the network's input nodes.
   *                           The length must match the network's `input` size.
   * @returns {number[]} An array of numerical values representing the activations of the network's output nodes.
   *
   * @see {@link Node.noTraceActivate}
   */
  // Delegated activation helpers
  noTraceActivate(input: number[] | Float32Array): number[] {
    return _noTraceActivate.call(this, input as number[]);
  }

  /**
   * Raw activation that can return a reusable typed array when pooling is enabled.
   * If `reuseActivationArrays` is disabled this falls back to the standard plain-array activation path.
   *
   * @param input Input vector.
   * @param training Whether to enable training-time stochastic paths.
   * @param maxActivationDepth Maximum graph depth for activation.
   * @returns Output activations as either a plain array or a reusable typed activation buffer.
   */
  activateRaw(
    input: number[],
    training = false,
    maxActivationDepth = 1000,
  ): number[] | ActivationArray {
    return _activateRaw.call(this, input, training, maxActivationDepth);
  }

  /**
   * Activate the network over a batch of input vectors (micro-batching).
   *
   * Currently iterates sample-by-sample while reusing the network's internal
   * fast-path allocations. Outputs are cloned number[] arrays for API
   * compatibility. Future optimizations can vectorize this path.
   *
   * @param inputs Array of input vectors, each length must equal this.input
   * @param training Whether to run with training-time stochastic features
   * @returns Array of output vectors, each length equals this.output
   */
  activateBatch(inputs: number[][], training = false): number[][] {
    return _activateBatch.call(this, inputs, training);
  }

  /**
   * Activate one input sequence in bounded windows while preserving carried recurrent state.
   *
   * This keeps the same output contract as repeated `activate()` calls, while
   * adding bounded window callbacks and an opt-out from collecting the full
   * output matrix when the caller wants lower sequence-retention pressure.
   *
   * @param inputs Ordered sequence of input vectors.
   * @param options Optional windowed activation settings.
   * @returns Output vectors aligned to the input order.
   */
  forwardWindowed(
    inputs: number[][],
    options?: NetworkForwardWindowOptions,
  ): number[][] {
    return _forwardWindowed.call(this, inputs, options);
  }

  /**
   * Activate one input sequence in bounded windows with cooperative runtime yields.
   *
   * Browser runtimes can use this to yield after a configurable number of
   * emitted windows so long-running sequence inference remains responsive.
   *
   * @param inputs Ordered sequence of input vectors.
   * @param options Optional async windowed activation settings.
   * @returns Output vectors aligned to the input order.
   */
  forwardWindowedAsync(
    inputs: number[][],
    options?: NetworkForwardWindowAsyncOptions,
  ): Promise<number[][]> {
    return _forwardWindowedAsync.call(this, inputs, options);
  }

  /**
   * Propagates the error backward through the network (backpropagation).
   * Calculates the error gradient for each node and connection.
   * If `update` is true, it adjusts the weights and biases based on the calculated gradients,
   * learning rate, momentum, and optional L2 regularization.
   *
   * The process starts from the output nodes and moves backward layer by layer (or topologically for recurrent nets).
   *
   * @param {number} rate - The learning rate (controls the step size of weight adjustments).
   * @param {number} momentum - The momentum factor (helps overcome local minima and speeds up convergence). Typically between 0 and 1.
   * @param {boolean} update - If true, apply the calculated weight and bias updates. If false, only calculate gradients (e.g., for batch accumulation).
   * @param {number[]} target - An array of target values corresponding to the network's output nodes.
   *                            The length must match the network's `output` size.
   * @param {number} [regularization=0] - The L2 regularization factor (lambda). Helps prevent overfitting by penalizing large weights.
   * @param {(target: number, output: number) => number} [costDerivative] - Optional derivative of the cost function for output nodes.
   * @throws {Error} If the `target` array length does not match the network's `output` size.
   *
   * @see {@link Node.propagate} for the node-level backpropagation logic.
   */
  propagate(
    rate: number,
    momentum: number,
    update: boolean,
    target: number[],
    regularization: number = 0, // L2 regularization factor (lambda)
    costDerivative?: (target: number, output: number) => number,
  ): void {
    _propagate.call(
      this,
      rate,
      momentum,
      update,
      target,
      regularization,
      costDerivative,
    );
  }

  /**
   * Clears the internal state of all nodes in the network.
   * Resets node activation, state, eligibility traces, and extended traces to their initial values (usually 0).
   * This is typically done before processing a new input sequence in recurrent networks or between training epochs if desired.
   *
   * @see {@link Node.clear}
   */
  clear(): void {
    _clearState.call(this);
  }

  /**
   * Mutates the network's structure or parameters according to the specified method.
   * This is a core operation for neuro-evolutionary algorithms (like NEAT).
   * The method argument should be one of the mutation types defined in `methods.mutation`.
   *
   * Some structural methods, especially `ADD_CONN` and `ADD_NODE`, silently
   * no-op when no eligible candidate exists (for example, a fully saturated
   * graph). The NGE juvenile applier checks the live node/edge count before and
   * after calling `mutate` so it can report the outcome truthfully as applied or
   * skipped rather than claiming growth that did not happen.
   *
   * @param method The mutation method to apply (e.g., `mutation.ADD_NODE`, `mutation.MOD_WEIGHT`).
   *                 Some methods might have associated parameters (e.g., `MOD_WEIGHT` uses `min`, `max`).
   * @throws {Error} If no valid mutation `method` is provided.
   *
   * @see {@link methods.mutation} for available mutation types.
   * @see {@link applyMorphDeltas} for the NGE wrapper that verifies structural changes.
   */
  mutate(method: MutationMethod): void {
    return _mutateImpl.call(this, method);
  }

  /**
   * Creates a connection between two nodes in the network.
   * Handles both regular connections and self-connections.
   * Adds the new connection object(s) to the appropriate network list (`connections` or `selfconns`).
   *
   * @param {Node} from - The source node of the connection.
   * @param {Node} to - The target node of the connection.
   * @param {number} [weight] - Optional weight for the connection. If not provided, a random weight is usually assigned by the underlying `Node.connect` method.
   * @returns {Connection[]} An array containing the newly created connection object(s). Typically contains one connection, but might be empty or contain more in specialized node types.
   *
   * @see {@link Node.connect}
   */
  connect(from: Node, to: Node, weight?: number): Connection[] {
    return _connect.call(this, from, to, weight);
  }

  /**
   * Creates many connections in one ordered structural edit batch.
   *
   * This preserves the same legality checks and deterministic default-weight
   * behavior as repeated `connect()` calls, but it reserves network-level
   * storage once for the whole request shelf.
   *
   * @param requests Ordered connection requests.
   * @returns Flattened created connection objects in request order.
   */
  connectBatch(requests: readonly NetworkConnectionRequest[]): Connection[] {
    return _connectBatch.call(this, requests);
  }

  /**
   * Gates a connection with a specified node.
   * The activation of the `node` (gater) will modulate the weight of the `connection`.
   * Adds the connection to the network's `gates` list.
   *
   * @param {Node} node - The node that will act as the gater. Must be part of this network.
   * @param {Connection} connection - The connection to be gated.
   * @throws {Error} If the provided `node` is not part of this network.
   * @throws {Error} If the `connection` is already gated (though currently handled with a warning).
   *
   * @see {@link Node.gate}
   */
  gate(node: Node, connection: Connection) {
    return _gate.call(this, node, connection);
  }

  /**
   * Removes a node from the network.
   * This involves:
   * 1. Disconnecting all incoming and outgoing connections associated with the node.
   * 2. Removing self-connections.
   * 3. Removing the node from the `nodes` array.
   * 4. Attempting to reconnect the node's direct predecessors to its direct successors
   *    to maintain network flow, if possible and configured.
   * 5. Handling gates involving the removed node (ungating connections gated *by* this node,
   *    and potentially re-gating connections that were gated *by other nodes* onto the removed node's connections).
   *
   * @param {Node} node - The node instance to remove. Must exist within the network's `nodes` list.
   * @throws {Error} If the specified `node` is not found in the network's `nodes` list.
   */
  remove(node: Node) {
    return _removeNodeStandalone.call(this, node);
  }

  /**
   * Disconnects two nodes, removing the connection between them.
   * Handles both regular connections and self-connections.
   * If the connection being removed was gated, it is also ungated.
   *
   * @param {Node} from - The source node of the connection to remove.
   * @param {Node} to - The target node of the connection to remove.
   *
   * @see {@link Node.disconnect}
   */
  disconnect(from: Node, to: Node): void {
    return _disconnect.call(this, from, to);
  }

  /**
   * Removes the gate from a specified connection.
   * The connection will no longer be modulated by its gater node.
   * Removes the connection from the network's `gates` list.
   *
   * @param {Connection} connection - The connection object to ungate.
   * @throws {Error} If the provided `connection` is not found in the network's `gates` list (i.e., it wasn't gated).
   *
   * @see {@link Node.ungate}
   */
  ungate(connection: Connection) {
    return _ungate.call(this, connection);
  }

  /**
   * Train the network against a supervised dataset using the gradient-based
   * training chapter.
   *
   * This wrapper keeps the public `Network` API stable while the training
   * helpers own batching, optimizer steps, regularization, and mixed-precision
   * runtime behavior.
   *
   * @param set Supervised samples with `input` and `output` vectors.
   * @param options Training options such as learning rate, iteration limits, batching, and optimizer settings.
   * @returns Aggregate training result with final error, iteration count, and elapsed time.
   */
  train(
    set: { input: number[]; output: number[] }[],
    options: unknown,
  ): { error: number; iterations: number; time: number } {
    return _trainImpl(this, set, options as TrainingOptions);
  }

  /** Returns last recorded raw (pre-update) gradient L2 norm. */
  getRawGradientNorm(): number {
    return _getRawGradientNorm.call(this);
  }
  /** Returns current mixed precision loss scale (1 if disabled). */
  getLossScale(): number {
    return _getLossScale.call(this);
  }
  /** Returns last gradient clipping group count (0 if no clipping yet). */
  getLastGradClipGroupCount(): number {
    return _getLastGradClipGroupCount.call(this);
  }
  /** Consolidated training stats snapshot. */
  getTrainingStats() {
    return _getTrainingStats.call(this);
  }
  /** Utility: adjust rate for accumulation mode (use result when switching to 'sum' to mimic 'average'). */
  static adjustRateForAccumulation(
    rate: number,
    accumulationSteps: number,
    reduction: 'average' | 'sum',
  ) {
    if (reduction === 'sum' && accumulationSteps > 1)
      return rate / accumulationSteps;
    return rate;
  }

  /**
   * Evolve the network against a dataset using the neuroevolution chapter.
   *
   * The implementation lives outside this class so the public surface stays
   * orchestration-first while population search, mutation policy, and stopping
   * criteria remain chapter-owned.
   *
   * @param set Evaluation samples with `input` and `output` vectors.
   * @param options Evolution options controlling population search and stopping criteria.
   * @returns Promise resolving to the final error, iteration count, and elapsed time.
   */
  async evolve(
    set: { input: number[]; output: number[] }[],
    options: Record<string, unknown> | undefined,
  ): Promise<{ error: number; iterations: number; time: number }> {
    const { evolveNetwork } = await import('./network.utils');
    return evolveNetwork.call(this, set, options as never);
  }

  /**
   * Tests the network's performance on a given dataset.
   * Calculates the average error over the dataset using a specified cost function.
   * Uses `noTraceActivate` for efficiency as gradients are not needed.
   * Handles dropout scaling if dropout was used during training.
   *
   * @param {{ input: number[]; output: number[] }[]} set - The test dataset, an array of objects with `input` and `output` arrays.
   * @param {function} [cost=methods.Cost.MSE] - The cost function to evaluate the error. Defaults to Mean Squared Error.
   * @returns {{ error: number; time: number }} An object containing the calculated average error over the dataset and the time taken for the test in milliseconds.
   */
  test(
    set: { input: number[]; output: number[] }[],
    cost?: (target: number[], output: number[]) => number,
  ): { error: number; time: number } {
    return _testNetwork.call(this, set, cost);
  }

  /** Lightweight tuple serializer delegating to network.serialize.ts */
  serialize(): CompactSerializedNetworkTuple {
    return _serialize.call(this);
  }

  /**
   * Creates a Network instance from serialized data produced by `serialize()`.
   * Reconstructs the network structure and state based on the provided arrays.
   *
   * @param {unknown[]} data - The serialized network data array, typically obtained from `network.serialize()`.
   *                       Expected format: `[activations, states, squashNames, connectionData, inputSize, outputSize]`
   *                       with optional trailing `nodeGeneIds` and `topologyIntent` slots for
   *                       identity-preserving restore paths.
   * @param {number} [inputSize] - Optional input size override.
   * @param {number} [outputSize] - Optional output size override.
   * @returns {Network} A new Network instance reconstructed from the serialized data.
   * @static
   */
  /** Static lightweight tuple deserializer delegate */
  static deserialize(
    data: CompactSerializedNetworkTuple | unknown[],
    inputSize?: number,
    outputSize?: number,
  ): Network {
    return _deserialize(data as never, inputSize, outputSize);
  }

  /**
   * Converts the network into a JSON object representation (latest standard).
   * Includes formatVersion, and only serializes properties needed for full reconstruction.
   * All references are by index. Excludes runtime-only properties (activation, state, traces).
   *
   * @returns {object} A JSON-compatible object representing the network.
   */
  /** Verbose JSON serializer delegate */
  toJSON(): Record<string, unknown> {
    return _toJSONImpl.call(this) as unknown as Record<string, unknown>;
  }

  /**
   * Resolves a stable architecture descriptor for telemetry/UI consumers.
   *
   * Prefers live graph analysis and only falls back to hydrated serialization
   * metadata when graph-based resolution is purely inferred.
   *
   * @returns Architecture descriptor with hidden-layer widths and provenance.
   */
  describeArchitecture(): NetworkArchitectureDescriptor {
    return _resolveArchitectureDescriptor(this);
  }

  /**
   * Resolves the validated temporal-module structure for diagnostics and visualization.
   *
   * Call this when the coarse hidden-layer descriptor is not enough and you
   * need the explicit recurrent-module and gated-block ownership story that the
   * runtime builders preserve for LSTM, GRU, and NARX networks.
   *
   * @returns Temporal-structure descriptor synchronized against the live graph.
   */
  describeTemporalStructure(): NetworkTemporalStructureDescriptor {
    return _describeTemporalStructure(this);
  }

  /**
   * Reconstructs a network from a JSON object (latest standard).
   * Handles formatVersion, robust error handling, and index-based references.
   * @param {object} json - The JSON object representing the network.
   * @returns {Network} The reconstructed network.
   */
  /** Verbose JSON static deserializer */
  static fromJSON(json: Record<string, unknown>): Network {
    return _fromJSONImpl(json as never);
  }

  /**
   * Creates a new offspring network by performing crossover between two parent networks.
   * This method implements the crossover mechanism inspired by the NEAT algorithm and described
   * in the Instinct paper, combining genes (nodes and connections) from both parents.
   * Fitness scores can influence the inheritance process. Matching genes are inherited randomly,
   * while disjoint/excess genes are typically inherited from the fitter parent (or randomly if fitness is equal or `equal` flag is set).
   *
   * @param {Network} network1 - The first parent network.
   * @param {Network} network2 - The second parent network.
   * @param {boolean} [equal=false] - If true, disjoint and excess genes are inherited randomly regardless of fitness.
   *                                  If false (default), they are inherited from the fitter parent.
   * @returns {Network} A new Network instance representing the offspring.
   * @throws {Error} If the input or output sizes of the parent networks do not match.
   *
   * @see Instinct Algorithm - Section 2 Crossover
   * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6}
   * @static
   */
  /** NEAT-style crossover delegate. */
  static crossOver(
    network1: Network,
    network2: Network,
    equal: boolean = false,
  ): Network {
    return _crossOver(network1, network2, equal);
  }

  /**
   * Sets specified properties (e.g., bias, squash function) for all nodes in the network.
   * Useful for initializing or resetting node properties uniformly.
   *
   * @param {object} values - An object containing the properties and values to set.
   * @param {number} [values.bias] - If provided, sets the bias for all nodes.
   * @param {function} [values.squash] - If provided, sets the squash (activation) function for all nodes.
   *                                     Should be a valid activation function (e.g., from `methods.Activation`).
   */
  set(values: {
    bias?: number;
    squash?: (x: number, derivate?: boolean) => number;
  }): void {
    // Iterate through all nodes in the network.
    this.nodes.forEach((node) => {
      // Update bias if provided in the values object.
      if (typeof values.bias !== 'undefined') {
        node.bias = values.bias;
      }
      // Update squash function if provided.
      if (typeof values.squash !== 'undefined') {
        node.squash = values.squash;
      }
    });
  }

  /**
   * Exports the network to ONNX format (JSON object, minimal MLP support).
   * Only standard feedforward architectures and standard activations are supported.
   * Gating, custom activations, and evolutionary features are ignored or replaced with Identity.
   *
   * @returns {import('./onnx').OnnxModel} ONNX model as a JSON object.
   */
  toONNX() {
    return exportToONNX(this);
  }

  /**
   * Creates a fully connected, strictly layered MLP network.
   * @param {number} inputCount - Number of input nodes
   * @param {number[]} hiddenCounts - Array of hidden layer sizes (e.g. [2,3] for two hidden layers)
   * @param {number} outputCount - Number of output nodes
   * @returns {Network} A new, fully connected, layered MLP
   */
  static createMLP(
    inputCount: number,
    hiddenCounts: number[],
    outputCount: number,
  ): Network {
    return _createMLP.call(this, inputCount, hiddenCounts, outputCount);
  }

  /**
   * Construct a runnable network from mixed `Node`, `Group`, and `Layer` parts.
   *
   * This builder compiles the provided parts into the ordinary `Network`
   * runtime, preserving explicit input/output ordering and then rebuilding the
   * scheduling cache in either acyclic or recurrent mode.
   *
   * @param parts Mixed architecture parts to flatten.
   * @param options Optional construct-time validation, ordering, and runtime flags.
   * @returns Materialized runtime plus lightweight diagnostics.
   *
   * @example
   * ```ts
   * const sensor = new Node('input');
   * const hidden = new Group(2);
   * const readout = Layer.dense(1, 'output');
   *
   * sensor.connect(hidden);
   * hidden.connect(readout);
   *
   * const { network } = Network.construct([sensor, hidden, readout]);
   * ```
   */
  static construct(
    parts: readonly ConstructPart[],
    options?: ConstructOptions,
  ): ConstructResult {
    return _constructNetwork.call(this, parts, options);
  }

  /**
   * Rebuilds the network's connections array from all per-node connections.
   * This ensures that the network.connections array is consistent with the actual
   * outgoing connections of all nodes. Useful after manual wiring or node manipulation.
   *
   * @param {Network} net - The network instance to rebuild connections for.
   * @returns {void}
   *
   * Example usage:
   *   Network.rebuildConnections(net);
   */
  static rebuildConnections(net: Network): void {
    _rebuildConnections(net);
  }
}

function collectExplicitIORoles(nodes: readonly Node[]): ExplicitIORoles {
  const explicitIORoles: ExplicitIORoles = {
    inputNodeIds: [],
    outputNodeIds: [],
  };

  for (const node of nodes) {
    if (node.type === 'input') {
      explicitIORoles.inputNodeIds.push(node.geneId);
      continue;
    }

    if (node.type === 'output') {
      explicitIORoles.outputNodeIds.push(node.geneId);
    }
  }

  return explicitIORoles;
}
