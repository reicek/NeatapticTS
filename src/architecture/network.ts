import Node from './node';
import Connection from './connection';
import Multi from '../multithreading/multi';
import * as methods from '../methods/methods';
import mutation from '../methods/mutation'; // Import mutation methods
import { config } from '../config'; // Import configuration settings
import { activationArrayPool, ActivationArray } from './activationArrayPool';
// ONNX export/import now lives in ./network/network.onnx (re-exported via ./onnx for backwards compat)
import { exportToONNX } from './onnx';
import { generateStandalone } from './network/network.standalone';
import {
  computeTopoOrder as _computeTopoOrder,
  hasPath as _hasPath,
} from './network/network.topology';
import {
  rebuildConnectionSlab as _rebuildConnectionSlab,
  fastSlabActivate as _fastSlabActivate,
  canUseFastSlab as _canUseFastSlab,
  getConnectionSlab as _getConnectionSlab,
} from './network/network.slab';
import {
  maybePrune as _maybePrune,
  pruneToSparsity as _pruneToSparsity,
  getCurrentSparsity as _getCurrentSparsity,
} from './network/network.prune';
import {
  gate as _gate,
  ungate as _ungate,
  removeNode as _removeNode,
} from './network/network.gating';
import {
  setSeed as _setSeed,
  snapshotRNG as _snapshotRNG,
  restoreRNG as _restoreRNG,
  getRNGState as _getRNGState,
  setRNGState as _setRNGState,
} from './network/network.deterministic';
import { getRegularizationStats as _getRegularizationStats } from './network/network.stats';
import { removeNode as _removeNodeStandalone } from './network/network.remove';
import {
  connect as _connect,
  disconnect as _disconnect,
} from './network/network.connect';
import {
  serialize as _serialize,
  deserialize as _deserialize,
  toJSONImpl as _toJSONImpl,
  fromJSONImpl as _fromJSONImpl,
} from './network/network.serialize';
import { crossOver as _crossOver } from './network/network.genetic';

export default class Network {
  input: number;
  output: number;
  score?: number;
  nodes: Node[];
  connections: Connection[];
  gates: Connection[];
  selfconns: Connection[];
  dropout: number = 0;
  private _dropConnectProb: number = 0;
  private _lastGradNorm?: number;
  private _optimizerStep: number = 0;
  private _weightNoiseStd: number = 0;
  private _weightNoisePerHidden: number[] = [];
  private _weightNoiseSchedule?: (step: number) => number;
  private _stochasticDepth: number[] = [];
  private _wnOrig?: number[];
  private _trainingStep: number = 0;
  private _rand: () => number = Math.random;
  private _rngState?: number;
  private _lastStats: any = null;
  private _stochasticDepthSchedule?: (
    step: number,
    current: number[]
  ) => number[];
  private _mixedPrecision: { enabled: boolean; lossScale: number } = {
    enabled: false,
    lossScale: 1,
  };
  private _mixedPrecisionState: {
    goodSteps: number;
    badSteps: number;
    minLossScale: number;
    maxLossScale: number;
    overflowCount?: number;
    scaleUpEvents?: number;
    scaleDownEvents?: number;
  } = {
    goodSteps: 0,
    badSteps: 0,
    minLossScale: 1,
    maxLossScale: 65536,
    overflowCount: 0,
    scaleUpEvents: 0,
    scaleDownEvents: 0,
  };
  private _gradAccumMicroBatches: number = 0;
  private _currentGradClip?: {
    mode: 'norm' | 'percentile' | 'layerwiseNorm' | 'layerwisePercentile';
    maxNorm?: number;
    percentile?: number;
  };
  private _lastRawGradNorm: number = 0;
  private _accumulationReduction: 'average' | 'sum' = 'average';
  private _gradClipSeparateBias: boolean = false;
  private _lastGradClipGroupCount: number = 0;
  private _lastOverflowStep: number = -1;
  private _forceNextOverflow: boolean = false;
  private _pruningConfig?: {
    start: number;
    end: number;
    targetSparsity: number;
    regrowFraction: number;
    frequency: number;
    method: 'magnitude' | 'snip';
    lastPruneIter?: number;
  };
  private _initialConnectionCount?: number;
  private _enforceAcyclic: boolean = false;
  private _topoOrder: Node[] | null = null;
  private _topoDirty: boolean = true;
  private _globalEpoch: number = 0;
  layers?: any[];
  private _evoInitialConnCount?: number; // baseline for evolution-time pruning
  private _activationPrecision: 'f64' | 'f32' = 'f64'; // typed array precision for compiled path
  private _reuseActivationArrays: boolean = false; // reuse pooled output arrays
  private _returnTypedActivations: boolean = false; // if true and reuse enabled, return typed array directly
  private _activationPool?: Float32Array | Float64Array; // pooled output array
  // Packed connection slab fields (for memory + cache efficiency when iterating connections)
  private _connWeights?: Float32Array | Float64Array;
  private _connFrom?: Uint32Array;
  private _connTo?: Uint32Array;
  private _slabDirty: boolean = true;
  private _useFloat32Weights: boolean = true;
  // Cached node.index maintenance (avoids repeated this.nodes.indexOf in hot paths like slab rebuild)
  private _nodeIndexDirty: boolean = true; // when true, node.index values must be reassigned sequentially
  // Fast slab forward path structures
  private _outStart?: Uint32Array;
  private _outOrder?: Uint32Array;
  private _adjDirty: boolean = true;
  // Cached typed arrays for fast slab forward pass
  private _fastA?: Float32Array | Float64Array;
  private _fastS?: Float32Array | Float64Array;
  // Internal hint: track a preferred linear chain edge to split on subsequent ADD_NODE mutations
  // to encourage deep path formation even in stochastic modes. Updated each time we split it.
  private _preferredChainEdge?: Connection;

  // Slab helpers delegated to network.slab.ts
  private _canUseFastSlab(training: boolean) {
    return _canUseFastSlab.call(this, training);
  }
  private _fastSlabActivate(input: number[]) {
    return _fastSlabActivate.call(this, input);
  }
  rebuildConnectionSlab(force = false) {
    return _rebuildConnectionSlab.call(this, force);
  }
  getConnectionSlab() {
    return _getConnectionSlab.call(this);
  }
  constructor(
    input: number,
    output: number,
    options?: {
      minHidden?: number;
      seed?: number;
      enforceAcyclic?: boolean;
      activationPrecision?: 'f32' | 'f64';
      reuseActivationArrays?: boolean;
      returnTypedActivations?: boolean;
    }
  ) {
    // Validate that input and output sizes are provided.
    if (typeof input === 'undefined' || typeof output === 'undefined') {
      throw new Error('No input or output size given');
    }

    // Initialize network properties
    this.input = input;
    this.output = output;
    this.nodes = [];
    this.connections = [];
    this.gates = [];
    this.selfconns = [];
    this.dropout = 0;
    this._enforceAcyclic = (options as any)?.enforceAcyclic || false;
    if (options?.activationPrecision) {
      this._activationPrecision = options.activationPrecision;
    } else if (config.float32Mode) {
      this._activationPrecision = 'f32';
    }
    if (options?.reuseActivationArrays) this._reuseActivationArrays = true;
    if (options?.returnTypedActivations) this._returnTypedActivations = true;
    // Configure and prewarm the activation pool based on global config
    try {
      if (typeof config.poolMaxPerBucket === 'number')
        activationArrayPool.setMaxPerBucket(config.poolMaxPerBucket);
      const prewarm =
        typeof config.poolPrewarmCount === 'number'
          ? config.poolPrewarmCount
          : 2;
      activationArrayPool.prewarm(this.output, prewarm);
    } catch {}

    if (options?.seed !== undefined) {
      this.setSeed(options.seed);
    }

    for (let i = 0; i < this.input + this.output; i++) {
      const type = i < this.input ? 'input' : 'output';
      this.nodes.push(new Node(type, undefined, this._rand));
    }
    for (let i = 0; i < this.input; i++) {
      for (let j = this.input; j < this.input + this.output; j++) {
        const weight = this._rand() * this.input * Math.sqrt(2 / this.input);
        this.connect(this.nodes[i], this.nodes[j], weight);
      }
    }

    const minHidden = options?.minHidden || 0;
    if (minHidden > 0) {
      while (this.nodes.length < this.input + this.output + minHidden) {
        this.addNodeBetween();
      }
    }
  }

  // --- Added: structural helper referenced by constructor (split a random connection) ---
  private addNodeBetween(): void {
    if (this.connections.length === 0) return;
    const idx = Math.floor(this._rand() * this.connections.length);
    const conn = this.connections[idx];
    if (!conn) return;
    // Remove original connection
    this.disconnect(conn.from, conn.to);
    // Create new hidden node
    const newNode = new Node('hidden', undefined, this._rand);
    this.nodes.push(newNode);
    // Connect from->newNode and newNode->to
    this.connect(conn.from, newNode, conn.weight); // keep original weight on first leg
    this.connect(newNode, conn.to, 1); // second leg weight initialised randomly or 1
    // Invalidate topo cache
    this._topoDirty = true;
    this._nodeIndexDirty = true; // structure changed
  }

  // --- DropConnect API (re-added for tests) ---
  enableDropConnect(p: number) {
    if (p < 0 || p >= 1)
      throw new Error('DropConnect probability must be in [0,1)');
    this._dropConnectProb = p;
  }
  disableDropConnect() {
    this._dropConnectProb = 0;
  }

  // --- Acyclic enforcement toggle (used by tests) ---
  setEnforceAcyclic(flag: boolean) {
    this._enforceAcyclic = !!flag;
  }
  private _computeTopoOrder() {
    return _computeTopoOrder.call(this);
  }
  private _hasPath(from: Node, to: Node) {
    return _hasPath.call(this, from, to);
  }

  // --- Pruning configuration & helpers ---
  configurePruning(cfg: {
    start: number;
    end: number;
    targetSparsity: number;
    regrowFraction?: number;
    frequency?: number;
    method?: 'magnitude' | 'snip';
  }) {
    const { start, end, targetSparsity } = cfg;
    if (start < 0 || end < start)
      throw new Error('Invalid pruning schedule window');
    if (targetSparsity <= 0 || targetSparsity >= 1)
      throw new Error('targetSparsity must be in (0,1)');
    this._pruningConfig = {
      start,
      end,
      targetSparsity,
      regrowFraction: cfg.regrowFraction ?? 0,
      frequency: cfg.frequency ?? 1,
      method: cfg.method || 'magnitude',
      lastPruneIter: undefined,
    };
    this._initialConnectionCount = this.connections.length;
  }
  getCurrentSparsity(): number {
    return _getCurrentSparsity.call(this);
  }
  private _maybePrune(iteration: number) {
    return _maybePrune.call(this, iteration);
  }

  /**
   * Immediately prune connections to reach (or approach) a target sparsity fraction.
   * Used by evolutionary pruning (generation-based) independent of training iteration schedule.
   * @param targetSparsity fraction in (0,1). 0.8 means keep 20% of original (if first call sets baseline)
   * @param method 'magnitude' | 'snip'
   */
  pruneToSparsity(
    targetSparsity: number,
    method: 'magnitude' | 'snip' = 'magnitude'
  ) {
    return _pruneToSparsity.call(this, targetSparsity, method);
  }

  /** Enable weight noise. Provide a single std dev number or { perHiddenLayer: number[] }. */
  enableWeightNoise(stdDev: number | { perHiddenLayer: number[] }) {
    if (typeof stdDev === 'number') {
      if (stdDev < 0) throw new Error('Weight noise stdDev must be >= 0');
      this._weightNoiseStd = stdDev;
      this._weightNoisePerHidden = [];
    } else if (stdDev && Array.isArray(stdDev.perHiddenLayer)) {
      if (!this.layers || this.layers.length < 3)
        throw new Error(
          'Per-hidden-layer weight noise requires a layered network with at least one hidden layer'
        );
      const hiddenLayerCount = this.layers.length - 2;
      if (stdDev.perHiddenLayer.length !== hiddenLayerCount)
        throw new Error(
          `Expected ${hiddenLayerCount} std dev entries (one per hidden layer), got ${stdDev.perHiddenLayer.length}`
        );
      if (stdDev.perHiddenLayer.some((s) => s < 0))
        throw new Error('Weight noise std devs must be >= 0');
      this._weightNoiseStd = 0; // disable global
      this._weightNoisePerHidden = stdDev.perHiddenLayer.slice();
    } else {
      throw new Error('Invalid weight noise configuration');
    }
  }
  disableWeightNoise() {
    this._weightNoiseStd = 0;
    this._weightNoisePerHidden = [];
  }
  setWeightNoiseSchedule(fn: (step: number) => number) {
    this._weightNoiseSchedule = fn;
  }
  clearWeightNoiseSchedule() {
    this._weightNoiseSchedule = undefined;
  }
  setRandom(fn: () => number) {
    this._rand = fn;
  }
  setSeed(seed: number) {
    _setSeed.call(this, seed);
  }
  testForceOverflow() {
    this._forceNextOverflow = true;
  }
  get trainingStep() {
    return this._trainingStep;
  }
  get lastSkippedLayers(): number[] {
    return (this as any)._lastSkippedLayers || [];
  }
  snapshotRNG(): any {
    return _snapshotRNG.call(this);
  }
  restoreRNG(fn: () => number) {
    _restoreRNG.call(this, fn);
  }
  getRNGState(): number | undefined {
    return _getRNGState.call(this);
  }
  setRNGState(state: number) {
    _setRNGState.call(this, state);
  }
  setStochasticDepthSchedule(
    fn: (step: number, current: number[]) => number[]
  ) {
    this._stochasticDepthSchedule = fn;
  }
  clearStochasticDepthSchedule() {
    this._stochasticDepthSchedule = undefined;
  }
  getRegularizationStats() {
    return _getRegularizationStats.call(this);
  }

  /** Configure stochastic depth with survival probabilities per hidden layer (length must match hidden layer count when using layered network). */
  setStochasticDepth(survival: number[]) {
    if (!Array.isArray(survival)) throw new Error('survival must be an array');
    if (survival.some((p) => p <= 0 || p > 1))
      throw new Error('Stochastic depth survival probs must be in (0,1]');
    if (!this.layers || this.layers.length === 0)
      throw new Error('Stochastic depth requires layer-based network');
    // layers includes input and output; hidden layers are layers[1..length-2]
    const hiddenLayerCount = Math.max(0, this.layers.length - 2);
    if (survival.length !== hiddenLayerCount)
      throw new Error(
        `Expected ${hiddenLayerCount} survival probabilities for hidden layers, got ${survival.length}`
      );
    this._stochasticDepth = survival.slice();
  }
  disableStochasticDepth() {
    this._stochasticDepth = [];
  }

  /**
   * Creates a deep copy of the network.
   * @returns {Network} A new Network instance that is a clone of the current network.
   */
  clone(): Network {
    return Network.fromJSON(this.toJSON());
  }

  /**
   * Resets all masks in the network to 1 (no dropout). Applies to both node-level and layer-level dropout.
   * Should be called after training to ensure inference is unaffected by previous dropout.
   */
  resetDropoutMasks(): void {
    if (this.layers && this.layers.length > 0) {
      for (const layer of this.layers) {
        if (typeof layer.nodes !== 'undefined') {
          for (const node of layer.nodes) {
            if (typeof node.mask !== 'undefined') node.mask = 1;
          }
        }
      }
    } else {
      for (const node of this.nodes) {
        if (typeof node.mask !== 'undefined') node.mask = 1;
      }
    }
  }

  // Delegated standalone generator
  standalone(): string {
    return generateStandalone(this as any);
  }

  /**
   * Activates the network using the given input array.
   * Performs a forward pass through the network, calculating the activation of each node.
   *
   * @param {number[]} input - An array of numerical values corresponding to the network's input nodes.
   * @param {boolean} [training=false] - Flag indicating if the activation is part of a training process.
   * @param {number} [maxActivationDepth=1000] - Maximum allowed activation depth to prevent infinite loops/cycles.
   * @returns {number[]} An array of numerical values representing the activations of the network's output nodes.
   */
  /**
   * Standard activation API returning a plain number[] for backward compatibility.
   * Internally may use pooled typed arrays; if so they are cloned before returning.
   */
  activate(
    input: number[],
    training = false,
    maxActivationDepth = 1000
  ): number[] {
    if (this._enforceAcyclic && this._topoDirty) this._computeTopoOrder();
    if (!Array.isArray(input) || input.length !== this.input) {
      throw new Error(
        `Input size mismatch: expected ${this.input}, got ${
          input ? input.length : 'undefined'
        }`
      );
    }
    // Fast slab path (inference-only, ungated, acyclic, no stochastic features)
    if (this._canUseFastSlab(training)) {
      try {
        return this._fastSlabActivate(input);
      } catch {
        /* fall back */
      }
    }
    // Acquire pooled activation array for outputs
    const outputArr = activationArrayPool.acquire(this.output);

    // Check for empty or corrupted network structure
    if (!this.nodes || this.nodes.length === 0) {
      throw new Error(
        'Network structure is corrupted or empty. No nodes found.'
      );
    }

    let output: ActivationArray = outputArr;
    (this as any)._lastSkippedLayers = [];
    const stats = {
      droppedHiddenNodes: 0,
      totalHiddenNodes: 0,
      droppedConnections: 0,
      totalConnections: this.connections.length,
      skippedLayers: [] as number[],
      weightNoise: { count: 0, sumAbs: 0, maxAbs: 0, meanAbs: 0 },
    };
    // Pre-apply weight noise
    let appliedWeightNoise = false;
    let dynamicStd = this._weightNoiseStd;
    if (training) {
      if (this._weightNoiseSchedule)
        dynamicStd = this._weightNoiseSchedule(this._trainingStep);
      if (dynamicStd > 0 || this._weightNoisePerHidden.length > 0) {
        for (const c of this.connections) {
          if ((c as any)._origWeightNoise != null) continue;
          (c as any)._origWeightNoise = c.weight;
          let std = dynamicStd;
          if (this._weightNoisePerHidden.length > 0 && this.layers) {
            let fromLayerIndex = -1;
            for (let li = 0; li < this.layers.length; li++) {
              if (this.layers[li].nodes.includes(c.from)) {
                fromLayerIndex = li;
                break;
              }
            }
            if (fromLayerIndex > 0 && fromLayerIndex < this.layers.length) {
              const hiddenIdx = fromLayerIndex - 1;
              if (
                hiddenIdx >= 0 &&
                hiddenIdx < this._weightNoisePerHidden.length
              )
                std = this._weightNoisePerHidden[hiddenIdx];
            }
          }
          if (std > 0) {
            const noise = std * Network._gaussianRand(this._rand);
            c.weight += noise;
            (c as any)._wnLast = noise;
            appliedWeightNoise = true;
          } else {
            (c as any)._wnLast = 0;
          }
        }
      }
    }
    // Optional stochastic depth schedule update
    if (
      training &&
      this._stochasticDepthSchedule &&
      this._stochasticDepth.length > 0
    ) {
      const updated = this._stochasticDepthSchedule(
        this._trainingStep,
        this._stochasticDepth.slice()
      );
      if (
        Array.isArray(updated) &&
        updated.length === this._stochasticDepth.length &&
        !updated.some((p) => p <= 0 || p > 1)
      ) {
        this._stochasticDepth = updated.slice();
      }
    }
    if (
      this.layers &&
      this.layers.length > 0 &&
      this._stochasticDepth.length > 0
    ) {
      // Layered activation with stochastic depth
      let acts: number[] | undefined;
      for (let li = 0; li < this.layers.length; li++) {
        const layer = this.layers[li];
        const isHidden = li > 0 && li < this.layers.length - 1;
        let skip = false;
        if (training && isHidden) {
          const hiddenIndex = li - 1;
          if (hiddenIndex < this._stochasticDepth.length) {
            const surviveProb = this._stochasticDepth[hiddenIndex];
            skip = this._rand() >= surviveProb;
            if (skip) {
              // Only skip if size matches previous outputs
              if (!acts || acts.length !== layer.nodes.length) skip = false;
            }
            if (!skip) {
              // Activate (input layer gets input array)
              const raw =
                li === 0
                  ? layer.activate(input, training)
                  : layer.activate(undefined, training);
              acts =
                surviveProb < 1
                  ? raw.map((a: number) => a * (1 / surviveProb))
                  : raw;
              continue;
            }
          }
        }
        if (skip) {
          (this as any)._lastSkippedLayers.push(li);
          stats.skippedLayers.push(li);
          // identity: acts unchanged
          continue;
        }
        const raw =
          li === 0
            ? layer.activate(input, training)
            : layer.activate(undefined, training);
        acts = raw;
      }
      if (acts) {
        for (let i = 0; i < acts.length && i < this.output; i++)
          output[i] = acts[i];
      }
    } else if (this.layers && this.layers.length > 0) {
      // Layered activation with optional node-level dropout (replicating legacy behavior expected by tests)
      let lastActs: number[] | undefined;
      for (let li = 0; li < this.layers.length; li++) {
        const layer = this.layers[li];
        const isHidden = li > 0 && li < this.layers.length - 1;
        // Always call layer.activate with training=false to avoid its uniform layer-level dropout; we'll handle per-node masks ourselves
        const raw =
          li === 0
            ? layer.activate(input, false)
            : layer.activate(undefined, false);
        // Apply node-level dropout to hidden layers if requested
        if (isHidden && training && this.dropout > 0) {
          let dropped = 0;
          for (const node of layer.nodes) {
            node.mask = this._rand() < this.dropout ? 0 : 1;
            stats.totalHiddenNodes++;
            if (node.mask === 0) stats.droppedHiddenNodes++;
            if (node.mask === 0) {
              node.activation = 0; // zero activation so downstream sees dropout
              dropped++;
            }
          }
          // Safeguard: ensure at least one active node remains
          if (dropped === layer.nodes.length && layer.nodes.length > 0) {
            const idx = Math.floor(this._rand() * layer.nodes.length);
            layer.nodes[idx].mask = 1;
            // Recompute activation for that single node using previous layer outputs
            // Simplified: keep existing raw value captured earlier in raw[idx]
            layer.nodes[idx].activation = raw[idx];
          }
        } else if (isHidden) {
          // Ensure masks are 1 during inference
          for (const node of layer.nodes) node.mask = 1;
        }
        lastActs = raw; // (raw may have been partially zeroed above via node.activation edits; raw array still original but not used after output layer)
      }
      if (lastActs) {
        if (this._reuseActivationArrays) {
          for (let i = 0; i < lastActs.length && i < this.output; i++)
            (output as any)[i] = lastActs[i];
        } else {
          for (let i = 0; i < lastActs.length && i < this.output; i++)
            (output as any)[i] = lastActs[i];
        }
      }
    } else {
      // Node-based activation (legacy, node-level dropout)
      let hiddenNodes = this.nodes.filter((node) => node.type === 'hidden');
      let droppedCount = 0;
      if (training && this.dropout > 0) {
        // Randomly drop hidden nodes
        for (const node of hiddenNodes) {
          node.mask = this._rand() < this.dropout ? 0 : 1;
          stats.totalHiddenNodes++;
          if (node.mask === 0) {
            droppedCount++;
            stats.droppedHiddenNodes++;
          }
        }
        // SAFEGUARD: Ensure at least one hidden node is active
        if (droppedCount === hiddenNodes.length && hiddenNodes.length > 0) {
          // Randomly pick one hidden node to keep active
          const idx = Math.floor(this._rand() * hiddenNodes.length);
          hiddenNodes[idx].mask = 1;
        }
      } else {
        for (const node of hiddenNodes) node.mask = 1;
      }
      // Optional weight noise (apply before node activations to all connection weights, store originals)
      if (training && this._weightNoiseStd > 0) {
        if (!this._wnOrig) this._wnOrig = new Array(this.connections.length);
        for (let ci = 0; ci < this.connections.length; ci++) {
          const c = this.connections[ci];
          if ((c as any)._origWeightNoise != null) continue; // already perturbed in recursive call
          (c as any)._origWeightNoise = c.weight;
          const noise =
            this._weightNoiseStd * Network._gaussianRand(this._rand);
          c.weight += noise;
        }
      }
      let outIndex = 0;
      this.nodes.forEach((node, index) => {
        if (node.type === 'input') {
          node.activate(input[index]);
        } else if (node.type === 'output') {
          const activation = node.activate();
          (output as any)[outIndex++] = activation;
        } else {
          node.activate();
        }
      });
      // Apply DropConnect masking to connections post-activation accumulation
      if (training && this._dropConnectProb > 0) {
        for (const conn of this.connections) {
          const mask = this._rand() < this._dropConnectProb ? 0 : 1;
          if (mask === 0) stats.droppedConnections++;
          (conn as any).dcMask = mask;
          if (mask === 0) {
            if ((conn as any)._origWeight == null)
              (conn as any)._origWeight = conn.weight;
            conn.weight = 0;
          } else if ((conn as any)._origWeight != null) {
            conn.weight = (conn as any)._origWeight;
            delete (conn as any)._origWeight;
          }
        }
      } else {
        // restore any temporarily zeroed weights
        for (const conn of this.connections) {
          if ((conn as any)._origWeight != null) {
            conn.weight = (conn as any)._origWeight;
            delete (conn as any)._origWeight;
          }
          (conn as any).dcMask = 1;
        }
      }
      // Restore weight noise
      if (training && appliedWeightNoise) {
        for (const c of this.connections) {
          if ((c as any)._origWeightNoise != null) {
            c.weight = (c as any)._origWeightNoise;
            delete (c as any)._origWeightNoise;
          }
        }
      }
    }
    if (training) this._trainingStep++;
    if (stats.weightNoise.count > 0)
      stats.weightNoise.meanAbs =
        stats.weightNoise.sumAbs / stats.weightNoise.count;
    this._lastStats = stats;
    // Clone and release pooled array for backward compatibility
    const result = Array.from(output as any) as number[];
    activationArrayPool.release(output);
    return result;
  }

  private static _gaussianRand(rng: () => number = Math.random): number {
    let u = 0,
      v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  /**
   * Activates the network without calculating eligibility traces.
   * This is a performance optimization for scenarios where backpropagation is not needed,
   * such as during testing, evaluation, or deployment (inference).
   *
   * @param {number[]} input - An array of numerical values corresponding to the network's input nodes.
   *                           The length must match the network's `input` size.
   * @returns {number[]} An array of numerical values representing the activations of the network's output nodes.
   *
   * @see {@link Node.noTraceActivate}
   */
  // Delegated activation helpers
  noTraceActivate(input: number[]): number[] {
    const { noTraceActivate } = require('./network/network.activate');
    return noTraceActivate.call(this, input);
  }

  /**
   * Raw activation that can return a typed array when pooling is enabled (zero-copy).
   * If reuseActivationArrays=false falls back to standard activate().
   */
  activateRaw(
    input: number[],
    training = false,
    maxActivationDepth = 1000
  ): any {
    const { activateRaw } = require('./network/network.activate');
    return activateRaw.call(this, input, training, maxActivationDepth);
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
    const { activateBatch } = require('./network/network.activate');
    return activateBatch.call(this, inputs, training);
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
    costDerivative?: (target: number, output: number) => number
  ): void {
    // Validate that the target array matches the network's output size.
    if (!target || target.length !== this.output) {
      throw new Error(
        'Output target length should match network output length'
      );
    }

    let targetIndex = target.length; // Initialize index for accessing target values in reverse order.

    // Propagate error starting from the output nodes (last nodes in the `nodes` array).
    // Iterate backward from the last node to the first output node.
    for (
      let i = this.nodes.length - 1;
      i >= this.nodes.length - this.output;
      i--
    ) {
      if (costDerivative) {
        (this.nodes[i] as any).propagate(
          rate,
          momentum,
          update,
          regularization,
          target[--targetIndex],
          costDerivative
        );
      } else {
        this.nodes[i].propagate(
          rate,
          momentum,
          update,
          regularization,
          target[--targetIndex]
        );
      }
    }

    // Propagate error backward through the hidden nodes.
    // Iterate backward from the last hidden node to the first hidden node.
    for (let i = this.nodes.length - this.output - 1; i >= this.input; i--) {
      this.nodes[i].propagate(rate, momentum, update, regularization); // Pass regularization factor
    }
  }

  /**
   * Clears the internal state of all nodes in the network.
   * Resets node activation, state, eligibility traces, and extended traces to their initial values (usually 0).
   * This is typically done before processing a new input sequence in recurrent networks or between training epochs if desired.
   *
   * @see {@link Node.clear}
   */
  clear(): void {
    // Iterate through all nodes and call their clear method.
    this.nodes.forEach((node) => node.clear());
  }

  /**
   * Mutates the network's structure or parameters according to the specified method.
   * This is a core operation for neuro-evolutionary algorithms (like NEAT).
   * The method argument should be one of the mutation types defined in `methods.mutation`.
   *
   * @param {any} method - The mutation method to apply (e.g., `mutation.ADD_NODE`, `mutation.MOD_WEIGHT`).
   *                       Some methods might have associated parameters (e.g., `MOD_WEIGHT` uses `min`, `max`).
   * @throws {Error} If no valid mutation `method` is provided.
   *
   * @see {@link methods.mutation} for available mutation types.
   */
  mutate(method: any): void {
    const { mutateImpl } = require('./network/network.mutate');
    return mutateImpl.call(this, method);
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
   * 2. Removing any self-connections.
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

  // slab rebuild + accessor moved to network.slab.ts

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
   * Trains the network on a given dataset subset for one pass (epoch or batch).
   * Performs activation and backpropagation for each item in the set.
   * Updates weights based on batch size configuration.
   *
   * @param {{ input: number[]; output: number[] }[]} set - The training dataset subset (e.g., a batch or the full set for one epoch).
   * @param {number} batchSize - The number of samples to process before updating weights.
   * @param {number} currentRate - The learning rate to use for this training pass.
   * @param {number} momentum - The momentum factor to use.
   * @param {any} regularization - The regularization configuration (L1, L2, or custom function).
   * @param {(target: number[], output: number[]) => number} costFunction - The function used to calculate the error between target and output.
   * @returns {number} The average error calculated over the provided dataset subset.
   * @private Internal method used by `train`.
   */
  // Removed legacy _trainSet; delegated to network.training.ts

  // Gradient clipping implemented in network.training.ts (applyGradientClippingImpl). Kept here only for backward compat if reflection used.
  private _applyGradientClipping(cfg: {
    mode: 'norm' | 'percentile' | 'layerwiseNorm' | 'layerwisePercentile';
    maxNorm?: number;
    percentile?: number;
  }) {
    const { applyGradientClippingImpl } = require('./network/network.training');
    applyGradientClippingImpl(this as any, cfg);
  }

  // Training is implemented in network.training.ts; this wrapper keeps public API stable.
  train(
    set: { input: number[]; output: number[] }[],
    options: any
  ): { error: number; iterations: number; time: number } {
    const { trainImpl } = require('./network/network.training');
    return trainImpl(this as any, set, options);
  }

  /** Returns last recorded raw (pre-update) gradient L2 norm. */
  getRawGradientNorm(): number {
    return this._lastRawGradNorm;
  }
  /** Returns current mixed precision loss scale (1 if disabled). */
  getLossScale(): number {
    return this._mixedPrecision.lossScale;
  }
  /** Returns last gradient clipping group count (0 if no clipping yet). */
  getLastGradClipGroupCount(): number {
    return this._lastGradClipGroupCount;
  }
  /** Consolidated training stats snapshot. */
  getTrainingStats() {
    return {
      gradNorm: this._lastGradNorm ?? 0,
      gradNormRaw: this._lastRawGradNorm,
      lossScale: this._mixedPrecision.lossScale,
      optimizerStep: this._optimizerStep,
      mp: {
        good: this._mixedPrecisionState.goodSteps,
        bad: this._mixedPrecisionState.badSteps,
        overflowCount: this._mixedPrecisionState.overflowCount || 0,
        scaleUps: this._mixedPrecisionState.scaleUpEvents || 0,
        scaleDowns: this._mixedPrecisionState.scaleDownEvents || 0,
        lastOverflowStep: this._lastOverflowStep,
      },
    };
  }
  /** Utility: adjust rate for accumulation mode (use result when switching to 'sum' to mimic 'average'). */
  static adjustRateForAccumulation(
    rate: number,
    accumulationSteps: number,
    reduction: 'average' | 'sum'
  ) {
    if (reduction === 'sum' && accumulationSteps > 1)
      return rate / accumulationSteps;
    return rate;
  }

  // Evolution wrapper delegates to network/network.evolve.ts implementation.
  async evolve(
    set: { input: number[]; output: number[] }[],
    options: any
  ): Promise<{ error: number; iterations: number; time: number }> {
    const { evolveNetwork } = await import('./network/network.evolve');
    return evolveNetwork.call(this, set, options);
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
    cost?: any
  ): { error: number; time: number } {
    // Dataset dimension validation
    if (!Array.isArray(set) || set.length === 0) {
      throw new Error('Test set is empty or not an array.');
    }
    for (const sample of set) {
      if (!Array.isArray(sample.input) || sample.input.length !== this.input) {
        throw new Error(
          `Test sample input size mismatch: expected ${this.input}, got ${
            sample.input ? sample.input.length : 'undefined'
          }`
        );
      }
      if (
        !Array.isArray(sample.output) ||
        sample.output.length !== this.output
      ) {
        throw new Error(
          `Test sample output size mismatch: expected ${this.output}, got ${
            sample.output ? sample.output.length : 'undefined'
          }`
        );
      }
    }

    let error = 0; // Accumulator for the total error.
    const costFn = cost || methods.Cost.mse; // Use provided cost function or default to MSE.
    const start = Date.now(); // Start time measurement.

    // --- Dropout/inference transition: Explicitly reset all hidden node masks to 1 for robust inference ---
    this.nodes.forEach((node) => {
      if (node.type === 'hidden') node.mask = 1;
    });

    const previousDropout = this.dropout; // Store current dropout rate
    if (this.dropout > 0) {
      // Temporarily disable dropout effect for testing.
      this.dropout = 0;
    }

    // Iterate through each sample in the test set.
    set.forEach((data) => {
      // Activate the network without calculating traces.
      const output = this.noTraceActivate(data.input);
      // Calculate the error for this sample and add it to the sum.
      error += costFn(data.output, output);
    });

    // Restore the previous dropout rate if it was changed.
    this.dropout = previousDropout;

    // Return the average error and the time taken.
    return { error: error / set.length, time: Date.now() - start };
  }

  /** Lightweight tuple serializer delegating to network.serialize.ts */
  serialize(): any[] {
    return _serialize.call(this);
  }

  /**
   * Creates a Network instance from serialized data produced by `serialize()`.
   * Reconstructs the network structure and state based on the provided arrays.
   *
   * @param {any[]} data - The serialized network data array, typically obtained from `network.serialize()`.
   *                       Expected format: `[activations, states, squashNames, connectionData, inputSize, outputSize]`.
   * @param {number} [inputSize] - Optional input size override.
   * @param {number} [outputSize] - Optional output size override.
   * @returns {Network} A new Network instance reconstructed from the serialized data.
   * @static
   */
  /** Static lightweight tuple deserializer delegate */
  static deserialize(
    data: any[],
    inputSize?: number,
    outputSize?: number
  ): Network {
    return _deserialize(data, inputSize, outputSize);
  }

  /**
   * Converts the network into a JSON object representation (latest standard).
   * Includes formatVersion, and only serializes properties needed for full reconstruction.
   * All references are by index. Excludes runtime-only properties (activation, state, traces).
   *
   * @returns {object} A JSON-compatible object representing the network.
   */
  /** Verbose JSON serializer delegate */
  toJSON(): object {
    return _toJSONImpl.call(this);
  }

  /**
   * Reconstructs a network from a JSON object (latest standard).
   * Handles formatVersion, robust error handling, and index-based references.
   * @param {object} json - The JSON object representing the network.
   * @returns {Network} The reconstructed network.
   */
  /** Verbose JSON static deserializer */
  static fromJSON(json: any): Network {
    return _fromJSONImpl(json);
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
    equal: boolean = false
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
  set(values: { bias?: number; squash?: any }): void {
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
    outputCount: number
  ): Network {
    // Create all nodes
    const inputNodes = Array.from(
      { length: inputCount },
      () => new Node('input')
    );
    const hiddenLayers: Node[][] = hiddenCounts.map((count) =>
      Array.from({ length: count }, () => new Node('hidden'))
    );
    const outputNodes = Array.from(
      { length: outputCount },
      () => new Node('output')
    );
    // Flatten all nodes in topological order
    const allNodes = [...inputNodes, ...hiddenLayers.flat(), ...outputNodes];
    // Create network instance
    const net = new Network(inputCount, outputCount);
    net.nodes = allNodes;
    // Connect layers
    let prevLayer = inputNodes;
    for (const layer of hiddenLayers) {
      for (const to of layer) {
        for (const from of prevLayer) {
          from.connect(to);
        }
      }
      prevLayer = layer;
    }
    // Connect last hidden (or input if no hidden) to output
    for (const to of outputNodes) {
      for (const from of prevLayer) {
        from.connect(to);
      }
    }
    // Rebuild net.connections from all per-node connections
    net.connections = net.nodes.flatMap((n) => n.connections.out);
    net._topoDirty = true;
    return net;
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
    const allConnections = new Set<Connection>();
    net.nodes.forEach((node) => {
      node.connections.out.forEach((conn) => {
        allConnections.add(conn);
      });
    });
    net.connections = Array.from(allConnections) as Connection[];
  }
}
