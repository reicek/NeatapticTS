import Node from './node';
import Connection from './connection';
import Multi from '../multithreading/multi';
import * as methods from '../methods/methods';
import mutation from '../methods/mutation'; // Import mutation methods
import { config } from '../config'; // Import configuration settings
import { exportToONNX } from './onnx'; // Import ONNX export function

/**
 * Helper function to remove Istanbul.js code coverage instrumentation artifacts.
 * This is useful for generating clean, production-ready code (e.g., for `standalone`).
 * It removes comments and counter increments inserted by the coverage tool.
 *
 * @param {string} code - The input JavaScript code string potentially containing coverage artifacts.
 * @returns {string} The cleaned JavaScript code string.
 * @private
 */
const stripCoverage = (code: string): string => {
  // 1. Remove Istanbul ignore comments (e.g., /* istanbul ignore next */)
  code = code.replace(/\/\*\s*istanbul\s+ignore\s+[\s\S]*?\*\//g, '');

  // 2. Remove coverage counter increments (e.g., cov_1a2b3c4d().s[0]++)
  // Matches patterns like cov_...().s[...][...]++ or cov_...().f[...]++ etc.
  code = code.replace(/cov_[\w$]+\(\)\.(s|f|b)\[\d+\](\[\d+\])?\+\+/g, ''); // Removed trailing ';' requirement

  // 3. Remove simple coverage function calls (e.g., cov_1a2b3c4d())
  code = code.replace(/cov_[\w$]+\(\)/g, ''); // Removed trailing ';' requirement

  // 4. Remove sourceMappingURL comments (often added during transpilation)
  code = code.replace(/^\s*\/\/# sourceMappingURL=.*\s*$/gm, '');

  // 5. Cleanup potential leftover artifacts from comma operator usage within coverage code
  // Example: transforms `( , false)` to `( false)` or `( true , )` to `( true )`
  code = code.replace(/\(\s*,\s*/g, '( '); // Remove leading comma after opening parenthesis
  code = code.replace(/\s*,\s*\)/g, ' )'); // Remove trailing comma before closing parenthesis

  // 6. Trim leading/trailing whitespace from the entire code string
  code = code.trim();

  // 7. Remove empty statements (lines containing only a semicolon) potentially left behind
  code = code.replace(/^\s*;\s*$/gm, ''); // Remove lines with only ';'
  code = code.replace(/;{2,}/g, ';'); // Replace multiple consecutive semicolons with a single one
  // Remove lines that might now only contain commas or whitespace after other replacements
  code = code.replace(/^\s*[,;]?\s*$/gm, '');

  return code;
};
export default class Network {
  input: number; output: number; score?: number; nodes: Node[]; connections: Connection[]; gates: Connection[]; selfconns: Connection[]; dropout: number = 0;
  private _dropConnectProb: number = 0; private _lastGradNorm?: number; private _optimizerStep: number = 0; private _weightNoiseStd: number = 0; private _weightNoisePerHidden: number[] = []; private _weightNoiseSchedule?: (step:number)=>number; private _stochasticDepth: number[] = []; private _wnOrig?: number[]; private _trainingStep: number = 0; private _rand: () => number = Math.random; private _rngState?: number; private _lastStats: any = null; private _stochasticDepthSchedule?: (step:number,current:number[])=>number[]; private _mixedPrecision: { enabled:boolean; lossScale:number } = { enabled:false, lossScale:1 }; private _mixedPrecisionState: { goodSteps:number; badSteps:number; minLossScale:number; maxLossScale:number; overflowCount?:number; scaleUpEvents?:number; scaleDownEvents?:number } = { goodSteps:0, badSteps:0, minLossScale:1, maxLossScale:65536, overflowCount:0, scaleUpEvents:0, scaleDownEvents:0 }; private _gradAccumMicroBatches: number = 0; private _currentGradClip?: { mode:'norm'|'percentile'|'layerwiseNorm'|'layerwisePercentile'; maxNorm?:number; percentile?:number }; private _lastRawGradNorm: number = 0; private _accumulationReduction: 'average'|'sum' = 'average'; private _gradClipSeparateBias: boolean = false; private _lastGradClipGroupCount: number = 0; private _lastOverflowStep: number = -1; private _forceNextOverflow: boolean = false; private _pruningConfig?: { start:number; end:number; targetSparsity:number; regrowFraction:number; frequency:number; method:'magnitude'|'snip'; lastPruneIter?:number }; private _initialConnectionCount?: number; private _enforceAcyclic: boolean = false; private _topoOrder: Node[] | null = null; private _topoDirty: boolean = true; private _globalEpoch: number = 0; layers?: any[];
  private _evoInitialConnCount?: number; // baseline for evolution-time pruning
  constructor(input: number, output: number, options?: { minHidden?: number; seed?: number; enforceAcyclic?: boolean }) {
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
  }

  // --- DropConnect API (re-added for tests) ---
  enableDropConnect(p: number) {
    if (p < 0 || p >= 1) throw new Error('DropConnect probability must be in [0,1)');
    this._dropConnectProb = p;
  }
  disableDropConnect() { this._dropConnectProb = 0; }

  // --- Acyclic enforcement toggle (used by tests) ---
  setEnforceAcyclic(flag: boolean) { this._enforceAcyclic = !!flag; }
  /** Recompute cached topological execution order (acyclic only). */
  private _computeTopoOrder(): void {
    if (!this._enforceAcyclic) { this._topoOrder = null; this._topoDirty = false; return; }
    // Kahn's algorithm over current connections (ignore self connections & disabled if added later)
    const inDegree: Map<Node, number> = new Map();
    this.nodes.forEach(n => inDegree.set(n, 0));
    for (const c of this.connections) { if (c.from !== c.to) inDegree.set(c.to, (inDegree.get(c.to) || 0) + 1); }
    const queue: Node[] = [];
    // Bias toward inputs first
    this.nodes.forEach(n => { if ((n as any).type === 'input' || (inDegree.get(n) || 0) === 0) queue.push(n); });
    const order: Node[] = [];
    while (queue.length) {
      const n = queue.shift()!;
      order.push(n);
      for (const out of n.connections.out) {
        if (out.to === n) continue; // self
        const deg = (inDegree.get(out.to) || 0) - 1;
        inDegree.set(out.to, deg);
        if (deg === 0) queue.push(out.to);
      }
    }
    if (order.length !== this.nodes.length) {
      // Graph has cycle (should not in acyclic mode) – fallback to existing order
      this._topoOrder = this.nodes.slice();
    } else {
      this._topoOrder = order;
    }
    this._topoDirty = false;
  }
  /** Internal reachability check used to avoid introducing cycles when acyclic enforced. */
  private _hasPath(from: Node, to: Node): boolean {
    if (from === to) return true;
    const seen = new Set<Node>();
    const stack: Node[] = [from];
    while (stack.length) {
      const cur = stack.pop()!;
      if (cur === to) return true;
      if (seen.has(cur)) continue;
      seen.add(cur);
      for (const c of cur.connections.out) { if (c.to !== cur) stack.push(c.to); }
    }
    return false;
  }

  // --- Pruning configuration & helpers ---
  configurePruning(cfg: { start:number; end:number; targetSparsity:number; regrowFraction?:number; frequency?:number; method?:'magnitude'|'snip' }) {
    const { start, end, targetSparsity } = cfg;
    if (start < 0 || end < start) throw new Error('Invalid pruning schedule window');
    if (targetSparsity <= 0 || targetSparsity >= 1) throw new Error('targetSparsity must be in (0,1)');
    this._pruningConfig = {
      start,
      end,
      targetSparsity,
      regrowFraction: cfg.regrowFraction ?? 0,
      frequency: cfg.frequency ?? 1,
      method: cfg.method || 'magnitude',
      lastPruneIter: undefined
    };
    this._initialConnectionCount = this.connections.length;
  }
  getCurrentSparsity(): number {
    if (!this._initialConnectionCount) return 0;
    return 1 - (this.connections.length / this._initialConnectionCount);
  }
  private _maybePrune(iteration: number) {
    const cfg = this._pruningConfig;
    if (!cfg) return;
    if (iteration < cfg.start || iteration > cfg.end) return;
    if (cfg.lastPruneIter != null && iteration === cfg.lastPruneIter) return;
    if ((iteration - cfg.start) % (cfg.frequency || 1) !== 0) return;
    if (!this._initialConnectionCount || this._initialConnectionCount === 0) return;
    const progress = (iteration - cfg.start) / Math.max(1, (cfg.end - cfg.start));
    const targetSparsityNow = cfg.targetSparsity * Math.min(1, Math.max(0, progress));
    const desiredRemaining = Math.max(1, Math.floor(this._initialConnectionCount * (1 - targetSparsityNow)));
    const excess = this.connections.length - desiredRemaining;
    if (excess <= 0) { cfg.lastPruneIter = iteration; return; }
    // Determine pruning ranking: magnitude or SNIP (saliency |w * grad|)
    let sorted = [...this.connections];
    if (cfg.method === 'snip') {
      sorted.sort((a: any, b: any) => {
        const ga = Math.abs(a.totalDeltaWeight) || Math.abs(a.previousDeltaWeight) || 0;
        const gb = Math.abs(b.totalDeltaWeight) || Math.abs(b.previousDeltaWeight) || 0;
        const sa = ga ? Math.abs(a.weight) * ga : Math.abs(a.weight);
        const sb = gb ? Math.abs(b.weight) * gb : Math.abs(b.weight);
        return sa - sb; // ascending saliency
      });
    } else {
      sorted.sort((a,b)=>Math.abs(a.weight)-Math.abs(b.weight));
    }
    const toPrune = sorted.slice(0, excess);
    toPrune.forEach(c => this.disconnect(c.from, c.to));
    // Optional regrow
    if (cfg.regrowFraction && cfg.regrowFraction > 0) {
      const regrowTarget = Math.floor(toPrune.length * cfg.regrowFraction);
      let attempts = 0;
      while (this.connections.length < desiredRemaining && attempts < regrowTarget * 10) {
        attempts++;
        const from = this.nodes[Math.floor(this._rand()*this.nodes.length)];
        const to = this.nodes[Math.floor(this._rand()*this.nodes.length)];
        if (!from || !to) continue;
        if (from === to) continue;
        // Prevent duplicate
        if (this.connections.some(c => c.from === from && c.to === to)) continue;
        // Respect acyclic constraint
        if (this._enforceAcyclic && this.nodes.indexOf(from) > this.nodes.indexOf(to)) continue;
        this.connect(from, to);
        if (this.connections.length >= desiredRemaining) break;
      }
    }
    cfg.lastPruneIter = iteration;
  }

  /**
   * Immediately prune connections to reach (or approach) a target sparsity fraction.
   * Used by evolutionary pruning (generation-based) independent of training iteration schedule.
   * @param targetSparsity fraction in (0,1). 0.8 means keep 20% of original (if first call sets baseline)
   * @param method 'magnitude' | 'snip'
   */
  pruneToSparsity(targetSparsity: number, method: 'magnitude'|'snip'='magnitude') {
    if (targetSparsity <= 0) return; // nothing to do
    if (targetSparsity >= 1) targetSparsity = 0.999; // safety
    if (!this._evoInitialConnCount) this._evoInitialConnCount = this.connections.length;
    const desiredRemaining = Math.max(1, Math.floor(this._evoInitialConnCount * (1 - targetSparsity)));
    const excess = this.connections.length - desiredRemaining;
    if (excess <= 0) return;
    let ranked = [...this.connections];
    if (method === 'snip') {
      ranked.sort((a: any, b: any) => {
        const ga = Math.abs(a.totalDeltaWeight) || Math.abs(a.previousDeltaWeight) || 0;
        const gb = Math.abs(b.totalDeltaWeight) || Math.abs(b.previousDeltaWeight) || 0;
        const sa = ga ? Math.abs(a.weight) * ga : Math.abs(a.weight);
        const sb = gb ? Math.abs(b.weight) * gb : Math.abs(b.weight);
        return sa - sb;
      });
    } else {
      ranked.sort((a,b)=> Math.abs(a.weight) - Math.abs(b.weight));
    }
    const toRemove = ranked.slice(0, excess);
    toRemove.forEach(c => this.disconnect(c.from, c.to));
    this._topoDirty = true;
  }

  /** Enable weight noise. Provide a single std dev number or { perHiddenLayer: number[] }. */
  enableWeightNoise(stdDev: number | { perHiddenLayer: number[] }) {
    if (typeof stdDev === 'number') {
      if (stdDev < 0) throw new Error('Weight noise stdDev must be >= 0');
      this._weightNoiseStd = stdDev;
      this._weightNoisePerHidden = [];
    } else if (stdDev && Array.isArray(stdDev.perHiddenLayer)) {
      if (!this.layers || this.layers.length < 3) throw new Error('Per-hidden-layer weight noise requires a layered network with at least one hidden layer');
      const hiddenLayerCount = this.layers.length - 2;
      if (stdDev.perHiddenLayer.length !== hiddenLayerCount) throw new Error(`Expected ${hiddenLayerCount} std dev entries (one per hidden layer), got ${stdDev.perHiddenLayer.length}`);
      if (stdDev.perHiddenLayer.some(s => s < 0)) throw new Error('Weight noise std devs must be >= 0');
      this._weightNoiseStd = 0; // disable global
      this._weightNoisePerHidden = stdDev.perHiddenLayer.slice();
    } else {
      throw new Error('Invalid weight noise configuration');
    }
  }
  disableWeightNoise() { this._weightNoiseStd = 0; this._weightNoisePerHidden = []; }
  setWeightNoiseSchedule(fn: (step: number) => number) { this._weightNoiseSchedule = fn; }
  clearWeightNoiseSchedule() { this._weightNoiseSchedule = undefined; }
  setRandom(fn: () => number) { this._rand = fn; }
  setSeed(seed: number) {
    this._rngState = seed >>> 0;
    this._rand = () => {
      this._rngState = (this._rngState! + 0x6D2B79F5) >>> 0;
      let r = Math.imul(this._rngState! ^ (this._rngState! >>> 15), 1 | this._rngState!);
      r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
  }
  testForceOverflow() { this._forceNextOverflow = true; }
  get trainingStep() { return this._trainingStep; }
  get lastSkippedLayers(): number[] { return (this as any)._lastSkippedLayers || []; }
  snapshotRNG(): any { return { step: this._trainingStep, state: this._rngState }; }
  restoreRNG(fn: () => number) { this._rand = fn; this._rngState = undefined; }
  getRNGState(): number | undefined { return this._rngState; }
  setRNGState(state: number) { if (typeof state === 'number') this._rngState = state >>> 0; }
  setStochasticDepthSchedule(fn: (step: number, current: number[]) => number[]) { this._stochasticDepthSchedule = fn; }
  clearStochasticDepthSchedule() { this._stochasticDepthSchedule = undefined; }
  getRegularizationStats() { return this._lastStats ? JSON.parse(JSON.stringify(this._lastStats)) : null; }

    /** Configure stochastic depth with survival probabilities per hidden layer (length must match hidden layer count when using layered network). */
    setStochasticDepth(survival: number[]) {
      if (!Array.isArray(survival)) throw new Error('survival must be an array');
      if (survival.some(p => p <= 0 || p > 1)) throw new Error('Stochastic depth survival probs must be in (0,1]');
      if (!this.layers || this.layers.length === 0) throw new Error('Stochastic depth requires layer-based network');
      // layers includes input and output; hidden layers are layers[1..length-2]
      const hiddenLayerCount = Math.max(0, this.layers.length - 2);
      if (survival.length !== hiddenLayerCount) throw new Error(`Expected ${hiddenLayerCount} survival probabilities for hidden layers, got ${survival.length}`);
      this._stochasticDepth = survival.slice();
    }
    disableStochasticDepth() { this._stochasticDepth = []; }

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

  /**
   * Generates a standalone JavaScript function that replicates the network's behavior.
   * This function can be executed independently of the NeatapticTS library.
   * It uses `Float64Array` for storing activations and states, potentially offering performance benefits.
   * The generated function includes definitions for the necessary activation (squash) functions.
   *
   * @returns {string} A string containing the source code of the standalone JavaScript function.
   *                   This function takes an `input` array and returns an `output` array.
   */
  standalone(): string {
    // Throw if there are no output nodes
    if (!this.nodes.some(n => n.type === 'output')) {
      throw new Error('Cannot create standalone function: network has no output nodes.');
    }

    const present: { [key: string]: string } = {}; // Tracks activation functions already defined (name -> definition string).
    const squashDefinitions: string[] = []; // Stores the string definitions of the activation functions used.
    const functionMap: { [key: string]: number } = {}; // Maps activation function name to an index in the `F` array.
    let functionIndexCounter = 0; // Counter for assigning indices to functions.

    const activations: number[] = []; // Stores initial activation values for all nodes.
    const states: number[] = []; // Stores initial state values for all nodes.
    const lines: string[] = []; // Stores the lines of code for the core activation logic within the standalone function.

    // Predefined activation functions. These are included directly for common cases.
    // Ensure these names match the keys in `methods.Activation`.
    // Definitions are minified/optimized where possible.
    const predefined: { [key: string]: string } = {
      logistic: 'function logistic(x){ return 1 / (1 + Math.exp(-x)); }',
      tanh: 'function tanh(x){ return Math.tanh(x); }',
      relu: 'function relu(x){ return x > 0 ? x : 0; }',
      identity: 'function identity(x){ return x; }',
      step: 'function step(x){ return x > 0 ? 1 : 0; }',
      softsign: 'function softsign(x){ return x / (1 + Math.abs(x)); }',
      sinusoid: 'function sinusoid(x){ return Math.sin(x); }',
      gaussian: 'function gaussian(x){ return Math.exp(-Math.pow(x, 2)); }',
      bentIdentity:
        'function bentIdentity(x){ return (Math.sqrt(Math.pow(x, 2) + 1) - 1) / 2 + x; }',
      bipolar: 'function bipolar(x){ return x > 0 ? 1 : -1; }',
      bipolarSigmoid:
        'function bipolarSigmoid(x){ return 2 / (1 + Math.exp(-x)) - 1; }',
      hardTanh: 'function hardTanh(x){ return Math.max(-1, Math.min(1, x)); }',
      absolute: 'function absolute(x){ return Math.abs(x); }',
      inverse: 'function inverse(x){ return 1 - x; }',
      selu:
        'function selu(x){ var a=1.6732632423543772,s=1.0507009873554805; var fx=x>0?x:a*Math.exp(x)-a; return fx*s; }', // Minified SELU
      softplus:
        'function softplus(x){ if(x>30)return x; if(x<-30)return Math.exp(x); return Math.max(0,x)+Math.log(1+Math.exp(-Math.abs(x))); }', // Numerically stable softplus
      swish: 'function swish(x){ var s=1/(1+Math.exp(-x)); return x*s; }', // Swish (Sigmoid Linear Unit)
      gelu:
        'function gelu(x){ var cdf=0.5*(1.0+Math.tanh(Math.sqrt(2.0/Math.PI)*(x+0.044715*Math.pow(x,3)))); return x*cdf; }', // Approximate GELU (Gaussian Error Linear Unit)
      mish:
        'function mish(x){ var sp_x; if(x>30){sp_x=x;}else if(x<-30){sp_x=Math.exp(x);}else{sp_x=Math.log(1+Math.exp(x));} var tanh_sp_x=Math.tanh(sp_x); return x*tanh_sp_x; }', // Mish (Self Regularized Non-Monotonic) with stable softplus
    };

    // Assign an index to each node and initialize activation/state arrays.
    // This ensures consistent referencing within the generated code.
    this.nodes.forEach((node, index) => {
      node.index = index; // Assign index used in the standalone code.
      activations.push(node.activation); // Store current activation (initial value).
      states.push(node.state); // Store current state (initial value).
    });

    // Code to copy input values to the activation array `A`.
    lines.push('for(var i = 0; i < input.length; i++) A[i] = input[i];');

    // Generate activation logic for hidden and output nodes.
    for (let i = this.input; i < this.nodes.length; i++) {
      const node = this.nodes[i];
      const squash = node.squash as any; // The activation function for this node.
      // Determine the name of the squash function. Use its `name` property or generate one.
      const squashName = squash.name || `anonymous_squash_${i}`;

      // Check if this squash function's definition is already included.
      if (!(squashName in present)) {
        let funcStr: string;
        // Use predefined definition if available.
        if (predefined[squashName]) {
          funcStr = predefined[squashName];
          // Ensure the function name in the definition string matches the key.
          if (!funcStr.startsWith(`function ${squashName}`)) {
            funcStr = `function ${squashName}${funcStr.substring(
              funcStr.indexOf('(')
            )}`;
          }
          funcStr = stripCoverage(funcStr); // Remove potential coverage artifacts.
        } else {
          // Handle custom or unknown functions by converting them to string.
          funcStr = squash.toString();
          funcStr = stripCoverage(funcStr); // Remove coverage artifacts.

          // Attempt to format the function string consistently.
          if (funcStr.startsWith('function')) {
            // Ensure name matches if it was anonymous or different.
            funcStr = `function ${squashName}${funcStr.substring(
              funcStr.indexOf('(')
            )}`;
          } else if (funcStr.includes('=>')) {
            // Basic conversion for arrow functions.
            funcStr = `function ${squashName}${funcStr.substring(
              funcStr.indexOf('(')
            )}`;
          } else {
            // Fallback for unusual function definitions.
            console.warn(
              `Standalone: Could not reliably get definition for squash function '${squashName}'. Using placeholder.`
            );
            funcStr = `function ${squashName}(x){ /* unknown definition */ return x; }`; // Provide a safe placeholder.
          }
        }
        present[squashName] = funcStr; // Mark as present.
        squashDefinitions.push(present[squashName]); // Add definition to the list.
        functionMap[squashName] = functionIndexCounter++; // Assign an index.
      }
      const functionIndex = functionMap[squashName]; // Get the index for this function.

      // Build the calculation for the node's state (sum of weighted inputs + bias).
      const incoming: string[] = [];
      // Process incoming connections.
      for (const conn of node.connections.in) {
        // Ensure the source node has a valid index.
        if (typeof conn.from.index === 'undefined') continue;
        // Base computation: activation[from] * weight
        let computation = `A[${conn.from.index}] * ${conn.weight}`;
        // Apply gating if the connection is gated and the gater node has a valid index.
        if (conn.gater && typeof conn.gater.index !== 'undefined') {
          computation += ` * A[${conn.gater.index}]`; // Multiply by gater activation.
        }
        incoming.push(computation);
      }

      // Process self-connection.
      // Check if a self-connection exists in the array.
      if (node.connections.self.length > 0) {
        const selfConn = node.connections.self[0]; // Access the first (and likely only) self-connection.
        // Base computation: state[self] * weight
        let computation = `S[${i}] * ${selfConn.weight}`;
        // Apply gating if the self-connection is gated and the gater node has a valid index.
        if (
          selfConn.gater &&
          typeof selfConn.gater.index !== 'undefined'
        ) {
          computation += ` * A[${selfConn.gater.index}]`; // Multiply by gater activation.
        }
        incoming.push(computation);
      }

      // Combine incoming calculations. Default to '0' if no incoming connections.
      const incomingCalculation =
        incoming.length > 0 ? incoming.join(' + ') : '0';
      // Calculate the node's state (S[i] = sum + bias).
      lines.push(`S[${i}] = ${incomingCalculation} + ${node.bias};`);
      // Calculate the node's activation (A[i] = squash(S[i]) * mask).
      // Use node.mask if defined and not 1, otherwise default to 1 (no multiplication needed).
      const maskValue =
        typeof node.mask === 'number' && node.mask !== 1 ? node.mask : 1;
      // Apply squash function (referenced by index from F array) and mask.
      lines.push(
        `A[${i}] = F[${functionIndex}](S[${i}])${
          maskValue !== 1 ? ` * ${maskValue}` : '' // Apply mask only if it's not 1.
        };`
      );
    }

    // Identify the indices of the output nodes.
    const outputIndices: number[] = [];
    for (let i = this.nodes.length - this.output; i < this.nodes.length; i++) {
      // Ensure the output node and its index are valid.
      if (typeof this.nodes[i]?.index !== 'undefined') {
        outputIndices.push(this.nodes[i].index!); // Add valid index.
      } else {
        // Warn if an output node seems invalid (should not happen in normal operation).
        console.warn(`Standalone: Invalid output node index ${i}`);
      }
    }
    // Generate the return statement, collecting activations from output nodes.
    lines.push(
      `return [${outputIndices.map((idx) => `A[${idx}]`).join(',')}];`
    );

    // Assemble the final standalone function string.
    let total = '';
    total += `(function(){\n`; // Start IIFE (Immediately Invoked Function Expression) to encapsulate scope.
    // Define all required squash functions first.
    total += `${squashDefinitions.join('\n')}\n`;
    // Create the `F` array, mapping indices to the actual function objects (by name).
    const fArrayContent = Object.entries(functionMap)
      .sort(([, a], [, b]) => a - b) // Sort functions by their assigned index.
      .map(([name]) => name) // Get the function names in the correct order.
      .join(','); // Join names into a comma-separated list.
    total += `var F = [${fArrayContent}];\n`; // Define the F array.
    // Initialize activation (A) and state (S) arrays using Float64Array.
    total += `var A = new Float64Array([${activations.join(',')}]);\n`;
    total += `var S = new Float64Array([${states.join(',')}]);\n`;
    // Define the main `activate` function within the IIFE.
    total += `function activate(input){\n`;
    // Add a safety check for the input array length.
    total += `if (!input || input.length !== ${this.input}) { throw new Error('Invalid input size. Expected ${this.input}, got ' + (input ? input.length : 'undefined')); }\n`;
    // Insert the generated activation logic lines.
    total += `${lines.join('\n')}\n}\n`;
    // Return the `activate` function from the IIFE.
    total += `return activate;\n`;
    total += `})()`; // End and execute the IIFE.

    // Final stripCoverage call as a safeguard.
    return stripCoverage(total);
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
  activate(input: number[], training = false, maxActivationDepth = 1000): number[] {
  if (this._enforceAcyclic && this._topoDirty) this._computeTopoOrder();
    if (!Array.isArray(input) || input.length !== this.input) {
      throw new Error(
        `Input size mismatch: expected ${this.input}, got ${input ? input.length : 'undefined'}`
      );
    }
    
    // Check for empty or corrupted network structure
    if (!this.nodes || this.nodes.length === 0) {
      throw new Error('Network structure is corrupted or empty. No nodes found.');
    }
    
    const output: number[] = [];
    (this as any)._lastSkippedLayers = [];
      const stats = {
        droppedHiddenNodes: 0,
        totalHiddenNodes: 0,
        droppedConnections: 0,
        totalConnections: this.connections.length,
        skippedLayers: [] as number[],
        weightNoise: { count: 0, sumAbs: 0, maxAbs: 0, meanAbs: 0 }
      };
    // Pre-apply weight noise
    let appliedWeightNoise = false;
    let dynamicStd = this._weightNoiseStd;
    if (training) {
      if (this._weightNoiseSchedule) dynamicStd = this._weightNoiseSchedule(this._trainingStep);
      if (dynamicStd > 0 || this._weightNoisePerHidden.length > 0) {
        for (const c of this.connections) {
          if ((c as any)._origWeightNoise != null) continue;
          (c as any)._origWeightNoise = c.weight;
          let std = dynamicStd;
          if (this._weightNoisePerHidden.length > 0 && this.layers) {
            let fromLayerIndex = -1;
            for (let li = 0; li < this.layers.length; li++) {
              if (this.layers[li].nodes.includes(c.from)) { fromLayerIndex = li; break; }
            }
            if (fromLayerIndex > 0 && fromLayerIndex < this.layers.length) {
              const hiddenIdx = fromLayerIndex - 1;
              if (hiddenIdx >= 0 && hiddenIdx < this._weightNoisePerHidden.length) std = this._weightNoisePerHidden[hiddenIdx];
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
    if (training && this._stochasticDepthSchedule && this._stochasticDepth.length > 0) {
      const updated = this._stochasticDepthSchedule(this._trainingStep, this._stochasticDepth.slice());
      if (Array.isArray(updated) && updated.length === this._stochasticDepth.length && !updated.some(p => p <= 0 || p > 1)) {
        this._stochasticDepth = updated.slice();
      }
    }
    if (this.layers && this.layers.length > 0 && this._stochasticDepth.length > 0) {
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
              const raw = li === 0 ? layer.activate(input, training) : layer.activate(undefined, training);
              acts = surviveProb < 1 ? raw.map((a: number) => a * (1 / surviveProb)) : raw;
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
        const raw = li === 0 ? layer.activate(input, training) : layer.activate(undefined, training);
        acts = raw;
      }
      if (acts) output.push(...acts);
    } else if (this.layers && this.layers.length > 0) {
      // Layered activation with optional node-level dropout (replicating legacy behavior expected by tests)
      let lastActs: number[] | undefined;
      for (let li = 0; li < this.layers.length; li++) {
        const layer = this.layers[li];
        const isHidden = li > 0 && li < this.layers.length - 1;
        // Always call layer.activate with training=false to avoid its uniform layer-level dropout; we'll handle per-node masks ourselves
        const raw = li === 0 ? layer.activate(input, false) : layer.activate(undefined, false);
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
      if (lastActs) output.push(...lastActs);
    } else {
      // Node-based activation (legacy, node-level dropout)
      let hiddenNodes = this.nodes.filter(node => node.type === 'hidden');
      let droppedCount = 0;
      if (training && this.dropout > 0) {
        // Randomly drop hidden nodes
        for (const node of hiddenNodes) {
          node.mask = this._rand() < this.dropout ? 0 : 1;
          stats.totalHiddenNodes++;
          if (node.mask === 0) { droppedCount++; stats.droppedHiddenNodes++; }
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
          const noise = this._weightNoiseStd * Network._gaussianRand(this._rand);
          c.weight += noise;
        }
      }
      this.nodes.forEach((node, index) => {
        if (node.type === 'input') {
          node.activate(input[index]);
        } else if (node.type === 'output') {
          const activation = node.activate();
          output.push(activation);
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
            if ((conn as any)._origWeight == null) (conn as any)._origWeight = conn.weight;
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
  if (stats.weightNoise.count > 0) stats.weightNoise.meanAbs = stats.weightNoise.sumAbs / stats.weightNoise.count;
  this._lastStats = stats;
  return output;
  }


    private static _gaussianRand(rng: () => number = Math.random): number {
      let u = 0, v = 0;
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
  noTraceActivate(input: number[]): number[] {
  if (this._enforceAcyclic && this._topoDirty) this._computeTopoOrder();
    // Input size validation
    if (!Array.isArray(input) || input.length !== this.input) {
      throw new Error(
        `Input size mismatch: expected ${this.input}, got ${input ? input.length : 'undefined'}`
      );
    }

    const output: number[] = []; // Array to store the activations of output nodes.

    // Iterate through all nodes in the network.
    this.nodes.forEach((node, index) => {
      if (node.type === 'input') {
        // Activate input node without calculating traces.
        node.noTraceActivate(input[index]);
      } else if (node.type === 'output') {
        // Activate output node without calculating traces.
        const activation = node.noTraceActivate();
        output.push(activation); // Store the activation value.
      } else {
        // Activate hidden node without calculating traces.
        // Note: Dropout masking (`node.mask`) might still be active if set previously,
        // ensure `clear()` or manual reset if testing after dropout training.
        node.noTraceActivate();
      }
    });

    return output; // Return the collected output activations.
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
    // Validate that a mutation method was provided.
    if (!method) {
      throw new Error('No (correct) mutate method given!');
    }

    // Declare variables used within the switch statement.
    let connection, node, index, possible, available, pair, randomConn;

    // Apply the specified mutation method.
    switch (method) {
      case mutation.ADD_NODE:
  if (this._enforceAcyclic) this._topoDirty = true;
        // Adds a new hidden node by splitting an existing connection.
        // 1. Select a random existing connection.
        if (this.connections.length === 0) break; // Cannot add node if no connections exist
        connection = this.connections[
          Math.floor(this._rand() * this.connections.length)
        ];
        const gater = connection.gater; // Store original gater, if any.
        // 2. Disconnect the original connection.
        this.disconnect(connection.from, connection.to);

        // 3. Create a new hidden node. Optionally mutate its activation function.
  node = new Node('hidden', undefined, this._rand);
        node.mutate(mutation.MOD_ACTIVATION); // Randomize activation function.

        // 4. Insert the new node into the `nodes` array.
        // Try to insert topologically between the original `from` and `to` nodes,
        // but ensure it's placed before the output layer.
        const toIndex = this.nodes.indexOf(connection.to);
        const minBound = Math.min(toIndex, this.nodes.length - this.output);
        this.nodes.splice(minBound, 0, node);

        // 5. Create two new connections: from -> new_node, and new_node -> to.
        const newConn1 = this.connect(connection.from, node)[0]; // Weight might be 1 initially.
        const newConn2 = this.connect(node, connection.to)[0]; // Weight might inherit original weight.

        // 6. If the original connection was gated, re-apply the gate to one of the new connections randomly.
        if (gater) {
          this.gate(gater, this._rand() >= 0.5 ? newConn1 : newConn2);
        }
        break;

      case mutation.SUB_NODE:
        // Removes a hidden node from the network.
        // Filter for hidden nodes only.
        const hiddenNodes = this.nodes.filter(n => n.type === 'hidden');
        if (hiddenNodes.length === 0) {
          if (config.warnings) console.warn('No hidden nodes left to remove!');
          break;
        }
        // 1. Select a random hidden node.
  node = hiddenNodes[Math.floor(this._rand() * hiddenNodes.length)];
        // 2. Remove the selected node using the `remove` method, which handles reconnection.
        if (node.type === 'hidden') {
          this.remove(node);
        } else {
          // This should never happen, but guard just in case.
          if (config.warnings) console.warn('Attempted to remove a non-hidden node. Skipping.');
        }
        break;

      case mutation.ADD_CONN:
  if (this._enforceAcyclic) this._topoDirty = true;
        // Adds a new connection between two previously unconnected nodes.
        available = []; // Stores pairs of [fromNode, toNode] that can be connected.
        // Iterate through possible 'from' nodes (input and hidden).
        for (let i = 0; i < this.nodes.length - this.output; i++) {
          const node1 = this.nodes[i];
          // Iterate through possible 'to' nodes (hidden and output, must come after 'from' node topologically for feedforward).
          for (
            let j = Math.max(i + 1, this.input); // Ensure 'to' is not input and comes after 'from'.
            j < this.nodes.length;
            j++
          ) {
            const node2 = this.nodes[j];
            // Check if a connection doesn't already exist.
            if (!node1.isProjectingTo(node2)) available.push([node1, node2]);
          }
        }

        // If no possible new connections can be made, exit.
        if (available.length === 0) {
          // Avoid warning spam: console.warn('No possible new connections to add.');
          break;
        }

        // 1. Select a random pair of unconnected nodes.
  pair = available[Math.floor(this._rand() * available.length)];
        // 2. Connect the selected pair with a random weight.
        this.connect(pair[0], pair[1]);
        break;

      case mutation.SUB_CONN:
        // Removes an existing connection.
        // Find connections that can be safely removed (nodes should have other connections).
        possible = this.connections.filter(
          (conn) => {
            // Basic check: both nodes need to have multiple connections
            const fromHasMultiple = conn.from.connections.out.length > 1;
            const toHasMultiple = conn.to.connections.in.length > 1;
            
            // Verify we're not removing the last connection from a specific source to a layer
            // Get all nodes in the same "layer" as the target (to) node
            const toLayer = this.nodes.filter(n => 
              n.type === conn.to.type && 
              Math.abs(this.nodes.indexOf(n) - this.nodes.indexOf(conn.to)) < Math.max(this.input, this.output)
            );
            
            // Check if removing this would disconnect a source node from an entire layer
            let wouldDisconnectLayer = false;
            if (toLayer.length > 0) {
              // Count connections from the same source to this layer
              const connectionsToLayer = this.connections.filter(c => 
                c.from === conn.from && 
                toLayer.includes(c.to)
              );
              // If this is the only connection to this layer, don't remove it
              if (connectionsToLayer.length <= 1) {
                wouldDisconnectLayer = true;
              }
            }
            
            return fromHasMultiple && toHasMultiple && 
                  this.nodes.indexOf(conn.to) > this.nodes.indexOf(conn.from) && // Ensure it's not a back-connection
                  !wouldDisconnectLayer; // Don't disconnect layers completely
          }
        );

        // If no connections can be safely removed, exit.
        if (possible.length === 0) {
          // Avoid warning spam: console.warn('No connections available to remove.');
          break;
        }

        // 1. Select a random connection from the list of removable connections.
  randomConn = possible[Math.floor(this._rand() * possible.length)];
        // 2. Disconnect the nodes associated with the selected connection.
        this.disconnect(randomConn.from, randomConn.to);
        break;

      case mutation.MOD_WEIGHT:
        // Modifies the weight of a random existing connection (including self-connections).
        const allConnections = this.connections.concat(this.selfconns); // Combine regular and self-connections.
        if (allConnections.length === 0) break; // Exit if no connections exist.
        // 1. Select a random connection.
        connection =
          allConnections[Math.floor(this._rand() * allConnections.length)];
        // 2. Calculate a weight modification value based on method parameters (or defaults).
        // The `method` object itself might contain `max` and `min` properties.
        const modification =
          this._rand() * (method.max - method.min) + method.min;
        // 3. Add the modification to the connection's weight.
        connection.weight += modification;
        break;

      case mutation.MOD_BIAS:
        // Modifies the bias of a random hidden or output node.
        if (this.nodes.length <= this.input) break; // Exit if only input nodes exist.
        // 1. Select a random hidden or output node index.
        index = Math.floor(
          this._rand() * (this.nodes.length - this.input) + this.input // Range covers hidden and output nodes.
        );
        node = this.nodes[index];
        // 2. Call the node's mutate method to modify its bias.
        node.mutate(method);
        break;

      case mutation.MOD_ACTIVATION:
        // Changes the activation (squash) function of a random hidden or output node.
        // Check if mutation of output node activations is allowed by the method config.
        const canMutateOutput = method.mutateOutput ?? true; // Default to true if not specified
        const numMutableNodes =
          this.nodes.length - this.input - (canMutateOutput ? 0 : this.output);

        if (numMutableNodes <= 0) {
          if (config.warnings)
            console.warn(
              'No nodes available for activation function mutation based on config.'
            );
          break;
        }

        // 1. Select a random node index from the allowed range (hidden, potentially output).
        index = Math.floor(
          this._rand() * numMutableNodes + this.input // Offset by input nodes
        );
        node = this.nodes[index];
        // 2. Call the node's mutate method to change its squash function.
        node.mutate(method);
        break;

      case mutation.ADD_SELF_CONN:
  if (this._enforceAcyclic) { break; }
        // Adds a self-connection (node connects to itself) to a random node that doesn't already have one.
        // Find nodes (hidden or output) that currently have no self-connection (array is empty).
        possible = this.nodes.filter(
          (node, idx) => idx >= this.input && node.connections.self.length === 0 // Check array length
        );

        // If all eligible nodes already have self-connections, exit.
        if (possible.length === 0) {
          if (config.warnings)
            console.warn('All eligible nodes already have self-connections.');
          break;
        }

        // 1. Select a random node from the list of possibilities.
  node = possible[Math.floor(this._rand() * possible.length)];
        // 2. Connect the node to itself with a random weight.
        this.connect(node, node);
        break;

      case mutation.SUB_SELF_CONN:
        // Removes an existing self-connection.
        // Check if any self-connections exist.
        if (this.selfconns.length === 0) {
          if (config.warnings)
            console.warn('No self-connections exist to remove.');
          break;
        }

        // 1. Select a random self-connection from the `selfconns` list.
        connection = this.selfconns[
          Math.floor(this._rand() * this.selfconns.length)
        ];
        // 2. Disconnect the node from itself.
        this.disconnect(connection.from, connection.to); // from and to are the same node here.
        break;

      case mutation.ADD_GATE:
        // Adds gating to a random existing connection using a random node as the gater.
        const allConns = this.connections.concat(this.selfconns); // Consider all connection types.
        // Find connections that are not currently gated.
        possible = allConns.filter((conn) => conn.gater === null);

        // If all connections are already gated, exit.
        if (possible.length === 0) {
          if (config.warnings)
            console.warn('All connections are already gated.');
          break;
        }
        // Ensure there's at least one potential gating node (hidden/output)
        if (this.nodes.length <= this.input) break;

        // 1. Select a random node (hidden or output) to act as the gater.
        index = Math.floor(
          this._rand() * (this.nodes.length - this.input) + this.input
        );
        node = this.nodes[index]; // This node will control the gate.
        // 2. Select a random un-gated connection.
  connection = possible[Math.floor(this._rand() * possible.length)];
        // 3. Apply the gate using the selected node and connection.
        this.gate(node, connection);
        break;

      case mutation.SUB_GATE:
        // Removes the gating mechanism from a random gated connection.
        // Check if any gated connections exist.
        if (this.gates.length === 0) {
          if (config.warnings) console.warn('No gated connections to ungate.');
          break;
        }

        // 1. Select a random gated connection from the `gates` list.
  index = Math.floor(this._rand() * this.gates.length);
        const gatedConn = this.gates[index];
        // 2. Remove the gate from the selected connection.
        this.ungate(gatedConn);
        break;

      case mutation.ADD_BACK_CONN:
  if (this._enforceAcyclic) { break; }
        // Adds a recurrent connection where the 'from' node appears later in the `nodes` array than the 'to' node.
        available = []; // Stores possible [fromNode, toNode] pairs for back-connections.
        // Iterate through possible 'from' nodes (hidden and output).
        for (let i = this.input; i < this.nodes.length; i++) {
          const node1 = this.nodes[i];
          // Iterate through possible 'to' nodes (hidden only, must appear *before* 'from' node).
          for (let j = this.input; j < i; j++) {
            const node2 = this.nodes[j];
            // Check if a connection (in this direction) doesn't already exist.
            if (!node1.isProjectingTo(node2)) available.push([node1, node2]);
          }
        }

        // If no possible back-connections can be made, exit.
        if (available.length === 0) {
          // Avoid warning spam: console.warn('No possible new back-connections to add.');
          break;
        }

        // 1. Select a random pair for the back-connection.
  pair = available[Math.floor(this._rand() * available.length)];
        // 2. Connect the selected pair (note: connection weight is random).
        this.connect(pair[0], pair[1]);
        break;

      case mutation.SUB_BACK_CONN:
        // Removes an existing recurrent (back-) connection.
        // Find back-connections that can be safely removed.
        possible = this.connections.filter(
          (conn) =>
            conn.from.connections.out.length > 1 && // 'from' node has other outputs.
            conn.to.connections.in.length > 1 && // 'to' node has other inputs.
            this.nodes.indexOf(conn.from) > this.nodes.indexOf(conn.to) // Identify back-connections.
        );

        // If no removable back-connections are found, exit.
        if (possible.length === 0) {
          // Avoid warning spam: console.warn('No back-connections available to remove.');
          break;
        }

        // 1. Select a random back-connection from the list.
  randomConn = possible[Math.floor(this._rand() * possible.length)];
        // 2. Disconnect the nodes associated with the selected back-connection.
        this.disconnect(randomConn.from, randomConn.to);
        break;

      case mutation.SWAP_NODES:
        // Swaps the bias and activation function between two random nodes (hidden or output).
        // Check if there are at least two nodes eligible for swapping.
        const canSwapOutput = method.mutateOutput ?? true; // Check method config
        const numSwappableNodes =
          this.nodes.length - this.input - (canSwapOutput ? 0 : this.output);

        if (numSwappableNodes < 2) {
          // Avoid warning spam: console.warn('Not enough nodes to perform swap.');
          break;
        }

        // 1. Select two distinct random indices for eligible nodes.
        let node1Index = Math.floor(
          this._rand() * numSwappableNodes + this.input
        );
        let node2Index = Math.floor(
          this._rand() * numSwappableNodes + this.input
        );
        // Ensure the indices are different.
        while (node1Index === node2Index) {
          node2Index = Math.floor(
            this._rand() * numSwappableNodes + this.input
          );
        }

        const node1 = this.nodes[node1Index];
        const node2 = this.nodes[node2Index];

        // 2. Swap the bias and squash function between the two selected nodes.
        const tempBias = node1.bias;
        const tempSquash = node1.squash;

        node1.bias = node2.bias;
        node1.squash = node2.squash;
        node2.bias = tempBias;
        node2.squash = tempSquash;
        break;

      case mutation.ADD_LSTM_NODE: {
  if (this._enforceAcyclic) { break; }
        if (this.connections.length === 0) break;
        connection = this.connections[
          Math.floor(Math.random() * this.connections.length)
        ];
        const gaterLSTM = connection.gater;
        this.disconnect(connection.from, connection.to);
        // Dynamically import Layer for compatibility
        const Layer = require('./layer').default;
        const lstmLayer = Layer.lstm(1);
        // Insert all LSTM nodes into the network as hidden nodes
        lstmLayer.nodes.forEach((n: import('./node').default) => { n.type = 'hidden'; this.nodes.push(n); });
        // Connect from->LSTM input, LSTM output->to
        this.connect(connection.from, lstmLayer.nodes[0]);
        this.connect(lstmLayer.output.nodes[0], connection.to);
        if (gaterLSTM) {
          this.gate(gaterLSTM, this.connections[this.connections.length-1]);
        }
        break;
      }
      case mutation.ADD_GRU_NODE: {
  if (this._enforceAcyclic) { break; }
        if (this.connections.length === 0) break;
        connection = this.connections[
          Math.floor(Math.random() * this.connections.length)
        ];
        const gaterGRU = connection.gater;
        this.disconnect(connection.from, connection.to);
        const Layer = require('./layer').default;
        const gruLayer = Layer.gru(1);
        gruLayer.nodes.forEach((n: import('./node').default) => { n.type = 'hidden'; this.nodes.push(n); });
        this.connect(connection.from, gruLayer.nodes[0]);
        this.connect(gruLayer.output.nodes[0], connection.to);
        if (gaterGRU) {
          this.gate(gaterGRU, this.connections[this.connections.length-1]);
        }
        break;
      }
    }
  // structural mutations invalidate topo cache
  this._topoDirty = true;
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
    if (this._enforceAcyclic && this.nodes.indexOf(from) > this.nodes.indexOf(to)) {
      // Disallow back-edges in acyclic mode
      return [];
    }
    // Delegate the actual connection creation to the 'from' node.
    const connections = from.connect(to, weight);

    // Add the created connection(s) to the network's lists.
    for (const connection of connections) {
      if (from !== to) {
        // Regular connection (feedforward or backward)
        this.connections.push(connection);
      } else {
        // Self-connection
    if (this._enforceAcyclic) continue; // ignore in acyclic mode
        this.selfconns.push(connection);
      }
    }
  if (connections.length) this._topoDirty = true;

    return connections; // Return the created connection object(s).
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
  gate(node: Node, connection: Connection): void {
    // Validate that the gating node belongs to this network.
    if (!this.nodes.includes(node)) {
      throw new Error(
        'Gating node must be part of the network to gate a connection!'
      );
    }
    // Check if the connection is already gated. Avoids redundant gating.
    if (connection.gater) {
      if (config.warnings)
        console.warn('Connection is already gated. Skipping.');
      return; // Exit if already gated.
    }
    // Delegate the gating logic to the gating node.
    node.gate(connection);
    // Add the newly gated connection to the network's list of gates.
    this.gates.push(connection);
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
  remove(node: Node): void {
    // Prevent removal of input or output nodes
    if (node.type === 'input' || node.type === 'output') {
      throw new Error('Cannot remove input or output node from the network.');
    }

    // Find the index of the node to remove.
    const index = this.nodes.indexOf(node);

    // Validate that the node exists in the network.
    if (index === -1) {
      throw new Error('Node not found in the network for removal.');
    }

    // Keep track of nodes that were gating connections involving the removed node.
    const gaters: Node[] = [];

    // Disconnect and remove the self-connection, if it exists.
    this.disconnect(node, node); // Handles removal from `selfconns` list.

    // Disconnect all incoming connections.
    const inputs: Node[] = []; // Store the 'from' nodes of incoming connections.
    // Iterate backwards as `disconnect` modifies the `node.connections.in` array.
    for (let i = node.connections.in.length - 1; i >= 0; i--) {
      const connection = node.connections.in[i];
      // If configured to keep gates and the connection was gated by another node, store the gater.
      if (
        mutation.SUB_NODE.keep_gates && // Check configuration option
        connection.gater !== null &&
        connection.gater !== node // Ensure gater is not the node being removed
      ) {
        gaters.push(connection.gater);
      }
      inputs.push(connection.from); // Store the input node.
      this.disconnect(connection.from, node); // Disconnect input -> node.
    }

    // Disconnect all outgoing connections.
    const outputs: Node[] = []; // Store the 'to' nodes of outgoing connections.
    // Iterate backwards as `disconnect` modifies the `node.connections.out` array.
    for (let i = node.connections.out.length - 1; i >= 0; i--) {
      const connection = node.connections.out[i];
      // If configured to keep gates and the connection was gated by another node, store the gater.
      if (
        mutation.SUB_NODE.keep_gates && // Check configuration option
        connection.gater !== null &&
        connection.gater !== node // Ensure gater is not the node being removed
      ) {
        gaters.push(connection.gater);
      }
      outputs.push(connection.to); // Store the output node.
      this.disconnect(node, connection.to); // Disconnect node -> output.
    }

    // Reconnect inputs to outputs to bridge the gap, if they are not already connected.
    const connections: Connection[] = []; // Store newly created connections.
    for (const input of inputs) {
      for (const output of outputs) {
        // Avoid creating duplicate connections or self-loops if input === output.
        if (input !== output && !input.isProjectingTo(output)) {
          const conn = this.connect(input, output); // Create new connection.
          if (conn.length > 0) connections.push(conn[0]); // Store the new connection.
        }
      }
    }

    // Re-apply gates (if `keep_gates` is true) to some of the newly created connections.
    // This attempts to preserve the gating logic that involved the removed node's connections.
    for (const gater of gaters) {
      if (connections.length === 0) break; // Stop if no new connections to gate.

      // Select a random new connection to gate.
      const connIndex = Math.floor(Math.random() * connections.length);
      this.gate(gater, connections[connIndex]); // Apply the gate.
      connections.splice(connIndex, 1); // Remove the connection from the list so it's not gated again.
    }

    // Ungate any connections that were gated *by* the node being removed.
    // Iterate backwards as `ungate` modifies the `node.connections.gated` array.
    for (let i = node.connections.gated.length - 1; i >= 0; i--) {
      const conn = node.connections.gated[i];
      this.ungate(conn); // Remove the gate.
    }

    // Remove the node itself from the network's list.
    this.nodes.splice(index, 1);
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
    // Determine which list holds the connection (`connections` or `selfconns`).
    const connectionsList = from === to ? this.selfconns : this.connections;

    // Find and remove the connection from the network's list.
    for (let i = 0; i < connectionsList.length; i++) {
      const connection = connectionsList[i];
      if (connection.from === from && connection.to === to) {
        // If the connection is gated, remove the gate first.
        if (connection.gater !== null) {
          this.ungate(connection); // Removes from `this.gates` list.
        }
        // Remove the connection from the appropriate list (`connections` or `selfconns`).
        connectionsList.splice(i, 1);
        // Connection found and removed, break the loop.
        break;
      }
    }

    // Delegate to the 'from' node to remove the connection reference from its internal lists.
    from.disconnect(to);
  this._topoDirty = true;
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
  ungate(connection: Connection): void {
    // Find the index of the connection in the `gates` list.
    const index = this.gates.indexOf(connection);
    // If not found, just return (defensive: should never throw in production)
    if (index === -1) {
      if (config.warnings) {
        console.warn('Attempted to ungate a connection not in the gates list.');
      }
      return;
    }

    // Remove the connection from the `gates` list.
    this.gates.splice(index, 1);
    // Delegate to the gater node (if it exists) to remove the gating reference from its internal list.
    connection.gater?.ungate(connection); // Safely call ungate on the gater.
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
  // Removed legacy _trainSet; delegated to network.train.ts

  // Gradient clipping implemented in network.train.ts (applyGradientClippingImpl). Kept here only for backward compat if reflection used.
  private _applyGradientClipping(cfg: { mode: 'norm'|'percentile'|'layerwiseNorm'|'layerwisePercentile'; maxNorm?: number; percentile?: number }) {
    const { applyGradientClippingImpl } = require('./network.train');
    applyGradientClippingImpl(this as any, cfg);
  }

  // Training is implemented in network.train.ts; this wrapper keeps public API stable.
  train(set: { input: number[]; output: number[] }[], options: any): { error: number; iterations: number; time: number } {
    const { trainImpl } = require('./network.train');
    return trainImpl(this as any, set, options);
  }

  /** Returns last recorded raw (pre-update) gradient L2 norm. */
  getRawGradientNorm(): number { return this._lastRawGradNorm; }
  /** Returns current mixed precision loss scale (1 if disabled). */
  getLossScale(): number { return this._mixedPrecision.lossScale; }
  /** Returns last gradient clipping group count (0 if no clipping yet). */
  getLastGradClipGroupCount(): number { return this._lastGradClipGroupCount; }
  /** Consolidated training stats snapshot. */
  getTrainingStats() { return { gradNorm: this._lastGradNorm ?? 0, gradNormRaw: this._lastRawGradNorm, lossScale: this._mixedPrecision.lossScale, optimizerStep: this._optimizerStep, mp: { good: this._mixedPrecisionState.goodSteps, bad: this._mixedPrecisionState.badSteps, overflowCount: this._mixedPrecisionState.overflowCount||0, scaleUps: this._mixedPrecisionState.scaleUpEvents||0, scaleDowns: this._mixedPrecisionState.scaleDownEvents||0, lastOverflowStep: this._lastOverflowStep } }; }
  /** Utility: adjust rate for accumulation mode (use result when switching to 'sum' to mimic 'average'). */
  static adjustRateForAccumulation(rate: number, accumulationSteps: number, reduction: 'average'|'sum') {
    if (reduction === 'sum' && accumulationSteps > 1) return rate / accumulationSteps;
    return rate;
  }


  // Evolution wrapper delegates to network.evolve.ts implementation.
  async evolve(
    set: { input: number[]; output: number[] }[],
    options: any
  ): Promise<{ error: number; iterations: number; time: number }> {
  const { evolveNetwork } = await import('./network.evolve');
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
          `Test sample input size mismatch: expected ${this.input}, got ${sample.input ? sample.input.length : 'undefined'}`
        );
      }
      if (!Array.isArray(sample.output) || sample.output.length !== this.output) {
        throw new Error(
          `Test sample output size mismatch: expected ${this.output}, got ${sample.output ? sample.output.length : 'undefined'}`
        );
      }
    }

    let error = 0; // Accumulator for the total error.
    const costFn = cost || methods.Cost.mse; // Use provided cost function or default to MSE.
    const start = Date.now(); // Start time measurement.

    // --- Dropout/inference transition: Explicitly reset all hidden node masks to 1 for robust inference ---
    this.nodes.forEach(node => {
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

  

  /**
   * Serializes the network's current state into a compact array format.
   * This format is primarily intended for efficient transfer, e.g., between web workers.
   * It includes activations, states, squash function names, and connection details.
   * Note: This is a lightweight serialization, different from the more comprehensive `toJSON`.
   *
   * @returns {any[]} An array containing serialized network data:
   *                  `[activations, states, squashFunctionNames, connectionData, inputSize, outputSize]`
   *                  where `connectionData` is an array of objects like
   *                  `{ from: number, to: number, weight: number, gater: number | null }`.
   *                  Indices refer to the node's position in the `nodes` array.
   */
  serialize(): any[] {
    // Ensure all nodes have an index assigned (should be done by methods like standalone or toJSON, but good practice here too).
    this.nodes.forEach((node, index) => (node.index = index));

    // Extract current activations and states from all nodes.
    const activations = this.nodes.map((node) => node.activation);
    const states = this.nodes.map((node) => node.state);
    // Extract the names of the squash functions used by each node.
    const squashes = this.nodes.map((node) => node.squash.name);
    // Serialize connection data, including self-connections.
    const connections = this.connections.concat(this.selfconns).map((conn) => ({
      from: conn.from.index, // Source node index.
      to: conn.to.index, // Target node index.
      weight: conn.weight, // Connection weight.
      gater: conn.gater ? conn.gater.index : null, // Gating node index or null.
    }));
    // Include input/output sizes for robust deserialization
    return [activations, states, squashes, connections, this.input, this.output];
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
  static deserialize(data: any[], inputSize?: number, outputSize?: number): Network {
    // Unpack the serialized data.
    const [activations, states, squashes, connections, serializedInput, serializedOutput] = data;
    // Use provided input/output size, or fallback to serialized values, or fallback to 0
    const input = typeof inputSize === 'number' ? inputSize : (typeof serializedInput === 'number' ? serializedInput : 0);
    const output = typeof outputSize === 'number' ? outputSize : (typeof serializedOutput === 'number' ? serializedOutput : 0);
    // Create a new network shell. The actual structure will be defined by the nodes and connections data.
    const network = new Network(input, output);
    network.nodes = [];
    network.connections = [];
    network.selfconns = [];
    network.gates = [];

    // Recreate nodes and set their state.
    activations.forEach((activation: number, index: number) => {
      let type: string;
      if (index < input) type = 'input';
      else if (index >= activations.length - output) type = 'output';
      else type = 'hidden';
  const node = new Node(type);
      node.activation = activation;
      node.state = states[index];
      const squashName = squashes[index] as keyof typeof methods.Activation;
      const squashFn = methods.Activation[squashName];
      if (squashFn) {
        node.squash = squashFn as (x: number, derivate?: boolean) => number;
      } else {
        console.warn(
          `Deserialization: Unknown squash function '${squashName}'. Using identity.`
        );
        node.squash = methods.Activation.identity;
      }
      node.index = index;
      network.nodes.push(node);
    });

    // Recreate connections.
    connections.forEach(
      (connData: {
        from: number;
        to: number;
        weight: number;
        gater: number | null;
      }) => {
        if (
          connData.from < network.nodes.length &&
          connData.to < network.nodes.length
        ) {
          const fromNode: Node = network.nodes[connData.from];
          const toNode: Node = network.nodes[connData.to];
          const connection: Connection | undefined = network.connect(
            fromNode,
            toNode,
            connData.weight
          )[0];
          if (connection && connData.gater !== null) {
            if (connData.gater < network.nodes.length) {
              network.gate(network.nodes[connData.gater], connection);
            } else {
              console.warn(
                `Deserialization: Invalid gater index ${connData.gater}.`
              );
            }
          }
        } else {
          console.warn(
            `Deserialization: Invalid connection indices ${connData.from} -> ${connData.to}.`
          );
        }
      }
    );

    return network;
  }

  /**
   * Converts the network into a JSON object representation (latest standard).
   * Includes formatVersion, and only serializes properties needed for full reconstruction.
   * All references are by index. Excludes runtime-only properties (activation, state, traces).
   *
   * @returns {object} A JSON-compatible object representing the network.
   */
  toJSON(): object {
    const json: any = {
      formatVersion: 2,
      input: this.input,
      output: this.output,
      dropout: this.dropout,
      nodes: [],
      connections: []
    };
    // Serialize nodes (type, bias, squash, index)
    this.nodes.forEach((node, index) => {
      node.index = index;
      json.nodes.push({
        type: node.type,
        bias: node.bias,
        squash: node.squash.name,
  index,
  geneId: (node as any).geneId
      });
      // Self-connection (if any)
      if (node.connections.self.length > 0) {
        const selfConn = node.connections.self[0];
        json.connections.push({
          from: index,
          to: index,
          weight: selfConn.weight,
            gater: selfConn.gater ? selfConn.gater.index : null,
            enabled: (selfConn as any).enabled !== false
        });
      }
    });
    // Serialize regular connections
    this.connections.forEach(conn => {
      if (typeof conn.from.index !== 'number' || typeof conn.to.index !== 'number') return;
      json.connections.push({
        from: conn.from.index,
        to: conn.to.index,
        weight: conn.weight,
  gater: conn.gater ? conn.gater.index : null,
  enabled: (conn as any).enabled !== false
      });
    });
    return json;
  }

  /**
   * Reconstructs a network from a JSON object (latest standard).
   * Handles formatVersion, robust error handling, and index-based references.
   * @param {object} json - The JSON object representing the network.
   * @returns {Network} The reconstructed network.
   */
  static fromJSON(json: any): Network {
    if (!json || typeof json !== 'object') throw new Error('Invalid JSON for network.');
    if (json.formatVersion !== 2) {
      console.warn('fromJSON: Unknown or missing formatVersion. Attempting best-effort import.');
    }
    const network = new Network(json.input, json.output);
    network.dropout = json.dropout || 0;
    network.nodes = [];
    network.connections = [];
    network.selfconns = [];
    network.gates = [];
    // Recreate nodes
    json.nodes.forEach((n: any, i: number) => {
  const node = new Node(n.type);
      node.bias = n.bias;
      node.squash = methods.Activation[n.squash] || methods.Activation.identity;
      node.index = i;
      if (typeof n.geneId === 'number') (node as any).geneId = n.geneId; // preserve stable gene id
      network.nodes.push(node);
    });
    // Recreate connections
    json.connections.forEach((c: any) => {
      if (typeof c.from !== 'number' || typeof c.to !== 'number') return;
      const from = network.nodes[c.from];
      const to = network.nodes[c.to];
  const conn = network.connect(from, to, c.weight)[0];
      if (conn && c.gater !== null && typeof c.gater === 'number' && network.nodes[c.gater]) {
        network.gate(network.nodes[c.gater], conn);
      }
  if (conn && typeof c.enabled !== 'undefined') (conn as any).enabled = c.enabled;
    });
    return network;
  }

  /**
   * Merges two networks sequentially.
   * Creates a new network where the output nodes of the first network effectively become
   * the input nodes for the second network's hidden/output layers.
   * This is a structural merge; weights and biases are copied. It does not automatically
   * connect the output layer of network1 to the input layer of network2.
   * The current implementation seems flawed and might not produce a functional merged network as intended.
   * It primarily concatenates node and connection lists with some index adjustments.
   *
   * @param {Network} network1 - The first network (provides input layer and potentially hidden layers).
   * @param {Network} network2 - The second network (provides hidden/output layers).
   * @returns {Network} A new, fully connected, layered MLP
   * @throws {Error} If the output size of `network1` does not match the input size of `network2`. (This check might be misleading given the implementation).
   * @static
   * @deprecated The current implementation is likely incorrect for functional merging. Review needed.
   */
  static merge(network1: Network, network2: Network): Network {
    // Validate that the interface between the networks matches.
    if (network1.output !== network2.input) {
      throw new Error(
        'Output size of network1 must match input size of network2 for merging!'
      );
    }

    // Create a new network shell with the input size of network1 and output size of network2.
    const merged = new Network(network1.input, network2.output);

    // Combine nodes: Take all nodes from network1, and then the hidden/output nodes from network2.
    // This assumes network2's input nodes are conceptually replaced by network1's output nodes.
    merged.nodes = [
      ...network1.nodes, // All nodes from network1
      ...network2.nodes.slice(network2.input), // Hidden and output nodes from network2
    ];
    // Re-index all nodes in the merged network.
    merged.nodes.forEach((node, index) => (node.index = index));

    // Combine connections. This part is complex and potentially incorrect.
    // It copies connections from network1 directly.
    // It copies connections from network2, attempting to remap connections originating from network2's input layer
    // to connect from network1's output layer instead. This remapping logic seems fragile.
    merged.connections = [
      ...network1.connections, // All connections from network1
      ...network2.connections.map((conn) => {
        // Remap connections starting from network2's input nodes.
        if (conn.from.type === 'input') {
          // Find the corresponding output node in network1.
          // This logic assumes a direct mapping based on index, which might be wrong.
          // `network2.nodes.indexOf(conn.from)` gives the index within network2's node list.
          // We need to map this input index (0 to network2.input-1) to the corresponding output node index in network1
          // (network1.input to network1.input + network1.output - 1).
          const inputIndexInNet2 = network2.nodes.indexOf(conn.from);
          if (inputIndexInNet2 >= 0 && inputIndexInNet2 < network2.input) {
            const correspondingOutputIndexInNet1 =
              network1.input + inputIndexInNet2;
            if (correspondingOutputIndexInNet1 < network1.nodes.length) {
              conn.from = network1.nodes[correspondingOutputIndexInNet1]; // Remap 'from' node.
            } else {
              // Handle error: Corresponding output node not found in network1.
              console.warn("Merge: Error remapping connection 'from' node.");
            }
          }
        }
        // Remap 'to' nodes and 'gater' nodes if they came from network2's hidden/output layers.
        // Their indices need to be offset by the number of nodes in network1.
        const toIndexInNet2 = network2.nodes.indexOf(conn.to);
        if (toIndexInNet2 >= network2.input) {
          conn.to =
            merged.nodes[
              network1.nodes.length + (toIndexInNet2 - network2.input)
            ];
        }
        if (conn.gater) {
          const gaterIndexInNet2 = network2.nodes.indexOf(conn.gater);
          if (gaterIndexInNet2 >= network2.input) {
            conn.gater =
              merged.nodes[
                network1.nodes.length + (gaterIndexInNet2 - network2.input)
              ];
          } else if (gaterIndexInNet2 >= 0) {
            // Gater was an input node in network2, needs remapping similar to 'from'.
            const correspondingOutputIndexInNet1 =
              network1.input + gaterIndexInNet2;
            if (correspondingOutputIndexInNet1 < network1.nodes.length) {
              conn.gater = network1.nodes[correspondingOutputIndexInNet1];
            } else {
              conn.gater = null; // Gater cannot be remapped.
            }
          }
        }
        return conn; // Return the potentially modified connection.
      }),
    ];
    // TODO: This merge logic needs significant review and likely correction to be functionally useful.
    // It doesn't handle self-connections or gates from network2 properly after index changes.

    return merged; // Return the merged network structure.
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
  static crossOver(
    network1: Network,
    network2: Network,
    equal: boolean = false // Default to fitness-based inheritance for disjoint/excess genes.
  ): Network {
    // Validate that parents have compatible input/output dimensions.
    if (
      network1.input !== network2.input ||
      network1.output !== network2.output
    ) {
      throw new Error(
        'Parent networks must have the same input and output sizes for crossover.'
      );
    }

    // Create a new network shell for the offspring.
    const offspring = new Network(network1.input, network1.output);
    offspring.connections = []; // Clear default connections.
    offspring.nodes = []; // Clear default nodes.
    offspring.selfconns = [];
    offspring.gates = [];

    // Get parent fitness scores (default to 0 if undefined).
    const score1 = network1.score || 0;
    const score2 = network2.score || 0;

    // Determine the number of non-output nodes in the offspring network.
    // This influences which nodes and connections are potentially included.
    let size; // Number of nodes to inherit (excluding output layer initially).
    const n1Size = network1.nodes.length;
    const n2Size = network2.nodes.length;
    if (equal || score1 === score2) {
      // If equal fitness or `equal` flag is true, choose a random size between the parents' sizes.
      const max = Math.max(n1Size, n2Size);
      const min = Math.min(n1Size, n2Size);
      size = Math.floor(Math.random() * (max - min + 1) + min);
    } else if (score1 > score2) {
      // Inherit size from the fitter parent (network1).
      size = n1Size;
    } else {
      // Inherit size from the fitter parent (network2).
      size = n2Size;
    }

    const outputSize = network1.output; // Number of output nodes is fixed.

    // Assign indices to nodes in parent networks for consistent referencing.
    network1.nodes.forEach((node, index) => (node.index = index));
    network2.nodes.forEach((node, index) => (node.index = index));

    // Inherit nodes up to the determined size.
    for (let i = 0; i < size; i++) {
      // Input nodes (i < inputSize) and Output nodes (handled later) must always be included structurally.
      // Hidden nodes are chosen based on parentage and random chance.

      let chosenNode: Node | undefined = undefined;
      const node1 = i < n1Size ? network1.nodes[i] : undefined; // Get node from parent 1 if index is valid.
      const node2 = i < n2Size ? network2.nodes[i] : undefined; // Get node from parent 2 if index is valid.

      if (i < network1.input) {
        // Input nodes: Always inherit structure, choose bias/squash randomly?
        // NEAT typically keeps input nodes standard. Let's inherit from parent 1 arbitrarily.
        chosenNode = node1;
      } else if (i >= size - outputSize) {
        // Output nodes: Determine which parent's output node corresponds to this position.
        // This logic assumes output nodes are at the end and indices align after potential hidden node differences.
        const outputIndexInParent1 = n1Size - (size - i);
        const outputIndexInParent2 = n2Size - (size - i);
        const node1Output =
          outputIndexInParent1 >= network1.input &&
          outputIndexInParent1 < n1Size
            ? network1.nodes[outputIndexInParent1]
            : undefined;
        const node2Output =
          outputIndexInParent2 >= network2.input &&
          outputIndexInParent2 < n2Size
            ? network2.nodes[outputIndexInParent2]
            : undefined;

        if (node1Output && node2Output) {
          // Both parents have a corresponding output node. Choose randomly.
          const _rngC = (network1 as any)._rand || (network2 as any)._rand || Math.random;
          chosenNode = _rngC() >= 0.5 ? node1Output : node2Output;
        } else {
          // Only one parent has this output node (due to size difference). Inherit from that parent.
          chosenNode = node1Output || node2Output;
        }
      } else {
        // Hidden nodes:
        if (node1 && node2) {
          // Both parents have a node at this index (matching gene). Choose randomly.
          const _rngH = (network1 as any)._rand || (network2 as any)._rand || Math.random;
          chosenNode = _rngH() >= 0.5 ? node1 : node2;
        } else if (node1 && (score1 >= score2 || equal)) {
          // Only parent 1 has this node (disjoint/excess) and is fitter or `equal` is true.
          chosenNode = node1;
        } else if (node2 && (score2 >= score1 || equal)) {
          // Only parent 2 has this node (disjoint/excess) and is fitter or `equal` is true.
          chosenNode = node2;
        }
        // If neither condition met (e.g., node exists only in weaker parent and equal=false), it's not inherited.
      }

      // If a node was chosen for inheritance, create a copy for the offspring.
      if (chosenNode) {
  const newNode = new Node(chosenNode.type); // Copy type.
        newNode.bias = chosenNode.bias; // Copy bias.
        newNode.squash = chosenNode.squash; // Copy squash function reference.
        // Note: Connections are handled separately below.
        offspring.nodes.push(newNode);
      } else {
        // This case (a gap in inherited nodes) might complicate connection inheritance.
        // NEAT usually includes all nodes up to the max index of inherited connections.
        // Let's push a placeholder or reconsider the node inheritance logic.
        // For simplicity now, we assume `size` correctly reflects the nodes to inherit.
        // If a node isn't inherited, connections to/from it also won't be inherited.
      }
    }
    // Re-index offspring nodes.
    offspring.nodes.forEach((node, index) => (node.index = index));
    const offspringNodeCount = offspring.nodes.length; // Actual number of nodes inherited.

    // --- Inherit Connections ---

    // Create dictionaries to store connection data keyed by innovation ID for efficient lookup.
    const n1conns: Record<string, any> = {}; // Connections from parent 1.
    const n2conns: Record<string, any> = {}; // Connections from parent 2.

    // Populate connection dictionaries for parent 1.
    const allParent1Conns = network1.connections.concat(network1.selfconns);
  allParent1Conns.forEach((conn) => {
      // Ensure indices are valid before creating innovation ID.
      if (
        typeof conn.from.index === 'number' &&
        typeof conn.to.index === 'number'
      ) {
        const innovId = Connection.innovationID(conn.from.index, conn.to.index);
        n1conns[innovId] = {
          weight: conn.weight,
          from: conn.from.index,
          to: conn.to.index,
          // Store gater index (-1 if no gater).
      gater: conn.gater ? conn.gater.index : -1,
      enabled: (conn as any).enabled !== false
        };
      }
    });

    // Populate connection dictionaries for parent 2.
    const allParent2Conns = network2.connections.concat(network2.selfconns);
  allParent2Conns.forEach((conn) => {
      // Ensure indices are valid.
      if (
        typeof conn.from.index === 'number' &&
        typeof conn.to.index === 'number'
      ) {
        const innovId = Connection.innovationID(conn.from.index, conn.to.index);
        n2conns[innovId] = {
          weight: conn.weight,
          from: conn.from.index,
          to: conn.to.index,
      gater: conn.gater ? conn.gater.index : -1,
      enabled: (conn as any).enabled !== false
        };
      }
    });

    // Determine which connections to inherit based on matching, disjoint, and excess genes.
    const connectionsToInherit: any[] = []; // List to store chosen connection data.
    const keys1 = Object.keys(n1conns); // Innovation IDs from parent 1.
    const keys2 = Object.keys(n2conns); // Innovation IDs from parent 2.

    // Process connections present in parent 1.
    keys1.forEach((key) => {
      const conn1Data = n1conns[key];
      if (n2conns[key]) {
        // Matching gene: Connection exists in both parents. Inherit randomly.
        const conn2Data = n2conns[key];
  const _rngConn = (network1 as any)._rand || (network2 as any)._rand || Math.random;
        const chosen = _rngConn() >= 0.5 ? conn1Data : conn2Data;
        // If either parent has it disabled, inherit disabled unless re-enabled probability triggers
        if ((conn1Data.enabled === false || conn2Data.enabled === false)) {
          const reenableProb = (network1 as any)._reenableProb ?? (network2 as any)._reenableProb ?? 0.25;
          const reenabled = Math.random() < reenableProb;
          chosen.enabled = reenabled;
          if (reenabled) {
            // track success counters
            (network1 as any)._reenableSuccess = ((network1 as any)._reenableSuccess||0)+1;
          }
          (network1 as any)._reenableAttempts = ((network1 as any)._reenableAttempts||0)+1;
        }
        connectionsToInherit.push(chosen);
        // Remove from parent 2's keys to avoid processing again.
        delete n2conns[key];
      } else if (score1 >= score2 || equal) {
        // Disjoint/Excess gene in parent 1: Inherit if parent 1 is fitter or `equal` is true.
        // Disjoint/excess: carry enabled status with chance to re-enable if disabled in fitter parent
        if (conn1Data.enabled === false) {
          const reenableProb = (network1 as any)._reenableProb ?? 0.25;
          const reenabled = Math.random() < reenableProb;
          conn1Data.enabled = reenabled;
          if (reenabled) (network1 as any)._reenableSuccess = ((network1 as any)._reenableSuccess||0)+1;
          (network1 as any)._reenableAttempts = ((network1 as any)._reenableAttempts||0)+1;
        }
        connectionsToInherit.push(conn1Data);
      }
    });

    // Process remaining connections (only present in parent 2).
    if (score2 >= score1 || equal) {
      // Inherit disjoint/excess genes from parent 2 if it's fitter or `equal` is true.
      Object.keys(n2conns).forEach((key) => {
        const d = n2conns[key];
        if (d.enabled === false) {
          const reenableProb = (network2 as any)._reenableProb ?? 0.25;
          const reenabled = Math.random() < reenableProb;
          d.enabled = reenabled;
          if (reenabled) (network2 as any)._reenableSuccess = ((network2 as any)._reenableSuccess||0)+1;
          (network2 as any)._reenableAttempts = ((network2 as any)._reenableAttempts||0)+1;
        }
        connectionsToInherit.push(d);
      });
    }

    // Create the inherited connections in the offspring network.
    connectionsToInherit.forEach((connData) => {
      // Check if the 'from' and 'to' nodes for this connection were actually inherited by the offspring.
      // Use `offspringNodeCount` which reflects the actual number of nodes added.
      if (
        connData.to < offspringNodeCount &&
        connData.from < offspringNodeCount
      ) {
        // Get the corresponding nodes from the offspring's list.
        const from = offspring.nodes[connData.from];
        const to = offspring.nodes[connData.to];

        // Check if connection already exists (shouldn't happen with this logic, but good practice).
        if (!from.isProjectingTo(to)) {
          // Create the connection using the offspring's method.
          const conn = offspring.connect(from, to)[0];
          if (conn) {
            conn.weight = connData.weight; // Set the inherited weight.
            (conn as any).enabled = connData.enabled !== false; // apply enabled status

            // Re-apply gating if the inherited connection was gated and the gater node was also inherited.
            if (connData.gater !== -1 && connData.gater < offspringNodeCount) {
              offspring.gate(offspring.nodes[connData.gater], conn);
            }
          }
        }
      }
      // If nodes were not inherited, the connection is implicitly dropped.
    });

    return offspring; // Return the created offspring network.
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
  static createMLP(inputCount: number, hiddenCounts: number[], outputCount: number): Network {
    // Create all nodes
  const inputNodes = Array.from({ length: inputCount }, () => new Node('input'));
  const hiddenLayers: Node[][] = hiddenCounts.map(count => Array.from({ length: count }, () => new Node('hidden')));
  const outputNodes = Array.from({ length: outputCount }, () => new Node('output'));
    // Flatten all nodes in topological order
    const allNodes = [
      ...inputNodes,
      ...hiddenLayers.flat(),
      ...outputNodes
    ];
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
    net.connections = net.nodes.flatMap(n => n.connections.out);
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
    net.nodes.forEach(node => {
      node.connections.out.forEach(conn => {
        allConnections.add(conn);
      });
    });
    net.connections = Array.from(allConnections) as Connection[];
  }
}
