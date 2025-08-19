/**
 * Connection (aka Synapse / Link)
 * ===============================
 * A `Connection` represents a directed, weighted edge between two `Node`s in a neural network graph.
 *
 * Responsibilities / Stored State
 * - References to the source (`from`) and target (`to`) nodes
 * - The weight scalar applied to the source activation when propagating to the target
 * - (Optional) a gating node (`gater`) whose activation modulates an additional multiplicative gain
 * - Training traces: standard eligibility trace plus optional extended / modulatory traces (`xtrace`)
 * - Bookkeeping information for evolutionary algorithms (NEAT-style innovation number)
 * - Bookkeeping for optimizers (Adam / RMSProp / Adagrad / Lion / Lookahead etc.) – allocated lazily
 * - Lightweight binary flags for enable / disable (genetic expression) and DropConnect masks
 *
 * Design Goals
 * 1. Memory efficiency: rarely-used optimizer fields & non-neutral gain are omitted until needed.
 * 2. Low GC churn: a simple object pool (`acquire` / `release`) recycles instances during topology evolution.
 * 3. Educational clarity: rich docs & examples show how to use connections in evolutionary + gradient settings.
 *
 * (Edited: gater is now virtualized – non-enumerable symbol storage + hasGater bit (bit2).)
 */
import Node from './node'; // Import Node type

// Symbol used for optional gain storage (non-enumerable). Neutral gain=1 omitted entirely.
const kGain = Symbol('connGain');
// Symbol used for optional gater storage (non-enumerable). Presence tracked with bit2 flag.
const kGater = Symbol('connGater');
// Symbol-backed optimizer moment bag (amortizes 7 rarely-used numeric fields into a single optional object).
// Accessed via prototype accessors so assigning e.g. `conn.firstMoment = x` does NOT create an enumerable
// own property (slimming the field audit key count back to baseline). The bag itself lives on a symbol key
// (non-enumerable) allocated lazily on first write to any optimizer field.
const kOpt = Symbol('connOptMoments');

export default class Connection {
  /** The source (pre-synaptic) node supplying activation. */
  from: Node;
  /** The target (post-synaptic) node receiving activation. */
  to: Node;
  /** Scalar multiplier applied to the source activation (prior to gain modulation). */
  weight: number;
  /** Standard eligibility trace (e.g., for RTRL / policy gradient credit assignment). */
  eligibility: number;
  /** Last applied delta weight (used by classic momentum). */
  previousDeltaWeight: number;
  /** Accumulated (batched) delta weight awaiting an apply step. */
  totalDeltaWeight: number;
  /** Extended trace structure for modulatory / eligibility propagation algorithms. Parallel arrays for cache-friendly iteration. */
  xtrace: { nodes: Node[]; values: number[] };
  /** Unique historical marking (auto-increment) for evolutionary alignment. */
  innovation: number;
  // enabled handled via bitfield (see _flags) exposed through accessor (enumerability removed for slimming)
  // --- Optimizer moment states (virtualized via symbol-backed bag + accessors) ---
  // NOTE: Accessor implementations below manage a lazily-created non-enumerable object containing:
  // { firstMoment, secondMoment, gradientAccumulator, maxSecondMoment, infinityNorm, secondMomentum, lookaheadShadowWeight }
  /**
   * Packed state flags (private for future-proofing hidden class):
   * bit0 => enabled gene expression (1 = active)
   * bit1 => DropConnect active mask (1 = not dropped this forward pass)
   * bit2 => hasGater (1 = symbol field present)
   * bits3+ reserved.
   */
  private _flags: number; // bit0 enabled, bit1 dcActive, bit2 hasGater

  /**
   * Construct a new connection between two nodes.
   *
   * @param from Source node.
   * @param to Target node.
   * @param weight Optional initial weight (default: small random in [-0.1, 0.1]).
   *
   * @example
   * const link = new Connection(nodeA, nodeB, 0.42);
   * console.log(link.weight); // 0.42
   * link.enabled = false;     // disable during mutation
   * link.enabled = true;      // re-enable later
   */
  constructor(from: Node, to: Node, weight?: number) {
    this.from = from;
    this.to = to;
    this.weight = weight ?? Math.random() * 0.2 - 0.1;
    // gater default: absent (bit2 clear, no symbol)
    this.eligibility = 0;

    // For tracking momentum
    this.previousDeltaWeight = 0;

    // Batch training
    this.totalDeltaWeight = 0;

    this.xtrace = {
      nodes: [],
      values: [],
    };

  // Optimizer moment fields left undefined until an optimizer that needs them runs (pay-for-use)
  // Initialize dropconnect mask
  // Bitfield initialization: enabled (bit0)=1, dropActive (bit1)=1
  this._flags = 0b11;
  this.innovation = Connection._nextInnovation++;
  }

  /**
   * Serialize to a minimal JSON-friendly shape (used for saving genomes / networks).
   * Undefined indices are preserved as `undefined` to allow later resolution / remapping.
   *
   * @returns Object with node indices, weight, gain, gater index (if any), innovation id & enabled flag.
   * @example
   * const json = connection.toJSON();
   * // => { from: 0, to: 3, weight: 0.12, gain: 1, innovation: 57, enabled: true }
   */
  toJSON() {
    const json: any = {
      from: this.from.index ?? undefined,
      to: this.to.index ?? undefined,
      weight: this.weight,
      gain: this.gain,
      innovation: this.innovation,
      enabled: this.enabled,
    };
    if ((this as any)._flags & 0b100) {
      const g = (this as any)[kGater];
      if (g && typeof g.index !== 'undefined') json.gater = g.index;
    }
    return json;
  }

  /**
   * Deterministic Cantor pairing function for a (sourceNodeId, targetNodeId) pair.
   * Useful when you want a stable innovation id without relying on global mutable counters
   * (e.g., for hashing or reproducible experiments).
   *
   * NOTE: For large indices this can overflow 53-bit safe integer space; keep node indices reasonable.
   *
   * @param sourceNodeId Source node integer id / index.
   * @param targetNodeId Target node integer id / index.
   * @returns Unique non-negative integer derived from the ordered pair.
   * @see https://en.wikipedia.org/wiki/Pairing_function
   * @example
   * const id = Connection.innovationID(2, 5); // deterministic
   */
  static innovationID(sourceNodeId: number, targetNodeId: number): number {
    return 0.5 * (sourceNodeId + targetNodeId) * (sourceNodeId + targetNodeId + 1) + targetNodeId;
  }
  private static _nextInnovation: number = 1;
  /**
   * Reset the monotonic auto-increment innovation counter (used for newly constructed / pooled instances).
   * You normally only call this at the start of an experiment or when deserializing a full population.
   *
   * @param value New starting value (default 1).
   * @example
   * Connection.resetInnovationCounter();     // back to 1
   * Connection.resetInnovationCounter(1000); // start counting from 1000
   */
  static resetInnovationCounter(value: number = 1) { Connection._nextInnovation = value; }

  // --- Simple object pool to reduce GC churn when connections are frequently created/removed ---
  private static _pool: Connection[] = [];
  /**
   * Acquire a `Connection` from the pool (or construct new). Fields are fully reset & given
   * a fresh sequential `innovation` id. Prefer this in evolutionary algorithms that mutate
   * topology frequently to reduce GC pressure.
   *
   * @param from Source node.
   * @param to Target node.
   * @param weight Optional initial weight.
   * @returns Reinitialized connection instance.
   * @example
   * const conn = Connection.acquire(a, b);
   * // ... use conn ...
   * Connection.release(conn); // when permanently removed
   */
  static acquire(from: Node, to: Node, weight?: number): Connection {
    let c: Connection;
    if (Connection._pool.length) {
      c = Connection._pool.pop()!;
      (c as any).from = from; (c as any).to = to; c.weight = weight ?? Math.random() * 0.2 - 0.1;
      if ((c as any)[kGain] !== undefined) delete (c as any)[kGain];
      if ((c as any)[kGater] !== undefined) delete (c as any)[kGater];
      c._flags = 0b11; // enabled + dcActive
      c.eligibility = 0; c.previousDeltaWeight = 0; c.totalDeltaWeight = 0;
      c.xtrace.nodes.length = 0; c.xtrace.values.length = 0;
  // Clear optimizer bag if present
  if ((c as any)[kOpt]) delete (c as any)[kOpt];
      (c as any).innovation = Connection._nextInnovation++;
    } else c = new Connection(from, to, weight);
    return c;
  }
  /**
   * Return a `Connection` to the internal pool for later reuse. Do NOT use the instance again
   * afterward unless re-acquired (treat as surrendered). Optimizer / trace fields are not
   * scrubbed here (they're overwritten during `acquire`).
   *
   * @param conn The connection instance to recycle.
   */
  static release(conn: Connection) { Connection._pool.push(conn); }
  /** Whether the gene (connection) is currently expressed (participates in forward pass). */
  get enabled(): boolean { return (this._flags & 0b1) !== 0; }
  set enabled(v: boolean) { this._flags = v ? (this._flags | 0b1) : (this._flags & ~0b1); }
  /** DropConnect active mask: 1 = not dropped (active), 0 = dropped for this stochastic pass. */
  get dcMask(): number { return (this._flags & 0b10) !== 0 ? 1 : 0; }
  set dcMask(v: number) { this._flags = v ? (this._flags | 0b10) : (this._flags & ~0b10); }
  /** Whether a gater node is assigned (modulates gain); true if the gater symbol field is present. */
  get hasGater(): boolean { return (this._flags & 0b100) !== 0; }

  // --- Virtualized gain property ---
  /**
   * Multiplicative modulation applied *after* weight. Default is `1` (neutral). We only store an
   * internal symbol-keyed property when the gain is non-neutral, reducing memory usage across
   * large populations where most connections are ungated.
   */
  get gain(): number { return (this as any)[kGain] === undefined ? 1 : (this as any)[kGain]; }
  set gain(v: number) {
    if (v === 1) {
      if ((this as any)[kGain] !== undefined) delete (this as any)[kGain];
    } else {
      (this as any)[kGain] = v;
    }
  }

  // --- Optimizer field accessors (prototype-level to avoid per-instance enumerable keys) ---
  private _ensureOptBag(): any {
    let bag = (this as any)[kOpt];
    if (!bag) {
      bag = {};
      (this as any)[kOpt] = bag; // symbol-keyed; non-enumerable
    }
    return bag;
  }
  private _getOpt<K extends keyof any>(k: string): number | undefined {
    const bag = (this as any)[kOpt];
    return bag ? bag[k] : undefined;
  }
  private _setOpt(k: string, v: number | undefined): void {
    if (v === undefined) {
      const bag = (this as any)[kOpt];
      if (bag) delete bag[k];
    } else {
      this._ensureOptBag()[k] = v;
    }
  }
  /** First moment estimate (Adam / AdamW) (was opt_m). */
  get firstMoment(): number | undefined { return this._getOpt('firstMoment'); }
  set firstMoment(v: number | undefined) { this._setOpt('firstMoment', v); }
  /** Second raw moment estimate (Adam family) (was opt_v). */
  get secondMoment(): number | undefined { return this._getOpt('secondMoment'); }
  set secondMoment(v: number | undefined) { this._setOpt('secondMoment', v); }
  /** Generic gradient accumulator (RMSProp / AdaGrad) (was opt_cache). */
  get gradientAccumulator(): number | undefined { return this._getOpt('gradientAccumulator'); }
  set gradientAccumulator(v: number | undefined) { this._setOpt('gradientAccumulator', v); }
  /** AMSGrad: Maximum of past second moment (was opt_vhat). */
  get maxSecondMoment(): number | undefined { return this._getOpt('maxSecondMoment'); }
  set maxSecondMoment(v: number | undefined) { this._setOpt('maxSecondMoment', v); }
  /** Adamax: Exponential moving infinity norm (was opt_u). */
  get infinityNorm(): number | undefined { return this._getOpt('infinityNorm'); }
  set infinityNorm(v: number | undefined) { this._setOpt('infinityNorm', v); }
  /** Secondary momentum (Lion variant) (was opt_m2). */
  get secondMomentum(): number | undefined { return this._getOpt('secondMomentum'); }
  set secondMomentum(v: number | undefined) { this._setOpt('secondMomentum', v); }
  /** Lookahead: shadow (slow) weight parameter (was _la_shadowWeight). */
  get lookaheadShadowWeight(): number | undefined { return this._getOpt('lookaheadShadowWeight'); }
  set lookaheadShadowWeight(v: number | undefined) { this._setOpt('lookaheadShadowWeight', v); }

  // --- Virtualized gater property (non-enumerable) ---
  /** Optional gating node whose activation can modulate effective weight (symbol-backed). */
  get gater(): Node | null { return (this._flags & 0b100) !== 0 ? (this as any)[kGater] : null; }
  set gater(node: Node | null) {
    if (node === null) {
      if ((this._flags & 0b100) !== 0) {
        this._flags &= ~0b100;
        if ((this as any)[kGater] !== undefined) delete (this as any)[kGater];
      }
    } else {
      (this as any)[kGater] = node;
      this._flags |= 0b100;
    }
  }

  // ---------------------------------------------------------------------------
  // Backward compatibility accessors for previously abbreviated property names
  // (opt_m, opt_v, opt_cache, opt_vhat, opt_u, opt_m2, _la_shadowWeight)
  // These keep external code & tests functioning while encouraging clearer names.
  // ---------------------------------------------------------------------------
  /** @deprecated Use firstMoment instead. */
  get opt_m(): number | undefined { return this.firstMoment; }
  set opt_m(v: number | undefined) { this.firstMoment = v; }
  /** @deprecated Use secondMoment instead. */
  get opt_v(): number | undefined { return this.secondMoment; }
  set opt_v(v: number | undefined) { this.secondMoment = v; }
  /** @deprecated Use gradientAccumulator instead. */
  get opt_cache(): number | undefined { return this.gradientAccumulator; }
  set opt_cache(v: number | undefined) { this.gradientAccumulator = v; }
  /** @deprecated Use maxSecondMoment instead. */
  get opt_vhat(): number | undefined { return this.maxSecondMoment; }
  set opt_vhat(v: number | undefined) { this.maxSecondMoment = v; }
  /** @deprecated Use infinityNorm instead. */
  get opt_u(): number | undefined { return this.infinityNorm; }
  set opt_u(v: number | undefined) { this.infinityNorm = v; }
  /** @deprecated Use secondMomentum instead. */
  get opt_m2(): number | undefined { return this.secondMomentum; }
  set opt_m2(v: number | undefined) { this.secondMomentum = v; }
  /** @deprecated Use lookaheadShadowWeight instead. */
  get _la_shadowWeight(): number | undefined { return this.lookaheadShadowWeight; }
  set _la_shadowWeight(v: number | undefined) { this.lookaheadShadowWeight = v; }

  /** Convenience alias for DropConnect mask with clearer naming. */
  get dropConnectActiveMask(): number { return this.dcMask; }
  set dropConnectActiveMask(v: number) { this.dcMask = v; }
}
