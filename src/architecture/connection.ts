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
 * Typical Lifecycle
 * ```ts
 * // Construct directly (simplest)
 * const conn = new Connection(nodeA, nodeB, 0.5);
 * conn.weight *= 1.1;                  // adjust weight
 * conn.gain = 0.8;                     // set gating gain manually (rare)
 * conn.enabled = false;                // temporarily disable gene expression (NEAT mutation)
 * conn.enabled = true;                 // re‑enable
 *
 * // OR use the pool for performance in algorithms that create/destroy many links
 * const pooled = Connection.acquire(nodeA, nodeB); // random small initial weight
 * // ... use pooled connection in a network draft ...
 * Connection.release(pooled); // recycle when removed from topology
 * ```
 *
 * Gating vs Gain
 * ---------------
 * A connection's effective strength during forward propagation is: `effective = weight * gain`.
 * By default `gain = 1`. If a `gater` node is assigned elsewhere in the library, `gain` may be
 * updated dynamically each tick using the gater's activation. To minimize object shape bloat, a
 * neutral gain of `1` is *not stored*; only non‑neutral gains materialize an internal symbol field.
 *
 * Innovation Numbers
 * ------------------
 * NEAT-style historical markings let crossover align homologous genes. We provide both:
 * - `Connection.innovationID(sourceId, targetId)` – deterministic Cantor pairing for a pair of node indices
 * - Sequential auto-increment (`innovation` instance field) – simpler uniqueness for pooled instances
 *
 * Serialization
 * -------------
 * The `toJSON()` method exposes a concise representation suitable for saving / exchanging genomes.
 *
 * Performance Note
 * ----------------
 * Most numeric fields are plain JS numbers for JIT friendliness. Avoid adding ad-hoc properties at
 * runtime (it would deopt hidden classes). Use provided accessors for flags & gain.
 */
import Node from './node'; // Import Node type

// Symbol used for optional gain storage (non-enumerable). Neutral gain=1 omitted entirely.
const kGain = Symbol('connGain');

export default class Connection {
  /** The source (pre-synaptic) node supplying activation. */
  from: Node;
  /** The target (post-synaptic) node receiving activation. */
  to: Node;
  /** Scalar multiplier applied to the source activation (prior to gain modulation). */
  weight: number;
  /** Optional gating node that modulates the connection's gain (handled externally). */
  gater: Node | null;
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
  // --- Optimizer moment states (allocated lazily when an optimizer uses them) ---
  /** First moment estimate (Adam / AdamW) (was opt_m). */
  firstMoment?: number;
  /** Second raw moment estimate (Adam family) (was opt_v). */
  secondMoment?: number;
  /** Generic gradient accumulator (RMSProp / AdaGrad) (was opt_cache). */
  gradientAccumulator?: number;
  /** AMSGrad: Maximum of past second moment (was opt_vhat). */
  maxSecondMoment?: number;
  /** Adamax: Exponential moving infinity norm (was opt_u). */
  infinityNorm?: number;
  /** Secondary momentum (Lion variant) (was opt_m2). */
  secondMomentum?: number;
  /** Lookahead: shadow (slow) weight parameter (was _la_shadowWeight). */
  lookaheadShadowWeight?: number;
  /**
   * Packed state flags (private for future-proofing hidden class):
   * bit0 => enabled gene expression (1 = active)
   * bit1 => DropConnect active mask (1 = not dropped this forward pass)
   */
  private _flags: number; // bit0: enabled, bit1: dc active

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
    this.gater = null;
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
    if (this.gater && typeof this.gater.index !== 'undefined') {
      json.gater = this.gater.index;
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
    let connection: Connection;
    if (Connection._pool.length) {
      connection = Connection._pool.pop()!;
      // Reset core references
      (connection as any).from = from;
      (connection as any).to = to;
      connection.weight = weight ?? Math.random() * 0.2 - 0.1;
      if ((connection as any)[kGain] !== undefined) delete (connection as any)[kGain]; // neutral gain
      connection.gater = null;
      connection.eligibility = 0;
      connection.previousDeltaWeight = 0;
      connection.totalDeltaWeight = 0;
      connection.xtrace.nodes.length = 0;
      connection.xtrace.values.length = 0;
      // Optimizer fields intentionally left undefined (pay-for-use)
      connection._flags = 0b11; // enabled + active
  connection.lookaheadShadowWeight = undefined;
      (connection as any).innovation = Connection._nextInnovation++;
    } else {
      connection = new Connection(from, to, weight);
    }
    return connection;
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
