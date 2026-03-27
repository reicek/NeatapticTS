/**
 * Core connection chapter for the architecture surface.
 *
 * This folder owns the library's directed edge primitive: the mutable link that
 * carries weight, gain modulation, gating metadata, eligibility traces,
 * optimizer scratch state, and evolutionary innovation bookkeeping between two
 * nodes.
 *
 * Read this chapter in three passes:
 *
 * 1. start with the `Connection` class overview to understand which edge fields
 *    are always present versus allocated lazily,
 * 2. continue to the serialization and innovation helpers when you need to see
 *    how a connection survives genome alignment, pooling, and persistence,
 * 3. finish with the virtualized accessors when you want the memory-shaping
 *    details behind gating, plasticity, and optimizer moments.
 */
import Node from '../node';

// Symbol used for optional gain storage (non-enumerable). Neutral gain=1 omitted entirely.
const kGain = Symbol('connGain');
// Symbol used for optional gater storage (non-enumerable). Presence tracked with bit2 flag.
const kGater = Symbol('connGater');
// Symbol-backed optimizer moment bag (amortizes 7 rarely-used numeric fields into a single optional object).
// Accessed via prototype accessors so assigning e.g. `conn.firstMoment = x` does NOT create an enumerable
// own property (slimming the field audit key count back to baseline). The bag itself lives on a symbol key
// (non-enumerable) allocated lazily on first write to an optimizer field.
const kOpt = Symbol('connOptMoments');
// Symbol used for optional plasticity learning rate (non-enumerable) (bit3 flag presence)
const kPlasticRate = Symbol('connPlasticRate');

/**
 * Internal interface for accessing symbol-keyed properties on Connection instances.
 * Used for type-safe access to dynamic symbol properties.
 */
interface ConnectionSymbolProps {
  [kGain]?: number;
  [kGater]?: Node;
  [kOpt]?: Record<string, number | undefined>;
  [kPlasticRate]?: number;
}

/**
 * Connection (Synapse / Edge)
 * ===========================
 * Directed weighted link between two nodes. The connection keeps the everyday
 * graph fields (`from`, `to`, `weight`, `innovation`) directly on the instance,
 * then pushes rarer capabilities behind symbol-backed accessors so large
 * populations do not pay object-shape costs for features they are not using.
 *
 * This makes the boundary useful in three different modes:
 *
 * - ordinary feed-forward links that only need endpoints and weight,
 * - gated or plastic links that gradually opt into extra runtime state,
 * - optimizer-heavy training paths that need moment buffers without turning
 *   every connection into a bloated record.
 *
 * @example
 * ```ts
 * const source = new Node('input');
 * const target = new Node('output');
 * const edge = new Connection(source, target, 0.42);
 *
 * edge.gain = 1.5;
 * edge.enabled = true;
 * ```
 */
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
   * bit3 => plastic (plasticityRate > 0)
   * bits4+ reserved.
   */
  private _flags: number;

  /**
   * Construct a new connection between two nodes.
   *
   * @param from Source node.
   * @param to Target node.
   * @param weight Optional initial weight (default: small random in [-0.1, 0.1]).
   * @returns A live connection instance that can participate in activation, mutation, and serialization flows.
   *
   * @example
   * ```ts
   * const link = new Connection(nodeA, nodeB, 0.42);
   * link.enabled = false;
   * link.enabled = true;
   * ```
   */
  constructor(from: Node, to: Node, weight?: number) {
    this.from = from;
    this.to = to;
    this.weight = weight ?? Math.random() * 0.2 - 0.1;
    this.eligibility = 0;
    this.previousDeltaWeight = 0;
    this.totalDeltaWeight = 0;
    this.xtrace = {
      nodes: [],
      values: [],
    };
    this._flags = 0b11;
    this.innovation = Connection._nextInnovation++;
  }

  /**
   * Serialize to a minimal JSON-friendly shape used by genome and network save flows.
   * Undefined node indices are preserved so callers can resolve or remap them later.
   *
   * @returns Object with node indices, weight, gain, innovation id, enabled flag, and gater index when one exists.
   * @example
   * ```ts
   * const json = connection.toJSON();
   * // => { from: 0, to: 3, weight: 0.12, gain: 1, innovation: 57, enabled: true }
   * ```
   */
  toJSON(): {
    from: number | undefined;
    to: number | undefined;
    weight: number;
    gain: number;
    innovation: number;
    enabled: boolean;
    gater?: number;
  } {
    const json: {
      from: number | undefined;
      to: number | undefined;
      weight: number;
      gain: number;
      innovation: number;
      enabled: boolean;
      gater?: number;
    } = {
      from: this.from.index ?? undefined,
      to: this.to.index ?? undefined,
      weight: this.weight,
      gain: this.gain,
      innovation: this.innovation,
      enabled: this.enabled,
    };
    if (this._flags & 0b100) {
      const gaterNode = (this as unknown as ConnectionSymbolProps)[kGater];
      if (gaterNode && typeof gaterNode.index !== 'undefined') {
        json.gater = gaterNode.index;
      }
    }
    return json;
  }

  /**
   * Deterministic Cantor pairing function for a `(sourceNodeId, targetNodeId)` pair.
   * Use it when you need a stable edge identifier without relying on the mutable
   * auto-increment counter.
   *
   * @param sourceNodeId Source node integer id or index.
   * @param targetNodeId Target node integer id or index.
   * @returns Unique non-negative integer derived from the ordered pair.
   * @see https://en.wikipedia.org/wiki/Pairing_function
   * @example
   * ```ts
   * const id = Connection.innovationID(2, 5);
   * ```
   */
  static innovationID(sourceNodeId: number, targetNodeId: number): number {
    return (
      0.5 * (sourceNodeId + targetNodeId) * (sourceNodeId + targetNodeId + 1) +
      targetNodeId
    );
  }

  private static _nextInnovation: number = 1;

  /**
   * Reset the monotonic innovation counter used for newly constructed or pooled connections.
   * You usually call this at the start of an experiment or before rebuilding a whole population.
   *
   * @param value New starting value.
   * @returns Nothing.
   */
  static resetInnovationCounter(value: number = 1) {
    Connection._nextInnovation = value;
  }

  private static _pool: Connection[] = [];

  /**
   * Acquire a connection from the internal pool, or construct a fresh one when the pool is empty.
   * This is the low-allocation path used by topology mutation and other edge-churn heavy flows.
   *
   * @param from Source node.
   * @param to Target node.
   * @param weight Optional initial weight.
   * @returns Reinitialized connection instance.
   */
  static acquire(from: Node, to: Node, weight?: number): Connection {
    let connectionInstance: Connection;
    if (Connection._pool.length) {
      connectionInstance = Connection._pool.pop()!;
      const symbolProps =
        connectionInstance as unknown as ConnectionSymbolProps;
      const mutableConnection = connectionInstance as unknown as {
        from: Node;
        to: Node;
        innovation: number;
      };
      mutableConnection.from = from;
      mutableConnection.to = to;
      connectionInstance.weight = weight ?? Math.random() * 0.2 - 0.1;
      if (symbolProps[kGain] !== undefined) delete symbolProps[kGain];
      if (symbolProps[kGater] !== undefined) delete symbolProps[kGater];
      connectionInstance._flags = 0b11;
      connectionInstance.eligibility = 0;
      connectionInstance.previousDeltaWeight = 0;
      connectionInstance.totalDeltaWeight = 0;
      connectionInstance.xtrace.nodes.length = 0;
      connectionInstance.xtrace.values.length = 0;
      if (symbolProps[kOpt]) delete symbolProps[kOpt];
      mutableConnection.innovation = Connection._nextInnovation++;
    } else {
      connectionInstance = new Connection(from, to, weight);
    }

    return connectionInstance;
  }

  /**
   * Return a connection instance to the internal pool for later reuse.
   * Treat the instance as surrendered after calling this method.
   *
   * @param conn The connection instance to recycle.
   * @returns Nothing.
   */
  static release(conn: Connection) {
    Connection._pool.push(conn);
  }

  /** Whether the gene is currently expressed and participates in the forward pass. */
  get enabled(): boolean {
    return (this._flags & 0b1) !== 0;
  }

  set enabled(isEnabled: boolean) {
    this._flags = isEnabled ? this._flags | 0b1 : this._flags & ~0b1;
  }

  /** DropConnect active mask: `1` means active for this stochastic pass, `0` means dropped. */
  get dcMask(): number {
    return (this._flags & 0b10) !== 0 ? 1 : 0;
  }

  set dcMask(maskValue: number) {
    this._flags = maskValue ? this._flags | 0b10 : this._flags & ~0b10;
  }

  /** Whether a gater node is assigned to modulate this connection's effective weight. */
  get hasGater(): boolean {
    return (this._flags & 0b100) !== 0;
  }

  /** Whether this connection participates in plastic adaptation. */
  get plastic(): boolean {
    return (this._flags & 0b1000) !== 0;
  }

  set plastic(isPlastic: boolean) {
    if (isPlastic) this._flags |= 0b1000;
    else this._flags &= ~0b1000;

    const symbolProps = this as unknown as ConnectionSymbolProps;
    if (!isPlastic && symbolProps[kPlasticRate] !== undefined) {
      delete symbolProps[kPlasticRate];
    }
  }

  /**
   * Multiplicative modulation applied after weight. Neutral gain `1` is omitted from storage.
   */
  get gain(): number {
    const symbolProps = this as unknown as ConnectionSymbolProps;
    return symbolProps[kGain] === undefined ? 1 : symbolProps[kGain];
  }

  set gain(gainValue: number) {
    const symbolProps = this as unknown as ConnectionSymbolProps;
    if (gainValue === 1) {
      if (symbolProps[kGain] !== undefined) delete symbolProps[kGain];
    } else {
      symbolProps[kGain] = gainValue;
    }
  }

  private _ensureOptBag(): Record<string, number | undefined> {
    const symbolProps = this as unknown as ConnectionSymbolProps;
    let bag = symbolProps[kOpt];
    if (!bag) {
      bag = {};
      symbolProps[kOpt] = bag;
    }
    return bag;
  }

  private _getOpt(optKey: string): number | undefined {
    const symbolProps = this as unknown as ConnectionSymbolProps;
    const bag = symbolProps[kOpt];
    return bag ? bag[optKey] : undefined;
  }

  private _setOpt(optKey: string, value: number | undefined): void {
    const symbolProps = this as unknown as ConnectionSymbolProps;
    if (value === undefined) {
      const bag = symbolProps[kOpt];
      if (bag) delete bag[optKey];
      return;
    }

    this._ensureOptBag()[optKey] = value;
  }

  /** First moment estimate used by Adam-family optimizers. */
  get firstMoment(): number | undefined {
    return this._getOpt('firstMoment');
  }

  set firstMoment(value: number | undefined) {
    this._setOpt('firstMoment', value);
  }

  /** Second raw moment estimate used by Adam-family optimizers. */
  get secondMoment(): number | undefined {
    return this._getOpt('secondMoment');
  }

  set secondMoment(value: number | undefined) {
    this._setOpt('secondMoment', value);
  }

  /** Generic gradient accumulator used by RMSProp and AdaGrad. */
  get gradientAccumulator(): number | undefined {
    return this._getOpt('gradientAccumulator');
  }

  set gradientAccumulator(value: number | undefined) {
    this._setOpt('gradientAccumulator', value);
  }

  /** AMSGrad maximum of past second-moment estimates. */
  get maxSecondMoment(): number | undefined {
    return this._getOpt('maxSecondMoment');
  }

  set maxSecondMoment(value: number | undefined) {
    this._setOpt('maxSecondMoment', value);
  }

  /** Adamax infinity norm accumulator. */
  get infinityNorm(): number | undefined {
    return this._getOpt('infinityNorm');
  }

  set infinityNorm(value: number | undefined) {
    this._setOpt('infinityNorm', value);
  }

  /** Secondary momentum buffer used by Lion-style updates. */
  get secondMomentum(): number | undefined {
    return this._getOpt('secondMomentum');
  }

  set secondMomentum(value: number | undefined) {
    this._setOpt('secondMomentum', value);
  }

  /** Lookahead slow-weight snapshot. */
  get lookaheadShadowWeight(): number | undefined {
    return this._getOpt('lookaheadShadowWeight');
  }

  set lookaheadShadowWeight(value: number | undefined) {
    this._setOpt('lookaheadShadowWeight', value);
  }

  /** Optional gating node whose activation modulates effective weight. */
  get gater(): Node | null {
    const symbolProps = this as unknown as ConnectionSymbolProps;
    return (this._flags & 0b100) !== 0 ? (symbolProps[kGater] ?? null) : null;
  }

  set gater(node: Node | null) {
    const symbolProps = this as unknown as ConnectionSymbolProps;
    if (node === null) {
      if ((this._flags & 0b100) !== 0) {
        this._flags &= ~0b100;
        if (symbolProps[kGater] !== undefined) delete symbolProps[kGater];
      }
      return;
    }

    symbolProps[kGater] = node;
    this._flags |= 0b100;
  }

  /** Per-connection plasticity rate. `0` means the connection is not plastic. */
  get plasticityRate(): number {
    const symbolProps = this as unknown as ConnectionSymbolProps;
    return symbolProps[kPlasticRate] === undefined
      ? 0
      : symbolProps[kPlasticRate];
  }

  set plasticityRate(value: number) {
    const symbolProps = this as unknown as ConnectionSymbolProps;
    if (value === undefined || value === 0) {
      if (symbolProps[kPlasticRate] !== undefined)
        delete symbolProps[kPlasticRate];
      this._flags &= ~0b1000;
      return;
    }

    symbolProps[kPlasticRate] = value;
    this._flags |= 0b1000;
  }

  /** Convenience alias for DropConnect mask with clearer naming. */
  get dropConnectActiveMask(): number {
    return this.dcMask;
  }

  set dropConnectActiveMask(value: number) {
    this.dcMask = value;
  }
}
