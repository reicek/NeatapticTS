import type Network from '../network';
import { activationArrayPool } from '../activationArrayPool';
import { config } from '../../config';

/**
 * Slab Packing / Structure‑of‑Arrays Backend (Educational Module)
 * ==============================================================
 * Packs per‑connection data into parallel typed arrays (SoA) to accelerate
 * forward passes and to illustrate memory/layout optimizations.
 *
 * Why SoA?
 *  - Locality & fewer cache misses.
 *  - Predictable tight numeric loops (JIT / SIMD friendly).
 *  - Easy instrumentation (single contiguous blocks to measure & diff).
 *
 * Key Arrays (logical length = `used`): weights | from | to | flags | (optional) gain | (optional) plastic.
 * Adjacency (CSR style): outStart (nodeCount+1), outOrder (per‑source permutation) enabling fast fan‑out.
 *
 * On‑Demand & Omission:
 *  - Gain/plastic slabs allocated only when a non‑neutral value appears; freed if neutrality returns.
 *  - `getConnectionSlab()` synthesizes a neutral gain view if omitted internally (keeps teaching tools simple).
 *
 * Capacity Strategy: geometric growth (1.25x browser / 1.75x Node) amortizes realloc cost.
 * Pooling (config gated) reuses typed arrays (see `getSlabAllocationStats`).
 *
 * Rebuild Steps (sync): reindex nodes → grow/allocate if needed → single pass populate → optional slabs → version++.
 * Async variant slices the population loop into microtasks to reduce long main‑thread blocks.
 *
 * Example (inspection):
 * ```ts
 * const slab = (net as any).getConnectionSlab();
 * console.log('Edges', slab.used, 'Version', slab.version, 'Cap', slab.capacity);
 * console.log('First weight from->to', slab.weights[0], slab.from[0], slab.to[0]);
 * ```
 */

/**
 * Internal typed array pool keyed by a composite string (kind:bytes:length).
 * Acts as a tiny per-key LRU (stack discipline via push/pop) to reuse large
 * slabs when geometric growth forces reallocation. Pooling is gated by
 * `config.enableSlabArrayPooling` so benchmarks / tests can measure both
 * allocation churn scenarios. Arrays are only retained up to the cap returned
 * by `_slabPoolCap()` (default 4) which empirically captures most reuse wins
 * while tightly bounding retained memory. DO NOT mutate arrays pulled from
 * here outside the intended lifecycle (acquire -> use -> _releaseTA).
 */
const _slabArrayPool: Record<string, Array<TypedArray>> = Object.create(null);
/**
 * Per-pool-key allocation & reuse counters (educational / diagnostics).
 * Tracks how many slabs were freshly created vs reused plus the high‑water
 * mark (maxRetained) of simultaneously retained arrays for the key. Exposed
 * indirectly via `getSlabAllocationStats()` so users can introspect the
 * effectiveness of pooling under their workload.
 */
interface PoolKeyMetrics {
  created: number;
  reused: number;
  maxRetained: number;
}
/**
 * Map backing metrics storage. Keys align 1:1 with `_slabArrayPool` entries.
 */
const _slabPoolMetrics: Record<string, PoolKeyMetrics> = Object.create(null);
/**
 * Union of slab typed array element container types. We purposefully restrict
 * to the specific constructors actually used by this module so TypeScript can
 * narrow accurately and editors display concise hover info.
 */
type TypedArray = Float32Array | Float64Array | Uint32Array | Uint8Array;
/**
 * Compute the effective per‑key retention cap for slab pooling.
 *
 * RATIONALE
 * ---------
 * The default (4) was selected after observing diminishing reuse gains beyond the
 * 3rd–4th cached buffer in mutation / prune churn micro‑benchmarks; larger caps
 * produced a higher long‑tail of retained bytes with negligible hit‑rate benefit.
 *
 * CONFIG
 * ------
 * Users can override via `config.slabPoolMaxPerKey`:
 *   undefined → default 4
 *   0         → keep metrics but do not retain slabs (max reuse pressure scenario)
 *   <0        → coerced to 0 (safety)
 *
 * @returns Integer retention cap (≥0).
 */
function _slabPoolCap(): number {
  const configuredCap = config.slabPoolMaxPerKey;
  if (configuredCap === undefined) return 4;
  return configuredCap < 0 ? 0 : configuredCap | 0; // coerce to int, clamp at 0
}
/**
 * Global allocation counters since process start / last manual reset:
 *  - fresh: number of newly constructed typed arrays (misses or pooling disabled)
 *  - pooled: number of arrays satisfied from the reuse pool.
 * Used (with per-key metrics) to evaluate memory reuse efficiency.
 */
const _slabAllocStats = { fresh: 0, pooled: 0 };
/**
 * Construct a unique pool key encoding kind + element byte size + logical length.
 * This granularity prevents mismatched reuse (different lengths / element sizes).
 * @param kind Short discriminator (e.g. 'w','f','t','fl','g','p').
 * @param bytes Bytes per element (1,4,8).
 * @param length Typed array length.
 * @returns Stable string key used in pool maps.
 */
function _poolKey(kind: string, bytes: number, length: number) {
  return kind + ':' + bytes + ':' + length;
}
/**
 * Acquire (or reuse) a typed array slab, updating allocation statistics.
 *
 * Behaviour:
 *  - Pooling disabled: always allocate fresh.
 *  - Pooling enabled: reuse last retained array for identical key if present.
 *  - Metrics updated (fresh/pooled + per-key created/reused counters).
 *
 * @param kind Pool discriminator (see `_poolKey`).
 * @param ctor Typed array constructor.
 * @param length Desired element count.
 * @param bytesPerElement Byte width used to form key (guards reuse correctness).
 * @returns The acquired typed array (possibly recycled).
 */
function _acquireTA(
  kind: string,
  ctor: any,
  length: number,
  bytesPerElement: number
): TypedArray {
  if (!config.enableSlabArrayPooling) {
    _slabAllocStats.fresh++;
    return new ctor(length);
  }
  const key = _poolKey(kind, bytesPerElement, length);
  const list = _slabArrayPool[key];
  if (list && list.length) {
    _slabAllocStats.pooled++;
    (_slabPoolMetrics[key] ||= { created: 0, reused: 0, maxRetained: 0 })
      .reused++;
    return list.pop()! as TypedArray;
  }
  _slabAllocStats.fresh++;
  (_slabPoolMetrics[key] ||= { created: 0, reused: 0, maxRetained: 0 })
    .created++;
  return new ctor(length);
}
/**
 * Return a typed array slab to the per‑key bounded pool.
 * No-op if pooling disabled. Pool functions as small LRU (push/pop).
 * @param kind Pool discriminator.
 * @param bytesPerElement Byte width for key regeneration.
 * @param arr The typed array instance to consider retaining.
 */
function _releaseTA(kind: string, bytesPerElement: number, arr: TypedArray) {
  if (!config.enableSlabArrayPooling) return;
  const key = _poolKey(kind, bytesPerElement, arr.length);
  const list = (_slabArrayPool[key] ||= []);
  if (list.length < _slabPoolCap()) list.push(arr);
  const m = (_slabPoolMetrics[key] ||= {
    created: 0,
    reused: 0,
    maxRetained: 0,
  });
  if (list.length > m.maxRetained) m.maxRetained = list.length;
}

/**
 * Allocation statistics snapshot for slab typed arrays.
 *
 * Includes:
 *  - fresh: number of newly constructed typed arrays since process start / metrics reset.
 *  - pooled: number of arrays served from the pool.
 *  - pool: per‑key metrics (created, reused, maxRetained) for educational inspection.
 *
 * NOTE: Stats are cumulative (not auto‑reset); callers may diff successive snapshots.
 * @returns Plain object copy (safe to serialize) of current allocator counters.
 */
export function getSlabAllocationStats() {
  return { ..._slabAllocStats, pool: Object.assign({}, _slabPoolMetrics) };
}

/**
 * Build (or refresh) the packed connection slabs for the network synchronously.
 *
 * ACTIONS
 * -------
 * 1. Optionally reindex nodes if structural mutations invalidated indices.
 * 2. Grow (geometric) or reuse existing typed arrays to ensure capacity >= active connections.
 * 3. Populate the logical slice [0, connectionCount) with weight/from/to/flag data.
 * 4. Lazily allocate gain & plastic slabs only on first non‑neutral / plastic encounter; omit otherwise.
 * 5. Release previously allocated optional slabs when they revert to neutral / unused (omission optimization).
 * 6. Update internal bookkeeping: logical count, dirty flags, version counter.
 *
 * PERFORMANCE
 * -----------
 * O(C) over active connections with amortized allocation cost due to geometric growth.
 *
 * @param force When true forces rebuild even if network not marked dirty (useful for timing tests).
 */
export function rebuildConnectionSlab(this: Network, force = false): void {
  const internalNet = this as any;
  if (!force && !internalNet._slabDirty) return; // Already current; avoid reallocation churn.
  if (internalNet._nodeIndexDirty) _reindexNodes.call(this); // Ensure node.index stable before packing.
  /** Active connection count requiring packing (logical size). */
  const connectionCount = this.connections.length;
  /** Previous reserved capacity (physical typed array length). */
  let capacity: number = internalNet._connCapacity || 0;
  const growthFactor = typeof window === 'undefined' ? 1.75 : 1.25; // Node larger growth; Browser smaller increments.
  // Decide if we must (re)allocate (first build OR insufficient capacity after structural change).
  const needAllocate = capacity < connectionCount;
  if (needAllocate) {
    // Geometric growth: enlarge to at least connectionCount, else grow previous capacity * factor until sufficient.
    capacity =
      capacity === 0 ? Math.ceil(connectionCount * growthFactor) : capacity;
    while (capacity < connectionCount)
      capacity = Math.ceil(capacity * growthFactor);
    // Allocate fresh parallel arrays sized to new capacity.
    // Release old slabs back to pool (if present)
    if (internalNet._connWeights)
      _releaseTA(
        'w',
        internalNet._useFloat32Weights ? 4 : 8,
        internalNet._connWeights
      );
    if (internalNet._connFrom)
      _releaseTA('f', 4, internalNet._connFrom as Uint32Array);
    if (internalNet._connTo)
      _releaseTA('t', 4, internalNet._connTo as Uint32Array);
    if (internalNet._connFlags)
      _releaseTA('fl', 1, internalNet._connFlags as Uint8Array);
    if (internalNet._connGain)
      _releaseTA(
        'g',
        internalNet._useFloat32Weights ? 4 : 8,
        internalNet._connGain as Float32Array | Float64Array
      );
    if (internalNet._connPlastic)
      _releaseTA(
        'p',
        internalNet._useFloat32Weights ? 4 : 8,
        internalNet._connPlastic as Float32Array | Float64Array
      );
    // Acquire (possibly pooled) slabs with new capacity
    internalNet._connWeights = _acquireTA(
      'w',
      internalNet._useFloat32Weights ? Float32Array : Float64Array,
      capacity,
      internalNet._useFloat32Weights ? 4 : 8
    );
    internalNet._connFrom = _acquireTA('f', Uint32Array, capacity, 4);
    internalNet._connTo = _acquireTA('t', Uint32Array, capacity, 4);
    internalNet._connFlags = _acquireTA('fl', Uint8Array, capacity, 1);
    // Gain slab now allocated lazily (gain omission optimization); set null placeholder
    internalNet._connGain = null;
    // Plasticity slab allocated lazily later IF any connection sets plastic flag
    internalNet._connPlastic = null;
    internalNet._connCapacity = capacity;
  } else {
    capacity = internalNet._connCapacity; // reuse existing arrays (logical size may grow within capacity)
  }
  // Populate ONLY the active logical slice [0, connectionCount)
  const weightArray = internalNet._connWeights as Float32Array | Float64Array;
  const fromIndexArray = internalNet._connFrom as Uint32Array;
  const toIndexArray = internalNet._connTo as Uint32Array;
  const flagArray = internalNet._connFlags as Uint8Array;
  let gainArray = internalNet._connGain as Float32Array | Float64Array | null;
  let anyNonNeutralGain = false;
  let plasticArray = internalNet._connPlastic as
    | Float32Array
    | Float64Array
    | null;
  let anyPlastic = false;
  for (
    let connectionIndex = 0;
    connectionIndex < connectionCount;
    connectionIndex++
  ) {
    const connection: any = this.connections[connectionIndex];
    weightArray[connectionIndex] = connection.weight;
    fromIndexArray[connectionIndex] = (connection.from as any).index >>> 0;
    toIndexArray[connectionIndex] = (connection.to as any).index >>> 0;
    // Bit-pack enabled/dropConnect/gater bits directly from Connection._flags (bit0 enabled, bit1 dcMask, bit2 hasGater)
    // Future bits (plasticity, freeze, mutation lineage) can be OR'ed here with documented positions.
    flagArray[connectionIndex] = (connection as any)._flags & 0xff; // mask to one byte
    // Gain: if virtualized gain !== 1 we snapshot it, else store 1 (keeps forward math branch-free)
    const gainValue = connection.gain;
    if (gainValue !== 1) {
      if (!gainArray) {
        gainArray = _acquireTA(
          'g',
          internalNet._useFloat32Weights ? Float32Array : Float64Array,
          capacity,
          internalNet._useFloat32Weights ? 4 : 8
        ) as any;
        internalNet._connGain = gainArray;
        for (let j = 0; j < connectionIndex; j++) (gainArray as any)[j] = 1;
      }
      (gainArray as any)[connectionIndex] = gainValue;
      anyNonNeutralGain = true;
    } else if (gainArray) {
      (gainArray as any)[connectionIndex] = 1;
    }
    if ((connection as any)._flags & 0b1000) anyPlastic = true;
  }
  // Omission optimization: if we allocated a gain array but all gains were neutral revert to null (tests for omission expect this)
  if (!anyNonNeutralGain && gainArray) {
    _releaseTA(
      'g',
      internalNet._useFloat32Weights ? 4 : 8,
      gainArray as Float32Array | Float64Array
    );
    internalNet._connGain = null;
  }
  if (anyPlastic && !plasticArray) {
    // allocate plastic slab & second pass fill
    plasticArray = _acquireTA(
      'p',
      internalNet._useFloat32Weights ? Float32Array : Float64Array,
      capacity,
      internalNet._useFloat32Weights ? 4 : 8
    ) as any;
    internalNet._connPlastic = plasticArray;
    for (let i = 0; i < connectionCount; i++) {
      const c: any = this.connections[i];
      plasticArray![i] = (c as any).plasticityRate || 0;
    }
  } else if (!anyPlastic && plasticArray) {
    // release existing plastic array if no longer needed
    _releaseTA(
      'p',
      internalNet._useFloat32Weights ? 4 : 8,
      plasticArray as Float32Array | Float64Array
    );
    internalNet._connPlastic = null;
  }
  // Optional: zero out tail of reused arrays (not strictly needed, left intact for potential diff debugging)
  internalNet._connCount = connectionCount; // record logical size
  internalNet._slabDirty = false;
  internalNet._adjDirty = true; // adjacency invalidated every structural rebuild
  internalNet._slabVersion = (internalNet._slabVersion || 0) + 1;
}

/**
 * Cooperative asynchronous slab rebuild (Browser only).
 *
 * Strategy:
 *  - Perform capacity decision + allocation up front (mirrors sync path).
 *  - Populate connection data in microtask slices (yield via resolved Promise) to avoid long main‑thread stalls.
 *  - Adaptive slice sizing for very large graphs if `config.browserSlabChunkTargetMs` set.
 *
 * Metrics: Increments `_slabAsyncBuilds` for observability.
 * Fallback: On Node (no `window`) defers to synchronous rebuild for simplicity.
 *
 * @param chunkSize Initial maximum connections per slice (may be reduced adaptively for huge graphs).
 * @returns Promise resolving once rebuild completes.
 */
export async function rebuildConnectionSlabAsync(
  this: Network,
  chunkSize = 50_000
): Promise<void> {
  const internalNet = this as any;
  if (typeof window === 'undefined')
    return rebuildConnectionSlab.call(this, true);
  if (!internalNet._slabDirty) return; // already clean
  if (internalNet._nodeIndexDirty) _reindexNodes.call(this);
  const total = this.connections.length;
  let capacity: number = internalNet._connCapacity || 0;
  const growthFactor = 1.25;
  if (capacity < total) {
    // Geometric growth mirroring sync path behavior
    capacity = capacity === 0 ? Math.ceil(total * growthFactor) : capacity;
    while (capacity < total) capacity = Math.ceil(capacity * growthFactor);
    // Release old slabs (if any) back to pool for reuse elsewhere
    if (internalNet._connWeights)
      _releaseTA(
        'w',
        internalNet._useFloat32Weights ? 4 : 8,
        internalNet._connWeights
      );
    if (internalNet._connFrom)
      _releaseTA('f', 4, internalNet._connFrom as Uint32Array);
    if (internalNet._connTo)
      _releaseTA('t', 4, internalNet._connTo as Uint32Array);
    if (internalNet._connFlags)
      _releaseTA('fl', 1, internalNet._connFlags as Uint8Array);
    if (internalNet._connGain)
      _releaseTA(
        'g',
        internalNet._useFloat32Weights ? 4 : 8,
        internalNet._connGain as Float32Array | Float64Array
      );
    if (internalNet._connPlastic)
      _releaseTA(
        'p',
        internalNet._useFloat32Weights ? 4 : 8,
        internalNet._connPlastic as Float32Array | Float64Array
      );
    // Acquire slabs (pooled or fresh) so allocation stats reflect this async path too
    internalNet._connWeights = _acquireTA(
      'w',
      internalNet._useFloat32Weights ? Float32Array : Float64Array,
      capacity,
      internalNet._useFloat32Weights ? 4 : 8
    );
    internalNet._connFrom = _acquireTA('f', Uint32Array, capacity, 4);
    internalNet._connTo = _acquireTA('t', Uint32Array, capacity, 4);
    internalNet._connFlags = _acquireTA('fl', Uint8Array, capacity, 1);
    internalNet._connGain = _acquireTA(
      'g',
      internalNet._useFloat32Weights ? Float32Array : Float64Array,
      capacity,
      internalNet._useFloat32Weights ? 4 : 8
    );
    internalNet._connPlastic = null;
    internalNet._connCapacity = capacity;
  }
  const weights = internalNet._connWeights as Float32Array | Float64Array;
  const fromIndices = internalNet._connFrom as Uint32Array;
  const toIndices = internalNet._connTo as Uint32Array;
  const flagBytes = internalNet._connFlags as Uint8Array;
  const gainArray = internalNet._connGain as Float32Array | Float64Array | null;
  let anyNonNeutralGain = false;
  let plasticArray = internalNet._connPlastic as
    | Float32Array
    | Float64Array
    | null;
  let anyPlastic = false;
  // Adaptive chunk sizing: if very large and config specifies a target ms, reduce slice size conservatively.
  if (total > 200_000) {
    const target = config.browserSlabChunkTargetMs;
    if (typeof target === 'number' && target > 0) {
      // Heuristic: assume ~50k simple copy ops ~= 2-4ms on mid-tier hardware; scale linearly.
      const baseOpsPerMs = 15000; // coarse empirical constant; refine later.
      const estOps = Math.max(
        5_000,
        Math.min(50_000, Math.floor(baseOpsPerMs * target))
      );
      chunkSize = Math.min(chunkSize, estOps);
    } else {
      // No target; still clamp to 50k to avoid pathological large default if caller passed bigger.
      chunkSize = Math.min(chunkSize, 50_000);
    }
  }
  let idx = 0;
  while (idx < total) {
    const end = Math.min(total, idx + chunkSize);
    for (let i = idx; i < end; i++) {
      const connection: any = this.connections[i];
      weights[i] = connection.weight;
      fromIndices[i] = (connection.from as any).index >>> 0;
      toIndices[i] = (connection.to as any).index >>> 0;
      flagBytes[i] = connection._flags & 0xff;
      const gainValue = connection.gain;
      if (gainArray) gainArray[i] = gainValue === 1 ? 1 : gainValue;
      if (gainValue !== 1) anyNonNeutralGain = true;
      if (connection._flags & 0b1000) anyPlastic = true;
    }
    idx = end;
    if (idx < total) await Promise.resolve(); // yield microtask
  }
  if (!anyNonNeutralGain && gainArray) {
    // Release neutral gain slab to honor omission policy
    _releaseTA(
      'g',
      internalNet._useFloat32Weights ? 4 : 8,
      gainArray as Float32Array | Float64Array
    );
    internalNet._connGain = null;
  }
  if (anyPlastic && !plasticArray) {
    plasticArray = _acquireTA(
      'p',
      internalNet._useFloat32Weights ? Float32Array : Float64Array,
      internalNet._connCapacity,
      internalNet._useFloat32Weights ? 4 : 8
    ) as any;
    internalNet._connPlastic = plasticArray;
    for (let i = 0; i < total; i++)
      (plasticArray as any)[i] =
        (this.connections[i] as any).plasticityRate || 0;
  } else if (!anyPlastic && plasticArray) {
    _releaseTA(
      'p',
      internalNet._useFloat32Weights ? 4 : 8,
      plasticArray as Float32Array | Float64Array
    );
    internalNet._connPlastic = null;
  }
  internalNet._connCount = total;
  internalNet._slabDirty = false;
  internalNet._adjDirty = true;
  internalNet._slabVersion = (internalNet._slabVersion || 0) + 1;
  internalNet._slabAsyncBuilds = (internalNet._slabAsyncBuilds || 0) + 1; // track async path usage
}

/**
 * Obtain (and lazily rebuild if dirty) the current packed SoA view of connections.
 *
 * Gain Omission: If the internal gain slab is absent (all gains neutral) a synthetic
 * neutral array is created and returned (NOT retained) to keep external educational
 * tooling branch‑free while preserving omission memory savings internally.
 *
 * @returns Read‑only style view (do not mutate) containing typed arrays + metadata.
 */
export function getConnectionSlab(this: Network): ConnectionSlabView {
  rebuildConnectionSlab.call(this); // Lazy rebuild if needed.
  const internalNet = this as any;
  let gain: Float32Array | Float64Array | null = internalNet._connGain || null;
  if (!gain) {
    // Provide a synthetic neutral gain view for educational/tests expecting parity while preserving omission semantics.
    const cap =
      internalNet._connCapacity ||
      (internalNet._connWeights && internalNet._connWeights.length) ||
      0;
    gain = internalNet._useFloat32Weights
      ? new Float32Array(cap)
      : new Float64Array(cap);
    for (let i = 0; i < (internalNet._connCount || 0); i++) gain[i] = 1;
  }
  return {
    weights: internalNet._connWeights!,
    from: internalNet._connFrom!,
    to: internalNet._connTo!,
    flags: internalNet._connFlags!,
    gain,
    plastic: internalNet._connPlastic || null,
    version: internalNet._slabVersion || 0,
    used: internalNet._connCount || 0,
    capacity:
      internalNet._connCapacity ||
      (internalNet._connWeights && internalNet._connWeights.length) ||
      0,
  };
}

/**
 * Assign sequential indices to each node (stable ordering prerequisite for slab packing).
 * Clears `_nodeIndexDirty` flag.
 */
function _reindexNodes(this: Network) {
  const internalNet = this as any;
  for (let nodeIndex = 0; nodeIndex < this.nodes.length; nodeIndex++)
    (this.nodes[nodeIndex] as any).index = nodeIndex;
  internalNet._nodeIndexDirty = false;
}

/**
 * Build / refresh CSR‑style adjacency (outStart + outOrder) enabling fast fan‑out traversal.
 * Only rebuilds when marked dirty. Stores arrays on internal network instance.
 */
function _buildAdjacency(this: Network) {
  const internalNet = this as any;
  if (!internalNet._connFrom || !internalNet._connTo) return; // Nothing to build yet.
  /** Number of nodes in current network. */
  const nodeCount = this.nodes.length;
  /** Number of packed (active logical) connections. */
  const connectionCount =
    internalNet._connCount ?? internalNet._connFrom.length;
  /** Fan-out counts per source node (populated first pass). */
  const fanOutCounts = new Uint32Array(nodeCount);
  for (
    let connectionIndex = 0;
    connectionIndex < connectionCount;
    connectionIndex++
  ) {
    fanOutCounts[internalNet._connFrom[connectionIndex]]++; // Tally outgoing edges per source.
  }
  /** CSR row pointer style start indices (length = nodeCount + 1). */
  const outgoingStartIndices = new Uint32Array(nodeCount + 1);
  /** Running offset while computing prefix sum of fanOutCounts. */
  let runningOffset = 0;
  for (let nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++) {
    outgoingStartIndices[nodeIndex] = runningOffset;
    runningOffset += fanOutCounts[nodeIndex];
  }
  outgoingStartIndices[nodeCount] = runningOffset; // Sentinel (total connections).
  /** Permutation of connection indices grouped by source for contiguous traversal. */
  const outgoingOrder = new Uint32Array(connectionCount);
  /** Working cursor array (clone) used to place each connection into its slot. */
  const insertionCursor = outgoingStartIndices.slice();
  for (
    let connectionIndex = 0;
    connectionIndex < connectionCount;
    connectionIndex++
  ) {
    const fromNodeIndex = internalNet._connFrom[connectionIndex];
    outgoingOrder[insertionCursor[fromNodeIndex]++] = connectionIndex;
  }
  internalNet._outStart = outgoingStartIndices;
  internalNet._outOrder = outgoingOrder;
  internalNet._adjDirty = false;
}

/**
 * Predicate gating usage of high‑performance slab forward pass.
 * Disallows training / stochastic / dynamic edge behaviours (gating, dropout, noise, self‑connections).
 * @param training Whether caller is in training mode (disables fast path for gradient/time reasons).
 * @returns True if fast path can be safely used for deterministic forward activation.
 */
function _canUseFastSlab(this: Network, training: boolean): boolean {
  const internalNet = this as any;
  return (
    !training && // Training may require gradients / noise injection.
    internalNet._enforceAcyclic && // Must have acyclic guarantee for single forward sweep.
    !internalNet._topoDirty && // Topological order must be current.
    this.gates.length === 0 && // Gating implies dynamic per-edge behavior.
    this.selfconns.length === 0 && // Self connections require recurrent handling.
    this.dropout === 0 && // Dropout introduces stochastic masking.
    internalNet._weightNoiseStd === 0 && // Global weight noise disables deterministic slab pass.
    internalNet._weightNoisePerHidden.length === 0 && // Per hidden noise variants.
    internalNet._stochasticDepth.length === 0 // Layer drop also stochastic.
  );
}

/**
 * High‑performance forward pass using packed slabs + CSR adjacency.
 *
 * Fallback Conditions (auto‑detected):
 *  - Missing slabs / adjacency structures.
 *  - Topology/gating/stochastic predicates fail (see `_canUseFastSlab`).
 *  - Any gating present (explicit guard).
 *
 * Implementation Notes:
 *  - Reuses internal activation/state buffers to reduce per‑step allocation churn.
 *  - Applies gain multiplication if optional gain slab exists.
 *  - Assumes acyclic graph; topological order recomputed on demand if marked dirty.
 *
 * @param input Input vector (length must equal `network.input`).
 * @returns Output activations (detached plain array) of length `network.output`.
 */
export function fastSlabActivate(this: Network, input: number[]): number[] {
  const internalNet = this as any;
  rebuildConnectionSlab.call(this); // Ensure slabs up-to-date (no-op if clean).
  if (internalNet._adjDirty) _buildAdjacency.call(this); // Build CSR adjacency if needed.
  // Gating incompatibility guard: if gating is present, always fallback to legacy path (dynamic per-edge behavior)
  if (this.gates && this.gates.length > 0)
    return (this as any).activate(input, false);
  if (
    !internalNet._connWeights ||
    !internalNet._connFrom ||
    !internalNet._connTo ||
    !internalNet._outStart ||
    !internalNet._outOrder
  ) {
    return (this as any).activate(input, false); // Fallback: prerequisites missing.
  }
  if (internalNet._topoDirty) (this as any)._computeTopoOrder();
  if (internalNet._nodeIndexDirty) _reindexNodes.call(this);
  /** Topologically sorted nodes (or original order if already acyclic & clean). */
  const topoOrder = internalNet._topoOrder || this.nodes;
  /** Total node count. */
  const nodeCount = this.nodes.length;
  /** Whether to store activations in 32-bit for memory/bandwidth or 64-bit for precision. */
  const useFloat32Activation = internalNet._activationPrecision === 'f32';
  // Allocate / reuse activation & state typed arrays (avoid reallocating each forward pass).
  if (
    !internalNet._fastA ||
    internalNet._fastA.length !== nodeCount ||
    (useFloat32Activation && !(internalNet._fastA instanceof Float32Array)) ||
    (!useFloat32Activation && !(internalNet._fastA instanceof Float64Array))
  ) {
    internalNet._fastA = useFloat32Activation
      ? new Float32Array(nodeCount)
      : new Float64Array(nodeCount);
  }
  if (
    !internalNet._fastS ||
    internalNet._fastS.length !== nodeCount ||
    (useFloat32Activation && !(internalNet._fastS instanceof Float32Array)) ||
    (!useFloat32Activation && !(internalNet._fastS instanceof Float64Array))
  ) {
    internalNet._fastS = useFloat32Activation
      ? new Float32Array(nodeCount)
      : new Float64Array(nodeCount);
  }
  /** Activation buffer (post-squash outputs). */
  const activationBuffer = internalNet._fastA as Float32Array | Float64Array;
  /** Pre-activation sum buffer (accumulates weighted inputs). */
  const stateBuffer = internalNet._fastS as Float32Array | Float64Array;
  stateBuffer.fill(0);
  // Seed input activations directly (no accumulation for inputs).
  for (let inputIndex = 0; inputIndex < this.input; inputIndex++) {
    activationBuffer[inputIndex] = input[inputIndex];
    (this.nodes[inputIndex] as any).activation = input[inputIndex];
    (this.nodes[inputIndex] as any).state = 0;
  }
  /** Packed connection weights. */
  const weightArray = internalNet._connWeights;
  /** Packed destination node indices per connection. */
  const toIndexArray = internalNet._connTo;
  /** Connection index order grouped by source (CSR style). */
  const outgoingOrder = internalNet._outOrder;
  /** Row pointer style start offsets for each source node. */
  const outgoingStartIndices = internalNet._outStart;
  // Iterate nodes in topological order, computing activations then streaming contributions forward.
  for (let topoIdx = 0; topoIdx < topoOrder.length; topoIdx++) {
    const node: any = topoOrder[topoIdx];
    const nodeIndex = node.index >>> 0;
    if (nodeIndex >= this.input) {
      /** Weighted input sum plus bias. */
      const weightedSum = stateBuffer[nodeIndex] + node.bias;
      /** Activated output via node's squash function. */
      const activated = node.squash(weightedSum);
      node.state = stateBuffer[nodeIndex];
      node.activation = activated;
      activationBuffer[nodeIndex] = activated;
    }
    // Propagate activation along outgoing edges.
    const edgeStart = outgoingStartIndices[nodeIndex];
    const edgeEnd = outgoingStartIndices[nodeIndex + 1];
    const sourceActivation = activationBuffer[nodeIndex];
    for (let cursorIdx = edgeStart; cursorIdx < edgeEnd; cursorIdx++) {
      const connectionIndex = outgoingOrder[cursorIdx];
      let w = weightArray[connectionIndex];
      const gainArr = internalNet._connGain;
      if (gainArr) w *= gainArr[connectionIndex];
      stateBuffer[toIndexArray[connectionIndex]] += sourceActivation * w;
    }
  }
  // Collect outputs: final output nodes occupy the tail of the node list.
  const outputBaseIndex = nodeCount - this.output;
  const pooledOutputArray = activationArrayPool.acquire(this.output);
  for (let outputOffset = 0; outputOffset < this.output; outputOffset++) {
    (pooledOutputArray as any)[outputOffset] =
      activationBuffer[outputBaseIndex + outputOffset];
  }
  const result = Array.from(pooledOutputArray as any) as number[]; // Detach buffer into regular array.
  activationArrayPool.release(pooledOutputArray);
  return result;
}

/**
 * Public convenience wrapper exposing fast path eligibility.
 * Mirrors `_canUseFastSlab` internal predicate.
 * @param training Whether caller is performing training (disables fast path if true).
 * @returns True when slab fast path predicates hold.
 */
export function canUseFastSlab(this: Network, training: boolean) {
  return _canUseFastSlab.call(this, training);
}

/**
 * Retrieve current monotonic slab version (increments on each successful rebuild).
 * @returns Non‑negative integer (0 if slab never built yet).
 */
export function getSlabVersion(this: Network): number {
  return (this as any)._slabVersion || 0;
}

/**
 * Shape returned by `getConnectionSlab()` describing the packed SoA view.
 * Note: The arrays SHOULD NOT be mutated by callers; treat as read‑only.
 */
export interface ConnectionSlabView {
  /** Packed connection weights (length >= used; logical slice = used). */
  weights: Float32Array | Float64Array;
  /** Source node indices per connection. */
  from: Uint32Array;
  /** Target node indices per connection. */
  to: Uint32Array;
  /** Bitfield flags per connection (see module header). */
  flags: Uint8Array;
  /** Gain array (synthetic neutral array if omission optimization active). */
  gain: Float32Array | Float64Array | null;
  /** Plasticity rate array (null if no plastic connections). */
  plastic: Float32Array | Float64Array | null;
  /** Monotonic rebuild counter (0 if never built). */
  version: number;
  /** Logical number of active connections packed into the leading slice of arrays. */
  used: number;
  /** Physical capacity (allocated length) of the parallel arrays. */
  capacity: number;
}
