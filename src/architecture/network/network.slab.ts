import type Network from '../network';
import { activationArrayPool } from '../activationArrayPool';
import { config } from '../../config';

/**
 * Fast slab (structure-of-arrays) acceleration layer (Phase 3 foundation).
 * ----------------------------------------------------------------------
 * Motivation:
 *  Object graphs suffer from pointer chasing & polymorphic inline caches in large forward passes.
 *  By packing connection attributes into contiguous typed arrays we:
 *    - Improve spatial locality (sequential memory scans).
 *    - Enable simpler tight loops amenable to JIT & future WASM SIMD lowering.
 *    - Provide a staging ground for subsequent Phase 3+ memory slimming (flags / precision / plasticity).
 *
 * Phase 3 Additions (initial commit):
 *  - Slab version counter (_slabVersion) incremented on each structural rebuild (educational introspection).
 *  - Flags array (_connFlags Uint8Array) and gain array (_connGain Float32/64) allocated in parallel (placeholders
 *    for enabled bits, drop masks, future plasticity multipliers). Currently all flags=1, gains=1.
 *  - getConnectionSlab() now returns flags & gain alongside weights/from/to.
 *
 * Future (later Phase 3 iterations):
 *  - Geometric capacity growth (avoid realloc on small structural deltas). (Implemented: capacity reuse with growth factor)
 *  - Typed array pooling (current commit) to reuse large buffers and reduce GC churn on frequent rebuilds.
 *  - Plasticity / mask bits stored compactly (bitpacking) to reduce per-connection bytes.
 *  - Typed array pooling / recycling to limit GC churn on frequent rebuilds.
 *
 * Core Data Structures:
 *  weights (Float32Array|Float64Array)    : connection weights
 *  from    (Uint32Array)                  : source node indices
 *  to      (Uint32Array)                  : target node indices
 *  flags   (Uint8Array)                   : connection enable / mask bits (placeholder value=1)
 *  gain    (Float32Array|Float64Array)    : multiplicative gain (placeholder value=1)
 *  outStart (Uint32Array)                 : CSR row pointer style offsets (length nodeCount+1)
 *  outOrder (Uint32Array)                 : permutation of connection indices grouped by source
 *
 * Rebuild Workflow:
 *  1. Reindex nodes if dirty.
 *  2. Allocate typed arrays sized to current connection count.
 *  3. Populate parallel arrays in a single linear pass.
 *  4. Mark adjacency dirty; increment version.
 */

/**
 * (Re)build packed connection slabs (SoA layout) for fast, cache-friendly forward passes.
 *
 * Slab arrays:
 *  - weights: Float32/64 contiguous weights
 *  - from: Uint32 source node indices
 *  - to:   Uint32 target node indices
 *
 * These enable tight loops free of object indirection; we update only when structure/weights marked dirty.
 */
// Simple slab typed array pool keyed by element byte size + capacity for reuse.
// We intentionally keep a small LRU list per key to avoid unbounded retention.
const _slabArrayPool: Record<string, Array<TypedArray>> = Object.create(null);
type TypedArray =
  | Float32Array
  | Float64Array
  | Uint32Array
  | Uint8Array;
const SLAB_POOL_MAX_PER_KEY = 4; // small cap; configurable later.
const _slabAllocStats = { fresh: 0, pooled: 0 };
function _poolKey(kind: string, bytes: number, length: number) {
  return kind + ':' + bytes + ':' + length;
}
function _acquireTA(kind: string, ctor: any, length: number, bytesPerElement: number): TypedArray {
  if (!config.enableSlabArrayPooling) {
    _slabAllocStats.fresh++;
    return new ctor(length);
  }
  const key = _poolKey(kind, bytesPerElement, length);
  const list = _slabArrayPool[key];
  if (list && list.length) {
    _slabAllocStats.pooled++;
    return list.pop()! as TypedArray;
  }
  _slabAllocStats.fresh++;
  return new ctor(length);
}
function _releaseTA(kind: string, bytesPerElement: number, arr: TypedArray) {
  if (!config.enableSlabArrayPooling) return;
  const key = _poolKey(kind, bytesPerElement, arr.length);
  const list = (_slabArrayPool[key] ||= []);
  if (list.length < SLAB_POOL_MAX_PER_KEY) list.push(arr);
}

export function getSlabAllocationStats() {
  return { ..._slabAllocStats };
}

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
    internalNet._connGain = _acquireTA(
      'g',
      internalNet._useFloat32Weights ? Float32Array : Float64Array,
      capacity,
      internalNet._useFloat32Weights ? 4 : 8
    );
    internalNet._connCapacity = capacity;
  } else {
    capacity = internalNet._connCapacity; // reuse existing arrays (logical size may grow within capacity)
  }
  // Populate ONLY the active logical slice [0, connectionCount)
  const weightArray = internalNet._connWeights as Float32Array | Float64Array;
  const fromIndexArray = internalNet._connFrom as Uint32Array;
  const toIndexArray = internalNet._connTo as Uint32Array;
  const flagArray = internalNet._connFlags as Uint8Array;
  const gainArray = internalNet._connGain as Float32Array | Float64Array;
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
    const g = connection.gain;
    gainArray[connectionIndex] = g === 1 ? 1 : g;
  }
  // Optional: zero out tail of reused arrays (not strictly needed, left intact for potential diff debugging)
  internalNet._connCount = connectionCount; // record logical size
  internalNet._slabDirty = false;
  internalNet._adjDirty = true; // adjacency invalidated every structural rebuild
  internalNet._slabVersion = (internalNet._slabVersion || 0) + 1;
}

// Browser cooperative async rebuild (experimental). Splits large copy into microtasks to reduce jank.
/**
 * Cooperative asynchronous slab rebuild (Browser focused) that slices the packing work
 * across microtasks to mitigate long blocking copy loops for very large networks.
 *
 * Instrumentation / Phase 3 extensions:
 *  - Shares geometric capacity growth & typed array pooling with the synchronous path.
 *  - Increments internal _slabAsyncBuilds counter for memoryStats aggregation.
 *  - Updates shared allocation statistics (fresh vs pooled) only once per reallocation.
 *
 * NOTE: The async copy loop ONLY yields between chunk slices; typed array allocation
 * occurs up-front so allocStats.fresh should increase by exactly the number of newly
 * allocated slabs (5 arrays: weights/from/to/flags/gain) when capacity grows, and 0
 * when reusing existing capacity.
 */
export async function rebuildConnectionSlabAsync(this: Network, chunkSize = 50_000): Promise<void> {
  const internalNet = this as any;
  if (typeof window === 'undefined') return rebuildConnectionSlab.call(this, true);
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
    if (internalNet._connFrom) _releaseTA('f', 4, internalNet._connFrom as Uint32Array);
    if (internalNet._connTo) _releaseTA('t', 4, internalNet._connTo as Uint32Array);
    if (internalNet._connFlags) _releaseTA('fl', 1, internalNet._connFlags as Uint8Array);
    if (internalNet._connGain)
      _releaseTA(
        'g',
        internalNet._useFloat32Weights ? 4 : 8,
        internalNet._connGain as Float32Array | Float64Array
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
    internalNet._connCapacity = capacity;
  }
  const w = internalNet._connWeights as Float32Array | Float64Array;
  const fr = internalNet._connFrom as Uint32Array;
  const to = internalNet._connTo as Uint32Array;
  const fl = internalNet._connFlags as Uint8Array;
  const g = internalNet._connGain as Float32Array | Float64Array;
  // Adaptive chunk sizing: if very large and config specifies a target ms, reduce slice size conservatively.
  if (total > 200_000) {
    const target = config.browserSlabChunkTargetMs;
    if (typeof target === 'number' && target > 0) {
      // Heuristic: assume ~50k simple copy ops ~= 2-4ms on mid-tier hardware; scale linearly.
      const baseOpsPerMs = 15000; // coarse empirical constant; refine later.
      const estOps = Math.max(5_000, Math.min(50_000, Math.floor(baseOpsPerMs * target)));
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
      const c: any = this.connections[i];
      w[i] = c.weight;
      fr[i] = (c.from as any).index >>> 0;
      to[i] = (c.to as any).index >>> 0;
      fl[i] = c._flags & 0xff;
      const gn = c.gain;
      g[i] = gn === 1 ? 1 : gn;
    }
    idx = end;
    if (idx < total) await Promise.resolve(); // yield microtask
  }
  internalNet._connCount = total;
  internalNet._slabDirty = false;
  internalNet._adjDirty = true;
  internalNet._slabVersion = (internalNet._slabVersion || 0) + 1;
  internalNet._slabAsyncBuilds = (internalNet._slabAsyncBuilds || 0) + 1; // track async path usage
}

/** Return current slab (building lazily). */
export function getConnectionSlab(this: Network) {
  rebuildConnectionSlab.call(this); // Lazy rebuild if needed.
  const internalNet = this as any;
  return {
    weights: internalNet._connWeights!,
    from: internalNet._connFrom!,
    to: internalNet._connTo!,
    flags: internalNet._connFlags!,
    gain: internalNet._connGain!,
    version: internalNet._slabVersion || 0,
    used: internalNet._connCount || 0,
    capacity:
      internalNet._connCapacity ||
      (internalNet._connWeights && internalNet._connWeights.length) ||
      0,
  };
}

// Assign sequential indices (stable across slabs) to nodes.
function _reindexNodes(this: Network) {
  const internalNet = this as any;
  for (let nodeIndex = 0; nodeIndex < this.nodes.length; nodeIndex++)
    (this.nodes[nodeIndex] as any).index = nodeIndex;
  internalNet._nodeIndexDirty = false;
}

// Build CSR-like adjacency (outgoing edge index ranges) for fast propagation in slab mode.
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

// Eligibility conditions for fast slab path (must avoid scenarios needing per-edge dynamic behavior)
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
 * High-performance forward pass using packed slabs + CSR adjacency.
 * Falls back to generic activate if prerequisites unavailable.
 */
export function fastSlabActivate(this: Network, input: number[]): number[] {
  const internalNet = this as any;
  rebuildConnectionSlab.call(this); // Ensure slabs up-to-date (no-op if clean).
  if (internalNet._adjDirty) _buildAdjacency.call(this); // Build CSR adjacency if needed.
  // Gating incompatibility guard: if gating is present, always fallback to legacy path (dynamic per-edge behavior)
  if (this.gates && this.gates.length > 0) return (this as any).activate(input, false);
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
      stateBuffer[toIndexArray[connectionIndex]] +=
        sourceActivation * weightArray[connectionIndex];
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

/** Public helper: indicates whether fast slab path is currently viable. */
export function canUseFastSlab(this: Network, training: boolean) {
  return _canUseFastSlab.call(this, training);
}

/** Public accessor for current slab version (0 if never built). */
export function getSlabVersion(this: Network): number {
  return (this as any)._slabVersion || 0;
}
