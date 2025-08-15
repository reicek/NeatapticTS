import type Network from '../network';
import { activationArrayPool } from '../activationArrayPool';

/**
 * Fast slab (structure-of-arrays) acceleration layer.
 *
 * Rationale:
 *  Typical neural network graphs represented as object graphs incur significant overhead during
 *  forward passes due to pointer chasing (cache misses) and dynamic property lookups. For large
 *  evolving populations where topologies change infrequently compared to evaluation frequency,
 *  we can amortize a one-off packing cost into contiguous typed arrays, dramatically improving
 *  memory locality and enabling tight inner loops.
 *
 * Core Data Structures:
 *  - weightArray     (Float32Array|Float64Array): connection weights
 *  - fromIndexArray  (Uint32Array): source node indices per connection
 *  - toIndexArray    (Uint32Array): destination node indices per connection
 *  - outgoingStartIndices (Uint32Array length = nodeCount + 1): CSR row pointer style offsets
 *  - outgoingOrder   (Uint32Array): permutation of connection indices grouped by source node
 *
 * Workflow:
 *  1. rebuildConnectionSlab packs connections into SoA arrays when dirty.
 *  2. _buildAdjacency converts fromIndexArray into CSR-like adjacency for each source node.
 *  3. fastSlabActivate uses the packed arrays + precomputed topological order to perform a forward pass
 *     with minimal branching and object access.
 *
 * Constraints for Fast Path (_canUseFastSlab):
 *  - Acyclic enforced (no recurrence) so single topological sweep suffices.
 *  - No gating, self-connections, dropout, stochastic depth, or per-hidden noise.
 *  - Topological order and node indices must be clean.
 *
 * Dirty Flags Touched:
 *  - _slabDirty: slab arrays need rebuild
 *  - _adjDirty: adjacency mapping (CSR) invalid
 *  - _nodeIndexDirty: node.index values invalid
 *  - _topoDirty: topological ordering invalid
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
export function rebuildConnectionSlab(this: Network, force = false): void {
  const internalNet = this as any;
  if (!force && !internalNet._slabDirty) return; // Already current; avoid reallocation churn.
  if (internalNet._nodeIndexDirty) _reindexNodes.call(this); // Ensure node.index stable before packing.
  /** Total number of forward connections to pack. */
  const connectionCount = this.connections.length;
  /** Contiguous weight buffer matching connection order. */
  const weightArray = internalNet._useFloat32Weights
    ? new Float32Array(connectionCount)
    : new Float64Array(connectionCount);
  /** Source node indices per connection (parallel to weightArray). */
  const fromIndexArray = new Uint32Array(connectionCount);
  /** Target node indices per connection (parallel to weightArray). */
  const toIndexArray = new Uint32Array(connectionCount);
  for (
    let connectionIndex = 0;
    connectionIndex < connectionCount;
    connectionIndex++
  ) {
    /** Original connection object (read-only during packing). */
    const connection = this.connections[connectionIndex];
    weightArray[connectionIndex] = connection.weight; // Snapshot weight (mutations will mark dirty next time).
    fromIndexArray[connectionIndex] = (connection.from as any).index >>> 0;
    toIndexArray[connectionIndex] = (connection.to as any).index >>> 0;
  }
  internalNet._connWeights = weightArray;
  internalNet._connFrom = fromIndexArray;
  internalNet._connTo = toIndexArray;
  internalNet._slabDirty = false;
  internalNet._adjDirty = true; // CSR adjacency invalidated by rebuild.
}

/** Return current slab (building lazily). */
export function getConnectionSlab(this: Network) {
  rebuildConnectionSlab.call(this); // Lazy rebuild if needed.
  const internalNet = this as any;
  return {
    weights: internalNet._connWeights!,
    from: internalNet._connFrom!,
    to: internalNet._connTo!,
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
  /** Number of packed connections. */
  const connectionCount = internalNet._connFrom.length;
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
