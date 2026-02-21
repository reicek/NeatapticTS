/**
 * Internal fast slab activation helpers extracted from network.slab.utils.ts.
 */
import type Network from '../../network';
import type Node from '../../node';
import { activationArrayPool } from '../../activationArrayPool';
import type {
  FastSlabNodeRuntime,
  NetworkActivationRuntime,
  NetworkSlabProps,
  NetworkTopoRuntime,
} from './network.slab.utils.types';

const ZERO = 0;
const ONE = 1;

/**
 * Predicate gating usage of high-performance slab forward pass.
 *
 * @param training - Whether caller is in training mode.
 * @returns True if fast path can be safely used.
 */
export function _canUseFastSlab(this: Network, training: boolean): boolean {
  // Step 1: Evaluate deterministic fast-path eligibility predicates.
  const internalNet = this as unknown as NetworkSlabProps;
  return !!(
    !training &&
    internalNet._enforceAcyclic &&
    !internalNet._topoDirty &&
    this.gates.length === 0 &&
    this.selfconns.length === 0 &&
    this.dropout === 0 &&
    internalNet._weightNoiseStd === 0 &&
    (internalNet._weightNoisePerHidden?.length || 0) === 0 &&
    (internalNet._stochasticDepth?.length || 0) === 0
  );
}

/**
 * Falls back to legacy activation when gating is present.
 *
 * @param network - Target network.
 * @param input - Activation input.
 * @returns Legacy output or null when fast path may continue.
 */
export function _tryFastSlabFallbackForGating(
  network: Network,
  input: number[],
): number[] | null {
  // Step 1: Return null when no gating exists.
  if (!network.gates || network.gates.length === ZERO) {
    return null;
  }
  // Step 2: Delegate to legacy activation path for gated networks.
  return _activateThroughLegacyPath(network, input);
}

/**
 * Falls back to legacy activation when slab prerequisites are missing.
 *
 * @param network - Target network.
 * @param internalNet - Internal slab runtime shape.
 * @param input - Activation input.
 * @returns Legacy output or null when fast path may continue.
 */
export function _tryFastSlabFallbackForMissingPrerequisites(
  network: Network,
  internalNet: NetworkSlabProps,
  input: number[],
): number[] | null {
  // Step 1: Continue fast path when all slab prerequisites are available.
  if (_hasFastSlabPrerequisites(internalNet)) {
    return null;
  }
  // Step 2: Delegate to legacy activation path when prerequisites are missing.
  return _activateThroughLegacyPath(network, input);
}

/**
 * Prepares topology and indices for fast slab pass.
 *
 * @param network - Target network.
 * @param internalNet - Internal slab runtime shape.
 * @param reindexNodes - Callback used to reindex nodes when needed.
 * @returns Nothing.
 */
export function _prepareFastSlabRuntime(
  network: Network,
  internalNet: NetworkSlabProps,
  reindexNodes: (network: Network) => void,
): void {
  // Step 1: Recompute topological order when it is marked dirty.
  if (internalNet._topoDirty) {
    _recomputeTopologyOrder(network);
  }
  // Step 2: Refresh node indices when structural changes occurred.
  if (internalNet._nodeIndexDirty) {
    reindexNodes(network);
  }
}

/**
 * Resolves topological iteration order for fast slab pass.
 *
 * @param network - Target network.
 * @param internalNet - Internal slab runtime shape.
 * @returns Topological node order.
 */
export function _resolveFastTopoOrder(
  network: Network,
  internalNet: NetworkSlabProps,
): FastSlabNodeRuntime[] {
  // Step 1: Prefer cached topological order and fall back to node order.
  return (internalNet._topoOrder || network.nodes) as FastSlabNodeRuntime[];
}

/**
 * Ensures fast activation/state buffers are allocated and shape-compatible.
 *
 * @param internalNet - Internal slab runtime shape.
 * @param nodeCount - Node count.
 * @returns Nothing.
 */
export function _ensureFastSlabBuffers(
  internalNet: NetworkSlabProps,
  nodeCount: number,
): void {
  // Step 1: Resolve activation precision for reusable working buffers.
  const useFloat32Activation = internalNet._activationPrecision === 'f32';
  if (
    _needsFastBufferReplacement(
      internalNet._fastA,
      nodeCount,
      useFloat32Activation,
    )
  ) {
    internalNet._fastA = _createFastActivationBuffer(
      useFloat32Activation,
      nodeCount,
    );
  }
  if (
    _needsFastBufferReplacement(
      internalNet._fastS,
      nodeCount,
      useFloat32Activation,
    )
  ) {
    internalNet._fastS = _createFastActivationBuffer(
      useFloat32Activation,
      nodeCount,
    );
  }
}

/**
 * Seeds input-layer activations for fast slab pass.
 *
 * @param network - Target network.
 * @param input - Activation input.
 * @param activationBuffer - Activation buffer.
 * @returns Nothing.
 */
export function _seedFastInputLayer(
  network: Network,
  input: number[],
  activationBuffer: Float32Array | Float64Array,
): void {
  // Step 1: Seed input activations and mirror runtime node fields.
  for (let inputIndex = ZERO; inputIndex < network.input; inputIndex++) {
    activationBuffer[inputIndex] = input[inputIndex];
    _writeInputNodeRuntime(network.nodes[inputIndex], input[inputIndex]);
  }
}

/**
 * Propagates activations through topology using slab arrays.
 *
 * @param network - Target network.
 * @param internalNet - Internal slab runtime shape.
 * @param topoOrder - Topological node order.
 * @param activationBuffer - Activation buffer.
 * @param stateBuffer - State buffer.
 * @returns Nothing.
 */
export function _propagateFastSlabActivations(
  network: Network,
  internalNet: NetworkSlabProps,
  topoOrder: FastSlabNodeRuntime[],
  activationBuffer: Float32Array | Float64Array,
  stateBuffer: Float32Array | Float64Array,
): void {
  // Step 1: Resolve packed slab references used during propagation.
  const weightArray = internalNet._connWeights as Float32Array | Float64Array;
  const toIndexArray = internalNet._connTo as Uint32Array;
  const outgoingOrder = internalNet._outOrder as Uint32Array;
  const outgoingStartIndices = internalNet._outStart as Uint32Array;

  // Step 2: Traverse nodes in topological order and fan out activations.
  for (let topoIndex = ZERO; topoIndex < topoOrder.length; topoIndex++) {
    const node = topoOrder[topoIndex];
    const nodeIndex = node.index >>> ZERO;
    _maybeActivateNonInputNode(
      network,
      node,
      nodeIndex,
      stateBuffer,
      activationBuffer,
    );
    _propagateNodeOutgoingEdges(
      internalNet,
      nodeIndex,
      activationBuffer,
      stateBuffer,
      weightArray,
      toIndexArray,
      outgoingOrder,
      outgoingStartIndices,
    );
  }
}

/**
 * Collects output activations into detached number array.
 *
 * @param network - Target network.
 * @param activationBuffer - Activation buffer.
 * @param nodeCount - Node count.
 * @returns Output activation array.
 */
export function _collectFastSlabOutput(
  network: Network,
  activationBuffer: Float32Array | Float64Array,
  nodeCount: number,
): number[] {
  // Step 1: Copy tail output activations into a pooled temporary array.
  const outputBaseIndex = nodeCount - network.output;
  const pooledOutputArray = activationArrayPool.acquire(
    network.output,
  ) as Float64Array;
  for (let outputOffset = ZERO; outputOffset < network.output; outputOffset++) {
    pooledOutputArray[outputOffset] =
      activationBuffer[outputBaseIndex + outputOffset];
  }
  // Step 2: Detach pooled values into a plain array and release pool slot.
  const output = Array.from(pooledOutputArray) as number[];
  activationArrayPool.release(pooledOutputArray);
  return output;
}

/**
 * Checks whether core slab prerequisites are available.
 *
 * @param internalNet - Internal slab runtime shape.
 * @returns True when all required slabs/adjacency arrays exist.
 */
function _hasFastSlabPrerequisites(internalNet: NetworkSlabProps): boolean {
  // Step 1: Verify all core slabs and adjacency slabs are present.
  return !!(
    internalNet._connWeights &&
    internalNet._connFrom &&
    internalNet._connTo &&
    internalNet._outStart &&
    internalNet._outOrder
  );
}

/**
 * Executes legacy network activation fallback.
 *
 * @param network - Target network.
 * @param input - Activation input.
 * @returns Legacy activation output.
 */
function _activateThroughLegacyPath(
  network: Network,
  input: number[],
): number[] {
  // Step 1: Delegate activation to legacy runtime path.
  return (network as unknown as Network & NetworkActivationRuntime).activate(
    input,
    false,
  );
}

/**
 * Recomputes topological order on demand.
 *
 * @param network - Target network.
 * @returns Nothing.
 */
function _recomputeTopologyOrder(network: Network): void {
  // Step 1: Invoke network-specific topology recomputation hook.
  const computeTopoOrder = (network as unknown as NetworkTopoRuntime)
    ._computeTopoOrder;
  computeTopoOrder.call(network);
}

/**
 * Checks whether a fast buffer requires replacement.
 *
 * @param buffer - Existing buffer.
 * @param nodeCount - Node count.
 * @param useFloat32Activation - True when 32-bit buffer is required.
 * @returns True when replacement is needed.
 */
function _needsFastBufferReplacement(
  buffer: Float32Array | Float64Array | undefined,
  nodeCount: number,
  useFloat32Activation: boolean,
): boolean {
  // Step 1: Replace when buffer is missing or length mismatches node count.
  if (!buffer || buffer.length !== nodeCount) {
    return true;
  }
  // Step 2: Replace when buffer precision does not match requested precision.
  if (useFloat32Activation && !(buffer instanceof Float32Array)) {
    return true;
  }
  if (!useFloat32Activation && !(buffer instanceof Float64Array)) {
    return true;
  }
  return false;
}

/**
 * Creates typed fast activation/state buffer.
 *
 * @param useFloat32Activation - True when 32-bit buffer is required.
 * @param nodeCount - Node count.
 * @returns New typed buffer.
 */
function _createFastActivationBuffer(
  useFloat32Activation: boolean,
  nodeCount: number,
): Float32Array | Float64Array {
  // Step 1: Allocate the requested typed buffer precision.
  if (useFloat32Activation) {
    return new Float32Array(nodeCount);
  }
  return new Float64Array(nodeCount);
}

/**
 * Writes runtime activation/state for one input node.
 *
 * @param node - Input node.
 * @param inputValue - Input activation value.
 * @returns Nothing.
 */
function _writeInputNodeRuntime(node: Node, inputValue: number): void {
  // Step 1: Mirror seeded input value into node runtime fields.
  const nodeRuntime = node as unknown as Node & {
    activation: number;
    state: number;
  };
  nodeRuntime.activation = inputValue;
  nodeRuntime.state = ZERO;
}

/**
 * Activates one non-input node when required.
 *
 * @param network - Target network.
 * @param node - Current node.
 * @param nodeIndex - Node index.
 * @param stateBuffer - State buffer.
 * @param activationBuffer - Activation buffer.
 * @returns Nothing.
 */
function _maybeActivateNonInputNode(
  network: Network,
  node: FastSlabNodeRuntime,
  nodeIndex: number,
  stateBuffer: Float32Array | Float64Array,
  activationBuffer: Float32Array | Float64Array,
): void {
  // Step 1: Skip activation for input-layer nodes.
  if (nodeIndex < network.input) {
    return;
  }

  // Step 2: Activate hidden/output node and store runtime values.
  const weightedSum = stateBuffer[nodeIndex] + node.bias;
  const activated = node.squash(weightedSum);
  node.state = stateBuffer[nodeIndex];
  node.activation = activated;
  activationBuffer[nodeIndex] = activated;
}

/**
 * Propagates one node activation over all outgoing slab edges.
 *
 * @param internalNet - Internal slab runtime shape.
 * @param nodeIndex - Source node index.
 * @param activationBuffer - Activation buffer.
 * @param stateBuffer - State buffer.
 * @param weightArray - Weight slab.
 * @param toIndexArray - Destination-index slab.
 * @param outgoingOrder - Outgoing edge order slab.
 * @param outgoingStartIndices - Outgoing start-offset slab.
 * @returns Nothing.
 */
function _propagateNodeOutgoingEdges(
  internalNet: NetworkSlabProps,
  nodeIndex: number,
  activationBuffer: Float32Array | Float64Array,
  stateBuffer: Float32Array | Float64Array,
  weightArray: Float32Array | Float64Array,
  toIndexArray: Uint32Array,
  outgoingOrder: Uint32Array,
  outgoingStartIndices: Uint32Array,
): void {
  // Step 1: Resolve outgoing adjacency window for the source node.
  const edgeStart = outgoingStartIndices[nodeIndex];
  const edgeEnd = outgoingStartIndices[nodeIndex + ONE];
  const sourceActivation = activationBuffer[nodeIndex];

  // Step 2: Accumulate weighted source contribution into destination states.
  for (let cursorIndex = edgeStart; cursorIndex < edgeEnd; cursorIndex++) {
    const connectionIndex = outgoingOrder[cursorIndex];
    const weightedValue = _resolveWeightedConnectionValue(
      internalNet,
      weightArray,
      connectionIndex,
    );
    stateBuffer[toIndexArray[connectionIndex]] +=
      sourceActivation * weightedValue;
  }
}

/**
 * Resolves effective connection weight including optional gain.
 *
 * @param internalNet - Internal slab runtime shape.
 * @param weightArray - Weight slab.
 * @param connectionIndex - Connection index.
 * @returns Effective weighted value.
 */
function _resolveWeightedConnectionValue(
  internalNet: NetworkSlabProps,
  weightArray: Float32Array | Float64Array,
  connectionIndex: number,
): number {
  // Step 1: Use raw weight when gain slab is omitted.
  const gainArray = internalNet._connGain;
  if (!gainArray) {
    return weightArray[connectionIndex];
  }
  // Step 2: Multiply weight by gain when gain slab exists.
  return weightArray[connectionIndex] * gainArray[connectionIndex];
}
