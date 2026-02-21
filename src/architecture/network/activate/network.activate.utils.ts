import type Network from '../../network';
import { activationArrayPool } from '../../activationArrayPool';
import type { ActivateNetworkInternals as NetworkInternals } from '../network.types';

/**
 * Perform a forward pass without creating or updating any training / gradient traces.
 *
 * This is the most allocation‑sensitive activation path. Internally it will attempt
 * to leverage a compact "fast slab" routine (an optimized, vectorized broadcast over
 * contiguous activation buffers) when the Network instance indicates that such a path
 * is currently valid. If that attempt fails (for instance because the slab is stale
 * after a structural mutation) execution gracefully falls back to a node‑by‑node loop.
 *
 * Algorithm outline:
 *  1. (Optional) Refresh cached topological order if the network enforces acyclicity
 *     and a structural change marked the order as dirty.
 *  2. Validate the input dimensionality.
 *  3. Try the fast slab path; if it throws, continue with the standard path.
 *  4. Acquire a pooled output buffer sized to the number of output neurons.
 *  5. Iterate all nodes in their internal order:
 *       - Input nodes: directly assign provided input values.
 *       - Hidden nodes: compute activation via Node.noTraceActivate (no bookkeeping).
 *       - Output nodes: compute activation and store it (in sequence) inside the
 *         pooled output buffer.
 *  6. Copy the pooled buffer into a fresh array (detaches user from the pool) and
 *     release the pooled buffer back to the pool.
 *
 * Complexity considerations:
 *  - Time: O(N + E) where N = number of nodes, E = number of inbound edges processed
 *    inside each Node.noTraceActivate call (not explicit here but inside the node).
 *  - Space: O(O) transient (O = number of outputs) due to the pooled output buffer.
 *
 * @param this - Bound {@link Network} instance.
 * @param input - Flat numeric vector whose length must equal network.input.
 * @returns Array of output neuron activations (length == network.output).
 * @throws {Error} If the provided input vector length mismatches the network's input size.
 * @example
 * const out = net.noTraceActivate([0.1, 0.2, 0.3]);
 * console.log(out); // => e.g. [0.5123, 0.0441]
 * @remarks Safe for inference hot paths; not suitable when gradients / training traces are required.
 */
export function noTraceActivate(this: Network, input: number[]): number[] {
  const networkInternal = this as unknown as NetworkInternals;

  // Step 1: Keep ordering guarantees current when acyclic constraints are enabled.
  refreshTopologicalOrderWhenRequired(networkInternal);

  // Step 2: Fail fast on invalid dimensionality.
  assertInputMatchesNetworkInputSize(input, this.input);

  // Step 3: Opportunistically use fast slab inference when available.
  const fastSlabResult = tryActivateWithFastSlab(networkInternal, input);
  if (fastSlabResult !== null) return fastSlabResult;

  // Step 4: Fall back to deterministic node-by-node activation.
  return activateWithoutTraceUsingNodeIteration(this, input);

  /**
   * Refresh the cached topological order if network topology changed while acyclic mode is active.
   *
   * @param internalState - Runtime network internals used by activation pipeline.
   * @returns Nothing.
   */
  function refreshTopologicalOrderWhenRequired(
    internalState: NetworkInternals,
  ): void {
    if (internalState._enforceAcyclic && internalState._topoDirty)
      internalState._computeTopoOrder();
  }

  /**
   * Ensure provided input vector has expected dimensionality.
   *
   * @param inputVector - Candidate activation input vector.
   * @param expectedInputSize - Network input dimensionality.
   * @returns Nothing.
   */
  function assertInputMatchesNetworkInputSize(
    inputVector: number[],
    expectedInputSize: number,
  ): void {
    if (
      !Array.isArray(inputVector) ||
      inputVector.length !== expectedInputSize
    ) {
      throw new Error(
        `Input size mismatch: expected ${expectedInputSize}, got ${
          inputVector ? inputVector.length : 'undefined'
        }`,
      );
    }
  }

  /**
   * Attempt vectorized fast slab activation and gracefully fall back on failure.
   *
   * @param internalState - Runtime network internals used by activation pipeline.
   * @param inputVector - Validated activation input vector.
   * @returns Fast slab output when successful, otherwise null.
   */
  function tryActivateWithFastSlab(
    internalState: NetworkInternals,
    inputVector: number[],
  ): number[] | null {
    if (!internalState._canUseFastSlab(false)) return null;

    try {
      return internalState._fastSlabActivate(inputVector);
    } catch {
      return null;
    }
  }

  /**
   * Execute inference by iterating nodes and collecting output activations in pooled storage.
   *
   * @param network - Network instance owning nodes and output shape.
   * @param inputVector - Validated activation input vector.
   * @returns Detached output activation vector.
   */
  function activateWithoutTraceUsingNodeIteration(
    network: Network,
    inputVector: number[],
  ): number[] {
    const pooledOutputBuffer = activationArrayPool.acquire(network.output);

    try {
      populatePooledOutputBufferFromNodes(
        network.nodes,
        inputVector,
        pooledOutputBuffer,
      );
      return detachPooledOutputBuffer(pooledOutputBuffer);
    } finally {
      activationArrayPool.release(pooledOutputBuffer);
    }
  }

  /**
   * Visit each node in order and write output-neuron activations into pooled output storage.
   *
   * @param networkNodes - Network nodes in activation order.
   * @param inputVector - Validated activation input vector.
   * @param pooledOutputBuffer - Mutable pooled output storage.
   * @returns Nothing.
   */
  function populatePooledOutputBufferFromNodes(
    networkNodes: Network['nodes'],
    inputVector: number[],
    pooledOutputBuffer: ReturnType<typeof activationArrayPool.acquire>,
  ): void {
    let outputWriteIndex = 0;

    networkNodes.forEach((networkNode, nodeIndex) => {
      outputWriteIndex = activateSingleNodeWithoutTrace(
        networkNode,
        nodeIndex,
        inputVector,
        pooledOutputBuffer,
        outputWriteIndex,
      );
    });
  }

  /**
   * Activate a single node according to its role and update output write position if needed.
   *
   * @param networkNode - Node to activate.
   * @param nodeIndex - Current node index in network traversal order.
   * @param inputVector - Validated activation input vector.
   * @param pooledOutputBuffer - Mutable pooled output storage.
   * @param outputWriteIndex - Current output buffer write position.
   * @returns Next output write position.
   */
  function activateSingleNodeWithoutTrace(
    networkNode: Network['nodes'][number],
    nodeIndex: number,
    inputVector: number[],
    pooledOutputBuffer: ReturnType<typeof activationArrayPool.acquire>,
    outputWriteIndex: number,
  ): number {
    if (networkNode.type === 'input') {
      networkNode.noTraceActivate(inputVector[nodeIndex]);
      return outputWriteIndex;
    }

    if (networkNode.type === 'output') {
      pooledOutputBuffer[outputWriteIndex] = networkNode.noTraceActivate();
      return outputWriteIndex + 1;
    }

    networkNode.noTraceActivate();
    return outputWriteIndex;
  }

  /**
   * Clone pooled output storage into a detached plain array.
   *
   * @param pooledOutputBuffer - Pooled activation output storage.
   * @returns Detached output activation vector.
   */
  function detachPooledOutputBuffer(
    pooledOutputBuffer: ReturnType<typeof activationArrayPool.acquire>,
  ): number[] {
    return Array.from(pooledOutputBuffer) as number[];
  }
}

/**
 * Thin semantic alias to the network's main activation path.
 *
 * At present this simply forwards to {@link Network.activate}. The indirection is useful for:
 *  - Future differentiation between raw (immediate) activation and a mode that performs reuse /
 *    staged batching logic.
 *  - Providing a stable exported symbol for external tooling / instrumentation.
 *
 * @param this - Bound {@link Network} instance.
 * @param input - Input vector (length == network.input).
 * @param training - Whether to retain training traces / gradients (delegated downstream).
 * @param maxActivationDepth - Guard against runaway recursion / cyclic activation attempts.
 * @returns Implementation-defined result of Network.activate (typically an output vector).
 * @example
 * const y = net.activateRaw([0,1,0]);
 * @remarks Keep this wrapper lightweight; heavy logic should live inside Network.activate itself.
 */
export function activateRaw(
  this: Network,
  input: number[],
  training = false,
  maxActivationDepth = 1000,
): number[] {
  const networkInternal = this as unknown as NetworkInternals;

  // Keep orchestrator declarative by delegating activation dispatch decision.
  return activateWithSelectedReusePath(
    networkInternal,
    input,
    training,
    maxActivationDepth,
  );

  /**
   * Select activation path based on reuse configuration.
   *
   * @param internalState - Runtime network internals used by activation pipeline.
   * @param inputVector - Input activation vector.
   * @param isTraining - Whether activation should retain training traces.
   * @param maximumActivationDepth - Guard against runaway activation depth.
   * @returns Activation output vector.
   */
  function activateWithSelectedReusePath(
    internalState: NetworkInternals,
    inputVector: number[],
    isTraining: boolean,
    maximumActivationDepth: number,
  ): number[] {
    if (internalState._reuseActivationArrays)
      return activateViaNetworkDelegate(
        internalState,
        inputVector,
        isTraining,
        maximumActivationDepth,
      );

    return activateViaNetworkDelegate(
      internalState,
      inputVector,
      isTraining,
      maximumActivationDepth,
    );
  }

  /**
   * Delegate activation to the network's core activation implementation.
   *
   * @param internalState - Runtime network internals used by activation pipeline.
   * @param inputVector - Input activation vector.
   * @param isTraining - Whether activation should retain training traces.
   * @param maximumActivationDepth - Guard against runaway activation depth.
   * @returns Activation output vector.
   */
  function activateViaNetworkDelegate(
    internalState: NetworkInternals,
    inputVector: number[],
    isTraining: boolean,
    maximumActivationDepth: number,
  ): number[] {
    return internalState.activate(
      inputVector,
      isTraining,
      maximumActivationDepth,
    );
  }
}

/**
 * Activate the network over a mini‑batch (array) of input vectors, returning a 2‑D array of outputs.
 *
 * This helper simply loops, invoking {@link Network.activate} (or its bound variant) for each
 * sample. It is intentionally naive: no attempt is made to fuse operations across the batch.
 * For very large batch sizes or performance‑critical paths consider implementing a custom
 * vectorized backend that exploits SIMD, GPU kernels, or parallel workers.
 *
 * Input validation occurs per row to surface the earliest mismatch with a descriptive index.
 *
 * @param this - Bound {@link Network} instance.
 * @param inputs - Array of input vectors; each must have length == network.input.
 * @param training - Whether each activation should keep training traces.
 * @returns 2‑D array: outputs[i] is the activation result for inputs[i].
 * @throws {Error} If inputs is not an array, or any contained vector has an incorrect length.
 * @example
 * const batchOut = net.activateBatch([[0,0,1],[1,0,0],[0,1,0]]);
 * console.log(batchOut.length); // 3 rows
 * @remarks For small batches this is perfectly adequate and clear.
 */
export function activateBatch(
  this: Network,
  inputs: number[][],
  training = false,
): number[][] {
  const networkInternal = this as unknown as NetworkInternals;

  // Step 1: Validate that we received a batch matrix.
  assertBatchInputCollection(inputs);

  // Step 2: Activate each input row with per-row validation.
  return activateValidatedBatchRows(
    inputs,
    this.input,
    networkInternal,
    training,
  );

  /**
   * Validate top-level batch container shape.
   *
   * @param batchInputs - Candidate matrix of input vectors.
   * @returns Nothing.
   */
  function assertBatchInputCollection(batchInputs: number[][]): void {
    if (!Array.isArray(batchInputs))
      throw new Error('inputs must be an array of input arrays');
  }

  /**
   * Activate each row of a validated batch matrix.
   *
   * @param batchInputs - Matrix of input vectors.
   * @param expectedInputSize - Network input dimensionality.
   * @param internalState - Runtime network internals used by activation pipeline.
   * @param isTraining - Whether each row activation should retain traces.
   * @returns Matrix of activation outputs.
   */
  function activateValidatedBatchRows(
    batchInputs: number[][],
    expectedInputSize: number,
    internalState: NetworkInternals,
    isTraining: boolean,
  ): number[][] {
    return batchInputs.map((inputVector, batchIndex) =>
      activateSingleBatchRow(
        inputVector,
        batchIndex,
        expectedInputSize,
        internalState,
        isTraining,
      ),
    );
  }

  /**
   * Validate and activate one batch row.
   *
   * @param inputVector - Input vector at one batch index.
   * @param batchIndex - Position in the batch matrix.
   * @param expectedInputSize - Network input dimensionality.
   * @param internalState - Runtime network internals used by activation pipeline.
   * @param isTraining - Whether activation should retain traces.
   * @returns Activation output vector.
   */
  function activateSingleBatchRow(
    inputVector: number[],
    batchIndex: number,
    expectedInputSize: number,
    internalState: NetworkInternals,
    isTraining: boolean,
  ): number[] {
    assertBatchRowInputSize(inputVector, batchIndex, expectedInputSize);
    return internalState.activate(inputVector, isTraining);
  }

  /**
   * Validate one batch row dimensionality.
   *
   * @param inputVector - Input vector at one batch index.
   * @param batchIndex - Position in the batch matrix.
   * @param expectedInputSize - Network input dimensionality.
   * @returns Nothing.
   */
  function assertBatchRowInputSize(
    inputVector: number[],
    batchIndex: number,
    expectedInputSize: number,
  ): void {
    if (
      !Array.isArray(inputVector) ||
      inputVector.length !== expectedInputSize
    ) {
      throw new Error(
        `Input[${batchIndex}] size mismatch: expected ${expectedInputSize}, got ${
          inputVector ? inputVector.length : 'undefined'
        }`,
      );
    }
  }
}
