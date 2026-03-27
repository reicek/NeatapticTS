import type Network from '../../network/network';
export { activate, gaussianRand } from './network.activate.core.utils';
import {
  createBatchActivationContext,
  createNoTraceActivationContext,
  createRawActivationContext,
  executeBatchActivation,
  executeNoTraceActivation,
  executeRawActivation,
} from './network.activate.helpers.utils';
import { DEFAULT_MAX_ACTIVATION_DEPTH } from './network.activate.utils.types';

/**
 * Perform a forward pass without creating or updating training / gradient traces.
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
 * @param this - Bound Network instance.
 * @param input - Flat numeric vector whose length must equal network.input.
 * @returns Array of output neuron activations (length == network.output).
 * @throws {Error} If the provided input vector length mismatches the network's input size.
 * @example
 * const out = net.noTraceActivate([0.1, 0.2, 0.3]);
 * console.log(out); // => e.g. [0.5123, 0.0441]
 * @remarks Safe for inference hot paths; not suitable when gradients / training traces are required.
 */
export function noTraceActivate(this: Network, input: number[]): number[] {
  const activationContext = createNoTraceActivationContext(this, input);

  // Step 1: Delegate no-trace activation phases to dedicated helper orchestrator.
  return executeNoTraceActivation(activationContext);
}

/**
 * Thin semantic alias to the network's main activation path.
 *
 * At present this simply forwards to {@link Network.activate}. The indirection is useful for:
 *  - Future differentiation between raw (immediate) activation and a mode that performs reuse /
 *    staged batching logic.
 *  - Providing a stable exported symbol for external tooling / instrumentation.
 *
 * @param this - Bound Network instance.
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
  maxActivationDepth = DEFAULT_MAX_ACTIVATION_DEPTH,
): number[] {
  const activationContext = createRawActivationContext(
    this,
    input,
    training,
    maxActivationDepth,
  );

  // Step 1: Delegate raw activation execution to specialized helper orchestration.
  return executeRawActivation(activationContext);
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
 * @param this - Bound Network instance.
 * @param inputs - Array of input vectors; each must have length == network.input.
 * @param training - Whether each activation should keep training traces.
 * @returns 2‑D array: outputs[i] is the activation result for inputs[i].
 * @throws {Error} If inputs is not an array, or one of its vectors has an incorrect length.
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
  const activationContext = createBatchActivationContext(
    this,
    inputs,
    training,
  );

  // Step 1: Delegate batch activation phases to specialized helper orchestration.
  return executeBatchActivation(activationContext);
}
