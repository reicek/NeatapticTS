import type Network from '../network';
import { activationArrayPool } from '../activationArrayPool';

/**
 * Network activation helpers (forward pass utilities).
 *
 * This module provides progressively lower–overhead entry points for performing
 * forward propagation through a {@link Network}. The emphasis is on:
 *  1. Educative clarity – each step is documented so newcomers can follow the
 *     life‑cycle of a forward pass in a neural network graph.
 *  2. Performance – fast paths avoid unnecessary allocation and bookkeeping when
 *     gradients / evolution traces are not needed.
 *  3. Safety – pooled buffers are never exposed directly to the public API.
 *
 * Exported functions:
 *  - {@link noTraceActivate}: ultra‑light inference (no gradients, minimal allocation).
 *  - {@link activateRaw}: thin semantic alias around the canonical Network.activate path.
 *  - {@link activateBatch}: simple mini‑batch loop utility.
 *
 * Design terminology used below:
 *  - Topological order: a sequence of nodes such that all directed connections flow forward.
 *  - Slab: a contiguous typed‑array structure packing node activations for vectorized math.
 *  - Trace / gradient bookkeeping: auxiliary data (e.g. eligibility traces, derivative caches)
 *    required for training algorithms; skipped in inference‑only modes.
 *  - Pool: an object managing reusable arrays to reduce garbage collection pressure.
 *
 * @module network.activate
 */

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
  /**
   * Reference to the network instance cast to any so internal/private helper properties
   * (underscored fields & fast path flags) can be accessed without TypeScript complaints.
   */
  const self = this as any;

  // Step 1: Ensure that if we require an acyclic graph, our cached topological
  // ordering of nodes is current. A fresh order guarantees deterministic forward propagation.
  if (self._enforceAcyclic && self._topoDirty)
    (this as any)._computeTopoOrder();

  // Step 2: Basic validation – mismatched length typically indicates a user error.
  if (!Array.isArray(input) || input.length !== this.input) {
    throw new Error(
      `Input size mismatch: expected ${this.input}, got ${
        input ? (input as any).length : 'undefined'
      }`
    );
  }

  // Step 3: Attempt a zero‑allocation vectorized activation over a packed slab. We wrap
  // the call in a try/catch to avoid penalizing typical paths with conditional prechecks.
  if ((this as any)._canUseFastSlab(false)) {
    try {
      return (this as any)._fastSlabActivate(input);
    } catch {
      // Silent fallback – correctness first; performance is opportunistic here.
    }
  }

  // Step 4: Acquire a pooled typed array (or array‑like) sized to the number of outputs.
  /** Pooled buffer to collect output activations in order. */
  /**
   * Pooled activation output buffer sized to the number of output neurons; will be cloned
   * into a plain array before returning to the caller to avoid external mutation of pooled memory.
   */
  const output = activationArrayPool.acquire(this.output);

  // Maintain a manual write index to decouple node iteration order from output layout.
  /**
   * Sequential index into the pooled output buffer. Increments each time we process
   * an output node so we produce a dense, zero‑gap array matching logical output order.
   */
  /** Sequential write index into the pooled output buffer. */
  let outIndex = 0;

  // Step 5: Iterate every node once. For hidden nodes we simply invoke noTraceActivate;
  // its internal logic will read predecessor activations already set during earlier steps.
  this.nodes.forEach((node, index) => {
    // Input nodes: feed value directly from the corresponding slot in the provided input vector.
    if (node.type === 'input') node.noTraceActivate(input[index]);
    // Output nodes: compute their activation (which implicitly uses upstream hidden/input nodes) and store.
    else if (node.type === 'output')
      (output as any)[outIndex++] = node.noTraceActivate();
    // Hidden nodes: just activate (value stored internally on the node itself).
    else node.noTraceActivate();
  });

  // Step 6: Copy pooled buffer to a fresh standard array so external callers cannot mutate
  // the pooled object after it's released (which would create hard‑to‑trace bugs).
  /** Detached plain array containing final output activations. */
  /** Final detached output activation vector. */
  const result = Array.from(output as any) as number[];

  // Always release pooled resources promptly to keep memory pressure low for future calls.
  activationArrayPool.release(output);

  return result;
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
  maxActivationDepth = 1000
): any {
  /** Access internal flags / helpers (private-ish) via a loose cast. */
  const self = this as any;

  // If the network is not reusing activation arrays there's nothing special to do – delegate.
  if (!self._reuseActivationArrays)
    return (this as any).activate(input, training, maxActivationDepth);

  // Even when reuse is enabled we currently still just delegate; hook point for future optimization.
  return (this as any).activate(input, training, maxActivationDepth);
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
  training = false
): number[][] {
  // Global validation – ensure we can iterate as expected.
  if (!Array.isArray(inputs))
    throw new Error('inputs must be an array of input arrays');

  /** Preallocate the output matrix at the correct height (one row per input). */
  /** Output matrix (row-major) where each row corresponds to activation of one input vector. */
  const out: number[][] = new Array(inputs.length);

  // Iterate sequentially – early exit behavior (via throw) will surface the first invalid row.
  for (let i = 0; i < inputs.length; i++) {
    /** Current input vector under evaluation. */
    /** Input vector at batch index i currently being processed. */
    const x = inputs[i];
    // Validate row dimensionality with a descriptive index for easier debugging.
    if (!Array.isArray(x) || x.length !== this.input) {
      throw new Error(
        `Input[${i}] size mismatch: expected ${this.input}, got ${
          x ? x.length : 'undefined'
        }`
      );
    }
    // Delegate to the network's activation (may perform tracing if training=true).
    out[i] = (this as any).activate(x, training);
  }

  return out;
}
