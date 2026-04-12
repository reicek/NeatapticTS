/**
 * Activation chapter for `Network` execution policy.
 *
 * This folder answers the moment when a graph already exists and the next
 * question becomes: how should signal move through it right now? The same
 * network may be stepped for ordinary inference, training-aware forward passes,
 * zero-copy raw output reuse, or a sequence of batch rows. Keeping those paths
 * together makes the execution tradeoffs visible without mixing them into
 * topology or serialization code.
 *
 * The important split is between graph meaning and graph execution. Node and
 * connection chapters explain what the structure is. `activate/` explains how
 * that structure is stepped: validate inputs, decide whether the slab fast path
 * is still legal, preserve or skip training traces, and return outputs in the
 * shape the caller requested.
 *
 * A second useful lens is to read the public exports as four modes.
 * `activate()` is the ordinary compatibility path. `noTraceActivate()` is the
 * hot inference path when trace bookkeeping would be wasteful. `activateRaw()`
 * keeps typed-array reuse available when pooling matters more than boxed
 * outputs. `activateBatch()` is the clear orchestration layer for repeated
 * forward passes over many rows.
 *
 * The performance lesson here is not "always choose the fastest path." It is
 * "choose the narrowest path that still matches the caller's semantics." If a
 * network is slab-ready, this chapter can exploit contiguous typed arrays. If a
 * structural edit made that layout stale, the same boundary falls back to node
 * traversal instead of forcing callers to understand storage internals first.
 *
 * ```mermaid
 * flowchart LR
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   Input[caller input]:::base --> Modes[activate chapter]:::accent
 *   Modes --> Trace[activate<br/>keep traces]:::base
 *   Modes --> NoTrace[noTraceActivate<br/>inference hot path]:::base
 *   Modes --> Raw[activateRaw<br/>typed output reuse]:::base
 *   Modes --> Batch[activateBatch<br/>repeat over rows]:::base
 * ```
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   ActivateChapter[activate/]:::accent --> Validation[input validation and contexts]:::base
 *   ActivateChapter --> FastPath[slab fast path when layout is ready]:::base
 *   ActivateChapter --> Traversal[node traversal fallback]:::base
 *   ActivateChapter --> Buffers[pooled activation buffers]:::base
 * ```
 *
 * For background on why some activation paths preserve training traces while
 * others skip them, see Wikipedia contributors,
 * [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation). This
 * chapter sits at the forward-pass side of that story and decides how much
 * training bookkeeping each call should carry along.
 *
 * Example: use the no-trace path when you only need inference outputs.
 *
 * ```ts
 * const network = Network.createMLP(2, [3], 1);
 * const outputValues = network.noTraceActivate([0.2, 0.8]);
 * ```
 *
 * Example: run the same network over several input rows with one orchestration
 * call.
 *
 * ```ts
 * const network = Network.createMLP(2, [3], 1);
 * const batchOutputs = network.activateBatch(
 *   [
 *     [0, 1],
 *     [1, 0],
 *   ],
 *   true,
 * );
 * ```
 *
 * Practical reading order:
 *
 * 1. Start here for the public activation modes and their semantic differences.
 * 2. Continue into `network.activate.core.utils.ts` when you want the ordinary
 *    forward-pass pipeline.
 * 3. Continue into `network.activate.raw.utils.ts` and the no-trace helpers
 *    when typed-array reuse or inference hot paths are the next question.
 * 4. Finish with the context and helper files when you want the orchestration
 *    details behind validation, batching, and fallback behavior.
 */

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
 * 1. (Optional) Refresh the compiled activation schedule when a structural change
 *    marked topology as dirty.
 *  2. Validate the input dimensionality.
 *  3. Try the fast slab path; if it throws, continue with the standard path.
 *  4. Acquire a pooled output buffer sized to the number of output neurons.
 * 5. Traverse nodes in the compiled activation order when available:
 *      - Input nodes: assign values by explicit `inputNodeIds`, not raw node position.
 *      - Hidden and recurrent-component nodes: compute activation via
 *        Node.noTraceActivate without training traces.
 *      - Output nodes: activate in schedule order, then read out results in explicit
 *        `outputNodeIds` order so vector semantics stay stable even if storage order drifts.
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
