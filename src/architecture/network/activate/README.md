# architecture/network/activate

## architecture/network/activate/network.activate.utils.ts

### activateBatch

`(inputs: number[][], training: boolean) => number[][]`

Activate the network over a mini‑batch (array) of input vectors, returning a 2‑D array of outputs.

This helper simply loops, invoking {@link Network.activate} (or its bound variant) for each
sample. It is intentionally naive: no attempt is made to fuse operations across the batch.
For very large batch sizes or performance‑critical paths consider implementing a custom
vectorized backend that exploits SIMD, GPU kernels, or parallel workers.

Input validation occurs per row to surface the earliest mismatch with a descriptive index.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *
- `inputs` - - Array of input vectors; each must have length == network.input.
- `training` - - Whether each activation should keep training traces.

Returns: 2‑D array: outputs[i] is the activation result for inputs[i].

### activateRaw

`(input: number[], training: boolean, maxActivationDepth: number) => number[]`

Thin semantic alias to the network's main activation path.

At present this simply forwards to {@link Network.activate}. The indirection is useful for:
 - Future differentiation between raw (immediate) activation and a mode that performs reuse /
   staged batching logic.
 - Providing a stable exported symbol for external tooling / instrumentation.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *
- `input` - - Input vector (length == network.input).
- `training` - - Whether to retain training traces / gradients (delegated downstream).
- `maxActivationDepth` - - Guard against runaway recursion / cyclic activation attempts.

Returns: Implementation-defined result of Network.activate (typically an output vector).

### noTraceActivate

`(input: number[]) => number[]`

Perform a forward pass without creating or updating any training / gradient traces.

This is the most allocation‑sensitive activation path. Internally it will attempt
to leverage a compact "fast slab" routine (an optimized, vectorized broadcast over
contiguous activation buffers) when the Network instance indicates that such a path
is currently valid. If that attempt fails (for instance because the slab is stale
after a structural mutation) execution gracefully falls back to a node‑by‑node loop.

Algorithm outline:
 1. (Optional) Refresh cached topological order if the network enforces acyclicity
    and a structural change marked the order as dirty.
 2. Validate the input dimensionality.
 3. Try the fast slab path; if it throws, continue with the standard path.
 4. Acquire a pooled output buffer sized to the number of output neurons.
 5. Iterate all nodes in their internal order:
      - Input nodes: directly assign provided input values.
      - Hidden nodes: compute activation via Node.noTraceActivate (no bookkeeping).
      - Output nodes: compute activation and store it (in sequence) inside the
        pooled output buffer.
 6. Copy the pooled buffer into a fresh array (detaches user from the pool) and
    release the pooled buffer back to the pool.

Complexity considerations:
 - Time: O(N + E) where N = number of nodes, E = number of inbound edges processed
   inside each Node.noTraceActivate call (not explicit here but inside the node).
 - Space: O(O) transient (O = number of outputs) due to the pooled output buffer.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *
- `input` - - Flat numeric vector whose length must equal network.input.

Returns: Array of output neuron activations (length == network.output).
