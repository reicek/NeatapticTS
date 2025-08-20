# architecture/network

## architecture/network/network.activate.ts

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

`(input: number[], training: boolean, maxActivationDepth: number) => any`

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

Network activation helpers (forward pass utilities).

This module provides progressively lower–overhead entry points for performing
forward propagation through a {@link Network}. The emphasis is on:
 1. Educative clarity – each step is documented so newcomers can follow the
    life‑cycle of a forward pass in a neural network graph.
 2. Performance – fast paths avoid unnecessary allocation and bookkeeping when
    gradients / evolution traces are not needed.
 3. Safety – pooled buffers are never exposed directly to the public API.

Exported functions:
 - {@link noTraceActivate}: ultra‑light inference (no gradients, minimal allocation).
 - {@link activateRaw}: thin semantic alias around the canonical Network.activate path.
 - {@link activateBatch}: simple mini‑batch loop utility.

Design terminology used below:
 - Topological order: a sequence of nodes such that all directed connections flow forward.
 - Slab: a contiguous typed‑array structure packing node activations for vectorized math.
 - Trace / gradient bookkeeping: auxiliary data (e.g. eligibility traces, derivative caches)
   required for training algorithms; skipped in inference‑only modes.
 - Pool: an object managing reusable arrays to reduce garbage collection pressure.

## architecture/network/network.connect.ts

### connect

`(from: import("D:/code-practice/NeatapticTS/src/architecture/node").default, to: import("D:/code-practice/NeatapticTS/src/architecture/node").default, weight: number | undefined) => import("D:/code-practice/NeatapticTS/src/architecture/connection").default[]`

Network structural mutation helpers (connect / disconnect).

This module centralizes the logic for adding and removing edges (connections) between
nodes in a {@link Network}. By isolating the book‑keeping here we keep the primary
Network class lean and ensure consistent handling of:
 - Acyclic constraints
 - Multiple low‑level connections returned by composite node operations
 - Gating & self‑connection invariants
 - Cache invalidation (topological order + packed activation slabs)

Exported functions:
 - {@link connect}: Create one or more connections from a source node to a target node.
 - {@link disconnect}: Remove (at most) one direct connection from source to target.

Key terminology:
 - Self‑connection: An edge where from === to (loop). Usually disallowed under acyclicity.
 - Gating: A mechanism where a third node modulates (gates) the weight / influence of a connection.
 - Slab: Packed typed‑array representation of connections for vectorized forward passes.

### disconnect

`(from: import("D:/code-practice/NeatapticTS/src/architecture/node").default, to: import("D:/code-practice/NeatapticTS/src/architecture/node").default) => void`

Remove (at most) one directed connection from source 'from' to target 'to'.

Only a single direct edge is removed because typical graph configurations maintain at most
one logical connection between a given pair of nodes (excluding potential future multi‑edge
semantics). If the target edge is gated we first call {@link Network.ungate} to maintain
gating invariants (ensuring the gater node's internal gate list remains consistent).

Algorithm outline:
 1. Choose the correct list (selfconns vs connections) based on whether from === to.
 2. Linear scan to find the first edge with matching endpoints.
 3. If gated, ungate to detach gater bookkeeping.
 4. Splice the edge out; exit loop (only one expected).
 5. Delegate per‑node cleanup via from.disconnect(to) (clears reverse references, traces, etc.).
 6. Mark structural caches dirty for lazy recomputation.

Complexity:
 - Time: O(m) where m is length of the searched list (connections or selfconns).
 - Space: O(1) extra.

Idempotence: If no such edge exists we still perform node-level disconnect and flag caches dirty –
this conservative approach simplifies callers (they need not pre‑check existence).

Parameters:
- `this` - - Bound  {@link Network} instance.
 *
- `from` - - Source node.
- `to` - - Target node.

## architecture/network/network.deterministic.ts

### getRandomFn

`() => (() => number) | undefined`

Retrieve the active random function reference (for testing, instrumentation, or swapping).

Mutating the returned function's closure variables (if any) is not recommended; prefer using
higher-level APIs (setSeed / restoreRNG) to manage state.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *

Returns: Function producing numbers in [0,1). May be undefined if never seeded (call setSeed first).

### getRNGState

`() => number | undefined`

Get the current internal 32‑bit RNG state value.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *

Returns: Unsigned 32‑bit state integer or undefined if generator not yet seeded or was reset.

### network.deterministic

Default export bundle for convenient named imports.

### restoreRNG

`(fn: () => number) => void`

Restore a previously captured RNG function implementation (advanced usage).

This does NOT rehydrate _rngState (it explicitly sets it to undefined). Intended for scenarios
where a caller has customly serialized a full RNG closure or wants to inject a deterministic stub.
If you only need to restore the raw state word produced by {@link snapshotRNG}, prefer
{@link setRNGState} instead.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *
- `fn` - - Function returning a pseudo‑random number in [0,1). Caller guarantees determinism if required.

### RNGSnapshot

Deterministic pseudo‑random number generation (PRNG) utilities for {@link Network}.

Why this module exists:
 - Facilitates reproducible evolutionary runs / gradient training by allowing explicit seeding.
 - Centralizes RNG state management & snapshot/restore operations (useful for rollbacks or
   deterministic tests around mutation sequences).
 - Keeps the core Network class focused by extracting ancillary RNG concerns.

Implementation notes:
 - Uses a small, fast 32‑bit xorshift / mix style generator (same semantics as the legacy inline version)
   combining an additive Weyl sequence step plus a few avalanche-style integer mixes.
 - Not cryptographically secure. Do not use for security / fairness sensitive applications.
 - Produces floating point numbers in [0,1) with 2^32 (~4.29e9) discrete possible mantissa states.

Public surface:
 - {@link setSeed}: Initialize deterministic generator with a numeric seed.
 - {@link snapshotRNG}: Capture current training step + raw internal RNG state.
 - {@link restoreRNG}: Provide an externally saved RNG function (advanced) & clear stored state.
 - {@link getRNGState} / {@link setRNGState}: Low-level accessors for the internal 32‑bit state word.
 - {@link getRandomFn}: Retrieve the active random() function reference (primarily for tests / tooling).

Design rationale:
 - Storing both a state integer (_rngState) and a function (_rand) allows hot-swapping alternative
   RNG implementations (e.g., for benchmarking or pluggable randomness strategies) without rewriting
   callsites inside Network algorithms.

### setRNGState

`(state: number) => void`

Explicitly set (override) the internal 32‑bit RNG state without changing the generator function.

This is a low‑level operation; typical clients should call {@link setSeed}. Provided for advanced
replay functionality where the same PRNG algorithm is assumed but you want to resume exactly at a
known state word.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *
- `state` - - Any finite number (only low 32 bits used). Ignored if not numeric.

### setSeed

`(seed: number) => void`

Seed the internal PRNG and install a deterministic random() implementation on the Network instance.

Process:
 1. Coerce the provided seed to an unsigned 32‑bit integer (>>> 0) for predictable wraparound behavior.
 2. Define an inline closure that advances an internal 32‑bit state using:
      a. A Weyl increment (adding constant 0x6D2B79F5 each call) ensuring full-period traversal of
         the 32‑bit space when combined with mixing.
      b. Two rounds of xorshift / integer mixing (xor, shifts, multiplications) to decorrelate bits.
      c. Normalization to [0,1) by dividing the final 32‑bit unsigned integer by 2^32.

Bit-mixing explanation (rough intuition):
 - XOR with shifted versions spreads high-order entropy to lower bits.
 - Multiplication (Math.imul) with carefully chosen odd constants introduces non-linear mixing.
 - The final right shift & xor avalanche aims to reduce sequential correlation.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *
- `seed` - - Any finite number; only its lower 32 bits are used.

### snapshotRNG

`() => import("D:/code-practice/NeatapticTS/src/architecture/network/network.deterministic").RNGSnapshot`

Capture a snapshot of the RNG state together with the network's training step.

Useful for implementing speculative evolutionary mutations where you may revert both the
structural change and the randomness timeline if accepting/rejecting a candidate.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *

Returns: Object containing current training step & 32‑bit RNG state (both possibly undefined if unseeded).

## architecture/network/network.evolve.ts

### buildMultiThreadFitness

`(set: TrainingSample[], cost: any, amount: number, growth: number, threads: number, options: any) => Promise<{ fitnessFunction: (genome: import("D:/code-practice/NeatapticTS/src/architecture/network").default) => number; threads: number; } | { fitnessFunction: (population: import("D:/code-practice/NeatapticTS/src/architecture/network").default[]) => Promise<void>; threads: number; }>`

Build a multi-threaded (worker-based) population fitness evaluator if worker infrastructure is available.

Strategy:
 - Attempt to dynamically obtain a Worker constructor (node or browser variant).
 - If not possible, gracefully fall back to single-thread evaluation.
 - Spawn N workers (threads) each capable of evaluating genomes by calling worker.evaluate(genome).
 - Provide a fitness function that takes the whole population and returns a Promise that resolves
   when all queued genomes have been processed. Each genome's score is written in-place.

Implementation details:
 - Queue: simple FIFO (array shift) suffices because ordering is not critical.
 - Robustness: Each worker evaluation is wrapped with error handling to prevent a single failure
   from stalling the batch; failed evaluations simply proceed to next genome.
 - Complexity penalty applied after raw result retrieval: genome.score = -result - penalty.

Returned metadata sets options.fitnessPopulation=true so downstream NEAT logic treats the fitness
function as operating over the entire population at once (rather than per-genome).

Parameters:
- `set` - - Dataset.
- `cost` - - Cost function.
- `amount` - - Repetition count (unused directly here; assumed handled inside worker.evaluate result metric if needed).
- `growth` - - Complexity penalty scalar.
- `threads` - - Desired worker count.
- `options` - - Evolution options object (mutated to add cleanup hooks & flags).

Returns: Object with fitnessFunction (population evaluator) and resolved thread count.

### buildSingleThreadFitness

`(set: TrainingSample[], cost: any, amount: number, growth: number) => (genome: import("D:/code-practice/NeatapticTS/src/architecture/network").default) => number`

Build a single-threaded fitness evaluation function (classic NEAT style) evaluating a genome
over the provided dataset and returning a scalar score where higher is better.

Fitness Definition:
  fitness = -averageError - complexityPenalty
We accumulate negative error (so lower error => higher fitness) over `amount` independent
evaluations (amount>1 can smooth stochastic evaluation noise) then subtract complexity penalty.

Error handling: If evaluation throws (numerical instability, internal error) we return -Infinity
so such genomes are strongly disfavored.

Parameters:
- `set` - - Dataset of training samples.
- `cost` - - Cost function reference (should expose error computation in genome.test).
- `amount` - - Number of repeated evaluations to average.
- `growth` - - Complexity penalty scalar.

Returns: Function mapping a Network genome to a numeric fitness.

### computeComplexityPenalty

`(genome: import("D:/code-practice/NeatapticTS/src/architecture/network").default, growth: number) => number`

Compute a structural complexity penalty scaled by a growth factor.

Complexity heuristic:
  (hidden nodes) + (connections) + (gates)
hidden nodes = total nodes - input - output (to avoid penalizing fixed I/O interface size).

Rationale: Encourages minimal / parsimonious networks by subtracting a term from fitness
proportional to network size, counteracting bloat. Growth hyper‑parameter tunes pressure.

Caching strategy: We memoize the base complexity (pre‑growth scaling) per genome when its
structural counts (nodes / connections / gates) are unchanged. This is safe because only
structural mutations alter these counts, and those invalidate earlier entries naturally
(since mutated genomes are distinct object references in typical NEAT flows).

Parameters:
- `genome` - - Candidate network whose complexity to measure.
- `growth` - - Positive scalar controlling strength of parsimony pressure.

Returns: Complexity * growth (used directly to subtract from fitness score).

### EvolutionConfig

Internal evolution configuration summary (for potential logging / debugging)
capturing normalized option values used by the local evolutionary loop.

### evolveNetwork

`(set: TrainingSample[], options: any) => Promise<{ error: number; iterations: number; time: number; }>`

Evolve (optimize) the current network's topology and weights using a NEAT-like evolutionary loop
until a stopping criterion (target error or max iterations) is met.

High-level process:
 1. Validate dataset shape (input/output vector sizes must match network I/O counts).
 2. Normalize / default option values and construct an internal configuration summary.
 3. Build appropriate fitness evaluation function (single or multi-thread).
 4. Initialize a Neat population (optionally with speciation) seeded by this network.
 5. Iteratively call neat.evolve():
      - Retrieve fittest genome + its fitness.
      - Derive an error metric from fitness (inverse relationship considering complexity penalty).
      - Track best genome overall (elitism) and perform logging/scheduling callbacks.
      - Break if error criterion satisfied or iterations exceeded.
 6. Replace this network's internal structural arrays with the best discovered genome's (in-place upgrade).
 7. Cleanup any worker threads and report final statistics.

Fitness / Error relationship:
  fitness = -error - complexityPenalty  =>  error = -(fitness - complexityPenalty)
We recompute error from the stored fitness plus penalty to ensure consistent reporting.

Resilience strategies:
 - Guard against infinite / NaN errors; after MAX_INF consecutive invalid errors we abort.
 - Fallback for tiny populations: increase mutation aggressiveness to prevent premature convergence.

Parameters:
- `this` - - Bound  {@link Network} instance being evolved in-place.
 *
- `set` - - Supervised dataset (array of {input, output}).
- `options` - - Evolution options (see README / docs). Key fields include:
- iterations: maximum generations (if omitted must supply error target)
- error: target error threshold (if omitted must supply iterations)
- growth: complexity penalty scaling
- amount: number of score evaluations (averaged) per genome
- threads: desired worker count (>=2 enables multi-thread path if available)
- popsize / populationSize: population size
- schedule: { iterations: number, function: (ctx) => void } periodic callback
- log: generation interval for console logging
- clear: whether to call network.clear() after adopting best genome

Returns: Summary object { error, iterations, time(ms) }.

### TrainingSample

A single supervised training example used to evaluate fitness.

## architecture/network/network.gating.ts

### gate

`(node: import("D:/code-practice/NeatapticTS/src/architecture/node").default, connection: import("D:/code-practice/NeatapticTS/src/architecture/connection").default) => void`

Gating & node removal utilities for {@link Network}.

Gating concept:
 - A "gater" node modulates the effective weight of a target connection. Conceptually the raw
   connection weight w is multiplied (or otherwise transformed) by a function of the gater node's
   activation a_g (actual math lives in {@link Node.gate}). This enables dynamic, context-sensitive
   routing (similar in spirit to attention mechanisms or LSTM-style gates) within an evolved topology.

Removal strategy (removeNode):
 - When excising a hidden node we attempt to preserve overall connectivity by creating bridging
   connections from each of its predecessors to each of its successors if such edges do not already
   exist. Optional logic reassigns previous gater nodes to these new edges (best-effort) to preserve
   modulation diversity.

Mutation interplay:
 - The flag `mutation.SUB_NODE.keep_gates` determines whether gating nodes associated with edges
   passing through the removed node should be retained and reassigned.

Determinism note:
 - Bridging gate reassignment currently uses Math.random directly; for fully deterministic runs
   you may consider replacing with the network's seeded RNG (if provided) in future refactors.

Exported functions:
 - {@link gate}: Attach a gater to a connection.
 - {@link ungate}: Remove gating from a connection.
 - {@link removeNode}: Remove a hidden node while attempting to preserve connectivity & gating.

### removeNode

`(node: import("D:/code-practice/NeatapticTS/src/architecture/node").default) => void`

Remove a hidden node from the network while attempting to preserve functional connectivity.

Algorithm outline:
 1. Reject removal if node is input/output (structural invariants) or absent (error).
 2. Optionally collect gating nodes (if keep_gates flag) from inbound & outbound connections.
 3. Remove self-loop (if present) to simplify subsequent edge handling.
 4. Disconnect all inbound edges (record their source nodes) and all outbound edges (record targets).
 5. For every (input predecessor, output successor) pair create a new connection unless:
      a. input === output (avoid trivial self loops) OR
      b. an existing projection already connects them.
 6. Reassign preserved gater nodes randomly onto newly created bridging connections.
 7. Ungate any connections that were gated BY this node (where node acted as gater).
 8. Remove node from network node list and flag node index cache as dirty.

Complexity summary:
 - Let I = number of inbound edges, O = number of outbound edges.
 - Disconnect phase: O(I + O)
 - Bridging phase: O(I * O) connection existence checks (isProjectingTo) + potential additions.
 - Gater reassignment: O(min(G, newConnections)) where G is number of preserved gaters.

Preservation rationale:
 - Reassigning gaters maintains some of the dynamic modulation capacity that would otherwise
   be lost, aiding continuity during topology simplification.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *
- `node` - - Hidden node to remove.

### ungate

`(connection: import("D:/code-practice/NeatapticTS/src/architecture/connection").default) => void`

Remove gating from a connection, restoring its static weight contribution.

Idempotent: If the connection is not currently gated, the call performs no structural changes
(and optionally logs a warning). After ungating, the connection's weight will be used directly
without modulation by a gater activation.

Complexity: O(n) where n = number of gated connections (indexOf lookup) – typically small.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *
- `connection` - - Connection to ungate.

## architecture/network/network.genetic.ts

### crossOver

`(network1: import("D:/code-practice/NeatapticTS/src/architecture/network").default, network2: import("D:/code-practice/NeatapticTS/src/architecture/network").default, equal: boolean) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Genetic operator: NEAT‑style crossover (legacy merge operator removed).

This module now focuses solely on producing recombinant offspring via {@link crossOver}.
The previous experimental Network.merge has been removed to reduce maintenance surface area
and avoid implying a misleading “sequential composition” guarantee.

## architecture/network/network.mutate.ts

### _addBackConn

`() => void`

ADD_BACK_CONN: Add a backward (recurrent) connection (acyclic mode must be off).

### _addConn

`() => void`

ADD_CONN: Add a new forward (acyclic) connection between two previously unconnected nodes.
Recurrent edges are handled separately by ADD_BACK_CONN.

### _addGate

`() => void`

ADD_GATE: Assign a random (hidden/output) node to gate a random ungated connection.

### _addGRUNode

`() => void`

ADD_GRU_NODE: Replace a random connection with a minimal 1‑unit GRU block.

### _addLSTMNode

`() => void`

ADD_LSTM_NODE: Replace a random connection with a minimal 1‑unit LSTM block (macro mutation).

### _addNode

`() => void`

ADD_NODE: Insert a new hidden node by splitting an existing connection.

Deterministic test mode (config.deterministicChainMode):
 - Maintain an internal linear chain (input → hidden* → output).
 - Always split the chain's terminal edge, guaranteeing depth +1 per call.
 - Prune side edges from chain nodes to keep depth measurement unambiguous.

Standard evolutionary mode:
 - Sample a random existing connection and perform the classical NEAT split.

Core algorithm (stochastic variant):
 1. Pick connection (random).
 2. Disconnect it (preserve any gater reference).
 3. Create hidden node (random activation mutation).
 4. Insert before output tail to preserve ordering invariants.
 5. Connect source→hidden and hidden→target.
 6. Reassign gater uniformly to one of the new edges.

### _addSelfConn

`() => void`

ADD_SELF_CONN: Add a self loop to a random eligible node (only when cycles allowed).

### _batchNorm

`() => void`

BATCH_NORM: Placeholder mutation – marks a random hidden node with a flag for potential
future batch normalization integration. Currently a no-op beyond tagging.

### _modActivation

`(method: any) => void`

MOD_ACTIVATION: Swap activation (squash) of a random eligible node; may exclude outputs.

### _modBias

`(method: any) => void`

MOD_BIAS: Delegate to node.mutate to adjust bias of a random non‑input node.

### _modWeight

`(method: any) => void`

MOD_WEIGHT: Perturb a single (possibly self) connection weight by uniform delta in [min,max].

### _reinitWeight

`(method: any) => void`

REINIT_WEIGHT: Reinitialize all incoming/outgoing/self connection weights for a random node.
Useful as a heavy mutation to escape local minima. Falls back silently if no eligible node.

### _subBackConn

`() => void`

SUB_BACK_CONN: Remove a backward connection meeting redundancy heuristics.

### _subConn

`() => void`

SUB_CONN: Remove a forward connection chosen under redundancy heuristics to avoid disconnects.

### _subGate

`() => void`

SUB_GATE: Remove gating from a random previously gated connection.

### _subNode

`() => void`

SUB_NODE: Remove a random hidden node (if any remain).
After removal a tiny deterministic weight nudge encourages observable phenotype change in tests.

### _subSelfConn

`() => void`

SUB_SELF_CONN: Remove a random existing self loop.

### _swapNodes

`(method: any) => void`

SWAP_NODES: Exchange bias & activation function between two random eligible nodes.

### mutateImpl

`(method: any) => void`

Public entry point: apply a single mutation operator to the network.

Steps:
 1. Validate the supplied method (enum value or descriptor object).
 2. Resolve helper implementation from the dispatch map (supports objects exposing name/type/identity).
 3. Invoke helper (passing through method for parameterized operators).
 4. Flag topology caches dirty so ordering / slabs rebuild lazily.

Accepts either the raw enum value (e.g. `mutation.ADD_NODE`) or an object carrying an
identifying `name | type | identity` field allowing future parameterization without breaking call sites.

Parameters:
- `this` - Network instance (bound).
- `method` - Mutation enum value or descriptor object.

## architecture/network/network.onnx.ts

### assignActivationFunctions

`(network: import("D:/code-practice/NeatapticTS/src/architecture/network").default, onnx: import("D:/code-practice/NeatapticTS/src/architecture/network/network.onnx").OnnxModel, hiddenLayerSizes: number[]) => void`

Map activation op_types from ONNX nodes back to internal activation functions.

### assignWeightsAndBiases

`(network: import("D:/code-practice/NeatapticTS/src/architecture/network").default, onnx: import("D:/code-practice/NeatapticTS/src/architecture/network/network.onnx").OnnxModel, hiddenLayerSizes: number[], metadataProps: { key: string; value: string; }[] | undefined) => void`

Apply weights & biases from ONNX initializers onto the newly created network.

### buildOnnxModel

`(network: import("D:/code-practice/NeatapticTS/src/architecture/network").default, layers: any[][], options: import("D:/code-practice/NeatapticTS/src/architecture/network/network.onnx").OnnxExportOptions) => import("D:/code-practice/NeatapticTS/src/architecture/network/network.onnx").OnnxModel`

Construct the ONNX model graph (initializers + nodes) given validated layers.

### Conv2DMapping

Mapping declaration for treating a fully-connected layer as a 2D convolution during export.
This assumes the dense layer was originally synthesized from a convolution with weight sharing; we reconstitute spatial metadata.
Each mapping references an export-layer index (1-based across hidden layers, output layer would be hiddenCount+1) and supplies spatial/kernel hyperparameters.
Validation ensures that input spatial * channels product equals the previous layer width and that output channels * output spatial equals the current layer width.

### deriveHiddenLayerSizes

`(initializers: OnnxTensor[], metadataProps: { key: string; value: string; }[] | undefined) => number[]`

Extract hidden layer sizes from ONNX initializers (weight tensors).

### exportToONNX

`(network: import("D:/code-practice/NeatapticTS/src/architecture/network").default, options: import("D:/code-practice/NeatapticTS/src/architecture/network/network.onnx").OnnxExportOptions) => import("D:/code-practice/NeatapticTS/src/architecture/network/network.onnx").OnnxModel`

Export a minimal multilayer perceptron Network to a lightweight ONNX JSON object.

Steps:
 1. Rebuild connection cache ensuring up-to-date adjacency.
 2. Index nodes for error messaging.
 3. Infer strict layer ordering (throws if structure unsupported).
 4. Validate homogeneity & full connectivity layer-to-layer.
 5. Build initializer tensors (weights + biases) and node list (Gemm + activation pairs).

Constraints: See module doc. Throws descriptive errors when assumptions violated.

### importFromONNX

`(onnx: import("D:/code-practice/NeatapticTS/src/architecture/network/network.onnx").OnnxModel) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Import a model previously produced by {@link exportToONNX} into a fresh Network instance.

Core Steps:
 1. Parse input/output tensor shapes (supports optional symbolic batch dim).
 2. Derive hidden layer sizes (prefer `layer_sizes` metadata; fallback to weight tensor grouping heuristic).
 3. Instantiate matching layered MLP (inputs -> hidden[] -> outputs); remove placeholder hidden nodes for single layer perceptrons.
 4. Assign weights & biases (aggregated or per-neuron) from W/B initializers.
 5. Reconstruct activation functions from Activation node op_types (layer or per-neuron).
 6. Restore recurrent self connections from recorded diagonal Rk matrices if `recurrent_single_step` metadata present.
 7. Experimental: Reconstruct LSTM / GRU layers when fused initializers & metadata (`lstm_emitted_layers`, `gru_emitted_layers`) detected
    by replacing the corresponding hidden node block with a freshly constructed Layer.lstm / Layer.gru instance and remapping weights.
 8. Rebuild flat connection array for downstream invariants.

Experimental Behavior:
 - LSTM/GRU reconstruction is best-effort; inconsistencies in tensor shapes or gate counts result in silent skip (import still succeeds).
 - Recurrent biases (Rb) absent; self-connection diagonal only restored for cell/candidate groups.

Limitations:
 - Only guaranteed for self-produced models; arbitrary ONNX graphs or differing op orderings are unsupported.
 - Fused recurrent node emission currently leaves original unfused Gemm/Activation path in exported model (import ignores duplicates).

### inferLayerOrdering

`(network: import("D:/code-practice/NeatapticTS/src/architecture/network").default) => any[][]`

Infer strictly layered ordering from a network, ensuring feed-forward fully-connected structure.

### mapActivationToOnnx

`(squash: any) => string`

Map an internal activation function (squash) to an ONNX op_type, defaulting to Identity.

### OnnxExportOptions

Options controlling ONNX export behavior (Phase 1).

### OnnxModel

### Pool2DMapping

Mapping describing a pooling operation inserted after a given export-layer index.

### rebuildConnectionsLocal

`(networkLike: any) => void`

Rebuild the network's flat connections array from each node's outgoing list (avoids circular import).

### validateLayerHomogeneityAndConnectivity

`(layers: any[][], network: import("D:/code-practice/NeatapticTS/src/architecture/network").default, options: import("D:/code-practice/NeatapticTS/src/architecture/network/network.onnx").OnnxExportOptions) => void`

Validate layer connectivity and (optionally) homogeneity; mixed activations allowed with per-neuron decomposition.

## architecture/network/network.prune.ts

### getCurrentSparsity

`() => number`

Current sparsity fraction relative to the training-time pruning baseline.

### maybePrune

`(iteration: number) => void`

Opportunistically perform scheduled pruning during gradient-based training.

Scheduling model:
 - start / end define an iteration window (inclusive) during which pruning may occur
 - frequency defines cadence (every N iterations inside the window)
 - targetSparsity is linearly annealed from 0 to its final value across the window
 - method chooses ranking heuristic (magnitude | snip)
 - optional regrowFraction allows dynamic sparse training: after removing edges we probabilistically regrow
   a fraction of them at random unused positions (respecting acyclic constraint if enforced)

SNIP heuristic:
 - Uses |w * grad| style saliency approximation (here reusing stored delta stats as gradient proxy)
 - Falls back to pure magnitude if gradient stats absent.

### pruneToSparsity

`(targetSparsity: number, method: "magnitude" | "snip") => void`

Evolutionary (generation-based) pruning toward a target sparsity baseline.
Unlike maybePrune this operates immediately relative to the first invocation's connection count
(stored separately as _evoInitialConnCount) and does not implement scheduling or regrowth.

### rankConnections

`(conns: import("D:/code-practice/NeatapticTS/src/architecture/connection").default[], method: "magnitude" | "snip") => import("D:/code-practice/NeatapticTS/src/architecture/connection").default[]`

Structured and dynamic pruning utilities for networks.

Features:
 - Scheduled pruning during gradient-based training ({@link maybePrune}) with linear sparsity ramp.
 - Evolutionary generation pruning toward a target sparsity ({@link pruneToSparsity}).
 - Two ranking heuristics:
     magnitude: |w|
     snip: |w * g| approximation (g approximated via accumulated delta stats; falls back to |w|)
 - Optional stochastic regrowth during scheduled pruning (dynamic sparse training), preserving acyclic constraints.

Internal State Fields (attached to Network via `any` casting):
 - _pruningConfig: user-specified schedule & options (start, end, frequency, targetSparsity, method, regrowFraction, lastPruneIter)
 - _initialConnectionCount: baseline connection count captured outside (first training iteration)
 - _evoInitialConnCount: baseline for evolutionary pruning (first invocation of pruneToSparsity)
 - _rand: deterministic RNG function
 - _enforceAcyclic: boolean flag enforcing forward-only connectivity ordering
 - _topoDirty: topology order invalidation flag consumed by activation fast path / topological sorting

### regrowConnections

`(network: import("D:/code-practice/NeatapticTS/src/architecture/network").default, desiredRemaining: number, maxAttempts: number) => void`

Attempt stochastic regrowth of pruned connections up to a desired remaining count.

## architecture/network/network.remove.ts

### removeNode

`(node: import("D:/code-practice/NeatapticTS/src/architecture/node").default) => void`

Node removal utilities.

This module provides a focused implementation for removing a single hidden node from a network
while attempting to preserve overall functional connectivity. The removal procedure mirrors the
legacy Neataptic logic but augments it with clearer documentation and explicit invariants.

High‑level algorithm (removeNode):
 1. Guard: ensure the node exists and is not an input or output (those are structural anchors).
 2. Ungate: detach any connections gated BY the node (we don't currently reassign gater roles).
 3. Snapshot inbound / outbound connections (before mutation of adjacency lists).
 4. Disconnect all inbound, outbound, and self connections.
 5. Physically remove the node from the network's node array.
 6. Simple path repair heuristic: for every former inbound source and outbound target, add a
    direct connection if (a) both endpoints still exist, (b) they are distinct, and (c) no
    direct connection already exists. This keeps forward information flow possibilities.
 7. Mark topology / caches dirty so that subsequent activation / ordering passes rebuild state.

Notes / Limitations:
 - We do NOT attempt to clone weights or distribute the removed node's function across new
   connections (more sophisticated strategies could average or compose weights).
 - Gating effects involving the removed node as a gater are dropped; downstream behavior may
   change—callers relying heavily on gating may want a custom remap strategy.
 - Self connections are simply removed; no attempt is made to emulate recursion via alternative
   structures.

## architecture/network/network.serialize.ts

### deserialize

`(data: any[], inputSize: number | undefined, outputSize: number | undefined) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Static counterpart to {@link serialize}. Rebuilds a Network from the compact tuple form.
Accepts optional explicit input/output size overrides (useful when piping through evolvers that trim IO).

### fromJSONImpl

`(json: any) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Reconstruct a Network from the verbose JSON produced by {@link toJSONImpl} (formatVersion 2).
Defensive parsing retains forward compatibility (warns on unknown versions rather than aborting).

### network.serialize

### serialize

`() => any[]`

Serialization & deserialization helpers for Network instances.

Provides two independent formats:
 1. Compact tuple (serialize/deserialize): optimized for fast structured clone / worker transfer.
 2. Verbose JSON (toJSONImpl/fromJSONImpl): stable, versioned representation retaining structural genes.

Compact tuple format layout:
 [ activations: number[], states: number[], squashes: string[],
   connections: { from:number; to:number; weight:number; gater:number|null }[],
   inputSize: number, outputSize: number ]

Design Principles:
 - Avoid deep nested objects to reduce serialization overhead.
 - Use current node ordering as canonical index mapping (caller must keep ordering stable between peers).
 - Include current activation/state for scenarios resuming partially evaluated populations.
 - Self connections placed in the same array as normal connections for uniform reconstruction.

Verbose JSON (formatVersion = 2) adds:
 - Enabled flag for connections (innovation toggling).
 - Stable geneId (if tracked) on nodes.
 - Dropout probability.

Future Ideas:
 - Delta / patch serialization for large evolving populations.
 - Compressed binary packing (e.g., Float32Array segments) for WASM pipelines.

### toJSONImpl

`() => object`

Verbose JSON export (stable formatVersion). Omits transient runtime fields but keeps structural genetics.
formatVersion=2 adds: enabled flags, stable geneId (if present), dropout value.

### default

#### _flags

Packed state flags (private for future-proofing hidden class):
bit0 => enabled gene expression (1 = active)
bit1 => DropConnect active mask (1 = not dropped this forward pass)
bit2 => hasGater (1 = symbol field present)
bits3+ reserved.

#### _la_shadowWeight

**Deprecated:** Use lookaheadShadowWeight instead.

#### acquire

`(from: import("D:/code-practice/NeatapticTS/src/architecture/node").default, to: import("D:/code-practice/NeatapticTS/src/architecture/node").default, weight: number | undefined) => import("D:/code-practice/NeatapticTS/src/architecture/connection").default`

Acquire a `Connection` from the pool (or construct new). Fields are fully reset & given
a fresh sequential `innovation` id. Prefer this in evolutionary algorithms that mutate
topology frequently to reduce GC pressure.

Parameters:
- `from` - Source node.
- `to` - Target node.
- `weight` - Optional initial weight.

Returns: Reinitialized connection instance.

#### dcMask

DropConnect active mask: 1 = not dropped (active), 0 = dropped for this stochastic pass.

#### dropConnectActiveMask

Convenience alias for DropConnect mask with clearer naming.

#### eligibility

Standard eligibility trace (e.g., for RTRL / policy gradient credit assignment).

#### enabled

Whether the gene (connection) is currently expressed (participates in forward pass).

#### firstMoment

First moment estimate (Adam / AdamW) (was opt_m).

#### from

The source (pre-synaptic) node supplying activation.

#### gain

Multiplicative modulation applied *after* weight. Default is `1` (neutral). We only store an
internal symbol-keyed property when the gain is non-neutral, reducing memory usage across
large populations where most connections are ungated.

#### gater

Optional gating node whose activation can modulate effective weight (symbol-backed).

#### gradientAccumulator

Generic gradient accumulator (RMSProp / AdaGrad) (was opt_cache).

#### hasGater

Whether a gater node is assigned (modulates gain); true if the gater symbol field is present.

#### infinityNorm

Adamax: Exponential moving infinity norm (was opt_u).

#### innovation

Unique historical marking (auto-increment) for evolutionary alignment.

#### innovationID

`(sourceNodeId: number, targetNodeId: number) => number`

Deterministic Cantor pairing function for a (sourceNodeId, targetNodeId) pair.
Useful when you want a stable innovation id without relying on global mutable counters
(e.g., for hashing or reproducible experiments).

NOTE: For large indices this can overflow 53-bit safe integer space; keep node indices reasonable.

Parameters:
- `sourceNodeId` - Source node integer id / index.
- `targetNodeId` - Target node integer id / index.

Returns: Unique non-negative integer derived from the ordered pair.

#### lookaheadShadowWeight

Lookahead: shadow (slow) weight parameter (was _la_shadowWeight).

#### maxSecondMoment

AMSGrad: Maximum of past second moment (was opt_vhat).

#### opt_cache

**Deprecated:** Use gradientAccumulator instead.

#### opt_m

**Deprecated:** Use firstMoment instead.

#### opt_m2

**Deprecated:** Use secondMomentum instead.

#### opt_u

**Deprecated:** Use infinityNorm instead.

#### opt_v

**Deprecated:** Use secondMoment instead.

#### opt_vhat

**Deprecated:** Use maxSecondMoment instead.

#### previousDeltaWeight

Last applied delta weight (used by classic momentum).

#### release

`(conn: import("D:/code-practice/NeatapticTS/src/architecture/connection").default) => void`

Return a `Connection` to the internal pool for later reuse. Do NOT use the instance again
afterward unless re-acquired (treat as surrendered). Optimizer / trace fields are not
scrubbed here (they're overwritten during `acquire`).

Parameters:
- `conn` - The connection instance to recycle.

#### resetInnovationCounter

`(value: number) => void`

Reset the monotonic auto-increment innovation counter (used for newly constructed / pooled instances).
You normally only call this at the start of an experiment or when deserializing a full population.

Parameters:
- `value` - New starting value (default 1).

#### secondMoment

Second raw moment estimate (Adam family) (was opt_v).

#### secondMomentum

Secondary momentum (Lion variant) (was opt_m2).

#### to

The target (post-synaptic) node receiving activation.

#### toJSON

`() => any`

Serialize to a minimal JSON-friendly shape (used for saving genomes / networks).
Undefined indices are preserved as `undefined` to allow later resolution / remapping.

Returns: Object with node indices, weight, gain, gater index (if any), innovation id & enabled flag.

#### totalDeltaWeight

Accumulated (batched) delta weight awaiting an apply step.

#### weight

Scalar multiplier applied to the source activation (prior to gain modulation).

#### xtrace

Extended trace structure for modulatory / eligibility propagation algorithms. Parallel arrays for cache-friendly iteration.

## architecture/network/network.slab.ts

### canUseFastSlab

`(training: boolean) => boolean`

Public helper: indicates whether fast slab path is currently viable.

### fastSlabActivate

`(input: number[]) => number[]`

High-performance forward pass using packed slabs + CSR adjacency.
Falls back to generic activate if prerequisites unavailable.

### getConnectionSlab

`() => { weights: any; from: any; to: any; flags: any; gain: any; version: any; used: any; capacity: any; }`

Return current slab (building lazily).

### getSlabVersion

`() => number`

Public accessor for current slab version (0 if never built).

### rebuildConnectionSlab

`(force: boolean) => void`

Fast slab (structure-of-arrays) acceleration layer (Phase 3 foundation).
----------------------------------------------------------------------
Motivation:
 Object graphs suffer from pointer chasing & polymorphic inline caches in large forward passes.
 By packing connection attributes into contiguous typed arrays we:
   - Improve spatial locality (sequential memory scans).
   - Enable simpler tight loops amenable to JIT & future WASM SIMD lowering.
   - Provide a staging ground for subsequent Phase 3+ memory slimming (flags / precision / plasticity).

Phase 3 Additions (initial commit):
 - Slab version counter (_slabVersion) incremented on each structural rebuild (educational introspection).
 - Flags array (_connFlags Uint8Array) and gain array (_connGain Float32/64) allocated in parallel (placeholders
   for enabled bits, drop masks, future plasticity multipliers). Currently all flags=1, gains=1.
 - getConnectionSlab() now returns flags & gain alongside weights/from/to.

Future (later Phase 3 iterations):
 - Geometric capacity growth (avoid realloc on small structural deltas). (Implemented: capacity reuse with growth factor)
 - Plasticity / mask bits stored compactly (bitpacking) to reduce per-connection bytes.
 - Typed array pooling / recycling to limit GC churn on frequent rebuilds.

Core Data Structures:
 weights (Float32Array|Float64Array)    : connection weights
 from    (Uint32Array)                  : source node indices
 to      (Uint32Array)                  : target node indices
 flags   (Uint8Array)                   : connection enable / mask bits (placeholder value=1)
 gain    (Float32Array|Float64Array)    : multiplicative gain (placeholder value=1)
 outStart (Uint32Array)                 : CSR row pointer style offsets (length nodeCount+1)
 outOrder (Uint32Array)                 : permutation of connection indices grouped by source

Rebuild Workflow:
 1. Reindex nodes if dirty.
 2. Allocate typed arrays sized to current connection count.
 3. Populate parallel arrays in a single linear pass.
 4. Mark adjacency dirty; increment version.

## architecture/network/network.standalone.ts

### generateStandalone

`(net: import("D:/code-practice/NeatapticTS/src/architecture/network").default) => string`

Generate a standalone JavaScript source string that returns an `activate(input:number[])` function.

Implementation Steps:
 1. Validate presence of output nodes (must produce something observable).
 2. Assign stable sequential indices to nodes (used as array offsets in generated code).
 3. Collect initial activation/state values into typed array initializers for warm starting.
 4. For each non-input node, build a line computing S[i] (pre-activation sum with bias) and A[i]
    (post-activation output). Gating multiplies activation by gate activations; self-connection adds
    recurrent term S[i] * weight before activation.
 5. De-duplicate activation functions: each unique squash name is emitted once; references become
    indices into array F of function references for compactness.
 6. Emit an IIFE producing the activate function with internal arrays A (activations) and S (states).

Parameters:
- `net` - Network instance to snapshot.

Returns: Source string (ES5-compatible) – safe to eval in sandbox to obtain activate function.

## architecture/network/network.stats.ts

### deepCloneValue

`(value: T) => T`

Network statistics accessors.

Currently exposes a single helper for retrieving the most recent regularization / stochasticity
metrics snapshot recorded during training or evaluation. The internal `_lastStats` field (on the
Network instance, typed as any) is expected to be populated elsewhere in the training loop with
values such as:
 - l1Penalty, l2Penalty
 - dropoutApplied (fraction of units dropped last pass)
 - weightNoiseStd (effective std dev used if noise injected)
 - sparsityRatio, prunedConnections
 - any custom user extensions (object is not strictly typed to allow experimentation)

Design decision: We return a deep copy to prevent external mutation of internal accounting state.
If the object is large and copying becomes a bottleneck, future versions could offer a freeze
option or incremental diff interface.

### getRegularizationStats

`() => any`

Obtain the last recorded regularization / stochastic statistics snapshot.

Returns a defensive deep copy so callers can inspect metrics without risking mutation of the
internal `_lastStats` object maintained by the training loop (e.g., during pruning, dropout, or
noise scheduling updates).

Returns: A deep-cloned stats object or null if no stats have been recorded yet.

## architecture/network/network.topology.ts

### computeTopoOrder

`() => void`

Topology utilities.

Provides:
 - computeTopoOrder: Kahn-style topological sorting with graceful fallback when cycles detected.
 - hasPath: depth-first reachability query (used to prevent cycle introduction when acyclicity enforced).

Design Notes:
 - We deliberately tolerate cycles by falling back to raw node ordering instead of throwing; this
   allows callers performing interim structural mutations to proceed (e.g. during evolve phases)
   while signaling that the fast acyclic optimizations should not be used.
 - Input nodes are seeded into the queue immediately regardless of in-degree to keep them early in
   the ordering even if an unusual inbound edge was added (defensive redundancy).
 - Self loops are ignored for in-degree accounting and queue progression (they neither unlock new
   nodes nor should they block ordering completion).

### hasPath

`(from: import("D:/code-practice/NeatapticTS/src/architecture/node").default, to: import("D:/code-practice/NeatapticTS/src/architecture/node").default) => boolean`

Depth-first reachability test (avoids infinite loops via visited set).

## architecture/network/network.training.ts

### __trainingInternals

### applyGradientClippingImpl

`(net: import("D:/code-practice/NeatapticTS/src/architecture/network").default, cfg: { mode: "norm" | "percentile" | "layerwiseNorm" | "layerwisePercentile"; maxNorm?: number | undefined; percentile?: number | undefined; }) => void`

Apply gradient clipping to accumulated connection deltas / bias deltas.

Modes:
 - norm / layerwiseNorm: L2 norm scaling (global vs per group).
 - percentile / layerwisePercentile: element-wise clamp at absolute percentile threshold.

Grouping:
 - If layerwise* and net.layers exists -> each defined layer is a group.
 - Else if layerwise* -> each non-input node becomes its own group.
 - Otherwise a single global group containing all learnable params.

### applyOptimizerStep

`(net: import("D:/code-practice/NeatapticTS/src/architecture/network").default, optimizer: any, currentRate: number, momentum: number, internalNet: any) => number`

Apply optimizer update step across all nodes; returns gradient L2 norm (approx).

### averageAccumulatedGradients

`(net: import("D:/code-practice/NeatapticTS/src/architecture/network").default, accumulationSteps: number) => void`

Divide accumulated gradients by accumulationSteps (average reduction mode).

### CheckpointConfig

Checkpoint callback spec.

### computeMonitoredError

`(trainError: number, recentErrors: number[], cfg: MonitoredSmoothingConfig, state: PrimarySmoothingState) => number`

Compute the monitored (primary) smoothed error given recent raw errors.

Behavior:
 - For SMA-like strategies uses the supplied window slice directly.
 - For EMA it mutates state.emaValue.
 - For adaptive-ema maintains dual EMA tracks inside state and returns the min for stability.
 - For median / gaussian / trimmed / wma applies algorithmic weighting as documented inline.

Inputs:
 - trainError: Current raw mean error for this iteration.
 - recentErrors: Chronological array (oldest->newest) of last N raw errors.
 - cfg: Algorithm selection + parameters.
 - state: Mutable smoothing state (ema / adaptive fields updated in-place).

Returns: Smoothed/monitored error metric (may equal trainError if no smoothing active).

### computePlateauMetric

`(trainError: number, plateauErrors: number[], cfg: PlateauSmoothingConfig, state: PlateauSmoothingState) => number`

Compute plateau metric (may differ in strategy from primary monitored error).
Only algorithms actually supported for plateau in current pipeline are SMA, median and EMA.
Provided flexibility keeps room for extension; unsupported types silently fallback to mean.

### CostFunction

`(target: number[], output: number[]) => number`

-----------------------------------------------------------------------------
Internal Type Definitions (documentation only; optional for callers)
-----------------------------------------------------------------------------

### detectMixedPrecisionOverflow

`(net: import("D:/code-practice/NeatapticTS/src/architecture/network").default, internalNet: any) => boolean`

Detect mixed precision overflow (NaN / Inf) in bias values if mixed precision enabled.
Side-effect: may clear internal trigger _forceNextOverflow.

### GradientClipConfig

Gradient clipping configuration accepted by options.gradientClip.

### handleOverflow

`(internalNet: any) => void`

Respond to a mixed precision overflow by shrinking loss scale & bookkeeping.

### maybeIncreaseLossScale

`(internalNet: any) => void`

Update dynamic loss scaling after a successful (non-overflow) optimizer step.

### MetricsHook

`(m: { iteration: number; error: number; plateauError?: number | undefined; gradNorm: number; }) => void`

Metrics hook signature.

### MixedPrecisionConfig

### MixedPrecisionDynamicConfig

Mixed precision configuration.

### MonitoredSmoothingConfig

Configuration passed to monitored (primary) smoothing computation.

### MovingAverageType

Moving average strategy identifiers.

### OptimizerConfigBase

Optimizer configuration (subset – delegated to node.applyBatchUpdatesWithOptimizer).

### PlateauSmoothingConfig

Configuration for plateau smoothing computation.

### PlateauSmoothingState

State container for plateau EMA smoothing.

### PrimarySmoothingState

---------------------------------------------------------------------------
Internal Helper Utilities (non-exported)
---------------------------------------------------------------------------
These functions encapsulate cohesive sub-steps of the training pipeline so the
main exported functions remain readable while preserving original behavior.
Each helper is intentionally pure where reasonable or documents its side-effects.

### ScheduleConfig

Schedule hook executed every N iterations.

### trainImpl

`(net: import("D:/code-practice/NeatapticTS/src/architecture/network").default, set: { input: number[]; output: number[]; }[], options: import("D:/code-practice/NeatapticTS/src/architecture/network/network.training").TrainingOptions) => { error: number; iterations: number; time: number; }`

High-level training orchestration with early stopping, smoothing & callbacks.

### TrainingOptions

Primary training options object (public shape).

### trainSetImpl

`(net: import("D:/code-practice/NeatapticTS/src/architecture/network").default, set: { input: number[]; output: number[]; }[], batchSize: number, accumulationSteps: number, currentRate: number, momentum: number, regularization: any, costFunction: (target: number[], output: number[]) => number, optimizer: any) => number`

Execute one full pass over dataset (epoch) with optional accumulation & adaptive optimizer.
Returns mean cost across processed samples.

### zeroAccumulatedGradients

`(net: import("D:/code-practice/NeatapticTS/src/architecture/network").default) => void`

Zero-out accumulated gradient buffers after an overflow to discard invalid updates.
