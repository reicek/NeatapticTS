# architecture/network/activate

## architecture/network/activate/network.activate.utils.types.ts

### ActivateRuntimeNetworkProps

Runtime network view used by the object-graph activation pipeline.

This intentionally describes the internal fields activation reads/writes
(training step, RNG, regularization knobs, and slab fast-path hooks).

### ActivationOutputBuffer

Pooled activation output array type acquired from the shared activation array pool.

### ActivationStats

Activation telemetry collected during a single activation pass.

### BATCH_INPUTS_COLLECTION_ERROR_MESSAGE

Error message used when batch activation receives a non-array container.

### BatchActivationContext

Shared state used by batch activation orchestration.

### BatchRowActivationContext

Shared state used while validating and activating one row in a batch.

### DEFAULT_MAX_ACTIVATION_DEPTH

Default hard limit for recursive activation depth in raw activation mode.

### INITIAL_OUTPUT_WRITE_INDEX

Initial write index used when collecting output activations.

### INPUT_NODE_TYPE

Node role label used by activation traversal for input neurons.

### NetworkLayer

Layer container type used by the layered activation paths.

### NetworkLayerNodes

Node collection type attached to a single network layer.

### NO_TRACE_FAST_SLAB_TRAINING_FLAG

Training flag value used by no-trace fast slab eligibility checks.

### NoTraceActivationContext

Shared state used by no-trace activation orchestration and helpers.

### NoTraceNodeTraversalContext

Shared state used for node traversal during no-trace activation.

### OUTPUT_NODE_TYPE

Node role label used by activation traversal for output neurons.

### OUTPUT_WRITE_INDEX_INCREMENT

Increment applied after writing one output activation value.

### RawActivationContext

Shared state used by raw activation orchestration.

### SingleNodeNoTraceActivationContext

Shared state used while activating one node during no-trace traversal.

### UNDEFINED_INPUT_LENGTH_TEXT

Fallback text for undefined input lengths when formatting validation errors.

### WeightNoiseApplyResult

Marker returned by weight-noise application to drive safe restore logic.

### WeightNoiseStats

Weight-noise telemetry collected during a single activation pass.

## architecture/network/activate/network.activate.utils.ts

### activate

```ts
activate(
  input: number[],
  training: boolean,
): number[]
```

Execute the main activation routine and return plain numeric outputs.

Parameters:
- `this` - Bound network instance.
- `input` - Input values with length matching network input count.
- `training` - Whether training-time stochastic behavior is enabled.

Returns: Output activation values.

### activateBatch

```ts
activateBatch(
  inputs: number[][],
  training: boolean,
): number[][]
```

Activate the network over a mini‑batch (array) of input vectors, returning a 2‑D array of outputs.

This helper simply loops, invoking {@link Network.activate} (or its bound variant) for each
sample. It is intentionally naive: no attempt is made to fuse operations across the batch.
For very large batch sizes or performance‑critical paths consider implementing a custom
vectorized backend that exploits SIMD, GPU kernels, or parallel workers.

Input validation occurs per row to surface the earliest mismatch with a descriptive index.

Parameters:
- `this` - - Bound Network instance.
- `inputs` - - Array of input vectors; each must have length == network.input.
- `training` - - Whether each activation should keep training traces.

Returns: 2‑D array: outputs[i] is the activation result for inputs[i].

Example:

const batchOut = net.activateBatch([[0,0,1],[1,0,0],[0,1,0]]);
console.log(batchOut.length); // 3 rows

### activateRaw

```ts
activateRaw(
  input: number[],
  training: boolean,
  maxActivationDepth: number,
): number[]
```

Thin semantic alias to the network's main activation path.

At present this simply forwards to {@link Network.activate}. The indirection is useful for:
 - Future differentiation between raw (immediate) activation and a mode that performs reuse /
   staged batching logic.
 - Providing a stable exported symbol for external tooling / instrumentation.

Parameters:
- `this` - - Bound Network instance.
- `input` - - Input vector (length == network.input).
- `training` - - Whether to retain training traces / gradients (delegated downstream).
- `maxActivationDepth` - - Guard against runaway recursion / cyclic activation attempts.

Returns: Implementation-defined result of Network.activate (typically an output vector).

Example:

const y = net.activateRaw([0,1,0]);

### gaussianRand

```ts
gaussianRand(
  rng: () => number,
): number
```

Produce a normally distributed random sample using the Box-Muller transform.

Parameters:
- `rng` - Pseudo-random source in the interval [0, 1).

Returns: Standard normal sample with mean 0 and variance 1.

### noTraceActivate

```ts
noTraceActivate(
  input: number[],
): number[]
```

Perform a forward pass without creating or updating training / gradient traces.

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
- `this` - - Bound Network instance.
- `input` - - Flat numeric vector whose length must equal network.input.

Returns: Array of output neuron activations (length == network.output).

Example:

const out = net.noTraceActivate([0.1, 0.2, 0.3]);
console.log(out); // => e.g. [0.5123, 0.0441]

## architecture/network/activate/network.activate.raw.utils.ts

### activateViaNetworkDelegate

```ts
activateViaNetworkDelegate(
  activationContext: RawActivationContext,
): number[]
```

Delegate raw activation to the core network activation implementation.

Parameters:
- `activationContext` - - Shared raw activation state.

Returns: Activation output vector.

### activateWithSelectedReusePath

```ts
activateWithSelectedReusePath(
  activationContext: RawActivationContext,
): number[]
```

Select the raw activation execution path based on runtime reuse configuration.

Parameters:
- `activationContext` - - Shared raw activation state.

Returns: Activation output vector.

### executeRawActivation

```ts
executeRawActivation(
  activationContext: RawActivationContext,
): number[]
```

Execute raw activation through the network delegate using a compact orchestration flow.

This helper keeps the exported activation method focused on context creation while this
module owns the execution path and future branching behavior.

Parameters:
- `activationContext` - - Shared raw activation state.

Returns: Activation output vector from the network delegate.

## architecture/network/activate/network.activate.core.utils.ts

### acquireOutputBuffer

```ts
acquireOutputBuffer(
  outputSize: number,
): ActivationArray
```

Acquire a pooled activation output buffer for the current output width.

Parameters:
- `outputSize` - Number of output slots.

Returns: Mutable pooled output buffer.

### activate

```ts
activate(
  input: number[],
  training: boolean,
): number[]
```

Execute the main activation routine and return plain numeric outputs.

Parameters:
- `this` - Bound network instance.
- `input` - Input values with length matching network input count.
- `training` - Whether training-time stochastic behavior is enabled.

Returns: Output activation values.

### activateLayer

```ts
activateLayer(
  currentLayer: default,
  layerIndex: number,
  inputVector: number[],
  isTraining: boolean,
): number[]
```

Activate one layer, routing input only for the first layer.

Parameters:
- `currentLayer` - Layer instance to activate.
- `layerIndex` - Layer index.
- `inputVector` - Network input vector.
- `isTraining` - Training-time flag.

Returns: Layer activations.

### activateLayeredNetworkWithDropout

```ts
activateLayeredNetworkWithDropout(
  network: default,
  runtimeNetwork: ActivateRuntimeNetworkProps,
  inputVector: number[],
  isTraining: boolean,
  outputBuffer: ActivationArray,
  stats: ActivationStats,
): void
```

Run layered activation with dropout masks and no stochastic-depth skips.

Parameters:
- `network` - Network being activated.
- `runtimeNetwork` - Runtime activation internals.
- `inputVector` - Input vector.
- `isTraining` - Training-time flag.
- `outputBuffer` - Mutable output buffer.
- `stats` - Activation stats accumulator.

Returns: Nothing.

### activateLayeredNetworkWithStochasticDepth

```ts
activateLayeredNetworkWithStochasticDepth(
  network: default,
  runtimeNetwork: ActivateRuntimeNetworkProps,
  inputVector: number[],
  isTraining: boolean,
  outputBuffer: ActivationArray,
  stats: ActivationStats,
): void
```

Run layered activation with stochastic-depth skipping and inverse-survival scaling.

Parameters:
- `network` - Network being activated.
- `runtimeNetwork` - Runtime activation internals.
- `inputVector` - Input vector.
- `isTraining` - Training-time flag.
- `outputBuffer` - Mutable output buffer.
- `stats` - Activation stats accumulator.

Returns: Nothing.

### activateNodeNetworkFallback

```ts
activateNodeNetworkFallback(
  network: default,
  runtimeNetwork: ActivateRuntimeNetworkProps,
  inputVector: number[],
  isTraining: boolean,
  outputBuffer: ActivationArray,
  stats: ActivationStats,
): void
```

Run fallback node-by-node activation for networks without explicit layer definitions.

Parameters:
- `network` - Network being activated.
- `runtimeNetwork` - Runtime activation internals.
- `inputVector` - Input vector.
- `isTraining` - Training-time flag.
- `outputBuffer` - Mutable output buffer.
- `stats` - Activation stats accumulator.

Returns: Nothing.

### activateNodesAndCollectOutputs

```ts
activateNodesAndCollectOutputs(
  nodes: default[],
  inputVector: number[],
  outputBuffer: ActivationArray,
): void
```

Activate raw nodes in order and collect output-node activations into output buffer.

Parameters:
- `nodes` - Network nodes in activation order.
- `inputVector` - Input vector.
- `outputBuffer` - Mutable output buffer.

Returns: Nothing.

### applyDropConnect

```ts
applyDropConnect(
  network: default,
  runtimeNetwork: ActivateRuntimeNetworkProps,
  isTraining: boolean,
  stats: ActivationStats,
): void
```

Apply drop-connect masking and restore original weights where required.

Parameters:
- `network` - Network being activated.
- `runtimeNetwork` - Runtime activation internals.
- `isTraining` - Training-time flag.
- `stats` - Activation stats accumulator.

Returns: Nothing.

### applyFallbackHiddenDropout

```ts
applyFallbackHiddenDropout(
  hiddenNodes: default[],
  runtimeNetwork: ActivateRuntimeNetworkProps,
  dropoutProbability: number,
  isTraining: boolean,
  stats: ActivationStats,
): void
```

Apply fallback dropout for hidden nodes in raw node traversal mode.

Parameters:
- `hiddenNodes` - Hidden nodes.
- `runtimeNetwork` - Runtime activation internals.
- `dropoutProbability` - Dropout probability.
- `isTraining` - Training-time flag.
- `stats` - Activation stats accumulator.

Returns: Nothing.

### applyFallbackWeightNoise

```ts
applyFallbackWeightNoise(
  network: default,
  runtimeNetwork: ActivateRuntimeNetworkProps,
  isTraining: boolean,
): void
```

Apply raw fallback weight noise to all connections using global standard deviation.

Parameters:
- `network` - Network being activated.
- `runtimeNetwork` - Runtime activation internals.
- `isTraining` - Training-time flag.

Returns: Nothing.

### applyHiddenLayerDropout

```ts
applyHiddenLayerDropout(
  layer: default,
  rawActivations: number[],
  runtimeNetwork: ActivateRuntimeNetworkProps,
  dropoutProbability: number,
  isTraining: boolean,
  stats: ActivationStats,
): void
```

Apply dropout masks to hidden layer nodes and enforce at least one active node.

Parameters:
- `layer` - Hidden layer instance.
- `rawActivations` - Raw layer activations.
- `runtimeNetwork` - Runtime activation internals.
- `dropoutProbability` - Layer dropout probability.
- `isTraining` - Training-time flag.
- `stats` - Activation stats accumulator.

Returns: Nothing.

### applyTrainingDropConnect

```ts
applyTrainingDropConnect(
  network: default,
  runtimeNetwork: ActivateRuntimeNetworkProps,
  stats: ActivationStats,
): void
```

Apply training-time drop-connect masks to each connection.

Parameters:
- `network` - Network being activated.
- `runtimeNetwork` - Runtime activation internals.
- `stats` - Activation stats accumulator.

Returns: Nothing.

### applyTrainingWeightNoise

```ts
applyTrainingWeightNoise(
  network: default,
  runtimeNetwork: ActivateRuntimeNetworkProps,
  isTraining: boolean,
): WeightNoiseApplyResult
```

Apply per-connection training noise for the main activation flow.

Parameters:
- `network` - Network being activated.
- `runtimeNetwork` - Runtime activation internals.
- `isTraining` - Training-time flag.

Returns: Applied-state information for downstream restore logic.

### collectHiddenNodes

```ts
collectHiddenNodes(
  nodes: default[],
): default[]
```

Collect hidden nodes from a raw node list.

Parameters:
- `nodes` - Network node collection.

Returns: Hidden-only node list.

### containsInvalidProbability

```ts
containsInvalidProbability(
  probabilities: number[],
): boolean
```

Check whether a probability vector contains values outside the (0, 1] interval.

Parameters:
- `probabilities` - Candidate probability vector.

Returns: True when one or more probabilities are invalid.

### createActivationStats

```ts
createActivationStats(
  totalConnections: number,
): ActivationStats
```

Create activation statistics container for the current pass.

Parameters:
- `totalConnections` - Number of network connections.

Returns: Initialized activation stats object.

### createWeightNoiseStats

```ts
createWeightNoiseStats(): WeightNoiseStats
```

Create the weight-noise statistics record with zeroed aggregates.

Returns: Zero-initialized weight-noise stats.

### decideLayerSkip

```ts
decideLayerSkip(
  network: default,
  runtimeNetwork: ActivateRuntimeNetworkProps,
  currentLayerNodeCount: number,
  layerIndex: number,
  isTraining: boolean,
  previousLayerActivations: number[] | undefined,
): { shouldSkipLayer: boolean; surviveProbability: number; }
```

Decide whether a hidden layer should be skipped in stochastic-depth mode.

Parameters:
- `network` - Network being activated.
- `runtimeNetwork` - Runtime activation internals.
- `currentLayerNodeCount` - Number of nodes in current layer.
- `layerIndex` - Current layer index.
- `isTraining` - Training-time flag.
- `previousLayerActivations` - Last computed layer activations.

Returns: Skip decision and survival probability for the layer.

### executeActivationPath

```ts
executeActivationPath(
  network: default,
  runtimeNetwork: ActivateRuntimeNetworkProps,
  inputVector: number[],
  isTraining: boolean,
  outputBuffer: ActivationArray,
  stats: ActivationStats,
): void
```

Execute one of the three activation branches: stochastic layers, standard layers, or raw nodes.

Parameters:
- `network` - Network being activated.
- `runtimeNetwork` - Runtime activation internals.
- `inputVector` - Input vector.
- `isTraining` - Training-time flag.
- `outputBuffer` - Mutable output buffer.
- `stats` - Activation stats accumulator.

Returns: Nothing.

### finalizeNodePathWeightNoiseRestore

```ts
finalizeNodePathWeightNoiseRestore(
  network: default,
  isTraining: boolean,
  appliedWeightNoise: boolean,
): void
```

Restore temporary weight-noise values for fallback node path only.

Parameters:
- `network` - Network being activated.
- `isTraining` - Training-time flag.
- `appliedWeightNoise` - Whether weight noise was applied during this pass.

Returns: Nothing.

### finalizeTrainingStepAndStats

```ts
finalizeTrainingStepAndStats(
  runtimeNetwork: ActivateRuntimeNetworkProps,
  stats: ActivationStats,
  isTraining: boolean,
): void
```

Finalize training counters and attach activation statistics to runtime state.

Parameters:
- `runtimeNetwork` - Runtime activation internals.
- `stats` - Activation stats.
- `isTraining` - Training-time flag.

Returns: Nothing.

### findSourceLayerIndex

```ts
findSourceLayerIndex(
  network: default,
  connection: default,
): number
```

Find the layer index containing a connection source node.

Parameters:
- `network` - Network being activated.
- `connection` - Connection to inspect.

Returns: Layer index for source node, or -1 when not found.

### gaussianRand

```ts
gaussianRand(
  rng: () => number,
): number
```

Produce a normally distributed random sample using the Box-Muller transform.

Parameters:
- `rng` - Pseudo-random source in the interval [0, 1).

Returns: Standard normal sample with mean 0 and variance 1.

### hasCompatibleSkipState

```ts
hasCompatibleSkipState(
  previousLayerActivations: number[] | undefined,
  currentLayerNodeCount: number,
): boolean
```

Validate whether previous activations can be reused as skip pass-through output.

Parameters:
- `previousLayerActivations` - Last computed layer activations.
- `currentLayerNodeCount` - Current layer node count.

Returns: True when pass-through activations are compatible.

### hasLayeredNetwork

```ts
hasLayeredNetwork(
  network: default,
): boolean
```

Check whether the network has at least one explicit layer.

Parameters:
- `network` - Network being activated.

Returns: True when layered activation path should run.

### hasLayeredNetworkWithStochasticDepth

```ts
hasLayeredNetworkWithStochasticDepth(
  network: default,
  runtimeNetwork: ActivateRuntimeNetworkProps,
): boolean
```

Check whether the network has layers and stochastic-depth configuration for layer skipping path.

Parameters:
- `network` - Network being activated.
- `runtimeNetwork` - Runtime activation internals.

Returns: True when stochastic-depth layer path should run.

### hasOriginalWeightNoise

```ts
hasOriginalWeightNoise(
  connection: default,
): boolean
```

Check whether a connection already has an original weight-noise snapshot.

Parameters:
- `connection` - Connection to inspect.

Returns: True when snapshot exists.

### isHiddenLayer

```ts
isHiddenLayer(
  layerIndex: number,
  totalLayerCount: number,
): boolean
```

Check whether a layer index refers to a hidden layer in a layered network.

Parameters:
- `layerIndex` - Current layer index.
- `totalLayerCount` - Number of layers in the network.

Returns: True when the layer is hidden.

### persistOriginalWeightNoise

```ts
persistOriginalWeightNoise(
  connection: default,
): void
```

Store current connection weight before applying temporary weight-noise modifications.

Parameters:
- `connection` - Connection to persist.

Returns: Nothing.

### prepareTopologyForActivation

```ts
prepareTopologyForActivation(
  runtimeNetwork: ActivateRuntimeNetworkProps,
): void
```

Ensure topological order is refreshed before activation when acyclic mode requires it.

Parameters:
- `runtimeNetwork` - Runtime activation internals.

Returns: Nothing.

### recordSkippedLayer

```ts
recordSkippedLayer(
  network: default,
  stats: ActivationStats,
  layerIndex: number,
): void
```

Record a skipped layer in runtime and stats trackers.

Parameters:
- `network` - Network being activated.
- `stats` - Activation stats accumulator.
- `layerIndex` - Skipped layer index.

Returns: Nothing.

### releaseBufferAndCreateResult

```ts
releaseBufferAndCreateResult(
  outputBuffer: ActivationArray,
): number[]
```

Release pooled output buffer and return a detached plain array copy.

Parameters:
- `outputBuffer` - Mutable pooled output buffer.

Returns: Plain array of output values.

### resetSkippedLayers

```ts
resetSkippedLayers(
  network: default,
): void
```

Clear the runtime list of skipped layers before current activation pass.

Parameters:
- `network` - Network runtime owner.

Returns: Nothing.

### resolveConnectionNoiseStd

```ts
resolveConnectionNoiseStd(
  network: default,
  runtimeNetwork: ActivateRuntimeNetworkProps,
  connection: default,
  fallbackStandardDeviation: number,
): number
```

Resolve connection-specific weight-noise standard deviation, including per-hidden overrides.

Parameters:
- `network` - Network being activated.
- `runtimeNetwork` - Runtime activation internals.
- `connection` - Current connection.
- `fallbackStandardDeviation` - Global fallback deviation.

Returns: Effective standard deviation for this connection.

### resolveDynamicWeightNoiseStd

```ts
resolveDynamicWeightNoiseStd(
  runtimeNetwork: ActivateRuntimeNetworkProps,
): number
```

Resolve the training-step adjusted global weight-noise standard deviation.

Parameters:
- `runtimeNetwork` - Runtime activation internals.

Returns: Effective weight-noise standard deviation for current training step.

### restoreDropConnectWeights

```ts
restoreDropConnectWeights(
  network: default,
): void
```

Restore drop-connect modified weights and normalize all masks back to one.

Parameters:
- `network` - Network being activated.

Returns: Nothing.

### restoreOriginalDropConnectWeight

```ts
restoreOriginalDropConnectWeight(
  connection: default,
): void
```

Restore and clear original connection weight after drop-connect.

Parameters:
- `connection` - Connection to restore.

Returns: Nothing.

### restoreOriginalWeightNoise

```ts
restoreOriginalWeightNoise(
  connection: default,
): void
```

Restore and clear the original weight-noise snapshot for a connection.

Parameters:
- `connection` - Connection to restore.

Returns: Nothing.

### scaleActivations

```ts
scaleActivations(
  activations: number[],
  scaleFactor: number,
): number[]
```

Create a new activation vector by multiplying each activation by a scale factor.

Parameters:
- `activations` - Source activation vector.
- `scaleFactor` - Multiplicative scale factor.

Returns: Scaled activation vector.

### setAllMasksToOne

```ts
setAllMasksToOne(
  nodes: default[],
): void
```

Set mask value to one for every node in a layer.

Parameters:
- `nodes` - Layer nodes to normalize.

Returns: Nothing.

### setDropConnectMask

```ts
setDropConnectMask(
  connection: default,
  dropConnectMask: number,
): void
```

Set drop-connect mask value for a connection.

Parameters:
- `connection` - Connection to annotate.
- `dropConnectMask` - Drop-connect mask value.

Returns: Nothing.

### setLastSampledNoise

```ts
setLastSampledNoise(
  connection: default,
  sampledNoise: number,
): void
```

Persist last sampled weight-noise value for a connection.

Parameters:
- `connection` - Connection to annotate.
- `sampledNoise` - Last sampled noise.

Returns: Nothing.

### stashOriginalDropConnectWeight

```ts
stashOriginalDropConnectWeight(
  connection: default,
): void
```

Store original connection weight before drop-connect zeroing.

Parameters:
- `connection` - Connection to persist.

Returns: Nothing.

### tryFastSlabActivation

```ts
tryFastSlabActivation(
  runtimeNetwork: ActivateRuntimeNetworkProps,
  inputVector: number[],
  isTraining: boolean,
): number[] | undefined
```

Attempt fast slab activation and safely fall back to regular activation on failure.

Parameters:
- `runtimeNetwork` - Runtime activation internals.
- `inputVector` - Input vector.
- `isTraining` - Training-time flag.

Returns: Fast slab output when available, otherwise undefined.

### updateStochasticDepthFromSchedule

```ts
updateStochasticDepthFromSchedule(
  runtimeNetwork: ActivateRuntimeNetworkProps,
  isTraining: boolean,
): void
```

Update stochastic depth probabilities using a training schedule when valid.

Parameters:
- `runtimeNetwork` - Runtime activation internals.
- `isTraining` - Training-time flag.

Returns: Nothing.

### validateInputVector

```ts
validateInputVector(
  network: default,
  inputVector: number[],
): void
```

Validate that the incoming input vector exists and matches expected input size.

Parameters:
- `network` - Network being activated.
- `inputVector` - Input vector to validate.

Returns: Nothing.

### validateNetworkNodes

```ts
validateNetworkNodes(
  network: default,
): void
```

Assert that the network contains nodes before executing activation routines.

Parameters:
- `network` - Network being activated.

Returns: Nothing.

### writeLayerActivationsToOutput

```ts
writeLayerActivationsToOutput(
  layerActivations: number[] | undefined,
  outputBuffer: ActivationArray,
  outputSize: number,
): void
```

Copy final layer activations into the pooled network output buffer.

Parameters:
- `layerActivations` - Final layer activations.
- `outputBuffer` - Mutable output buffer.
- `outputSize` - Maximum output width.

Returns: Nothing.

## architecture/network/activate/network.activate.batch.utils.ts

### activateSingleBatchRow

```ts
activateSingleBatchRow(
  rowActivationContext: BatchRowActivationContext,
): number[]
```

Validate and activate one batch row.

Parameters:
- `rowActivationContext` - - Shared state for one batch-row activation.

Returns: Activation output vector for the row.

### activateValidatedBatchRows

```ts
activateValidatedBatchRows(
  activationContext: BatchActivationContext,
): number[][]
```

Activate each row in a validated batch matrix.

Parameters:
- `activationContext` - - Shared batch activation state.

Returns: Matrix of activation outputs.

### assertBatchInputCollection

```ts
assertBatchInputCollection(
  batchInputs: number[][],
): void
```

Validate that the batch input collection is an array of rows.

Parameters:
- `batchInputs` - - Candidate batch input collection.

Returns: Nothing.

### assertBatchRowInputSize

```ts
assertBatchRowInputSize(
  rowActivationContext: BatchRowActivationContext,
): void
```

Validate one batch row dimensionality.

Parameters:
- `rowActivationContext` - - Shared state for one batch-row activation.

Returns: Nothing.

### buildBatchRowInputSizeMismatchMessage

```ts
buildBatchRowInputSizeMismatchMessage(
  rowActivationContext: BatchRowActivationContext,
): string
```

Build a descriptive mismatch message for invalid batch row input dimensions.

Parameters:
- `rowActivationContext` - - Shared state for one batch-row activation.

Returns: Formatted error message for invalid row dimensionality.

### executeBatchActivation

```ts
executeBatchActivation(
  activationContext: BatchActivationContext,
): number[][]
```

Execute mini-batch activation with top-level shape validation and per-row checks.

The orchestration keeps behavior deterministic by validating the container first,
then validating each row before delegating to the core network activation function.

Parameters:
- `activationContext` - - Shared batch activation state.

Returns: Matrix of activation outputs.

### formatInputLengthForMessage

```ts
formatInputLengthForMessage(
  inputVector: number[],
): string
```

Convert input length into a display-safe string for error messaging.

Parameters:
- `inputVector` - - Candidate batch row input vector.

Returns: Numeric length as string or predefined undefined text.

### isBatchRowInputSizeValid

```ts
isBatchRowInputSizeValid(
  rowActivationContext: BatchRowActivationContext,
): boolean
```

Determine whether one batch row matches the expected input dimensionality.

Parameters:
- `rowActivationContext` - - Shared state for one batch-row activation.

Returns: True when row size is valid.

## architecture/network/activate/network.activate.helpers.utils.ts

### createBatchActivationContext

```ts
createBatchActivationContext(
  network: default,
  batchInputs: number[][],
  isTraining: boolean,
): BatchActivationContext
```

Build shared batch activation context for helper orchestration.

Parameters:
- `network` - - Network instance bound to the activation call.
- `batchInputs` - - Input matrix supplied by the caller.
- `isTraining` - - Whether activation should retain training traces.

Returns: Fully populated batch activation context.

### createNoTraceActivationContext

```ts
createNoTraceActivationContext(
  network: default,
  inputVector: number[],
): NoTraceActivationContext
```

Build shared no-trace activation context for helper orchestration.

Parameters:
- `network` - - Network instance bound to the activation call.
- `inputVector` - - Input activation vector supplied by the caller.

Returns: Fully populated no-trace activation context.

### createRawActivationContext

```ts
createRawActivationContext(
  network: default,
  inputVector: number[],
  isTraining: boolean,
  maximumActivationDepth: number,
): RawActivationContext
```

Build shared raw activation context for helper orchestration.

Parameters:
- `network` - - Network instance bound to the activation call.
- `inputVector` - - Input activation vector supplied by the caller.
- `isTraining` - - Whether activation should retain training traces.
- `maximumActivationDepth` - - Guard against runaway activation depth.

Returns: Fully populated raw activation context.

### executeBatchActivation

```ts
executeBatchActivation(
  activationContext: BatchActivationContext,
): number[][]
```

Execute mini-batch activation with top-level shape validation and per-row checks.

The orchestration keeps behavior deterministic by validating the container first,
then validating each row before delegating to the core network activation function.

Parameters:
- `activationContext` - - Shared batch activation state.

Returns: Matrix of activation outputs.

### executeNoTraceActivation

```ts
executeNoTraceActivation(
  activationContext: NoTraceActivationContext,
): number[]
```

Execute no-trace activation with a fast-path attempt and deterministic fallback traversal.

The orchestration follows a strict sequence: refresh order guarantees, validate input shape,
try fast slab inference, then compute outputs through node traversal when needed.

Parameters:
- `activationContext` - - Shared no-trace activation state.

Returns: Output activation vector detached from pooled storage.

### executeRawActivation

```ts
executeRawActivation(
  activationContext: RawActivationContext,
): number[]
```

Execute raw activation through the network delegate using a compact orchestration flow.

This helper keeps the exported activation method focused on context creation while this
module owns the execution path and future branching behavior.

Parameters:
- `activationContext` - - Shared raw activation state.

Returns: Activation output vector from the network delegate.

## architecture/network/activate/network.activate.notrace.utils.ts

### activateWithoutTraceUsingNodeIteration

```ts
activateWithoutTraceUsingNodeIteration(
  activationContext: NoTraceActivationContext,
): number[]
```

Execute no-trace activation through node traversal and pooled output collection.

Parameters:
- `activationContext` - - Shared no-trace activation state.

Returns: Detached output activation vector.

### assertInputMatchesNetworkInputSize

```ts
assertInputMatchesNetworkInputSize(
  activationContext: NoTraceActivationContext,
): void
```

Validate that the input vector length matches expected network input dimensionality.

Parameters:
- `activationContext` - - Shared no-trace activation state.

Returns: Nothing.

### buildInputSizeMismatchMessage

```ts
buildInputSizeMismatchMessage(
  activationContext: NoTraceActivationContext,
): string
```

Build a descriptive input mismatch message for activation validation errors.

Parameters:
- `activationContext` - - Shared no-trace activation state.

Returns: Formatted mismatch error message.

### canUseNoTraceFastSlab

```ts
canUseNoTraceFastSlab(
  activationContext: NoTraceActivationContext,
): boolean
```

Determine whether fast slab activation is available for no-trace execution mode.

Parameters:
- `activationContext` - - Shared no-trace activation state.

Returns: True when slab execution is available for inference mode.

### detachPooledOutputBuffer

```ts
detachPooledOutputBuffer(
  pooledOutputBuffer: ActivationArray,
): number[]
```

Clone pooled output storage into a detached plain array.

Parameters:
- `pooledOutputBuffer` - - Pooled activation output storage.

Returns: Detached output activation vector.

### executeNoTraceActivation

```ts
executeNoTraceActivation(
  activationContext: NoTraceActivationContext,
): number[]
```

Execute no-trace activation with a fast-path attempt and deterministic fallback traversal.

The orchestration follows a strict sequence: refresh order guarantees, validate input shape,
try fast slab inference, then compute outputs through node traversal when needed.

Parameters:
- `activationContext` - - Shared no-trace activation state.

Returns: Output activation vector detached from pooled storage.

### formatInputLengthForMessage

```ts
formatInputLengthForMessage(
  inputVector: number[],
): string
```

Convert input length into a display-safe string for error messaging.

Parameters:
- `inputVector` - - Candidate activation input vector.

Returns: Numeric length as string or predefined undefined text.

### isInputVectorLengthValid

```ts
isInputVectorLengthValid(
  activationContext: NoTraceActivationContext,
): boolean
```

Check whether the input vector has a valid length for activation.

Parameters:
- `activationContext` - - Shared no-trace activation state.

Returns: True when the input vector is an array with expected length.

### refreshTopologicalOrderWhenRequired

```ts
refreshTopologicalOrderWhenRequired(
  activationContext: NoTraceActivationContext,
): void
```

Refresh cached topological order when acyclic mode is active and marked dirty.

Parameters:
- `activationContext` - - Shared no-trace activation state.

Returns: Nothing.

### tryActivateWithFastSlab

```ts
tryActivateWithFastSlab(
  activationContext: NoTraceActivationContext,
): number[] | null
```

Attempt fast slab activation and return null when slab execution is unavailable or fails.

Parameters:
- `activationContext` - - Shared no-trace activation state.

Returns: Fast slab output when successful, otherwise null.

## architecture/network/activate/network.activate.contexts.utils.ts

### createBatchActivationContext

```ts
createBatchActivationContext(
  network: default,
  batchInputs: number[][],
  isTraining: boolean,
): BatchActivationContext
```

Build shared batch activation context for helper orchestration.

Parameters:
- `network` - - Network instance bound to the activation call.
- `batchInputs` - - Input matrix supplied by the caller.
- `isTraining` - - Whether activation should retain training traces.

Returns: Fully populated batch activation context.

### createNoTraceActivationContext

```ts
createNoTraceActivationContext(
  network: default,
  inputVector: number[],
): NoTraceActivationContext
```

Build shared no-trace activation context for helper orchestration.

Parameters:
- `network` - - Network instance bound to the activation call.
- `inputVector` - - Input activation vector supplied by the caller.

Returns: Fully populated no-trace activation context.

### createRawActivationContext

```ts
createRawActivationContext(
  network: default,
  inputVector: number[],
  isTraining: boolean,
  maximumActivationDepth: number,
): RawActivationContext
```

Build shared raw activation context for helper orchestration.

Parameters:
- `network` - - Network instance bound to the activation call.
- `inputVector` - - Input activation vector supplied by the caller.
- `isTraining` - - Whether activation should retain training traces.
- `maximumActivationDepth` - - Guard against runaway activation depth.

Returns: Fully populated raw activation context.

### toNetworkInternals

```ts
toNetworkInternals(
  network: default,
): ActivateNetworkInternals
```

Convert a network instance into the activation internals interface used by helper modules.

Parameters:
- `network` - - Runtime network instance.

Returns: Network internals view used by activation helper modules.

## architecture/network/activate/network.activate.notrace.traversal.utils.ts

### activateHiddenNode

```ts
activateHiddenNode(
  networkNode: default,
): void
```

Activate a hidden node without trace bookkeeping.

Parameters:
- `networkNode` - - Hidden-role node to activate.

Returns: Nothing.

### activateInputNode

```ts
activateInputNode(
  activationContext: SingleNodeNoTraceActivationContext,
): void
```

Activate an input node using the matching input vector value.

Parameters:
- `activationContext` - - Node-specific activation state.

Returns: Nothing.

### activateOutputNodeAndAdvanceIndex

```ts
activateOutputNodeAndAdvanceIndex(
  activationContext: SingleNodeNoTraceActivationContext,
): number
```

Activate an output node, write the activation value, and advance the output index.

Parameters:
- `activationContext` - - Node-specific activation state.

Returns: Next output write index.

### activateSingleNodeWithoutTrace

```ts
activateSingleNodeWithoutTrace(
  activationContext: SingleNodeNoTraceActivationContext,
): number
```

Activate one node and return the next output write index.

Parameters:
- `activationContext` - - Node-specific activation state.

Returns: Updated output write index.

### isInputNode

```ts
isInputNode(
  networkNode: default,
): boolean
```

Determine whether a node is an input-role node.

Parameters:
- `networkNode` - - Node under traversal.

Returns: True when node role is input.

### isOutputNode

```ts
isOutputNode(
  networkNode: default,
): boolean
```

Determine whether a node is an output-role node.

Parameters:
- `networkNode` - - Node under traversal.

Returns: True when node role is output.

### populatePooledOutputBufferFromNodes

```ts
populatePooledOutputBufferFromNodes(
  traversalContext: NoTraceNodeTraversalContext,
): void
```

Traverse nodes in activation order and write output activations into pooled storage.

This helper isolates traversal concerns from no-trace orchestration so the main flow
can remain focused on high-level activation phases.

Parameters:
- `traversalContext` - - Inputs required to process each node and collect outputs.

Returns: Nothing.
