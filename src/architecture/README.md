# architecture

## architecture/node.ts

### node

Node (Neuron)
=============
Fundamental computational unit: aggregates weighted inputs, applies an activation
function (squash) and emits an activation value. Supports:
 - Types: 'input' | 'hidden' | 'output' (affects bias initialization & error handling)
 - Recurrent self‑connections & gated connections (for dynamic / RNN behavior)
 - Dropout mask (`mask`), momentum terms, eligibility & extended traces (for
   a variety of learning rules beyond simple backprop).

Educational note: Traces (`eligibility` and `xtrace`) illustrate how recurrent credit
assignment works in algorithms like RTRL / policy gradients. They are updated only when
using the traced activation path (`activate`) vs `noTraceActivate` (inference fast path).

### default

#### _activateCore

```ts
_activateCore(
  withTrace: boolean,
  input: number | undefined,
): number
```

Internal shared implementation for activate/noTraceActivate.

Parameters:
- `withTrace` - Whether to update eligibility traces.
- `input` - Optional externally supplied activation (bypasses weighted sum if provided).

#### _globalNodeIndex

Global index counter for assigning unique indices to nodes.

#### _safeUpdateWeight

```ts
_safeUpdateWeight(
  connection: default,
  delta: number,
): void
```

Internal helper to safely update a connection weight with clipping and NaN checks.

#### activate

```ts
activate(
  input: number | undefined,
): number
```

Activates the node, calculating its output value based on inputs and state.
This method also calculates eligibility traces (`xtrace`) used for training recurrent connections.

The activation process involves:
1. Calculating the node's internal state (`this.state`) based on:
   - Incoming connections' weighted activations.
   - The recurrent self-connection's weighted state from the previous timestep (`this.old`).
   - The node's bias.
2. Applying the activation function (`this.squash`) to the state to get the activation (`this.activation`).
3. Applying the dropout mask (`this.mask`).
4. Calculating the derivative of the activation function.
5. Updating the gain of connections gated by this node.
6. Calculating and updating eligibility traces for incoming connections.

Parameters:
- `input` - Optional input value. If provided, sets the node's activation directly (used for input nodes).

Returns: The calculated activation value of the node.

#### activation

The output value of the node after applying the activation function. This is the value transmitted to connected nodes.

#### applyBatchUpdates

```ts
applyBatchUpdates(
  momentum: number,
): void
```

Applies accumulated batch updates to incoming and self connections and this node's bias.
Uses momentum in a Nesterov-compatible way: currentDelta = accumulated + momentum * previousDelta.
Resets accumulators after applying. Safe to call on any node type.

Parameters:
- `momentum` - Momentum factor (0 to disable)

#### applyBatchUpdatesWithOptimizer

```ts
applyBatchUpdatesWithOptimizer(
  opts: { type: "sgd" | "rmsprop" | "adagrad" | "adam" | "adamw" | "amsgrad" | "adamax" | "nadam" | "radam" | "lion" | "adabelief" | "lookahead"; momentum?: number | undefined; beta1?: number | undefined; beta2?: number | undefined; eps?: number | undefined; weightDecay?: number | undefined; lrScale?: number | undefined; t?: number | undefined; baseType?: string | undefined; la_k?: number | undefined; la_alpha?: number | undefined; },
): void
```

Extended batch update supporting multiple optimizers.

Applies accumulated (batch) gradients stored in `totalDeltaWeight` / `totalDeltaBias` to the
underlying weights and bias using the selected optimization algorithm. Supports both classic
SGD (with Nesterov-style momentum via preceding propagate logic) and a collection of adaptive
optimizers. After applying an update, gradient accumulators are reset to 0.

Supported optimizers (type):
 - 'sgd'      : Standard gradient descent with optional momentum.
 - 'rmsprop'  : Exponential moving average of squared gradients (cache) to normalize step.
 - 'adagrad'  : Accumulate squared gradients; learning rate effectively decays per weight.
 - 'adam'     : Bias‑corrected first (m) & second (v) moment estimates.
 - 'adamw'    : Adam with decoupled weight decay (applied after adaptive step).
 - 'amsgrad'  : Adam variant maintaining a maximum of past v (vhat) to enforce non‑increasing step size.
 - 'adamax'   : Adam variant using the infinity norm (u) instead of second moment.
 - 'nadam'    : Adam + Nesterov momentum style update (lookahead on first moment).
 - 'radam'    : Rectified Adam – warms up variance by adaptively rectifying denominator when sample size small.
 - 'lion'     : Uses sign of combination of two momentum buffers (beta1 & beta2) for update direction only.
 - 'adabelief': Adam-like but second moment on (g - m) (gradient surprise) for variance reduction.
 - 'lookahead': Wrapper; performs k fast optimizer steps then interpolates (alpha) towards a slow (shadow) weight.

Options:
 - momentum     : (SGD) momentum factor (Nesterov handled in propagate when update=true).
 - beta1/beta2  : Exponential decay rates for first/second moments (Adam family, Lion, AdaBelief, etc.).
 - eps          : Numerical stability epsilon added to denominator terms.
 - weightDecay  : Decoupled weight decay (AdamW) or additionally applied after main step when adamw selected.
 - lrScale      : Learning rate scalar already scheduled externally (passed as currentRate).
 - t            : Global step (1-indexed) for bias correction / rectification.
 - baseType     : Underlying optimizer for lookahead (not itself lookahead).
 - la_k         : Lookahead synchronization interval (number of fast steps).
 - la_alpha     : Interpolation factor towards slow (shadow) weights/bias at sync points.

Internal per-connection temp fields (created lazily):
 - firstMoment / secondMoment / maxSecondMoment / infinityNorm : Moment / variance / max variance / infinity norm caches.
 - gradientAccumulator : Single accumulator (RMSProp / AdaGrad).
 - previousDeltaWeight : For classic SGD momentum.
 - lookaheadShadowWeight / _la_shadowBias : Lookahead shadow copies.

Safety: We clip extreme weight / bias magnitudes and guard against NaN/Infinity.

Parameters:
- `opts` - Optimizer configuration (see above).

#### bias

The bias value of the node. Added to the weighted sum of inputs before activation.
Input nodes typically have a bias of 0.

#### clear

```ts
clear(): void
```

Clears the node's dynamic state information.
Resets activation, state, previous state, error signals, and eligibility traces.
Useful for starting a new activation sequence (e.g., for a new input pattern).

#### connect

```ts
connect(
  target: default | { nodes: default[]; },
  weight: number | undefined,
): default[]
```

Creates a connection from this node to a target node or all nodes in a group.

Parameters:
- `target` - The target Node or a group object containing a `nodes` array.
- `weight` - The weight for the new connection(s). If undefined, a default or random weight might be assigned by the Connection constructor (currently defaults to 0, consider changing).

Returns: An array containing the newly created Connection object(s).

#### connections

Stores incoming, outgoing, gated, and self-connections for this node.

#### derivative

The derivative of the activation function evaluated at the node's current state. Used in backpropagation.

#### disconnect

```ts
disconnect(
  target: default,
  twosided: boolean,
): void
```

Removes the connection from this node to the target node.

Parameters:
- `target` - The target node to disconnect from.
- `twosided` - If true, also removes the connection from the target node back to this node (if it exists). Defaults to false.

#### error

Stores error values calculated during backpropagation.

#### fromJSON

```ts
fromJSON(
  json: { bias: number; type: string; squash: string; mask: number; },
): default
```

Creates a Node instance from a JSON object.

Parameters:
- `json` - The JSON object containing node configuration.

Returns: A new Node instance configured according to the JSON object.

#### gate

```ts
gate(
  connections: default | default[],
): void
```

Makes this node gate the provided connection(s).
The connection's gain will be controlled by this node's activation value.

Parameters:
- `connections` - A single Connection object or an array of Connection objects to be gated.

#### geneId

Stable per-node gene identifier for NEAT innovation reuse

#### index

Optional index, potentially used to identify the node's position within a layer or network structure. Not used internally by the Node class itself.

#### isActivating

Internal flag to detect cycles during activation

#### isConnectedTo

```ts
isConnectedTo(
  target: default,
): boolean
```

Checks if this node is connected to another node.

Parameters:
- `target` - The target node to check the connection with.

Returns: True if connected, otherwise false.

#### isProjectedBy

```ts
isProjectedBy(
  node: default,
): boolean
```

Checks if the given node has a direct outgoing connection to this node.
Considers both regular incoming connections and the self-connection.

Parameters:
- `node` - The potential source node.

Returns: True if the given node projects to this node, false otherwise.

#### isProjectingTo

```ts
isProjectingTo(
  node: default,
): boolean
```

Checks if this node has a direct outgoing connection to the given node.
Considers both regular outgoing connections and the self-connection.

Parameters:
- `node` - The potential target node.

Returns: True if this node projects to the target node, false otherwise.

#### mask

A mask factor (typically 0 or 1) used for implementing dropout. If 0, the node's output is effectively silenced.

#### mutate

```ts
mutate(
  method: unknown,
): void
```

Applies a mutation method to the node. Used in neuro-evolution.

This allows modifying the node's properties, such as its activation function or bias,
based on predefined mutation methods.

Parameters:
- `method` - A mutation method object, typically from `methods.mutation`. It should define the type of mutation and its parameters (e.g., allowed functions, modification range).

#### noTraceActivate

```ts
noTraceActivate(
  input: number | undefined,
): number
```

Activates the node without calculating eligibility traces (`xtrace`).
This is a performance optimization used during inference (when the network
is just making predictions, not learning) as trace calculations are only needed for training.

Parameters:
- `input` - Optional input value. If provided, sets the node's activation directly (used for input nodes).

Returns: The calculated activation value of the node.

#### old

The node's state from the previous activation cycle. Used for recurrent self-connections.

#### previousDeltaBias

The change in bias applied in the previous training iteration. Used for calculating momentum.

#### propagate

```ts
propagate(
  rate: number,
  momentum: number,
  update: boolean,
  regularization: number | { type: "L1" | "L2"; lambda: number; } | ((weight: number) => number),
  target: number | undefined,
): void
```

Back-propagates the error signal through the node and calculates weight/bias updates.

This method implements the backpropagation algorithm, including:
1. Calculating the node's error responsibility based on errors from subsequent nodes (`projected` error)
   and errors from connections it gates (`gated` error).
2. Calculating the gradient for each incoming connection's weight using eligibility traces (`xtrace`).
3. Calculating the change (delta) for weights and bias, incorporating:
   - Learning rate.
   - L1/L2/custom regularization.
   - Momentum (using Nesterov Accelerated Gradient - NAG).
4. Optionally applying the calculated updates immediately or accumulating them for batch training.

Parameters:
- `rate` - The learning rate (controls the step size of updates).
- `momentum` - The momentum factor (helps accelerate learning and overcome local minima). Uses NAG.
- `update` - If true, apply the calculated weight/bias updates immediately. If false, accumulate them in `totalDelta*` properties for batch updates.
- `regularization` - The regularization setting. Can be:
- number (L2 lambda)
- { type: 'L1'|'L2', lambda: number }
- (weight: number) => number (custom function)
- `target` - The target output value for this node. Only used if the node is of type 'output'.

#### setActivation

```ts
setActivation(
  fn: (x: number, derivate?: boolean | undefined) => number,
): void
```

Sets a custom activation function for this node at runtime.

Parameters:
- `fn` - The activation function (should handle derivative if needed).

#### squash

```ts
squash(
  x: number,
  derivate: boolean | undefined,
): number
```

The activation function (squashing function) applied to the node's state.
Maps the internal state to the node's output (activation).

Parameters:
- `x` - The node's internal state (sum of weighted inputs + bias).
- `derivate` - If true, returns the derivative of the function instead of the function value.

Returns: The activation value or its derivative.

#### state

The internal state of the node (sum of weighted inputs + bias) before the activation function is applied.

#### toJSON

```ts
toJSON(): { index: number | undefined; bias: number; type: string; squash: string | null; mask: number; }
```

Converts the node's essential properties to a JSON object for serialization.
Does not include state, activation, error, or connection information, as these
are typically transient or reconstructed separately.

Returns: A JSON representation of the node's configuration.

#### totalDeltaBias

Accumulates changes in bias over a mini-batch during batch training. Reset after each weight update.

#### type

The type of the node: 'input', 'hidden', or 'output'.
Determines behavior (e.g., input nodes don't have biases modified typically, output nodes calculate error differently).

#### ungate

```ts
ungate(
  connections: default | default[],
): void
```

Removes this node's gating control over the specified connection(s).
Resets the connection's gain to 1 and removes it from the `connections.gated` list.

Parameters:
- `connections` - A single Connection object or an array of Connection objects to ungate.

## architecture/onnx.ts

### exportToONNX

```ts
exportToONNX(
  network: default,
  options: OnnxExportOptions,
): OnnxModel
```

Export a NeatapticTS network to an ONNX-like **JSON object** (`OnnxModel`).

What you get:
- A plain object that you can persist with `JSON.stringify()`.
- A minimal ONNX-ish graph (`model.graph`) plus optional metadata (`model.metadata_props`).

When to use this:
- You want a portable snapshot that can be inspected/diffed as JSON.
- You want to reconstruct the network later via `importFromONNX()`.

Tradeoffs:
- The output is ONNX-like, but **not** intended to be universally compatible with all ONNX
  runtimes.
- Some advanced features (partial connectivity, mixed activations, recurrent heuristics)
  may produce graphs that are primarily meant for this library’s importer.

High-level algorithm:
 1) Normalize/rebuild local connection state for deterministic traversal.
 2) Infer an ordered layer view and validate export constraints.
 3) Materialize graph nodes/tensors and (optionally) attach metadata.

Example (export → JSON text):

```ts
const model = exportToONNX(network, { includeMetadata: true });
const jsonText = JSON.stringify(model);
```

Parameters:
- `network` - Source network instance to serialize.
- `options` - Export controls (validation strictness and metadata behavior).

Returns: ONNX-like model object suitable for persistence or re-import.

### importFromONNX

```ts
importFromONNX(
  onnx: OnnxModel,
): default
```

Reconstruct a NeatapticTS network from an exported `OnnxModel`.

Expected input:
- A model produced by `exportToONNX()` (same repo/version family).

Trust boundary:
- Do not import untrusted blobs. A malformed model can be extremely large or internally
  inconsistent and may cause errors or high memory usage.

High-level behavior:
 1) Build a perceptron-shaped scaffold from the payload layer sizes.
 2) Assign weights/biases and activation functions.
 3) Re-apply recurrent and pooling metadata when present.

Example (JSON text → restore):

```ts
const model = JSON.parse(jsonText) as OnnxModel;
const restored = importFromONNX(model);
const output = restored.activate([0.1, 0.9]);
```

Parameters:
- `onnx` - ONNX-like model to reconstruct.

Returns: Reconstructed network ready for inference/evolution workflows.

### Conv2DMapping

Mapping declaration for treating a fully-connected layer as a 2D convolution during export.

This does **not** magically turn an MLP into a convolutional network at runtime.
It annotates a particular export-layer index with a conv interpretation so that:
- The exported graph uses conv-shaped tensors/operators, and
- Import can re-attach pooling/flatten metadata appropriately.

Pitfall: mappings must match the actual layer sizes. If `inHeight * inWidth * inChannels`
does not correspond to the prior layer width (and similarly for outputs), export or import
may reject the model.

### OnnxExportOptions

Options controlling ONNX-like export.

These options trade off strictness, portability, and fidelity:

- **Strict (default-ish)** export tries to keep the graph easy to interpret:
  layered topology, homogeneous activations per layer, and fully-connected layers.

- **Relaxed** export (`allowPartialConnectivity` / `allowMixedActivations`) can represent
  more networks, but it may generate graphs that are primarily meant for NeatapticTS’s
  importer (and may be less friendly to external ONNX tooling).

- **Recurrent export** (`allowRecurrent`) is intentionally conservative and currently
  focuses on a constrained single-step representation and optional fused heuristics.

Key fields (high-level):
- `includeMetadata`: includes `metadata_props` with architecture hints.
- `opset`: numeric opset version stored in the exported model metadata (default is
  resolved by the exporter; commonly 18 in this codebase).
- `legacyNodeOrdering`: keeps older node ordering for backward compatibility.
- `conv2dMappings` / `pool2dMappings`: encode conv/pool semantics for fully-connected
  layers via explicit mapping declarations.

### OnnxModel

ONNX-like model container (JSON-serializable).

This is the main “wire format” object in this folder. Persist it as JSON text:

```ts
const jsonText = JSON.stringify(model);
const restoredModel = JSON.parse(jsonText) as OnnxModel;
```

Notes:
- `metadata_props` contains NeatapticTS-specific keys (layer sizes, recurrent flags,
  conv/pool mappings, etc.). This is where most round-trip hints live.
- Initializers currently store floating-point weights in `float_data`.

Security/trust boundary:
- Treat this as untrusted input if it comes from outside your process.

### Pool2DMapping

Mapping describing a pooling operation inserted after a given export-layer index.

This is represented as metadata and optional graph nodes during export.
Import uses it to attach pooling-related runtime metadata back onto the reconstructed
network (when supported).

## architecture/group.ts

### group

Composite node block for architecture construction.

A group is the first place where the architecture surface stops talking about
one primitive at a time and starts exposing small graph motifs. It owns a set
of nodes plus the connection bookkeeping needed to treat that set as one
wiring target, one wiring source, and one propagation unit.

This makes the boundary useful in three different modes:

- dense or structured connection building between graph regions,
- collective activation and propagation when a block should act as one unit,
- recurrent and gated substructures where node-level behavior is still
  needed but orchestration should stay above the single-neuron level.

Example:

```ts
const encoderBlock = new Group(4);
const decoderBlock = new Group(4);

encoderBlock.connect(
  decoderBlock,
  methods.groupConnection.ONE_TO_ONE,
);
```

### default

#### activate

```ts
activate(
  value: number[] | undefined,
): number[]
```

Activates all nodes in the group.

Parameters:
- `value` - Optional array of input values. Its length must match the number of nodes in the group.

Returns: Activation value of each node in the group, in order.

#### clear

```ts
clear(): void
```

Resets the state of all nodes in the group.

Returns: Nothing.

#### connect

```ts
connect(
  target: default | default | default,
  method: unknown,
  weight: number | undefined,
): default[]
```

Establishes connections from all nodes in this group to a target group, layer, or node.

Parameters:
- `target` - Destination entity to connect to.
- `method` - Connection pattern to use.
- `weight` - Optional fixed weight for all created connections.

Returns: All connection objects created during this wiring step.

#### connections

Stores connection information related to this group.
`in`: Connections coming into any node in this group from outside.
`out`: Connections going out from any node in this group to outside.
`self`: Connections between nodes within this same group.

#### disconnect

```ts
disconnect(
  target: default | default,
  twosided: boolean,
): void
```

Removes connections between nodes in this group and a target group or node.

Parameters:
- `target` - Group or node to disconnect from.
- `twosided` - Whether to also remove reciprocal connections.

Returns: Nothing.

#### gate

```ts
gate(
  connections: default | default[],
  method: unknown,
): void
```

Configures nodes within this group to act as gates for the specified connection set.

Parameters:
- `connections` - Single connection or list of connections to gate.
- `method` - Gating mechanism to use.

Returns: Nothing.

#### nodes

An array holding all the nodes within this group.

#### propagate

```ts
propagate(
  rate: number,
  momentum: number,
  target: number[] | undefined,
): void
```

Propagates the error backward through all nodes in the group.

Parameters:
- `rate` - Learning rate to apply during weight updates.
- `momentum` - Momentum factor to apply during weight updates.
- `target` - Optional target values for error calculation. Its length must match the number of nodes.

Returns: Nothing.

#### set

```ts
set(
  values: { bias?: number | undefined; squash?: ((x: number, derivate?: boolean | undefined) => number) | undefined; type?: string | undefined; },
): void
```

Sets specific properties for all nodes within the group.

Parameters:
- `values` - Property values to apply to every node.

Returns: Nothing.

#### toJSON

```ts
toJSON(): { size: number; nodeIndices: (number | undefined)[]; connections: { in: number; out: number; self: number; }; }
```

Serializes the group into a JSON-compatible format, avoiding circular references.

Returns: JSON-friendly representation with node indices and connection counts.

## architecture/layer.ts

### layer

Represents a functional layer within a neural network architecture.

Layers act as organizational units for nodes, facilitating the creation of
complex network structures like Dense, LSTM, GRU, or Memory layers.
They manage the collective behavior of their nodes, including activation,
propagation, and connection to other network components.

### Layer

Represents a functional layer within a neural network architecture.

Layers act as organizational units for nodes, facilitating the creation of
complex network structures like Dense, LSTM, GRU, or Memory layers.
They manage the collective behavior of their nodes, including activation,
propagation, and connection to other network components.

### default

#### activate

```ts
activate(
  value: number[] | undefined,
  training: boolean,
): number[]
```

Activates all nodes within the layer, computing their output values.

If an input `value` array is provided, it's used as the initial activation
for the corresponding nodes in the layer. Otherwise, nodes compute their
activation based on their incoming connections.

During training, layer-level dropout is applied, masking all nodes in the layer together.
During inference, all masks are set to 1.

Parameters:
- `value` - - An optional array of activation values to set for the layer's nodes. The length must match the number of nodes.
- `training` - - A boolean indicating whether the layer is in training mode. Defaults to false.

Returns: An array containing the activation value of each node in the layer after activation.

#### attention

```ts
attention(
  size: number,
  heads: number,
): default
```

Creates a multi-head self-attention layer (stub implementation).

Parameters:
- `size` - - Number of output nodes.
- `heads` - - Number of attention heads (default 1).

Returns: A new Layer instance representing an attention layer.

#### batchNorm

```ts
batchNorm(
  size: number,
): default
```

Creates a batch normalization layer.
Applies batch normalization to the activations of the nodes in this layer during activation.

Parameters:
- `size` - - The number of nodes in this layer.

Returns: A new Layer instance configured as a batch normalization layer.

#### clear

```ts
clear(): void
```

Resets the activation state of all nodes within the layer.
This is typically done before processing a new input sequence or sample.

#### connect

```ts
connect(
  target: default | default | LayerLike,
  method: unknown,
  weight: number | undefined,
): default[]
```

Connects this layer's output to a target component (Layer, Group, or Node).

This method delegates the connection logic primarily to the layer's `output` group
or the target layer's `input` method. It establishes the forward connections
necessary for signal propagation.

Parameters:
- `target` - - The destination Layer, Group, or Node to connect to.
- `method` - - The connection method (e.g., `ALL_TO_ALL`, `ONE_TO_ONE`) defining the connection pattern. See `methods.groupConnection`.
- `weight` - - An optional fixed weight to assign to all created connections.

Returns: An array containing the newly created connection objects.

#### connections

Stores connection information related to this layer. This is often managed
by the network or higher-level structures rather than directly by the layer itself.
`in`: Incoming connections to the layer's nodes.
`out`: Outgoing connections from the layer's nodes.
`self`: Self-connections within the layer's nodes.

#### conv1d

```ts
conv1d(
  size: number,
  kernelSize: number,
  stride: number,
  padding: number,
): default
```

Creates a 1D convolutional layer (stub implementation).

Parameters:
- `size` - - Number of output nodes (filters).
- `kernelSize` - - Size of the convolution kernel.
- `stride` - - Stride of the convolution (default 1).
- `padding` - - Padding (default 0).

Returns: A new Layer instance representing a 1D convolutional layer.

#### dense

```ts
dense(
  size: number,
): default
```

Creates a standard fully connected (dense) layer.

All nodes in the source layer/group will connect to all nodes in this layer
when using the default `ALL_TO_ALL` connection method via `layer.input()`.

Parameters:
- `size` - - The number of nodes (neurons) in this layer.

Returns: A new Layer instance configured as a dense layer.

#### disconnect

```ts
disconnect(
  target: default | default,
  twosided: boolean | undefined,
): void
```

Removes connections between this layer's nodes and a target Group or Node.

Parameters:
- `target` - - The Group or Node to disconnect from.
- `twosided` - - If true, removes connections in both directions (from this layer to target, and from target to this layer). Defaults to false.

#### dropout

Dropout rate for this layer (0 to 1). If > 0, all nodes in the layer are masked together during training.
Layer-level dropout takes precedence over node-level dropout for nodes in this layer.

#### gate

```ts
gate(
  connections: default[],
  method: unknown,
): void
```

Applies gating to a set of connections originating from this layer's output group.

Gating allows the activity of nodes in this layer (specifically, the output group)
to modulate the flow of information through the specified `connections`.

Parameters:
- `connections` - - An array of connection objects to be gated.
- `method` - - The gating method (e.g., `INPUT`, `OUTPUT`, `SELF`) specifying how the gate influences the connection. See `methods.gating`.

#### gru

```ts
gru(
  size: number,
): default
```

Creates a Gated Recurrent Unit (GRU) layer.

GRUs are another type of recurrent neural network cell, often considered
simpler than LSTMs but achieving similar performance on many tasks.
They use an update gate and a reset gate to manage information flow.

Parameters:
- `size` - - The number of GRU units (and nodes in each gate/cell group).

Returns: A new Layer instance configured as a GRU layer.

#### input

```ts
input(
  from: default | LayerLike,
  method: unknown,
  weight: number | undefined,
): default[]
```

Handles the connection logic when this layer is the *target* of a connection.

It connects the output of the `from` layer or group to this layer's primary
input mechanism (which is often the `output` group itself, but depends on the layer type).
This method is usually called by the `connect` method of the source layer/group.

Parameters:
- `from` - - The source Layer or Group connecting *to* this layer.
- `method` - - The connection method (e.g., `ALL_TO_ALL`). Defaults to `ALL_TO_ALL`.
- `weight` - - An optional fixed weight for the connections.

Returns: An array containing the newly created connection objects.

#### layerNorm

```ts
layerNorm(
  size: number,
): default
```

Creates a layer normalization layer.
Applies layer normalization to the activations of the nodes in this layer during activation.

Parameters:
- `size` - - The number of nodes in this layer.

Returns: A new Layer instance configured as a layer normalization layer.

#### lstm

```ts
lstm(
  size: number,
): default
```

Creates a Long Short-Term Memory (LSTM) layer.

LSTMs are a type of recurrent neural network (RNN) cell capable of learning
long-range dependencies. This implementation uses standard LSTM architecture
with input, forget, and output gates, and a memory cell.

Parameters:
- `size` - - The number of LSTM units (and nodes in each gate/cell group).

Returns: A new Layer instance configured as an LSTM layer.

#### memory

```ts
memory(
  size: number,
  memory: number,
): default
```

Creates a Memory layer, designed to hold state over a fixed number of time steps.

This layer consists of multiple groups (memory blocks), each holding the state
from a previous time step. The input connects to the most recent block, and
information propagates backward through the blocks. The layer's output
concatenates the states of all memory blocks.

Parameters:
- `size` - - The number of nodes in each memory block (must match the input size).
- `memory` - - The number of time steps to remember (number of memory blocks).

Returns: A new Layer instance configured as a Memory layer.

#### nodes

An array containing all the nodes (neurons or groups) that constitute this layer.
The order of nodes might be relevant depending on the layer type and its connections.

#### output

Represents the primary output group of nodes for this layer.
This group is typically used when connecting this layer *to* another layer or group.
It might be null if the layer is not yet fully constructed or is an input layer.

#### propagate

```ts
propagate(
  rate: number,
  momentum: number,
  target: number[] | undefined,
): void
```

Propagates the error backward through all nodes in the layer.

This is a core step in the backpropagation algorithm used for training.
If a `target` array is provided (typically for the output layer), it's used
to calculate the initial error for each node. Otherwise, nodes calculate
their error based on the error propagated from subsequent layers.

Parameters:
- `rate` - - The learning rate, controlling the step size of weight adjustments.
- `momentum` - - The momentum factor, used to smooth weight updates and escape local minima.
- `target` - - An optional array of target values (expected outputs) for the layer's nodes. The length must match the number of nodes.

#### set

```ts
set(
  values: { bias?: number | undefined; squash?: ((x: number, derivate?: boolean | undefined) => number) | undefined; type?: string | undefined; },
): void
```

Configures properties for all nodes within the layer.

Allows batch setting of common node properties like bias, activation function (`squash`),
or node type. If a node within the `nodes` array is actually a `Group` (e.g., in memory layers),
the configuration is applied recursively to the nodes within that group.

Parameters:
- `values` - - An object containing the properties and their values to set.
  Example: `{ bias: 0.5, squash: methods.Activation.ReLU }`

## architecture/network.ts

### network

### resolveTopologyIntent

```ts
resolveTopologyIntent(
  options: NetworkConstructorOptions | undefined,
): NetworkTopologyIntent
```

Resolves the public topology intent for one constructor call.

Parameters:
- `options` - Optional constructor options.

Returns: Resolved topology intent.

### validateTopologyIntentConfiguration

```ts
validateTopologyIntentConfiguration(
  options: NetworkConstructorOptions | undefined,
): void
```

Validates that legacy acyclic flags do not contradict public topology intent.

Parameters:
- `options` - Optional constructor options.

Returns: Nothing.

### resolveAcyclicEnforcement

```ts
resolveAcyclicEnforcement(
  options: NetworkConstructorOptions | undefined,
  topologyIntent: NetworkTopologyIntent,
): boolean
```

Resolves whether acyclic enforcement should be enabled for one constructor call.

Parameters:
- `options` - Optional constructor options.
- `topologyIntent` - Resolved public topology intent.

Returns: True when acyclic enforcement should be enabled.

### default

#### _accumulationReduction

Accumulation reduction mode.

#### _activationPool

Cached pooled activation output array.

#### _activationPrecision

Typed-array precision used by compiled activation paths.

#### _adjDirty

Adjacency dirty marker for slab structures.

#### _applyGradientClipping

```ts
_applyGradientClipping(
  cfg: { mode: "norm" | "percentile" | "layerwiseNorm" | "layerwisePercentile"; maxNorm?: number | undefined; percentile?: number | undefined; },
): void
```

Apply gradient clipping configuration.

Parameters:
- `cfg` - Gradient clipping configuration.

#### _canUseFastSlab

```ts
_canUseFastSlab(
  training: boolean,
): boolean
```

Check if fast-slab activation can be used.

Parameters:
- `training` - Whether training mode is active.

Returns: True when fast-slab activation can be used.

#### _computeTopoOrder

```ts
_computeTopoOrder(): void
```

Recompute and cache topological node ordering.

Returns: Topological order payload from the delegate.

#### _connFrom

Packed connection slab source indices.

#### _connTo

Packed connection slab target indices.

#### _connWeights

Packed connection slab weights.

#### _currentGradClip

Gradient clip configuration for the current step.

#### _dropConnectProb

DropConnect probability.

#### _enforceAcyclic

Whether to enforce acyclic connectivity.

#### _evoInitialConnCount

Baseline connection count used by evolution-time pruning.

#### _fastA

Cached fast activation array A.

#### _fastS

Cached fast activation array S.

#### _fastSlabActivate

```ts
_fastSlabActivate(
  input: number[],
): number[]
```

Execute the fast slab activation path.

Parameters:
- `input` - Input vector.

Returns: Activation output.

#### _forceNextOverflow

Flag to force a mixed-precision overflow path.

#### _gaussianRand

```ts
_gaussianRand(
  rng: () => number,
): number
```

Sample a Gaussian random value with an optional RNG.

Parameters:
- `rng` - RNG function.

Returns: Gaussian random value.

#### _globalEpoch

Global epoch counter.

#### _gradAccumMicroBatches

Accumulated micro-batch counter.

#### _gradClipSeparateBias

Whether to apply separate bias clipping.

#### _hasPath

```ts
_hasPath(
  from: default,
  to: default,
): boolean
```

Check whether a directed path exists between two nodes.

Parameters:
- `from` - Source node.
- `to` - Target node.

Returns: True when a path exists.

#### _initialConnectionCount

Initial connection count used for pruning baselines.

#### _lastGradClipGroupCount

Last gradient clipping group count.

#### _lastGradNorm

Last recorded gradient norm.

#### _lastOverflowStep

Last overflow training step index.

#### _lastRawGradNorm

Last recorded raw (pre-update) gradient norm.

#### _lastStats

Last recorded stats payload.

#### _maybePrune

```ts
_maybePrune(
  iteration: number,
): void
```

Apply scheduled pruning if current iteration matches pruning policy.

Parameters:
- `iteration` - Current training iteration.

Returns: Delegate result for pruning attempt.

#### _mixedPrecision

Mixed precision runtime configuration.

#### _mixedPrecisionState

Mixed precision state counters.

#### _nodeIndexDirty

Node index dirty marker.

#### _optimizerStep

Optimizer step counter.

#### _outOrder

Output-order array for slab forward pass.

#### _outStart

Output-start array for slab forward pass.

#### _preferredChainEdge

Preferred linear-chain edge for node-split mutations.

#### _pruningConfig

Pruning configuration for scheduled pruning.

#### _rand

```ts
_rand(): number
```

Random number generator used for stochastic operations.

#### _returnTypedActivations

Whether pooled typed activations can be returned directly.

#### _reuseActivationArrays

Whether pooled activation arrays are reused across activations.

#### _rngState

Raw RNG state word.

#### _slabDirty

Slab dirty marker.

#### _stochasticDepth

Stochastic depth schedule values.

#### _stochasticDepthSchedule

Dynamic stochastic depth schedule.

#### _topoDirty

Topology dirty marker.

#### _topologyIntent

Public topology intent used to preserve semantic API choices.

#### _topoOrder

Cached topological order.

#### _trainingStep

Training step counter.

#### _useFloat32Weights

Whether to store slab weights in float32.

#### _weightNoisePerHidden

Per-hidden-layer weight-noise standard deviations.

#### _weightNoiseSchedule

Dynamic weight-noise schedule function.

#### _weightNoiseStd

Global weight-noise standard deviation.

#### _wnOrig

Original weights captured for weight-noise recovery.

#### activate

```ts
activate(
  input: number[],
  training: boolean,
  _maxActivationDepth: number,
): number[]
```

Standard activation API returning a plain number[] for backward compatibility.
Internally may use pooled typed arrays; if so they are cloned before returning.

#### activateBatch

```ts
activateBatch(
  inputs: number[][],
  training: boolean,
): number[][]
```

Activate the network over a batch of input vectors (micro-batching).

Currently iterates sample-by-sample while reusing the network's internal
fast-path allocations. Outputs are cloned number[] arrays for API
compatibility. Future optimizations can vectorize this path.

Parameters:
- `inputs` - Array of input vectors, each length must equal this.input
- `training` - Whether to run with training-time stochastic features

Returns: Array of output vectors, each length equals this.output

#### activateRaw

```ts
activateRaw(
  input: number[],
  training: boolean,
  maxActivationDepth: number,
): ActivationArray
```

Raw activation that can return a typed array when pooling is enabled (zero-copy).
If reuseActivationArrays=false falls back to standard activate().

Parameters:
- `input` - Input vector.
- `training` - Whether to enable training-time stochastic paths.
- `maxActivationDepth` - Maximum graph depth for activation.

Returns: Output activations (typed array when pooling is enabled).

#### addNodeBetween

```ts
addNodeBetween(): void
```

Split a random existing connection by inserting one hidden node.

#### adjustRateForAccumulation

```ts
adjustRateForAccumulation(
  rate: number,
  accumulationSteps: number,
  reduction: "average" | "sum",
): number
```

Utility: adjust rate for accumulation mode (use result when switching to 'sum' to mimic 'average').

#### clear

```ts
clear(): void
```

Clears the internal state of all nodes in the network.
Resets node activation, state, eligibility traces, and extended traces to their initial values (usually 0).
This is typically done before processing a new input sequence in recurrent networks or between training epochs if desired.

#### clearStochasticDepthSchedule

```ts
clearStochasticDepthSchedule(): void
```

Clear stochastic-depth schedule function.

#### clearWeightNoiseSchedule

```ts
clearWeightNoiseSchedule(): void
```

Clear the dynamic global weight-noise schedule.

#### clone

```ts
clone(): default
```

Creates a deep copy of the network.

Returns: A new Network instance that is a clone of the current network.

#### configurePruning

```ts
configurePruning(
  cfg: { start: number; end: number; targetSparsity: number; regrowFraction?: number | undefined; frequency?: number | undefined; method?: "magnitude" | "snip" | undefined; },
): void
```

Configure scheduled pruning during training.

Parameters:
- `cfg` - Pruning schedule and strategy configuration.

#### connect

```ts
connect(
  from: default,
  to: default,
  weight: number | undefined,
): default[]
```

Creates a connection between two nodes in the network.
Handles both regular connections and self-connections.
Adds the new connection object(s) to the appropriate network list (`connections` or `selfconns`).

Returns: An array containing the newly created connection object(s). Typically contains one connection, but might be empty or contain more in specialized node types.

#### connections

Connection list.

#### createMLP

```ts
createMLP(
  inputCount: number,
  hiddenCounts: number[],
  outputCount: number,
): default
```

Creates a fully connected, strictly layered MLP network.

Returns: A new, fully connected, layered MLP

#### crossOver

```ts
crossOver(
  network1: default,
  network2: default,
  equal: boolean,
): default
```

NEAT-style crossover delegate.

#### describeArchitecture

```ts
describeArchitecture(): NetworkArchitectureDescriptor
```

Resolves a stable architecture descriptor for telemetry/UI consumers.

Prefers live graph analysis and only falls back to hydrated serialization
metadata when graph-based resolution is purely inferred.

Returns: Architecture descriptor with hidden-layer widths and provenance.

#### deserialize

```ts
deserialize(
  data: unknown[] | [number[], number[], string[], { from: number; to: number; weight: number; gater: number | null; }[], number, number],
  inputSize: number | undefined,
  outputSize: number | undefined,
): default
```

Static lightweight tuple deserializer delegate

#### disableDropConnect

```ts
disableDropConnect(): void
```

Disable DropConnect.

#### disableStochasticDepth

```ts
disableStochasticDepth(): void
```

Disable stochastic depth.

#### disableWeightNoise

```ts
disableWeightNoise(): void
```

Disable all weight-noise settings.

#### disconnect

```ts
disconnect(
  from: default,
  to: default,
): void
```

Disconnects two nodes, removing the connection between them.
Handles both regular connections and self-connections.
If the connection being removed was gated, it is also ungated.

#### dropout

Dropout probability.

#### enableDropConnect

```ts
enableDropConnect(
  p: number,
): void
```

Enable DropConnect with a probability in $[0,1)$.

Parameters:
- `p` - DropConnect probability.

#### enableWeightNoise

```ts
enableWeightNoise(
  stdDev: number | { perHiddenLayer: number[]; },
): void
```

Enable weight noise using either a global standard deviation or per-hidden-layer values.

Parameters:
- `stdDev` - Global standard deviation or hidden-layer schedule.

#### fastSlabActivate

```ts
fastSlabActivate(
  input: number[],
): number[]
```

Public wrapper for fast slab forward pass.

Parameters:
- `input` - Input vector.

Returns: Activation output.

#### fromJSON

```ts
fromJSON(
  json: Record<string, unknown>,
): default
```

Verbose JSON static deserializer

#### gate

```ts
gate(
  node: default,
  connection: default,
): void
```

Gates a connection with a specified node.
The activation of the `node` (gater) will modulate the weight of the `connection`.
Adds the connection to the network's `gates` list.

#### gates

Network gates collection.

#### getConnectionSlab

```ts
getConnectionSlab(): ConnectionSlabView
```

Read slab structures for fast activation.

Returns: Slab connection structures.

#### getCurrentSparsity

```ts
getCurrentSparsity(): number
```

Compute the current connection sparsity ratio.

Returns: Current sparsity in $[0,1]$.

#### getLastGradClipGroupCount

```ts
getLastGradClipGroupCount(): number
```

Returns last gradient clipping group count (0 if no clipping yet).

#### getLossScale

```ts
getLossScale(): number
```

Returns current mixed precision loss scale (1 if disabled).

#### getRawGradientNorm

```ts
getRawGradientNorm(): number
```

Returns last recorded raw (pre-update) gradient L2 norm.

#### getRegularizationStats

```ts
getRegularizationStats(): Record<string, unknown> | null
```

Read regularization statistics collected during training.

Returns: Regularization stats payload.

#### getRNGState

```ts
getRNGState(): number | undefined
```

Read the raw deterministic RNG state word.

Returns: RNG state value when present.

#### getTopologyIntent

```ts
getTopologyIntent(): NetworkTopologyIntent
```

Returns the public topology intent for this network.

Returns: Current topology intent.

#### getTrainingStats

```ts
getTrainingStats(): { gradNorm: number; gradNormRaw: number; lossScale: number; optimizerStep: number; mp: { good: number; bad: number; overflowCount: number; scaleUps: number; scaleDowns: number; lastOverflowStep: number; }; }
```

Consolidated training stats snapshot.

#### input

Input node count.

#### lastSkippedLayers

Last skipped stochastic-depth layers from activation runtime state.

#### layers

Optional layered view cache.

#### mutate

```ts
mutate(
  method: MutationMethod,
): void
```

Mutates the network's structure or parameters according to the specified method.
This is a core operation for neuro-evolutionary algorithms (like NEAT).
The method argument should be one of the mutation types defined in `methods.mutation`.

Parameters:
- `method` - - The mutation method to apply (e.g., `mutation.ADD_NODE`, `mutation.MOD_WEIGHT`).
  Some methods might have associated parameters (e.g., `MOD_WEIGHT` uses `min`, `max`).

#### nodes

Network node collection.

#### noTraceActivate

```ts
noTraceActivate(
  input: number[],
): number[]
```

Activates the network without calculating eligibility traces.
This is a performance optimization for scenarios where backpropagation is not needed,
such as during testing, evaluation, or deployment (inference).

Returns: An array of numerical values representing the activations of the network's output nodes.

#### output

Output node count.

#### propagate

```ts
propagate(
  rate: number,
  momentum: number,
  update: boolean,
  target: number[],
  regularization: number,
  costDerivative: ((target: number, output: number) => number) | undefined,
): void
```

Propagates the error backward through the network (backpropagation).
Calculates the error gradient for each node and connection.
If `update` is true, it adjusts the weights and biases based on the calculated gradients,
learning rate, momentum, and optional L2 regularization.

The process starts from the output nodes and moves backward layer by layer (or topologically for recurrent nets).

#### pruneToSparsity

```ts
pruneToSparsity(
  targetSparsity: number,
  method: "magnitude" | "snip",
): void
```

Immediately prune connections to reach (or approach) a target sparsity fraction.
Used by evolutionary pruning (generation-based) independent of training iteration schedule.

Parameters:
- `targetSparsity` - fraction in (0,1). 0.8 means keep 20% of original (if first call sets baseline)
- `method` - 'magnitude' | 'snip'

#### rebuildConnections

```ts
rebuildConnections(
  net: default,
): void
```

Rebuilds the network's connections array from all per-node connections.
This ensures that the network.connections array is consistent with the actual
outgoing connections of all nodes. Useful after manual wiring or node manipulation.

Returns: Example usage:
  Network.rebuildConnections(net);

#### rebuildConnectionSlab

```ts
rebuildConnectionSlab(
  force: boolean,
): void
```

Rebuild slab structures for fast activation.

Parameters:
- `force` - Whether to force a rebuild.

Returns: Slab rebuild result.

#### remove

```ts
remove(
  node: default,
): void
```

Removes a node from the network.
This involves:
1. Disconnecting all incoming and outgoing connections associated with the node.
2. Removing any self-connections.
3. Removing the node from the `nodes` array.
4. Attempting to reconnect the node's direct predecessors to its direct successors
   to maintain network flow, if possible and configured.
5. Handling gates involving the removed node (ungating connections gated *by* this node,
   and potentially re-gating connections that were gated *by other nodes* onto the removed node's connections).

#### resetDropoutMasks

```ts
resetDropoutMasks(): void
```

Resets all masks in the network to 1 (no dropout). Applies to both node-level and layer-level dropout.
Should be called after training to ensure inference is unaffected by previous dropout.

#### restoreRNG

```ts
restoreRNG(
  fn: () => number,
): void
```

Restore deterministic RNG function from a snapshot source.

Parameters:
- `fn` - RNG function to restore.

#### score

Optional fitness score.

#### selfconns

Self-connection list.

#### serialize

```ts
serialize(): [number[], number[], string[], SerializedConnection[], number, number]
```

Lightweight tuple serializer delegating to network.serialize.ts

#### set

```ts
set(
  values: { bias?: number | undefined; squash?: ((x: number, derivate?: boolean | undefined) => number) | undefined; },
): void
```

Sets specified properties (e.g., bias, squash function) for all nodes in the network.
Useful for initializing or resetting node properties uniformly.

#### setEnforceAcyclic

```ts
setEnforceAcyclic(
  flag: boolean,
): void
```

Enable or disable acyclic topology enforcement.

Parameters:
- `flag` - Whether to enforce acyclic connectivity.

#### setRandom

```ts
setRandom(
  fn: () => number,
): void
```

Replace the network random number generator.

Parameters:
- `fn` - RNG function returning values in $[0,1)$.

#### setRNGState

```ts
setRNGState(
  state: number,
): void
```

Set the raw deterministic RNG state word.

Parameters:
- `state` - RNG state value.

#### setSeed

```ts
setSeed(
  seed: number,
): void
```

Seed the internal deterministic RNG.

Parameters:
- `seed` - Seed value.

#### setStochasticDepth

```ts
setStochasticDepth(
  survival: number[],
): void
```

Configure stochastic depth with survival probabilities per hidden layer.

Parameters:
- `survival` - Survival probabilities for hidden layers.

#### setStochasticDepthSchedule

```ts
setStochasticDepthSchedule(
  fn: (step: number, current: number[]) => number[],
): void
```

Set stochastic-depth schedule function.

Parameters:
- `fn` - Function mapping step and current schedule to next schedule.

#### setTopologyIntent

```ts
setTopologyIntent(
  topologyIntent: NetworkTopologyIntent,
): void
```

Sets the public topology intent and keeps acyclic enforcement aligned.

Parameters:
- `topologyIntent` - Desired topology intent.

Returns: Nothing.

#### setWeightNoiseSchedule

```ts
setWeightNoiseSchedule(
  fn: (step: number) => number,
): void
```

Set a dynamic scheduler for global weight noise.

Parameters:
- `fn` - Function mapping training step to noise standard deviation.

#### snapshotRNG

```ts
snapshotRNG(): RNGSnapshot
```

Snapshot deterministic RNG runtime state.

Returns: Current RNG snapshot.

#### test

```ts
test(
  set: { input: number[]; output: number[]; }[],
  cost: ((target: number[], output: number[]) => number) | undefined,
): { error: number; time: number; }
```

Tests the network's performance on a given dataset.
Calculates the average error over the dataset using a specified cost function.
Uses `noTraceActivate` for efficiency as gradients are not needed.
Handles dropout scaling if dropout was used during training.

Returns: An object containing the calculated average error over the dataset and the time taken for the test in milliseconds.

#### testForceOverflow

```ts
testForceOverflow(): void
```

Force the next mixed-precision overflow path (test utility).

#### toJSON

```ts
toJSON(): Record<string, unknown>
```

Verbose JSON serializer delegate

#### toONNX

```ts
toONNX(): OnnxModel
```

Exports the network to ONNX format (JSON object, minimal MLP support).
Only standard feedforward architectures and standard activations are supported.
Gating, custom activations, and evolutionary features are ignored or replaced with Identity.

Returns: ONNX model as a JSON object.

#### trainingStep

Current training step counter.

#### ungate

```ts
ungate(
  connection: default,
): void
```

Removes the gate from a specified connection.
The connection will no longer be modulated by its gater node.
Removes the connection from the network's `gates` list.

## architecture/nodePool.ts

### acquireNode

```ts
acquireNode(
  opts: AcquireNodeOptions,
): default
```

Acquire a node instance from the pool, or construct a fresh one when the
pool is empty.

The returned node is guaranteed to have detached connections, cleared error
state, and a fresh gene id for its next lifecycle.

Parameters:
- `opts` - Optional acquisition settings.

Returns: A ready-to-use node instance.

### nodePoolStats

```ts
nodePoolStats(): { size: number; highWaterMark: number; reused: number; fresh: number; recycledRatio: number; }
```

Get current pool statistics for diagnostics and memory reporting.

Returns: Pool size, reuse counters, and the long-run recycled ratio.

### releaseNode

```ts
releaseNode(
  node: default,
): void
```

Release a detached node back into the pool.

Callers must ensure the node is no longer part of any live graph. The pool
keeps the object shell, not the prior topology membership.

Parameters:
- `node` - Detached node instance to recycle.

Returns: Nothing.

### resetNodePool

```ts
resetNodePool(): void
```

Drop all retained pooled nodes and reset instrumentation counters.

Returns: Nothing.

### AcquireNodeOptions

Options bag for acquiring a node.

## architecture/architect.ts

### architect

Provides static methods for constructing predefined neural network
architectures.

`Architect` is the point where the low-level graph primitives stop being
raw building blocks and start becoming named network recipes. It assembles
nodes, groups, and layers into complete graphs, then normalizes the final
`Network` surface so callers can activate, train, serialize, or evolve the
result without manually wiring each primitive.

This boundary matters when you want one of three things:

- a deterministic builder for common feed-forward shapes,
- a quick way to sample or mutate topology-oriented starting graphs,
- recurrent presets that reuse the same lower-level chapters instead of
  hiding a separate graph implementation.

Example:

```ts
const network = Architect.perceptron(2, 4, 1);
const output = network.activate([0, 1]);
```

### default

#### construct

```ts
construct(
  list: (default | default | default)[],
): default
```

Constructs a network instance from an array of interconnected layers,
groups, or nodes.

This method is the bridge between manual graph assembly and a runnable
`Network`. It walks the supplied primitives, collects the unique nodes and
connections they reference, infers input/output counts from node types, and
folds the result into one normalized network object.

Parameters:
- `list` - Building blocks that are already interconnected.

Returns: A network representing the supplied architecture.

#### enforceMinimumHiddenLayerSizes

```ts
enforceMinimumHiddenLayerSizes(
  network: default,
): default
```

Enforces the minimum hidden layer size rule on a network.

Parameters:
- `network` - The network to normalize.

Returns: The same network with hidden layers grown to the minimum size when needed.

#### gru

```ts
gru(
  layers: number[],
): default
```

Creates a Gated Recurrent Unit network.

Parameters:
- `layers` - Layer sizes starting with input and ending with output.

Returns: The constructed GRU network.

#### hopfield

```ts
hopfield(
  size: number,
): default
```

Creates a Hopfield network.

Parameters:
- `size` - The number of nodes in the network.

Returns: The constructed Hopfield network.

#### lstm

```ts
lstm(
  layerArgs: (number | { inputToOutput?: boolean | undefined; })[],
): default
```

Creates a Long Short-Term Memory network.

Parameters:
- `layerArgs` - Layer sizes plus an optional trailing options object.

Returns: The constructed LSTM network.

#### narx

```ts
narx(
  inputSize: number,
  hiddenLayers: number | number[],
  outputSize: number,
  previousInput: number,
  previousOutput: number,
): default
```

Creates a Nonlinear AutoRegressive network with eXogenous inputs.

Parameters:
- `inputSize` - The exogenous input size at each time step.
- `hiddenLayers` - Hidden layer sizes, or zero / empty for none.
- `outputSize` - The prediction output size.
- `previousInput` - The number of delayed input steps.
- `previousOutput` - The number of delayed output steps.

Returns: The constructed NARX network.

#### perceptron

```ts
perceptron(
  layers: number[],
): default
```

Creates a standard multi-layer perceptron network.

The returned network is marked with the public `feed-forward` topology
intent so acyclic enforcement and slab fast-path eligibility stay aligned
with the builder users already chose.

Parameters:
- `layers` - Layer sizes starting with input, followed by hidden layers,
and ending with output.

Returns: The constructed MLP network.

#### random

```ts
random(
  input: number,
  hidden: number,
  output: number,
  options: { connections?: number | undefined; backconnections?: number | undefined; selfconnections?: number | undefined; gates?: number | undefined; },
): default
```

Creates a randomly structured network based on node counts and connection
options.

Parameters:
- `input` - The number of input nodes.
- `hidden` - The number of hidden nodes to add.
- `output` - The number of output nodes.
- `options` - Optional configuration for connection counts and gates.

Returns: The constructed randomized network.

## architecture/connection.ts

### connection

Connection (Synapse / Edge)
===========================
Directed weighted link between two nodes. The connection keeps the everyday
graph fields (`from`, `to`, `weight`, `innovation`) directly on the instance,
then pushes rarer capabilities behind symbol-backed accessors so large
populations do not pay object-shape costs for features they are not using.

This makes the boundary useful in three different modes:

- ordinary feed-forward links that only need endpoints and weight,
- gated or plastic links that gradually opt into extra runtime state,
- optimizer-heavy training paths that need moment buffers without turning
  every connection into a bloated record.

Example:

```ts
const source = new Node('input');
const target = new Node('output');
const edge = new Connection(source, target, 0.42);

edge.gain = 1.5;
edge.enabled = true;
```

### default

#### _flags

Packed state flags (private for future-proofing hidden class):
bit0 => enabled gene expression (1 = active)
bit1 => DropConnect active mask (1 = not dropped this forward pass)
bit2 => hasGater (1 = symbol field present)
bit3 => plastic (plasticityRate > 0)
bits4+ reserved.

#### acquire

```ts
acquire(
  from: default,
  to: default,
  weight: number | undefined,
): default
```

Acquire a connection from the internal pool, or construct a fresh one when the pool is empty.
This is the low-allocation path used by topology mutation and other edge-churn heavy flows.

Parameters:
- `from` - Source node.
- `to` - Target node.
- `weight` - Optional initial weight.

Returns: Reinitialized connection instance.

#### dcMask

DropConnect active mask: `1` means active for this stochastic pass, `0` means dropped.

#### dropConnectActiveMask

Convenience alias for DropConnect mask with clearer naming.

#### eligibility

Standard eligibility trace (e.g., for RTRL / policy gradient credit assignment).

#### enabled

Whether the gene is currently expressed and participates in the forward pass.

#### firstMoment

First moment estimate used by Adam-family optimizers.

#### from

The source (pre-synaptic) node supplying activation.

#### gain

Multiplicative modulation applied after weight. Neutral gain `1` is omitted from storage.

#### gater

Optional gating node whose activation modulates effective weight.

#### gradientAccumulator

Generic gradient accumulator used by RMSProp and AdaGrad.

#### hasGater

Whether a gater node is assigned to modulate this connection's effective weight.

#### infinityNorm

Adamax infinity norm accumulator.

#### innovation

Unique historical marking (auto-increment) for evolutionary alignment.

#### innovationID

```ts
innovationID(
  sourceNodeId: number,
  targetNodeId: number,
): number
```

Deterministic Cantor pairing function for a `(sourceNodeId, targetNodeId)` pair.
Use it when you need a stable edge identifier without relying on the mutable
auto-increment counter.

Parameters:
- `sourceNodeId` - Source node integer id or index.
- `targetNodeId` - Target node integer id or index.

Returns: Unique non-negative integer derived from the ordered pair.

Example:

```ts
const id = Connection.innovationID(2, 5);
```

#### lookaheadShadowWeight

Lookahead slow-weight snapshot.

#### maxSecondMoment

AMSGrad maximum of past second-moment estimates.

#### plastic

Whether this connection participates in plastic adaptation.

#### plasticityRate

Per-connection plasticity rate. `0` means the connection is not plastic.

#### previousDeltaWeight

Last applied delta weight (used by classic momentum).

#### release

```ts
release(
  conn: default,
): void
```

Return a connection instance to the internal pool for later reuse.
Treat the instance as surrendered after calling this method.

Parameters:
- `conn` - The connection instance to recycle.

Returns: Nothing.

#### resetInnovationCounter

```ts
resetInnovationCounter(
  value: number,
): void
```

Reset the monotonic innovation counter used for newly constructed or pooled connections.
You usually call this at the start of an experiment or before rebuilding a whole population.

Parameters:
- `value` - New starting value.

Returns: Nothing.

#### secondMoment

Second raw moment estimate used by Adam-family optimizers.

#### secondMomentum

Secondary momentum buffer used by Lion-style updates.

#### to

The target (post-synaptic) node receiving activation.

#### toJSON

```ts
toJSON(): { from: number | undefined; to: number | undefined; weight: number; gain: number; innovation: number; enabled: boolean; gater?: number | undefined; }
```

Serialize to a minimal JSON-friendly shape used by genome and network save flows.
Undefined node indices are preserved so callers can resolve or remap them later.

Returns: Object with node indices, weight, gain, innovation id, enabled flag, and gater index when one exists.

Example:

```ts
const json = connection.toJSON();
// => { from: 0, to: 3, weight: 0.12, gain: 1, innovation: 57, enabled: true }
```

#### totalDeltaWeight

Accumulated (batched) delta weight awaiting an apply step.

#### weight

Scalar multiplier applied to the source activation (prior to gain modulation).

#### xtrace

Extended trace structure for modulatory / eligibility propagation algorithms. Parallel arrays for cache-friendly iteration.

## architecture/activationArrayPool.ts

### ActivationArray

Allowed activation array shapes for pooling.

The runtime prefers typed arrays when float32 mode is enabled, but keeps
plain numeric arrays available for code paths that expect standard JS array
behavior.

### activationArrayPool

Shared singleton instance used across the library for maximal reuse.
