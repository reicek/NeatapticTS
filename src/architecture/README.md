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

### Node

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

### NodeOptimizerProps

Internal interface for accessing dynamic optimizer properties on Node instances.
These properties are lazily allocated and not part of the main class definition.

### default

#### _activateCore

`(withTrace: boolean, input: number | undefined) => number`

Internal shared implementation for activate/noTraceActivate.

Parameters:
- `withTrace` - Whether to update eligibility traces.
- `input` - Optional externally supplied activation (bypasses weighted sum if provided).

#### _globalNodeIndex

Global index counter for assigning unique indices to nodes.

#### _safeUpdateWeight

`(connection: import("src/architecture/connection").default, delta: number) => void`

Internal helper to safely update a connection weight with clipping and NaN checks.

#### activate

`(input: number | undefined) => number`

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

`(momentum: number) => void`

Applies accumulated batch updates to incoming and self connections and this node's bias.
Uses momentum in a Nesterov-compatible way: currentDelta = accumulated + momentum * previousDelta.
Resets accumulators after applying. Safe to call on any node type.

Parameters:
- `momentum` - Momentum factor (0 to disable)

#### applyBatchUpdatesWithOptimizer

`(opts: { type: "sgd" | "rmsprop" | "adagrad" | "adam" | "adamw" | "amsgrad" | "adamax" | "nadam" | "radam" | "lion" | "adabelief" | "lookahead"; momentum?: number | undefined; beta1?: number | undefined; beta2?: number | undefined; eps?: number | undefined; weightDecay?: number | undefined; lrScale?: number | undefined; t?: number | undefined; baseType?: string | undefined; la_k?: number | undefined; la_alpha?: number | undefined; }) => void`

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

`() => void`

Clears the node's dynamic state information.
Resets activation, state, previous state, error signals, and eligibility traces.
Useful for starting a new activation sequence (e.g., for a new input pattern).

#### connect

`(target: import("src/architecture/node").default | { nodes: import("src/architecture/node").default[]; }, weight: number | undefined) => import("src/architecture/connection").default[]`

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

`(target: import("src/architecture/node").default, twosided: boolean) => void`

Removes the connection from this node to the target node.

Parameters:
- `target` - The target node to disconnect from.
- `twosided` - If true, also removes the connection from the target node back to this node (if it exists). Defaults to false.

#### error

Stores error values calculated during backpropagation.

#### fromJSON

`(json: { bias: number; type: string; squash: string; mask: number; }) => import("src/architecture/node").default`

Creates a Node instance from a JSON object.

Parameters:
- `json` - The JSON object containing node configuration.

Returns: A new Node instance configured according to the JSON object.

#### gate

`(connections: import("src/architecture/connection").default | import("src/architecture/connection").default[]) => void`

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

`(target: import("src/architecture/node").default) => boolean`

Checks if this node is connected to another node.

Parameters:
- `target` - The target node to check the connection with.

Returns: True if connected, otherwise false.

#### isProjectedBy

`(node: import("src/architecture/node").default) => boolean`

Checks if the given node has a direct outgoing connection to this node.
Considers both regular incoming connections and the self-connection.

Parameters:
- `node` - The potential source node.

Returns: True if the given node projects to this node, false otherwise.

#### isProjectingTo

`(node: import("src/architecture/node").default) => boolean`

Checks if this node has a direct outgoing connection to the given node.
Considers both regular outgoing connections and the self-connection.

Parameters:
- `node` - The potential target node.

Returns: True if this node projects to the target node, false otherwise.

#### mask

A mask factor (typically 0 or 1) used for implementing dropout. If 0, the node's output is effectively silenced.

#### mutate

`(method: unknown) => void`

Applies a mutation method to the node. Used in neuro-evolution.

This allows modifying the node's properties, such as its activation function or bias,
based on predefined mutation methods.

Parameters:
- `method` - A mutation method object, typically from `methods.mutation`. It should define the type of mutation and its parameters (e.g., allowed functions, modification range).

#### noTraceActivate

`(input: number | undefined) => number`

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

`(rate: number, momentum: number, update: boolean, regularization: number | { type: "L1" | "L2"; lambda: number; } | ((weight: number) => number), target: number | undefined) => void`

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

`(fn: (x: number, derivate?: boolean | undefined) => number) => void`

Sets a custom activation function for this node at runtime.

Parameters:
- `fn` - The activation function (should handle derivative if needed).

#### squash

`(x: number, derivate: boolean | undefined) => number`

The activation function (squashing function) applied to the node's state.
Maps the internal state to the node's output (activation).

Parameters:
- `x` - The node's internal state (sum of weighted inputs + bias).
- `derivate` - If true, returns the derivative of the function instead of the function value.

Returns: The activation value or its derivative.

#### state

The internal state of the node (sum of weighted inputs + bias) before the activation function is applied.

#### toJSON

`() => { index: number | undefined; bias: number; type: string; squash: string | null; mask: number; }`

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

`(connections: import("src/architecture/connection").default | import("src/architecture/connection").default[]) => void`

Removes this node's gating control over the specified connection(s).
Resets the connection's gain to 1 and removes it from the `connections.gated` list.

Parameters:
- `connections` - A single Connection object or an array of Connection objects to ungate.

## architecture/onnx.ts

### Conv2DMapping

Mapping declaration for treating a fully-connected layer as a 2D convolution during export.

This does **not** magically turn an MLP into a convolutional network at runtime.
It annotates a particular export-layer index with a conv interpretation so that:
- The exported graph uses conv-shaped tensors/operators, and
- Import can re-attach pooling/flatten metadata appropriately.

Pitfall: mappings must match the actual layer sizes. If `inHeight * inWidth * inChannels`
does not correspond to the prior layer width (and similarly for outputs), export or import
may reject the model.

### exportToONNX

`(network: import("src/architecture/network").default, options: import("src/architecture/network/onnx/network.onnx.utils.types").OnnxExportOptions) => import("src/architecture/network/onnx/network.onnx.utils.types").OnnxModel`

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

`(onnx: import("src/architecture/network/onnx/network.onnx.utils.types").OnnxModel) => import("src/architecture/network").default`

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

Represents a collection of nodes functioning as a single unit within a network architecture.
Groups facilitate operations like collective activation, propagation, and connection management.

### Group

Represents a collection of nodes functioning as a single unit within a network architecture.
Groups facilitate operations like collective activation, propagation, and connection management.

### default

#### activate

`(value: number[] | undefined) => number[]`

Activates all nodes in the group. If input values are provided, they are assigned
sequentially to the nodes before activation. Otherwise, nodes activate based on their
existing states and incoming connections.

Returns: An array containing the activation value of each node in the group, in order.

#### clear

`() => void`

Resets the state of all nodes in the group. This typically involves clearing
activation values, state, and propagated errors, preparing the group for a new input pattern,
especially relevant in recurrent networks or sequence processing.

#### connect

`(target: import("src/architecture/node").default | import("src/architecture/layer").default | import("src/architecture/group").default, method: unknown, weight: number | undefined) => import("src/architecture/connection").default[]`

Establishes connections from all nodes in this group to a target Group, Layer, or Node.
The connection pattern (e.g., all-to-all, one-to-one) can be specified.

Returns: An array containing all the connection objects created.

#### connections

Stores connection information related to this group.
`in`: Connections coming into any node in this group from outside.
`out`: Connections going out from any node in this group to outside.
`self`: Connections between nodes within this same group (e.g., in ONE_TO_ONE connections).

#### disconnect

`(target: import("src/architecture/node").default | import("src/architecture/group").default, twosided: boolean) => void`

Removes connections between nodes in this group and a target Group or Node.

#### gate

`(connections: import("src/architecture/connection").default | import("src/architecture/connection").default[], method: unknown) => void`

Configures nodes within this group to act as gates for the specified connection(s).
Gating allows the output of a node in this group to modulate the flow of signal through the gated connection.

#### nodes

An array holding all the nodes within this group.

#### propagate

`(rate: number, momentum: number, target: number[] | undefined) => void`

Propagates the error backward through all nodes in the group. If target values are provided,
the error is calculated against these targets (typically for output layers). Otherwise,
the error is calculated based on the error propagated from subsequent layers/nodes.

#### set

`(values: { bias?: number | undefined; squash?: ((x: number, derivate?: boolean | undefined) => number) | undefined; type?: string | undefined; }) => void`

Sets specific properties (like bias, squash function, or type) for all nodes within the group.

#### toJSON

`() => { size: number; nodeIndices: (number | undefined)[]; connections: { in: number; out: number; self: number; }; }`

Serializes the group into a JSON-compatible format, avoiding circular references.
Only includes node indices and connection counts.

Returns: A JSON-compatible representation of the group.

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

`(value: number[] | undefined, training: boolean) => number[]`

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

`(size: number, heads: number) => import("src/architecture/layer").default`

Creates a multi-head self-attention layer (stub implementation).

Parameters:
- `size` - - Number of output nodes.
- `heads` - - Number of attention heads (default 1).

Returns: A new Layer instance representing an attention layer.

#### batchNorm

`(size: number) => import("src/architecture/layer").default`

Creates a batch normalization layer.
Applies batch normalization to the activations of the nodes in this layer during activation.

Parameters:
- `size` - - The number of nodes in this layer.

Returns: A new Layer instance configured as a batch normalization layer.

#### clear

`() => void`

Resets the activation state of all nodes within the layer.
This is typically done before processing a new input sequence or sample.

#### connect

`(target: import("src/architecture/node").default | import("src/architecture/group").default | import("src/architecture/layer/layer.utils.types").LayerLike, method: unknown, weight: number | undefined) => import("src/architecture/connection").default[]`

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

`(size: number, kernelSize: number, stride: number, padding: number) => import("src/architecture/layer").default`

Creates a 1D convolutional layer (stub implementation).

Parameters:
- `size` - - Number of output nodes (filters).
- `kernelSize` - - Size of the convolution kernel.
- `stride` - - Stride of the convolution (default 1).
- `padding` - - Padding (default 0).

Returns: A new Layer instance representing a 1D convolutional layer.

#### dense

`(size: number) => import("src/architecture/layer").default`

Creates a standard fully connected (dense) layer.

All nodes in the source layer/group will connect to all nodes in this layer
when using the default `ALL_TO_ALL` connection method via `layer.input()`.

Parameters:
- `size` - - The number of nodes (neurons) in this layer.

Returns: A new Layer instance configured as a dense layer.

#### disconnect

`(target: import("src/architecture/node").default | import("src/architecture/group").default, twosided: boolean | undefined) => void`

Removes connections between this layer's nodes and a target Group or Node.

Parameters:
- `target` - - The Group or Node to disconnect from.
- `twosided` - - If true, removes connections in both directions (from this layer to target, and from target to this layer). Defaults to false.

#### dropout

Dropout rate for this layer (0 to 1). If > 0, all nodes in the layer are masked together during training.
Layer-level dropout takes precedence over node-level dropout for nodes in this layer.

#### gate

`(connections: import("src/architecture/connection").default[], method: unknown) => void`

Applies gating to a set of connections originating from this layer's output group.

Gating allows the activity of nodes in this layer (specifically, the output group)
to modulate the flow of information through the specified `connections`.

Parameters:
- `connections` - - An array of connection objects to be gated.
- `method` - - The gating method (e.g., `INPUT`, `OUTPUT`, `SELF`) specifying how the gate influences the connection. See `methods.gating`.

#### gru

`(size: number) => import("src/architecture/layer").default`

Creates a Gated Recurrent Unit (GRU) layer.

GRUs are another type of recurrent neural network cell, often considered
simpler than LSTMs but achieving similar performance on many tasks.
They use an update gate and a reset gate to manage information flow.

Parameters:
- `size` - - The number of GRU units (and nodes in each gate/cell group).

Returns: A new Layer instance configured as a GRU layer.

#### input

`(from: import("src/architecture/group").default | import("src/architecture/layer/layer.utils.types").LayerLike, method: unknown, weight: number | undefined) => import("src/architecture/connection").default[]`

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

`(size: number) => import("src/architecture/layer").default`

Creates a layer normalization layer.
Applies layer normalization to the activations of the nodes in this layer during activation.

Parameters:
- `size` - - The number of nodes in this layer.

Returns: A new Layer instance configured as a layer normalization layer.

#### lstm

`(size: number) => import("src/architecture/layer").default`

Creates a Long Short-Term Memory (LSTM) layer.

LSTMs are a type of recurrent neural network (RNN) cell capable of learning
long-range dependencies. This implementation uses standard LSTM architecture
with input, forget, and output gates, and a memory cell.

Parameters:
- `size` - - The number of LSTM units (and nodes in each gate/cell group).

Returns: A new Layer instance configured as an LSTM layer.

#### memory

`(size: number, memory: number) => import("src/architecture/layer").default`

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

`(rate: number, momentum: number, target: number[] | undefined) => void`

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

`(values: { bias?: number | undefined; squash?: ((x: number, derivate?: boolean | undefined) => number) | undefined; type?: string | undefined; }) => void`

Configures properties for all nodes within the layer.

Allows batch setting of common node properties like bias, activation function (`squash`),
or node type. If a node within the `nodes` array is actually a `Group` (e.g., in memory layers),
the configuration is applied recursively to the nodes within that group.

Parameters:
- `values` - - An object containing the properties and their values to set.
  Example: `{ bias: 0.5, squash: methods.Activation.ReLU }`

## architecture/network.ts

### network

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

`(cfg: { mode: "norm" | "percentile" | "layerwiseNorm" | "layerwisePercentile"; maxNorm?: number | undefined; percentile?: number | undefined; }) => void`

Apply gradient clipping configuration.

Parameters:
- `cfg` - Gradient clipping configuration.

#### _canUseFastSlab

`(training: boolean) => boolean`

Check if fast-slab activation can be used.

Parameters:
- `training` - Whether training mode is active.

Returns: True when fast-slab activation can be used.

#### _computeTopoOrder

`() => void`

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

`(input: number[]) => number[]`

Execute the fast slab activation path.

Parameters:
- `input` - Input vector.

Returns: Activation output.

#### _forceNextOverflow

Flag to force a mixed-precision overflow path.

#### _gaussianRand

`(rng: () => number) => number`

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

`(from: import("src/architecture/node").default, to: import("src/architecture/node").default) => boolean`

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

`(iteration: number) => void`

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

`() => number`

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

`(input: number[], training: boolean, _maxActivationDepth: number) => number[]`

Activates the network using the given input array.
Performs a forward pass through the network, calculating the activation of each node.

Returns: An array of numerical values representing the activations of the network's output nodes.

#### activateBatch

`(inputs: number[][], training: boolean) => number[][]`

Activate the network over a batch of input vectors (micro-batching).

Currently iterates sample-by-sample while reusing the network's internal
fast-path allocations. Outputs are cloned number[] arrays for API
compatibility. Future optimizations can vectorize this path.

Parameters:
- `inputs` - Array of input vectors, each length must equal this.input
- `training` - Whether to run with training-time stochastic features

Returns: Array of output vectors, each length equals this.output

#### activateRaw

`(input: number[], training: boolean, maxActivationDepth: number) => import("src/architecture/activationArrayPool").ActivationArray`

Raw activation that can return a typed array when pooling is enabled (zero-copy).
If reuseActivationArrays=false falls back to standard activate().

Parameters:
- `input` - Input vector.
- `training` - Whether to enable training-time stochastic paths.
- `maxActivationDepth` - Maximum graph depth for activation.

Returns: Output activations (typed array when pooling is enabled).

#### addNodeBetween

`() => void`

Split a random existing connection by inserting one hidden node.

#### adjustRateForAccumulation

`(rate: number, accumulationSteps: number, reduction: "average" | "sum") => number`

Utility: adjust rate for accumulation mode (use result when switching to 'sum' to mimic 'average').

#### clear

`() => void`

Clears the internal state of all nodes in the network.
Resets node activation, state, eligibility traces, and extended traces to their initial values (usually 0).
This is typically done before processing a new input sequence in recurrent networks or between training epochs if desired.

#### clearStochasticDepthSchedule

`() => void`

Clear stochastic-depth schedule function.

#### clearWeightNoiseSchedule

`() => void`

Clear the dynamic global weight-noise schedule.

#### clone

`() => import("src/architecture/network").default`

Creates a deep copy of the network.

Returns: A new Network instance that is a clone of the current network.

#### configurePruning

`(cfg: { start: number; end: number; targetSparsity: number; regrowFraction?: number | undefined; frequency?: number | undefined; method?: "magnitude" | "snip" | undefined; }) => void`

Configure scheduled pruning during training.

Parameters:
- `cfg` - Pruning schedule and strategy configuration.

#### connect

`(from: import("src/architecture/node").default, to: import("src/architecture/node").default, weight: number | undefined) => import("src/architecture/connection").default[]`

Creates a connection between two nodes in the network.
Handles both regular connections and self-connections.
Adds the new connection object(s) to the appropriate network list (`connections` or `selfconns`).

Returns: An array containing the newly created connection object(s). Typically contains one connection, but might be empty or contain more in specialized node types.

#### connections

Connection list.

#### createMLP

`(inputCount: number, hiddenCounts: number[], outputCount: number) => import("src/architecture/network").default`

Creates a fully connected, strictly layered MLP network.

Returns: A new, fully connected, layered MLP

#### crossOver

`(network1: import("src/architecture/network").default, network2: import("src/architecture/network").default, equal: boolean) => import("src/architecture/network").default`

Creates a new offspring network by performing crossover between two parent networks.
This method implements the crossover mechanism inspired by the NEAT algorithm and described
in the Instinct paper, combining genes (nodes and connections) from both parents.
Fitness scores can influence the inheritance process. Matching genes are inherited randomly,
while disjoint/excess genes are typically inherited from the fitter parent (or randomly if fitness is equal or `equal` flag is set).

Returns: A new Network instance representing the offspring.

#### describeArchitecture

`() => import("src/architecture/network/network.types").NetworkArchitectureDescriptor`

Resolves a stable architecture descriptor for telemetry/UI consumers.

Prefers live graph analysis and only falls back to hydrated serialization
metadata when graph-based resolution is purely inferred.

Returns: Architecture descriptor with hidden-layer widths and provenance.

#### deserialize

`(data: [number[], number[], string[], { from: number; to: number; weight: number; gater: number | null; }[], number, number] | unknown[], inputSize: number | undefined, outputSize: number | undefined) => import("src/architecture/network").default`

Creates a Network instance from serialized data produced by `serialize()`.
Reconstructs the network structure and state based on the provided arrays.

Returns: A new Network instance reconstructed from the serialized data.

#### disableDropConnect

`() => void`

Disable DropConnect.

#### disableStochasticDepth

`() => void`

Disable stochastic depth.

#### disableWeightNoise

`() => void`

Disable all weight-noise settings.

#### disconnect

`(from: import("src/architecture/node").default, to: import("src/architecture/node").default) => void`

Disconnects two nodes, removing the connection between them.
Handles both regular connections and self-connections.
If the connection being removed was gated, it is also ungated.

#### dropout

Dropout probability.

#### enableDropConnect

`(p: number) => void`

Enable DropConnect with a probability in $[0,1)$.

Parameters:
- `p` - DropConnect probability.

#### enableWeightNoise

`(stdDev: number | { perHiddenLayer: number[]; }) => void`

Enable weight noise using either a global standard deviation or per-hidden-layer values.

Parameters:
- `stdDev` - Global standard deviation or hidden-layer schedule.

#### fastSlabActivate

`(input: number[]) => number[]`

Public wrapper for fast slab forward pass.

Parameters:
- `input` - Input vector.

Returns: Activation output.

#### fromJSON

`(json: Record<string, unknown>) => import("src/architecture/network").default`

Reconstructs a network from a JSON object (latest standard).
Handles formatVersion, robust error handling, and index-based references.

Returns: The reconstructed network.

#### gate

`(node: import("src/architecture/node").default, connection: import("src/architecture/connection").default) => void`

Gates a connection with a specified node.
The activation of the `node` (gater) will modulate the weight of the `connection`.
Adds the connection to the network's `gates` list.

#### gates

Network gates collection.

#### getConnectionSlab

`() => import("src/architecture/network/slab/network.slab.utils.types").ConnectionSlabView`

Read slab structures for fast activation.

Returns: Slab connection structures.

#### getCurrentSparsity

`() => number`

Compute the current connection sparsity ratio.

Returns: Current sparsity in $[0,1]$.

#### getLastGradClipGroupCount

`() => number`

Returns last gradient clipping group count (0 if no clipping yet).

#### getLossScale

`() => number`

Returns current mixed precision loss scale (1 if disabled).

#### getRawGradientNorm

`() => number`

Returns last recorded raw (pre-update) gradient L2 norm.

#### getRegularizationStats

`() => Record<string, unknown> | null`

Read regularization statistics collected during training.

Returns: Regularization stats payload.

#### getRNGState

`() => number | undefined`

Read the raw deterministic RNG state word.

Returns: RNG state value when present.

#### getTrainingStats

`() => { gradNorm: number; gradNormRaw: number; lossScale: number; optimizerStep: number; mp: { good: number; bad: number; overflowCount: number; scaleUps: number; scaleDowns: number; lastOverflowStep: number; }; }`

Consolidated training stats snapshot.

#### input

Input node count.

#### lastSkippedLayers

Last skipped stochastic-depth layers from activation runtime state.

#### layers

Optional layered view cache.

#### mutate

`(method: import("src/architecture/network/network.types").MutationMethod) => void`

Mutates the network's structure or parameters according to the specified method.
This is a core operation for neuro-evolutionary algorithms (like NEAT).
The method argument should be one of the mutation types defined in `methods.mutation`.

Parameters:
- `method` - - The mutation method to apply (e.g., `mutation.ADD_NODE`, `mutation.MOD_WEIGHT`).
  Some methods might have associated parameters (e.g., `MOD_WEIGHT` uses `min`, `max`).

#### nodes

Network node collection.

#### noTraceActivate

`(input: number[]) => number[]`

Activates the network without calculating eligibility traces.
This is a performance optimization for scenarios where backpropagation is not needed,
such as during testing, evaluation, or deployment (inference).

Returns: An array of numerical values representing the activations of the network's output nodes.

#### output

Output node count.

#### propagate

`(rate: number, momentum: number, update: boolean, target: number[], regularization: number, costDerivative: ((target: number, output: number) => number) | undefined) => void`

Propagates the error backward through the network (backpropagation).
Calculates the error gradient for each node and connection.
If `update` is true, it adjusts the weights and biases based on the calculated gradients,
learning rate, momentum, and optional L2 regularization.

The process starts from the output nodes and moves backward layer by layer (or topologically for recurrent nets).

#### pruneToSparsity

`(targetSparsity: number, method: "magnitude" | "snip") => void`

Immediately prune connections to reach (or approach) a target sparsity fraction.
Used by evolutionary pruning (generation-based) independent of training iteration schedule.

Parameters:
- `targetSparsity` - fraction in (0,1). 0.8 means keep 20% of original (if first call sets baseline)
- `method` - 'magnitude' | 'snip'

#### rebuildConnections

`(net: import("src/architecture/network").default) => void`

Rebuilds the network's connections array from all per-node connections.
This ensures that the network.connections array is consistent with the actual
outgoing connections of all nodes. Useful after manual wiring or node manipulation.

Returns: Example usage:
  Network.rebuildConnections(net);

#### rebuildConnectionSlab

`(force: boolean) => void`

Rebuild slab structures for fast activation.

Parameters:
- `force` - Whether to force a rebuild.

Returns: Slab rebuild result.

#### remove

`(node: import("src/architecture/node").default) => void`

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

`() => void`

Resets all masks in the network to 1 (no dropout). Applies to both node-level and layer-level dropout.
Should be called after training to ensure inference is unaffected by previous dropout.

#### restoreRNG

`(fn: () => number) => void`

Restore deterministic RNG function from a snapshot source.

Parameters:
- `fn` - RNG function to restore.

#### score

Optional fitness score.

#### selfconns

Self-connection list.

#### serialize

`() => [number[], number[], string[], import("src/architecture/network/network.types").SerializedConnection[], number, number]`

Lightweight tuple serializer delegating to network.serialize.ts

#### set

`(values: { bias?: number | undefined; squash?: ((x: number, derivate?: boolean | undefined) => number) | undefined; }) => void`

Sets specified properties (e.g., bias, squash function) for all nodes in the network.
Useful for initializing or resetting node properties uniformly.

#### setEnforceAcyclic

`(flag: boolean) => void`

Enable or disable acyclic topology enforcement.

Parameters:
- `flag` - Whether to enforce acyclic connectivity.

#### setRandom

`(fn: () => number) => void`

Replace the network random number generator.

Parameters:
- `fn` - RNG function returning values in $[0,1)$.

#### setRNGState

`(state: number) => void`

Set the raw deterministic RNG state word.

Parameters:
- `state` - RNG state value.

#### setSeed

`(seed: number) => void`

Seed the internal deterministic RNG.

Parameters:
- `seed` - Seed value.

#### setStochasticDepth

`(survival: number[]) => void`

Configure stochastic depth with survival probabilities per hidden layer.

Parameters:
- `survival` - Survival probabilities for hidden layers.

#### setStochasticDepthSchedule

`(fn: (step: number, current: number[]) => number[]) => void`

Set stochastic-depth schedule function.

Parameters:
- `fn` - Function mapping step and current schedule to next schedule.

#### setWeightNoiseSchedule

`(fn: (step: number) => number) => void`

Set a dynamic scheduler for global weight noise.

Parameters:
- `fn` - Function mapping training step to noise standard deviation.

#### snapshotRNG

`() => import("src/architecture/network/network.types").RNGSnapshot`

Snapshot deterministic RNG runtime state.

Returns: Current RNG snapshot.

#### test

`(set: { input: number[]; output: number[]; }[], cost: ((target: number[], output: number[]) => number) | undefined) => { error: number; time: number; }`

Tests the network's performance on a given dataset.
Calculates the average error over the dataset using a specified cost function.
Uses `noTraceActivate` for efficiency as gradients are not needed.
Handles dropout scaling if dropout was used during training.

Returns: An object containing the calculated average error over the dataset and the time taken for the test in milliseconds.

#### testForceOverflow

`() => void`

Force the next mixed-precision overflow path (test utility).

#### toJSON

`() => Record<string, unknown>`

Converts the network into a JSON object representation (latest standard).
Includes formatVersion, and only serializes properties needed for full reconstruction.
All references are by index. Excludes runtime-only properties (activation, state, traces).

Returns: A JSON-compatible object representing the network.

#### toONNX

`() => import("src/architecture/network/onnx/network.onnx.utils.types").OnnxModel`

Exports the network to ONNX format (JSON object, minimal MLP support).
Only standard feedforward architectures and standard activations are supported.
Gating, custom activations, and evolutionary features are ignored or replaced with Identity.

Returns: ONNX model as a JSON object.

#### trainingStep

Current training step counter.

#### ungate

`(connection: import("src/architecture/connection").default) => void`

Removes the gate from a specified connection.
The connection will no longer be modulated by its gater node.
Removes the connection from the network's `gates` list.

## architecture/nodePool.ts

### nodePool

NodePool (Phase 2 – COMPLETE)
=============================
Lightweight object pool for `Node` instances mirroring (future) connection pooling patterns.

Objectives:
1. Reduce GC pressure during topology mutation / morphogenesis (frequent add/remove of nodes).
2. Provide deterministic, fully-reset instances on `acquire()` so algorithms can assume a fresh state.
3. Provide instrumentation (reused vs fresh, highWaterMark, recycledRatio) consumed by benchmarks.
4. Serve as a future anchor for slab-backed / SoA node state (Phase 3) without altering the public API.

Phase 2 Deliverables Implemented Here:
- acquire / release with thorough reset and defensive scrub on release.
- highWaterMark updated ONLY on release (tracks retained capacity not transient demand).
- Counters reusedCount & freshCount powering recycledRatio assertions.
- resetNodePool() for deterministic test harness setup.

Deferred (Phase 3+): preWarm(count), adaptive trim(), leak pattern heuristics, slab field hydration.

### acquireNode

`(opts: import("src/architecture/nodePool").AcquireNodeOptions) => import("src/architecture/node").default`

### AcquireNodeOptions

Options bag for acquiring a node.

### nodePoolStats

`() => { size: number; highWaterMark: number; reused: number; fresh: number; recycledRatio: number; }`

### releaseNode

`(node: import("src/architecture/node").default) => void`

### resetNodePool

`() => void`

## architecture/architect.ts

### architect

Provides static methods for constructing various predefined neural network architectures.

The Architect class simplifies the creation of common network types like Multi-Layer Perceptrons (MLPs),
Long Short-Term Memory (LSTM) networks, Gated Recurrent Units (GRUs), and more complex structures
inspired by neuro-evolutionary algorithms. It leverages the underlying `Layer`, `Group`, and `Node`
components to build interconnected `Network` objects.

Methods often utilize helper functions from `Layer` (e.g., `Layer.dense`, `Layer.lstm`) and
connection strategies from `methods.groupConnection`.

### Architect

Provides static methods for constructing various predefined neural network architectures.

The Architect class simplifies the creation of common network types like Multi-Layer Perceptrons (MLPs),
Long Short-Term Memory (LSTM) networks, Gated Recurrent Units (GRUs), and more complex structures
inspired by neuro-evolutionary algorithms. It leverages the underlying `Layer`, `Group`, and `Node`
components to build interconnected `Network` objects.

Methods often utilize helper functions from `Layer` (e.g., `Layer.dense`, `Layer.lstm`) and
connection strategies from `methods.groupConnection`.

### default

#### construct

`(list: (import("src/architecture/node").default | import("src/architecture/layer").default | import("src/architecture/group").default)[]) => import("src/architecture/network").default`

Constructs a Network instance from an array of interconnected Layers, Groups, or Nodes.

This method processes the input list, extracts all unique nodes, identifies connections,
gates, and self-connections, and determines the network's input and output sizes based
on the `type` property ('input' or 'output') set on the nodes. It uses Sets internally
for efficient handling of unique elements during construction.

Returns: A Network object representing the constructed architecture.

#### enforceMinimumHiddenLayerSizes

`(network: import("src/architecture/network").default) => import("src/architecture/network").default`

Enforces the minimum hidden layer size rule on a network.

This ensures that all hidden layers have at least min(input, output) + 1 nodes,
which is a common heuristic to ensure networks have adequate representation capacity.

Returns: The same network with properly sized hidden layers

#### gru

`(layers: number[]) => import("src/architecture/network").default`

Creates a Gated Recurrent Unit (GRU) network.
GRUs are another type of recurrent neural network, similar to LSTMs but often simpler.
This constructor uses `Layer.gru` to create the core GRU blocks.

Returns: The constructed GRU network.

#### hopfield

`(size: number) => import("src/architecture/network").default`

Creates a Hopfield network.
Hopfield networks are a form of recurrent neural network often used for associative memory tasks.
This implementation creates a simple, fully connected structure.

Returns: The constructed Hopfield network.

#### lstm

`(layerArgs: (number | { inputToOutput?: boolean | undefined; })[]) => import("src/architecture/network").default`

Creates a Long Short-Term Memory (LSTM) network.
LSTMs are a type of recurrent neural network (RNN) capable of learning long-range dependencies.
This constructor uses `Layer.lstm` to create the core LSTM blocks.

Returns: The constructed LSTM network.

#### narx

`(inputSize: number, hiddenLayers: number | number[], outputSize: number, previousInput: number, previousOutput: number) => import("src/architecture/network").default`

Creates a Nonlinear AutoRegressive network with eXogenous inputs (NARX).
NARX networks are recurrent networks often used for time series prediction.
They predict the next value of a time series based on previous values of the series
and previous values of external (exogenous) input series.

Returns: The constructed NARX network.

#### perceptron

`(layers: number[]) => import("src/architecture/network").default`

Creates a standard Multi-Layer Perceptron (MLP) network.
An MLP consists of an input layer, one or more hidden layers, and an output layer,
fully connected layer by layer.

Returns: The constructed MLP network.

#### random

`(input: number, hidden: number, output: number, options: { connections?: number | undefined; backconnections?: number | undefined; selfconnections?: number | undefined; gates?: number | undefined; }) => import("src/architecture/network").default`

Creates a randomly structured network based on specified node counts and connection options.

This method allows for the generation of networks with a less rigid structure than MLPs.
It initializes a network with input and output nodes and then iteratively adds hidden nodes
and various types of connections (forward, backward, self) and gates using mutation methods.
This approach is inspired by neuro-evolution techniques where network topology evolves.

Returns: The constructed network with a randomized topology.

## architecture/connection.ts

### connection

Connection (Synapse / Edge)
===========================
Directed weighted link between two nodes. Extends the minimal (from,to,weight)
trio with optional features that are *allocated lazily* for efficiency:
 - Gain modulation (virtualized gain property; omitted when 1)
 - Gating node (symbol-backed; presence tracked via bit flag)
 - Plasticity rate (bit flag + optional slab field)
 - Optimizer moment bag (created only when an optimizer writes to it)

Educational design pattern: Use bit flags + symbol-backed optional fields to illustrate
how to minimize hidden class bloat while keeping the API ergonomic.

### ConnectionSymbolProps

Internal interface for accessing symbol-keyed properties on Connection instances.
Used for type-safe access to dynamic symbol properties.

### default

#### _flags

Packed state flags (private for future-proofing hidden class):
bit0 => enabled gene expression (1 = active)
bit1 => DropConnect active mask (1 = not dropped this forward pass)
bit2 => hasGater (1 = symbol field present)
bit3 => plastic (plasticityRate > 0)
bits4+ reserved.

#### acquire

`(from: import("src/architecture/node").default, to: import("src/architecture/node").default, weight: number | undefined) => import("src/architecture/connection").default`

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

#### plastic

Whether this connection participates in plastic adaptation (rate > 0).

#### plasticityRate

Per-connection plasticity / learning rate (0 means non-plastic). Setting >0 marks plastic flag.

#### previousDeltaWeight

Last applied delta weight (used by classic momentum).

#### release

`(conn: import("src/architecture/connection").default) => void`

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

`() => { from: number | undefined; to: number | undefined; weight: number; gain: number; innovation: number; enabled: boolean; gater?: number | undefined; }`

Serialize to a minimal JSON-friendly shape (used for saving genomes / networks).
Undefined indices are preserved as `undefined` to allow later resolution / remapping.

Returns: Object with node indices, weight, gain, gater index (if any), innovation id & enabled flag.

#### totalDeltaWeight

Accumulated (batched) delta weight awaiting an apply step.

#### weight

Scalar multiplier applied to the source activation (prior to gain modulation).

#### xtrace

Extended trace structure for modulatory / eligibility propagation algorithms. Parallel arrays for cache-friendly iteration.

## architecture/activationArrayPool.ts

### activationArrayPool

Activation array pooling utilities.

Size-bucketed pool for reusable activation arrays to reduce allocations in
hot forward paths. Reused arrays are zero-filled to prevent stale data.
Array type honors global precision via `config.float32Mode`.

### ActivationArray

Allowed activation array shapes for pooling.
- number[]: default JS array
- Float32Array: compact typed array when float32 mode is enabled
- Float64Array: supported for compatibility with typed math paths

### ActivationArrayPool

A size-bucketed pool of activation arrays.

Buckets map array length -> stack of arrays. Acquire pops and zero-fills, or
allocates a new array when empty. Release pushes back up to a configurable
per-bucket cap to avoid unbounded growth.

Note: not thread-safe; intended for typical single-threaded JS execution.
