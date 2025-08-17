# architecture

## architecture/activationArrayPool.ts

### ActivationArray

Allowed activation array shapes for pooling.
- number[]: default JS array
- Float32Array: compact typed array when float32 mode is enabled
- Float64Array: supported for compatibility with typed math paths

### activationArrayPool

### ActivationArrayPool

A size-bucketed pool of activation arrays.

Buckets map array length -> stack of arrays. Acquire pops and zero-fills, or
allocates a new array when empty. Release pushes back up to a configurable
per-bucket cap to avoid unbounded growth.

Note: not thread-safe; intended for typical single-threaded JS execution.

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

`(list: (import("D:/code-practice/NeatapticTS/src/architecture/node").default | import("D:/code-practice/NeatapticTS/src/architecture/layer").default | import("D:/code-practice/NeatapticTS/src/architecture/group").default)[]) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Constructs a Network instance from an array of interconnected Layers, Groups, or Nodes.

This method processes the input list, extracts all unique nodes, identifies connections,
gates, and self-connections, and determines the network's input and output sizes based
on the `type` property ('input' or 'output') set on the nodes. It uses Sets internally
for efficient handling of unique elements during construction.

Parameters:
- `` - - An array containing the building blocks (Nodes, Layers, Groups) of the network, assumed to be already interconnected.

Returns: A Network object representing the constructed architecture.

#### enforceMinimumHiddenLayerSizes

`(network: import("D:/code-practice/NeatapticTS/src/architecture/network").default) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Enforces the minimum hidden layer size rule on a network.

This ensures that all hidden layers have at least min(input, output) + 1 nodes,
which is a common heuristic to ensure networks have adequate representation capacity.

Parameters:
- `` - - The network to enforce minimum hidden layer sizes on

Returns: The same network with properly sized hidden layers

#### gru

`(layers: number[]) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Creates a Gated Recurrent Unit (GRU) network.
GRUs are another type of recurrent neural network, similar to LSTMs but often simpler.
This constructor uses `Layer.gru` to create the core GRU blocks.

Parameters:
- `` - - A sequence of numbers representing the size (number of units) of each layer: input layer size, hidden GRU layer sizes..., output layer size. Must include at least input, one hidden, and output layer sizes.

Returns: The constructed GRU network.

#### hopfield

`(size: number) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Creates a Hopfield network.
Hopfield networks are a form of recurrent neural network often used for associative memory tasks.
This implementation creates a simple, fully connected structure.

Parameters:
- `` - - The number of nodes in the network (input and output layers will have this size).

Returns: The constructed Hopfield network.

#### lstm

`(layerArgs: (number | { inputToOutput?: boolean | undefined; })[]) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Creates a Long Short-Term Memory (LSTM) network.
LSTMs are a type of recurrent neural network (RNN) capable of learning long-range dependencies.
This constructor uses `Layer.lstm` to create the core LSTM blocks.

Parameters:
- `` - - A sequence of arguments defining the network structure:
- Numbers represent the size (number of units) of each layer: input layer size, hidden LSTM layer sizes..., output layer size.
- An optional configuration object can be provided as the last argument.
- `` - - Configuration options (if passed as the last argument).

Returns: The constructed LSTM network.

#### narx

`(inputSize: number, hiddenLayers: number | number[], outputSize: number, previousInput: number, previousOutput: number) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Creates a Nonlinear AutoRegressive network with eXogenous inputs (NARX).
NARX networks are recurrent networks often used for time series prediction.
They predict the next value of a time series based on previous values of the series
and previous values of external (exogenous) input series.

Parameters:
- `` - - The number of input nodes for the exogenous inputs at each time step.
- `` - - The size of the hidden layer(s). Can be a single number for one hidden layer, or an array of numbers for multiple hidden layers. Use 0 or [] for no hidden layers.
- `` - - The number of output nodes (predicting the time series).
- `` - - The number of past time steps of the exogenous input to feed back into the network.
- `` - - The number of past time steps of the network's own output to feed back into the network (autoregressive part).

Returns: The constructed NARX network.

#### perceptron

`(layers: number[]) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Creates a standard Multi-Layer Perceptron (MLP) network.
An MLP consists of an input layer, one or more hidden layers, and an output layer,
fully connected layer by layer.

Parameters:
- `` - - A sequence of numbers representing the size (number of nodes) of each layer, starting with the input layer, followed by hidden layers, and ending with the output layer. Must include at least input, one hidden, and output layer sizes.

Returns: The constructed MLP network.

#### random

`(input: number, hidden: number, output: number, options: { connections?: number | undefined; backconnections?: number | undefined; selfconnections?: number | undefined; gates?: number | undefined; }) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Creates a randomly structured network based on specified node counts and connection options.

This method allows for the generation of networks with a less rigid structure than MLPs.
It initializes a network with input and output nodes and then iteratively adds hidden nodes
and various types of connections (forward, backward, self) and gates using mutation methods.
This approach is inspired by neuro-evolution techniques where network topology evolves.

Parameters:
- `` - - The number of input nodes.
- `` - - The number of hidden nodes to add.
- `` - - The number of output nodes.
- `` - - Optional configuration for the network structure.

Returns: The constructed network with a randomized topology.

## architecture/connection.ts

### connection

### default

#### acquire

`(from: import("D:/code-practice/NeatapticTS/src/architecture/node").default, to: import("D:/code-practice/NeatapticTS/src/architecture/node").default, weight: number | undefined) => import("D:/code-practice/NeatapticTS/src/architecture/connection").default`

Acquire a Connection from the pool or construct a new one. Ensures fresh innovation id.

#### innovationID

`(a: number, b: number) => number`

Generates a unique innovation ID for the connection.

The innovation ID is calculated using the Cantor pairing function, which maps two integers
(representing the source and target nodes) to a unique integer.

Parameters:
- `` - - The ID of the source node.
- `` - - The ID of the target node.

Returns: The innovation ID based on the Cantor pairing function.

#### release

`(conn: import("D:/code-practice/NeatapticTS/src/architecture/connection").default) => void`

Return a Connection to the pool for reuse.

#### toJSON

`() => any`

Converts the connection to a JSON object for serialization.

Returns: A JSON representation of the connection.

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

Parameters:
- `` - - An optional array of input values. If provided, its length must match the number of nodes in the group.

Returns: An array containing the activation value of each node in the group, in order.

#### clear

`() => void`

Resets the state of all nodes in the group. This typically involves clearing
activation values, state, and propagated errors, preparing the group for a new input pattern,
especially relevant in recurrent networks or sequence processing.

#### connect

`(target: import("D:/code-practice/NeatapticTS/src/architecture/node").default | import("D:/code-practice/NeatapticTS/src/architecture/layer").default | import("D:/code-practice/NeatapticTS/src/architecture/group").default, method: any, weight: number | undefined) => any[]`

Establishes connections from all nodes in this group to a target Group, Layer, or Node.
The connection pattern (e.g., all-to-all, one-to-one) can be specified.

Parameters:
- `` - - The destination entity (Group, Layer, or Node) to connect to.
- `` - - The connection method/type (e.g., `methods.groupConnection.ALL_TO_ALL`, `methods.groupConnection.ONE_TO_ONE`). Defaults depend on the target type and whether it's the same group.
- `` - - An optional fixed weight to assign to all created connections. If not provided, weights might be initialized randomly or based on node defaults.

Returns: An array containing all the connection objects created. Consider using a more specific type like `Connection[]`.

#### connections

Stores connection information related to this group.
`in`: Connections coming into any node in this group from outside.
`out`: Connections going out from any node in this group to outside.
`self`: Connections between nodes within this same group (e.g., in ONE_TO_ONE connections).

#### disconnect

`(target: import("D:/code-practice/NeatapticTS/src/architecture/node").default | import("D:/code-practice/NeatapticTS/src/architecture/group").default, twosided: boolean) => void`

Removes connections between nodes in this group and a target Group or Node.

Parameters:
- `` - - The Group or Node to disconnect from.
- `` - - If true, also removes connections originating from the `target` and ending in this group. Defaults to false (only removes connections from this group to the target).

#### gate

`(connections: any, method: any) => void`

Configures nodes within this group to act as gates for the specified connection(s).
Gating allows the output of a node in this group to modulate the flow of signal through the gated connection.

Parameters:
- `` - - A single connection object or an array of connection objects to be gated. Consider using a more specific type like `Connection | Connection[]`.
- `` - - The gating mechanism to use (e.g., `methods.gating.INPUT`, `methods.gating.OUTPUT`, `methods.gating.SELF`). Specifies which part of the connection is influenced by the gater node.

#### nodes

An array holding all the nodes within this group.

#### propagate

`(rate: number, momentum: number, target: number[] | undefined) => void`

Propagates the error backward through all nodes in the group. If target values are provided,
the error is calculated against these targets (typically for output layers). Otherwise,
the error is calculated based on the error propagated from subsequent layers/nodes.

Parameters:
- `` - - The learning rate to apply during weight updates.
- `` - - The momentum factor to apply during weight updates.
- `` - - Optional target values for error calculation. If provided, its length must match the number of nodes.

#### set

`(values: { bias?: number | undefined; squash?: any; type?: string | undefined; }) => void`

Sets specific properties (like bias, squash function, or type) for all nodes within the group.

Parameters:
- `` - - An object containing the properties and their new values. Only provided properties are updated.
`bias`: Sets the bias term for all nodes.
`squash`: Sets the activation function (squashing function) for all nodes.
`type`: Sets the node type (e.g., 'input', 'hidden', 'output') for all nodes.

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

`(size: number, heads: number) => import("D:/code-practice/NeatapticTS/src/architecture/layer").default`

Creates a multi-head self-attention layer (stub implementation).

Parameters:
- `size` - - Number of output nodes.
- `heads` - - Number of attention heads (default 1).

Returns: A new Layer instance representing an attention layer.

#### batchNorm

`(size: number) => import("D:/code-practice/NeatapticTS/src/architecture/layer").default`

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

`(target: import("D:/code-practice/NeatapticTS/src/architecture/node").default | import("D:/code-practice/NeatapticTS/src/architecture/layer").default | import("D:/code-practice/NeatapticTS/src/architecture/group").default, method: any, weight: number | undefined) => any[]`

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

`(size: number, kernelSize: number, stride: number, padding: number) => import("D:/code-practice/NeatapticTS/src/architecture/layer").default`

Creates a 1D convolutional layer (stub implementation).

Parameters:
- `size` - - Number of output nodes (filters).
- `kernelSize` - - Size of the convolution kernel.
- `stride` - - Stride of the convolution (default 1).
- `padding` - - Padding (default 0).

Returns: A new Layer instance representing a 1D convolutional layer.

#### dense

`(size: number) => import("D:/code-practice/NeatapticTS/src/architecture/layer").default`

Creates a standard fully connected (dense) layer.

All nodes in the source layer/group will connect to all nodes in this layer
when using the default `ALL_TO_ALL` connection method via `layer.input()`.

Parameters:
- `size` - - The number of nodes (neurons) in this layer.

Returns: A new Layer instance configured as a dense layer.

#### disconnect

`(target: import("D:/code-practice/NeatapticTS/src/architecture/node").default | import("D:/code-practice/NeatapticTS/src/architecture/group").default, twosided: boolean | undefined) => void`

Removes connections between this layer's nodes and a target Group or Node.

Parameters:
- `target` - - The Group or Node to disconnect from.
- `twosided` - - If true, removes connections in both directions (from this layer to target, and from target to this layer). Defaults to false.

#### dropout

Dropout rate for this layer (0 to 1). If > 0, all nodes in the layer are masked together during training.
Layer-level dropout takes precedence over node-level dropout for nodes in this layer.

#### gate

`(connections: any[], method: any) => void`

Applies gating to a set of connections originating from this layer's output group.

Gating allows the activity of nodes in this layer (specifically, the output group)
to modulate the flow of information through the specified `connections`.

Parameters:
- `connections` - - An array of connection objects to be gated.
- `method` - - The gating method (e.g., `INPUT`, `OUTPUT`, `SELF`) specifying how the gate influences the connection. See `methods.gating`.

#### gru

`(size: number) => import("D:/code-practice/NeatapticTS/src/architecture/layer").default`

Creates a Gated Recurrent Unit (GRU) layer.

GRUs are another type of recurrent neural network cell, often considered
simpler than LSTMs but achieving similar performance on many tasks.
They use an update gate and a reset gate to manage information flow.

Parameters:
- `size` - - The number of GRU units (and nodes in each gate/cell group).

Returns: A new Layer instance configured as a GRU layer.

#### input

`(from: import("D:/code-practice/NeatapticTS/src/architecture/layer").default | import("D:/code-practice/NeatapticTS/src/architecture/group").default, method: any, weight: number | undefined) => any[]`

Handles the connection logic when this layer is the *target* of a connection.

It connects the output of the `from` layer or group to this layer's primary
input mechanism (which is often the `output` group itself, but depends on the layer type).
This method is usually called by the `connect` method of the source layer/group.

Parameters:
- `from` - - The source Layer or Group connecting *to* this layer.
- `method` - - The connection method (e.g., `ALL_TO_ALL`). Defaults to `ALL_TO_ALL`.
- `weight` - - An optional fixed weight for the connections.

Returns: An array containing the newly created connection objects.

#### isGroup

`(obj: any) => boolean`

Type guard to check if an object is likely a `Group`.

This is a duck-typing check based on the presence of expected properties
(`set` method and `nodes` array). Used internally where `layer.nodes`
might contain `Group` instances (e.g., in `Memory` layers).

Parameters:
- `obj` - - The object to inspect.

Returns: `true` if the object has `set` and `nodes` properties matching a Group, `false` otherwise.

#### layerNorm

`(size: number) => import("D:/code-practice/NeatapticTS/src/architecture/layer").default`

Creates a layer normalization layer.
Applies layer normalization to the activations of the nodes in this layer during activation.

Parameters:
- `size` - - The number of nodes in this layer.

Returns: A new Layer instance configured as a layer normalization layer.

#### lstm

`(size: number) => import("D:/code-practice/NeatapticTS/src/architecture/layer").default`

Creates a Long Short-Term Memory (LSTM) layer.

LSTMs are a type of recurrent neural network (RNN) cell capable of learning
long-range dependencies. This implementation uses standard LSTM architecture
with input, forget, and output gates, and a memory cell.

Parameters:
- `size` - - The number of LSTM units (and nodes in each gate/cell group).

Returns: A new Layer instance configured as an LSTM layer.

#### memory

`(size: number, memory: number) => import("D:/code-practice/NeatapticTS/src/architecture/layer").default`

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

`(values: { bias?: number | undefined; squash?: any; type?: string | undefined; }) => void`

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

#### _applyGradientClipping

`(cfg: { mode: "norm" | "percentile" | "layerwiseNorm" | "layerwisePercentile"; maxNorm?: number | undefined; percentile?: number | undefined; }) => void`

Trains the network on a given dataset subset for one pass (epoch or batch).
Performs activation and backpropagation for each item in the set.
Updates weights based on batch size configuration.

Parameters:
- `` - - The training dataset subset (e.g., a batch or the full set for one epoch).
- `` - - The number of samples to process before updating weights.
- `` - - The learning rate to use for this training pass.
- `` - - The momentum factor to use.
- `` - - The regularization configuration (L1, L2, or custom function).
- `` - - The function used to calculate the error between target and output.

Returns: The average error calculated over the provided dataset subset.

#### activate

`(input: number[], training: boolean, maxActivationDepth: number) => number[]`

Activates the network using the given input array.
Performs a forward pass through the network, calculating the activation of each node.

Parameters:
- `` - - An array of numerical values corresponding to the network's input nodes.
- `` - - Flag indicating if the activation is part of a training process.
- `` - - Maximum allowed activation depth to prevent infinite loops/cycles.

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

`(input: number[], training: boolean, maxActivationDepth: number) => any`

Raw activation that can return a typed array when pooling is enabled (zero-copy).
If reuseActivationArrays=false falls back to standard activate().

#### adjustRateForAccumulation

`(rate: number, accumulationSteps: number, reduction: "average" | "sum") => number`

Utility: adjust rate for accumulation mode (use result when switching to 'sum' to mimic 'average').

#### clear

`() => void`

Clears the internal state of all nodes in the network.
Resets node activation, state, eligibility traces, and extended traces to their initial values (usually 0).
This is typically done before processing a new input sequence in recurrent networks or between training epochs if desired.

#### clone

`() => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Creates a deep copy of the network.

Returns: A new Network instance that is a clone of the current network.

#### connect

`(from: import("D:/code-practice/NeatapticTS/src/architecture/node").default, to: import("D:/code-practice/NeatapticTS/src/architecture/node").default, weight: number | undefined) => import("D:/code-practice/NeatapticTS/src/architecture/connection").default[]`

Creates a connection between two nodes in the network.
Handles both regular connections and self-connections.
Adds the new connection object(s) to the appropriate network list (`connections` or `selfconns`).

Parameters:
- `` - - The source node of the connection.
- `` - - The target node of the connection.
- `` - - Optional weight for the connection. If not provided, a random weight is usually assigned by the underlying `Node.connect` method.

Returns: An array containing the newly created connection object(s). Typically contains one connection, but might be empty or contain more in specialized node types.

#### createMLP

`(inputCount: number, hiddenCounts: number[], outputCount: number) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Creates a fully connected, strictly layered MLP network.

Parameters:
- `` - - Number of input nodes
- `` - - Array of hidden layer sizes (e.g. [2,3] for two hidden layers)
- `` - - Number of output nodes

Returns: A new, fully connected, layered MLP

#### crossOver

`(network1: import("D:/code-practice/NeatapticTS/src/architecture/network").default, network2: import("D:/code-practice/NeatapticTS/src/architecture/network").default, equal: boolean) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Creates a new offspring network by performing crossover between two parent networks.
This method implements the crossover mechanism inspired by the NEAT algorithm and described
in the Instinct paper, combining genes (nodes and connections) from both parents.
Fitness scores can influence the inheritance process. Matching genes are inherited randomly,
while disjoint/excess genes are typically inherited from the fitter parent (or randomly if fitness is equal or `equal` flag is set).

Parameters:
- `` - - The first parent network.
- `` - - The second parent network.
- `` - - If true, disjoint and excess genes are inherited randomly regardless of fitness.
   If false (default), they are inherited from the fitter parent.

Returns: A new Network instance representing the offspring.

#### deserialize

`(data: any[], inputSize: number | undefined, outputSize: number | undefined) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Creates a Network instance from serialized data produced by `serialize()`.
Reconstructs the network structure and state based on the provided arrays.

Parameters:
- `` - - The serialized network data array, typically obtained from `network.serialize()`.
  Expected format: `[activations, states, squashNames, connectionData, inputSize, outputSize]`.
- `` - - Optional input size override.
- `` - - Optional output size override.

Returns: A new Network instance reconstructed from the serialized data.

#### disconnect

`(from: import("D:/code-practice/NeatapticTS/src/architecture/node").default, to: import("D:/code-practice/NeatapticTS/src/architecture/node").default) => void`

Disconnects two nodes, removing the connection between them.
Handles both regular connections and self-connections.
If the connection being removed was gated, it is also ungated.

Parameters:
- `` - - The source node of the connection to remove.
- `` - - The target node of the connection to remove.

#### enableWeightNoise

`(stdDev: number | { perHiddenLayer: number[]; }) => void`

Enable weight noise. Provide a single std dev number or { perHiddenLayer: number[] }.

#### fromJSON

`(json: any) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Reconstructs a network from a JSON object (latest standard).
Handles formatVersion, robust error handling, and index-based references.

Parameters:
- `` - - The JSON object representing the network.

Returns: The reconstructed network.

#### gate

`(node: import("D:/code-practice/NeatapticTS/src/architecture/node").default, connection: import("D:/code-practice/NeatapticTS/src/architecture/connection").default) => void`

Gates a connection with a specified node.
The activation of the `node` (gater) will modulate the weight of the `connection`.
Adds the connection to the network's `gates` list.

Parameters:
- `` - - The node that will act as the gater. Must be part of this network.
- `` - - The connection to be gated.

#### getLastGradClipGroupCount

`() => number`

Returns last gradient clipping group count (0 if no clipping yet).

#### getLossScale

`() => number`

Returns current mixed precision loss scale (1 if disabled).

#### getRawGradientNorm

`() => number`

Returns last recorded raw (pre-update) gradient L2 norm.

#### getTrainingStats

`() => { gradNorm: number; gradNormRaw: number; lossScale: number; optimizerStep: number; mp: { good: number; bad: number; overflowCount: number; scaleUps: number; scaleDowns: number; lastOverflowStep: number; }; }`

Consolidated training stats snapshot.

#### mutate

`(method: any) => void`

Mutates the network's structure or parameters according to the specified method.
This is a core operation for neuro-evolutionary algorithms (like NEAT).
The method argument should be one of the mutation types defined in `methods.mutation`.

Parameters:
- `` - - The mutation method to apply (e.g., `mutation.ADD_NODE`, `mutation.MOD_WEIGHT`).
  Some methods might have associated parameters (e.g., `MOD_WEIGHT` uses `min`, `max`).

#### noTraceActivate

`(input: number[]) => number[]`

Activates the network without calculating eligibility traces.
This is a performance optimization for scenarios where backpropagation is not needed,
such as during testing, evaluation, or deployment (inference).

Parameters:
- `` - - An array of numerical values corresponding to the network's input nodes.
  The length must match the network's `input` size.

Returns: An array of numerical values representing the activations of the network's output nodes.

#### propagate

`(rate: number, momentum: number, update: boolean, target: number[], regularization: number, costDerivative: ((target: number, output: number) => number) | undefined) => void`

Propagates the error backward through the network (backpropagation).
Calculates the error gradient for each node and connection.
If `update` is true, it adjusts the weights and biases based on the calculated gradients,
learning rate, momentum, and optional L2 regularization.

The process starts from the output nodes and moves backward layer by layer (or topologically for recurrent nets).

Parameters:
- `` - - The learning rate (controls the step size of weight adjustments).
- `` - - The momentum factor (helps overcome local minima and speeds up convergence). Typically between 0 and 1.
- `` - - If true, apply the calculated weight and bias updates. If false, only calculate gradients (e.g., for batch accumulation).
- `` - - An array of target values corresponding to the network's output nodes.
  The length must match the network's `output` size.
- `` - - The L2 regularization factor (lambda). Helps prevent overfitting by penalizing large weights.
- `` - - Optional derivative of the cost function for output nodes.

#### pruneToSparsity

`(targetSparsity: number, method: "magnitude" | "snip") => void`

Immediately prune connections to reach (or approach) a target sparsity fraction.
Used by evolutionary pruning (generation-based) independent of training iteration schedule.

Parameters:
- `targetSparsity` - fraction in (0,1). 0.8 means keep 20% of original (if first call sets baseline)
- `method` - 'magnitude' | 'snip'

#### rebuildConnections

`(net: import("D:/code-practice/NeatapticTS/src/architecture/network").default) => void`

Rebuilds the network's connections array from all per-node connections.
This ensures that the network.connections array is consistent with the actual
outgoing connections of all nodes. Useful after manual wiring or node manipulation.

Parameters:
- `` - - The network instance to rebuild connections for.

Returns: Example usage:
  Network.rebuildConnections(net);

#### remove

`(node: import("D:/code-practice/NeatapticTS/src/architecture/node").default) => void`

Removes a node from the network.
This involves:
1. Disconnecting all incoming and outgoing connections associated with the node.
2. Removing any self-connections.
3. Removing the node from the `nodes` array.
4. Attempting to reconnect the node's direct predecessors to its direct successors
   to maintain network flow, if possible and configured.
5. Handling gates involving the removed node (ungating connections gated *by* this node,
   and potentially re-gating connections that were gated *by other nodes* onto the removed node's connections).

Parameters:
- `` - - The node instance to remove. Must exist within the network's `nodes` list.

#### resetDropoutMasks

`() => void`

Resets all masks in the network to 1 (no dropout). Applies to both node-level and layer-level dropout.
Should be called after training to ensure inference is unaffected by previous dropout.

#### serialize

`() => any[]`

Lightweight tuple serializer delegating to network.serialize.ts

#### set

`(values: { bias?: number | undefined; squash?: any; }) => void`

Sets specified properties (e.g., bias, squash function) for all nodes in the network.
Useful for initializing or resetting node properties uniformly.

Parameters:
- `` - - An object containing the properties and values to set.

#### setStochasticDepth

`(survival: number[]) => void`

Configure stochastic depth with survival probabilities per hidden layer (length must match hidden layer count when using layered network).

#### test

`(set: { input: number[]; output: number[]; }[], cost: any) => { error: number; time: number; }`

Tests the network's performance on a given dataset.
Calculates the average error over the dataset using a specified cost function.
Uses `noTraceActivate` for efficiency as gradients are not needed.
Handles dropout scaling if dropout was used during training.

Parameters:
- `` - - The test dataset, an array of objects with `input` and `output` arrays.
- `` - - The cost function to evaluate the error. Defaults to Mean Squared Error.

Returns: An object containing the calculated average error over the dataset and the time taken for the test in milliseconds.

#### toJSON

`() => object`

Converts the network into a JSON object representation (latest standard).
Includes formatVersion, and only serializes properties needed for full reconstruction.
All references are by index. Excludes runtime-only properties (activation, state, traces).

Returns: A JSON-compatible object representing the network.

#### toONNX

`() => import("D:/code-practice/NeatapticTS/src/architecture/network/network.onnx").OnnxModel`

Exports the network to ONNX format (JSON object, minimal MLP support).
Only standard feedforward architectures and standard activations are supported.
Gating, custom activations, and evolutionary features are ignored or replaced with Identity.

Returns: ONNX model as a JSON object.

#### ungate

`(connection: import("D:/code-practice/NeatapticTS/src/architecture/connection").default) => void`

Removes the gate from a specified connection.
The connection will no longer be modulated by its gater node.
Removes the connection from the network's `gates` list.

Parameters:
- `` - - The connection object to ungate.

## architecture/node.ts

### node

Represents a node (neuron) in a neural network graph.

Nodes are the fundamental processing units. They receive inputs, apply an activation function,
and produce an output. Nodes can be of type 'input', 'hidden', or 'output'. Hidden and output
nodes have biases and activation functions, which can be mutated during neuro-evolution.
This class also implements mechanisms for backpropagation, including support for momentum (NAG),
L2 regularization, dropout, and eligibility traces for recurrent connections.

### Node

Represents a node (neuron) in a neural network graph.

Nodes are the fundamental processing units. They receive inputs, apply an activation function,
and produce an output. Nodes can be of type 'input', 'hidden', or 'output'. Hidden and output
nodes have biases and activation functions, which can be mutated during neuro-evolution.
This class also implements mechanisms for backpropagation, including support for momentum (NAG),
L2 regularization, dropout, and eligibility traces for recurrent connections.

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

`(connection: import("D:/code-practice/NeatapticTS/src/architecture/connection").default, delta: number) => void`

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

`(opts: { type: "sgd" | "rmsprop" | "adagrad" | "adam" | "adamw" | "amsgrad" | "adamax" | "nadam" | "radam" | "lion" | "adabelief" | "lookahead"; momentum?: number | undefined; beta1?: number | undefined; beta2?: number | undefined; eps?: number | undefined; weightDecay?: number | undefined; lrScale?: number | undefined; t?: number | undefined; baseType?: any; la_k?: number | undefined; la_alpha?: number | undefined; }) => void`

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
 - opt_m / opt_v / opt_vhat / opt_u : Moment / variance / max variance / infinity norm caches.
 - opt_cache : Single accumulator (RMSProp / AdaGrad).
 - previousDeltaWeight : For classic SGD momentum.
 - _la_shadowWeight / _la_shadowBias : Lookahead shadow copies.

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

`(target: import("D:/code-practice/NeatapticTS/src/architecture/node").default | { nodes: import("D:/code-practice/NeatapticTS/src/architecture/node").default[]; }, weight: number | undefined) => import("D:/code-practice/NeatapticTS/src/architecture/connection").default[]`

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

`(target: import("D:/code-practice/NeatapticTS/src/architecture/node").default, twosided: boolean) => void`

Removes the connection from this node to the target node.

Parameters:
- `target` - The target node to disconnect from.
- `twosided` - If true, also removes the connection from the target node back to this node (if it exists). Defaults to false.

#### error

Stores error values calculated during backpropagation.

#### fromJSON

`(json: { bias: number; type: string; squash: string; mask: number; }) => import("D:/code-practice/NeatapticTS/src/architecture/node").default`

Creates a Node instance from a JSON object.

Parameters:
- `json` - The JSON object containing node configuration.

Returns: A new Node instance configured according to the JSON object.

#### gate

`(connections: import("D:/code-practice/NeatapticTS/src/architecture/connection").default | import("D:/code-practice/NeatapticTS/src/architecture/connection").default[]) => void`

Makes this node gate the provided connection(s).
The connection's gain will be controlled by this node's activation value.

Parameters:
- `connections` - A single Connection object or an array of Connection objects to be gated.

#### gates

**Deprecated:** Use connections.gated; retained for legacy tests

#### geneId

Stable per-node gene identifier for NEAT innovation reuse

#### index

Optional index, potentially used to identify the node's position within a layer or network structure. Not used internally by the Node class itself.

#### isActivating

Internal flag to detect cycles during activation

#### isConnectedTo

`(target: import("D:/code-practice/NeatapticTS/src/architecture/node").default) => boolean`

Checks if this node is connected to another node.

Parameters:
- `target` - The target node to check the connection with.

Returns: True if connected, otherwise false.

#### isProjectedBy

`(node: import("D:/code-practice/NeatapticTS/src/architecture/node").default) => boolean`

Checks if the given node has a direct outgoing connection to this node.
Considers both regular incoming connections and the self-connection.

Parameters:
- `node` - The potential source node.

Returns: True if the given node projects to this node, false otherwise.

#### isProjectingTo

`(node: import("D:/code-practice/NeatapticTS/src/architecture/node").default) => boolean`

Checks if this node has a direct outgoing connection to the given node.
Considers both regular outgoing connections and the self-connection.

Parameters:
- `node` - The potential target node.

Returns: True if this node projects to the target node, false otherwise.

#### mask

A mask factor (typically 0 or 1) used for implementing dropout. If 0, the node's output is effectively silenced.

#### mutate

`(method: any) => void`

Applies a mutation method to the node. Used in neuro-evolution.

This allows modifying the node's properties, such as its activation function or bias,
based on predefined mutation methods.

Parameters:
- `method` - A mutation method object, typically from `methods.mutation`. It should define the type of mutation and its parameters (e.g., allowed functions, modification range).

#### nodes

**Deprecated:** Placeholder kept for legacy structural algorithms. No longer populated.

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

`(connections: import("D:/code-practice/NeatapticTS/src/architecture/connection").default | import("D:/code-practice/NeatapticTS/src/architecture/connection").default[]) => void`

Removes this node's gating control over the specified connection(s).
Resets the connection's gain to 1 and removes it from the `connections.gated` list.

Parameters:
- `connections` - A single Connection object or an array of Connection objects to ungate.

## architecture/onnx.ts

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

Steps:
 1. Read input/output dimensions.
 2. Derive hidden layer sizes from weight tensor shapes.
 3. Create corresponding MLP with identical layer counts.
 4. Assign weights & biases.
 5. Map activation op_types back to internal activation functions.
 6. Rebuild flat connection list.

Limitations: Only guaranteed for self-produced ONNX; inconsistent naming or ordering will break.

### OnnxExportOptions

Options controlling ONNX export behavior (Phase 1).

### OnnxModel
