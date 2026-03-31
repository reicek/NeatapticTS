# architecture/architect

Core architect chapter for the architecture surface.

This folder owns the top-level builder entrypoint that turns folderized
node, connection, group, layer, and network chapters into named neural
network presets.

Read this chapter in three passes:

1. start with `construct()` to see how pre-wired primitives become one
   `Network` instance,
2. continue to `perceptron()` and `random()` when you want feed-forward and
   topology-search-friendly builders,
3. finish with `lstm()`, `gru()`, `hopfield()`, and `narx()` when you need
   recurrent presets built from the lower-level architecture chapters.

Example:

```ts
const network = Architect.perceptron(2, 4, 1);
const output = network.activate([0, 1]);
```

## architecture/architect/architect.ts

### Architect

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

## architecture/architect/architect.errors.ts

Raised when architect construction cannot infer input/output nodes from supplied primitives.

### ArchitectInputOutputTypeResolutionError

Raised when architect construction cannot infer input/output nodes from supplied primitives.

### ArchitectInvalidGruConfigurationError

Raised when a GRU builder receives too few layer sizes.

### ArchitectInvalidLstmConfigurationError

Raised when an LSTM builder receives too few layer sizes.

### ArchitectInvalidLstmLayerArgumentsError

Raised when LSTM builder arguments contain invalid layer-size values.

### ArchitectInvalidPerceptronConfigurationError

Raised when an MLP builder receives too few layer sizes.

### ArchitectZeroInputOutputNodesError

Raised when architect construction produces a network with zero inputs or outputs.
