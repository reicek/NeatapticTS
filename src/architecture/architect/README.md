# architecture/architect

Core architect chapter for the architecture surface.

This folder owns the top-level builder entrypoint that turns folderized
node, connection, group, layer, and network chapters into named neural
network presets.

Read this chapter in three passes:

1. start with `construct()` to see how pre-wired primitives become one
   `Network` instance,
2. continue to `perceptron()`, `randomSparse()`, and `random()` when you
   want feed-forward and topology-search-friendly builders,
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
  layerArgs: (number | ArchitectRecurrentShortcutOptions)[],
): default
```

Creates a Gated Recurrent Unit network.

This builder keeps the GRU graph explicit: update, inverse-update, reset,
memory, output, and previous-output groups are all wired from the public
primitive surface rather than hidden behind a fused recurrent runtime.

The optional `inputToOutput` shortcut is disabled by default so existing
GRU topologies keep their historical shape. Enable it when you want a
direct input-to-readout path in addition to the recurrent block stack.

Parameters:
- `layerArgs` - Layer sizes plus an optional trailing options object.

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
  layerArgs: (number | ArchitectRecurrentShortcutOptions)[],
): default
```

Creates a Long Short-Term Memory network.

This builder keeps the LSTM graph explicit: each recurrent block is
assembled from gate groups, a memory-cell group, and an output block using
the same primitive wiring surface used elsewhere in the architecture layer.

The optional `inputToOutput` shortcut preserves the historical builder
behavior by default. Disable it when you want the public preset to route
information strictly through the recurrent block stack.

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

This is the smallest stateful preset in the public builder surface. The
main processing path receives the current exogenous input plus two explicit
delay lines: one for recent inputs and one for recent outputs. That keeps
the temporal story readable for debugging, visualization, and evolution
work without immediately jumping to gated cells.

Both delay lines are built from `Layer.memory(...)` blocks. Those memory
blocks use identity activation, zero bias, and one-to-one unit carry links,
so remembered values behave like a deterministic rolling window rather than
a learned recurrent cell.

Clear-state guidance: call `network.clear()` before starting a new
independent sequence, episode, or evaluation run. If you keep activating
the same runtime without clearing it, the delay lines intentionally carry
their terminal state into the next activation stream.

Parameters:
- `inputSize` - The exogenous input size at each time step.
- `hiddenLayers` - Hidden layer sizes, or zero / empty for none.
- `outputSize` - The prediction output size.
- `previousInput` - The number of delayed input steps.
- `previousOutput` - The number of delayed output steps.

Returns: The constructed NARX network.

Example:

```ts
const network = Architect.narx(1, [4], 1, 2, 1);

for (const sample of sequenceA) {
  network.activate(sample.input);
}

network.clear();

for (const sample of sequenceB) {
  network.activate(sample.input);
}
```

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
  options: ArchitectLegacyRandomOptions,
): default
```

Creates a randomly structured network based on node counts and connection
options.

This compatibility wrapper preserves the historical `random()` surface
while forwarding to the stricter `randomSparse()` builder that uses the
Phase 2 sparse-profile vocabulary.

Parameters:
- `input` - The number of input nodes.
- `hidden` - The number of hidden nodes to add.
- `output` - The number of output nodes.
- `options` - Optional legacy configuration using lowercase back/self keys.

Returns: The constructed randomized network.

#### randomSparse

```ts
randomSparse(
  input: number,
  hidden: number,
  output: number,
  options: ArchitectRandomSparseOptions,
): default
```

Creates a sparse random starting graph for topology-oriented search.

This builder is the explicit sparse-profile entrypoint for the public
architecture set. It accepts camelCase option names, validates the request
up front, and throws a clear error as soon as one requested structural edit
cannot be satisfied instead of silently leaving the graph underspecified.

Parameters:
- `input` - The number of input nodes.
- `hidden` - The number of hidden nodes to add.
- `output` - The number of output nodes.
- `options` - Optional sparse-structure counts for forward connections,
back connections, self connections, gates, and an optional deterministic seed.

Returns: The constructed sparse random network.

Example:

```ts
const network = Architect.randomSparse(3, 6, 2, {
  connections: 12,
  backConnections: 2,
  selfConnections: 1,
  seed: 7,
});
```

## architecture/architect/architect.errors.ts

Raised when architect construction cannot infer input/output nodes from supplied primitives.

### ArchitectInputOutputTypeResolutionError

Raised when architect construction cannot infer input/output nodes from supplied primitives.

### ArchitectInvalidGruConfigurationError

Raised when a GRU builder receives too few layer sizes.

### ArchitectInvalidGruLayerArgumentsError

Raised when GRU builder arguments contain invalid layer-size values.

### ArchitectInvalidLstmConfigurationError

Raised when an LSTM builder receives too few layer sizes.

### ArchitectInvalidLstmLayerArgumentsError

Raised when LSTM builder arguments contain invalid layer-size values.

### ArchitectInvalidPerceptronConfigurationError

Raised when an MLP builder receives too few layer sizes.

### ArchitectInvalidRandomSparseConfigurationError

Raised when a sparse architect builder receives invalid dimensions or
requests more structural edits than the graph can satisfy.

### ArchitectZeroInputOutputNodesError

Raised when architect construction produces a network with zero inputs or outputs.
