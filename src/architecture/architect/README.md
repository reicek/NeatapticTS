# architecture/architect

Core architect chapter for the architecture surface.

This folder owns the top-level builder entrypoint that turns folderized
node, connection, group, layer, and network chapters into named neural
network presets.

One practical selection guide is to choose the smallest builder that matches
the kind of memory your task actually needs:

- `perceptron()` for static input-to-output mappings and dense feed-forward baselines,
- `randomSparse()` when evolution should begin from a lighter graph with room to grow,
- `narx()` when a short explicit window of past inputs and outputs is enough,
- `gru()` and `lstm()` when the state itself should be learned inside gated recurrent blocks.

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

Recommended starting knobs:

- start with one small recurrent block before stacking multiple GRU stages,
- supervised fine-tuning: keep `rate` low, usually around `0.001` to
  `0.01`, and preserve sequence order,
- evolution: compare runs with a fixed `seed`, moderate `mutationRate`,
  and only enable `inputToOutput` when the task clearly benefits from a
  direct readout shortcut.

Common pitfalls:

- forgetting to call `clear()` between independent sequences or episodes,
- enabling the shortcut before checking whether the recurrent block alone
  already solves the task,
- shuffling timesteps rather than whole sequences.

Parameters:
- `layerArgs` - Layer sizes plus an optional trailing options object.

Returns: The constructed GRU network.

Example:

```ts
const network = Architect.gru(1, 3, 1, { inputToOutput: true });

const outputs = [0.1, 0.4, 0.2].map(
  (value) => network.activate([value])[0],
);

network.clear();

console.log(outputs);
```

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

Recommended starting knobs:

- start with one small block such as `Architect.lstm(input, 4, output)`
  before stacking multiple recurrent stages,
- supervised fine-tuning: keep `rate` low, usually around `0.001` to
  `0.01`, and feed one ordered sequence at a time,
- evolution: keep `seed` fixed for comparisons and prefer `NARX` first
  when a short explicit window already explains the task.

Common pitfalls:

- forgetting that `inputToOutput` defaults to `true`,
- flattening unrelated episodes into one activation stream without calling
  `clear()`,
- shuffling timesteps inside a sequence and destroying the temporal story
  the recurrent block is supposed to learn.

Parameters:
- `layerArgs` - Layer sizes plus an optional trailing options object.

Returns: The constructed LSTM network.

Example:

```ts
const network = Architect.lstm(1, 4, 1, { inputToOutput: false });

const firstPass = [0.1, 0.4, 0.2].map(
  (value) => network.activate([value])[0],
);

network.clear();

const secondPass = [0.1, 0.4, 0.2].map(
  (value) => network.activate([value])[0],
);

console.log(firstPass, secondPass);
```

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

Recommended starting knobs:

- start with short shelves such as `previousInput` and `previousOutput`
  in the `1` to `3` range,
- prefer `narx()` before gated builders when the task is really "predict
  from a small rolling window",
- keep sequence order intact during fine-tuning or evaluation and reset the
  carried state between independent runs.

Common pitfalls:

- forgetting that the delay lines intentionally carry state until
  `clear()` is called,
- shuffling timesteps from different sequences together,
- using long memory shelves when a shorter explicit window would be easier
  to debug, visualize, and evolve.

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
const firstSequence = [[0.2], [0.7], [0.4]];
const secondSequence = [[1], [0], [0]];

const firstOutputs = firstSequence.map((input) => network.activate(input)[0]);

network.clear();

const secondOutputs = secondSequence.map((input) => network.activate(input)[0]);

console.log(firstOutputs, secondOutputs);
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

Recommended starting knobs:

- gradient training: start with `iterations` in the low thousands, `rate`
  around `0.1` to `0.3`, and `momentum` around `0.9` for small normalized tasks,
- evolution: use this as the deterministic baseline profile with a fixed
  `seed`, `popsize` around `50` to `150`, and moderate `mutationRate`.

Common pitfalls:

- expecting a feed-forward graph to remember prior timesteps,
- keeping the exact same sample order every epoch on small i.i.d. datasets
  when some order randomization would reduce training bias.

Parameters:
- `layers` - Layer sizes starting with input, followed by hidden layers,
and ending with output.

Returns: The constructed MLP network.

Example:

```ts
const network = Architect.perceptron(2, 4, 1);

network.train(
  [
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [1] },
    { input: [1, 0], output: [1] },
    { input: [1, 1], output: [0] },
  ],
  {
    iterations: 3_000,
    rate: 0.3,
    momentum: 0.9,
  },
);

const output = network.activate([1, 0])[0];
```

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

Recommended starting knobs:

- topology search: keep `connections` only a little above the hidden-node
  count, then add `backConnections`, `selfConnections`, and `gates`
  gradually as the task proves it benefits from richer recurrence,
- deterministic replay: set `seed` whenever you want to compare topology
  changes instead of random initializers,
- NEAT seeding: pair this preset with `popsize` around `100` and moderate
  `mutationRate` so evolution still has structural headroom left.

Common pitfalls:

- requesting more structural edits than the graph can satisfy,
- starting from a graph that is already so dense that topology search has
  little left to discover.

Parameters:
- `input` - The number of input nodes.
- `hidden` - The number of hidden nodes to add.
- `output` - The number of output nodes.
- `options` - Optional sparse-structure counts for forward connections,
back connections, self connections, gates, and an optional deterministic seed.

Returns: The constructed sparse random network.

Example:

```ts
const network = Architect.randomSparse(3, 6, 1, {
  connections: 10,
  backConnections: 1,
  selfConnections: 1,
  seed: 7,
});

const output = network.activate([0.2, -0.1, 0.8])[0];
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
