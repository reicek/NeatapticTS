# architecture/group

Core group chapter for the architecture surface.

This folder owns the library's first composite graph primitive: a small
collection of nodes that can be activated, propagated, wired, gated, and
serialized as one architectural block.

Read this chapter in three passes:

1. start with the `Group` class overview to understand how the boundary
   wraps many node-level operations into one reusable building block,
2. continue to `connect()` and `gate()` when you need the structural
   vocabulary for wiring groups into larger graphs,
3. finish with `disconnect()`, `clear()`, and `toJSON()` when you want the
   lifecycle and persistence view for composite primitives.

Groups also carry the same two-level story as nodes: construction-time role
describes the runtime semantics of their allocated nodes, while
`describe({ label, intent, metadata })` lets later tooling remember why this
block exists without changing how activation or propagation works.

Example:

```ts
const sensorBlock = new Group(4, 'input');
const readoutBlock = new Group(2, 'output');

sensorBlock.describe({
  label: 'sensorBlock',
  metadata: { stage: 'encoder' },
});
readoutBlock.describe({
  label: 'readoutBlock',
  metadata: { stage: 'readout' },
});

sensorBlock.connect(
  readoutBlock,
  methods.groupConnection.ALL_TO_ALL,
);
```

## architecture/group/group.ts

### Group

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

The practical pattern is usually: allocate the group with the right runtime
role when the whole block is clearly input- or output-oriented, then add a
descriptor only when the boundary should stay visible in diagnostics or
later graph assembly.

Example:

```ts
const sensorBlock = new Group(4, 'input');
const readoutBlock = new Group(2, 'output');

sensorBlock.describe({
  label: 'sensorBlock',
  metadata: { stage: 'encoder' },
});
readoutBlock.describe({
  label: 'readoutBlock',
  metadata: { stage: 'readout' },
});

sensorBlock.connect(
  readoutBlock,
  methods.groupConnection.ALL_TO_ALL,
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
`in`: Connections coming into nodes in this group from outside.
`out`: Connections going out from nodes in this group to outside.
`self`: Connections between nodes within this same group.

#### describe

```ts
describe(
  descriptor: PrimitiveDescriptor,
): void
```

Attaches optional descriptor metadata to the group boundary.

Use this when a group represents a named stage, gate bundle, or other
meaningful architecture unit that later diagnostics should recognize
without inferring from node order alone.

Parameters:
- `descriptor` - Optional label, intent, and scalar metadata to merge.

Returns: Nothing.

Example:

```ts
const forgetGate = new Group(8);

forgetGate.describe({
  label: 'forgetGate',
  intent: 'gate',
  metadata: { family: 'lstm' },
});
```

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

#### intent

Optional semantic intent for architecture tooling and diagnostics.

#### label

Optional human-readable descriptor label for architecture tooling.

#### metadata

Optional scalar metadata retained on the primitive boundary.

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

## architecture/group/group.errors.ts

Raised when caller-provided group values do not match the number of nodes.

### GroupGatingMethodRequiredError

Raised when gating is requested without specifying a gating method.

### GroupOneToOneSizeMismatchError

Raised when ONE_TO_ONE group connections are requested for groups of different sizes.

### GroupSizeMismatchError

Raised when caller-provided group values do not match the number of nodes.
