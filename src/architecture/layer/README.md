# architecture/layer

## architecture/layer/layer.utils.types.ts

### LayerActivationContext

Minimal state required to run layer activation helpers.

`dropout` is layer-level dropout probability, while `nodes` holds the
activation units that will be read/written during forward activation.

Example:

```ts
const activationContext: LayerActivationContext = {
  nodes: layer.nodes,
  dropout: 0.2,
};

// activateLayer(activationContext, values, true)
```

### LayerConnectionContext

Context bundle required by connection/disconnection orchestration.

It packages raw connection arrays, a layer type guard, and the active layer
references so helpers remain pure and testable.

Example:

```ts
const connectionContext: LayerConnectionContext = {
  connections: { in: [], out: [], self: [] },
  isLayer: (value): value is LayerLike =>
    !!value && typeof (value as LayerLike).input === 'function',
  layer: someLayerLike,
  nodes: someLayerLike.nodes,
  output: someLayerLike.output,
};
```

### LayerFactoryContext

Generic factory context used to create layers without circular imports.

`createLayer` builds the target instance, and `isLayer` enables structural
narrowing whenever helpers accept mixed layer/group values.

Example:

```ts
const factoryContext: LayerFactoryContext<MyLayer> = {
  createLayer: () => new MyLayer(),
  isLayer: (value): value is LayerLike =>
    !!value && typeof (value as LayerLike).input === 'function',
};
```

### LayerFactoryLayer

Public layer surface required by factory construction helpers.

Factories only depend on activation/input wiring and output/node containers.

Example:

```ts
// Factory builders create an object that has these members.
// (Concrete layer classes typically provide many more helpers.)
```

### LayerLike

Structural contract for "layer-like" objects used in utility wiring.

This type avoids direct class coupling while preserving the behaviors needed
by connection helpers (`input`, `nodes`, and `output`).

Example:

```ts
// A real layer class typically satisfies this shape.
const layerLike: LayerLike = {
  input: (from) => [],
  nodes: [],
  output: null,
};
```

### LayerPropagationContext

Minimal state required to run backpropagation helpers.

Helpers only need access to the node sequence to propagate in reverse order.

Example:

```ts
const propagationContext: LayerPropagationContext = { nodes: layer.nodes };

// propagateLayer(propagationContext, 0.3, 0.1, targets)
```

## architecture/layer/layer.utils.ts

### activateLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerActivationContext, values: number[] | undefined, training: boolean) => number[]`

Orchestrates layer activation behavior with a high-level flow.

This is the recommended entry point for forward activation when you already
have a `LayerActivationContext`. It handles:
1) Input validation
2) Layer-level dropout masking (when `training` is true)
3) Activation into a pooled buffer
4) Cloning into a stable array for the caller

Examples:

```ts
// Typical usage: activate without explicit per-node inputs.
const output = activateLayer({ nodes: layer.nodes, dropout: layer.dropout });

// Explicit inputs: one number per node.
const output2 = activateLayer(
  { nodes: layer.nodes, dropout: layer.dropout },
  [0.1, 0.2, 0.3],
  true,
);
```

Parameters:
- `context` - - The layer state needed for activation.
- `values` - - Optional activation values to set per node.
- `training` - - Whether to apply dropout masking for training.

Returns: A cloned array of activation values.

### clearLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerConnectionContext) => void`

Orchestrates clearing node activation state with a high-level flow.

Parameters:
- `context` - - The layer state needed to reset nodes.
Example:

```ts
clearLayer(layerConnectionContext);
```

### connectLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerConnectionContext, target: import("src/architecture/node").default | import("src/architecture/group").default | import("src/architecture/layer/layer.utils.types").LayerLike, method: unknown, weight: number | undefined) => import("src/architecture/connection").default[]`

Orchestrates layer connection behavior with a high-level flow.

This is a small wrapper around the focused helper in
`layer.connection.utils.ts`, kept here so the public layer API stays compact.

Example:

```ts
connectLayer(layerConnectionContext, nextLayerLike);
```

Parameters:
- `context` - - The layer state needed for connections.
- `target` - - The layer, group, or node to connect to.
- `method` - - Optional connection method override.
- `weight` - - Optional fixed weight to apply.

Returns: The created connection list.

### createAttentionLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerFactoryContext<TLayer>, size: number, heads: number) => TLayer`

Orchestrates attention layer creation with a high-level flow.

Parameters:
- `context` - - Factory helpers for constructing the layer instance.
- `size` - - Number of output nodes.
- `heads` - - Number of attention heads.

Returns: The configured layer instance.
Example:

```ts
const attention = createAttentionLayer(factoryContext, 8, 4);
```

### createBatchNormLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerFactoryContext<TLayer>, size: number) => TLayer`

Orchestrates batch normalization layer creation with a high-level flow.

Parameters:
- `context` - - Factory helpers for constructing the layer instance.
- `size` - - Number of nodes in the normalization layer.

Returns: The configured layer instance.
Example:

```ts
const batchNorm = createBatchNormLayer(factoryContext, 16);
```

### createConv1dLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerFactoryContext<TLayer>, size: number, kernelSize: number, stride: number, padding: number) => TLayer`

Orchestrates 1D convolution layer creation with a high-level flow.

Parameters:
- `context` - - Factory helpers for constructing the layer instance.
- `size` - - Number of output nodes.
- `kernelSize` - - Size of the convolution kernel.
- `stride` - - Stride of the convolution.
- `padding` - - Padding size for the convolution.

Returns: The configured layer instance.
Example:

```ts
const conv1d = createConv1dLayer(factoryContext, 8, 3);
```

### createDenseLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerFactoryContext<TLayer>, size: number) => TLayer`

Orchestrates dense layer creation with a high-level flow.

Parameters:
- `context` - - Factory helpers for constructing the layer instance.
- `size` - - Number of nodes in the dense layer.

Returns: The configured layer instance.
Example:

```ts
const dense = createDenseLayer(factoryContext, 8);
```

### createGruLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerFactoryContext<TLayer>, size: number) => TLayer`

Orchestrates GRU layer creation with a high-level flow.

Parameters:
- `context` - - Factory helpers for constructing the layer instance.
- `size` - - Number of units in the GRU layer.

Returns: The configured layer instance.
Example:

```ts
const gru = createGruLayer(factoryContext, 8);
```

### createLayerNormLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerFactoryContext<TLayer>, size: number) => TLayer`

Orchestrates layer normalization layer creation with a high-level flow.

Parameters:
- `context` - - Factory helpers for constructing the layer instance.
- `size` - - Number of nodes in the normalization layer.

Returns: The configured layer instance.
Example:

```ts
const layerNorm = createLayerNormLayer(factoryContext, 16);
```

### createLstmLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerFactoryContext<TLayer>, size: number) => TLayer`

Orchestrates LSTM layer creation with a high-level flow.

Parameters:
- `context` - - Factory helpers for constructing the layer instance.
- `size` - - Number of units in the LSTM layer.

Returns: The configured layer instance.
Example:

```ts
const lstm = createLstmLayer(factoryContext, 8);
```

### createMemoryLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerFactoryContext<TLayer>, size: number, memory: number) => TLayer`

Orchestrates Memory layer creation with a high-level flow.

Parameters:
- `context` - - Factory helpers for constructing the layer instance.
- `size` - - Number of nodes in each memory block.
- `memory` - - Number of time steps to remember.

Returns: The configured layer instance.
Example:

```ts
const memoryLayer = createMemoryLayer(factoryContext, 4, 3);
```

### disconnectLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerConnectionContext, target: import("src/architecture/node").default | import("src/architecture/group").default, twoSided: boolean) => void`

Orchestrates disconnection behavior with a high-level flow.

Example:

```ts
disconnectLayer(layerConnectionContext, someGroup, true);
```

Parameters:
- `context` - - The layer state needed for disconnecting.
- `target` - - The group or node to disconnect.
- `twoSided` - - Whether to remove reciprocal connections as well.

### gateLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerConnectionContext, connections: import("src/architecture/connection").default[], method: unknown) => void`

Orchestrates layer gating behavior with a high-level flow.

Example:

```ts
gateLayer(layerConnectionContext, someConnections, method);
```

Parameters:
- `context` - - The layer state needed for gating.
- `connections` - - The connections to gate.
- `method` - - The gating method.

### inputLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerConnectionContext, from: import("src/architecture/group").default | import("src/architecture/layer/layer.utils.types").LayerLike, method: unknown, weight: number | undefined) => import("src/architecture/connection").default[]`

Orchestrates layer input wiring with a high-level flow.

Example:

```ts
inputLayer(layerConnectionContext, previousLayerLike);
```

Parameters:
- `context` - - The layer state needed for input wiring.
- `from` - - The source layer or group.
- `method` - - Optional connection method override.
- `weight` - - Optional fixed weight to apply.

Returns: The created connection list.

### propagateLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerPropagationContext, rate: number, momentum: number, targets: number[] | undefined) => void`

Orchestrates layer backpropagation behavior with a high-level flow.

If `targets` is provided, it must be one value per node (output layer).
If omitted, propagation behaves like a hidden layer.

Examples:

```ts
// Hidden layer propagation.
propagateLayer({ nodes: layer.nodes }, 0.3, 0.1);

// Output layer propagation.
propagateLayer({ nodes: layer.nodes }, 0.3, 0.1, [1, 0, 0]);
```

Parameters:
- `context` - - The layer state needed for propagation.
- `rate` - - The learning rate for weight updates.
- `momentum` - - The momentum factor for smoothing updates.
- `targets` - - Optional target values for output layers.

## architecture/layer/layer.guard.utils.ts

### isGroup

`(candidate: unknown) => boolean`

Checks whether an unknown value is group-like.

This is a structural runtime guard used by layer helpers that must safely
operate on mixed node/group collections.

Parameters:
- `candidate` - - The value to inspect.

Returns: True when the value exposes group-like members.

Example:

```ts
if (isGroup(value)) {
value.set({ bias: 0 });
}
```

## architecture/layer/layer.activation.utils.ts

### acquireActivationOutput

`(nodeCount: number) => number[]`

Acquires a pooled output buffer sized for the current activation call.

Pooling avoids frequent temporary allocations in hot activation paths.

The returned array is owned by the pool. Treat it as **temporary**:
- Fill it.
- Clone it (if you need a stable output).
- Release it back to the pool.

Example (typical pattern):

```ts
const pooled = acquireActivationOutput(nodes.length);
fillActivationOutput(nodes, values, pooled);
const output = cloneActivationOutput(pooled);
releaseActivationOutput(pooled);
```

Parameters:
- `nodeCount` - - Number of nodes in the layer.

Returns: A pooled output array.

### applyLayerMask

`(nodeList: import("src/architecture/node").default[], mask: number) => void`

Applies one mask value to every node in the layer.

In this library, a node-level `mask` is used as a lightweight dropout control.
A mask of `0` effectively disables the node for the current activation step.

Example:

```ts
applyLayerMask(layer.nodes, 1);
```

Parameters:
- `nodeList` - - The layer nodes to update.
- `mask` - - The mask value to apply.

### assertActivationInputSize

`(nodeCount: number, inputValues: number[] | undefined) => void`

Ensures optional activation inputs align 1:1 with layer nodes.

This guard prevents silent index skew where one node might accidentally
reuse another node's value.

In practice, this enables two safe activation modes:
- **Implicit activation**: omit `inputValues` and let each node compute its
  activation from its inbound connections.
- **Explicit activation**: pass a `number[]` with exactly one value per node.

This function exists because a mismatched array length is almost always a
caller bug, and failing fast is easier to debug than producing subtly wrong
activations.

Throws when `inputValues.length !== nodeCount`.

Example:

```ts
assertActivationInputSize(3, [0.1, 0.2, 0.3]);
assertActivationInputSize(3, [0.1, 0.2]); // throws
```

Parameters:
- `nodeCount` - - Number of nodes in the layer.
- `inputValues` - - Optional activation values provided by the caller.

### cloneActivationOutput

`(output: number[]) => number[]`

Clones pooled output into a stable caller-owned array.

This is the "escape hatch" that turns a pooled scratch buffer into a normal
array you can safely return from APIs.

Example:

```ts
const stable = cloneActivationOutput(pooled);
```

Parameters:
- `output` - - The pooled output array to clone.

Returns: A cloned output array.

### fillActivationOutput

`(nodeList: import("src/architecture/node").default[], inputValues: number[] | undefined, output: number[]) => void`

Activates each node and writes outputs into the provided buffer.

When `inputValues` is provided, each node receives the corresponding input
value. Otherwise each node self-activates from incoming state.

This function is deliberately low-level: it does not allocate and it does not
return anything. That makes it ideal for hot paths where you want to reuse a
buffer (typically a pooled one).

Example:

```ts
fillActivationOutput(layer.nodes, [0.2, 0.4], pooled);
```

Parameters:
- `nodeList` - - Nodes to activate.
- `inputValues` - - Optional activation values for each node.
- `output` - - Output buffer to populate.

### releaseActivationOutput

`(output: number[]) => void`

Releases a pooled activation output buffer back to the pool.

Call this after cloning/consuming the buffer to keep memory reuse effective.

Important: do not keep using `output` after releasing it.

Parameters:
- `output` - - The pooled output array to release.

### resolveLayerMask

`(layerDropout: number, isTraining: boolean) => number`

Resolves a shared dropout mask for the full layer.

In this layer-level dropout model, all nodes receive the same mask per call,
which keeps activation behavior synchronized for grouped layer semantics.

Notes:
- Dropout is only applied when `isTraining` is true.
- A return value of `1` means "keep" and `0` means "drop".
- This helper intentionally does **not** rescale activations (some dropout
  implementations divide by $(1 - p)$ during training). In this codebase the
  mask is a simple on/off switch.

Example:

```ts
resolveLayerMask(0.5, false); // => 1 (dropout disabled)
resolveLayerMask(0.5, true); // => 0 or 1
```

Parameters:
- `layerDropout` - - The dropout rate configured for the layer.
- `isTraining` - - Whether the layer is running in training mode.

Returns: A mask value of 1 or 0 for all nodes in the layer.

## architecture/layer/layer.connection.utils.ts

### clearLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerConnectionContext) => void`

Clears activation state for all nodes in a layer.

Use this when you want to reset per-node transient state between runs
(especially helpful in recurrent networks that keep state across timesteps).

Example:

```ts
clearLayer(layerContext);
```

Parameters:
- `context` - - The layer state needed to reset nodes.

### connectLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerConnectionContext, target: import("src/architecture/node").default | import("src/architecture/group").default | import("src/architecture/layer/layer.utils.types").LayerLike, method: unknown, weight: number | undefined) => import("src/architecture/connection").default[]`

Connects a layer's output group to a target.

Conceptually, this is the "forward wiring" helper: it takes whatever this
layer exposes as its `output` group and connects it to another structure.

Target handling:
- **Layer-like target**: calls `target.input(layer, method, weight)` so the
  target can decide how to interpret the source.
- **Group/Node target**: calls `output.connect(target, method, weight)`.

Throws when `context.output` is `null` (some layer types may not expose an
output group).

Example:

```ts
connectLayer(layerAContext, layerBLike);
connectLayer(layerAContext, someGroup, methods.groupConnection.ALL_TO_ALL);
```

Parameters:
- `context` - - The layer state needed for connections.
- `target` - - The layer, group, or node to connect to.
- `method` - - Optional connection method override.
- `weight` - - Optional fixed weight to apply.

Returns: The created connection list.

### disconnectFromGroup

`(layerNodes: import("src/architecture/node").default[], targetGroup: import("src/architecture/group").default, layerConnections: { in: import("src/architecture/connection").default[]; out: import("src/architecture/connection").default[]; self: import("src/architecture/connection").default[]; }, removeTwoSided: boolean) => void`

Disconnects all layer nodes from a target group.

This is a "cartesian disconnect": every node in this layer is disconnected
from every node in the target group.

Parameters:
- `layerNodes` - - Nodes in the layer.
- `targetGroup` - - Group to disconnect from.
- `layerConnections` - - Connection tracking for the layer.
- `removeTwoSided` - - Whether to remove reciprocal connections as well.

### disconnectFromNode

`(layerNodes: import("src/architecture/node").default[], targetNode: import("src/architecture/node").default, layerConnections: { in: import("src/architecture/connection").default[]; out: import("src/architecture/connection").default[]; self: import("src/architecture/connection").default[]; }, removeTwoSided: boolean) => void`

Disconnects all layer nodes from a target node.

Parameters:
- `layerNodes` - - Nodes in the layer.
- `targetNode` - - Node to disconnect from.
- `layerConnections` - - Connection tracking for the layer.
- `removeTwoSided` - - Whether to remove reciprocal connections as well.

### disconnectLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerConnectionContext, target: import("src/architecture/node").default | import("src/architecture/group").default, twoSided: boolean) => void`

Disconnects nodes in this layer from a target group or node.

This iterates through this layer's nodes and calls `node.disconnect(...)`.
It also updates the layer's tracked `connections.in/out` arrays so they
remain consistent with the underlying node graph.

Example:

```ts
disconnectLayer(layerContext, someNode, false);
```

Parameters:
- `context` - - The layer state needed for disconnecting.
- `target` - - The group or node to disconnect.
- `twoSided` - - Whether to remove reciprocal connections as well.

### gateLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerConnectionContext, connections: import("src/architecture/connection").default[], method: unknown) => void`

Applies gating to the provided connections using the layer output group.

Gating lets a third party (here, this layer's output group) *modulate* a set
of connections. This is used in recurrent architectures (e.g. LSTM/GRU) to
implement gate-controlled flow.

Throws when `context.output` is `null`.

Example:

```ts
gateLayer(layerContext, connections, methods.gating.OUTPUT);
```

Parameters:
- `context` - - The layer state needed for gating.
- `connections` - - The connections to gate.
- `method` - - The gating method.

### inputLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerConnectionContext, from: import("src/architecture/group").default | import("src/architecture/layer/layer.utils.types").LayerLike, method: unknown, weight: number | undefined) => import("src/architecture/connection").default[]`

Connects a source group or layer to this layer's input target.

This is the inverse of `connectLayer(...)`: instead of using *this* layer as
the source, we treat this layer as the *target* and connect a provided
source into `context.output`.

If `from` is a layer-like object, its `output` group is used as the source.
If no `method` is provided, this defaults to `ALL_TO_ALL` wiring.

Throws when either the resolved source group or this layer's `context.output`
is `null`.

Example:

```ts
inputLayer(thisLayerContext, previousLayerLike);
inputLayer(thisLayerContext, someGroup, methods.groupConnection.ONE_TO_ONE);
```

Parameters:
- `context` - - The layer state needed for input wiring.
- `from` - - The source layer or group.
- `method` - - Optional connection method override.
- `weight` - - Optional fixed weight to apply.

Returns: The created connection list.

### removeIncomingConnection

`(layerConnections: { in: import("src/architecture/connection").default[]; out: import("src/architecture/connection").default[]; self: import("src/architecture/connection").default[]; }, sourceNode: import("src/architecture/node").default, targetNode: import("src/architecture/node").default) => void`

Removes an incoming connection from layer tracking.

This scans in reverse so we can `splice(...)` safely while iterating.

Parameters:
- `layerConnections` - - Connection tracking for the layer.
- `sourceNode` - - Source node for the connection.
- `targetNode` - - Target node for the connection.

### removeOutgoingConnection

`(layerConnections: { in: import("src/architecture/connection").default[]; out: import("src/architecture/connection").default[]; self: import("src/architecture/connection").default[]; }, sourceNode: import("src/architecture/node").default, targetNode: import("src/architecture/node").default) => void`

Removes an outgoing connection from layer tracking.

This scans in reverse so we can `splice(...)` safely while iterating.

Parameters:
- `layerConnections` - - Connection tracking for the layer.
- `sourceNode` - - Source node for the connection.
- `targetNode` - - Target node for the connection.

## architecture/layer/layer.propagation.utils.ts

### assertTargetInputSize

`(nodeCount: number, inputTargets: number[] | undefined) => void`

Ensures target values align with the node count.

In backpropagation, a `targets` array is only meaningful for layers that are
acting as an output layer (supervised training). Hidden layers typically
propagate without explicit targets.

Example:

```ts
assertTargetInputSize(2, [1, 0]);
```

Parameters:
- `nodeCount` - - Number of nodes in the layer.
- `inputTargets` - - Optional target values provided by the caller.

### propagateNodesInReverse

`(context: import("src/architecture/layer/layer.utils.types").LayerPropagationContext, rate: number, momentum: number, targets: number[] | undefined) => void`

Propagates errors through all nodes in reverse order.

Reverse iteration matches the historical ordering used by Neataptic-style
implementations, and can matter when a node's propagation uses state that is
mutated as you traverse.

Target handling:
- When `targets` is omitted, each node propagates based on its accumulated
  error from downstream connections (hidden layer behavior).
- When `targets` is provided, each node receives a corresponding target value
  (output layer behavior).

Examples:

```ts
// Hidden layer:
propagateNodesInReverse({ nodes }, 0.3, 0.1);

// Output layer:
propagateNodesInReverse({ nodes }, 0.3, 0.1, [1, 0, 0]);
```

Parameters:
- `context` - - The layer state needed for propagation.
- `rate` - - The learning rate for weight updates.
- `momentum` - - The momentum factor for smoothing updates.
- `targets` - - Optional target values for output layers.

## architecture/layer/layer.factory.core.utils.ts

### buildDenseLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerFactoryContext<TLayer>, size: number) => TLayer`

Builds a standard dense (fully connected) layer.

The produced layer exposes its `Group` as `layer.output` and defines
`layer.input(...)` so callers can connect a source group or layer using
the selected connection strategy.

This helper is intentionally "low ceremony": it does not decide *where* the
layer is used in a network. It only creates nodes, creates the output group,
and provides an `input(...)` function so external code can wire it.

Parameters:
- `context` - - Factory helpers for constructing the layer instance.
- `size` - - Number of nodes to create in the dense layer.

Returns: The configured layer instance.

Example:

```ts
// Create a dense layer with 8 nodes.
const dense = buildDenseLayer(factoryContext, 8);

// Wire: previous -> dense (the dense layer decides how to accept inputs).
dense.input(previousLayerLike);
```

## architecture/layer/layer.factory.recurrent.utils.ts

### buildGruLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerFactoryContext<TLayer>, size: number) => TLayer`

Builds a GRU layer using the provided factory context.

Educational overview:
- GRU is a gated recurrent unit with fewer gates than LSTM.
- This implementation wires update/reset gates and a memory cell, then
exposes a standard `layer.output` group.

Parameters:
- `context` - - Factory helpers for constructing the layer instance.
- `size` - - Number of units in each GRU gate and cell.

Returns: The configured layer instance.

Example:

```ts
const gru = buildGruLayer(factoryContext, 8);
gru.input(previousLayerLike);
```

### buildLstmLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerFactoryContext<TLayer>, size: number) => TLayer`

Builds an LSTM layer using the provided factory context.

Educational overview:
- **Gates** control information flow (input / forget / output).
- The **memory cell** stores recurrent state via a self-connection.
- The **output block** is what this layer exposes as `layer.output`.

This builder wires the classic LSTM topology using `Group` blocks and gating.
It returns a layer object that is compatible with the rest of the layer
utilities (`layer.input(...)`, `layer.activate(...)`, `layer.output`, etc.).

Parameters:
- `context` - - Factory helpers for constructing the layer instance.
- `size` - - Number of units in each LSTM gate and cell.

Returns: The configured layer instance.

Example:

```ts
const lstm = buildLstmLayer(factoryContext, 8);

// Wire a previous layer (or Group) into the LSTM.
lstm.input(previousLayerLike);
```

### buildMemoryLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerFactoryContext<TLayer>, size: number, memory: number) => TLayer`

Builds a Memory layer using the provided factory context.

A memory layer is a simple way to provide a fixed window of past values.
Internally it creates `memory` blocks, links them one-to-one, and exposes a
flattened output group containing all block nodes.

Important: the input connector for a memory layer enforces **one-to-one**
wiring with unit weights to preserve the intended delay-line behavior.

Parameters:
- `context` - - Factory helpers for constructing the layer instance.
- `size` - - Number of nodes in each memory block.
- `memory` - - Number of time steps to remember.

Returns: The configured layer instance.

Example:

```ts
// A memory layer with 4 nodes per timestep, remembering 3 steps.
const memoryLayer = buildMemoryLayer(factoryContext, 4, 3);
memoryLayer.input(previousLayerLike);
```

### flattenConnections

`(connectionLists: import("src/architecture/connection").default[][]) => import("src/architecture/connection").default[]`

Flattens grouped connection arrays into a single list.

This keeps builder code declarative: build connections per gate/block,
then flatten once at the end.

Parameters:
- `connectionLists` - - Connection groups to flatten.

Returns: Flattened connection list.

Example:

```ts
const connections = flattenConnections([gateConnections, cellConnections]);
```

### resolveConnectionMethod

`(method: unknown) => unknown`

Resolves an optional connection method to a concrete method.

When users don't specify a method, we default to a dense-style
`ALL_TO_ALL` group connection.

Parameters:
- `method` - - Optional user-supplied connection method.

Returns: Connection method to apply.

Example:

```ts
const method = resolveConnectionMethod(undefined);
```

### resolveSourceGroup

`(factoryContext: import("src/architecture/layer/layer.utils.types").LayerFactoryContext<TLayer>, from: import("src/architecture/group").default | import("src/architecture/layer/layer.utils.types").LayerLike) => import("src/architecture/group").default`

Resolves a source group from a layer-like or group input.

Many wiring helpers accept either a `Group` or a "layer-like" object.
When a layer is provided, we treat `layer.output` as the actual source group.

Parameters:
- `factoryContext` - - Factory context providing layer guards.
- `from` - - Source input candidate.

Returns: Source group used for connections.

Example:

```ts
const sourceGroup = resolveSourceGroup(factoryContext, previousLayerLike);
```

## architecture/layer/layer.factory.experimental.utils.ts

### activateStubNodes

`(layer: TLayer) => number[]`

Activates all nodes in a stub layer and returns their outputs.

This helper keeps fallback activation behavior identical across experimental
layer variants.

Parameters:
- `layer` - - Layer containing nodes to activate.

Returns: Activated node outputs.

Example:

```ts
const outputs = activateStubNodes(layer);
```

### buildAttentionLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerFactoryContext<TLayer>, size: number, heads: number) => TLayer`

Builds a lightweight attention-style stub layer.

This placeholder stores head count metadata and uses a simple averaging
behavior for provided values. It is useful as an integration seam while
full attention internals are still under development.

Educational note: because this is a stub, "heads" are metadata only. The
activation behavior is intentionally simple: it collapses provided values to
their average.

Parameters:
- `context` - - Factory helpers for constructing the layer instance.
- `size` - - Number of output nodes.
- `heads` - - Number of attention heads.

Returns: The configured layer instance.

Example:

```ts
const attention = buildAttentionLayer(factoryContext, 8, 4);
const out = attention.activate([1, 2, 3, 4]);
// out is length 8, every entry is the average (2.5)
```

### buildConv1dLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerFactoryContext<TLayer>, size: number, kernelSize: number, stride: number, padding: number) => TLayer`

Builds a lightweight Conv1D-style stub layer.

This is an experimental placeholder: it stores Conv1D metadata and returns
either activated node outputs (no input values) or a bounded slice from
provided values. It does not perform real convolution math.

Educational note: this is designed as an integration seam. It lets you
prototype graphs that *mention* Conv1D without requiring a full convolution
implementation yet.

Parameters:
- `context` - - Factory helpers for constructing the layer instance.
- `size` - - Number of output nodes (filters).
- `kernelSize` - - Size of the convolution kernel.
- `stride` - - Stride of the convolution.
- `padding` - - Padding size for the convolution.

Returns: The configured layer instance.

Example:

```ts
const conv = buildConv1dLayer(factoryContext, 8, 3, 1, 1);

// When called with values, the stub returns a slice (not a true convolution).
const out = conv.activate([10, 11, 12, 13]);
```

### createAttentionActivator

`(layer: TLayer, size: number) => (values?: number[] | undefined) => number[]`

Builds the activation function used by the attention stub.

When values are provided, all outputs are filled with their average.
When values are omitted, it delegates to node activation.

This behavior is *not* meant to represent real attention math; it simply
produces a stable, shape-correct output while attention internals evolve.

Parameters:
- `layer` - - Layer whose nodes can self-activate.
- `size` - - Number of output values to return.

Returns: Activation callback for attention behavior.

Example:

```ts
const activate = createAttentionActivator(layer, 4);
activate([1, 3]); // -> [2, 2, 2, 2]
```

### createConv1dActivator

`(layer: TLayer, size: number) => (values?: number[] | undefined) => number[]`

Builds the activation function used by the Conv1D stub.

When values are provided, this function returns a length-bounded slice.
When values are omitted, it delegates to node activation.

This keeps the call signature compatible with real layers while remaining
intentionally cheap.

Parameters:
- `layer` - - Layer whose nodes can self-activate.
- `size` - - Number of output values to return.

Returns: Activation callback for Conv1D behavior.

Example:

```ts
const activate = createConv1dActivator(layer, 3);
activate([9, 8, 7, 6]); // -> [9, 8, 7]
```

### createStubLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerFactoryContext<TLayer>, size: number) => TLayer`

Creates shared node/output scaffolding for experimental layers.

Centralizing this setup keeps the experimental builders focused on their
metadata and activation behavior.

Implementation detail: the returned layer contains a `nodes` list (for basic
activation) and an `output` group (for API compatibility with the rest of the
architecture). In these stubs, the output group is not intended to be a fully
wired projection of `nodes`.

Parameters:
- `context` - - Factory helpers for constructing the layer instance.
- `size` - - Number of output nodes to allocate.

Returns: Initialized experimental layer.

## architecture/layer/layer.factory.normalization.utils.ts

### applyNormalizationActivation

`(layer: TLayer) => void`

Wraps a layer activation function with normalization post-processing.

The wrapper preserves existing activation semantics, then applies
`normalizeActivations(...)` to produce zero-centered, variance-scaled output.

This is implemented as a function wrapper rather than modifying node math.
That makes it easy to layer normalization behavior onto any dense layer.

Parameters:
- `layer` - - Dense layer to decorate with normalization behavior.

Returns: No return value.

Example (conceptual flow):

```ts
// 1) baseActivate(...) computes raw activations
// 2) wrapper normalizes the vector before returning
```

### buildBatchNormLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerFactoryContext<TLayer>, size: number) => TLayer`

Builds a dense layer decorated with batch-style normalization.

This helper keeps dense connectivity but post-processes activations so
each activation vector is centered and scaled using mean/variance.

Educational intuition:
- Centering subtracts the mean so the vector has average ~0.
- Scaling divides by standard deviation so typical magnitude is ~1.

In this implementation, the normalization is applied to the activation vector
produced *per call*.

Parameters:
- `context` - - Factory helpers for constructing the layer instance.
- `size` - - Number of nodes in the normalization layer.

Returns: The configured layer instance.

Example:

```ts
const normalized = buildBatchNormLayer(factoryContext, 16);
```

### buildLayerNormLayer

`(context: import("src/architecture/layer/layer.utils.types").LayerFactoryContext<TLayer>, size: number) => TLayer`

Builds a dense layer decorated with layer-style normalization.

This helper mirrors the batch variant in this implementation, applying
normalization to the activation vector produced for the current call.

Note: in many frameworks "batch norm" and "layer norm" differ in how they
compute statistics. Here they share the same post-processing to keep the code
simple and educational.

Parameters:
- `context` - - Factory helpers for constructing the layer instance.
- `size` - - Number of nodes in the normalization layer.

Returns: The configured layer instance.

Example:

```ts
const normalized = buildLayerNormLayer(factoryContext, 16);
```

### computeMean

`(activations: number[]) => number`

Computes the arithmetic mean for a vector of activations.

Parameters:
- `activations` - - Activation values to summarize.

Returns: Mean activation value.

Example:

```ts
computeMean([1, 2, 3]); // -> 2
```

### computeVariance

`(activations: number[], mean: number) => number`

Computes activation variance relative to a known mean.

Variance is computed as the average squared deviation from the mean.

Educational note: the standard deviation is `Math.sqrt(variance)`.

Parameters:
- `activations` - - Activation values to summarize.
- `mean` - - Mean value used for centering.

Returns: Variance of the activation values.

Example:

```ts
const mean = computeMean([1, 2, 3]);
computeVariance([1, 2, 3], mean); // -> 2/3
```

### normalizeActivations

`(activations: number[], mean: number, variance: number) => number[]`

Normalizes activation values using mean and variance.

A small epsilon (`NORM_EPSILON`) is added for numerical stability so
division remains well-defined when variance is very small.

The transformation is applied per element:
$(x - \mu) / \sqrt{\sigma^2 + \epsilon}$

Parameters:
- `activations` - - Activation values to normalize.
- `mean` - - Mean activation value.
- `variance` - - Variance activation value.

Returns: Normalized activation values.

Example:

```ts
const mean = computeMean([1, 2, 3]);
const variance = computeVariance([1, 2, 3], mean);
const normalized = normalizeActivations([1, 2, 3], mean, variance);
```
