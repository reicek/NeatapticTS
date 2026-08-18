# architecture/node

Core neuron chapter for the architecture surface.

This folder now owns the library's lowest-level unit of computation: the
mutable neuron object that stores bias, state, activation history, tracing
information, connection references, and optimizer scratch space.

Read this chapter in three passes:

1. start with the `Node` class overview to understand the long-lived state a
   neuron carries between activation, propagation, mutation, and
   serialization,
2. continue to the activation methods when you want the difference between
   traced and no-trace execution,
3. finish with connectivity, gating, and optimizer helpers when you need to
   understand how one node participates in larger graph changes.

Architecture building asks two different questions at this boundary:
what role does the neuron play at runtime, and what human-facing meaning
should later tooling remember about it? The runtime role lives in `type`
(`input`, `hidden`, `output`) and changes execution semantics. The optional
descriptor surface (`label`, `intent`, `metadata`) does not change
activation math; it keeps architecture intent visible for later diagnostics,
visualization, and construct-from-parts work.

Example:

```ts
const sensor = new Node('input');
const readout = new Node('output');

readout.describe({
  label: 'readoutNode',
  metadata: { stage: 'policy' },
});
```

## architecture/node/node.ts

### Node

Node (Neuron)
=============
Fundamental computational unit: aggregates weighted inputs, applies an activation
function (squash) and emits an activation value. Supports:
 - Types: 'input' | 'hidden' | 'output' (affects bias initialization & error handling)
 - Recurrent self‑connections & gated connections (for dynamic / RNN behavior)
 - Dropout mask (`mask`), momentum terms, eligibility & extended traces (for
   a variety of learning rules beyond simple backprop).
 - Optional descriptors via `describe({ label, intent, metadata })` when a
   low-level node should keep a stable human-facing identity inside a larger
   architecture story.

Educational note: Traces (`eligibility` and `xtrace`) illustrate how recurrent credit
assignment works in algorithms like RTRL / policy gradients. They are updated only when
using the traced activation path (`activate`) vs `noTraceActivate` (inference fast path).

Most architecture code should only attach a descriptor when a node boundary
matters outside the current function. That keeps the primitive cheap for raw
graph math while still letting later passes recover names such as
`readoutNode`, `memoryCell`, or `temperatureGate`.

Example:

```ts
const sensor = new Node('input');
const readout = new Node('output');

readout.describe({
  label: 'readoutNode',
  metadata: { stage: 'policy' },
});
```

### NodeOptimizerProps

Internal interface for accessing dynamic optimizer properties on Node instances.
These properties are lazily allocated and not part of the main class definition.

### PrimitiveDescriptor

Optional human-readable descriptor attached to one architecture primitive.

Descriptors are intentionally lightweight. They keep the public primitive
API teachable by separating runtime behavior from architectural meaning:
`type` still controls execution semantics, while `label`, `intent`, and
scalar metadata keep the boundary recognizable to later tooling.

Example:

```ts
const readout = new Node('output');

readout.describe({
  label: 'policyLogits',
  metadata: { stage: 'readout' },
});
```

### PrimitiveIntent

Semantic role hints that later architecture tooling can attach to individual primitives.

These labels do not change activation math — they let diagnostics, visualizers, and
builders identify what a node represents conceptually (e.g. `'gate'` for gating neurons,
`'memory'` for LSTM cell nodes, `'attention'` for attention-head units).

### PrimitiveMetadata

Lightweight key-value metadata bag attached to architecture primitives. Keys are free-form strings; values are restricted to serializable scalars.

### PrimitiveMetadataValue

Scalar metadata value types retained on architecture primitives. Kept narrow so metadata bags are safe to serialize and diff across checkpoints.

### PrimitiveNodeType

Runtime-supported primitive roles for architecture-building surfaces.

- `'input'`: receives the external stimulus vector; no bias computation.
- `'hidden'`: standard neuron with bias, activation function, and trace storage.
- `'output'`: emits the network result; error and gradient accumulation begin here during back-propagation.

### resolvePrimitiveIntent

```ts
resolvePrimitiveIntent(
  nodeType: string,
): PrimitiveIntent | null
```

Resolve a public primitive intent from a runtime node type string.

Returns the intent that matches the type when the type is one of the
three standard roles (`'input'`, `'hidden'`, `'output'`), or `null`
for any other string. Used by descriptor helpers to stamp a default
intent without requiring the caller to repeat the role mapping.

Parameters:
- `nodeType` - Runtime node type string to resolve.

Returns: The matching primitive intent or null for non-standard types.

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

#### _safeUpdateBias

```ts
_safeUpdateBias(
  delta: number,
): void
```

Internal helper to safely update the node bias with clipping and NaN checks.

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
Resets accumulators after applying. Safe to call on every node type.

Parameters:
- `momentum` - Momentum factor (0 to disable)

#### applyBatchUpdatesWithOptimizer

```ts
applyBatchUpdatesWithOptimizer(
  opts: BatchOptimizerOptions,
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

#### applyCtrnnActivation

```ts
applyCtrnnActivation(
  inputSum: number,
  dt: number,
): number
```

Applies CTRNN (Continuous Time Recurrent Neural Network) exponential Euler
integration to advance the node's internal state.

**Scaffolding:** This method is library scaffolding for a future CTRNN
activation path. It is not yet routed through the standard network
`activate` / `noTraceActivate` pipeline. The Neatenstein demo uses MLP
inference (not `Node` objects), so integration is deferred to a later
step that adds a recurrent activation mode to the network.

Updates `this.state` in place using the formula:
  `state += (inputSum - state) * (dt / timeConstant)`

Then applies the squash function to compute the new activation. The
`timeConstant` property controls temporal memory: larger values produce
slower, more inertial responses.

Parameters:
- `inputSum` - The weighted sum of inputs (including bias if desired).
- `dt` - The discrete time step for integration.

Returns: The node's activation after integration.

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

#### describe

```ts
describe(
  descriptor: PrimitiveDescriptor,
): void
```

Attaches optional descriptor metadata to the primitive boundary.

This descriptor is advisory only. It does not change runtime activation,
mutation, or serialization behavior, but it gives later architecture
assembly, diagnostics, and visualization passes a stable place to read
human-facing labels and intent.

Reach for this when the node is still the right abstraction but a later
reader should not have to infer its purpose from connection order alone.

Parameters:
- `descriptor` - Optional label, intent, and scalar metadata to merge.

Returns: Nothing.

Example:

```ts
const readout = new Node('output');
readout.describe({
  label: 'readoutNode',
  metadata: { stage: 'readout' },
});
```

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
  json: { bias: number; response?: number | undefined; timeConstant?: number | undefined; type: string; squash: string | null; mask: number; },
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

Stable per-node gene identifier for NEAT innovation reuse.

#### index

Optional index, potentially used to identify the node's position within a layer or network structure. Not used internally by the Node class itself.

#### intent

Optional semantic intent for architecture tooling and diagnostics.

#### isActivating

Internal flag to detect cycles during activation.

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

#### label

Optional human-readable descriptor label for architecture tooling.

#### mask

A mask factor (typically 0 or 1) used for implementing dropout. If 0, the node's output is effectively silenced.

#### metadata

Optional scalar metadata retained on the primitive boundary.

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

#### response

Response multiplier applied to the node state before the squash function.

A neutral response of `1` preserves the historical runtime behavior. Values
above or below `1` steepen or flatten the node's effective transfer curve
without changing the chosen activation family.

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

#### syncGeneIdCounter

```ts
syncGeneIdCounter(
  maxObservedGeneId: number,
): void
```

Advances the global gene-id cursor past a restored maximum.

Restore flows use this after hydrating persisted genomes so the next freshly
created node cannot collide with an older serialized `geneId`.

Parameters:
- `maxObservedGeneId` - Highest restored node gene id currently in memory.

Returns: Nothing.

#### timeConstant

Per-node evolvable time constant for CTRNN (Continuous Time Recurrent Neural Network) formulation.

Controls the speed of temporal integration during activation. A value of 1.0
gives fast (near-instant) response; larger values introduce slower temporal
memory. Mutated by the `MOD_TIME_CONSTANT` operator.

Defaults to `1.0`.

#### toJSON

```ts
toJSON(): { index: number | undefined; bias: number; response: number; timeConstant: number; type: string; squash: string | null; mask: number; }
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

## architecture/node/node.errors.ts

Raised when a node mutation call receives a null or undefined method.

### NodeInvalidConnectionTargetTypeError

Raised when a node connection target is neither a node nor a group-like object.

### NodeMutationMethodRequiredError

Raised when a node mutation call receives a null or undefined method.

### NodeUndefinedConnectionTargetError

Raised when a node connection target reference is missing or undefined.

### NodeUnknownMutationMethodError

Raised when a node mutation method name is not recognized.

### NodeUnsupportedMutationMethodError

Raised when a known mutation call reaches an unsupported mutation branch.
