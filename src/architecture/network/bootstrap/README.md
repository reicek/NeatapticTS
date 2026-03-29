# architecture/network/bootstrap

Raised when constructor topology intent conflicts with legacy acyclic flags.

## architecture/network/bootstrap/network.bootstrap.errors.ts

### NetworkBootstrapTopologyIntentConflictError

Raised when constructor topology intent conflicts with legacy acyclic flags.

## architecture/network/bootstrap/network.bootstrap.utils.ts

Bootstrap helpers for the public `Network` class.

This chapter owns the work that happens exactly once during construction:
resolve the caller's topology contract, initialize runtime flags, prewarm
pooled activation storage, seed deterministic randomness when requested, and
materialize the first input-to-output graph.

Keeping that one-time setup here lets `network.ts` stay focused on the
long-lived runtime surface while this file teaches the difference between
constructor policy and the ongoing activation or training lifecycle.

### applySeedOption

```ts
applySeedOption(
  network: NetworkBootstrapInternals,
  options: NetworkConstructorOptions | undefined,
): void
```

Applies the optional deterministic seed from constructor options.

Parameters:
- `network` - Network instance being constructed.
- `options` - Optional constructor options.

Returns: Nothing.

### bootstrapNetwork

```ts
bootstrapNetwork(
  network: NetworkBootstrapInternals,
  bootstrapContext: NetworkBootstrapContext,
): void
```

Performs one-time `Network` construction setup.

This orchestration keeps the constructor readable by separating four concerns:
public topology-policy resolution, runtime flag initialization, pooled memory
warmup, and synthesis of the initial fully connected IO graph.

Parameters:
- `network` - Network instance being constructed.
- `bootstrapContext` - Constructor inputs and resolved topology policy.

Returns: Nothing.

### connectInitialInputToOutputGraph

```ts
connectInitialInputToOutputGraph(
  network: NetworkBootstrapInternals,
): void
```

Connects every input node to every output node to form the starter graph.

The initial weight scaling mirrors the existing Network constructor behavior
so this split remains a pure structural refactor.

Parameters:
- `network` - Network instance being constructed.

Returns: Nothing.

### ensureMinimumHiddenNodes

```ts
ensureMinimumHiddenNodes(
  network: NetworkBootstrapInternals,
  minimumHiddenNodes: number,
): void
```

Ensures the network reaches the caller's requested minimum hidden width.

This relies on the public node-split mutation flow so the constructor and the
evolutionary runtime keep growing hidden structure the same way.

Parameters:
- `network` - Network instance being constructed.
- `minimumHiddenNodes` - Requested minimum hidden-node count.

Returns: Nothing.

### initializeIONodes

```ts
initializeIONodes(
  network: NetworkBootstrapInternals,
): void
```

Creates the initial input and output nodes for a new network.

When node pooling is enabled, each acquired node is reset before use so the
freshly constructed graph still starts from deterministic runtime state.

Parameters:
- `network` - Network instance being constructed.

Returns: Nothing.

### initializeRuntimeState

```ts
initializeRuntimeState(
  network: NetworkBootstrapInternals,
  input: number,
  output: number,
  options: NetworkConstructorOptions | undefined,
  topologyIntent: NetworkTopologyIntent,
  enforceAcyclic: boolean,
): void
```

Initializes core runtime state for a new network instance.

Parameters:
- `network` - Network instance being constructed.
- `input` - Number of input nodes.
- `output` - Number of output nodes.
- `options` - Optional constructor options.
- `topologyIntent` - Resolved public topology intent.
- `enforceAcyclic` - Resolved low-level acyclic policy.

Returns: Nothing.

### prewarmActivationPool

```ts
prewarmActivationPool(
  output: number,
): void
```

Prewarms pooled activation buffers for the initial output width.

Pool configuration errors remain non-fatal because allocation policy is an
optimization layer, not a correctness requirement for constructing a graph.

Parameters:
- `output` - Number of output nodes.

Returns: Nothing.

### resolveAcyclicEnforcement

```ts
resolveAcyclicEnforcement(
  options: NetworkConstructorOptions | undefined,
  topologyIntent: NetworkTopologyIntent,
): boolean
```

Resolves whether acyclic enforcement should be enabled for one constructor call.

The semantic topology intent remains the source of truth unless the caller
explicitly opted into the legacy boolean toggle.

Parameters:
- `options` - Optional constructor options.
- `topologyIntent` - Resolved public topology intent.

Returns: True when acyclic enforcement should be enabled.

### resolveTopologyIntent

```ts
resolveTopologyIntent(
  options: NetworkConstructorOptions | undefined,
): NetworkTopologyIntent
```

Resolves the public topology intent for one constructor call.

Prefer this semantic contract over the legacy low-level acyclic flag when
both are available.

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

This protects callers from creating a constructor packet that says
"feed-forward" in one field and "cyclic is allowed" in another.

Parameters:
- `options` - Optional constructor options.

Returns: Nothing.
