# architecture/network/training

Training pipeline utilities (migrated from legacy architecture/network.train.ts).

Provides:
 - Gradient clipping (global / layerwise; norm / percentile variants).
 - Mini & micro-batch gradient accumulation.
 - Optimizer step dispatch (SGD + adaptive optimizers + lookahead wrapper).
 - Simple mixed precision dynamic loss scaling (overflow detection heuristic).
 - Multiple moving-average smoothing strategies for error monitoring (SMA, EMA, adaptive EMA,
   median, gaussian, trimmed mean, WMA) plus separate plateau averaging.
 - Early stopping, schedule hooks, pruning hooks, and checkpoint callbacks.

Notes:
 - This module intentionally keeps imperative style for clarity/perf (avoids heap churn in hot loops).
 - Refactor changes here are documentation & naming only; numerical behavior preserved.

## architecture/network/training/network.training.utils.ts

### __trainingInternals

Test-only internal helper bundle.

This is exported so unit tests can cover edge-cases in the smoothing logic without
running full end-to-end training loops.

Important: this is **not** considered stable public API. It may change between releases.

### applyGradientClippingImpl

```ts
applyGradientClippingImpl(
  net: default,
  cfg: GradientClipRuntimeConfig,
): void
```

Apply gradient clipping to a network using a normalized runtime configuration.

This is a small wrapper that forwards to the concrete implementation used by training.

Parameters:
- `net` - Network instance to update.
- `cfg` - Normalized clipping settings.

### CheckpointConfig

Checkpoint callback configuration.

Training can periodically call `save(...)` with a serialized network snapshot.
You can persist these snapshots to disk, upload them, or keep them in-memory.

### clearState

```ts
clearState(): void
```

Clear all node runtime traces and states.

Parameters:
- `this` - Bound network instance.

### CostFunction

```ts
CostFunction(
  target: number[],
  output: number[],
): number
```

Cost / loss function used during supervised training.

A cost function compares an expected `target` vector with the network's produced `output`
vector, returning a scalar error where **lower is better**.

Design notes:
- This is called frequently (often once per training sample), so implementations should be
  **pure** and **allocation-light**.
- Most built-in training loops assume the returned value is non-negative.

Example (mean squared error):

```ts
export const mse: CostFunction = (target, output) => {
  const sum = target.reduce((acc, targetValue, index) => {
    const diff = targetValue - (output[index] ?? 0);
    return acc + diff * diff;
  }, 0);
  return sum / Math.max(1, target.length);
};
```

### GradientClipConfig

Gradient clipping configuration.

Clipping prevents rare large gradients from causing unstable weight updates.
It is most useful for recurrent networks and noisy datasets.

Conceptual modes:
- `norm`: clip by a global $L_2$ norm threshold.
- `percentile`: clip using a running percentile estimate (robust to outliers).
- `layerwise*`: apply the same idea per-layer (useful when layers have very different scales).

### MetricsHook

```ts
MetricsHook(
  m: { iteration: number; error: number; plateauError?: number | undefined; gradNorm: number; },
): void
```

Metrics hook signature.

If provided, this callback receives summarized metrics after each iteration.
It is designed for lightweight telemetry, not heavy data export.

### MixedPrecisionConfig

Mixed-precision configuration.

Mixed precision can improve throughput by running some math in lower precision while
keeping a stable FP32 master copy of parameters when needed.

### MixedPrecisionDynamicConfig

Dynamic mixed-precision configuration.

When enabled, training uses a loss-scaling heuristic that attempts to keep gradients
in a numerically stable range. Overflow pressure scales the loss down, while
persistent tiny gradients can scale it back up.

### MovingAverageType

Moving-average strategy identifier.

These strategies are used to smooth the monitored error curve during training.
Smoothing can make early stopping and progress logging less noisy.

### OptimizerConfigBase

Base optimizer configuration.

Training accepts either an optimizer name (`"adam"`, `"sgd"`, ...) or an object.
This object form is useful when you want to pin numeric hyperparameters or wrap a base
optimizer (e.g. lookahead).

Example:

```ts
net.train(set, {
  iterations: 1_000,
  rate: 0.001,
  optimizer: { type: 'adamw', beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.01 },
});
```

Notes:
- Exact supported `type` values are validated by training utilities.
- Unspecified fields fall back to sensible defaults per optimizer.

### propagate

```ts
propagate(
  rate: number,
  momentum: number,
  update: boolean,
  target: number[],
  regularization: number,
  costDerivative: CostDerivative | undefined,
): void
```

Contract for propagate.

### ScheduleConfig

Schedule callback configuration.

A schedule callback is a simple "tick hook" that runs every N iterations.
Typical uses include logging, custom learning-rate schedules, or diagnostics.

### SerializedNetwork

Serialized network payload used in checkpoint callbacks.

This is intentionally loose: serialization formats evolve and may include nested
structures. Treat this as an opaque snapshot blob.

### trainImpl

```ts
trainImpl(
  net: default,
  set: TrainingSample[],
  options: TrainingOptions,
): { error: number; iterations: number; time: number; }
```

Contract for trainImpl.

### TrainingOptions

Public training options accepted by the high-level training orchestration.

Training in this codebase is conceptually:
1) forward activation
2) backward propagation
3) optimizer update
repeated until a stopping condition is met.

Minimal example:

```ts
net.train(set, {
  iterations: 500,
  rate: 0.3,
  batchSize: 16,
  gradientClip: { mode: 'norm', maxNorm: 1 },
});
```

Stopping conditions:
- Provide at least one of `iterations` or `error`.
- `earlyStopPatience` adds an additional "stop when no improvement" guard.

### trainSetImpl

```ts
trainSetImpl(
  net: default,
  set: TrainingSample[],
  batchSize: number,
  accumulationSteps: number,
  currentRate: number,
  momentum: number,
  regularization: RegularizationConfig,
  costFunction: CostFunction | CostFunctionOrObject,
  optimizer: OptimizerConfigBase | undefined,
): number
```

Execute one full pass over dataset (epoch) with optional accumulation & adaptive optimizer.
Returns mean cost across processed samples.

This is the core "one epoch" primitive used by higher-level training orchestration.

Parameters:
- `net` - Network instance receiving training updates.
- `set` - Training samples.
- `batchSize` - Mini-batch size (use 1 for pure SGD).
- `accumulationSteps` - Micro-batch accumulation steps.
- `currentRate` - Current learning rate (may be scheduled by caller).
- `momentum` - Momentum used by some optimizers (when applicable).
- `regularization` - Regularization configuration passed down to nodes.
- `costFunction` - Cost function selector (function or compatible object).
- `optimizer` - Optional optimizer configuration.

Returns: Mean cost across the processed samples.

## architecture/network/training/network.training.utils.types.ts

### ALLOWED_OPTIMIZERS

Allow-list of optimizer identifiers accepted by training options before
optimizer-specific runtime state is initialized.

### buildMonitoredSmoothingConfig

```ts
buildMonitoredSmoothingConfig(
  type: MovingAverageType,
  window: number,
  emaAlpha: number | undefined,
  trimmedRatio: number | undefined,
): MonitoredSmoothingConfig
```

Build monitored smoothing configuration from options and defaults.

This keeps call sites declarative by normalizing all monitored-smoothing
fields into one explicit configuration object.

Parameters:
- `type` - Selected monitored smoothing mode.
- `window` - Monitored smoothing window length.
- `emaAlpha` - Optional monitored EMA alpha.
- `trimmedRatio` - Optional trimmed-mean ratio.

Returns: Normalized monitored smoothing configuration.

### CostDerivative

```ts
CostDerivative(
  target: number,
  output: number,
): number
```

Derivative callback used by output-node backpropagation.

Inputs are `(target, output)` so custom objectives can match the built-in
training loop without changing node internals.

### GradientClipRuntimeConfig

Normalized runtime gradient clipping configuration.

Optional fields are mode-dependent (`maxNorm` for norm modes, `percentile`
for percentile modes).

### NetworkNode

Node instance type used by training helpers.

This alias keeps helper signatures short while preserving the exact node
contract exposed by the owning `Network` instance.

### OutputNodeWithCostDerivative

Output-node contract for propagation paths that provide a custom derivative.

### PropagationContext

Immutable context shared by propagation helpers.

Keeping these values in a single object avoids argument drift across helper
boundaries and keeps orchestration code declarative.

### RegularizationArgument

Regularization payload accepted by `Node.propagate`.

Helpers pass this through unchanged so callers can centralize L1/L2
configuration at the training entrypoint.

### resolveEmaAlpha

```ts
resolveEmaAlpha(
  smoothingWindow: number,
  explicitAlpha: number | undefined,
): number
```

Resolve default EMA alpha using a window length.

When the caller omits a valid explicit alpha, this helper applies the
standard EMA conversion `2 / (window + 1)`.

Parameters:
- `smoothingWindow` - Window length for moving average operations.
- `explicitAlpha` - Optional user-provided alpha override.

Returns: A valid EMA alpha in the range (0, 1].

### TrainingSample

Input/output pair consumed by dataset training loops for one supervision
step during iterative optimization.

## architecture/network/training/network.training.finalize.utils.ts

### trainFinalizeCore

```ts
trainFinalizeCore(
  net: default,
  set: { input: number[]; output: number[]; }[],
  options: TrainingOptions,
): { error: number; iterations: number; time: number; }
```

Run the full training orchestration loop with smoothing, callbacks, and early stopping.

Parameters:
- `net` - Network instance to train.
- `set` - Training dataset.
- `options` - Training options.

Returns: Final training summary including error, iteration count, and elapsed time.

## architecture/network/training/network.training.backprop.utils.ts

Propagate output and hidden error signals backward through the network graph.

### clearNodeState

```ts
clearNodeState(
  node: default,
): void
```

Clear runtime state for a single node.

Parameters:
- `node` - Node to clear.

### clearState

```ts
clearState(): void
```

Clear all node runtime traces and states.

Parameters:
- `this` - Bound network instance.

### createPropagationContext

```ts
createPropagationContext(
  network: default,
  rate: number,
  momentum: number,
  update: boolean,
  regularization: number | { type: "L1" | "L2"; lambda: number; } | ((weight: number) => number) | undefined,
  costDerivative: CostDerivative | undefined,
): PropagationContext
```

Build the shared propagation context consumed by layer helpers.

Parameters:
- `network` - Network instance receiving backpropagation.
- `rate` - Learning rate.
- `momentum` - Momentum factor.
- `update` - Whether updates are applied immediately.
- `regularization` - Regularization setting used by node propagation.
- `costDerivative` - Optional cost-derivative override for output nodes.

Returns: Immutable context consumed by propagation helpers.

### getLastNodeIndex

```ts
getLastNodeIndex(
  network: default,
): number
```

Resolve the last node index in the network.

Parameters:
- `network` - Network instance.

Returns: Last valid node index.

### getOutputLayerStartIndex

```ts
getOutputLayerStartIndex(
  network: default,
): number
```

Resolve the first index of the output layer.

Parameters:
- `network` - Network instance.

Returns: Index at which output nodes begin.

### propagate

```ts
propagate(
  rate: number,
  momentum: number,
  update: boolean,
  target: number[],
  regularization: number,
  costDerivative: CostDerivative | undefined,
): void
```

Contract for propagate.

### propagateHiddenLayer

```ts
propagateHiddenLayer(
  context: PropagationContext,
): void
```

Propagate all hidden nodes in reverse topological order.

Parameters:
- `context` - Shared propagation context.

### propagateOutputLayer

```ts
propagateOutputLayer(
  context: PropagationContext,
  target: number[],
): void
```

Propagate all output nodes with explicit targets.

Parameters:
- `context` - Shared propagation context.
- `target` - Output target vector.

### propagateOutputNodeWithCostDerivative

```ts
propagateOutputNodeWithCostDerivative(
  node: default,
  context: PropagationContext,
  targetValue: number,
  costDerivative: CostDerivative,
): void
```

Propagate one output node using a custom cost derivative override.

Parameters:
- `node` - Output node to propagate.
- `context` - Shared propagation context.
- `targetValue` - Expected output value for this node.
- `costDerivative` - Cost derivative callback.

### propagateSingleHiddenNode

```ts
propagateSingleHiddenNode(
  context: PropagationContext,
  node: default,
): void
```

Propagate a single hidden node without a target value.

Parameters:
- `context` - Shared propagation context.
- `node` - Hidden node to propagate.

### propagateSingleOutputNode

```ts
propagateSingleOutputNode(
  context: PropagationContext,
  node: default,
  targetValue: number,
): void
```

Propagate a single output node with a target value.

Parameters:
- `context` - Shared propagation context.
- `node` - Output node to propagate.
- `targetValue` - Expected output value for this node.

### validateTargetLength

```ts
validateTargetLength(
  network: default,
  target: number[],
): void
```

Validate that target output count matches the network output width.

Parameters:
- `network` - Network instance receiving backpropagation.
- `target` - Output target vector.

## architecture/network/training/network.training.loop.utils.ts

Execute one dataset pass with mini-batching, accumulation, clipping, and optimizer updates.

### trainSetCore

```ts
trainSetCore(
  net: default,
  set: TrainingSample[],
  batchSize: number,
  accumulationSteps: number,
  currentRate: number,
  momentum: number,
  regularization: RegularizationConfig,
  costFunction: CostFunction | CostFunctionOrObject,
  optimizer: OptimizerConfigBase | undefined,
): number
```

Contract for trainSetCore.

## architecture/network/training/network.training.smoothing.utils.ts

### computeMonitoredError

```ts
computeMonitoredError(
  trainError: number,
  recentErrors: number[],
  cfg: MonitoredSmoothingConfig,
  state: PrimarySmoothingState,
): number
```

Compute monitored training error using the configured smoothing strategy.

The helper returns the raw error when smoothing is effectively disabled.
For stateful modes (`ema`, `adaptive-ema`), the provided state object is
updated in place so callers can keep continuity across iterations.

Parameters:
- `trainError` - Raw training error for the current iteration.
- `recentErrors` - Chronological recent error window (oldest to newest).
- `cfg` - Monitored smoothing configuration.
- `state` - Mutable smoothing state for EMA-based modes.

Returns: Smoothed monitored error.

### computePlateauMetric

```ts
computePlateauMetric(
  trainError: number,
  plateauErrors: number[],
  cfg: PlateauSmoothingConfig,
  state: PlateauSmoothingState,
): number
```

Compute plateau metric using the configured plateau smoothing strategy.

This metric is intentionally independent from the primary monitored metric so
plateau detection can use a different noise profile.

Parameters:
- `trainError` - Raw training error for the current iteration.
- `plateauErrors` - Plateau window of recent raw errors.
- `cfg` - Plateau smoothing configuration.
- `state` - Mutable state for plateau EMA.

Returns: Smoothed plateau metric.

## architecture/network/training/network.training.gradient-clip.utils.ts

### applyGradientClippingCore

```ts
applyGradientClippingCore(
  net: default,
  cfg: GradientClipRuntimeConfig,
): void
```

Apply gradient clipping to accumulated connection and bias delta buffers.

Parameters:
- `net` - Network instance whose accumulated gradients are clipped.
- `cfg` - Runtime clipping configuration.

Returns: Nothing.

## architecture/network/training/network.training.errors.ts

Raised when the training dataset is missing or does not match network IO dimensions.

### NetworkTrainingAccumulationStepsError

Raised when accumulation steps is invalid.

### NetworkTrainingBatchSizeError

Raised when configured batch size exceeds dataset size.

### NetworkTrainingDatasetCompatibilityError

Raised when the training dataset is missing or does not match network IO dimensions.

### NetworkTrainingDropoutRangeError

Raised when dropout is outside the expected range [0, 1).

### NetworkTrainingInvalidCostFunctionError

Raised when the provided cost function is not callable or recognized.

### NetworkTrainingInvalidOptimizerOptionError

Raised when optimizer option type is not supported.

### NetworkTrainingNestedLookaheadError

Raised when lookahead is configured with a nested lookahead base type.

### NetworkTrainingOutputTargetLengthError

Raised when output target length does not match the network output width.

### NetworkTrainingStoppingConditionRequiredError

Raised when no stopping condition is provided to training.

### NetworkTrainingUnknownLookaheadBaseTypeError

Raised when lookahead base optimizer type is unknown.

### NetworkTrainingUnknownOptimizerTypeError

Raised when optimizer type is unknown.

## architecture/network/training/network.training.isolate.utils.ts

### FineTuneOptions

Explicit settings for one isolated fine-tune pass.

`steps` maps to the training loop iteration count and `learningRate` maps to
the training rate. `seed` is optional because same-runtime ordered
determinism only becomes a strong claim when the caller supplies both a
stable dataset order and an explicit deterministic seed.

Without an explicit `seed`, the helper still guarantees isolation: the
original network and vector are never mutated, but repeated calls may
produce different trained vectors when the underlying training loop contains
stochastic behaviour such as dropout.

### FineTuneResult

Detached result from one isolated fine-tune pass.

The helper returns a trained parameter vector rather than a mutated network
so shared candidate state stays outside the training-owned boundary. Metrics
are numeric summaries from the existing training loop, not persisted
optimizer or runtime state.

The returned `trainedVector` can be compared against the original vector,
forwarded to a worker for scoring, persisted as a checkpoint delta, or
discarded when only the fitness score matters.

### fineTuneVector

```ts
fineTuneVector(
  baseNetwork: default,
  vector: ParameterVector,
  dataset: TrainingSample[],
  options: FineTuneOptions,
): FineTuneResult
```

Fine-tune one parameter vector against an ordered dataset without mutating shared state.

The helper clones `baseNetwork`, applies `vector` to that working copy,
optionally installs an explicit deterministic seed via `Network.setSeed(...)`,
runs the existing training loop in the caller-provided dataset order, and
returns a new `ParameterVector` exported from the trained working copy. The
supplied `baseNetwork` and `vector` are read-only inputs to this helper.

Determinism is intentionally scoped. On the same runtime, repeated calls can
return the same trained vector when topology, dataset order, training
settings, and explicit `seed` all match. This helper does not claim
cross-runtime exact replay, and it does not return transient optimizer,
activation, or recurrent runtime state.

```ts
// Export the current parameter vector, fine-tune a working copy, and
// inspect fitness metrics without modifying the shared candidate network.
const vector = toParameterVector(candidate);
const { trainedVector, metrics } = fineTuneVector(candidate, vector, dataset, {
  steps: 50,
  learningRate: 0.01,
  seed: 42,
});
console.log('training error:', metrics?.error);
// `candidate` and `vector` are unchanged after this call.
```

Parameters:
- `baseNetwork` - Topology source cloned for the isolated working copy.
- `vector` - Ordered parameter payload applied to the working copy only.
- `dataset` - Ordered training samples consumed without shuffling.
- `options` - Explicit training settings and optional deterministic seed.

Returns: Detached trained vector plus numeric training metrics.
