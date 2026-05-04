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
in a numerically stable range. If an overflow is detected, the scale is reduced.

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

Propagate output and hidden errors backward through the network.

Parameters:
- `this` - Bound network instance.
- `rate` - Learning rate.
- `momentum` - Momentum factor.
- `update` - Whether to apply updates immediately.
- `target` - Output target values.
- `regularization` - L2 regularization factor.
- `costDerivative` - Optional output-node derivative override.

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

High-level training orchestration with early stopping, smoothing & callbacks.

This is the main entrypoint used by `Network.train(...)`-style APIs.

Parameters:
- `net` - Network instance to train.
- `set` - Training dataset.
- `options` - Training options (stopping conditions, optimizer, hooks, etc.).

Returns: Summary payload containing final error, iteration count, and elapsed time.

Example:

```ts
const result = net.train(set, { iterations: 500, rate: 0.3 });
console.log(result.error);
```

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

Set of supported optimizer identifiers accepted by training options.

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

Cost-derivative callback shape for output-node backpropagation.

### GradientClipRuntimeConfig

Runtime gradient clipping configuration normalized from training options.

### NetworkNode

Local node shape alias used by training utility modules.

### OutputNodeWithCostDerivative

Extended output-node contract that supports custom cost derivatives.

### PropagationContext

Shared immutable context for network propagation helpers.

### RegularizationArgument

Regularization argument accepted by node-level propagation.

### resolveEmaAlpha

```ts
resolveEmaAlpha(
  smoothingWindow: number,
  explicitAlpha: number | undefined,
): number
```

Resolve default EMA alpha using a window length.

Parameters:
- `smoothingWindow` - Window length for moving average operations.
- `explicitAlpha` - Optional user-provided alpha override.

Returns: A valid EMA alpha in the range (0, 1].

### TrainingSample

Training sample consumed by training set loops.

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

Propagate output and hidden errors backward through the network.

Parameters:
- `this` - Bound network instance.
- `rate` - Learning rate.
- `momentum` - Momentum factor.
- `update` - Whether to apply updates immediately.
- `target` - Output target values.
- `regularization` - L2 regularization factor.
- `costDerivative` - Optional output-node derivative override.

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

Execute one dataset pass with mini-batching, accumulation, clipping, and optimizer updates.

Parameters:
- `net` - Network instance being trained.
- `set` - Training sample set.
- `batchSize` - Mini-batch size.
- `accumulationSteps` - Micro-batches per optimizer step.
- `currentRate` - Learning rate for this pass.
- `momentum` - Momentum value used by propagation paths.
- `regularization` - Regularization settings passed into propagation calls.
- `costFunction` - Cost function or cost-function object.
- `optimizer` - Optional optimizer configuration.

Returns: Mean cost over processed samples.

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

Apply gradient clipping to accumulated connection and bias deltas.

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
