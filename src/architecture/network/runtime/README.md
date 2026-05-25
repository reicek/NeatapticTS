# architecture/network/runtime

Runtime control utilities for advanced network inference features.

Provides:
 - Weight noise injection (global and per-layer).
 - DropConnect regularization during inference.
 - Stochastic depth (layer skipping) for layered networks.
 - Iterative magnitude-based weight pruning with configurable schedules.

## architecture/network/runtime/network.runtime.controls.utils.ts

### clearStochasticDepthSchedule

```ts
clearStochasticDepthSchedule(): void
```

Clear the stochastic-depth schedule function so runtime behavior reverts to the currently stored static survival probabilities without additional per-step schedule adjustments.

Parameters:
- `this` - Target network instance.

Returns: Nothing.

### clearWeightNoiseSchedule

```ts
clearWeightNoiseSchedule(): void
```

Clear the dynamic global weight-noise schedule so future steps stop applying schedule-driven standard-deviation updates and keep only explicit static configuration.

Parameters:
- `this` - Target network instance.

Returns: Nothing.

### configurePruning

```ts
configurePruning(
  configuration: PruningConfiguration,
): void
```

Configure scheduled pruning during training.

This stores the pruning window and target policy on the network so the
training loop can opportunistically apply structured sparsification later.

Parameters:
- `this` - Target network instance.
- `configuration` - Pruning schedule and ranking configuration.

Returns: Nothing.

### disableStochasticDepth

```ts
disableStochasticDepth(): void
```

Disable stochastic depth entirely so all hidden layers participate in every pass and no layer-skipping regularization is applied at runtime.

Parameters:
- `this` - Target network instance.

Returns: Nothing.

### disableWeightNoise

```ts
disableWeightNoise(): void
```

Disable all configured weight-noise mechanisms so subsequent training and inference passes execute without global or per-hidden-layer perturbation state, schedule updates, or hidden-layer noise carryover.

Parameters:
- `this` - Target network instance.

Returns: Nothing.

### enableWeightNoise

```ts
enableWeightNoise(
  configuration: WeightNoiseConfiguration,
): void
```

Enable weight noise using either one global standard deviation or per-hidden-layer values.

A single global value is useful for quick experiments, while the per-hidden
schedule keeps layered models explicit about which hidden stage receives how
much perturbation.

Parameters:
- `this` - Target network instance.
- `configuration` - Global standard deviation or one value per hidden layer.

Returns: Nothing.

### getLastSkippedLayers

```ts
getLastSkippedLayers(): number[]
```

Read the last hidden-layer indices skipped by stochastic depth so diagnostics can inspect which layers were bypassed in the most recent forward pass.

Parameters:
- `this` - Target network instance.

Returns: Snapshot of the last skipped hidden-layer indices.

### getRuntimeRegularizationStats

```ts
getRuntimeRegularizationStats(): Record<string, unknown> | null
```

Read regularization statistics collected during training so callers can inspect dropout, noise, and penalty telemetry without direct access to internal runtime fields.

Parameters:
- `this` - Target network instance.

Returns: Last regularization stats payload or `null` when none exists yet.

### getTrainingStep

```ts
getTrainingStep(): number
```

Read the current training-step counter so external schedulers, dashboards, and callback logic can align runtime control decisions with iteration progress.

Parameters:
- `this` - Target network instance.

Returns: Current training step.

### setRandom

```ts
setRandom(
  randomFunction: () => number,
): void
```

Replace the network random number generator.

This lets advanced callers share one deterministic source across mutation,
stochastic depth, DropConnect, and other runtime randomness.

Parameters:
- `this` - Target network instance.
- `randomFunction` - RNG function returning values in $[0,1)$.

Returns: Nothing.

### setStochasticDepth

```ts
setStochasticDepth(
  survivalProbabilities: number[],
): void
```

Configure stochastic depth with one survival probability per hidden layer.

Matching survival values to hidden layers keeps the runtime contract explicit
and avoids silently applying one layer's policy to another.

Parameters:
- `this` - Target network instance.
- `survivalProbabilities` - Survival probabilities for each hidden layer.

Returns: Nothing.

### setStochasticDepthSchedule

```ts
setStochasticDepthSchedule(
  schedule: StochasticDepthSchedule,
): void
```

Set the stochastic-depth schedule function that updates survival probabilities over time using the current training step and previous schedule state.

Parameters:
- `this` - Target network instance.
- `schedule` - Function mapping the current step and schedule to a new schedule.

Returns: Nothing.

### setWeightNoiseSchedule

```ts
setWeightNoiseSchedule(
  schedule: (step: number) => number,
): void
```

Set a dynamic scheduler for global weight noise so each training step can derive a new standard deviation from one explicit and testable policy function.

Parameters:
- `this` - Target network instance.
- `schedule` - Function mapping the current training step to a standard deviation.

Returns: Nothing.

### testForceOverflow

```ts
testForceOverflow(): void
```

Force the next mixed-precision overflow path.

This is a test-oriented hook used to exercise loss-scale recovery logic
without waiting for a real floating-point overflow.

Parameters:
- `this` - Target network instance.

Returns: Nothing.

## architecture/network/runtime/network.runtime.diagnostics.utils.ts

Runtime diagnostics and safety helpers for the public `Network` class.

This chapter owns the public readers and small runtime controls that expose
training-health state, activation-ordering diagnostics, DropConnect policy,
and dropout-mask cleanup without changing the network topology itself.

### appendSuggestion

```ts
appendSuggestion(
  suggestions: string[],
  suggestion: string,
): string[]
```

Append one suggestion string only when it is not already present.

Parameters:
- `suggestions` - Existing suggestions.
- `suggestion` - Suggested next action.

Returns: Updated suggestions list.

### cloneSchedulingDiagnostics

```ts
cloneSchedulingDiagnostics(
  diagnostics: ActivationSchedulingDiagnostics,
): ActivationSchedulingDiagnostics
```

Clone the diagnostics snapshot so callers cannot mutate runtime state.

Parameters:
- `diagnostics` - Diagnostics snapshot to clone.

Returns: Detached diagnostics snapshot.

### createCompiledSchedulingDiagnostics

```ts
createCompiledSchedulingDiagnostics(
  network: default,
  activationSchedule: ActivationSchedule,
): ActivationSchedulingDiagnostics
```

Build the standard diagnostics snapshot for a compiled activation schedule.

Parameters:
- `network` - Target network instance.
- `activationSchedule` - Cached compiled schedule.

Returns: Scheduling diagnostics snapshot.

### createDefaultSchedulingDiagnostics

```ts
createDefaultSchedulingDiagnostics(
  network: default,
  runtimeNetwork: NetworkRuntimeDiagnosticsInternals,
): ActivationSchedulingDiagnostics
```

Build a safe default diagnostics snapshot when no explicit scheduling record exists yet.

Parameters:
- `network` - Target network instance.
- `runtimeNetwork` - Runtime diagnostics internals.

Returns: Default scheduling diagnostics snapshot.

### disableDropConnect

```ts
disableDropConnect(): void
```

Disable DropConnect.

Parameters:
- `this` - Target network instance.

Returns: Nothing.

### enableDropConnect

```ts
enableDropConnect(
  probability: number,
): void
```

Enable DropConnect with a probability in $[0,1)$.

Parameters:
- `this` - Target network instance.
- `probability` - DropConnect probability.

Returns: Nothing.

### getActivationSchedulingDiagnostics

```ts
getActivationSchedulingDiagnostics(): ActivationSchedulingDiagnostics
```

Read a human-friendly snapshot of the current activation-ordering contract.

The snapshot explains whether activation is using a compiled schedule or a
fallback path, whether topology is currently dirty, and what callers should
do next when cycles or stale caches prevent the preferred schedule.

Parameters:
- `this` - Target network instance.

Returns: Activation scheduling diagnostics snapshot.

### getLastGradClipGroupCount

```ts
getLastGradClipGroupCount(): number
```

Read the last recorded gradient-clipping group count.

Parameters:
- `this` - Target network instance.

Returns: Last gradient-clipping group count.

### getLossScale

```ts
getLossScale(): number
```

Read the active mixed-precision loss scale.

Parameters:
- `this` - Target network instance.

Returns: Current loss scale.

### getRawGradientNorm

```ts
getRawGradientNorm(): number
```

Read the last recorded raw gradient norm.

Parameters:
- `this` - Target network instance.

Returns: Last raw gradient norm.

### getTrainingStats

```ts
getTrainingStats(): TrainingStatsSnapshot
```

Read a consolidated training-health snapshot.

Parameters:
- `this` - Target network instance.

Returns: Training statistics snapshot.

### resetDropoutMasks

```ts
resetDropoutMasks(): void
```

Reset every dropout mask to `1`.

This is useful after training so later inference does not inherit transient
node-level dropout state from a previous activation pass.

Parameters:
- `this` - Target network instance.

Returns: Nothing.

## architecture/network/runtime/network.runtime.errors.ts

Raised when a pruning schedule window is invalid.

### NetworkRuntimeDropConnectProbabilityRangeError

Raised when DropConnect probability is outside [0, 1).

### NetworkRuntimeLayeredWeightNoiseRequiredError

Raised when per-hidden-layer weight noise is requested on a non-layered network.

### NetworkRuntimePruningScheduleWindowError

Raised when a pruning schedule window is invalid.

### NetworkRuntimeStochasticDepthEntryCountError

Raised when stochastic-depth survival entries do not match hidden-layer count.

### NetworkRuntimeStochasticDepthLayeredNetworkRequiredError

Raised when stochastic depth is requested on a non-layered network.

### NetworkRuntimeStochasticDepthSurvivalArrayError

Raised when stochastic-depth survival input is not an array.

### NetworkRuntimeStochasticDepthSurvivalRangeError

Raised when a stochastic-depth survival probability falls outside (0, 1].

### NetworkRuntimeTargetSparsityRangeError

Raised when pruning target sparsity is outside the open interval (0, 1).

### NetworkRuntimeWeightNoiseConfigurationError

Raised when weight-noise configuration shape is invalid.

### NetworkRuntimeWeightNoiseEntryCountError

Raised when hidden-layer weight-noise entries do not match hidden-layer count.

### NetworkRuntimeWeightNoisePerLayerRangeError

Raised when a per-hidden-layer weight-noise value is negative.

### NetworkRuntimeWeightNoiseStdDevRangeError

Raised when weight-noise standard deviation is negative.
