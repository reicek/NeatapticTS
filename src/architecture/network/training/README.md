# architecture/network/training

## architecture/network/training/network.training.utils.ts

### __trainingInternals

### applyGradientClippingImpl

`(net: import("C:/NeatapticTS/src/architecture/network").default, cfg: { mode: "norm" | "percentile" | "layerwiseNorm" | "layerwisePercentile"; maxNorm?: number | undefined; percentile?: number | undefined; }) => void`

### CheckpointConfig

Checkpoint callback configuration.

### CostFunction

`(target: number[], output: number[]) => number`

Cost function signature used by training utilities.

### GradientClipConfig

Gradient clipping configuration.

### MetricsHook

`(m: { iteration: number; error: number; plateauError?: number | undefined; gradNorm: number; }) => void`

Metrics hook signature.

### MixedPrecisionConfig

Mixed-precision configuration.

### MixedPrecisionDynamicConfig

Dynamic mixed-precision configuration.

### MovingAverageType

Moving-average strategy identifier.

### OptimizerConfigBase

Base optimizer configuration.

### ScheduleConfig

Schedule callback configuration.

### SerializedNetwork

Serialized network payload used in checkpoint callbacks.

### trainImpl

`(net: import("C:/NeatapticTS/src/architecture/network").default, set: { input: number[]; output: number[]; }[], options: import("C:/NeatapticTS/src/architecture/network/network.types").TrainingOptions) => { error: number; iterations: number; time: number; }`

### TrainingOptions

Public training options shape.

### trainSetImpl

`(net: import("C:/NeatapticTS/src/architecture/network").default, set: { input: number[]; output: number[]; }[], batchSize: number, accumulationSteps: number, currentRate: number, momentum: number, regularization: import("C:/NeatapticTS/src/architecture/network/network.types").RegularizationConfig, costFunction: import("C:/NeatapticTS/src/architecture/network/network.types").CostFunction | import("C:/NeatapticTS/src/architecture/network/network.types").CostFunctionOrObject, optimizer: import("C:/NeatapticTS/src/architecture/network/network.types").OptimizerConfigBase | undefined) => number`
