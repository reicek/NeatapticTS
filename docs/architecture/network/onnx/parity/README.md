# architecture/network/onnx/parity

## architecture/network/onnx/parity/network.onnx.parity.types.ts

### OnnxRuntimeParityExecutedFixtureDescriptor

Executable fixture descriptor for runtime-parity lanes that are already approved.

This narrows the generic fixture descriptor to the executed subset so callers
such as the Phase 9C randomized runner do not need unreachable skipped-result
guards after they create lane-approved samples.

### OnnxRuntimeParityExecutedResult

Result for one executed Phase 9 runtime-parity fixture.

The harness reports both native and ONNX Runtime outputs so later golden and
randomized passes can reuse the same comparison packet without changing the
execution seam.

### OnnxRuntimeParityExecutionMode

Execution state for one Phase 9 runtime-parity fixture.

`execute` means the current harness should export, run, and compare the
binary model immediately. `skip` keeps the lane in the frozen inventory but
requires an explicit follow-up tranche before execution begins.

### OnnxRuntimeParityFixtureDescriptor

Deterministic fixture descriptor for one Phase 9 runtime-parity lane.

Each descriptor couples a stable network factory, the binary export options,
one native runtime sample, the ONNX Runtime feed shape, and the tolerance
packet that later golden or randomized passes must honor.

### OnnxRuntimeParityLane

Named Phase 9 runtime-parity lanes for the current ONNX execution boundary.

Phase 9A freezes this list before golden parity cases widen coverage. Each
lane names one exporter-owned subset whose binary `.onnx` artifact can later
be compared against `Network.activate()` with an explicit tolerance packet.

Example:

```ts
const baselineLane: OnnxRuntimeParityLane = 'baseline-float32-dense';
```

### OnnxRuntimeParityRandomizedRunOptions

Seeded randomized-parity options for one Phase 9C sample run.

Phase 9C keeps the runtime subset narrow, but broadens evidence from named
golden fixtures into reproducible randomized samples. The seed and sample
count make failures replayable.

### OnnxRuntimeParityRandomizedSampleResult

Flattened result for one seeded randomized parity sample.

The Phase 9C runner returns a compact packet per sample so tests can freeze
deterministic summaries without reimplementing the execution seam.

### OnnxRuntimeParityResult

Runtime-parity result union type for the Phase 9 test harness seam.

### OnnxRuntimeParitySkippedResult

Result for one inventory-only skipped Phase 9 runtime-parity test fixture.

### OnnxRuntimeParityTolerancePacket

Explicit tolerance packet for one runtime-parity lane.

Phase 9 keeps tolerances named and lane-specific so reduced-precision cases
cannot silently inherit looser thresholds from unrelated fixtures.

## architecture/network/onnx/parity/network.onnx.parity.ts

### assignDeterministicParameters

```ts
assignDeterministicParameters(
  network: default,
  parameterSeed: DeterministicParameterSeed,
): void
```

Apply deterministic scalar parameters to a network fixture.

Parameters:
- `network` - Target network fixture.
- `parameterSeed` - Arithmetic progression used for weights and biases.

### buildExecutedResult

```ts
buildExecutedResult(
  fixtureDescriptor: OnnxRuntimeParityFixtureDescriptor,
  inputNames: readonly string[],
  outputNames: readonly string[],
  nativeOutput: number[],
  runtimeOutput: number[],
): OnnxRuntimeParityExecutedResult
```

Build the executed runtime-parity result packet for one fixture.

Parameters:
- `fixtureDescriptor` - Executed deterministic fixture descriptor.
- `inputNames` - Resolved runtime input names.
- `outputNames` - Resolved runtime output names.
- `nativeOutput` - Output from `Network.activate()`.
- `runtimeOutput` - Output from ONNX Runtime.

Returns: Executed comparison packet for the current fixture.

### calculateMaxAbsoluteDifference

```ts
calculateMaxAbsoluteDifference(
  nativeOutput: readonly number[],
  runtimeOutput: readonly number[],
): number
```

Calculate the maximum absolute scalar difference for one parity comparison.

Parameters:
- `nativeOutput` - Native runtime output.
- `runtimeOutput` - ONNX Runtime output.

Returns: Maximum absolute difference across the compared output vector.

### calculateMeanSquaredError

```ts
calculateMeanSquaredError(
  nativeOutput: readonly number[],
  runtimeOutput: readonly number[],
): number
```

Calculate mean squared error for one parity comparison.

Parameters:
- `nativeOutput` - Native runtime output.
- `runtimeOutput` - ONNX Runtime output.

Returns: Mean squared error across the compared output vector.

### createBaselineFloat32DenseFixture

```ts
createBaselineFloat32DenseFixture(): OnnxRuntimeParityFixtureDescriptor
```

Create the Phase 9A dense baseline fixture with float32 export and strict tolerances.

Returns: Baseline same-family dense runtime-parity descriptor.

### createConvGroundworkScenario

```ts
createConvGroundworkScenario(): { createNetwork: () => default; mappings: Conv2DMapping[]; nativeInputValues: number[]; runtimeInputValues: number[]; }
```

Create the deterministic explicit-Conv groundwork scenario reused by Phase 9.

Returns: Deterministic network factory, mappings, and sample inputs for the explicit-Conv lane.

### createDynamicUint8DenseGuidanceFixture

```ts
createDynamicUint8DenseGuidanceFixture(): OnnxRuntimeParityFixtureDescriptor
```

Create the dynamic-uint8 dense fixture that validates DynamicQuantizeLinear guidance export.

Returns: Dynamic-guidance runtime-parity descriptor with an explicit tolerance packet.

### createRandomizedDenseFixture

```ts
createRandomizedDenseFixture(
  fixtureDescriptor: OnnxRuntimeParityFixtureDescriptor,
  sampleSeed: number,
  sampleIndex: number,
): OnnxRuntimeParityExecutedFixtureDescriptor
```

Create one seeded same-family dense randomized fixture.

Parameters:
- `fixtureDescriptor` - Base dense execute fixture.
- `sampleSeed` - Deterministic seed for this randomized case.

Returns: Lane-compatible randomized dense fixture descriptor.

### createRandomizedNumericVector

```ts
createRandomizedNumericVector(
  sampleGenerator: () => number,
  vectorLength: number,
  numericRange: NumericRange,
): number[]
```

Create a reproducible numeric vector for one randomized parity sample.

Parameters:
- `sampleGenerator` - Deterministic unit-interval generator.
- `vectorLength` - Target vector length.
- `numericRange` - Numeric range for each sampled value.

Returns: Deterministic numeric vector for the sample input.

### createRandomizedParameterSeed

```ts
createRandomizedParameterSeed(
  sampleGenerator: () => number,
  parameterSeedBounds: { weightStart: NumericRange; weightStep: NumericRange; biasStart: NumericRange; biasStep: NumericRange; },
): DeterministicParameterSeed
```

Create a reproducible arithmetic parameter seed packet.

Parameters:
- `sampleGenerator` - Deterministic unit-interval generator.
- `parameterSeedBounds` - Numeric ranges for each parameter-seed component.

Returns: Deterministic scalar parameter packet for one randomized network.

### createRandomizedParityFixture

```ts
createRandomizedParityFixture(
  fixtureDescriptor: OnnxRuntimeParityFixtureDescriptor,
  sampleSeed: number,
  sampleIndex: number,
): OnnxRuntimeParityExecutedFixtureDescriptor
```

Materialize one lane-approved randomized fixture descriptor.

Parameters:
- `fixtureDescriptor` - Base execute fixture from the frozen Phase 9 inventory.
- `sampleSeed` - Deterministic seed for this randomized sample.

Returns: Randomized fixture descriptor that stays within the declared lane boundary.

### createRandomizedStatic8BitConvFixture

```ts
createRandomizedStatic8BitConvFixture(
  fixtureDescriptor: OnnxRuntimeParityFixtureDescriptor,
  sampleSeed: number,
  sampleIndex: number,
): OnnxRuntimeParityExecutedFixtureDescriptor
```

Create one seeded explicit-Conv static-8bit randomized fixture.

Parameters:
- `fixtureDescriptor` - Base explicit-Conv execute fixture.
- `sampleSeed` - Deterministic seed for this randomized case.

Returns: Lane-compatible randomized explicit-Conv fixture descriptor.

### createRuntimeParityTensor

```ts
createRuntimeParityTensor(
  arg0: unknown,
  arg1: unknown,
  arg2: readonly number[] | undefined,
): Tensor
```

Create one tensor instance that matches the raw binding's float32 output quirks.

The current Node binding can surface float32 outputs through `ArrayBuffer` or
non-`Float32Array` views. This helper normalizes those payloads before they
reach the standard `Tensor` constructor.

Parameters:
- `arg0` - Raw tensor type or data argument.
- `arg1` - Raw tensor data argument.
- `arg2` - Optional tensor dimensions.

Returns: A standard ONNX Runtime tensor instance.

### createSeededUnitIntervalGenerator

```ts
createSeededUnitIntervalGenerator(
  initialSeed: number,
): () => number
```

Create one reproducible unit-interval generator from a 32-bit seed.

Parameters:
- `initialSeed` - Seed used for the linear congruential generator.

Returns: Deterministic floating-point generator in the range [0, 1).

### createStatic8BitConvQlinearFixture

```ts
createStatic8BitConvQlinearFixture(): OnnxRuntimeParityFixtureDescriptor
```

Create the explicit-conv static-8bit fixture for conv-lane runtime parity checks.

Returns: Static-8bit explicit-Conv runtime-parity descriptor.

### createStatic8BitDenseQlinearFixture

```ts
createStatic8BitDenseQlinearFixture(): OnnxRuntimeParityFixtureDescriptor
```

Create the static-8bit dense qlinear fixture with external calibration metadata.

Returns: Static-8bit dense qlinear runtime-parity descriptor.

### createStorageFp16DenseFixture

```ts
createStorageFp16DenseFixture(): OnnxRuntimeParityFixtureDescriptor
```

Create the storage-fp16 dense fixture used to validate cast-bridge parity behavior.

Returns: Storage-fp16 cast-bridge runtime-parity descriptor.

### cycleIntegerRangeBySampleIndex

```ts
cycleIntegerRangeBySampleIndex(
  sampleIndex: number,
  numericRange: NumericRange,
): number
```

Cycle deterministically through an inclusive integer range by sample index.

Parameters:
- `sampleIndex` - Zero-based randomized sample index.
- `numericRange` - Inclusive integer range.

Returns: Deterministic integer derived from the sample position.

### executeRuntimeParityInSubprocess

```ts
executeRuntimeParityInSubprocess(
  binaryModel: Uint8Array<ArrayBufferLike>,
  fixtureDescriptor: OnnxRuntimeParityFixtureDescriptor,
): OnnxRuntimeParitySubprocessResult
```

Execute one ONNX Runtime parity case in a fresh child Node process.

This keeps the raw binding isolated from the Jest process so Phase 8 runtime
validation and Phase 9 parity execution do not compete over native addon
initialization order.

Parameters:
- `binaryModel` - Binary `.onnx` payload produced by `exportToONNXBinary()`.
- `fixtureDescriptor` - Deterministic runtime-parity descriptor.

Returns: Runtime output packet emitted by the child process.

### findPhase9RuntimeParityFixture

```ts
findPhase9RuntimeParityFixture(
  fixtureId: OnnxRuntimeParityLane,
): OnnxRuntimeParityFixtureDescriptor
```

Resolve one named Phase 9 runtime-parity fixture from the frozen inventory.

Parameters:
- `fixtureId` - Named fixture identifier from the frozen Phase 9A inventory.

Returns: The matching deterministic fixture descriptor.

### getPhase9RuntimeParityInventory

```ts
getPhase9RuntimeParityInventory(): readonly OnnxRuntimeParityFixtureDescriptor[]
```

Return the frozen Phase 9A runtime-parity inventory.

This inventory is the narrow stop line between the closed Phase 8 binary
acceptance lane and the later Phase 9B golden parity tranche. Keeping the
list centralized prevents new runtime lanes from slipping in through ad hoc
tests.

Returns: The current Phase 9A fixture inventory in deterministic order.

### isArrayBufferLike

```ts
isArrayBufferLike(
  tensorData: unknown,
): boolean
```

Check whether one runtime payload should be treated as an ArrayBuffer source.

Parameters:
- `tensorData` - Raw tensor payload from the native binding.

Returns: True when the payload is a plain or shared ArrayBuffer-like object.

### resolveRuntimeSubprocessError

```ts
resolveRuntimeSubprocessError(
  subprocessResult: { stderr: string; stdout: string; },
): string
```

Resolve a stable parity-subprocess error message from stderr, stdout, or fallback text.

Parameters:
- `subprocessResult` - Child-process execution result.

Returns: Human-readable error message.

### resolveRuntimeTensorDimensions

```ts
resolveRuntimeTensorDimensions(
  tensorShape: readonly (string | number)[],
): number[]
```

Resolve runtime feed dimensions from the fixture packet or session metadata.

Parameters:
- `tensorShape` - Runtime tensor shape dimensions (numeric or symbolic).

Returns: Numeric input dimensions suitable for the feed tensor constructor.

### resolveValidatedRandomizedSampleCount

```ts
resolveValidatedRandomizedSampleCount(
  sampleCount: number,
): number
```

Resolve the allowed randomized sample count for Phase 9C.

Parameters:
- `sampleCount` - Requested randomized sample count.

Returns: Validated sample count within the supported Phase 9C bounds.

### runOnnxRuntimeParityFixture

```ts
runOnnxRuntimeParityFixture(
  fixtureDescriptor: OnnxRuntimeParityExecutedFixtureDescriptor,
): Promise<OnnxRuntimeParityExecutedResult>
```

Export, execute, and compare one Phase 9 runtime-parity fixture.

The harness always starts from binary `.onnx` bytes so Phase 9 stays
binary-first rather than drifting back to the JSON-first roundtrip lane.
Skipped fixtures are returned with their explicit reason instead of being
executed implicitly.

Parameters:
- `fixtureDescriptor` - Deterministic runtime-parity fixture descriptor.

Returns: Either a skipped result or an executed comparison packet.

### runSeededOnnxRuntimeParitySamples

```ts
runSeededOnnxRuntimeParitySamples(
  fixtureDescriptor: OnnxRuntimeParityFixtureDescriptor,
  randomizedRunOptions: OnnxRuntimeParityRandomizedRunOptions,
): Promise<OnnxRuntimeParityRandomizedSampleResult[]>
```

Execute reproducible randomized parity samples for one approved Phase 9 lane.

Phase 9C reuses the existing binary-first fixture seam, but broadens the
evidence from one named golden sample into several seeded randomized cases.
The runner keeps the subset narrow by generating only lane-approved shapes.

Parameters:
- `fixtureDescriptor` - Approved Phase 9 runtime-parity fixture descriptor.
- `randomizedRunOptions` - Seed and sample-count packet for this randomized pass.

Returns: Flattened executed results for each seeded randomized sample.

### sampleNumberInRange

```ts
sampleNumberInRange(
  sampleGenerator: () => number,
  numericRange: NumericRange,
): number
```

Sample one rounded floating-point value inside a numeric range.

Parameters:
- `sampleGenerator` - Deterministic unit-interval generator.
- `numericRange` - Inclusive numeric range.

Returns: Rounded floating-point sample inside the declared range.
