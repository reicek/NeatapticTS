# simulation-shared

## simulation-shared/simulation-shared.types.ts

### simulation-shared.types

Minimal deterministic random contract used by shared spawn helpers.

### SharedDifficultyProfile

Shared runtime difficulty profile used by browser and environment simulators.

### SharedObservationFeatures

Structured observation features for network input.

### SharedObservationInput

Input shape for observation-feature synthesis.

### SharedObservationMemoryState

Mutable temporal memory attached to one policy-controlled bird.

The memory stores recent core observation frames and recent action history,
allowing feedforward policies to consume short-term context without adding
recurrent connections.

### SharedPipeLike

Common pipe shape consumed by observation helpers.

### SharedRngLike

Minimal deterministic random contract used by shared spawn helpers.

## simulation-shared/simulation-shared.errors.ts

### simulation-shared.errors

Prefix used when formatting unexpected shared-simulation errors.

### FLAPPY_SHARED_SIMULATION_ERROR_PREFIX

### formatSharedSimulationErrorMessage

`(error: unknown) => string`

Formats unknown shared-simulation errors for stable logs.

Parameters:
- `error` - - Unknown error value.

Returns: Readable error message.

## simulation-shared/simulation-shared.constants.ts

### simulation-shared.constants

Default curriculum scale used when callers do not provide one.

A value of `1` means full adaptive difficulty behavior is enabled.

### FLAPPY_SHARED_DEFAULT_DIFFICULTY_SCALE

### FLAPPY_SHARED_DEFAULT_NORMALIZATION_EPSILON

## simulation-shared/simulation-shared.math.utils.ts

### simulation-shared.math.utils

Clamps a numeric value to the inclusive `[min, max]` interval.

@param value - Candidate value.
@param min - Inclusive lower bound.
@param max - Inclusive upper bound.
@returns Clamped value.

### clamp

`(value: number, min: number, max: number) => number`

Internal clamp primitive.

Parameters:
- `value` - - Candidate value.
- `min` - - Inclusive lower bound.
- `max` - - Inclusive upper bound.

Returns: Clamped value.

### clamp01

`(value: number) => number`

Clamps a numeric value to the inclusive `[0, 1]` interval.

Parameters:
- `value` - - Candidate value.

Returns: Value clamped between 0 and 1.

### clampValue

`(value: number, min: number, max: number) => number`

Clamps a numeric value to the inclusive `[min, max]` interval.

Parameters:
- `value` - - Candidate value.
- `min` - - Inclusive lower bound.
- `max` - - Inclusive upper bound.

Returns: Clamped value.

### interpolateValue

`(startValue: number, endValue: number, progress: number) => number`

Linear interpolation helper.

Parameters:
- `startValue` - - Start value at progress `0`.
- `endValue` - - End value at progress `1`.
- `progress` - - Normalized interpolation progress.

Returns: Interpolated value.

## simulation-shared/simulation-shared.spawn.utils.ts

### resolveNextSpawnGapCenterY

`(previousGapCenterYPx: number, rng: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedRngLike, maximumGapCenterYPx: number) => number`

Resolves next gap center with bounded per-pipe delta.

Parameters:
- `previousGapCenterYPx` - - Previous spawn gap center.
- `rng` - - Deterministic RNG.
- `maximumGapCenterYPx` - - Optional inclusive upper bound for smaller viewports.

Returns: Next gap center y-position.

### resolveNextSpawnGapSize

`(previousSpawnGapPx: number | undefined, difficultyProfile: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedDifficultyProfile, rng: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedRngLike) => number`

Resolves next spawn gap size using progressive shrink and jitter.

Parameters:
- `previousSpawnGapPx` - - Previous spawn gap size.
- `difficultyProfile` - - Active difficulty profile.
- `rng` - - Deterministic RNG.

Returns: Next spawn gap size.

### resolveNextSpawnIntervalFrames

`(previousSpawnIntervalFrames: number | undefined, difficultyProfile: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedDifficultyProfile) => number`

Resolves next spawn interval using progressive shrink.

Parameters:
- `previousSpawnIntervalFrames` - - Previous spawn interval.
- `difficultyProfile` - - Active difficulty profile.

Returns: Next spawn interval in frames.

### sampleGapCenterY

`(rng: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedRngLike, maximumGapCenterYPx: number) => number`

Samples a random gap center y-position.

Parameters:
- `rng` - - Deterministic RNG.
- `maximumGapCenterYPx` - - Optional inclusive upper bound for smaller viewports.

Returns: Sampled y-position.

## simulation-shared/simulation-shared.memory.utils.ts

### commitSharedObservationMemoryStep

`(observationMemoryState: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationMemoryState, features: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationFeatures, didFlap: boolean) => void`

Commits one observation-action step into temporal memory.

Parameters:
- `observationMemoryState` - - Mutable temporal memory for the active bird.
- `features` - - Structured observation features used for the decision.
- `didFlap` - - Decision taken at this step.

Returns: Nothing.

### createSharedObservationMemoryState

`() => import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationMemoryState`

Creates an empty temporal observation memory state.

Returns: Fresh mutable memory buffers for one bird/controller.

### resolvePreviousCoreFramesWithPadding

`(observationMemoryState: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationMemoryState) => number[][]`

Resolves previous core frames (newest-first) with deterministic zero padding.

Parameters:
- `observationMemoryState` - - Mutable temporal memory for the active bird.

Returns: Previous core frame list with fixed target length.

### resolveTemporalObservationVector

`(features: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationFeatures, observationMemoryState: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationMemoryState) => number[]`

Builds the temporal policy input vector (stacked observation + action memory).

Output layout:
1) current core observation frame
2) previous core frames (newest to oldest) with zero padding
3) last-action channel
4) recent flap-rate channel over a fixed window

Parameters:
- `features` - - Structured observation features for the current decision step.
- `observationMemoryState` - - Mutable temporal memory for the active bird.

Returns: Ordered temporal input vector for policy activation.

### resolveZeroCoreObservationFrame

`() => number[]`

Builds a zero-valued core frame with canonical length.

Returns: Zero core frame.

## simulation-shared/simulation-shared.control.utils.ts

### simulation-shared.control.utils

Resolves flap/no-flap decision from network outputs.

@param rawOutputs - Activation output payload.
@param flapThreshold - Scalar threshold for single-output policies.
@returns True when flap should trigger.

### resolveFlapDecision

`(rawOutputs: unknown, flapThreshold: number) => boolean`

Resolves flap/no-flap decision from network outputs.

Parameters:
- `rawOutputs` - - Activation output payload.
- `flapThreshold` - - Scalar threshold for single-output policies.

Returns: True when flap should trigger.

## simulation-shared/simulation-shared.difficulty.utils.ts

### resolveAdaptiveDifficultyProfile

`(pipesPassed: number, difficultyScale: number) => import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedDifficultyProfile`

Resolves adaptive difficulty profile from passed-pipe progress.

Parameters:
- `pipesPassed` - - Number of passed pipes.
- `difficultyScale` - - Curriculum scale in `[0, 1]`.

Returns: Active difficulty profile.

## simulation-shared/simulation-shared.statistics.utils.ts

### compareNumbersAscending

`(leftValue: number, rightValue: number) => number`

Compares two numeric values in ascending order.

Parameters:
- `leftValue` - - Left numeric value.
- `rightValue` - - Right numeric value.

Returns: Comparator delta for `Array.prototype.toSorted`.

### computeMean

`(values: readonly number[]) => number`

Computes arithmetic mean for numeric samples.

Parameters:
- `values` - - Numeric samples.

Returns: Arithmetic mean.

### computePercentile

`(values: readonly number[], percentile: number) => number`

Computes percentile value via linear interpolation between nearest ranks.

Parameters:
- `values` - - Numeric samples.
- `percentile` - - Percentile in [0, 1].

Returns: Percentile value, or `Number.NaN` when `values` is empty.

### computePopulationStandardDeviation

`(values: readonly number[], meanValue: number) => number`

Computes population standard deviation.

Parameters:
- `values` - - Numeric samples.
- `meanValue` - - Precomputed mean.

Returns: Population standard deviation.

## simulation-shared/simulation-shared.observation.utils.ts

### simulation-shared.observation.utils

Shared observation compatibility façade.

The observation implementation now lives under `simulation-shared/observation/`
so feature synthesis and vector projection can evolve behind a focused module
boundary. This file stays as the stable import path for existing callers.

### resolveCoreObservationVectorFromFeatures

`(features: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationFeatures) => number[]`

Resolves the compact core vector used for temporal stacking.

The core intentionally keeps directly observed kinematic and geometric
channels while dropping derived one-step predictors that become redundant
once short-term temporal memory is available.

Parameters:
- `features` - - Structured observation features.

Returns: Core per-frame vector.

### resolveObservationFeatures

`(input: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationInput) => import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationFeatures`

Builds the shared normalized observation feature set consumed by policies.

Educational note:
This helper stays focused on semantic feature assembly only. Projection into
the canonical network vectors now lives in the neighboring vector module so
observation policy and network-shape concerns can evolve independently.

Parameters:
- `input` - - Observation input bundle.

Returns: Structured observation features.

### resolveObservationVectorFromFeatures

`(features: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationFeatures) => number[]`

Converts observation features to the canonical 12-value network input vector.

Educational note:
This module owns the network-shape projection so feature semantics can change
independently from how the policy input is ordered.

Parameters:
- `features` - - Structured feature object.

Returns: Ordered feature vector.

### resolveUpcomingPipes

`(pipes: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedPipeLike[], birdCenterXPx: number, birdRadiusPx: number, pipeWidthPx: number) => [import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedPipeLike | undefined, import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedPipeLike | undefined]`

Resolves the next two upcoming pipes in front of the bird.

Parameters:
- `pipes` - - Current pipe list.
- `birdCenterXPx` - - Bird center x-position.
- `birdRadiusPx` - - Bird radius.
- `pipeWidthPx` - - Pipe width.

Returns: Tuple of first and second upcoming pipes.
