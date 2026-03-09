# simulation-shared/observation

## simulation-shared/observation/observation.ts

### observation

Shared observation public entry.

This focused boundary keeps observation feature assembly separate from
observation-vector projection so browser, environment, and worker callers can
depend on a smaller, clearer public surface.

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

## simulation-shared/observation/observation.vector.utils.ts

### resolveCoreObservationVectorFromFeatures

`(features: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationFeatures) => number[]`

Resolves the compact core vector used for temporal stacking.

The core intentionally keeps directly observed kinematic and geometric
channels while dropping derived one-step predictors that become redundant
once short-term temporal memory is available.

Parameters:

- `features` - - Structured observation features.

Returns: Core per-frame vector.

### resolveObservationVectorFromFeatures

`(features: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationFeatures) => number[]`

Converts observation features to the canonical 12-value network input vector.

Educational note:
This module owns the network-shape projection so feature semantics can change
independently from how the policy input is ordered.

Parameters:

- `features` - - Structured feature object.

Returns: Ordered feature vector.

## simulation-shared/observation/observation.features.utils.ts

### clamp

`(value: number, min: number, max: number) => number`

Clamps a numeric value to the inclusive [min, max] interval.

Parameters:

- `value` - - Candidate value.
- `min` - - Inclusive lower bound.
- `max` - - Inclusive upper bound.

Returns: Clamped value.

### clamp01

`(value: number) => number`

Clamps a numeric value to the inclusive [0, 1] interval.

Parameters:

- `value` - - Candidate value.

Returns: Value clamped between 0 and 1.

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

### resolvePredictedBirdYAtFrames

`(startYPx: number, initialVerticalVelocityPxPerFrame: number, frameHorizon: number) => number`

Predicts bird y-position after a frame horizon with constant gravity.

Parameters:

- `startYPx` - - Current bird y-position.
- `initialVerticalVelocityPxPerFrame` - - Initial vertical velocity.
- `frameHorizon` - - Predicted horizon in simulation frames.

Returns: Predicted y-position.

### resolveUpcomingPipes

`(pipes: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedPipeLike[], birdCenterXPx: number, birdRadiusPx: number, pipeWidthPx: number) => [import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedPipeLike | undefined, import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedPipeLike | undefined]`

Resolves the next two upcoming pipes in front of the bird.

Parameters:

- `pipes` - - Current pipe list.
- `birdCenterXPx` - - Bird center x-position.
- `birdRadiusPx` - - Bird radius.
- `pipeWidthPx` - - Pipe width.

Returns: Tuple of first and second upcoming pipes.
