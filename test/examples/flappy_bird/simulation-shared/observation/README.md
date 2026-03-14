# simulation-shared/observation

Shared observation public entry.

This focused boundary keeps observation feature assembly separate from
observation-vector projection so browser, environment, and worker callers can
depend on a smaller, clearer public surface.

Conceptually, this folder is the "state representation" layer for the
Flappy Bird example. Instead of learning directly from pixels, the agent sees
a compact set of geometric and kinematic signals derived from the current
world state. That makes the example easier to study, faster to evaluate, and
closer to classic feature-engineering pipelines used in small control tasks.

If you want background reading, the Wikipedia articles on "feature
engineering" and "state space representation" give useful intuition for why
this boundary exists at all.

## simulation-shared/observation/observation.ts

### resolveCoreObservationVectorFromFeatures

```ts
resolveCoreObservationVectorFromFeatures(
  features: SharedObservationFeatures,
): number[]
```

Resolves the compact core vector used for temporal stacking.

The core intentionally keeps directly observed kinematic and geometric
channels while dropping derived one-step predictors that become redundant
once short-term temporal memory is available.

This is the representation used when the example wants a short history of raw
observation slices. The idea is similar to frame stacking in reinforcement
learning: a feed-forward policy can recover some sense of motion by looking
at several recent compact frames at once.

The Wikipedia article on "frame stacking" is a useful conceptual reference.

Parameters:
- `features` - - Structured observation features.

Returns: Core per-frame vector.

Example:

```ts
const coreFrame = resolveCoreObservationVectorFromFeatures(features);
observationMemoryState.previousCoreFrames.push(coreFrame);
```

### resolveObservationFeatures

```ts
resolveObservationFeatures(
  input: SharedObservationInput,
): SharedObservationFeatures
```

Builds the shared normalized observation feature set consumed by policies.

Educational note:
This helper stays focused on semantic feature assembly only. Projection into
the canonical network vectors now lives in the neighboring vector module so
observation policy and network-shape concerns can evolve independently.

The features deliberately mix three kinds of signal:
1. Current state, such as bird height and vertical velocity.
2. Near-term geometry, such as gap bounds and upcoming-pipe distances.
3. Simple forward-looking control hints, such as urgency and one-flap
   reachability.

This is a compact example of feature engineering for control. Instead of
asking NEAT to rediscover basic geometry from raw sensory input, the example
hands the network semantically meaningful signals and lets evolution focus on
policy search.

For broader context, the Wikipedia article on "feature engineering" is a
good companion reference.

Parameters:
- `input` - - Observation input bundle.

Returns: Structured observation features.

Example:

```ts
const features = resolveObservationFeatures({
  birdYPx: 120,
  velocityYPxPerFrame: 2,
  pipes,
  visibleWorldWidthPx: 640,
  difficultyProfile,
  activeSpawnIntervalFrames: 90,
});

if (features.normalizedEntryUrgency > 0.8) {
  // The bird is misaligned and running out of time to recover.
}
```

### resolveObservationVectorFromFeatures

```ts
resolveObservationVectorFromFeatures(
  features: SharedObservationFeatures,
): number[]
```

Converts observation features to the canonical 12-value network input vector.

Educational note:
This module owns the network-shape projection so feature semantics can change
independently from how the policy input is ordered.

The 12-value vector is the compact feed-forward policy input used by the main
evaluation and training flow. Its ordering is stable on purpose: once a
network topology has evolved against one input layout, silent channel
reshuffles would invalidate learned behavior.

Parameters:
- `features` - - Structured feature object.

Returns: Ordered feature vector.

Example:

```ts
const features = resolveObservationFeatures(input);
const networkInput = resolveObservationVectorFromFeatures(features);
```

### resolveUpcomingPipes

```ts
resolveUpcomingPipes(
  pipes: SharedPipeLike[],
  birdCenterXPx: number,
  birdRadiusPx: number,
  pipeWidthPx: number,
): [SharedPipeLike | undefined, SharedPipeLike | undefined]
```

Resolves the next two upcoming pipes in front of the bird.

The observation pipeline only cares about the immediate near future, because
Flappy Bird decisions are dominated by the next gap and the transition after
it. Looking further ahead adds noise faster than it adds useful control
signal.

Parameters:
- `pipes` - - Current pipe list.
- `birdCenterXPx` - - Bird center x-position.
- `birdRadiusPx` - - Bird radius.
- `pipeWidthPx` - - Pipe width.

Returns: Tuple of first and second upcoming pipes.

Example:

```ts
const [nextPipe, secondPipe] = resolveUpcomingPipes(pipes);
```

## simulation-shared/observation/observation.vector.utils.ts

### resolveCoreObservationVectorFromFeatures

```ts
resolveCoreObservationVectorFromFeatures(
  features: SharedObservationFeatures,
): number[]
```

Resolves the compact core vector used for temporal stacking.

The core intentionally keeps directly observed kinematic and geometric
channels while dropping derived one-step predictors that become redundant
once short-term temporal memory is available.

This is the representation used when the example wants a short history of raw
observation slices. The idea is similar to frame stacking in reinforcement
learning: a feed-forward policy can recover some sense of motion by looking
at several recent compact frames at once.

The Wikipedia article on "frame stacking" is a useful conceptual reference.

Parameters:
- `features` - - Structured observation features.

Returns: Core per-frame vector.

Example:

```ts
const coreFrame = resolveCoreObservationVectorFromFeatures(features);
observationMemoryState.previousCoreFrames.push(coreFrame);
```

### resolveObservationVectorFromFeatures

```ts
resolveObservationVectorFromFeatures(
  features: SharedObservationFeatures,
): number[]
```

Converts observation features to the canonical 12-value network input vector.

Educational note:
This module owns the network-shape projection so feature semantics can change
independently from how the policy input is ordered.

The 12-value vector is the compact feed-forward policy input used by the main
evaluation and training flow. Its ordering is stable on purpose: once a
network topology has evolved against one input layout, silent channel
reshuffles would invalidate learned behavior.

Parameters:
- `features` - - Structured feature object.

Returns: Ordered feature vector.

Example:

```ts
const features = resolveObservationFeatures(input);
const networkInput = resolveObservationVectorFromFeatures(features);
```

## simulation-shared/observation/observation.features.utils.ts

### clamp

```ts
clamp(
  value: number,
  min: number,
  max: number,
): number
```

Clamps a numeric value to the inclusive [min, max] interval.

Observation synthesis normalizes many raw measurements, so this helper keeps
derived channels inside their documented ranges.

Parameters:
- `value` - - Candidate value.
- `min` - - Inclusive lower bound.
- `max` - - Inclusive upper bound.

Returns: Clamped value.

### clamp01

```ts
clamp01(
  value: number,
): number
```

Clamps a numeric value to the inclusive [0, 1] interval.

This is used for channels that are naturally interpreted as normalized
proportions or bounded progress values.

Parameters:
- `value` - - Candidate value.

Returns: Value clamped between 0 and 1.

### resolveObservationFeatures

```ts
resolveObservationFeatures(
  input: SharedObservationInput,
): SharedObservationFeatures
```

Builds the shared normalized observation feature set consumed by policies.

Educational note:
This helper stays focused on semantic feature assembly only. Projection into
the canonical network vectors now lives in the neighboring vector module so
observation policy and network-shape concerns can evolve independently.

The features deliberately mix three kinds of signal:
1. Current state, such as bird height and vertical velocity.
2. Near-term geometry, such as gap bounds and upcoming-pipe distances.
3. Simple forward-looking control hints, such as urgency and one-flap
   reachability.

This is a compact example of feature engineering for control. Instead of
asking NEAT to rediscover basic geometry from raw sensory input, the example
hands the network semantically meaningful signals and lets evolution focus on
policy search.

For broader context, the Wikipedia article on "feature engineering" is a
good companion reference.

Parameters:
- `input` - - Observation input bundle.

Returns: Structured observation features.

Example:

```ts
const features = resolveObservationFeatures({
  birdYPx: 120,
  velocityYPxPerFrame: 2,
  pipes,
  visibleWorldWidthPx: 640,
  difficultyProfile,
  activeSpawnIntervalFrames: 90,
});

if (features.normalizedEntryUrgency > 0.8) {
  // The bird is misaligned and running out of time to recover.
}
```

### resolvePredictedBirdYAtFrames

```ts
resolvePredictedBirdYAtFrames(
  startYPx: number,
  initialVerticalVelocityPxPerFrame: number,
  frameHorizon: number,
): number
```

Predicts bird y-position after a frame horizon with constant gravity.

This is a deliberately cheap forward model. It ignores richer control
sequences and simply answers: "where would the bird be after N frames under
the current physics assumption?" That small prediction is enough to build the
reachability and urgency features used by the controller.

Parameters:
- `startYPx` - - Current bird y-position.
- `initialVerticalVelocityPxPerFrame` - - Initial vertical velocity.
- `frameHorizon` - - Predicted horizon in simulation frames.

Returns: Predicted y-position.

### resolveUpcomingPipes

```ts
resolveUpcomingPipes(
  pipes: SharedPipeLike[],
  birdCenterXPx: number,
  birdRadiusPx: number,
  pipeWidthPx: number,
): [SharedPipeLike | undefined, SharedPipeLike | undefined]
```

Resolves the next two upcoming pipes in front of the bird.

The observation pipeline only cares about the immediate near future, because
Flappy Bird decisions are dominated by the next gap and the transition after
it. Looking further ahead adds noise faster than it adds useful control
signal.

Parameters:
- `pipes` - - Current pipe list.
- `birdCenterXPx` - - Bird center x-position.
- `birdRadiusPx` - - Bird radius.
- `pipeWidthPx` - - Pipe width.

Returns: Tuple of first and second upcoming pipes.

Example:

```ts
const [nextPipe, secondPipe] = resolveUpcomingPipes(pipes);
```
