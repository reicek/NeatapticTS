# neat/evolve/runtime

## neat/evolve/runtime/evolve.runtime.utils.ts

### buildFittestSnapshot

```ts
buildFittestSnapshot(
  internal: NeatControllerForEvolution,
): default
```

Build a cloned Network from the current best genome.

Parameters:
- `internal` - - NEAT controller instance.

Returns: best network snapshot.

### clearPopulationScores

```ts
clearPopulationScores(
  internal: NeatControllerForEvolution,
): void
```

Clear genome scores to force re-evaluation.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### computeElapsedTime

```ts
computeElapsedTime(
  startTimestamp: number,
): number
```

Compute elapsed time since the start of evolve().

Parameters:
- `startTimestamp` - - Start time resolved earlier.

Returns: elapsed time.

### ensurePopulationEvaluated

```ts
ensurePopulationEvaluated(
  internal: NeatControllerForEvolution,
): Promise<void>
```

Ensure the population is evaluated before evolution operations.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### resolveStartTime

```ts
resolveStartTime(): number
```

Resolve the start time for an evolution step.

Returns: timestamp in milliseconds or high-resolution units.

### trackGlobalImprovement

```ts
trackGlobalImprovement(
  internal: NeatControllerForEvolution,
  snapshot: default,
): void
```

Track global best improvement for stagnation logic.

Parameters:
- `internal` - - NEAT controller instance.
- `snapshot` - - Best network snapshot.

Returns: void.

### updateGlobalBestTracking

```ts
updateGlobalBestTracking(
  internal: NeatControllerForEvolution,
): void
```

Update generation-level best score tracking.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.
