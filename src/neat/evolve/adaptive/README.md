# neat/evolve/adaptive

## neat/evolve/adaptive/evolve.adaptive.utils.ts

### adaptReenableProbability

```ts
adaptReenableProbability(
  internal: NeatControllerForEvolution,
  config: { minSamples: number; target: number; min: number; max: number; deltaScale: number; },
): void
```

Adapt the re-enable probability based on recent success ratios.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### applyAdaptiveComplexityControllers

```ts
applyAdaptiveComplexityControllers(
  internal: NeatControllerForEvolution,
): Promise<void>
```

Apply adaptive complexity controllers if available.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### applyAncestorUniqAdaptiveSafe

```ts
applyAncestorUniqAdaptiveSafe(
  internal: NeatControllerForEvolution,
): Promise<void>
```

Apply ancestor uniqueness adaptation if available.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### applyAutoCompatibilityTuning

```ts
applyAutoCompatibilityTuning(
  internal: NeatControllerForEvolution,
  config: { targetMin: number; adjustRate: number; minCoeff: number; maxCoeff: number; randomScale: number; },
): void
```

Apply auto-compatibility tuning if enabled.

Parameters:
- `internal` - - NEAT controller instance.
- `config` - - Tuning constants.

Returns: void.

### applyMinimalCriterionAdaptiveSafe

```ts
applyMinimalCriterionAdaptiveSafe(
  internal: NeatControllerForEvolution,
): Promise<void>
```

Apply minimal criterion adaptive controller if available.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### applyOperatorAdaptationSafe

```ts
applyOperatorAdaptationSafe(
  internal: NeatControllerForEvolution,
): Promise<void>
```

Apply operator adaptation if available.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### applyPruningAndMutation

```ts
applyPruningAndMutation(
  internal: NeatControllerForEvolution,
): Promise<void>
```

Apply pruning and mutation phases.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### invalidateCompatibilityCaches

```ts
invalidateCompatibilityCaches(
  internal: NeatControllerForEvolution,
): void
```

Invalidate compatibility caches after mutations.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.
