# neat/evolve/telemetry

## neat/evolve/telemetry/evolve.telemetry.utils.ts

### computeDiversityStatsSafely

```ts
computeDiversityStatsSafely(
  internal: NeatControllerForEvolution,
): void
```

Compute diversity stats safely if the hook exists.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### recordTelemetryIfEnabled

```ts
recordTelemetryIfEnabled(
  internal: NeatControllerForEvolution,
  snapshot: default,
): Promise<void>
```

Record telemetry if enabled.

Parameters:
- `internal` - - NEAT controller instance.
- `snapshot` - - Best network snapshot for the generation.

Returns: void.
