# neat/evolve/objectives

## neat/evolve/objectives/evolve.objectives.utils.ts

### applyDynamicObjectiveSchedule

```ts
applyDynamicObjectiveSchedule(
  internal: NeatControllerForEvolution,
  currentObjectiveKeys: string[],
  config: { autoEntropyAddAt: number; },
): void
```

Apply dynamic objective scheduling and entropy rules.

Parameters:
- `internal` - - NEAT controller instance.
- `currentObjectiveKeys` - - Keys of active objectives.
- `config` - - Scheduling constants.

Returns: void.

### applyFitnessSuppressionForTests

```ts
applyFitnessSuppressionForTests(
  internal: NeatControllerForEvolution,
): void
```

Suppress fitness objective for specific test scenarios.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### captureObjectiveImportanceSnapshot

```ts
captureObjectiveImportanceSnapshot(
  internal: NeatControllerForEvolution,
): void
```

Capture objective importance stats for telemetry.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### handleEntropyDropAndReadd

```ts
handleEntropyDropAndReadd(
  internal: NeatControllerForEvolution,
  currentObjectiveKeys: string[],
  dynamicConfig: { enabled?: boolean | undefined; addComplexityAt?: number | undefined; addEntropyAt?: number | undefined; dropEntropyOnStagnation?: number | undefined; readdEntropyAfter?: number | undefined; } | undefined,
): void
```

Handle entropy removal and re-addition rules.

Parameters:
- `internal` - - NEAT controller instance.
- `currentObjectiveKeys` - - Active objective keys.
- `dynamicConfig` - - Dynamic objective config.

Returns: void.

### resetObjectivesCache

```ts
resetObjectivesCache(
  internal: NeatControllerForEvolution,
): void
```

Clear cached objectives so dynamic schedules can rebuild them.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### updateObjectiveScheduleAndAges

```ts
updateObjectiveScheduleAndAges(
  internal: NeatControllerForEvolution,
  helpers: { applyDynamicObjectiveSchedule: (currentObjectiveKeys: string[]) => void; },
): Promise<void>
```

Update objective schedule, pending adds/removes, and objective ages.

Parameters:
- `internal` - - NEAT controller instance.
- `helpers` - - Helper callbacks used by scheduling logic.
- `helpers` - - Dynamic objective scheduler.

Returns: void.
