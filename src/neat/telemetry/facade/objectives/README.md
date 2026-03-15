# neat/telemetry/facade/objectives

## neat/telemetry/facade/objectives/telemetry.facade.objectives.ts

### clearTelemetryObjectives

```ts
clearTelemetryObjectives(
  host: TelemetryFacadeObjectivesHost,
): void
```

Remove all registered custom objectives so only the default objective path remains.

Parameters:
- `host` - - `Neat` instance whose objective registry should be cleared.

Returns: Nothing. The helper mutates the objective registry in place.

### getObjectiveEvents

```ts
getObjectiveEvents(
  host: TelemetryFacadeObjectivesHost,
): { gen: number; type: "add" | "remove"; key: string; }[]
```

Snapshot recent objective add/remove events for telemetry consumers.

Parameters:
- `host` - - `Neat` instance storing objective lifecycle events.

Returns: Shallow copy of the recorded objective events.

### getObjectiveKeys

```ts
getObjectiveKeys(
  host: TelemetryFacadeObjectivesHost,
): string[]
```

Return just the registered objective keys in stable order.

This is the shortest inspection surface for tests and quick diagnostics that
only need to confirm which objectives are active, not the full descriptor
payload.

Parameters:
- `host` - - `Neat` instance exposing objective descriptors.

Returns: Ordered list of active objective keys.

### getObjectives

```ts
getObjectives(
  host: TelemetryFacadeObjectivesHost,
): { key: string; direction: "max" | "min"; }[]
```

Return a compact view of active objective descriptors.

The full objective descriptor includes accessors and internal metadata. This
read model trims that down to the pieces most useful in UI surfaces and
debugging output: the key and whether the objective is minimized or
maximized.

Parameters:
- `host` - - `Neat` instance exposing objective descriptors.

Returns: Compact objective summaries in evaluation order.

### registerTelemetryObjective

```ts
registerTelemetryObjective(
  host: TelemetryFacadeObjectivesHost,
  key: string,
  direction: "max" | "min",
  accessor: (genome: GenomeLike) => number,
): void
```

Register or replace a custom objective.

Parameters:
- `host` - - `Neat` instance whose multi-objective registry should change.
- `key` - - Unique objective key.
- `direction` - - Whether lower or higher values are considered better.
- `accessor` - - Function that reads the objective value from a genome.

Returns: Nothing. The objective registry on `host` is updated in place.

### TelemetryFacadeObjectivesHost

Narrow telemetry-facade host surface required by the objectives chapter.

This chapter keeps objective lifecycle reads and registration helpers beside
each other so the root telemetry facade can delegate that policy cluster as a
single concept instead of mixing it with telemetry buffers or lineage reads.
