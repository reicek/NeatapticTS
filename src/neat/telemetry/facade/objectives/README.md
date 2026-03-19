# neat/telemetry/facade/objectives

Objective-inspection and lifecycle helpers inside the public telemetry facade.

This chapter is the smallest public surface for answering three related
questions about the active objective set:

- which objectives are currently registered,
- what compact descriptor summary should a dashboard show,
- which add/remove events happened recently as the controller adapted.

It also carries the two lifecycle controls that let a caller register a new
objective or clear custom objectives without digging into the lower-level
objective subsystem directly.

Read this chapter after the root telemetry facade when the question is about
objective policy itself rather than telemetry buffers, species history, or
Pareto archive inspection.

```mermaid
flowchart TD
  Registry[Active objective registry] --> Keys[getObjectiveKeys()<br/>stable objective names]
  Registry --> Summary[getObjectives()<br/>compact descriptors]
  Events[Objective lifecycle events] --> History[getObjectiveEvents()<br/>recent add remove log]
  Register[registerTelemetryObjective()] --> Registry
  Clear[clearTelemetryObjectives()] --> Registry
```

## neat/telemetry/facade/objectives/telemetry.facade.objectives.ts

### clearTelemetryObjectives

```ts
clearTelemetryObjectives(
  host: TelemetryFacadeObjectivesHost,
): void
```

Remove all registered custom objectives so only the default objective path remains.

Reach for this when an experiment wants to reset the objective surface to its
baseline state without rebuilding the whole controller.

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

This is the historical companion to {@link getObjectives}. The descriptor
summary tells you what is active now; the event log tells you how the active
set changed across recent generations.

Parameters:
- `host` - - `Neat` instance storing objective lifecycle events.

Returns: Shallow copy of the recorded objective events.

Example:

```ts
const recentEvents = getObjectiveEvents(neat);
console.log(recentEvents.at(-1));
```

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

Example:

```ts
console.log(getObjectiveKeys(neat));
```

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

Example:

```ts
console.table(getObjectives(neat));
```

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

Use this when an experiment or dashboard wants to add a temporary objective
to the live registry without reaching into the deeper objective subsystem
directly through controller internals.

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

The host combines the read-side objective registry with the recent event log
because callers often want to inspect both the current state and the recent
policy changes in the same place.
