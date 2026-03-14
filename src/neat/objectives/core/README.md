# neat/objectives/core

Objective-list mechanics used by the NEAT controller.

This chapter holds the default fitness objective, validation helpers, and
the small list-management helpers behind objective registration.

## neat/objectives/core/objectives.types.ts

### NeatLikeWithObjectives

Minimal NEAT host contract required by objective-management helpers.

Objective registration only needs multi-objective configuration, a cached
objective list, and the flag that suppresses the default fitness objective.

## neat/objectives/core/objectives.core.ts

### buildDefaultFitnessObjective

```ts
buildDefaultFitnessObjective(): ObjectiveDescriptor
```

Build the default fitness objective descriptor.

Returns: Default fitness objective descriptor.

### collectDefaultObjectives

```ts
collectDefaultObjectives(
  neatInstance: NeatLikeWithObjectives,
): ObjectiveDescriptor[]
```

Collect the default objectives when fitness is not suppressed.

Parameters:
- `neatInstance` - - NEAT host exposing objective settings.

Returns: Default objective descriptors.

### collectUserObjectives

```ts
collectUserObjectives(
  neatInstance: NeatLikeWithObjectives,
): ObjectiveDescriptor[]
```

Collect valid user-registered objectives when multi-objective mode is enabled.

Parameters:
- `neatInstance` - - NEAT host exposing objective settings.

Returns: Valid user objective descriptors.

### ensureMultiObjectiveOptions

```ts
ensureMultiObjectiveOptions(
  neatInstance: NeatLikeWithObjectives,
): { enabled?: boolean | undefined; objectives?: ObjectiveDescriptor[] | undefined; }
```

Ensure the multi-objective options container exists.

Parameters:
- `neatInstance` - - NEAT host receiving the multi-objective container.

Returns: Initialized multi-objective options.

### ensureObjectivesList

```ts
ensureObjectivesList(
  multiObjectiveOptions: { enabled?: boolean | undefined; objectives?: ObjectiveDescriptor[] | undefined; },
): ObjectiveDescriptor[]
```

Ensure the objectives list exists on the multi-objective container.

Parameters:
- `multiObjectiveOptions` - - Multi-objective container to hydrate.

Returns: Objectives list ready for non-destructive operations.

### getObjectiveCandidates

```ts
getObjectiveCandidates(
  neatInstance: NeatLikeWithObjectives,
): ObjectiveDescriptor[]
```

Get the configured objective candidates.

Parameters:
- `neatInstance` - - NEAT host exposing objective settings.

Returns: Objective candidates from configuration.

### isMultiObjectiveEnabled

```ts
isMultiObjectiveEnabled(
  neatInstance: NeatLikeWithObjectives,
): boolean
```

Check whether multi-objective mode is enabled with a candidate list.

Parameters:
- `neatInstance` - - NEAT host exposing objective settings.

Returns: `true` when multi-objective mode is enabled and an objective list exists.

### isValidObjective

```ts
isValidObjective(
  candidateObjective: ObjectiveDescriptor | undefined,
): boolean
```

Validate that an objective descriptor has the required shape.

Parameters:
- `candidateObjective` - - Candidate descriptor to validate.

Returns: `true` when the descriptor can be used safely.

### replaceObjectiveByKey

```ts
replaceObjectiveByKey(
  objectivesList: ObjectiveDescriptor[],
  objectiveKey: string,
  objectiveDirection: "max" | "min",
  objectiveAccessor: (genome: GenomeLike) => number,
): ObjectiveDescriptor[]
```

Replace any existing objective with the same key and append the new descriptor.

Parameters:
- `objectivesList` - - Existing objectives to update.
- `objectiveKey` - - Key to replace.
- `objectiveDirection` - - Direction for the new objective.
- `objectiveAccessor` - - Accessor for the new objective.

Returns: Updated objectives list.
