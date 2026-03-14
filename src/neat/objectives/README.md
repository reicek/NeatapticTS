# neat/objectives

Objective-management helpers for the NEAT controller.

The root objectives chapter keeps the public registration and resolution
methods small, while `core/` holds the validation and list-management logic.

- `core/` explains default fitness objectives, user objective filtering, and list replacement rules.

## neat/objectives/objectives.ts

### _getObjectives

```ts
_getObjectives(): ObjectiveDescriptor[]
```

Build and return the list of registered objectives for this NEAT instance.

Parameters:
- `this` - - NEAT host exposing multi-objective options and the cached objective list.

Returns: Objective descriptors in the order they should be applied.

### clearObjectives

```ts
clearObjectives(): void
```

Clear all registered multi-objectives.

Parameters:
- `this` - - NEAT host exposing multi-objective options and the cached objective list.

Returns: Nothing. Registered user objectives and the cached objective list are cleared.

### registerObjective

```ts
registerObjective(
  key: string,
  direction: "max" | "min",
  accessor: (genome: GenomeLike) => number,
): void
```

Register a new objective descriptor.

Parameters:
- `this` - - NEAT host exposing multi-objective options and the cached objective list.
- `key` - - Unique name for the objective.
- `direction` - - Whether the objective should be minimized or maximized.
- `accessor` - - Function that extracts a numeric value from a genome.

Returns: Nothing. The host objective configuration is updated in place.
