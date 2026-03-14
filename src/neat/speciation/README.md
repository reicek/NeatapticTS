# neat/speciation

Assign genomes into species based on compatibility distance and maintain species structures.

This root entrypoint stays orchestration-first so the generated README reads
like the speciation lifecycle: snapshot the previous state, reassign the
population, tune the threshold, refresh representatives, apply optional
protection, and capture history. The detailed mechanics now live in focused
teaching folders for assignment, threshold control, history, and sharing.

## neat/speciation/speciation.ts

### _applyFitnessSharing

```ts
_applyFitnessSharing(): void
```

Apply fitness sharing to penalize similarity within species.

Parameters:
- `this` - - Neat instance context with species array and compatibility distance function.

### _sortSpeciesMembers

```ts
_sortSpeciesMembers(
  species: SpeciesLike,
): void
```

Sort species members by descending score.

Parameters:
- `species` - - Species to sort.

### _speciate

```ts
_speciate(): void
```

Assign genomes into species based on compatibility distance.

Parameters:
- `this` - - Speciation harness context.

Returns: Nothing.

### _updateSpeciesStagnation

```ts
_updateSpeciesStagnation(): void
```

Update stagnation counters for all species.

Parameters:
- `this` - - Neat instance context with species array and generation counter.
