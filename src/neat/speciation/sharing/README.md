# neat/speciation/sharing

Fitness-sharing and stagnation mechanics for speciation.

These helpers run after species are assigned. One normalizes scores within a
species so dense clusters do not dominate selection, and the other tracks
whether a species has stopped improving.

## neat/speciation/sharing/speciation.sharing.utils.ts

### applyFitnessSharing

```ts
applyFitnessSharing(
  speciationContext: FitnessSharingContext,
  sharingSigma: number,
): void
```

Apply fitness sharing to penalize similarity within species.

Parameters:
- `speciationContext` - - Neat instance context with species and distance function.
- `sharingSigma` - - Sharing radius used for distance weighting.

Returns: Nothing.

### updateSpeciesStagnation

```ts
updateSpeciesStagnation(
  speciationContext: StagnationContext,
  stagnationWindow: number,
  sortSpeciesMembers: (species: SpeciesLike) => void,
): void
```

Update stagnation counters and prune stagnant species.

Parameters:
- `speciationContext` - - Neat instance context with species array and generation counter.
- `stagnationWindow` - - Allowed stagnation window.
- `sortSpeciesMembers` - - Sort function for species members.

Returns: Nothing.
