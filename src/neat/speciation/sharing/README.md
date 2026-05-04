# neat/speciation/sharing

Fitness-sharing and stagnation mechanics for speciation.

This chapter owns the two post-assignment adjustments that make a species
registry useful over time rather than merely descriptive for one generation.
After assignment has grouped genomes and threshold tuning has updated the
future boundary, these helpers answer two follow-up questions:

1. should crowded species keep all of their raw score advantage,
2. has a species stopped improving long enough to be removed from the run.

Read the file in two halves. {@link applyFitnessSharing} reshapes scores so a
dense cluster of very similar genomes does not overwhelm selection simply by
volume. {@link updateSpeciesStagnation} then tracks whether each species is
still producing better members and prunes lineages that have gone stale.

Those concerns live together on purpose. Sharing answers the short-horizon
pressure question, "how much advantage should this species keep right now?"
Stagnation answers the longer-horizon survival question, "has this species
earned another generation in the run?" Together they form the post-
assignment pressure layer beneath later selection and maintenance steps.

Both helpers are deliberate controller-policy overlays. Fitness sharing
rewrites the current generation's score view within each species, and
stagnation updates controller-owned species bookkeeping such as `bestScore`
and `lastImproved`. Neither helper changes compatibility distance or the
structural identity used to form species.

Read the exported helpers in this order:

1. {@link applyFitnessSharing} adjusts member scores inside each already-
   assigned species,
2. {@link updateSpeciesStagnation} re-reads those species deterministically,
   refreshes progress bookkeeping, and prunes stale lineages.

The boundary stays intentionally narrow. These helpers do not decide species
membership, adapt compatibility thresholds, or write history rows. They take
the current species registry as given and adjust how that registry affects
later selection pressure and long-run survival.

```mermaid
flowchart TD
  Assigned[Assigned species registry]
  Sharing[Normalize scores within each species]
  Ranked[Species members re-read with shared scores]
  Stagnation[Update best-score progress and prune stale species]
  Output[Species registry ready for later controller phases]

  Assigned --> Sharing
  Sharing --> Ranked
  Ranked --> Stagnation
  Stagnation --> Output
```

## neat/speciation/sharing/speciation.sharing.utils.ts

### applyFitnessSharing

```ts
applyFitnessSharing(
  speciationContext: FitnessSharingContext,
  sharingSigma: number,
): void
```

Apply fitness sharing to penalize similarity within species.

Fitness sharing lowers the effective score of genomes that sit inside a dense
neighborhood of similar peers. That keeps one crowded species from dominating
later parent selection purely because many near-duplicates all retained their
full raw score.

The helper supports two modes:
- sigma-aware sharing, which weights neighbors by compatibility distance when
  the sharing radius is positive,
- uniform sharing, which falls back to dividing each member's score by the
  species size when no positive radius is configured.

The choice between those modes is really a choice about how much geometry the
controller wants from the current species. Sigma-aware sharing says nearby
neighbors should count more than distant ones inside the same species,
producing a softer gradient of pressure. Uniform sharing says the controller
only needs a simple crowding penalty, so every member of the species carries
the same share burden regardless of fine-grained distance.

Read the score writes here as shared-fitness overlays for this generation's
later selection and allocation steps, not as a replacement for the raw
evaluation evidence that assignment started from.

Parameters:
- `speciationContext` - Neat instance context with species and distance function.
- `sharingSigma` - Sharing radius used for distance weighting.

Returns: Nothing.

Example:

```ts
applyFitnessSharing(neat, 0);
applyFitnessSharing(neat, 3);
```

### updateSpeciesStagnation

```ts
updateSpeciesStagnation(
  speciationContext: StagnationContext,
  stagnationWindow: number,
  sortSpeciesMembers: (species: SpeciesLike) => void,
): void
```

Update stagnation counters and prune stagnant species.

This is the long-run maintenance half of post-assignment speciation. It first
sorts each species so the "best current member" read is deterministic, then
refreshes best-score and last-improved bookkeeping, and finally removes
species whose improvement gap has exceeded the allowed stagnation window.

Read this as the answer to "is this species still earning its place in the
population?" Species that keep improving remain eligible for future rounds;
species that stop improving long enough are pruned from the live registry.

Sorting first is not a cosmetic step. It makes the chapter's survival rule
deterministic: the same scored roster yields the same best-member read, which
means best-score updates and pruning decisions do not drift with incidental
member ordering.

The resulting `bestScore` and `lastImproved` values belong to species-side
controller bookkeeping, not to the canonical genome contract.

Parameters:
- `speciationContext` - Neat instance context with species array and generation counter.
- `stagnationWindow` - Allowed stagnation window.
- `sortSpeciesMembers` - Sort function for species members.

Returns: Nothing.

Example:

```ts
updateSpeciesStagnation(neat, 15, neat._sortSpeciesMembers.bind(neat));
```
