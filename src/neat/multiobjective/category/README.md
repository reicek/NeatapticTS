# neat/multiobjective/category

## neat/multiobjective/category/multiobjective.category.ts

### adaptDominanceEpsilon

```ts
adaptDominanceEpsilon(
  internal: NeatControllerForEvolution,
  paretoFronts: GenomeWithMetadata[][],
  config: { targetFrontMin: number; targetFrontUpperRatio: number; targetFrontLowerRatio: number; defaultEpsilonAdjust: number; defaultEpsilonMin: number; defaultEpsilonMax: number; defaultEpsilonCooldown: number; },
): void
```

Adapt dominance epsilon based on Pareto front size.

Parameters:
- `internal` - - NEAT controller instance.
- `paretoFronts` - - Non-dominated fronts.
- `config` - - Epsilon tuning constants.

Returns: void.

### computeCrowdingDistances

```ts
computeCrowdingDistances(
  internal: NeatControllerForEvolution,
  populationSnapshot: GenomeWithMetadata[],
  paretoFronts: GenomeWithMetadata[][],
  objectives: ObjectiveDescriptor[],
): number[]
```

Compute crowding distances for multi-objective fronts.

Parameters:
- `internal` - - NEAT controller instance.
- `populationSnapshot` - - Current population reference.
- `paretoFronts` - - Non-dominated fronts.
- `objectives` - - Active objectives.

Returns: crowding distances aligned with population order.

### processMultiObjective

```ts
processMultiObjective(
  internal: NeatControllerForEvolution,
  config: { paretoArchiveMax: number; targetFrontMin: number; targetFrontUpperRatio: number; targetFrontLowerRatio: number; defaultEpsilonAdjust: number; defaultEpsilonMin: number; defaultEpsilonMax: number; defaultEpsilonCooldown: number; pruneWindowDefault: number; pruneRangeEpsDefault: number; },
): void
```

Apply the multi-objective evolution policy for the current generation.

This is the bridge between the generic Pareto-ranking helpers and the larger
evolve loop. It runs the ranking pass, reorders the live population by
`(rank, crowding)`, snapshots the best fronts for telemetry, optionally
adjusts dominance epsilon to keep the frontier size useful, and prunes
objectives that have gone flat for long enough to stop influencing search.

Parameters:
- `internal` - - NEAT controller instance.
- `config` - - Multi-objective tuning constants.

Returns: Nothing. The controller is updated in place.

### pruneInactiveObjectives

```ts
pruneInactiveObjectives(
  internal: NeatControllerForEvolution,
  config: { pruneWindowDefault: number; pruneRangeEpsDefault: number; },
): void
```

Prune objectives that have collapsed ranges over a window.

Parameters:
- `internal` - - NEAT controller instance.
- `config` - - Pruning constants.

Returns: void.

### recordParetoArchives

```ts
recordParetoArchives(
  internal: NeatControllerForEvolution,
  paretoFronts: GenomeWithMetadata[][],
  objectives: ObjectiveDescriptor[],
  archiveMax: number,
): void
```

Record Pareto front archives for telemetry.

Parameters:
- `internal` - - NEAT controller instance.
- `paretoFronts` - - Non-dominated fronts.
- `objectives` - - Active objectives.
- `archiveMax` - - Maximum archive size.

Returns: void.

### sortPopulationByPareto

```ts
sortPopulationByPareto(
  internal: NeatControllerForEvolution,
  populationSnapshot: GenomeWithMetadata[],
  crowdingDistances: number[],
): void
```

Sort population by Pareto rank and crowding distance.

Parameters:
- `internal` - - NEAT controller instance.
- `populationSnapshot` - - Current population reference.
- `crowdingDistances` - - Crowding distances aligned with population order.

Returns: void.
