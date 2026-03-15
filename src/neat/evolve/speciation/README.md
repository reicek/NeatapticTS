# neat/evolve/speciation

## neat/evolve/speciation/evolve.speciation.utils.ts

### applyGlobalStagnationInjectionIfNeeded

```ts
applyGlobalStagnationInjectionIfNeeded(
  internal: NeatControllerForEvolution,
  helpers: { buildFreshGenomeForStagnation: () => Promise<GenomeWithMetadata>; replaceFraction: number; },
): Promise<void>
```

Apply global stagnation injection if configured.

Parameters:
- `internal` - - NEAT controller instance.
- `helpers` - - Helper callbacks for stagnation injection.
- `helpers` - - Genome builder for injection.

Returns: void.

### applySpeciationAndSharingIfEnabled

```ts
applySpeciationAndSharingIfEnabled(
  internal: NeatControllerForEvolution,
  helpers: { applyAutoCompatibilityTuning: () => void; recordSpeciesHistorySnapshot: () => void; },
): Promise<void>
```

Apply speciation, fitness sharing, and related side effects.

Parameters:
- `internal` - - NEAT controller instance.
- `helpers` - - Helper callbacks used for tuning and history.
- `helpers` - - Auto-compatibility adjustment helper.
- `helpers` - - Species history snapshot helper.

Returns: void.

### buildFreshGenomeForStagnation

```ts
buildFreshGenomeForStagnation(
  internal: NeatControllerForEvolution,
): Promise<GenomeWithMetadata>
```

Build a fresh genome for stagnation injection.

Parameters:
- `internal` - - NEAT controller instance.

Returns: new genome with minimum constraints.

### ensureHiddenNodeVariance

```ts
ensureHiddenNodeVariance(
  internal: NeatControllerForEvolution,
  genome: GenomeWithMetadata,
): Promise<void>
```

Ensure a minimal hidden-node variance in injected genomes.

Parameters:
- `internal` - - NEAT controller instance.
- `genome` - - Genome to adjust.

Returns: void.

### ensureSpeciesHistorySnapshot

```ts
ensureSpeciesHistorySnapshot(
  internal: NeatControllerForEvolution,
  maxHistory: number,
): void
```

Ensure a minimal species history snapshot exists for exports.

Parameters:
- `internal` - - NEAT controller instance.
- `maxHistory` - - Maximum history length.

Returns: void.

### recordSpeciesHistorySnapshot

```ts
recordSpeciesHistorySnapshot(
  internal: NeatControllerForEvolution,
  maxHistory: number,
): void
```

Record a species history snapshot when needed.

Parameters:
- `internal` - - NEAT controller instance.
- `maxHistory` - - Maximum history length.

Returns: void.

### updateSpeciesStagnationIfEnabled

```ts
updateSpeciesStagnationIfEnabled(
  internal: NeatControllerForEvolution,
): void
```

Update species stagnation status when speciation enabled.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.
