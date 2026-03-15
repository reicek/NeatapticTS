# neat/evolve/population

## neat/evolve/population/evolve.population.utils.ts

### addOffspring

```ts
addOffspring(
  internal: NeatControllerForEvolution,
  nextPopulation: default[],
  helpers: { addSpeciatedOffspring: (nextPopulation: default[], remainingSlots: number) => Promise<void>; addUnspeciatedOffspring: (nextPopulation: default[], remainingSlots: number) => Promise<void>; },
): Promise<void>
```

Add offspring to fill remaining population slots.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Target population array.
- `helpers` - - Helper callbacks for offspring selection.
- `helpers` - - Speciated offspring helper.
- `helpers` - - Unspeciated offspring helper.

Returns: void.

### addSpeciatedOffspring

```ts
addSpeciatedOffspring(
  internal: NeatControllerForEvolution,
  nextPopulation: default[],
  remainingSlots: number,
  config: { minOffspringDefault: number; survivalThresholdDefault: number; youngThresholdDefault: number; youngMultiplierDefault: number; oldThresholdDefault: number; oldMultiplierDefault: number; crossSpeciesGuardLimit: number; },
): Promise<void>
```

Add offspring when speciation is enabled.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Target population array.
- `remainingSlots` - - Slots remaining to fill.
- `config` - - Offspring allocation constants.

Returns: void.

### addUnspeciatedOffspring

```ts
addUnspeciatedOffspring(
  internal: NeatControllerForEvolution,
  nextPopulation: default[],
  remainingSlots: number,
): Promise<void>
```

Add offspring when speciation is disabled.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Target population array.
- `remainingSlots` - - Slots remaining to fill.

Returns: void.

### applyElitism

```ts
applyElitism(
  internal: NeatControllerForEvolution,
  nextPopulation: default[],
): void
```

Apply elitism for the next generation.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Target population array.

Returns: void.

### applyProvenance

```ts
applyProvenance(
  internal: NeatControllerForEvolution,
  nextPopulation: default[],
): void
```

Add provenance genomes into the next population.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Target population array.

Returns: void.

### buildNextPopulation

```ts
buildNextPopulation(
  internal: NeatControllerForEvolution,
  helpers: { applyElitism: (nextPopulation: default[]) => void; applyProvenance: (nextPopulation: default[]) => void; addOffspring: (nextPopulation: default[]) => Promise<void>; },
): Promise<default[]>
```

Build the next population (elitism, provenance, offspring).

Parameters:
- `internal` - - NEAT controller instance.
- `helpers` - - Helper callbacks for population construction.
- `helpers` - - Elitism helper.
- `helpers` - - Provenance helper.
- `helpers` - - Offspring helper.

Returns: next population array.

### buildSpeciesOffspring

```ts
buildSpeciesOffspring(
  internal: NeatControllerForEvolution,
  survivors: GenomeWithMetadata[],
  speciesIndex: number,
  crossSpeciesProbability: number,
  crossSpeciesGuardLimit: number,
  survivalThresholdDefault: number,
): GenomeWithMetadata
```

Build a single offspring within a species.

Parameters:
- `internal` - - NEAT controller instance.
- `survivors` - - Survivors pool for selection.
- `speciesIndex` - - Species index.
- `crossSpeciesProbability` - - Cross-species mating probability.
- `crossSpeciesGuardLimit` - - Retry guard for cross-species selection.

Returns: offspring genome.

### computeOffspringAllocation

```ts
computeOffspringAllocation(
  internal: NeatControllerForEvolution,
  remainingSlots: number,
  config: { minOffspringDefault: number; youngThresholdDefault: number; youngMultiplierDefault: number; oldThresholdDefault: number; oldMultiplierDefault: number; },
): number[]
```

Compute offspring allocation per species.

Parameters:
- `internal` - - NEAT controller instance.
- `remainingSlots` - - Slots remaining to fill.
- `config` - - Allocation constants.

Returns: allocation per species index.

### distributeRemainingSlots

```ts
distributeRemainingSlots(
  allocation: number[],
  rawShares: number[],
  remainingSlots: number,
): void
```

Distribute leftover slots by fractional remainders.

Parameters:
- `allocation` - - Allocation array to adjust.
- `rawShares` - - Raw fractional shares.
- `remainingSlots` - - Total slots available.

Returns: void.

### enforceMinimumOffspring

```ts
enforceMinimumOffspring(
  internal: NeatControllerForEvolution,
  allocation: number[],
  remainingSlots: number,
  minOffspringDefault: number,
): void
```

Enforce minimum offspring per species when possible.

Parameters:
- `internal` - - NEAT controller instance.
- `allocation` - - Allocation array to adjust.
- `remainingSlots` - - Total slots available.
- `minOffspringDefault` - - Default minimum offspring.

Returns: void.

### enforcePopulationConstraints

```ts
enforcePopulationConstraints(
  internal: NeatControllerForEvolution,
  nextPopulation: default[],
): Promise<void>
```

Ensure new population meets structural constraints.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Population to validate.

Returns: void.

### selectSecondParent

```ts
selectSecondParent(
  internal: NeatControllerForEvolution,
  survivors: GenomeWithMetadata[],
  speciesIndex: number,
  crossSpeciesProbability: number,
  crossSpeciesGuardLimit: number,
  survivalThresholdDefault: number,
): GenomeWithMetadata
```

Select a second parent, optionally from another species.

Parameters:
- `internal` - - NEAT controller instance.
- `survivors` - - Survivors pool from the current species.
- `speciesIndex` - - Current species index.
- `crossSpeciesProbability` - - Probability to cross species.
- `crossSpeciesGuardLimit` - - Retry guard for cross-species selection.

Returns: chosen parent genome.

### trimOversubscription

```ts
trimOversubscription(
  internal: NeatControllerForEvolution,
  allocation: number[],
  remainingSlots: number,
  minOffspringDefault: number,
): void
```

Trim allocations when oversubscribed.

Parameters:
- `internal` - - NEAT controller instance.
- `allocation` - - Allocation array to adjust.
- `remainingSlots` - - Total slots available.
- `minOffspringDefault` - - Default minimum offspring.

Returns: void.
