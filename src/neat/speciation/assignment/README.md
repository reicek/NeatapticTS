# neat/speciation/assignment

Speciation assignment mechanics.

This chapter covers the part of speciation that decides where genomes go:
snapshot the old memberships, clear the species, match genomes against
representatives, create new species when needed, and refresh representatives
once reassignment is complete.

## neat/speciation/assignment/speciation.assignment.utils.ts

### assignPopulationToSpecies

```ts
assignPopulationToSpecies(
  speciationContext: SpeciationHarnessContext<TOptions>,
  options: TOptions,
): void
```

Assign each genome in the population to a compatible species.

Parameters:
- `speciationContext` - - Speciation harness context.
- `options` - - Speciation options.

Returns: Nothing.

### createSpeciesForGenome

```ts
createSpeciesForGenome(
  speciationContext: SpeciationHarnessContext<TOptions>,
  genome: GenomeDetailed,
): void
```

Create a new species for the provided genome.

Parameters:
- `speciationContext` - - Speciation harness context.
- `genome` - - Genome that starts a new species.

Returns: Nothing.

### findCompatibleSpecies

```ts
findCompatibleSpecies(
  speciationContext: SpeciationHarnessContext<TOptions>,
  options: TOptions,
  genome: GenomeDetailed,
): SpeciesLike | undefined
```

Find a compatible species representative for the given genome.

Parameters:
- `speciationContext` - - Speciation harness context.
- `options` - - Speciation options.
- `genome` - - Genome to match.

Returns: Matching species or undefined.

### refreshSpeciesRepresentatives

```ts
refreshSpeciesRepresentatives(
  speciationContext: SpeciationHarnessContext<TOptions>,
): void
```

Refresh representatives and remove empty species.

Parameters:
- `speciationContext` - - Speciation harness context.

Returns: Nothing.

### resetSpeciesMembers

```ts
resetSpeciesMembers(
  speciationContext: SpeciationHarnessContext<TOptions>,
): void
```

Clear member lists for all species.

Parameters:
- `speciationContext` - - Speciation harness context.

Returns: Nothing.

### snapshotPreviousMembers

```ts
snapshotPreviousMembers(
  speciationContext: SpeciationHarnessContext<TOptions>,
): void
```

Snapshot current species memberships for telemetry.

Parameters:
- `speciationContext` - - Speciation harness context.

Returns: Nothing.
