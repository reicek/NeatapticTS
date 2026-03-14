# neat/speciation/history

History and telemetry mechanics for speciation.

This chapter captures what happened after speciation ran: age-based species
protection, per-generation history snapshots, and the extended innovation
statistics used by teaching and telemetry surfaces.

## neat/speciation/history/speciation.history.utils.ts

### applyAgeProtection

```ts
applyAgeProtection(
  speciationContext: SpeciationHarnessContext<TOptions>,
  options: TOptions,
): void
```

Apply age protection penalties to old species.

Parameters:
- `speciationContext` - - Speciation harness context.
- `options` - - Speciation options.

Returns: Nothing.

### averageNumbers

```ts
averageNumbers(
  values: number[],
): number
```

Average a list of numbers, returning zero when empty.

Parameters:
- `values` - - Numeric values to average.

Returns: Mean of the values or zero.

### buildExtendedHistoryStats

```ts
buildExtendedHistoryStats(
  speciationContext: SpeciationHarnessContext<TOptions>,
  species: SpeciesLike,
): Record<string, unknown>
```

Build extended history stats for a species.

Parameters:
- `speciationContext` - - Speciation harness context.
- `species` - - Species to snapshot.

Returns: Extended history entry.

### recordHistory

```ts
recordHistory(
  speciationContext: SpeciationHarnessContext<TOptions>,
  options: TOptions,
): void
```

Record the current species history snapshot.

Parameters:
- `speciationContext` - - Speciation harness context.
- `options` - - Speciation options.

Returns: Nothing.

### summarizeInnovations

```ts
summarizeInnovations(
  speciationContext: SpeciationHarnessContext<TOptions>,
  members: GenomeDetailed[],
): { meanInnovation: number; innovationRange: number; enabledRatio: number; }
```

Summarize innovation statistics for a set of members.

Parameters:
- `speciationContext` - - Speciation harness context.
- `members` - - Members to summarize.

Returns: Innovation summary statistics.

### trimHistory

```ts
trimHistory(
  speciationContext: SpeciationHarnessContext<TOptions>,
): void
```

Trim species history to the maximum buffer size.

Parameters:
- `speciationContext` - - Speciation harness context.

Returns: Nothing.
