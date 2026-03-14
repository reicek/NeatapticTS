# neat/species/core

Species-history augmentation mechanics used by NEAT reporting helpers.

This chapter holds the opt-in extended-history backfill logic that derives
innovation-range and enabled-ratio summaries from the current species state.

## neat/species/core/species.core.ts

### backfillExtendedHistory

```ts
backfillExtendedHistory(
  history: SpeciesHistoryEntry[],
  context: { _species?: SpeciesLike[] | undefined; _fallbackInnov?: ((connection: ConnectionLike) => number) | undefined; },
): void
```

Backfill missing extended history fields in place.

Parameters:
- `history` - - Recorded species history to enrich.
- `context` - - NEAT context exposing current species and optional fallback innovations.

Returns: Nothing. The history entries are mutated in place when backfill succeeds.

### shouldAugmentExtendedHistory

```ts
shouldAugmentExtendedHistory(
  options: NeatOptions | undefined,
): boolean
```

Check whether extended species history should be augmented.

Parameters:
- `options` - - Current NEAT options.

Returns: `true` when extended history is enabled.

### SPECIES_HISTORY_DEFAULT_ENABLED_RATIO

Default enabled ratio when no connections exist.

### SPECIES_HISTORY_DEFAULT_INNOVATION_ID

Default innovation id when none is present.

### SPECIES_HISTORY_DEFAULT_INNOVATION_RANGE

Default innovation range when data is missing.

### SPECIES_HISTORY_INITIAL_MAX_INNOVATION

Initial max tracker for innovation range aggregation.

### SPECIES_HISTORY_INITIAL_MIN_INNOVATION

Initial min tracker for innovation range aggregation.

### SPECIES_HISTORY_ZERO

Shared zero value for counters and defaults.
