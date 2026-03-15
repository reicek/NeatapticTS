# neat/species/core

Species-history augmentation mechanics used by NEAT reporting helpers.

This chapter holds the opt-in extended-history backfill logic that derives
innovation-range and enabled-ratio summaries from the current species state.

## neat/species/core/species.core.ts

### backfillExtendedHistory

```ts
backfillExtendedHistory(
  history: SpeciesHistoryEntry[],
  context: SpeciesHistoryBackfillContext,
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
