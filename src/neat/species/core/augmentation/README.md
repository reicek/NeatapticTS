# neat/species/core/augmentation

## neat/species/core/augmentation/species.core.augmentation.ts

### backfillExtendedHistoryEntries

```ts
backfillExtendedHistoryEntries(
  history: SpeciesHistoryEntry[],
  context: SpeciesHistoryBackfillContext,
): void
```

Backfill missing extended history fields in place.

This chapter owns the concrete augmentation workflow so the root
`species.core.ts` file can stay focused on policy decisions such as whether
the backfill should run at all.

Parameters:
- `history` - - Recorded species history to enrich.
- `context` - - NEAT context exposing current species and optional fallback innovations.

Returns: Nothing. The history entries are mutated in place when backfill succeeds.

### SpeciesHistoryBackfillContext

Runtime surface required to backfill extended species-history fields.

The augmentation chapter only needs the current live species registry plus an
optional innovation fallback for legacy connection records.
