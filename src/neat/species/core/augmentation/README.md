# neat/species/core/augmentation

Concrete extended-history augmentation helpers for the species core chapter.

The root `species.core.ts` file decides whether augmentation should run at
all. This companion file owns the narrower mechanics that actually walk the
history rows, find matching live species records, and copy derived structural
summaries into entries that are still missing them.

The implementation deliberately stays conservative:
- it enriches only rows that lack the extended fields,
- it skips history rows whose species no longer exist in the live registry,
- it derives only compact summary values instead of copying full genomes or
  connection lists into history.

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

The helper walks recorded generations first and species rows second. For each
row, it performs a small three-step decision:
1. skip rows that already contain the extended metrics,
2. look up the matching live species by id,
3. if that species still exists and has members, summarize its innovation
   coverage and copy the result into the history row.

This makes the backfill useful for read-time teaching and diagnostics without
pretending that old history can always be reconstructed perfectly from the
current runtime state.

Parameters:
- `history` - Recorded species history to enrich.
- `context` - NEAT context exposing current species and optional fallback innovations.

Returns: Nothing. The history entries are mutated in place when backfill succeeds.

### SpeciesHistoryBackfillContext

Runtime surface required to backfill extended species-history fields.

The augmentation chapter only needs the current live species registry plus an
optional innovation fallback for legacy connection records.

That narrowness is intentional. The backfill path is a read-side enrichment
step, not a second controller facade, so it should depend only on the live
species evidence and the small fallback hook needed to interpret older
connection records.
