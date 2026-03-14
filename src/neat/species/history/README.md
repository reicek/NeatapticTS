# neat/species/history

Default slice size when exporting species history as JSONL.

## neat/species/history/species.history.ts

### exportSpeciesHistoryJsonl

```ts
exportSpeciesHistoryJsonl(
  speciesHistory: unknown[],
  maxEntries: number,
): string
```

Export species history records as JSON Lines.

Parameters:
- `speciesHistory` - - Recorded species history entries to serialize.
- `maxEntries` - - Maximum number of recent entries to include.

Returns: JSONL payload containing the requested recent history slice.

### SPECIES_HISTORY_JSONL_MAX_DEFAULT

Default slice size when exporting species history as JSONL.
