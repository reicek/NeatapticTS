# neat/species/history/read

## neat/species/history/read/species.history.read.ts

### getSpeciesHistory

```ts
getSpeciesHistory(
  host: NeatLike,
): SpeciesHistoryEntry[]
```

Read the recorded species history for a NEAT host.

This history-read chapter keeps the public species facade thin by owning the
only remaining orchestration in the history path: resolve the stored history
context, optionally backfill extended metrics, then return the normalized
history buffer.

Parameters:
- `host` - - NEAT host exposing species history, species records, fallback innovation logic, and options.

Returns: Generation-stamped species history snapshots.

Example:

```ts
const historyEntries = getSpeciesHistory(neat);
console.log(historyEntries.at(-1));
```
