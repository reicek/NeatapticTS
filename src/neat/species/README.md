# neat/species

Species-reporting helpers for the NEAT controller.

The root species chapter keeps the public reporting flow small: one path
returns the current live species summaries, and the other returns the
recorded cross-generation history.

- `stats/` projects the live species registry into compact reporting rows.
- `core/` explains when extended species history is backfilled and how innovation-range summaries are derived.
- `history/` keeps the JSONL export surface scoped to one serialization concern.

## neat/species/species.ts

### getSpeciesHistory

```ts
getSpeciesHistory(): SpeciesHistoryEntry[]
```

Retrieve the recorded species history across generations.

Parameters:
- `this` - - NEAT host exposing species history, species records, fallback innovation logic, and options.

Returns: Generation-stamped species history snapshots.

### getSpeciesStats

```ts
getSpeciesStats(): { id: number; size: number; bestScore: number; lastImproved: number; }[]
```

Get lightweight per-species statistics for the current population.

This is the shortest read path into the species system. It is useful when a
caller needs dashboard-friendly snapshots such as current species sizes,
recent improvement timestamps, or the best score per species without pulling
the heavier generation-by-generation history buffer.

Parameters:
- `this` - - NEAT host exposing the internal species registry.

Returns: Compact per-species summaries suitable for reporting.

Example:

```ts
const speciesSummaries = neat.getSpeciesStats();
console.table(speciesSummaries);
```
