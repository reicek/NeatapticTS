# neat/species/history

Species-history export helpers for the NEAT controller.

This chapter keeps the JSONL export surface separate from the broader
species-reporting helpers so generated docs stay tightly scoped to one
export concern.

The root `species/` chapter explains when to read historical species data.
This narrower boundary explains how that history is serialized once a caller
decides it needs a portable export or a compact append-friendly artifact.

Read this chapter when you want to answer questions such as:
- Why does species history export live in its own tiny boundary?
- Why is JSONL a better fit here than one large JSON array for many tooling
  workflows?
- How does the export helper keep the history window bounded?
- Which part of the species story is preserved versus trimmed during export?

The mental model is intentionally simple:
1. take the most recent slice of species history,
2. serialize each history row independently,
3. join the rows into a newline-delimited stream.

That shape keeps exports easy to inspect, append, stream, and diff without
widening this chapter into the heavier history-reading and enrichment logic
owned elsewhere in the species subtree.

## neat/species/history/species.history.ts

### exportSpeciesHistoryJsonl

```ts
exportSpeciesHistoryJsonl(
  speciesHistory: unknown[],
  maxEntries: number,
): string
```

Export species history records as JSON Lines.

JSONL is a good fit for species-history export because each generation row
stays independently parseable. That makes the output convenient for log-like
storage, incremental processing, and quick inspection in tooling that prefers
one record per line.

Parameters:
- `speciesHistory` - - Recorded species history entries to serialize.
- `maxEntries` - - Maximum number of recent entries to include.

Returns: JSONL payload containing the requested recent history slice.

Example:

```ts
const jsonl = exportSpeciesHistoryJsonl(neat.getSpeciesHistory(), 50);
console.log(jsonl.split('\n').length);
```

### SPECIES_HISTORY_JSONL_MAX_DEFAULT

Default slice size when exporting species history as JSONL.

The default keeps history exports large enough to explain recent species
trends without forcing every export to serialize the entire retained buffer.
