# neat/species/stats

Current-roster snapshot boundary for species reporting.

The root `species/` chapter offers two reporting questions: what species exist
right now, and how did that roster change across generations? This file owns
the first question only. It projects the live species registry into compact
summary rows that are cheap to log, render in dashboards, or inspect during a
run.

That narrow scope is intentional. Historical interpretation belongs to the
history subtree, and optional enrichment of old rows belongs to the species
core augmentation policy. This chapter stays on the present-tense read: take
the live registry, extract a few dashboard-friendly fields, and return a
detached snapshot.

Read this chapter in two steps:

1. `SpeciesStatsHost` shows the tiny host seam needed for a current-roster
   snapshot.
2. `getSpeciesStats()` shows how live species records are projected into
   compact reporting rows.

## neat/species/stats/species.stats.ts

### getSpeciesStats

```ts
getSpeciesStats(
  host: NeatLike,
): { id: number; size: number; bestScore: number; lastImproved: number; }[]
```

Get lightweight per-species statistics for the current population.

The returned records are intentionally compact and detached from the live
species objects, which makes them safer to log, serialize, or hand to UI
code without leaking mutable internal member arrays.

Use this helper when a caller wants the present-tense roster rather than the
longer story. Each returned row answers one quick operational question about a
species:

- `id` identifies which species the row describes,
- `size` reports how many genomes are currently assigned to it,
- `bestScore` reports the best score the species has reached so far,
- `lastImproved` reports the last generation where that species improved.

That makes this the right read path for dashboards, debug tables, and concise
logs where historical replay would be unnecessary overhead. The helper does
not preserve live references to member arrays; it emits detached scalar data
that can be inspected freely without mutating controller state.

The projection stays intentionally simple:

1. read the current species registry,
2. normalize missing live values to safe reporting defaults,
3. return one compact row per species.

Parameters:

- `host` - NEAT host exposing the internal species registry.

Returns: Compact per-species summaries suitable for reporting.

Examples:

```ts
const summaries = getSpeciesStats(neat);
console.table(summaries);

const activeSpeciesCount = summaries.filter(
  (summary) => summary.size > 0,
).length;
console.log(activeSpeciesCount);
```

```ts
const summaries = getSpeciesStats(neat);
console.log(summaries.map((species) => `${species.id}:${species.size}`));
```

### SpeciesStatsHost

Narrow host surface required by the species-stats chapter.

Stats reads only the current species registry, so this helper keeps that
dependency explicit instead of coupling the chapter to the broader history
and fallback-innovation surface used by the root boundary.

That small seam is the point of the chapter. A current-roster snapshot does
not need stored history, augmentation context, or export helpers. It only
needs access to the live species registry that the controller is already
maintaining for speciation.
