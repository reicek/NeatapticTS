# neat/species/stats

Species-summary projection helpers for the NEAT controller.

This chapter owns the lightweight, read-only view used by dashboards, logs,
and quick diagnostics when callers only need the current species roster.
Keeping that projection here lets the root `species.ts` file focus on
orchestrating the broader reporting surface.

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

Parameters:
- `host` - - NEAT host exposing the internal species registry.

Returns: Compact per-species summaries suitable for reporting.

Example:

```ts
const summaries = getSpeciesStats(neat);
console.log(summaries.map((species) => `${species.id}:${species.size}`));
```

### SpeciesStatsHost

Narrow host surface required by the species-stats chapter.

Stats reads only the current species registry, so this helper keeps that
dependency explicit instead of coupling the chapter to the broader history
and fallback-innovation surface used by the root boundary.
