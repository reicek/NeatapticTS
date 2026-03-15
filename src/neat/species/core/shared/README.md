# neat/species/core/shared

## neat/species/core/shared/species.core.shared.ts

### SpeciesConnectionSummary

Aggregated innovation coverage for the genomes currently assigned to one species.

The history backfill path uses this compact shape to answer two questions that
are useful in telemetry dashboards:

- how wide the inherited innovation span is across the species members,
- how many of those structural genes are still enabled.

### summarizeSpeciesConnections

```ts
summarizeSpeciesConnections(
  members: GenomeDetailed[],
  fallbackInnov: ((connection: ConnectionLike) => number) | undefined,
): SpeciesConnectionSummary
```

Summarize innovation spread and enabled-connection ratio for one species.

This helper stays separate from the history backfill orchestration so the
arithmetic can be reused and documented independently from the policy that
decides when augmentation should happen.

Parameters:
- `members` - - Detailed member genomes for a single species.
- `fallbackInnov` - - Optional innovation resolver for legacy connections that do not carry a direct innovation id.

Returns: Compact innovation-range and enabled-ratio telemetry for the species.

Example:

```ts
const summary = summarizeSpeciesConnections(species.members, neat._fallbackInnov);
console.log(summary.innovationRange, summary.enabledRatio);
```
