# neat/multiobjective/archive

Compact Pareto archive helpers for the NEAT controller.

This chapter keeps the write-side archive mechanics together: when
multi-objective mode is enabled, it snapshots the leading Pareto fronts into
a small archive that downstream telemetry and visualization helpers can
inspect without retaining whole genomes.

## neat/multiobjective/archive/multiobjective.archive.ts

### archiveParetoFrontsIfEnabled

```ts
archiveParetoFrontsIfEnabled(
  neatInstance: NeatLikeWithMultiObjective,
  fronts: default[][],
): void
```

Archives a compact snapshot of the current Pareto fronts when
multi-objective mode is enabled.

This is intended for visualization/debugging:
- Stores only genome `_id` values, not full genomes.
- Keeps only the top `MAX_PARETO_ARCHIVE_FRONTS` fronts.
- Maintains a bounded archive by shifting the oldest entry.

Parameters:
- `neatInstance` - - Neat instance.
- `fronts` - - Pareto fronts to archive.

### MAX_PARETO_ARCHIVE_FRONTS

Maximum number of top Pareto fronts to retain per archive snapshot.

### MAX_PARETO_ARCHIVE_LENGTH

Maximum number of archive snapshots to retain.
