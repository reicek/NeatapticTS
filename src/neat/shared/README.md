# neat/shared

Shared structural contracts used across the NEAT controller chapters.

This shared chapter keeps the light-weight, controller-facing type surface in
one direct-path location so speciation, telemetry, objectives, species, and
tests can import the same contracts without depending on the heavier root
`Neat` implementation.

## neat/shared/neat.shared.types.ts

### AnyObj

Generic map type used as a stop‑gap where the precise shape is still in flux.
Prefer a specific interface once the surface stabilises.

### ComplexityMetrics

Aggregate structural complexity metrics capturing size & growth pressure.

### ConnectionLike

Lightweight connection representation used by telemetry and structural helpers.

### DiversityStats

Diversity statistics captured each generation. Individual fields may be
omitted in telemetry output if diversity tracking is partially disabled to
reduce runtime cost.

### GenomeDetailed

More concrete genome surface used by telemetry and lineage helpers.
Extends the minimal `GenomeLike` with node/connection shapes and a few
internal bookkeeping fields used by telemetry.

### GenomeLike

Minimal genome structural surface used by several helpers (incrementally expanded).

NOTE: `nodes` and `connections` remain intentionally structural/opaque
until a stable public abstraction is finalised.

### LineageSnapshot

Snapshot of lineage & ancestry statistics for the current generation.

### NeatLike

Minimal surface every helper currently expects from a NEAT instance while
extraction continues. Kept intentionally loose; prefer concrete fields
when helpers are stabilised. Represented as a simple record to avoid an
empty interface that duplicates its supertype.

### NeatOptions

Options subset used by telemetry helpers. Kept narrow to avoid leaking
full runtime options into the helper type surface.

### NodeLike

Lightweight node representation used by telemetry and structural helpers.

### ObjAges

Map of objective key to age in generations since introduction.

### ObjectiveDescriptor

Descriptor for a single optimisation objective (single or multi‑objective runs).

Examples:

Add a maximisation objective for accuracy
```ts
const accuracyObj: ObjectiveDescriptor = {
  key: 'accuracy',
  direction: 'max',
  accessor: g => g.score ?? 0
};
```

Add a minimisation objective for network complexity
```ts
const complexityObj: ObjectiveDescriptor = {
  key: 'complexity',
  direction: 'min',
  accessor: g => (g.nodes.length + g.connections.length)
};
```

### ObjectiveEvent

Objective add/remove lifecycle event for telemetry and auditing.

### ObjEvent

Dynamic objective lifecycle event (addition or removal).

**Deprecated:** Use `ObjectiveEvent` instead.

### ObjImportance

Map of objective key to its importance metrics (range / variance).

### ObjImportanceEntry

Contribution / dispersion metrics for an objective over a recent window.
Used to gauge whether an objective meaningfully influences selection.

### OperatorStat

Per-generation statistic for a genetic operator.

Success is operator‑specific (e.g. produced a structurally valid mutation).
A high attempt count with low success can indicate constraints becoming tight
(e.g. structural budgets reached) – useful for adaptive operator scheduling.

### OperatorStatsRecord

Aggregated success / attempt counters over a window or entire run.

### ParetoArchiveEntry

Pareto archive entry capturing a genome plus its objective values.

### PerformanceMetrics

Timing metrics for coarse evolutionary phases (milliseconds).

### SpeciationHarnessContext

Minimal runtime surface required by speciation helpers.
Tests and harnesses can narrow the options type via the generic parameter.

### SpeciationOptions

Speciation options for the NEAT speciation controller.

Extends {@link NeatOptions} with speciation-specific configuration used by:
- Compatibility-threshold based species assignment
- Adaptive threshold controllers (PID-like)
- Species allocation telemetry (history snapshots)

### SpeciesAlloc

Offspring allocation for a species during reproduction.

### SpeciesHistoryEntry

Species statistics captured for a particular generation.

### SpeciesHistoryStat

Species statistics at a single historical snapshot (generation boundary).

### SpeciesHistoryStatExtended

Extended per-species historical snapshot with optional backfilled metrics
that may be computed lazily (innovationRange, enabledRatio).

### SpeciesLastStats

Rolling statistics tracked for each species between generations.
These values inform stagnation heuristics and adaptive controllers.

### SpeciesLike

Internal species representation used by helpers. Kept minimal and structural.

### TelemetryEntry

Telemetry summary for one generation.

Optional properties are feature‑dependent; consumers MUST test for presence.

Example:

```ts
function logSummary(t: TelemetryEntry) {
  console.log(`Gen ${t.gen} best=${t.best.toFixed(4)} species=${t.species}`);
  if (t.diversity) console.log('Mean compat', t.diversity.meanCompat);
}
```
