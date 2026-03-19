# neat/telemetry/facade/operator-stats

Mutation-operator diagnostics inside the public telemetry facade.

This chapter is the read-side summary of how often mutation operators were
tried and how often they succeeded. That makes it the natural inspection
surface for adaptive-mutation tuning, debugging operator imbalance, or simply
understanding which structural edits are doing useful work during a run.

The boundary is intentionally narrow: one host contract carrying the recorded
operator counters and one accessor that projects those counters into a stable
dashboard-friendly summary.

Read this chapter after the root telemetry facade when the open question is
"which operators are being used successfully?" rather than "what happened in
the telemetry buffer?" or "what fronts are currently on the Pareto archive?"

```mermaid
flowchart TD
  Stats[Recorded operator counters] --> Summary[getOperatorStats()<br/>name success attempts]
  Summary --> Inspect[Adaptive tuning and diagnostics]
```

## neat/telemetry/facade/operator-stats/telemetry.facade.operator-stats.ts

### getOperatorStats

```ts
getOperatorStats(
  host: TelemetryFacadeOperatorStatsHost,
): { name: string; success: number; attempts: number; }[]
```

Return aggregated mutation/operator statistics.

Keeping this accessor in its own chapter makes the telemetry facade easier
to scan: archive helpers answer multi-objective questions, novelty helpers
manage behavior descriptors, and this chapter explains operator activity.

Parameters:
- `host` - - `Neat` instance recording operator attempts and successes.

Returns: Operator summaries suitable for dashboards and debugging.

Example:

```ts
const operatorStats = getOperatorStats(neat);
console.table(operatorStats);
```

### TelemetryFacadeOperatorStatsHost

Narrow telemetry-facade host surface required by the operator-stats chapter.

This chapter isolates the mutation-operator summary read path so adaptive
scheduling diagnostics stay separate from novelty maintenance and archive
inspection in the root telemetry facade.

Only the operator statistics map is needed because this subchapter is purely
about summarizing recorded attempts and successes, not mutating controller
policy directly.
