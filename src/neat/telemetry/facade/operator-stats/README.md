# neat/telemetry/facade/operator-stats

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

### TelemetryFacadeOperatorStatsHost

Narrow telemetry-facade host surface required by the operator-stats chapter.

This chapter isolates the mutation-operator summary read path so adaptive
scheduling diagnostics stay separate from novelty maintenance and archive
inspection in the root telemetry facade.
