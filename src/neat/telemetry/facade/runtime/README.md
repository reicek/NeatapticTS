# neat/telemetry/facade/runtime

## neat/telemetry/facade/runtime/telemetry.facade.runtime.ts

### getDiversityStats

```ts
getDiversityStats(
  host: TelemetryFacadeRuntimeHost,
): DiversityStats
```

Return cached diversity metrics, computing a fallback snapshot when needed.

This keeps the public facade resilient: callers can always ask for diversity
stats even before a full metrics pass has run.

Parameters:
- `host` - - `Neat` instance exposing cached diversity state.

Returns: Diversity metrics for the current population.

### getPerformanceStats

```ts
getPerformanceStats(
  host: TelemetryFacadeRuntimeHost,
): { lastEvalMs: number | undefined; lastEvolveMs: number | undefined; }
```

Return coarse timing metrics for the last evaluation and evolution passes.

Keeping this beside the diversity snapshot helper makes the runtime chapter a
compact place to inspect the latest controller-health signals without mixing
them with lineage, species, or archive reads.

Parameters:
- `host` - - `Neat` instance tracking performance timings.

Returns: Snapshot of the last evaluation and evolution durations.

### TelemetryFacadeRuntimeHost

Narrow telemetry-facade host surface required by the runtime-metrics chapter.

This chapter groups the two read paths that summarize recent runtime state:
coarse timing snapshots and the cached diversity view used by diagnostics and
telemetry dashboards.
