# neat/telemetry/recorder

Telemetry recorder orchestration for generation snapshots.

This chapter is the write-heavy counterpart to the read-oriented telemetry
facade. The neighboring `runtime/` and `metrics/` chapters own the small
mechanics and computation helpers, while this recorder chapter keeps the
end-to-end flow readable: shape a generation snapshot, optionally filter it,
and persist or stream it without destabilizing evolution.

## neat/telemetry/recorder/telemetry.recorder.ts

### applyTelemetrySelect

```ts
applyTelemetrySelect(
  entry: Record<string, unknown>,
): Record<string, unknown>
```

Apply a telemetry selection whitelist to a telemetry entry.

This helper inspects a per-instance Set of telemetry keys stored at
`this._telemetrySelect`. If present, only keys included in the set are
retained on the produced entry. Core fields (generation, best score and
species count) are always preserved.

Example:

Parameters:
- `entry` - - Raw telemetry object to be filtered in-place.

Returns: The filtered telemetry object (same reference as input).

Example:

// keep only 'gen', 'best', 'species' and 'diversity' fields
neat._telemetrySelect = new Set(['diversity']);
applyTelemetrySelect.call(neat, entry);

### buildTelemetryEntry

```ts
buildTelemetryEntry(
  fittest: Record<string, unknown>,
): TelemetryEntry
```

Build a comprehensive telemetry entry for the current generation.

The returned object contains a snapshot of population statistics, multi-
objective front sizes, operator statistics, lineage summaries and optional
complexity/performance metrics depending on configured telemetry options.

This function intentionally mirrors the legacy in-loop telemetry construction
to preserve behavior relied upon by tests and consumers.

Example:

Parameters:
- `fittest` - - The currently fittest genome (used to report `best` score).

Returns: A TelemetryEntry object suitable for recording/streaming.

Example:

// build a telemetry snapshot for the current generation
const snapshot = neat.buildTelemetryEntry(neat.population[0]);
neat.recordTelemetryEntry(snapshot);

### computeDiversityStats

```ts
computeDiversityStats(): void
```

Compute several diversity statistics used by telemetry reporting.

This helper is intentionally conservative in runtime: when `fastMode` is enabled it will automatically tune a few sampling defaults to keep the computation cheap. The computed statistics are written to `this._diversityStats` as an object with keys like `meanCompat` and `graphletEntropy`.

Example:

// compute and store diversity stats onto the neat instance
neat.options.diversityMetrics = { enabled: true };
neat.computeDiversityStats();
console.log(neat._diversityStats.meanCompat);

### createTelemetryEntryBase

```ts
createTelemetryEntryBase(
  generationIndex: number,
  bestScore: number,
  speciesCount: number,
): TelemetryEntry
```

Create a strict baseline telemetry entry with required fields populated.

This helper centralizes defaults so downstream telemetry producers can
extend the entry while keeping the strict `TelemetryEntry` contract.

Parameters:
- `generationIndex` - Generation index for the telemetry snapshot.
- `bestScore` - Best fitness value observed in the generation.
- `speciesCount` - Number of extant species.

Returns: A strict telemetry entry with required fields populated.

### recordTelemetryEntry

```ts
recordTelemetryEntry(
  entry: TelemetryEntry,
): void
```

Record a telemetry entry into the instance buffer and optionally stream it.

Steps:
This method performs the following steps to persist and optionally stream telemetry:
1. Apply `applyTelemetrySelect` to filter fields according to user selection.
2. Ensure `this._telemetry` buffer exists and push the entry.
3. If a telemetry stream callback is configured, call it.
4. Trim the buffer to a conservative max size (500 entries).

Example:

Parameters:
- `entry` - - Telemetry entry to record.

Example:

// record a simple telemetry entry from inside the evolve loop
neat.recordTelemetryEntry({ gen: neat.generation, best: neat.population[0].score });

### structuralEntropy

```ts
structuralEntropy(
  graph: { [key: string]: unknown; nodes: { geneId: number; }[]; connections: { from: { geneId: number; }; to: { geneId: number; }; enabled: boolean; }[]; },
): number
```

Lightweight proxy for structural entropy based on degree-distribution.

This function computes an approximate entropy of a graph topology by
counting node degrees and computing the entropy of the degree histogram.
The result is cached on the graph object for the current generation in
`_entropyVal` to avoid repeated expensive recomputation.

Example:

Parameters:
- `graph` - - A genome-like object with `nodes` and `connections` arrays.

Returns: A non-negative number approximating structural entropy.

Example:

const H = structuralEntropy.call(neat, genome);
console.log(`Structure entropy: ${H.toFixed(3)}`);

### TelemetryContext

Context view used within telemetry helpers to access optional internal
fields with descriptive names rather than repeated inline casts.
