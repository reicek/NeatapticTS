# utils

Memory instrumentation utilities (Phase 0).

Educational overview:
These helpers expose a *heuristic* snapshot of memory usage for the
evolutionary population and internal pools. The goal is to help learners
reason about how design choices (slab storage, pooling, typed arrays)
influence memory footprint *without* incurring heavy introspection costs.

Design principles:
- Lightweight: Avoid deep graph walks or JSON serialization.
- Pay-for-use: If no networks are registered the function returns a small, fast object.
- Cross‑environment: Works in both Browser and Node via feature detection.
- Extensible: Shape deliberately includes draft sections for later precise accounting phases.

## utils/memory.ts

### memoryStats

`(targetNetworks: import("src/utils/memory").NetworkView | import("src/utils/memory").NetworkView[] | undefined) => import("src/utils/memory").MemoryStats`

Capture heuristic memory statistics for one or more networks with a snapshot of active config flags.

Parameters:
- `targetNetworks` - - Optional single network or array. If omitted, uses registered networks.

Returns: MemoryStats heuristic snapshot.

### MemoryStats

Detailed statistics describing the current estimated memory footprint of
tracked networks plus supporting pools.

Important: All byte counts here are *estimates*. JavaScript engine object
overhead varies; once slab (Structure of Arrays) storage dominates, these
estimates get closer to real usage. Treat values as relative metrics for
comparing configurations (e.g. before / after enabling pooling) rather than
exact allocations.

### NetworkView

Minimal view of a network used for memory heuristics. Only properties
accessed by this module are declared. This keeps coupling light while
enabling typed local variables instead of `any` everywhere.

### registerTrackedNetwork

`(network: import("src/utils/memory").NetworkView | null | undefined) => void`

Register a network for inclusion in future `memoryStats()` calls made
without explicit parameters.

Duplicate registrations are ignored; insertion order is preserved which is
useful for deterministic test snapshots.

Parameters:
- `network` - Network instance (loose shape, validated at runtime).

Returns: void

### resetMemoryTracking

`() => void`

Clear the internal list of networks tracked by `memoryStats()` when no
explicit networks are provided. This does NOT free memory; it only
removes references held by the registry.

Returns: void

### SlabAllocStats

Minimal slab allocator stats shape used here. The real shape may
include additional fields; we only rely on fresh/pooled counts.

### unregisterTrackedNetwork

`(network: import("src/utils/memory").NetworkView) => void`

Remove a previously registered network from the tracking registry.
No-op if the network is not currently registered.

Parameters:
- `network` - Network instance to remove.

Returns: void

## utils/memory.utils.ts

### accumulateCapacitySlices

`(accumulators: import("src/utils/memory.utils").Accumulators, network: import("src/utils/memory").NetworkView, heuristics: import("src/utils/memory.utils").HeuristicBytes) => void`

Track reserved vs used bytes based on connection capacity slices.

Parameters:
- `accumulators` - Running totals for the memory snapshot.
- `network` - Network exposing capacity metadata.
- `heuristics` - Fallback byte weights for connection objects.

### accumulateSlabArrays

`(accumulators: import("src/utils/memory.utils").Accumulators, typedArrays: (Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike> | Uint32Array<ArrayBufferLike> | Uint8Array<ArrayBufferLike> | Int32Array<ArrayBufferLike>)[]) => void`

Sum slab-backed array counts and byte sizes into the accumulator.

Parameters:
- `accumulators` - Running totals for the memory snapshot.
- `typedArrays` - Connection-parallel arrays to measure.

### Accumulators

Running totals used while walking networks to summarize memory consumption.
Accumulates counts, slab byte totals, and reserved vs used capacity snapshots.

### aggregateNetworkStats

`(networksToSummarize: import("src/utils/memory").NetworkView[], heuristics: import("src/utils/memory.utils").HeuristicBytes) => import("src/utils/memory.utils").Accumulators`

Aggregate per-network counters and slab metrics into a single accumulator.

Parameters:
- `networksToSummarize` - Networks to include in the snapshot.
- `heuristics` - Heuristic byte weights for connections and nodes.

Returns: Accumulated summary of network metrics.

### buildFlagSnapshot

`(configSnapshot: import("src/utils/memory.utils").ConfigSnapshot, allocationStats: import("src/utils/memory").SlabAllocStats) => { warnings: unknown; float32Mode: unknown; deterministicChainMode: unknown; enableGatingTraces: unknown; poolMaxPerBucket: number | null; poolPrewarmCount: number | null; enableNodePooling: boolean; allocStats: unknown; }`

Build flag snapshot derived from config and allocator stats.

Parameters:
- `configSnapshot` - Relevant configuration values.
- `allocationStats` - Allocator stats (nullable on failure).

Returns: Flags snapshot for MemoryStats.

### BuildMemoryStatsInput

Structured inputs required to assemble a MemoryStats snapshot in one pass.
Bundles precomputed accumulators, environment info, allocator stats, and flags.

### buildMemoryStatsSnapshot

`(input: import("src/utils/memory.utils").BuildMemoryStatsInput) => import("src/utils/memory").MemoryStats`

Build the full MemoryStats snapshot from precomputed components.

Parameters:
- `input` - Structured inputs collected by the orchestrator.

Returns: Complete MemoryStats snapshot.

### buildSlabStats

`(accumulators: import("src/utils/memory.utils").Accumulators, networksToSummarize: import("src/utils/memory").NetworkView[], allocationStats: import("src/utils/memory").SlabAllocStats) => { slabBytes: number; slabArrayCount: number; fragmentationPct: number | null; reservedBytes: number | null; usedBytes: number | null; slabVersion: number | null; asyncBuilds: number; pooledFraction: number | null; }`

Assemble slab-related statistics for the MemoryStats payload.

Parameters:
- `accumulators` - Running totals collected during aggregation.
- `networksToSummarize` - Networks included in the snapshot.
- `allocationStats` - Optional allocator stats for pooled fraction.

Returns: Structured slab metrics block.

### calculateFragmentation

`(accumulators: import("src/utils/memory.utils").Accumulators) => number | null`

Compute fragmentation percentage from reserved vs used connection bytes.

Parameters:
- `accumulators` - Running totals holding reserved and used bytes.

Returns: Fragmentation percent (0-100) or null when undefined.

### calculatePooledFraction

`(allocationStats: import("src/utils/memory").SlabAllocStats) => number | null`

Calculate pooled fraction from allocator stats with four-decimal precision.

Parameters:
- `allocationStats` - Allocator snapshot or null when unavailable.

Returns: Fraction of pooled allocations or null if indeterminate.

### captureEnvironmentMetrics

`() => { isBrowser: boolean; usedJSHeapSize?: number | undefined; totalJSHeapSize?: number | undefined; jsHeapSizeLimit?: number | undefined; rss?: number | undefined; heapUsed?: number | undefined; heapTotal?: number | undefined; external?: number | undefined; }`

Capture environment memory metrics from browser or Node when available.

Returns: Environment metrics structure for the snapshot.

### captureVersionMetadata

`(accumulators: import("src/utils/memory.utils").Accumulators, network: import("src/utils/memory").NetworkView) => void`

Capture slab metadata (version and async builds) once across all networks.

Parameters:
- `accumulators` - Running totals with metadata slots.
- `network` - Network providing slab metadata fields.

### collectConnectionTypedArrays

`(network: import("src/utils/memory").NetworkView) => (Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike> | Uint32Array<ArrayBufferLike> | Uint8Array<ArrayBufferLike> | Int32Array<ArrayBufferLike>)[]`

Gather all typed arrays that represent connection-parallel data on a network.

Parameters:
- `network` - Network providing connection state arrays.

Returns: Typed arrays aligned to connections.

### computeCounts

`(network: import("src/utils/memory").NetworkView) => CountSnapshot`

Capture simple counts for nodes and connections on a network view.

Parameters:
- `network` - Network being summarized.

Returns: Connection and node counts.

### ConfigSnapshot

Captured configuration knobs that influence memory usage and pooling behavior.
Keeps only the flags relevant to the memory snapshot to avoid leaking full config.

### CONNECTION_OBJECT_BYTES

Estimated per-connection JS object footprint in bytes (includes metadata fields).
Used as a fallback when typed-array parallel data is unavailable.

### createEmptyAccumulators

`() => import("src/utils/memory.utils").Accumulators`

Initialize a fresh accumulator snapshot for memory summaries.

Returns: Zeroed accumulators ready for aggregation.

### describeConnectionBytes

`(network: import("src/utils/memory").NetworkView, heuristics: import("src/utils/memory.utils").HeuristicBytes) => number`

Determine bytes per connection using typed-array width or heuristic fallback.

Parameters:
- `network` - Network whose storage format drives the byte width.
- `heuristics` - Heuristic sizes for non-typed-array cases.

Returns: Estimated bytes per connection entry.

### HEURISTIC_BYTES

Default heuristics mapping human-readable weights to their byte estimates.
Centralizes the fallback values so downstream summaries stay consistent.

### HeuristicBytes

Heuristic byte weights used to approximate per-object overhead in the allocator.
These numbers represent typical JS object footprints, not exact runtime measurements.

### NODE_OBJECT_BYTES

Estimated per-node JS object footprint in bytes, covering activation state and IDs.
This heuristic keeps node weight comparable to connection objects during summaries.

### normalizeNetworks

`(targets: import("src/utils/memory").NetworkView | import("src/utils/memory").NetworkView[] | undefined, trackedNetworks: import("src/utils/memory").NetworkView[]) => import("src/utils/memory").NetworkView[]`

Normalize provided targets to an array of networks, falling back to tracked registry.

Parameters:
- `targets` - Optional single network or array.
- `trackedNetworks` - Internal registry of tracked networks.

Returns: Array of networks to summarize.

### safeGetSlabAllocationStats

`(getSlabAllocationStats: () => unknown) => import("src/utils/memory").SlabAllocStats`

Safely read slab allocation stats, guarding against provider errors.

Parameters:
- `getSlabAllocationStats` - Provider function returning allocator stats.

Returns: Slab allocation stats or null on failure.
