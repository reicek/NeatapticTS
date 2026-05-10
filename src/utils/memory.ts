/**
 * Memory instrumentation chapter for slab, pool, and heap comparisons.
 *
 * This chapter exists for one practical learning problem: memory behavior is
 * one of the easiest parts of an evolutionary system to feel, but one of the
 * hardest parts to explain from raw runtime objects alone. Networks grow,
 * slabs reserve capacity ahead of immediate need, pools trade fresh allocation
 * pressure for reuse, and the JavaScript engine adds object overhead that is
 * difficult to see directly from the outside.
 *
 * Instead of pretending to be a precise profiler, this boundary offers a fast,
 * educational snapshot. The goal is to make design choices visible enough that
 * readers can compare runs and ask better questions: did slab-backed storage
 * reduce object-heavy overhead, is pooling shifting pressure away from fresh
 * allocations, how much reserved typed-array capacity is currently unused, and
 * are browser or Node heap readings moving in the same direction as the
 * library-specific heuristics?
 *
 * The most important design choice is scope. `memoryStats()` is not trying to
 * replace a full heap profiler. It is trying to put library-shaped numbers next
 * to runtime-shaped numbers so a reader can compare architecture decisions with
 * less guesswork. The result is intentionally strongest at trend questions such
 * as "did the slab version get leaner?" or "did pooling reduce fresh pressure?"
 * rather than forensic questions about one exact byte count.
 *
 * A useful mental model is to treat the snapshot as four cooperating layers.
 * The network layer counts connections, nodes, and estimated totals. The slab
 * layer explains reserved versus used typed-array capacity. The pool layer
 * exposes reuse infrastructure. The environment layer shows the coarser browser
 * or Node counters surrounding the library's own heuristics.
 *
 * The environment metrics are intentionally coarser than the network heuristics.
 * See MDN,
 * [Performance.memory](https://developer.mozilla.org/en-US/docs/Web/API/Performance/memory),
 * and the Node.js docs,
 * [process.memoryUsage()](https://nodejs.org/api/process.html#processmemoryusage),
 * for the runtime-level counters this chapter folds alongside its own snapshot.
 * They are useful context, but they do not know the difference between live
 * network structure, reserved slab capacity, and reusable pools the way this
 * module does.
 *
 * Read the chapter in three passes:
 *
 * 1. {@link memoryStats} for the top-level snapshot and registry behavior.
 * 2. {@link MemoryStats} when you want to interpret the resulting sections.
 * 3. `memory.utils.ts` when you want the aggregation, environment probing, and
 *    slab-accounting mechanics behind the snapshot.
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   Networks[Tracked or explicit networks]:::base --> Aggregation[Heuristic network aggregation]:::accent
 *   Aggregation --> Snapshot[MemoryStats snapshot]:::base
 *   Config[Config and allocator flags]:::base --> Snapshot
 *   Environment[Browser or Node heap probes]:::base --> Snapshot
 *   Snapshot --> Questions[Compare storage strategy, pooling, and capacity behavior]:::base
 * ```
 *
 * ```mermaid
 * flowchart LR
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   Stats[MemoryStats]:::accent --> NetworkLayer[connections nodes estimatedTotalBytes]:::base
 *   Stats --> SlabLayer[slabBytes reservedBytes usedBytes fragmentationPct]:::base
 *   Stats --> PoolLayer[nodePool reuse snapshot]:::base
 *   Stats --> EnvironmentLayer[browser heap or Node RSS counters]:::base
 * ```
 *
 * Example: register one network once, then capture snapshots later without
 * threading the target through every call.
 *
 * ```ts
 * registerTrackedNetwork(network);
 * const snapshot = memoryStats();
 *
 * console.log(snapshot.estimatedTotalBytes);
 * console.log(snapshot.slabs.fragmentationPct);
 * ```
 *
 * Example: compare two explicit network sets when you want the memory story to
 * stay local to one experiment.
 *
 * ```ts
 * const baselineSnapshot = memoryStats([baselineNetwork]);
 * const pooledSnapshot = memoryStats([pooledNetwork]);
 *
 * console.log(baselineSnapshot.estimatedTotalBytes);
 * console.log(pooledSnapshot.slabs.pooledFraction);
 * ```
 */
import { nodePoolStats } from '../architecture/nodePool';
import { defaultMemoryManager } from '../memory/manager';
import { getSlabAllocationStats as _getSlabAllocationStats } from '../architecture/network/network.utils';
import {
  HEURISTIC_BYTES,
  aggregateNetworkStats,
  buildFlagSnapshot,
  buildMemoryStatsSnapshot,
  captureEnvironmentMetrics,
  normalizeNetworks,
  safeGetSlabAllocationStats,
} from './memory.utils';

/**
 * Detailed statistics describing the current estimated memory footprint of
 * tracked networks plus supporting pools.
 *
 * Important: All byte counts here are *estimates*. JavaScript engine object
 * overhead varies; once slab (Structure of Arrays) storage dominates, these
 * estimates get closer to real usage. Treat values as relative metrics for
 * comparing configurations (e.g. before / after enabling pooling) rather than
 * exact allocations.
 *
 * The payload is easiest to read as four cooperating layers:
 *
 * - `connections`, `nodes`, and `estimatedTotalBytes` summarize the tracked
 *   network footprint itself,
 * - `slabs` explains how much typed-array storage exists and how much of the
 *   reserved connection capacity is currently used,
 * - `pools` shows whether reusable allocation infrastructure is active,
 * - `env` exposes the coarser browser or Node memory readings that surround the
 *   heuristic network view.
 */
export interface MemoryStats {
  /** Epoch milliseconds when the snapshot was captured. */
  timestamp: number;
  /** Total active connection objects across all tracked networks (object + slab entries). */
  connections: number;
  /** Total active nodes across all tracked networks. */
  nodes: number;
  /** Heuristic bytes per connection (estimatedTotalBytes / connections, rounded). */
  bytesPerConnection: number;
  /** Aggregate estimated bytes combining node object overhead + connection object overhead + slab typed array bytes. */
  estimatedTotalBytes: number;
  slabs: {
    /** Sum of byteLength for all discovered typed arrays representing connection slabs & fast-path caches. */
    slabBytes: number;
    /** Count of individual typed array slabs encountered. */
    slabArrayCount: number;
    /** Percentage (0–100) of reserved slab bytes not currently used (null when capacity unknown). */
    fragmentationPct: number | null;
    /** Total reserved bytes implied by capacity * per-connection element width across slab arrays (null if unknown). */
    reservedBytes: number | null;
    /** Estimated used portion of reservedBytes corresponding to live connections (null if unknown). */
    usedBytes: number | null;
    /** Implementation / layout revision tag (null if network does not expose one). */
    slabVersion: number | null;
    /** Count of async slab builds performed (educational: shows deferred optimization activity). */
    asyncBuilds: number;
    /** Fraction of allocations served from slab pool vs fresh (null if allocator stats unavailable). */
    pooledFraction: number | null;
  };
  pools: {
    /** Node pool statistics or null when pooling disabled / not initialized. */
    nodePool: ReturnType<typeof nodePoolStats> | null;
  };
  flags: {
    /** Snapshot of selected global / experimental feature flags at sampling time. */
    snapshot: {
      warnings: unknown;
      float32Mode: unknown;
      deterministicChainMode: unknown;
      enableGatingTraces: unknown;
      poolMaxPerBucket: number | null;
      poolPrewarmCount: number | null;
      enableNodePooling: boolean;
      /** Raw allocator stats (shape may evolve). */
      allocStats: SlabAllocStats | unknown;
    };
  };
  env: {
    /** True if executing in a browser (window defined). */
    isBrowser: boolean;
    /** Browser: current used JS heap size (if available). */
    usedJSHeapSize?: number;
    /** Browser: total JS heap size (if available). */
    totalJSHeapSize?: number;
    /** Browser: heap size limit (if available). */
    jsHeapSizeLimit?: number;
    /** Node: resident set size in bytes. */
    rss?: number;
    /** Node: bytes of V8 heap used. */
    heapUsed?: number;
    /** Node: total V8 heap size. */
    heapTotal?: number;
    /** Node: external memory usage. */
    external?: number;
  };
}

/** Minimal view of a network used for memory heuristics. Only properties
 * accessed by this module are declared. This keeps coupling light while
 * enabling typed local variables instead of loose catch-all types everywhere. */
export interface NetworkView {
  connections?: unknown[];
  nodes?: unknown[];
  // Slab / pool internals (optional and feature-gated)
  _connWeights?: Float32Array | Float64Array;
  _connFrom?: Uint32Array;
  _connTo?: Uint32Array;
  _connFlags?: Uint8Array;
  _connGain?: Float32Array | Float64Array | null;
  _connPlastic?: Float32Array | Float64Array | null;
  _fastA?: Float32Array | Float64Array;
  _fastS?: Float32Array | Float64Array;
  _connCapacity?: number;
  _connCount?: number;
  _useFloat32Weights?: boolean;
  _slabVersion?: number;
  _slabAsyncBuilds?: number;
}

/** Minimal slab allocator stats shape used here. The real shape may
 * include additional fields; we only rely on fresh/pooled counts. */
export type SlabAllocStats = { fresh: number; pooled: number } | null;

/**
 * Capture heuristic memory statistics for one or more networks with a snapshot of active config flags.
 *
 * This is the main educational entrypoint for the utils chapter. It resolves
 * which networks to inspect, safely samples allocator and environment signals,
 * then folds those pieces into one comparable snapshot.
 *
 * The result is intentionally good at trend questions rather than forensic
 * accuracy. Use it to compare configurations, validate that pooling or slabs
 * changed the memory story in the expected direction, or teach why reserved and
 * used bytes can diverge even when the active connection count stays stable.
 *
 * @param targetNetworks Optional single network or array. If omitted, uses registered networks.
 * @returns MemoryStats heuristic snapshot.
 *
 * @example
 * ```ts
 * registerTrackedNetwork(network);
 * const snapshot = memoryStats();
 *
 * console.log(snapshot.estimatedTotalBytes, snapshot.slabs.fragmentationPct);
 * ```
 */
export const memoryStats = (
  targetNetworks?: NetworkView | NetworkView[],
): MemoryStats => {
  const memoryConfig = defaultMemoryManager.getConfig();
  const networks = normalizeNetworks(targetNetworks, _trackedNetworks);
  const slabAllocationStats = safeGetSlabAllocationStats(
    _getSlabAllocationStats,
  );
  const networkAccumulators = aggregateNetworkStats(networks, HEURISTIC_BYTES);
  const environmentMetrics = captureEnvironmentMetrics();
  const flagSnapshot = buildFlagSnapshot(memoryConfig, slabAllocationStats);
  const nodePoolSnapshot =
    typeof nodePoolStats === 'function'
      ? defaultMemoryManager.getPoolStats<ReturnType<typeof nodePoolStats>>(
          'nodePool',
        ) ?? nodePoolStats()
      : null;

  return buildMemoryStatsSnapshot({
    networks,
    accumulators: networkAccumulators,
    env: environmentMetrics,
    heuristics: HEURISTIC_BYTES,
    allocationStats: slabAllocationStats,
    nodePoolSnapshot,
    flagSnapshot,
  });
};

/**
 * Clear the internal list of networks tracked by `memoryStats()` when no
 * explicit networks are provided. This does NOT free memory; it only
 * removes references held by the registry.
 *
 * Use this when a teaching example, benchmark, or test wants a fresh registry
 * boundary before capturing the next snapshot.
 *
 * @returns void
 */
export const resetMemoryTracking = (): void => {
  _trackedNetworks.length = 0; // clear registered networks
};

/**
 * Register a network for inclusion in future `memoryStats()` calls made
 * without explicit parameters.
 *
 * Duplicate registrations are ignored; insertion order is preserved which is
 * useful for deterministic test snapshots.
 * This makes the registry convenient for demos and repeated observations: code
 * can opt into tracking once, then ask for snapshots later without threading a
 * network list through every call site.
 *
 * @param network Network instance (loose shape, validated at runtime).
 * @returns void
 */
export const registerTrackedNetwork = (
  network: NetworkView | null | undefined,
): void => {
  if (network && !_trackedNetworks.includes(network as NetworkView)) {
    _trackedNetworks.push(network as NetworkView);
  }
};

/**
 * Remove a previously registered network from the tracking registry.
 * No-op if the network is not currently registered.
 *
 * Use this when the chapter's tracked set should follow the active lifetime of
 * a network instead of accumulating historical references.
 *
 * @param network Network instance to remove.
 * @returns void
 */
export const unregisterTrackedNetwork = (network: NetworkView): void => {
  const index = _trackedNetworks.indexOf(network);
  if (index >= 0) _trackedNetworks.splice(index, 1);
};

// Internal registry (simple array to preserve insertion order for deterministic summaries)
const _trackedNetworks: NetworkView[] = [];
