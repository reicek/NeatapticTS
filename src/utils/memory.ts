/**
 * Memory instrumentation utilities (Phase 0).
 *
 * Educational overview:
 * These helpers expose a *heuristic* snapshot of memory usage for the
 * evolutionary population and internal pools. The goal is to help learners
 * reason about how design choices (slab storage, pooling, typed arrays)
 * influence memory footprint *without* incurring heavy introspection costs.
 *
 * Design principles:
 * - Lightweight: Avoid deep graph walks or JSON serialization.
 * - Pay-for-use: If no networks are registered the function returns a small, fast object.
 * - Cross‑environment: Works in both Browser and Node via feature detection.
 * - Extensible: Shape deliberately includes draft sections for later precise accounting phases.
 */

/**
 * Detailed statistics describing the current estimated memory footprint of
 * tracked networks plus supporting pools.
 *
 * Important: All byte counts here are *estimates*. JavaScript engine object
 * overhead varies; once slab (Structure of Arrays) storage dominates, these
 * estimates get closer to real usage. Treat values as relative metrics for
 * comparing configurations (e.g. before / after enabling pooling) rather than
 * exact allocations.
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

/**
 * Capture heuristic memory statistics for one or more networks and active config flags.
 *
 * Usage examples:
 * ```ts
 * import { memoryStats, registerTrackedNetwork } from '../utils/memory';
 * registerTrackedNetwork(myNetwork);
 * const snap = memoryStats();
 * console.log('Approx MB', (snap.estimatedTotalBytes / 1e6).toFixed(2));
 * ```
 * ```ts
 * // Ad-hoc comparison between two networks (no need to register globally)
 * const before = memoryStats(netA);
 * const after = memoryStats(netB);
 * console.log('Δ bytes', after.estimatedTotalBytes - before.estimatedTotalBytes);
 * ```
 *
 * Notes:
 * - Passing an explicit network (or array) bypasses the internal registry.
 * - Fields marked nullable are omitted when the implementation cannot infer
 *   capacity or allocator stats (e.g. early phases or when slabs disabled).
 * - Fragmentation percent is interpreted as (reserved - used)/reserved * 100.
 */
import { config } from '../config';
import { nodePoolStats } from '../architecture/nodePool';
import { getSlabAllocationStats as _getSlabAllocationStats } from '../architecture/network/network.slab';

/** Minimal view of a network used for memory heuristics. Only properties
 * accessed by this module are declared. This keeps coupling light while
 * enabling typed local variables instead of `any` everywhere. */
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
 * Capture heuristic memory statistics for one or more networks with snapshot of active config flags.
 * @param targetNetworks Optional single network or array. If omitted, uses registered networks.
 */
/**
 * Capture heuristic memory statistics for one or more networks with a
 * snapshot of active config flags.
 *
 * @param targetNetworks Optional single network or array. If omitted, uses registered networks.
 * @returns MemoryStats heuristic snapshot.
 */
export const memoryStats = (
  targetNetworks?: NetworkView | NetworkView[]
): MemoryStats => {
  const networks: NetworkView[] = Array.isArray(targetNetworks)
    ? targetNetworks
    : targetNetworks
    ? [targetNetworks]
    : _trackedNetworks;

  let totalConnections = 0;
  let totalNodes = 0;
  let slabBytes = 0;
  let slabArrayCount = 0;
  let totalReservedBytes = 0; // capacity * elementSize per array
  let totalUsedBytes = 0; // logical used slice
  let objectConnOverheadBytes = 0;

  /** Heuristic per-connection JS object overhead (fields / hidden class). Tuned later. */
  const JS_OBJECT_CONN_BYTES = 64; // conservative heuristic
  /** Heuristic per-node JS object overhead. */
  const JS_OBJECT_NODE_BYTES = 72; // includes bias, activation caches, refs

  for (const network of networks) {
    if (!network) continue;
    const connCount = Array.isArray(network.connections)
      ? network.connections.length
      : 0;
    const nodeCount = Array.isArray(network.nodes) ? network.nodes.length : 0;
    totalConnections += connCount;
    totalNodes += nodeCount;

    // Estimate object overhead (will be partly replaced once slabs dominate representation)
    objectConnOverheadBytes += connCount * JS_OBJECT_CONN_BYTES;

    // Examine slab typed arrays (private fields guarded by feature detection)
    const weightSlab = network._connWeights;
    const fromSlab = network._connFrom;
    const toSlab = network._connTo;
    const flagsSlab = network._connFlags; // Phase 3 flags
    const gainSlab = network._connGain; // optional with omission optimization
    const plasticSlab = network._connPlastic;
    const fastA = network._fastA;
    const fastS = network._fastS;

    const typedArrays: Array<
      Float32Array | Float64Array | Uint32Array | Uint8Array | Int32Array
    > = [];
    if (weightSlab) typedArrays.push(weightSlab);
    if (fromSlab) typedArrays.push(fromSlab);
    if (toSlab) typedArrays.push(toSlab);
    if (flagsSlab) typedArrays.push(flagsSlab);
    if (gainSlab) typedArrays.push(gainSlab);
    if (plasticSlab)
      typedArrays.push(plasticSlab as Float32Array | Float64Array);
    if (fastA) typedArrays.push(fastA);
    if (fastS) typedArrays.push(fastS);
    for (const ta of typedArrays) {
      slabArrayCount++;
      slabBytes += ta.byteLength;
    }
    // Capacity vs used (only for connection slabs when capacity tracked)
    const capacity = network._connCapacity;
    const used = network._connCount;
    if (capacity && used !== undefined && used <= capacity) {
      // Approximate per-connection bytes across parallel arrays we manage (weights/from/to/flags/gain)
      // Determine element sizes
      const weightBytes = network._useFloat32Weights ? 4 : 8;
      const gainBytes = gainSlab ? weightBytes : 0;
      const fromBytes = 4;
      const toBytes = 4;
      const flagBytes = 1;
      const plasticBytes = plasticSlab ? weightBytes : 0; // plastic slab mirrors weight precision
      const perConn =
        weightBytes +
        gainBytes +
        fromBytes +
        toBytes +
        flagBytes +
        plasticBytes;
      totalReservedBytes += perConn * capacity;
      totalUsedBytes += perConn * used;
    }
  }

  // Aggregate estimated totals
  const nodeBytes = totalNodes * JS_OBJECT_NODE_BYTES;
  const estimatedTotalBytes = nodeBytes + objectConnOverheadBytes + slabBytes;
  const bytesPerConnection = totalConnections
    ? Math.round(estimatedTotalBytes / totalConnections)
    : 0;

  // Environment metrics (feature-detected)
  const env: MemoryStats['env'] = {
    isBrowser: typeof window !== 'undefined',
  } as MemoryStats['env'];
  try {
    // Guarded access to browser performance.memory (non-standard in some envs)
    if (typeof performance !== 'undefined' && 'memory' in performance) {
      const mem = (performance as Performance & {
        memory?: {
          usedJSHeapSize: number;
          totalJSHeapSize: number;
          jsHeapSizeLimit: number;
        };
      }).memory;
      if (mem) {
        env.usedJSHeapSize = mem.usedJSHeapSize;
        env.totalJSHeapSize = mem.totalJSHeapSize;
        env.jsHeapSizeLimit = mem.jsHeapSizeLimit;
      }
    }
  } catch (err) {
    // Ignore instrumentation errors; this should not crash consumer code.
    void err;
  }
  try {
    // Node.js environment: use globalThis.process to avoid bundler shims
    const maybeProcess = ((globalThis as unknown) as {
      process?: { memoryUsage?: () => NodeJS.MemoryUsage };
    }).process;
    if (maybeProcess && typeof maybeProcess.memoryUsage === 'function') {
      const mu = maybeProcess.memoryUsage();
      env.rss = mu.rss;
      env.heapUsed = mu.heapUsed;
      env.heapTotal = mu.heapTotal;
      env.external = mu.external;
    }
  } catch (err) {
    // Ignore instrumentation errors; this should not crash consumer code.
    void err;
  }

  const stats: MemoryStats = {
    timestamp: Date.now(),
    connections: totalConnections,
    nodes: totalNodes,
    bytesPerConnection,
    estimatedTotalBytes,
    slabs: {
      slabBytes,
      slabArrayCount,
      fragmentationPct:
        totalReservedBytes > 0
          ? Math.round(
              (100 * (totalReservedBytes - totalUsedBytes)) / totalReservedBytes
            )
          : null,
      reservedBytes: totalReservedBytes || null,
      usedBytes: totalUsedBytes || null,
      slabVersion:
        (networks[0] as NetworkView | undefined)?._slabVersion ?? null, // example version (Phase 3 edu)
      asyncBuilds:
        (networks[0] as NetworkView | undefined)?._slabAsyncBuilds ?? 0,
      pooledFraction: (() => {
        try {
          const stats = _getSlabAllocationStats() as SlabAllocStats;
          if (!stats) return null;
          const denom = stats.fresh + stats.pooled;
          return denom > 0 ? Number((stats.pooled / denom).toFixed(4)) : null;
        } catch (err) {
          void err;
          return null;
        }
      })(),
    },
    pools: {
      nodePool: typeof nodePoolStats === 'function' ? nodePoolStats() : null,
      // future: connectionPool, slabAllocators, activation array pool etc.
    },
    flags: {
      snapshot: {
        warnings: config.warnings,
        float32Mode: config.float32Mode,
        deterministicChainMode: config.deterministicChainMode,
        enableGatingTraces: config.enableGatingTraces,
        poolMaxPerBucket: config.poolMaxPerBucket ?? null,
        poolPrewarmCount: config.poolPrewarmCount ?? null,
        enableNodePooling:
          ((config as unknown) as { enableNodePooling?: boolean })
            .enableNodePooling ?? false, // Phase 2 addition
        allocStats: (() => {
          try {
            return _getSlabAllocationStats();
          } catch (err) {
            void err;
            return null;
          }
        })(),
      },
    },
    env,
  };
  return stats;
};

/**
 * Reset internal tracking registry (and, in later phases, ancillary counters).
 *
 * Educational: Calling this does NOT free memory — it simply clears the list
 * of networks that will be included when `memoryStats()` is invoked without
 * arguments. Use it between benchmark runs to isolate scenarios.
 */
/**
 * Clear the internal list of networks tracked by `memoryStats()` when no
 * explicit networks are provided. This does NOT free memory; it only
 * removes references held by the registry.
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
 *
 * @param network Network instance (loosely typed to defer strict coupling).
 */
/**
 * Register a network instance for future anonymous `memoryStats()` calls.
 * Duplicate registrations are ignored.
 *
 * @param network Network instance (loose shape, validated at runtime).
 */
export const registerTrackedNetwork = (
  network: NetworkView | null | undefined
): void => {
  if (network && !_trackedNetworks.includes(network as NetworkView)) {
    _trackedNetworks.push(network as NetworkView);
  }
};

/**
 * Remove a previously registered network from the tracking registry.
 * No-op if the network is not currently registered.
 *
 * @param network Network instance.
 */
/**
 * Unregister a previously registered network. No-op when not found.
 * @param network Network instance to remove.
 */
export const unregisterTrackedNetwork = (network: NetworkView): void => {
  const index = _trackedNetworks.indexOf(network);
  if (index >= 0) _trackedNetworks.splice(index, 1);
};

// Internal registry (simple array to preserve insertion order for deterministic summaries)
// Internal registry (simple array to preserve insertion order for deterministic summaries)
const _trackedNetworks: NetworkView[] = [];
