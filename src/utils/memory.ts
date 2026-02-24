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
import { config } from '../config';
import { nodePoolStats } from '../architecture/nodePool';
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
 * Capture heuristic memory statistics for one or more networks with a snapshot of active config flags.
 *
 * @param targetNetworks - Optional single network or array. If omitted, uses registered networks.
 * @returns MemoryStats heuristic snapshot.
 */
export const memoryStats = (
  targetNetworks?: NetworkView | NetworkView[],
): MemoryStats => {
  const networks = normalizeNetworks(targetNetworks, _trackedNetworks);
  const slabAllocationStats = safeGetSlabAllocationStats(
    _getSlabAllocationStats,
  );
  const networkAccumulators = aggregateNetworkStats(networks, HEURISTIC_BYTES);
  const environmentMetrics = captureEnvironmentMetrics();
  const flagSnapshot = buildFlagSnapshot(config, slabAllocationStats);
  const nodePoolSnapshot =
    typeof nodePoolStats === 'function' ? nodePoolStats() : null;

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
 * @param network Network instance to remove.
 * @returns void
 */
export const unregisterTrackedNetwork = (network: NetworkView): void => {
  const index = _trackedNetworks.indexOf(network);
  if (index >= 0) _trackedNetworks.splice(index, 1);
};

// Internal registry (simple array to preserve insertion order for deterministic summaries)
const _trackedNetworks: NetworkView[] = [];
