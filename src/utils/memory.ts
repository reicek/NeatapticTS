/**
 * memory.ts
 *
 * Phase 0 memory instrumentation utilities.
 *
 * Design Goals (Phase 0):
 *  - Lightweight heuristic stats: no heavy reflection / JSON stringify cost.
 *  - Pay-for-use: if no networks registered, returns minimal structure.
 *  - Safe in both Node & Browser (feature-detect environment APIs).
 *  - Foundation for later precise slab + pool accounting.
 */

/**
 * Core memory statistics (draft shape) describing current network & pool state.
 *
 * @typedef {Object} MemoryStats
 * @property {number} timestamp Epoch ms when sampled.
 * @property {number} connections Number of active connections.
 * @property {number} nodes Number of active nodes.
 * @property {number} bytesPerConnection Approximate bytes per active connection (heuristic or measured).
 * @property {number} estimatedTotalBytes Aggregate estimated bytes across tracked structures.
 * @property {object} slabs Slab-related stats (weights, gains, flags, fragmentation%) (draft).
 * @property {object} pools Pool high-water marks & current sizes (draft).
 * @property {object} flags Active feature / optimization flags snapshot.
 * @property {object} env Environment heuristics (isBrowser, heapUsed, rss, usedJSHeapSize, etc.).
 */

/**
 * Collect approximate memory statistics across network structures and pools.
 * Placeholder returns minimal stub; implementation added later in Phase 0.
 *
 * @returns {MemoryStats} A memory stats object (partial placeholder fields).
 */
import { config } from '../config';
import { nodePoolStats } from '../architecture/nodePool';

/**
 * Capture heuristic memory statistics for one or more networks with snapshot of active config flags.
 * @param targetNetworks Optional single network or array. If omitted, uses registered networks.
 */
export function memoryStats(targetNetworks?: any | any[]) {
  const networks: any[] = Array.isArray(targetNetworks)
    ? targetNetworks
    : targetNetworks
    ? [targetNetworks]
    : _trackedNetworks;

  let totalConnections = 0;
  let totalNodes = 0;
  let slabBytes = 0;
  let slabArrayCount = 0;
  let objectConnOverheadBytes = 0;

  /** Heuristic per-connection JS object overhead (fields / hidden class). Tuned later. */
  const JS_OBJECT_CONN_BYTES = 64; // conservative heuristic
  /** Heuristic per-node JS object overhead. */
  const JS_OBJECT_NODE_BYTES = 72; // includes bias, activation caches, refs

  for (const net of networks) {
    if (!net) continue;
    const connCount = Array.isArray(net.connections) ? net.connections.length : 0;
    const nodeCount = Array.isArray(net.nodes) ? net.nodes.length : 0;
    totalConnections += connCount;
    totalNodes += nodeCount;

    // Estimate object overhead (will be partly replaced once slabs dominate representation)
    objectConnOverheadBytes += connCount * JS_OBJECT_CONN_BYTES;

    // Examine slab typed arrays (private fields guarded by feature detection)
    const weightSlab = (net as any)._connWeights as
      | Float32Array
      | Float64Array
      | undefined;
    const fromSlab = (net as any)._connFrom as Uint32Array | undefined;
    const toSlab = (net as any)._connTo as Uint32Array | undefined;
    const fastA = (net as any)._fastA as Float32Array | Float64Array | undefined;
    const fastS = (net as any)._fastS as Float32Array | Float64Array | undefined;

    const typedArrays: Array<
      | Float32Array
      | Float64Array
      | Uint32Array
      | Uint8Array
      | Int32Array
    > = [];
    if (weightSlab) typedArrays.push(weightSlab);
    if (fromSlab) typedArrays.push(fromSlab);
    if (toSlab) typedArrays.push(toSlab);
    if (fastA) typedArrays.push(fastA);
    if (fastS) typedArrays.push(fastS);
    for (const ta of typedArrays) {
      slabArrayCount++;
      slabBytes += ta.byteLength;
    }
  }

  // Aggregate estimated totals
  const nodeBytes = totalNodes * JS_OBJECT_NODE_BYTES;
  const estimatedTotalBytes = nodeBytes + objectConnOverheadBytes + slabBytes;
  const bytesPerConnection = totalConnections
    ? Math.round(estimatedTotalBytes / totalConnections)
    : 0;

  // Environment metrics (feature-detected)
  const env: any = { isBrowser: typeof window !== 'undefined' };
  try {
    if (typeof performance !== 'undefined' && (performance as any).memory) {
      const mem = (performance as any).memory;
      env.usedJSHeapSize = mem.usedJSHeapSize;
      env.totalJSHeapSize = mem.totalJSHeapSize;
      env.jsHeapSizeLimit = mem.jsHeapSizeLimit;
    }
  } catch {}
  try {
    if (typeof process !== 'undefined' && (process as any).memoryUsage) {
      const mu = (process as any).memoryUsage();
      env.rss = mu.rss;
      env.heapUsed = mu.heapUsed;
      env.heapTotal = mu.heapTotal;
      env.external = mu.external;
    }
  } catch {}

  return {
    timestamp: Date.now(),
    connections: totalConnections,
    nodes: totalNodes,
    bytesPerConnection,
    estimatedTotalBytes,
    slabs: {
      slabBytes,
      slabArrayCount,
      // fragmentation placeholder (computed in later phases when we track reserved vs used)
      fragmentationPct: null,
    },
    pools: {
      nodePool: nodePoolStats?.() ?? null,
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
      },
    },
    env,
  } as any;
}

/**
 * Reset internal counters / high-water marks used for memory tracking.
 * Placeholder for future pool tracking reset.
 */
export function resetMemoryTracking(): void {
  _trackedNetworks.length = 0; // clear registered networks
}

/**
 * Register a network instance for memory tracking (optional depending on design).
 * Allows memoryStats() to iterate over registered networks when no explicit instance passed.
 * @param {any} network Network instance (type narrowed later).
 */
export function registerTrackedNetwork(network: any): void {
  if (network && !_trackedNetworks.includes(network)) {
    _trackedNetworks.push(network);
  }
}

/**
 * Unregister a previously registered network instance.
 * @param {any} network Network instance.
 */
export function unregisterTrackedNetwork(network: any): void {
  const idx = _trackedNetworks.indexOf(network);
  if (idx >= 0) _trackedNetworks.splice(idx, 1);
}

// Internal registry (simple array to preserve insertion order for deterministic summaries)
const _trackedNetworks: any[] = [];
