import type { nodePoolStats } from '../architecture/nodePool';
import type { MemoryStats, NetworkView, SlabAllocStats } from './memory';

/**
 * Heuristic byte weights used to approximate per-object overhead in the allocator.
 * These numbers represent typical JS object footprints, not exact runtime measurements.
 */
export interface HeuristicBytes {
  connectionObjectBytes: number;
  nodeObjectBytes: number;
}

/**
 * Running totals used while walking networks to summarize memory consumption.
 * Accumulates counts, slab byte totals, and reserved vs used capacity snapshots.
 */
export interface Accumulators {
  totalConnections: number;
  totalNodes: number;
  slabBytes: number;
  slabArrayCount: number;
  totalReservedBytes: number;
  totalUsedBytes: number;
  objectConnOverheadBytes: number;
  slabVersion: number | null;
  slabAsyncBuilds: number;
}

/**
 * Captured configuration knobs that influence memory usage and pooling behavior.
 * Keeps only the flags relevant to the memory snapshot to avoid leaking full config.
 */
export interface ConfigSnapshot {
  warnings: unknown;
  float32Mode: unknown;
  deterministicChainMode?: unknown;
  enableGatingTraces?: unknown;
  poolMaxPerBucket?: number | null;
  poolPrewarmCount?: number | null;
  enableNodePooling?: boolean;
}

/**
 * Structured inputs required to assemble a MemoryStats snapshot in one pass.
 * Bundles precomputed accumulators, environment info, allocator stats, and flags.
 */
export interface BuildMemoryStatsInput {
  networks: NetworkView[];
  accumulators: Accumulators;
  env: MemoryStats['env'];
  heuristics: HeuristicBytes;
  allocationStats: SlabAllocStats | null;
  nodePoolSnapshot: ReturnType<typeof nodePoolStats> | null;
  flagSnapshot: MemoryStats['flags']['snapshot'];
}

/**
 * Estimated per-connection JS object footprint in bytes (includes metadata fields).
 * Used as a fallback when typed-array parallel data is unavailable.
 */
export const CONNECTION_OBJECT_BYTES = 64;

/**
 * Estimated per-node JS object footprint in bytes, covering activation state and IDs.
 * This heuristic keeps node weight comparable to connection objects during summaries.
 */
export const NODE_OBJECT_BYTES = 72;

const FLOAT32_BYTE_WIDTH = 4;
const FLOAT64_BYTE_WIDTH = 8;
const UINT32_BYTE_WIDTH = 4;
const FLAG_BYTE_WIDTH = 1;
const PERCENT_SCALE = 100;

/**
 * Default heuristics mapping human-readable weights to their byte estimates.
 * Centralizes the fallback values so downstream summaries stay consistent.
 */
export const HEURISTIC_BYTES: HeuristicBytes = {
  connectionObjectBytes: CONNECTION_OBJECT_BYTES,
  nodeObjectBytes: NODE_OBJECT_BYTES,
};

/**
 * Normalize provided targets to an array of networks, falling back to tracked registry.
 * @param targets Optional single network or array.
 * @param trackedNetworks Internal registry of tracked networks.
 * @returns Array of networks to summarize.
 */
export function normalizeNetworks(
  targets: NetworkView | NetworkView[] | undefined,
  trackedNetworks: NetworkView[],
): NetworkView[] {
  if (Array.isArray(targets)) return targets;
  if (targets) return [targets];
  return trackedNetworks;
}

/**
 * Safely read slab allocation stats, guarding against provider errors.
 * @param getSlabAllocationStats Provider function returning allocator stats.
 * @returns Slab allocation stats or null on failure.
 */
export function safeGetSlabAllocationStats(
  getSlabAllocationStats: () => SlabAllocStats | unknown,
): SlabAllocStats | null {
  try {
    const stats = getSlabAllocationStats();
    return stats as SlabAllocStats;
  } catch (error: unknown) {
    void error;
    return null;
  }
}

/**
 * Aggregate per-network counters and slab metrics into a single accumulator.
 * @param networksToSummarize Networks to include in the snapshot.
 * @param heuristics Heuristic byte weights for connections and nodes.
 * @returns Accumulated summary of network metrics.
 */
export function aggregateNetworkStats(
  networksToSummarize: NetworkView[],
  heuristics: HeuristicBytes,
): Accumulators {
  const accumulators = createEmptyAccumulators();

  for (const network of networksToSummarize) {
    if (!network) continue;
    const counts = computeCounts(network);
    accumulators.totalConnections += counts.connections;
    accumulators.totalNodes += counts.nodes;
    accumulators.objectConnOverheadBytes +=
      counts.connections * heuristics.connectionObjectBytes;

    const connectionTypedArrays = collectConnectionTypedArrays(network);
    accumulateSlabArrays(accumulators, connectionTypedArrays);
    accumulateCapacitySlices(accumulators, network, heuristics);
    captureVersionMetadata(accumulators, network);
  }

  return accumulators;
}

/**
 * Capture environment memory metrics from browser or Node when available.
 * @returns Environment metrics structure for the snapshot.
 */
export function captureEnvironmentMetrics(): MemoryStats['env'] {
  const environmentMetrics: MemoryStats['env'] = {
    isBrowser: typeof window !== 'undefined',
  } as MemoryStats['env'];

  try {
    if (typeof performance !== 'undefined' && 'memory' in performance) {
      const performanceMemory = (
        performance as Performance & {
          memory?: {
            usedJSHeapSize: number;
            totalJSHeapSize: number;
            jsHeapSizeLimit: number;
          };
        }
      ).memory;
      if (performanceMemory) {
        environmentMetrics.usedJSHeapSize = performanceMemory.usedJSHeapSize;
        environmentMetrics.totalJSHeapSize = performanceMemory.totalJSHeapSize;
        environmentMetrics.jsHeapSizeLimit = performanceMemory.jsHeapSizeLimit;
      }
    }
  } catch (error: unknown) {
    void error;
  }

  try {
    const maybeProcess = (
      globalThis as unknown as {
        process?: { memoryUsage?: () => NodeJS.MemoryUsage };
      }
    ).process;
    if (maybeProcess && typeof maybeProcess.memoryUsage === 'function') {
      const memoryUsageSnapshot = maybeProcess.memoryUsage();
      environmentMetrics.rss = memoryUsageSnapshot.rss;
      environmentMetrics.heapUsed = memoryUsageSnapshot.heapUsed;
      environmentMetrics.heapTotal = memoryUsageSnapshot.heapTotal;
      environmentMetrics.external = memoryUsageSnapshot.external;
    }
  } catch (error: unknown) {
    void error;
  }

  return environmentMetrics;
}

/**
 * Build the full MemoryStats snapshot from precomputed components.
 * @param input Structured inputs collected by the orchestrator.
 * @returns Complete MemoryStats snapshot.
 */
export function buildMemoryStatsSnapshot(
  input: BuildMemoryStatsInput,
): MemoryStats {
  const nodeBytes =
    input.accumulators.totalNodes * input.heuristics.nodeObjectBytes;
  const estimatedTotalBytes =
    nodeBytes +
    input.accumulators.objectConnOverheadBytes +
    input.accumulators.slabBytes;
  const bytesPerConnection = input.accumulators.totalConnections
    ? Math.round(estimatedTotalBytes / input.accumulators.totalConnections)
    : 0;

  return {
    timestamp: Date.now(),
    connections: input.accumulators.totalConnections,
    nodes: input.accumulators.totalNodes,
    bytesPerConnection,
    estimatedTotalBytes,
    slabs: buildSlabStats(
      input.accumulators,
      input.networks,
      input.allocationStats,
    ),
    pools: {
      nodePool: input.nodePoolSnapshot,
    },
    flags: {
      snapshot: input.flagSnapshot,
    },
    env: input.env,
  };
}

/**
 * Build flag snapshot derived from config and allocator stats.
 * @param configSnapshot Relevant configuration values.
 * @param allocationStats Allocator stats (nullable on failure).
 * @returns Flags snapshot for MemoryStats.
 */
export function buildFlagSnapshot(
  configSnapshot: ConfigSnapshot,
  allocationStats: SlabAllocStats | null,
): MemoryStats['flags']['snapshot'] {
  return {
    warnings: configSnapshot.warnings,
    float32Mode: configSnapshot.float32Mode,
    deterministicChainMode: configSnapshot.deterministicChainMode,
    enableGatingTraces: configSnapshot.enableGatingTraces,
    poolMaxPerBucket: configSnapshot.poolMaxPerBucket ?? null,
    poolPrewarmCount: configSnapshot.poolPrewarmCount ?? null,
    enableNodePooling: configSnapshot.enableNodePooling ?? false,
    allocStats: allocationStats,
  };
}

/**
 * Initialize a fresh accumulator snapshot for memory summaries.
 * @returns Zeroed accumulators ready for aggregation.
 */
function createEmptyAccumulators(): Accumulators {
  return {
    totalConnections: 0,
    totalNodes: 0,
    slabBytes: 0,
    slabArrayCount: 0,
    totalReservedBytes: 0,
    totalUsedBytes: 0,
    objectConnOverheadBytes: 0,
    slabVersion: null,
    slabAsyncBuilds: 0,
  };
}

interface CountSnapshot {
  connections: number;
  nodes: number;
}

/**
 * Capture simple counts for nodes and connections on a network view.
 * @param network Network being summarized.
 * @returns Connection and node counts.
 */
function computeCounts(network: NetworkView): CountSnapshot {
  return {
    connections: Array.isArray(network.connections)
      ? network.connections.length
      : 0,
    nodes: Array.isArray(network.nodes) ? network.nodes.length : 0,
  };
}

/**
 * Gather all typed arrays that represent connection-parallel data on a network.
 * @param network Network providing connection state arrays.
 * @returns Typed arrays aligned to connections.
 */
function collectConnectionTypedArrays(
  network: NetworkView,
): Array<Float32Array | Float64Array | Uint32Array | Uint8Array | Int32Array> {
  const typedArrays: Array<
    Float32Array | Float64Array | Uint32Array | Uint8Array | Int32Array
  > = [];
  if (network._connWeights) typedArrays.push(network._connWeights);
  if (network._connFrom) typedArrays.push(network._connFrom);
  if (network._connTo) typedArrays.push(network._connTo);
  if (network._connFlags) typedArrays.push(network._connFlags);
  if (network._connGain) typedArrays.push(network._connGain);
  if (network._connPlastic)
    typedArrays.push(network._connPlastic as Float32Array | Float64Array);
  if (network._fastA) typedArrays.push(network._fastA);
  if (network._fastS) typedArrays.push(network._fastS);
  return typedArrays;
}

/**
 * Sum slab-backed array counts and byte sizes into the accumulator.
 * @param accumulators Running totals for the memory snapshot.
 * @param typedArrays Connection-parallel arrays to measure.
 */
function accumulateSlabArrays(
  accumulators: Accumulators,
  typedArrays: Array<
    Float32Array | Float64Array | Uint32Array | Uint8Array | Int32Array
  >,
): void {
  for (const typedArray of typedArrays) {
    accumulators.slabArrayCount += 1;
    accumulators.slabBytes += typedArray.byteLength;
  }
}

/**
 * Track reserved vs used bytes based on connection capacity slices.
 * @param accumulators Running totals for the memory snapshot.
 * @param network Network exposing capacity metadata.
 * @param heuristics Fallback byte weights for connection objects.
 */
function accumulateCapacitySlices(
  accumulators: Accumulators,
  network: NetworkView,
  heuristics: HeuristicBytes,
): void {
  const capacity = network._connCapacity;
  const used = network._connCount;
  if (!capacity || used === undefined || used > capacity) return;

  const elementBytes = describeConnectionBytes(network, heuristics);
  accumulators.totalReservedBytes += elementBytes * capacity;
  accumulators.totalUsedBytes += elementBytes * used;
}

/**
 * Determine bytes per connection using typed-array width or heuristic fallback.
 * @param network Network whose storage format drives the byte width.
 * @param heuristics Heuristic sizes for non-typed-array cases.
 * @returns Estimated bytes per connection entry.
 */
function describeConnectionBytes(
  network: NetworkView,
  heuristics: HeuristicBytes,
): number {
  const weightBytes = network._useFloat32Weights
    ? FLOAT32_BYTE_WIDTH
    : FLOAT64_BYTE_WIDTH;
  const gainBytes = network._connGain ? weightBytes : 0;
  const fromBytes = UINT32_BYTE_WIDTH;
  const toBytes = UINT32_BYTE_WIDTH;
  const flagBytes = FLAG_BYTE_WIDTH;
  const plasticBytes = network._connPlastic ? weightBytes : 0;
  const totalParallelBytes =
    weightBytes + gainBytes + fromBytes + toBytes + flagBytes + plasticBytes;

  return totalParallelBytes || heuristics.connectionObjectBytes;
}

/**
 * Capture slab metadata (version and async builds) once across all networks.
 * @param accumulators Running totals with metadata slots.
 * @param network Network providing slab metadata fields.
 */
function captureVersionMetadata(
  accumulators: Accumulators,
  network: NetworkView,
): void {
  if (accumulators.slabVersion === null && network._slabVersion !== undefined) {
    accumulators.slabVersion = network._slabVersion ?? null;
  }
  if (
    accumulators.slabAsyncBuilds === 0 &&
    network._slabAsyncBuilds !== undefined
  ) {
    accumulators.slabAsyncBuilds = network._slabAsyncBuilds ?? 0;
  }
}

/**
 * Assemble slab-related statistics for the MemoryStats payload.
 * @param accumulators Running totals collected during aggregation.
 * @param networksToSummarize Networks included in the snapshot.
 * @param allocationStats Optional allocator stats for pooled fraction.
 * @returns Structured slab metrics block.
 */
function buildSlabStats(
  accumulators: Accumulators,
  networksToSummarize: NetworkView[],
  allocationStats: SlabAllocStats | null,
): MemoryStats['slabs'] {
  return {
    slabBytes: accumulators.slabBytes,
    slabArrayCount: accumulators.slabArrayCount,
    fragmentationPct: calculateFragmentation(accumulators),
    reservedBytes: accumulators.totalReservedBytes || null,
    usedBytes: accumulators.totalUsedBytes || null,
    slabVersion:
      (networksToSummarize[0] as NetworkView | undefined)?._slabVersion ??
      accumulators.slabVersion,
    asyncBuilds:
      (networksToSummarize[0] as NetworkView | undefined)?._slabAsyncBuilds ??
      accumulators.slabAsyncBuilds,
    pooledFraction: calculatePooledFraction(allocationStats),
  };
}

/**
 * Compute fragmentation percentage from reserved vs used connection bytes.
 * @param accumulators Running totals holding reserved and used bytes.
 * @returns Fragmentation percent (0-100) or null when undefined.
 */
function calculateFragmentation(accumulators: Accumulators): number | null {
  if (accumulators.totalReservedBytes <= 0) return null;
  const unusedBytes =
    accumulators.totalReservedBytes - accumulators.totalUsedBytes;
  const ratio = (PERCENT_SCALE * unusedBytes) / accumulators.totalReservedBytes;
  return Math.round(ratio);
}

/**
 * Calculate pooled fraction from allocator stats with four-decimal precision.
 * @param allocationStats Allocator snapshot or null when unavailable.
 * @returns Fraction of pooled allocations or null if indeterminate.
 */
function calculatePooledFraction(
  allocationStats: SlabAllocStats | null,
): number | null {
  if (!allocationStats) return null;
  const denominator = allocationStats.fresh + allocationStats.pooled;
  if (denominator <= 0) return null;
  return Number((allocationStats.pooled / denominator).toFixed(4));
}
