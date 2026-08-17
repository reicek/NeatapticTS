/**
 * Named constants for the generic network acceleration layer.
 *
 * Keeping thresholds as exported constants makes them easy to reference from
 * tests, documentation, and downstream configuration builders without hunting
 * for magic numbers in implementation code.
 */

/**
 * Minimum node count before the GPU backend is considered eligible.
 *
 * Networks smaller than this threshold are evaluated on CPU because the GPU
 * setup overhead dominates the compute cost.
 */
export const DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD = 1_024;

/**
 * Minimum number of parallel candidate evaluations before GPU batching is
 * preferred over sequential CPU evaluation.
 */
export const DEFAULT_ACCELERATION_GPU_BATCH_PARALLEL_THRESHOLD = 8;

/**
 * Minimum number of logical CPU cores required before worker evaluation is
 * considered. A machine with fewer cores is unlikely to benefit from the
 * scheduling overhead of worker threads.
 */
export const DEFAULT_ACCELERATION_WORKER_MIN_CORES = 2;

/**
 * Maximum number of workers to spawn when worker evaluation is enabled.
 *
 * The default is conservative to avoid starving the host process and other
 * browser tabs.
 */
export const DEFAULT_ACCELERATION_MAX_WORKERS = 4;

/**
 * Default number of weight variants evaluated concurrently in a single batch.
 *
 * A value of `16` provides modest parallel dispatch for typical networks while
 * still restoring connection weights between batches. Callers can override it
 * with explicit config.
 */
export const DEFAULT_ACCELERATION_PARALLEL_VARIANT_COUNT = 16;

/**
 * Default number of samples collected per backend during the regression guard
 * micro-benchmark.
 *
 * The sample count is selected deterministically from a small band around this
 * value using the caller-supplied seed, so repeated runs with the same seed
 * observe the same effective workload.
 */
export const DEFAULT_REGRESSION_GUARD_SAMPLES = 10;

/**
 * Default ratio threshold at which the GPU backend is blacklisted.
 *
 * When the GPU median micro-benchmark duration exceeds the CPU median duration
 * multiplied by this ratio, the GPU backend is considered a regression and is
 * blacklisted until `clearBlacklist()` is called.
 */
export const DEFAULT_REGRESSION_GUARD_RATIO_THRESHOLD = 2.0;

// ─────────────────────────────────────────────────────────────────────────────
// Buffer-pool cap constants
// ─────────────────────────────────────────────────────────────────────────────

import type {
  BufferPoolMaxPooledBytesOptions,
  BufferPoolWorkload,
} from './acceleration.types';

/**
 * Minimum bytes the GPU buffer-set pool cap is allowed to resolve to.
 *
 * Very small networks would otherwise produce a cap too small to retain even
 * a single buffer set, which would defeat the purpose of the pool on the next
 * slightly larger evaluation.
 */
export const MIN_BUFFER_POOL_BYTES = 256 * 1024;

/**
 * Default average in-degree used to estimate connection buffer needs.
 *
 * This is a conservative planning estimate, not a measured degree. A value of
 * 10 ensures the heuristic cap exceeds the actual packed buffer footprint for
 * typical NEAT topologies and triggers the device-limit clamp on large
 * networks as intended by the buffer-pool contract.
 */
export const DEFAULT_BUFFER_POOL_AVG_DEGREE = 10;

/**
 * Default number of distinct GPU buffers retained per pool entry.
 *
 * Each topology/variant key caches a full buffer set (connections, nodes,
 * outputs, params, topo levels, and in-start). The count is used by the cap
 * heuristic to reserve headroom for the whole set.
 */
export const DEFAULT_BUFFER_POOL_BUFFER_COUNT = 3;

/**
 * Bytes per Float32 element used by the buffer-pool cap heuristic.
 *
 * WebGPU buffer sizes are expressed in bytes; most numeric fields in the
 * compute shaders are 32-bit floats.
 */
export const DEFAULT_BUFFER_POOL_FLOAT32_BYTES = 4;

/**
 * Default safety multiplier over the raw byte estimate.
 *
 * Provides headroom for alignment padding, transient growth, and small
 * topology variations without allowing unbounded retention.
 */
export const DEFAULT_BUFFER_POOL_SAFETY_FACTOR = 1.5;

/**
 * Resolve the maximum pooled-byte cap for a GPU buffer-set pool.
 *
 * The cap is computed from the workload size (`nodeCount`) and a small set of
 * heuristic defaults. Callers can override the result with an explicit
 * `maxPooledBytes` budget or clamp it to one quarter of a device
 * `maxBufferSize`. The default knobs can also be overridden for benchmarking
 * or unusual network motifs.
 *
 * The formula is:
 *
 * ```text
 * max(minBytes, nodeCount * avgDegree * float32Bytes * bufferCount * safetyFactor)
 * ```
 *
 * and is clamped to `floor(maxBufferSize / 4)` when `maxBufferSize` is given.
 * `variantCount` is part of the workload contract for future scaling but does
 * not affect the current heuristic.
 *
 * ```mermaid
 * flowchart TD
 *   Start([Workload + options]) --> Explicit{maxPooledBytes given?}
 *   Explicit -->|yes| ReturnExplicit[Return explicit cap]
 *   Explicit -->|no| Estimate["bytes = nodeCount * avgDegree * float32Bytes * bufferCount * safetyFactor"]
 *   Estimate --> Floor["bytes = max(minBytes, bytes)"]
 *   Floor --> Clamp{maxBufferSize given?}
 *   Clamp -->|yes| ApplyClamp["bytes = min(bytes, floor(maxBufferSize / 4))"]
 *   Clamp -->|no| ReturnComputed[Return computed cap]
 *   ApplyClamp --> ReturnComputed
 * ```
 *
 * @param workload - Workload description; `nodeCount` drives the heuristic.
 * @param options - Optional explicit cap, device-size clamp, or heuristic overrides.
 * @returns The resolved byte cap.
 *
 * @example
 * ```ts
 * const cap = resolveBufferPoolMaxPooledBytes(
 *   { nodeCount: 1_024, variantCount: 8 },
 *   { maxBufferSize: 64 * 1024 * 1024 }
 * );
 * console.log(cap); // 262144 (min floor) for a 1k-node workload
 * ```
 */
const DEFAULT_BUFFER_POOL_OPTIONS: Required<
  Pick<
    BufferPoolMaxPooledBytesOptions,
    'avgDegree' | 'float32Bytes' | 'bufferCount' | 'safetyFactor' | 'minBytes'
  >
> = {
  avgDegree: DEFAULT_BUFFER_POOL_AVG_DEGREE,
  float32Bytes: DEFAULT_BUFFER_POOL_FLOAT32_BYTES,
  bufferCount: DEFAULT_BUFFER_POOL_BUFFER_COUNT,
  safetyFactor: DEFAULT_BUFFER_POOL_SAFETY_FACTOR,
  minBytes: MIN_BUFFER_POOL_BYTES,
};

/**
 * Clamp a computed byte cap to one quarter of the device `maxBufferSize`
 * when a device limit is supplied.
 *
 * @param floored - The computed cap after applying the minimum floor.
 * @param maxBufferSize - Optional device `maxBufferSize` for clamping.
 * @returns The clamped cap, or `floored` when no device limit is given.
 */
function clampToMaxBufferSize(floored: number, maxBufferSize?: number): number {
  if (maxBufferSize === undefined) return floored;
  return Math.min(floored, Math.floor(maxBufferSize / 4));
}

/**
 * Resolve the maximum pooled bytes for the buffer pool based on the given
 * workload and optional overrides. When `maxPooledBytes` is provided it is
 * returned directly; otherwise the value is estimated from workload metrics
 * and clamped to the configured maximum buffer size.
 *
 * @param workload - Buffer pool workload metrics (node count, connections, etc.).
 * @param options - Optional overrides for estimated calculation parameters.
 * @returns The resolved maximum pooled bytes value.
 */
export function resolveBufferPoolMaxPooledBytes(
  workload: BufferPoolWorkload,
  options?: BufferPoolMaxPooledBytesOptions,
): number {
  if (options?.maxPooledBytes !== undefined) {
    return options.maxPooledBytes;
  }

  const resolved = { ...DEFAULT_BUFFER_POOL_OPTIONS, ...options };
  const estimated =
    (workload.nodeCount ?? 0) *
    resolved.avgDegree *
    resolved.float32Bytes *
    resolved.bufferCount *
    resolved.safetyFactor;
  const floored = Math.max(resolved.minBytes, estimated);
  return clampToMaxBufferSize(floored, options?.maxBufferSize);
}
