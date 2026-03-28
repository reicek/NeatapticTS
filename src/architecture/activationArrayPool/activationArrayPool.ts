/**
 * Core activation-array-pool chapter for the architecture surface.
 *
 * This folder owns the reusable output buffers that support high-frequency
 * activation paths. It sits beside the network and layer chapters because the
 * pool is not a new runtime primitive; it is the memory policy that decides
 * when temporary activation storage should be reused instead of reallocated.
 *
 * Read this chapter in three passes:
 *
 * 1. start with `ActivationArray` to see which buffer shapes the runtime can
 *    safely recycle,
 * 2. continue to `activationArrayPool.acquire()` and `.release()` when you need
 *    the hot-path allocation story,
 * 3. finish with `stats()`, `setMaxPerBucket()`, and `prewarm()` when you want
 *    observability and capacity control.
 */

import { config } from '../../config';

/**
 * Allowed activation array shapes for pooling.
 *
 * The runtime prefers typed arrays when float32 mode is enabled, but keeps
 * plain numeric arrays available for code paths that expect standard JS array
 * behavior.
 */
export type ActivationArray = number[] | Float32Array | Float64Array;

/**
 * A size-bucketed pool of activation arrays.
 *
 * Buckets map array length to stacks of reusable buffers. Acquire returns a
 * zeroed buffer, either by recycling an existing one or by allocating a new
 * array when the requested bucket is empty.
 */
class ActivationArrayPool {
  /** Buckets keyed by length, storing reusable arrays. */
  private buckets: Map<number, ActivationArray[]> = new Map();
  /** Count of arrays created since last clear(), for diagnostics. */
  private created = 0;
  /** Count of successful reuses since last clear(), for diagnostics. */
  private reused = 0;
  /** Max arrays retained per size bucket; Infinity by default. */
  private maxPerBucket = Number.POSITIVE_INFINITY;

  /**
   * Acquire an activation array of fixed length.
   *
   * Reused arrays are zero-filled before they are handed back so activation
   * callers never observe stale values from a prior pass.
   *
   * @param size Required array length.
   * @returns Zeroed activation array of the requested size.
   */
  acquire(size: number): ActivationArray {
    const bucket = this.buckets.get(size);

    if (bucket && bucket.length > 0) {
      this.reused++;
      const activationArray = bucket.pop()!;

      if (Array.isArray(activationArray)) {
        activationArray.fill(0);
      } else if (activationArray instanceof Float32Array) {
        activationArray.fill(0);
      }

      return activationArray;
    }

    this.created++;
    return config.float32Mode
      ? new Float32Array(size)
      : new Array<number>(size).fill(0);
  }

  /**
   * Return an activation array to the pool.
   *
   * When a bucket is already at capacity, the buffer is dropped and normal GC
   * ownership resumes instead of letting retained memory grow without bound.
   *
   * @param array Array to release back to the pool.
   * @returns Nothing.
   */
  release(array: ActivationArray): void {
    const size = array.length >>> 0;

    if (!this.buckets.has(size)) {
      this.buckets.set(size, []);
    }

    const bucket = this.buckets.get(size)!;
    if (bucket.length < this.maxPerBucket) {
      bucket.push(array);
    }
  }

  /**
   * Clear all buckets and reset counters.
   *
   * @returns Nothing.
   */
  clear(): void {
    this.buckets.clear();
    this.created = 0;
    this.reused = 0;
  }

  /**
   * Snapshot diagnostics for the current pool state.
   *
   * @returns Creation count, reuse count, and active bucket count.
   */
  stats(): {
    created: number;
    reused: number;
    bucketCount: number;
  } {
    return {
      created: this.created,
      reused: this.reused,
      bucketCount: this.buckets.size,
    };
  }

  /**
   * Configure the retention cap for each size bucket.
   *
   * @param cap Non-negative capacity per bucket. `Infinity` keeps all buckets unbounded.
   * @returns Nothing.
   */
  setMaxPerBucket(cap: number): void {
    if (typeof cap === 'number' && cap >= 0) {
      this.maxPerBucket = cap;
    }
  }

  /**
   * Pre-allocate retained buffers for one size bucket.
   *
   * This is useful when callers know a hot activation size ahead of time and
   * want to avoid the first-wave allocation cost.
   *
   * @param size Array length for the bucket.
   * @param count Number of arrays to prepare.
   * @returns Nothing.
   */
  prewarm(size: number, count: number): void {
    const preparedCount = Math.max(0, Math.floor(count));

    if (!this.buckets.has(size)) {
      this.buckets.set(size, []);
    }

    const bucket = this.buckets.get(size)!;
    for (
      let preparedIndex = 0;
      preparedIndex < preparedCount && bucket.length < this.maxPerBucket;
      preparedIndex++
    ) {
      const activationArray = config.float32Mode
        ? new Float32Array(size)
        : new Array<number>(size).fill(0);
      bucket.push(activationArray);
      this.created++;
    }
  }

  /**
   * Get the retained size of one bucket.
   *
   * @param size Array length for the bucket.
   * @returns Number of reusable buffers currently retained for that length.
   */
  bucketSize(size: number): number {
    return this.buckets.get(size)?.length ?? 0;
  }
}

/**
 * Shared singleton instance used across the library for maximal reuse.
 */
export const activationArrayPool = new ActivationArrayPool();
