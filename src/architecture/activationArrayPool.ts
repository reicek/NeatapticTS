/**
 * Activation array pooling utilities.
 *
 * Size-bucketed pool for reusable activation arrays to reduce allocations in
 * hot forward paths. Reused arrays are zero-filled to prevent stale data.
 * Array type honors global precision via `config.float32Mode`.
 */

import { config } from '../config';

/**
 * Allowed activation array shapes for pooling.
 * - number[]: default JS array
 * - Float32Array: compact typed array when float32 mode is enabled
 * - Float64Array: supported for compatibility with typed math paths
 */
export type ActivationArray = number[] | Float32Array | Float64Array;

/**
 * A size-bucketed pool of activation arrays.
 *
 * Buckets map array length -> stack of arrays. Acquire pops and zero-fills, or
 * allocates a new array when empty. Release pushes back up to a configurable
 * per-bucket cap to avoid unbounded growth.
 *
 * Note: not thread-safe; intended for typical single-threaded JS execution.
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
   * Zero-fills reused arrays to guarantee clean state.
   *
   * @param size Required array length.
   * @returns Zeroed activation array of the requested size.
   */
  acquire(size: number): ActivationArray {
    const bucket = this.buckets.get(size);
    if (bucket && bucket.length > 0) {
      this.reused++;
      const arr = bucket.pop()!;
      // zero on reuse to avoid stale values
      (arr as any).fill(0);
      return arr;
    }
    this.created++;
    return config.float32Mode
      ? new Float32Array(size)
      : new Array<number>(size).fill(0);
  }

  /**
   * Return an activation array to the pool. If the bucket is full per
   * `maxPerBucket`, the array is dropped and left to GC.
   *
   * @param array Array to release back to the pool.
   */
  release(array: ActivationArray) {
    const size = array.length >>> 0;
    if (!this.buckets.has(size)) this.buckets.set(size, []);
    const bucket = this.buckets.get(size)!;
    if (bucket.length < this.maxPerBucket) bucket.push(array);
  }

  /**
   * Clear all buckets and reset counters. Frees references to pooled arrays.
   */
  clear() {
    this.buckets.clear();
    this.created = 0;
    this.reused = 0;
  }

  /**
   * Snapshot of diagnostics: creations, reuses, and number of active buckets.
   */
  stats() {
    return {
      created: this.created,
      reused: this.reused,
      bucketCount: this.buckets.size,
    };
  }

  /**
   * Configure a capacity cap per size bucket to avoid unbounded memory growth.
   *
   * @param cap Non-negative capacity per bucket (Infinity allowed).
   */
  setMaxPerBucket(cap: number) {
    if (typeof cap === 'number' && cap >= 0) this.maxPerBucket = cap;
  }

  /**
   * Pre-allocate and retain arrays for a given size bucket up to `count` items.
   *
   * @param size Array length (bucket key).
   * @param count Number of arrays to prepare (rounded down, min 0).
   */
  prewarm(size: number, count: number) {
    const n = Math.max(0, Math.floor(count));
    if (!this.buckets.has(size)) this.buckets.set(size, []);
    const bucket = this.buckets.get(size)!;
    for (let i = 0; i < n && bucket.length < this.maxPerBucket; i++) {
      const arr = config.float32Mode
        ? new Float32Array(size)
        : new Array<number>(size).fill(0);
      bucket.push(arr);
      this.created++;
    }
  }

  /**
   * Current retained count for a size bucket.
   *
   * @param size Array length (bucket key).
   * @returns Number of arrays available to reuse for that length.
   */
  bucketSize(size: number): number {
    return this.buckets.get(size)?.length ?? 0;
  }
}

/**
 * Shared singleton instance used across the library for maximal reuse.
 */
export const activationArrayPool = new ActivationArrayPool();
