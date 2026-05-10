/**
 * Core activation-array-pool chapter for the architecture surface.
 *
 * This folder owns the reusable output buffers that support high-frequency
 * activation paths. It sits beside the network and layer chapters because the
 * pool is not a new runtime primitive; it is the memory policy that decides
 * when temporary activation storage should be reused instead of reallocated.
 *
 * Think of this boundary as scratch-space policy for activation-heavy code.
 * Network execution repeatedly asks for short-lived arrays whose shapes often
 * repeat from sample to sample. Reallocating those arrays every pass adds
 * garbage-collector work without changing the model's behavior. This chapter
 * keeps the reusable buffer story explicit while leaving graph semantics to
 * the node, layer, and network chapters.
 *
 * The key design choice is that pooling happens by array length and resolved
 * activation precision, not by caller identity. A network slab path and a
 * layer helper can share the same retained storage as long as they request the
 * same length and precision and release the buffer after use. Acquire always
 * returns a zeroed buffer, so the pool promises clean scratch memory rather
 * than cached activations or memoized results.
 *
 * That distinction matters when you are debugging correctness. This folder is
 * not a semantic cache and it does not preserve intermediate values for later
 * reads. It only keeps array shells alive long enough to reduce allocation
 * churn in hot loops, worker batches, and repeated forward passes.
 *
 * ```mermaid
 * flowchart LR
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   Request[Activation request]:::base --> Acquire[acquire by length]:::accent
 *   Acquire -->|bucket hit| Reuse[Zeroed reused buffer]:::base
 *   Acquire -->|bucket miss| Fresh[Freshly allocated buffer]:::base
 *   Reuse --> Execute[Layer or network activation]:::base
 *   Fresh --> Execute
 *   Execute --> Release[release buffer]:::accent
 *   Release --> Bucket[Length bucket]:::base
 * ```
 *
 * For background on the broader software pattern, see Wikipedia contributors,
 * [Object pool pattern](https://en.wikipedia.org/wiki/Object_pool_pattern).
 * This chapter applies that idea to activation scratch space, not to long-lived
 * model parameters.
 *
 * Read this chapter in three passes:
 *
 * 1. start with `ActivationArray` to see which buffer shapes the runtime can
 *    safely recycle,
 * 2. continue to `activationArrayPool.acquire()` and `.release()` when you need
 *    the hot-path allocation story,
 * 3. finish with `stats()`, `setMaxPerBucket()`, `prewarm()`, and `compact()`
 *    when you want observability and capacity control.
 *
 * Example: reuse one scratch buffer around a custom activation-heavy loop.
 *
 * ```ts
 * const activationArray = activationArrayPool.acquire(128);
 * // fill and consume the buffer here
 * activationArrayPool.release(activationArray);
 * ```
 *
 * Example: prewarm one common bucket before a large evaluation batch.
 *
 * ```ts
 * activationArrayPool.setMaxPerBucket(32);
 * activationArrayPool.prewarm(256, 8);
 * const stats = activationArrayPool.stats();
 * ```
 */

import type { ActivationPrecision } from '../../config';
import { resolvePrecisionConfig } from '../../config';
import { defaultMemoryManager } from '../../memory/manager';

/**
 * Allowed activation array shapes for pooling.
 *
 * The runtime prefers typed arrays when float32 mode is enabled, but keeps
 * plain numeric arrays available for code paths that expect standard JS array
 * behavior.
 */
export type ActivationArray = number[] | Float32Array | Float64Array;

type ActivationArrayBucket = {
  arrays: ActivationArray[];
  lastTouchedTick: number;
};

/**
 * A size-bucketed pool of activation arrays.
 *
 * Buckets map array length plus resolved precision to stacks of reusable
 * buffers. Acquire returns a zeroed buffer, either by recycling an existing
 * one or by allocating a new array when the requested bucket is empty.
 */
class ActivationArrayPool {
  /** Buckets keyed by length and resolved precision, storing reusable arrays. */
  private buckets: Map<string, ActivationArrayBucket> = new Map();
  /** Count of arrays created since last clear(), for diagnostics. */
  private created = 0;
  /** Count of successful reuses since last clear(), for diagnostics. */
  private reused = 0;
  /** Count of compaction passes that trimmed retained activation storage. */
  private compactionCount = 0;
  /** Count of activation arrays trimmed across all compaction passes. */
  private trimmedArrays = 0;
  /** Count of buckets evicted across all compaction passes. */
  private trimmedBuckets = 0;
  /** Count of arrays currently retained across all buckets. */
  private retainedArrayCount = 0;
  /** Monotonic clock used to approximate least-recently-used bucket order. */
  private accessTick = 0;
  /** Max arrays retained per size bucket; Infinity by default. */
  private maxPerBucket = Number.POSITIVE_INFINITY;

  /**
   * Acquire an activation array of fixed length.
   *
   * Reused arrays are zero-filled before they are handed back so activation
   * callers never observe stale values from a prior pass.
   *
   * @param size Required array length.
   * @param activationPrecision Optional precision override for this acquisition.
   * @returns Zeroed activation array of the requested size.
   */
  acquire(
    size: number,
    activationPrecision?: ActivationPrecision,
  ): ActivationArray {
    const memoryConfig = defaultMemoryManager.getConfig();
    const precisionConfig = resolvePrecisionConfig(
      {
        activationPrecision,
      },
      memoryConfig,
    );
    const bucketKey = createActivationArrayBucketKey(
      size,
      precisionConfig.activationPrecision,
    );
    const bucket = this.buckets.get(bucketKey);

    if (bucket) {
      bucket.lastTouchedTick = this.nextAccessTick();
    }

    if (bucket && bucket.arrays.length > 0) {
      this.reused++;
      this.retainedArrayCount--;
      const activationArray = bucket.arrays.pop()!;

      activationArray.fill(0);

      return activationArray;
    }

    this.created++;
    return createActivationArray(size, memoryConfig, activationPrecision);
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
    const activationPrecision = resolveActivationArrayPrecision(array);
    const bucketKey = createActivationArrayBucketKey(size, activationPrecision);

    const bucket = this.getOrCreateBucket(bucketKey);
    bucket.lastTouchedTick = this.nextAccessTick();

    if (bucket.arrays.length < this.maxPerBucket) {
      bucket.arrays.push(array);
      this.retainedArrayCount++;
    }
  }

  /**
   * Clear all buckets and reset counters.
   *
   * @returns Nothing.
   */
  clear(): void {
    this.buckets.clear();
    this.accessTick = 0;
    this.compactionCount = 0;
    this.created = 0;
    this.retainedArrayCount = 0;
    this.reused = 0;
    this.trimmedArrays = 0;
    this.trimmedBuckets = 0;
  }

  /**
   * Snapshot diagnostics for the current pool state.
   *
   * @returns Creation count, reuse count, and active bucket count.
   */
  stats(): {
    reused: number;
    created: number;
    bucketCount: number;
    retainedArrayCount: number;
    compactionCount: number;
    trimmedArrays: number;
    trimmedBuckets: number;
  } {
    return {
      bucketCount: this.buckets.size,
      compactionCount: this.compactionCount,
      created: this.created,
      retainedArrayCount: this.retainedArrayCount,
      reused: this.reused,
      trimmedArrays: this.trimmedArrays,
      trimmedBuckets: this.trimmedBuckets,
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

      if (this.retainedArrayCount > 0) {
        this.compact();
      }
    }
  }

  /**
   * Trim retained activation storage by bucket recency.
   *
   * Compaction keeps the most recently touched buckets and discards colder
   * buckets first when the retained bucket budget is lowered. The same pass also
   * enforces the current per-bucket cap so lowering `setMaxPerBucket()` can
   * shrink already-retained storage instead of waiting for later releases.
   *
   * @param maxRetainedBuckets Optional retained bucket budget. Omit to compact only by the current per-bucket cap.
   * @returns Nothing.
   */
  compact(maxRetainedBuckets = Number.POSITIVE_INFINITY): void {
    const normalizedRetainedBucketCap =
      normalizeRetainedBucketCap(maxRetainedBuckets);
    const compactionResult = this.collectCompactionResult(
      normalizedRetainedBucketCap,
    );

    if (
      compactionResult.trimmedArrays === 0 &&
      compactionResult.trimmedBuckets === 0
    ) {
      return;
    }

    this.compactionCount++;
    this.trimmedArrays += compactionResult.trimmedArrays;
    this.trimmedBuckets += compactionResult.trimmedBuckets;
  }

  /**
   * Pre-allocate retained buffers for one size bucket.
   *
   * This is useful when callers know a hot activation size ahead of time and
   * want to avoid the first-wave allocation cost.
   *
   * @param size Array length for the bucket.
   * @param count Number of arrays to prepare.
   * @param activationPrecision Optional precision override for this bucket.
   * @returns Nothing.
   */
  prewarm(
    size: number,
    count: number,
    activationPrecision?: ActivationPrecision,
  ): void {
    const memoryConfig = defaultMemoryManager.getConfig();
    const preparedCount = Math.max(0, Math.floor(count));
    const precisionConfig = resolvePrecisionConfig(
      {
        activationPrecision,
      },
      memoryConfig,
    );
    const bucketKey = createActivationArrayBucketKey(
      size,
      precisionConfig.activationPrecision,
    );

    const bucket = this.getOrCreateBucket(bucketKey);
    bucket.lastTouchedTick = this.nextAccessTick();

    for (
      let preparedIndex = 0;
      preparedIndex < preparedCount && bucket.arrays.length < this.maxPerBucket;
      preparedIndex++
    ) {
      const activationArray = createActivationArray(
        size,
        memoryConfig,
        activationPrecision,
      );
      bucket.arrays.push(activationArray);
      this.created++;
      this.retainedArrayCount++;
    }
  }

  /**
   * Get the retained size of one bucket.
   *
   * @param size Array length for the bucket.
   * @param activationPrecision Optional precision override for the requested bucket.
   * @returns Number of reusable buffers currently retained for that length.
   */
  bucketSize(size: number, activationPrecision?: ActivationPrecision): number {
    const precisionConfig = resolvePrecisionConfig(
      {
        activationPrecision,
      },
      defaultMemoryManager.getConfig(),
    );
    const bucketKey = createActivationArrayBucketKey(
      size,
      precisionConfig.activationPrecision,
    );

    return this.buckets.get(bucketKey)?.arrays.length ?? 0;
  }

  private collectCompactionResult(maxRetainedBuckets: number): {
    trimmedArrays: number;
    trimmedBuckets: number;
  } {
    let trimmedArrays = 0;
    let trimmedBuckets = 0;

    for (const [bucketKey, bucket] of this.buckets.entries()) {
      const arraysToTrim = Math.max(0, bucket.arrays.length - this.maxPerBucket);

      if (arraysToTrim === 0) {
        continue;
      }

      // Step 1: Keep the most recently retained buffers inside this bucket.
      bucket.arrays = bucket.arrays.slice(arraysToTrim);
      this.retainedArrayCount -= arraysToTrim;
      trimmedArrays += arraysToTrim;

      if (bucket.arrays.length === 0) {
        this.buckets.delete(bucketKey);
        trimmedBuckets++;
      }
    }

    if (!Number.isFinite(maxRetainedBuckets)) {
      return { trimmedArrays, trimmedBuckets };
    }

    const bucketsToTrim = Math.max(0, this.buckets.size - maxRetainedBuckets);

    if (bucketsToTrim === 0) {
      return { trimmedArrays, trimmedBuckets };
    }

    const coldBuckets = [...this.buckets.entries()].toSorted(
      (leftEntry, rightEntry) => {
        return leftEntry[1].lastTouchedTick - rightEntry[1].lastTouchedTick;
      },
    );

    for (
      let bucketIndex = 0;
      bucketIndex < bucketsToTrim;
      bucketIndex++
    ) {
      const [bucketKey, bucket] = coldBuckets[bucketIndex];

      this.buckets.delete(bucketKey);
      this.retainedArrayCount -= bucket.arrays.length;
      trimmedArrays += bucket.arrays.length;
      trimmedBuckets++;
    }

    return { trimmedArrays, trimmedBuckets };
  }

  private getOrCreateBucket(bucketKey: string): ActivationArrayBucket {
    const existingBucket = this.buckets.get(bucketKey);

    if (existingBucket) {
      return existingBucket;
    }

    const createdBucket = {
      arrays: [],
      lastTouchedTick: this.nextAccessTick(),
    } satisfies ActivationArrayBucket;
    this.buckets.set(bucketKey, createdBucket);
    return createdBucket;
  }

  private nextAccessTick(): number {
    this.accessTick += 1;
    return this.accessTick;
  }
}

function normalizeRetainedBucketCap(maxRetainedBuckets: number): number {
  if (!Number.isFinite(maxRetainedBuckets)) {
    return Number.POSITIVE_INFINITY;
  }

  return Math.max(0, Math.floor(maxRetainedBuckets));
}

/**
 * Create one fresh activation buffer using the shared precision owner.
 *
 * @param size Required activation-array length.
 * @param precisionFlags Config-like precision flags for the current runtime.
 * @returns Fresh activation buffer with the resolved precision policy.
 */
function createActivationArray(
  size: number,
  precisionFlags: { float32Mode: boolean },
  activationPrecision?: ActivationPrecision,
): ActivationArray {
  const precisionConfig = resolvePrecisionConfig(
    {
      activationPrecision,
    },
    precisionFlags,
  );

  return precisionConfig.activationPrecision === 'f32'
    ? new Float32Array(size)
    : new Array<number>(size).fill(0);
}

/**
 * Build the pool key for one activation-array bucket.
 *
 * @param size Required activation-array length.
 * @param activationPrecision Resolved precision for this bucket.
 * @returns Stable bucket key string.
 */
function createActivationArrayBucketKey(
  size: number,
  activationPrecision: ActivationPrecision,
): string {
  return `${size}:${activationPrecision}`;
}

/**
 * Infer the retained precision for one released activation array.
 *
 * @param activationArray Activation buffer being returned to the pool.
 * @returns Precision bucket that owns this array.
 */
function resolveActivationArrayPrecision(
  activationArray: ActivationArray,
): ActivationPrecision {
  return activationArray instanceof Float32Array ? 'f32' : 'f64';
}

/**
 * Shared singleton instance used across the library for maximal reuse.
 */
export const activationArrayPool = new ActivationArrayPool();

defaultMemoryManager.registerPool('activationArrayPool', {
  reset: () => {
    activationArrayPool.clear();
  },
  stats: () => activationArrayPool.stats(),
});
