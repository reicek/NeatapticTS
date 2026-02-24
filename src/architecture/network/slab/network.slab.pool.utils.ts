/**
 * Internal slab pool/stat helpers extracted from network.slab.utils.ts.
 */
import { config } from '../../../config';
import type {
  PoolKeyMetrics,
  TypedArray,
  TypedArrayConstructor,
} from './network.slab.utils.types';

const DEFAULT_POOL_CAP = 4;
const ZERO = 0;

/**
 * Internal typed array pool keyed by kind:bytes:length.
 */
const _slabArrayPool: Record<string, Array<TypedArray>> = Object.create(null);

/**
 * Per-key pool metrics for educational diagnostics.
 */
const _slabPoolMetrics: Record<string, PoolKeyMetrics> = Object.create(null);

/**
 * Global allocation counters.
 */
const _slabAllocStats = { fresh: 0, pooled: 0 };

/**
 * Computes retention cap per key.
 *
 * @returns Non-negative max retained arrays per key.
 */
function _slabPoolCap(): number {
  // Step 1: Read optional per-key pool cap override.
  const configuredCap = config.slabPoolMaxPerKey;
  if (configuredCap === undefined) return DEFAULT_POOL_CAP;
  // Step 2: Clamp to non-negative integer retention cap.
  return configuredCap < ZERO ? ZERO : configuredCap | 0;
}

/**
 * Creates a stable pool key from kind, element width, and length.
 *
 * @param kind - Short pool kind discriminator.
 * @param bytes - Bytes per element.
 * @param length - Typed array logical length.
 * @returns Stable pool key.
 */
function _poolKey(kind: string, bytes: number, length: number): string {
  // Step 1: Compose stable key used for pool bucket lookup.
  return `${kind}:${bytes}:${length}`;
}

/**
 * Acquires a typed array from pool or allocates a fresh one.
 *
 * @param kind - Pool kind discriminator.
 * @param ctor - Typed array constructor.
 * @param length - Desired typed array length.
 * @param bytesPerElement - Element byte width for keying.
 * @returns Acquired typed array.
 */
export function _acquireTA(
  kind: string,
  ctor: TypedArrayConstructor,
  length: number,
  bytesPerElement: number,
): TypedArray {
  // Step 1: Allocate directly when pooling is disabled.
  if (!config.enableSlabArrayPooling) {
    _slabAllocStats.fresh++;
    return new ctor(length);
  }

  // Step 2: Attempt to reuse a retained typed array for this key.
  const key = _poolKey(kind, bytesPerElement, length);
  const retainedByKey = _slabArrayPool[key];
  if (retainedByKey && retainedByKey.length) {
    _slabAllocStats.pooled++;
    (_slabPoolMetrics[key] ||= { created: 0, reused: 0, maxRetained: 0 })
      .reused++;
    return retainedByKey.pop() as TypedArray;
  }

  // Step 3: Record miss and allocate a fresh array.
  _slabAllocStats.fresh++;
  (_slabPoolMetrics[key] ||= { created: 0, reused: 0, maxRetained: 0 })
    .created++;
  return new ctor(length);
}

/**
 * Releases a typed array back to bounded per-key pool.
 *
 * @param kind - Pool kind discriminator.
 * @param bytesPerElement - Element byte width for keying.
 * @param arr - Typed array instance to retain when room exists.
 * @returns Nothing.
 */
export function _releaseTA(
  kind: string,
  bytesPerElement: number,
  arr: TypedArray,
): void {
  // Step 1: Ignore release when pooling is disabled.
  if (!config.enableSlabArrayPooling) {
    return;
  }

  // Step 2: Return array to bounded per-key retention bucket.
  const key = _poolKey(kind, bytesPerElement, arr.length);
  const retainedByKey = (_slabArrayPool[key] ||= []);
  if (retainedByKey.length < _slabPoolCap()) {
    retainedByKey.push(arr);
  }

  const poolMetrics = (_slabPoolMetrics[key] ||= {
    created: 0,
    reused: 0,
    maxRetained: 0,
  });
  if (retainedByKey.length > poolMetrics.maxRetained) {
    // Step 3: Track maximum retained depth for diagnostics.
    poolMetrics.maxRetained = retainedByKey.length;
  }
}

/**
 * Returns allocation stats snapshot for slab typed arrays.
 *
 * @returns Serializable snapshot of fresh, pooled, and per-key metrics.
 */
export function _getSlabAllocationStatsSnapshot() {
  // Step 1: Return shallow-cloned counters to avoid external mutation.
  return { ..._slabAllocStats, pool: { ..._slabPoolMetrics } };
}
