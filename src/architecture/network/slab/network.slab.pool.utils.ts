/**
 * Internal helpers for typed-array slab allocation, release, and pool stats.
 */
import { defaultMemoryManager } from '../../../memory/manager';
import type {
  PoolKeyMetrics,
  TypedArray,
  TypedArrayConstructor,
} from './network.slab.utils.types';

/**
 * Acquire a typed-array slab for the requested key, reusing pooled capacity
 * when available and allocating only on pool miss.
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
  return defaultMemoryManager.allocateTypedArray(
    kind,
    ctor,
    length,
    bytesPerElement,
  ) as TypedArray;
}

/**
 * Return a typed array to its bounded per-key slab pool so later activation
 * passes can reuse capacity without reallocating.
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
  defaultMemoryManager.releaseTypedArray(kind, bytesPerElement, arr);
}

/**
 * Produce a serializable view of slab allocation telemetry, including global
 * fresh-versus-pooled counts and per-key pool depth.
 *
 * @returns Snapshot containing fresh, pooled, and per-key counters.
 */
export function _getSlabAllocationStatsSnapshot() {
  return defaultMemoryManager.getTypedArrayAllocationStats() as {
    fresh: number;
    pooled: number;
    pool: Record<string, PoolKeyMetrics>;
  };
}

defaultMemoryManager.registerPool('slabArrayPool', {
  stats: _getSlabAllocationStatsSnapshot,
});
