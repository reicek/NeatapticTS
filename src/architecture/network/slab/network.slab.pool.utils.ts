/**
 * Internal slab pool/stat helpers extracted from network.slab.utils.ts.
 */
import { defaultMemoryManager } from '../../../memory/manager';
import type {
  PoolKeyMetrics,
  TypedArray,
  TypedArrayConstructor,
} from './network.slab.utils.types';

const ZERO = 0;

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
  return defaultMemoryManager.allocateTypedArray(
    kind,
    ctor,
    length,
    bytesPerElement,
  ) as TypedArray;
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
  defaultMemoryManager.releaseTypedArray(kind, bytesPerElement, arr);
}

/**
 * Returns allocation stats snapshot for slab typed arrays.
 *
 * @returns Serializable snapshot of fresh, pooled, and per-key metrics.
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
