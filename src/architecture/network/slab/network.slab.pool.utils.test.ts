import type { config as sharedConfig } from '../../../config';
import type {
  _acquireTA,
  _getSlabAllocationStatsSnapshot,
  _releaseTA,
} from './network.slab.pool.utils';

type PoolHelperModule = {
  config: typeof sharedConfig;
  _acquireTA: typeof _acquireTA;
  _getSlabAllocationStatsSnapshot: typeof _getSlabAllocationStatsSnapshot;
  _releaseTA: typeof _releaseTA;
};

async function loadPoolHelpers(): Promise<PoolHelperModule> {
  jest.resetModules();

  const { config } = await import('../../../config');
  const poolHelpers = await import('./network.slab.pool.utils');

  return {
    config,
    _acquireTA: poolHelpers._acquireTA,
    _getSlabAllocationStatsSnapshot:
      poolHelpers._getSlabAllocationStatsSnapshot,
    _releaseTA: poolHelpers._releaseTA,
  };
}

describe('network slab chapter', () => {
  describe('slab pool utility helpers', () => {
    afterEach(() => {
      jest.restoreAllMocks();
      jest.resetModules();
    });

    describe('given slab array pooling is disabled', () => {
      describe('when one typed array is acquired and released', () => {
        it('records one fresh allocation and keeps the pool snapshot empty', async () => {
          // Arrange
          const {
            config,
            _acquireTA: acquireTypedArray,
            _getSlabAllocationStatsSnapshot: getAllocationStatsSnapshot,
            _releaseTA: releaseTypedArray,
          } = await loadPoolHelpers();
          config.enableSlabArrayPooling = false;
          const initialSnapshot = getAllocationStatsSnapshot();

          // Act
          const acquiredArray = acquireTypedArray(
            'disabled-path',
            Float32Array,
            3,
            Float32Array.BYTES_PER_ELEMENT,
          ) as Float32Array;
          releaseTypedArray(
            'disabled-path',
            Float32Array.BYTES_PER_ELEMENT,
            acquiredArray,
          );
          const finalSnapshot = getAllocationStatsSnapshot();

          // Assert
          expect({
            freshDelta: finalSnapshot.fresh - initialSnapshot.fresh,
            poolKeys: Object.keys(finalSnapshot.pool).length,
            pooledDelta: finalSnapshot.pooled - initialSnapshot.pooled,
          }).toStrictEqual({
            freshDelta: 1,
            poolKeys: 0,
            pooledDelta: 0,
          });
        });
      });
    });

    describe('given slab array pooling uses the default per-key cap', () => {
      describe('when one retained typed array is acquired again', () => {
        it('reuses the retained array and records one pooled reuse', async () => {
          // Arrange
          const {
            config,
            _acquireTA: acquireTypedArray,
            _getSlabAllocationStatsSnapshot: getAllocationStatsSnapshot,
            _releaseTA: releaseTypedArray,
          } = await loadPoolHelpers();
          config.enableSlabArrayPooling = true;
          delete config.slabPoolMaxPerKey;
          const poolKey = 'default-cap:4:5';
          const firstArray = acquireTypedArray(
            'default-cap',
            Float32Array,
            5,
            Float32Array.BYTES_PER_ELEMENT,
          ) as Float32Array;

          releaseTypedArray(
            'default-cap',
            Float32Array.BYTES_PER_ELEMENT,
            firstArray,
          );

          // Act
          const reacquiredArray = acquireTypedArray(
            'default-cap',
            Float32Array,
            5,
            Float32Array.BYTES_PER_ELEMENT,
          ) as Float32Array;
          const finalSnapshot = getAllocationStatsSnapshot();

          // Assert
          expect({
            maxRetained: finalSnapshot.pool[poolKey]?.maxRetained,
            reusedArray: reacquiredArray === firstArray,
            reusedCount: finalSnapshot.pool[poolKey]?.reused,
          }).toStrictEqual({
            maxRetained: 1,
            reusedArray: true,
            reusedCount: 1,
          });
        });
      });
    });

    describe('given slab array pooling uses a negative per-key cap override', () => {
      describe('when one typed array is released back to the pool', () => {
        it('clamps retention to zero and allocates a fresh replacement later', async () => {
          // Arrange
          const {
            config,
            _acquireTA: acquireTypedArray,
            _getSlabAllocationStatsSnapshot: getAllocationStatsSnapshot,
            _releaseTA: releaseTypedArray,
          } = await loadPoolHelpers();
          config.enableSlabArrayPooling = true;
          config.slabPoolMaxPerKey = -2;
          const poolKey = 'negative-cap:4:4';
          const firstArray = acquireTypedArray(
            'negative-cap',
            Float32Array,
            4,
            Float32Array.BYTES_PER_ELEMENT,
          ) as Float32Array;

          releaseTypedArray(
            'negative-cap',
            Float32Array.BYTES_PER_ELEMENT,
            firstArray,
          );

          // Act
          const secondArray = acquireTypedArray(
            'negative-cap',
            Float32Array,
            4,
            Float32Array.BYTES_PER_ELEMENT,
          ) as Float32Array;
          const finalSnapshot = getAllocationStatsSnapshot();

          // Assert
          expect({
            createdCount: finalSnapshot.pool[poolKey]?.created,
            maxRetained: finalSnapshot.pool[poolKey]?.maxRetained,
            reusedArray: secondArray === firstArray,
          }).toStrictEqual({
            createdCount: 2,
            maxRetained: 0,
            reusedArray: false,
          });
        });
      });
    });

    describe('given slab array pooling uses a fractional positive per-key cap override', () => {
      describe('when two same-key arrays are released back to the pool', () => {
        it('retains only one array because the cap is truncated to an integer', async () => {
          // Arrange
          const {
            config,
            _acquireTA: acquireTypedArray,
            _getSlabAllocationStatsSnapshot: getAllocationStatsSnapshot,
            _releaseTA: releaseTypedArray,
          } = await loadPoolHelpers();
          config.enableSlabArrayPooling = true;
          config.slabPoolMaxPerKey = 1.8;
          const poolKey = 'fractional-cap:4:6';
          const firstArray = acquireTypedArray(
            'fractional-cap',
            Float32Array,
            6,
            Float32Array.BYTES_PER_ELEMENT,
          ) as Float32Array;
          const secondArray = acquireTypedArray(
            'fractional-cap',
            Float32Array,
            6,
            Float32Array.BYTES_PER_ELEMENT,
          ) as Float32Array;

          releaseTypedArray(
            'fractional-cap',
            Float32Array.BYTES_PER_ELEMENT,
            firstArray,
          );
          releaseTypedArray(
            'fractional-cap',
            Float32Array.BYTES_PER_ELEMENT,
            secondArray,
          );

          // Act
          const reusedArray = acquireTypedArray(
            'fractional-cap',
            Float32Array,
            6,
            Float32Array.BYTES_PER_ELEMENT,
          ) as Float32Array;
          const freshReplacement = acquireTypedArray(
            'fractional-cap',
            Float32Array,
            6,
            Float32Array.BYTES_PER_ELEMENT,
          ) as Float32Array;
          const finalSnapshot = getAllocationStatsSnapshot();

          // Assert
          expect({
            createdCount: finalSnapshot.pool[poolKey]?.created,
            maxRetained: finalSnapshot.pool[poolKey]?.maxRetained,
            pooledCount: finalSnapshot.pooled,
            reusedMatchesOneReleasedArray:
              reusedArray === firstArray || reusedArray === secondArray,
            secondAcquireIsFresh: freshReplacement !== reusedArray,
          }).toStrictEqual({
            createdCount: 3,
            maxRetained: 1,
            pooledCount: 1,
            reusedMatchesOneReleasedArray: true,
            secondAcquireIsFresh: true,
          });
        });
      });
    });
  });
});