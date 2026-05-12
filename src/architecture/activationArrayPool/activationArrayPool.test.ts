import { config } from '../../config';
import { defaultMemoryManager } from '../../memory/manager';
import { activationArrayPool } from './activationArrayPool';

describe('activationArrayPool', () => {
  describe('given default array mode', () => {
    beforeEach(() => {
      config.float32Mode = false;
      activationArrayPool.clear();
    });

    describe('when acquiring a fresh buffer', () => {
      it('returns an array with the requested length', () => {
        // Arrange
        const requestedSize = 5;

        // Act
        const acquiredArray = activationArrayPool.acquire(requestedSize);

        // Assert
        expect(acquiredArray.length).toBe(requestedSize);
      });
    });

    describe('when re-acquiring a released buffer', () => {
      it('returns the same array reference from the pool', () => {
        // Arrange
        const requestedSize = 5;
        const acquiredArray = activationArrayPool.acquire(requestedSize);
        activationArrayPool.release(acquiredArray);

        // Act
        const reusedArray = activationArrayPool.acquire(requestedSize);

        // Assert
        expect(reusedArray).toBe(acquiredArray);
      });

      it('zero-fills the recycled buffer before reuse', () => {
        // Arrange
        const requestedSize = 5;
        const acquiredArray = activationArrayPool.acquire(requestedSize);
        acquiredArray[0] = 123;
        activationArrayPool.release(acquiredArray);

        // Act
        const reusedArray = activationArrayPool.acquire(requestedSize);

        // Assert
        expect(reusedArray[0]).toBe(0);
      });
    });

    describe('when clearing the pool after a release', () => {
      it('drops the retained reference', () => {
        // Arrange
        const requestedSize = 5;
        const acquiredArray = activationArrayPool.acquire(requestedSize);
        activationArrayPool.release(acquiredArray);
        activationArrayPool.clear();

        // Act
        const reacquiredArray = activationArrayPool.acquire(requestedSize);

        // Assert
        expect(reacquiredArray === acquiredArray).toBe(false);
      });
    });
  });

  describe('given float32 mode', () => {
    beforeEach(() => {
      config.float32Mode = true;
      activationArrayPool.clear();
    });

    afterEach(() => {
      config.float32Mode = false;
    });

    describe('when acquiring a fresh buffer', () => {
      it('returns a Float32Array instance', () => {
        // Arrange
        const requestedSize = 3;

        // Act
        const acquiredArray = activationArrayPool.acquire(requestedSize);

        // Assert
        expect(acquiredArray instanceof Float32Array).toBe(true);
      });
    });
  });

  describe('given the pool is registered with the default memory manager', () => {
    afterEach(() => {
      defaultMemoryManager.teardown();
      activationArrayPool.clear();
    });

    describe('when the registered pool snapshot and reset callbacks run', () => {
      it('reports pool stats and clears retained buckets through the manager surface', () => {
        // Arrange
        const requestedSize = 4;
        const acquiredArray = activationArrayPool.acquire(requestedSize);
        activationArrayPool.release(acquiredArray);
        defaultMemoryManager.init();

        // Act
        const snapshot = defaultMemoryManager.getPoolStats<{
          bucketCount: number;
        }>('activationArrayPool');
        defaultMemoryManager.teardown();

        // Assert
        expect({
          bucketCount: snapshot?.bucketCount ?? null,
          bucketSizeAfterTeardown:
            activationArrayPool.bucketSize(requestedSize),
        }).toStrictEqual({
          bucketCount: 1,
          bucketSizeAfterTeardown: 0,
        });
      });
    });
  });
});
