import { activationArrayPool } from './activationArrayPool';

interface RuntimeActivationArrayPool {
  clear: () => void;
  compact: (maxRetainedBuckets?: number) => void;
  setMaxPerBucket: (maxPerBucket: number) => void;
  prewarm: (size: number, count: number) => void;
  bucketSize: (size: number) => number;
}

describe('activationArrayPool capacity limiting', () => {
  describe('given a bucket cap smaller than the prewarm request', () => {
    describe('when prewarm is called', () => {
      it('retains no more than the configured bucket capacity', () => {
        // Arrange
        const pool =
          activationArrayPool as unknown as RuntimeActivationArrayPool;
        pool.clear();
        pool.setMaxPerBucket(2);
        pool.prewarm(4, 5);

        // Act
        const retainedBucketSize = pool.bucketSize(4);

        // Assert
        expect(retainedBucketSize).toBe(2);
      });
    });
  });

  describe('given compaction must shrink the retained bucket set', () => {
    describe('when one bucket was touched more recently than the others', () => {
      it('evicts the least recently used bucket first', () => {
        // Arrange
        const pool =
          activationArrayPool as unknown as RuntimeActivationArrayPool;
        pool.clear();
        pool.setMaxPerBucket(Number.POSITIVE_INFINITY);

        const firstBucketArray = activationArrayPool.acquire(2);
        activationArrayPool.release(firstBucketArray);

        const secondBucketArray = activationArrayPool.acquire(4);
        activationArrayPool.release(secondBucketArray);

        const thirdBucketArray = activationArrayPool.acquire(6);
        activationArrayPool.release(thirdBucketArray);

        const refreshedBucketArray = activationArrayPool.acquire(2);
        activationArrayPool.release(refreshedBucketArray);

        // Act
        pool.compact(2);

        // Assert
        expect({
          firstBucketSize: pool.bucketSize(2),
          secondBucketSize: pool.bucketSize(4),
          thirdBucketSize: pool.bucketSize(6),
        }).toEqual({
          firstBucketSize: 1,
          secondBucketSize: 0,
          thirdBucketSize: 1,
        });
      });
    });
  });
});
