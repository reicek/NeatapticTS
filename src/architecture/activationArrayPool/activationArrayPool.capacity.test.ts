import { activationArrayPool } from './activationArrayPool';

interface RuntimeActivationArrayPool {
  clear: () => void;
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
});
