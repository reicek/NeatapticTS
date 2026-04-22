import { config } from '../../config';
import { activationArrayPool } from './activationArrayPool';

describe('activationArrayPool direct coverage chapter', () => {
  beforeEach(() => {
    config.float32Mode = false;
    activationArrayPool.clear();
    activationArrayPool.setMaxPerBucket(Number.POSITIVE_INFINITY);
  });

  afterEach(() => {
    config.float32Mode = false;
    activationArrayPool.clear();
    activationArrayPool.setMaxPerBucket(Number.POSITIVE_INFINITY);
  });

  describe('given float32 mode reuses a retained typed buffer', () => {
    it('zero-fills the recycled Float32Array before reuse', () => {
      // Arrange
      config.float32Mode = true;
      const retainedArray = activationArrayPool.acquire(4);

      if (!(retainedArray instanceof Float32Array)) {
        throw new Error('Expected a Float32Array when float32 mode is enabled.');
      }

      retainedArray[0] = 7;
      activationArrayPool.release(retainedArray);

      // Act
      const reusedArray = activationArrayPool.acquire(4);

      // Assert
      expect({
        firstValue: reusedArray[0],
        sameReference: reusedArray === retainedArray,
      }).toEqual({
        firstValue: 0,
        sameReference: true,
      });
    });
  });

  describe('given a released Float64Array is reused from the pool', () => {
    it('zero-fills the recycled Float64Array before reuse', () => {
      // Arrange
      const retainedArray = new Float64Array([9, 8]);
      activationArrayPool.release(retainedArray);

      // Act
      const reusedArray = activationArrayPool.acquire(2);

      // Assert
      expect({
        firstValue: reusedArray[0],
        sameReference: reusedArray === retainedArray,
      }).toEqual({
        firstValue: 0,
        sameReference: true,
      });
    });
  });

  describe('given a bucket is already at its retention cap', () => {
    it('drops the extra released array instead of growing the bucket', () => {
      // Arrange
      activationArrayPool.setMaxPerBucket(1);
      const firstArray = activationArrayPool.acquire(2);
      const secondArray = activationArrayPool.acquire(2);
      activationArrayPool.release(firstArray);

      // Act
      activationArrayPool.release(secondArray);

      // Assert
      expect(activationArrayPool.bucketSize(2)).toBe(1);
    });
  });

  describe('given stats are requested after one creation and one reuse', () => {
    it('returns the current pool counters', () => {
      // Arrange
      const createdArray = activationArrayPool.acquire(3);
      activationArrayPool.release(createdArray);
      activationArrayPool.acquire(3);

      // Act
      const poolStats = activationArrayPool.stats();

      // Assert
      expect(poolStats).toEqual({
        bucketCount: 1,
        created: 1,
        reused: 1,
      });
    });
  });

  describe('given float32 mode prewarms a bucket', () => {
    it('retains a Float32Array for the later acquisition', () => {
      // Arrange
      config.float32Mode = true;
      activationArrayPool.prewarm(5, 1);

      // Act
      const prewarmedArray = activationArrayPool.acquire(5);

      // Assert
      expect(prewarmedArray instanceof Float32Array).toBe(true);
    });
  });

  describe('given an invalid cap and normalized prewarm counts are provided', () => {
    it('ignores the invalid cap, skips the negative request, and floors the fractional request', () => {
      // Arrange
      activationArrayPool.setMaxPerBucket(-1);
      activationArrayPool.prewarm(3, -2);

      // Act
      activationArrayPool.prewarm(3, 2.8);

      // Assert
      expect(activationArrayPool.bucketSize(3)).toBe(2);
    });
  });

  describe('given a missing bucket size is requested', () => {
    it('returns zero', () => {
      // Arrange
      const missingBucketSize = 99;

      // Act
      const retainedBucketSize = activationArrayPool.bucketSize(missingBucketSize);

      // Assert
      expect(retainedBucketSize).toBe(0);
    });
  });
});