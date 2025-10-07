import { activationArrayPool } from '../../src/architecture/activationArrayPool';

/**
 * Runtime interface for accessing private methods of activationArrayPool in tests.
 */
interface RuntimeActivationArrayPool {
  clear: () => void;
  setMaxPerBucket: (max: number) => void;
  prewarm: (size: number, count: number) => void;
  bucketSize: (size: number) => number;
}

/**
 * Capacity limiting & prewarm coverage: set small cap, prewarm beyond cap, ensure bucket never exceeds cap.
 */
describe('ActivationArrayPool capacity limiting', () => {
  describe('Scenario: prewarm respects maxPerBucket cap', () => {
    it('does not exceed configured bucket capacity', () => {
      // Arrange
      const pool = activationArrayPool as unknown as RuntimeActivationArrayPool;
      pool.clear();
      pool.setMaxPerBucket(2);
      pool.prewarm(4, 5); // request > cap
      // Act
      const bucketSize = pool.bucketSize(4);
      // Assert
      expect(bucketSize).toBe(2);
    });
  });
});
