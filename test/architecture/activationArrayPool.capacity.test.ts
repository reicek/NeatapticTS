import { activationArrayPool } from '../../src/architecture/activationArrayPool';

/**
 * Capacity limiting & prewarm coverage: set small cap, prewarm beyond cap, ensure bucket never exceeds cap.
 */
describe('ActivationArrayPool capacity limiting', () => {
  describe('Scenario: prewarm respects maxPerBucket cap', () => {
    it('does not exceed configured bucket capacity', () => {
      // Arrange
      activationArrayPool.clear();
      (activationArrayPool as any).setMaxPerBucket(2);
      (activationArrayPool as any).prewarm(4, 5); // request > cap
      // Act
      const bucketSize = (activationArrayPool as any).bucketSize(4);
      // Assert
      expect(bucketSize).toBe(2);
    });
  });
});
