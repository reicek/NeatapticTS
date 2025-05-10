import Rate from '../../src/methods/rate';

describe('Rate', () => {
  const baseRate = 0.1;

  describe('fixed()', () => {
    // Arrange: Get the fixed rate function
    const fixedRateFn = Rate.fixed();

    test('should return the base rate at iteration 0', () => {
      // Act: Calculate rate at iteration 0
      const rate = fixedRateFn(baseRate, 0);
      // Assert: Rate should equal baseRate and be non-negative
      expect(rate).toBe(baseRate);
      expect(rate).toBeGreaterThanOrEqual(0);
    });

    test('should return the base rate at iteration 100', () => {
      // Act: Calculate rate at iteration 100
      const rate = fixedRateFn(baseRate, 100);
      // Assert: Rate should equal baseRate and be non-negative
      expect(rate).toBe(baseRate);
      expect(rate).toBeGreaterThanOrEqual(0);
    });
  });

  describe('step()', () => {
    const gamma = 0.9;
    const stepSize = 10;

    // Arrange: Get the step rate function with custom parameters
    const stepRateFnCustom = Rate.step(gamma, stepSize);
    // Arrange: Get the step rate function with default parameters
    const stepRateFnDefault = Rate.step(); // gamma=0.9, stepSize=100

    test('should return the base rate before the first step (custom)', () => {
      // Act: Calculate rate at iteration 9
      const rate = stepRateFnCustom(baseRate, 9);
      // Assert: Rate should equal baseRate and be non-negative
      expect(rate).toBe(baseRate);
      expect(rate).toBeGreaterThanOrEqual(0);
    });

    test('should return the decayed rate at the first step (custom)', () => {
      // Act: Calculate rate at iteration 10
      const rate = stepRateFnCustom(baseRate, 10);
      // Assert: Rate should be baseRate * gamma^1, non-negative, and <= baseRate
      expect(rate).toBeCloseTo(baseRate * Math.pow(gamma, 1));
      expect(rate).toBeGreaterThanOrEqual(0);
      expect(rate).toBeLessThanOrEqual(baseRate);
    });

    test('should return the decayed rate after multiple steps (custom)', () => {
      // Act: Calculate rate at iteration 25 (floor(25/10) = 2 steps)
      const rate = stepRateFnCustom(baseRate, 25);
      // Assert: Rate should be baseRate * gamma^2, non-negative, and <= baseRate
      expect(rate).toBeCloseTo(baseRate * Math.pow(gamma, 2));
      expect(rate).toBeGreaterThanOrEqual(0);
      expect(rate).toBeLessThanOrEqual(baseRate);
    });

    test('should return the base rate before the first step (default)', () => {
      // Act: Calculate rate at iteration 99
      const rate = stepRateFnDefault(baseRate, 99);
      // Assert: Rate should equal baseRate and be non-negative
      expect(rate).toBe(baseRate);
      expect(rate).toBeGreaterThanOrEqual(0);
    });

    test('should return the decayed rate at the first step (default)', () => {
      // Act: Calculate rate at iteration 100
      const rate = stepRateFnDefault(baseRate, 100);
      // Assert: Rate should be baseRate * 0.9^1, non-negative, and <= baseRate
      expect(rate).toBeCloseTo(baseRate * Math.pow(0.9, 1));
      expect(rate).toBeGreaterThanOrEqual(0);
      expect(rate).toBeLessThanOrEqual(baseRate);
    });
  });

  describe('exp()', () => {
    const gamma = 0.95;

    // Arrange: Get the exponential rate function with custom gamma
    const expRateFnCustom = Rate.exp(gamma);
    // Arrange: Get the exponential rate function with default gamma
    const expRateFnDefault = Rate.exp(); // gamma=0.999

    test('should return the base rate at iteration 0 (custom)', () => {
      // Act: Calculate rate at iteration 0
      const rate = expRateFnCustom(baseRate, 0);
      // Assert: Rate should be baseRate * gamma^0 = baseRate and non-negative
      expect(rate).toBe(baseRate);
      expect(rate).toBeGreaterThanOrEqual(0);
    });

    test('should return the decayed rate at iteration 1 (custom)', () => {
      // Act: Calculate rate at iteration 1
      const rate = expRateFnCustom(baseRate, 1);
      // Assert: Rate should be baseRate * gamma^1, non-negative, and <= baseRate
      expect(rate).toBeCloseTo(baseRate * Math.pow(gamma, 1));
      expect(rate).toBeGreaterThanOrEqual(0);
      expect(rate).toBeLessThanOrEqual(baseRate);
    });

    test('should return the decayed rate at iteration 10 (custom)', () => {
      // Act: Calculate rate at iteration 10
      const rate = expRateFnCustom(baseRate, 10);
      // Assert: Rate should be baseRate * gamma^10, non-negative, and <= baseRate
      expect(rate).toBeCloseTo(baseRate * Math.pow(gamma, 10));
      expect(rate).toBeGreaterThanOrEqual(0);
      expect(rate).toBeLessThanOrEqual(baseRate);
    });

    test('should return the base rate at iteration 0 (default)', () => {
      // Act: Calculate rate at iteration 0
      const rate = expRateFnDefault(baseRate, 0);
      // Assert: Rate should be baseRate * 0.999^0 = baseRate and non-negative
      expect(rate).toBe(baseRate);
      expect(rate).toBeGreaterThanOrEqual(0);
    });

    test('should return the decayed rate at iteration 10 (default)', () => {
      // Act: Calculate rate at iteration 10
      const rate = expRateFnDefault(baseRate, 10);
      // Assert: Rate should be baseRate * 0.999^10, non-negative, and <= baseRate
      expect(rate).toBeCloseTo(baseRate * Math.pow(0.999, 10));
      expect(rate).toBeGreaterThanOrEqual(0);
      expect(rate).toBeLessThanOrEqual(baseRate);
    });
  });

  describe('inv()', () => {
    const gamma = 0.01;
    const power = 1.5;

    // Arrange: Get the inverse rate function with custom parameters
    const invRateFnCustom = Rate.inv(gamma, power);
    // Arrange: Get the inverse rate function with default parameters
    const invRateFnDefault = Rate.inv(); // gamma=0.001, power=2

    test('should return the decayed rate at iteration 0 (custom)', () => {
      // Act: Calculate rate at iteration 0 (uses currentIter = 1)
      const rate = invRateFnCustom(baseRate, 0);
      // Assert: Rate should be baseRate * (1 + gamma * 1)^(-power), non-negative, and <= baseRate
      expect(rate).toBeCloseTo(baseRate * Math.pow(1 + gamma * 1, -power));
      expect(rate).toBeGreaterThanOrEqual(0);
      expect(rate).toBeLessThanOrEqual(baseRate);
    });

    test('should return the decayed rate at iteration 9 (custom)', () => {
      // Act: Calculate rate at iteration 9 (uses currentIter = 10)
      const rate = invRateFnCustom(baseRate, 9);
      // Assert: Rate should be baseRate * (1 + gamma * 10)^(-power), non-negative, and <= baseRate
      expect(rate).toBeCloseTo(baseRate * Math.pow(1 + gamma * 10, -power));
      expect(rate).toBeGreaterThanOrEqual(0);
      expect(rate).toBeLessThanOrEqual(baseRate);
    });

    test('should return the decayed rate at iteration 0 (default)', () => {
      // Act: Calculate rate at iteration 0 (uses currentIter = 1)
      const rate = invRateFnDefault(baseRate, 0);
      // Assert: Rate should be baseRate * (1 + 0.001 * 1)^(-2), non-negative, and <= baseRate
      expect(rate).toBeCloseTo(baseRate * Math.pow(1 + 0.001 * 1, -2));
      expect(rate).toBeGreaterThanOrEqual(0);
      expect(rate).toBeLessThanOrEqual(baseRate);
    });

    test('should return the decayed rate at iteration 99 (default)', () => {
      // Act: Calculate rate at iteration 99 (uses currentIter = 100)
      const rate = invRateFnDefault(baseRate, 99);
      // Assert: Rate should be baseRate * (1 + 0.001 * 100)^(-2), non-negative, and <= baseRate
      expect(rate).toBeCloseTo(baseRate * Math.pow(1 + 0.001 * 100, -2));
      expect(rate).toBeGreaterThanOrEqual(0);
      expect(rate).toBeLessThanOrEqual(baseRate);
    });
  });

  describe('cosineAnnealing()', () => {
    const period = 100;
    const minRate = 0.01;

    // Arrange: Get the cosine annealing function with custom parameters
    const cosineFnCustom = Rate.cosineAnnealing(period, minRate);
    // Arrange: Get the cosine annealing function with default parameters
    const cosineFnDefault = Rate.cosineAnnealing(); // period=1000, minRate=0

    test('should return the base rate at the start of the cycle (custom)', () => {
      // Act: Calculate rate at iteration 0
      const rate = cosineFnCustom(baseRate, 0);
      // Assert: Rate should be close to baseRate and within [minRate, baseRate]
      expect(rate).toBeCloseTo(baseRate);
      expect(rate).toBeGreaterThanOrEqual(minRate);
      expect(rate).toBeLessThanOrEqual(baseRate);
    });

    test('should return the average rate at the middle of the cycle (custom)', () => {
      // Act: Calculate rate at iteration period / 2 = 50
      const rate = cosineFnCustom(baseRate, period / 2);
      // Assert: Rate should be close to (baseRate + minRate) / 2 and within [minRate, baseRate]
      expect(rate).toBeCloseTo(minRate + (baseRate - minRate) * 0.5);
      expect(rate).toBeGreaterThanOrEqual(minRate);
      expect(rate).toBeLessThanOrEqual(baseRate);
    });

    test('should return the min rate near the end of the cycle (custom)', () => {
      // Act: Calculate rate at iteration period - 1 = 99
      const rate = cosineFnCustom(baseRate, period - 1);
      // Assert: Rate should be close to minRate and within [minRate, baseRate]
      expect(rate).toBeCloseTo(minRate, 2);
      expect(rate).toBeGreaterThanOrEqual(minRate);
      expect(rate).toBeLessThanOrEqual(baseRate);
    });

    test('should return the base rate at the start of the second cycle (custom)', () => {
      // Act: Calculate rate at iteration period = 100
      const rate = cosineFnCustom(baseRate, period);
      // Assert: Rate should be close to baseRate and within [minRate, baseRate]
      expect(rate).toBeCloseTo(baseRate);
      expect(rate).toBeGreaterThanOrEqual(minRate);
      expect(rate).toBeLessThanOrEqual(baseRate);
    });

    test('should return the base rate at the start of the cycle (default)', () => {
      // Act: Calculate rate at iteration 0
      const rate = cosineFnDefault(baseRate, 0);
      // Assert: Rate should be close to baseRate and within [0, baseRate]
      expect(rate).toBeCloseTo(baseRate);
      expect(rate).toBeGreaterThanOrEqual(0); // Default minRate is 0
      expect(rate).toBeLessThanOrEqual(baseRate);
    });

    test('should return half the base rate at the middle of the cycle (default)', () => {
      // Act: Calculate rate at iteration 1000 / 2 = 500
      const rate = cosineFnDefault(baseRate, 500);
      // Assert: Rate should be close to baseRate / 2 and within [0, baseRate]
      expect(rate).toBeCloseTo(baseRate * 0.5);
      expect(rate).toBeGreaterThanOrEqual(0);
      expect(rate).toBeLessThanOrEqual(baseRate);
    });

    test('should return close to zero near the end of the cycle (default)', () => {
      // Act: Calculate rate at iteration 1000 - 1 = 999
      const rate = cosineFnDefault(baseRate, 999);
      // Assert: Rate should be close to 0 and within [0, baseRate]
      expect(rate).toBeCloseTo(0, 2);
      expect(rate).toBeGreaterThanOrEqual(0);
      expect(rate).toBeLessThanOrEqual(baseRate);
    });
  });
});
