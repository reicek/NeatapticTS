import Rate from '../../src/methods/rate';

describe('Rate', () => {
  const baseRate = 0.1;

  describe('fixed()', () => {
    const fixedRateFn = Rate.fixed();

    describe('Scenario: At iteration 0', () => {
      test('should return the base rate', () => {
        // Act
        const rate = fixedRateFn(baseRate, 0);
        // Assert
        expect(rate).toBe(baseRate);
      });
      test('should be non-negative', () => {
        // Act
        const rate = fixedRateFn(baseRate, 0);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });
    });

    describe('Scenario: At iteration 100', () => {
      test('should return the base rate', () => {
        // Act
        const rate = fixedRateFn(baseRate, 100);
        // Assert
        expect(rate).toBe(baseRate);
      });
      test('should be non-negative', () => {
        // Act
        const rate = fixedRateFn(baseRate, 100);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });
    });
  });

  describe('step()', () => {
    const gamma = 0.9;
    const stepSize = 10;
    const stepRateFnCustom = Rate.step(gamma, stepSize);
    const stepRateFnDefault = Rate.step();

    describe('Custom parameters', () => {
      test('should return the base rate before the first step', () => {
        // Act
        const rate = stepRateFnCustom(baseRate, 9);
        // Assert
        expect(rate).toBe(baseRate);
      });
      test('should be non-negative before the first step', () => {
        // Act
        const rate = stepRateFnCustom(baseRate, 9);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });
      test('should return the decayed rate at the first step', () => {
        // Act
        const rate = stepRateFnCustom(baseRate, 10);
        // Assert
        expect(rate).toBeCloseTo(baseRate * Math.pow(gamma, 1));
      });
      test('should be non-negative at the first step', () => {
        // Act
        const rate = stepRateFnCustom(baseRate, 10);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });
      test('should be less than or equal to baseRate at the first step', () => {
        // Act
        const rate = stepRateFnCustom(baseRate, 10);
        // Assert
        expect(rate).toBeLessThanOrEqual(baseRate);
      });
      test('should return the decayed rate after multiple steps', () => {
        // Act
        const rate = stepRateFnCustom(baseRate, 25);
        // Assert
        expect(rate).toBeCloseTo(baseRate * Math.pow(gamma, 2));
      });
      test('should be non-negative after multiple steps', () => {
        // Act
        const rate = stepRateFnCustom(baseRate, 25);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });
      test('should be less than or equal to baseRate after multiple steps', () => {
        // Act
        const rate = stepRateFnCustom(baseRate, 25);
        // Assert
        expect(rate).toBeLessThanOrEqual(baseRate);
      });
    });

    describe('Default parameters', () => {
      test('should return the base rate before the first step', () => {
        // Act
        const rate = stepRateFnDefault(baseRate, 99);
        // Assert
        expect(rate).toBe(baseRate);
      });
      test('should be non-negative before the first step', () => {
        // Act
        const rate = stepRateFnDefault(baseRate, 99);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });
      test('should return the decayed rate at the first step', () => {
        // Act
        const rate = stepRateFnDefault(baseRate, 100);
        // Assert
        expect(rate).toBeCloseTo(baseRate * Math.pow(0.9, 1));
      });
      test('should be non-negative at the first step', () => {
        // Act
        const rate = stepRateFnDefault(baseRate, 100);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });
      test('should be less than or equal to baseRate at the first step', () => {
        // Act
        const rate = stepRateFnDefault(baseRate, 100);
        // Assert
        expect(rate).toBeLessThanOrEqual(baseRate);
      });
    });
  });

  describe('exp()', () => {
    const gamma = 0.95;
    const expRateFnCustom = Rate.exp(gamma);
    const expRateFnDefault = Rate.exp();

    describe('Custom gamma', () => {
      test('should return the base rate at iteration 0', () => {
        // Act
        const rate = expRateFnCustom(baseRate, 0);
        // Assert
        expect(rate).toBe(baseRate);
      });
      test('should be non-negative at iteration 0', () => {
        // Act
        const rate = expRateFnCustom(baseRate, 0);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });
      test('should return the decayed rate at iteration 1', () => {
        // Act
        const rate = expRateFnCustom(baseRate, 1);
        // Assert
        expect(rate).toBeCloseTo(baseRate * Math.pow(gamma, 1));
      });
      test('should be non-negative at iteration 1', () => {
        // Act
        const rate = expRateFnCustom(baseRate, 1);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });
      test('should be less than or equal to baseRate at iteration 1', () => {
        // Act
        const rate = expRateFnCustom(baseRate, 1);
        // Assert
        expect(rate).toBeLessThanOrEqual(baseRate);
      });
      test('should return the decayed rate at iteration 10', () => {
        // Act
        const rate = expRateFnCustom(baseRate, 10);
        // Assert
        expect(rate).toBeCloseTo(baseRate * Math.pow(gamma, 10));
      });
      test('should be non-negative at iteration 10', () => {
        // Act
        const rate = expRateFnCustom(baseRate, 10);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });
      test('should be less than or equal to baseRate at iteration 10', () => {
        // Act
        const rate = expRateFnCustom(baseRate, 10);
        // Assert
        expect(rate).toBeLessThanOrEqual(baseRate);
      });
    });

    describe('Default gamma', () => {
      test('should return the base rate at iteration 0', () => {
        // Act
        const rate = expRateFnDefault(baseRate, 0);
        // Assert
        expect(rate).toBe(baseRate);
      });
      test('should be non-negative at iteration 0', () => {
        // Act
        const rate = expRateFnDefault(baseRate, 0);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });
      test('should return the decayed rate at iteration 10', () => {
        // Act
        const rate = expRateFnDefault(baseRate, 10);
        // Assert
        expect(rate).toBeCloseTo(baseRate * Math.pow(0.999, 10));
      });
      test('should be non-negative at iteration 10', () => {
        // Act
        const rate = expRateFnDefault(baseRate, 10);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });
      test('should be less than or equal to baseRate at iteration 10', () => {
        // Act
        const rate = expRateFnDefault(baseRate, 10);
        // Assert
        expect(rate).toBeLessThanOrEqual(baseRate);
      });
    });
  });

  describe('inv()', () => {
    const gamma = 0.01;
    const power = 1.5;
    const invRateFnCustom = Rate.inv(gamma, power);
    const invRateFnDefault = Rate.inv();

    describe('Custom parameters', () => {
      test('should return the base rate at iteration 0', () => {
        // Act
        const rate = invRateFnCustom(baseRate, 0);
        // Assert
        expect(rate).toBeCloseTo(baseRate, 5);
      });

      test('should be non-negative at iteration 0', () => {
        // Act
        const rate = invRateFnCustom(baseRate, 0);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });

      test('should return the decayed rate at iteration 1', () => {
        // Act
        const rate = invRateFnCustom(baseRate, 1);
        const expected = baseRate / (1 + gamma * Math.pow(1, power));
        // Assert
        expect(rate).toBeCloseTo(expected, 5);
      });

      test('should be non-negative at iteration 1', () => {
        // Act
        const rate = invRateFnCustom(baseRate, 1);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });

      test('should be less than or equal to baseRate at iteration 1', () => {
        // Act
        const rate = invRateFnCustom(baseRate, 1);
        // Assert
        expect(rate).toBeLessThanOrEqual(baseRate);
      });

      test('should return the decayed rate at iteration 10', () => {
        // Act
        const rate = invRateFnCustom(baseRate, 10);
        const expected = baseRate / (1 + gamma * Math.pow(10, power));
        // Assert
        expect(rate).toBeCloseTo(expected, 5);
      });

      test('should be non-negative at iteration 10', () => {
        // Act
        const rate = invRateFnCustom(baseRate, 10);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });

      test('should be less than or equal to baseRate at iteration 10', () => {
        // Act
        const rate = invRateFnCustom(baseRate, 10);
        // Assert
        expect(rate).toBeLessThanOrEqual(baseRate);
      });
    });

    describe('Default parameters', () => {
      test('should return the base rate at iteration 0', () => {
        // Act
        const rate = invRateFnDefault(baseRate, 0);
        // Assert
        expect(rate).toBeCloseTo(baseRate, 5);
      });

      test('should be non-negative at iteration 0', () => {
        // Act
        const rate = invRateFnDefault(baseRate, 0);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });

      test('should return the decayed rate at iteration 10', () => {
        // Act
        const rate = invRateFnDefault(baseRate, 10);
        const expected = baseRate / (1 + 0.001 * Math.pow(10, 2));
        // Assert
        expect(rate).toBeCloseTo(expected, 5);
      });

      test('should be non-negative at iteration 10', () => {
        // Act
        const rate = invRateFnDefault(baseRate, 10);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });

      test('should be less than or equal to baseRate at iteration 10', () => {
        // Act
        const rate = invRateFnDefault(baseRate, 10);
        // Assert
        expect(rate).toBeLessThanOrEqual(baseRate);
      });
    });
  });

  describe('cosineAnnealing()', () => {
    const period = 100;
    const minRate = 0.01;
    const cosineFnCustom = Rate.cosineAnnealing(period, minRate);
    const cosineFnDefault = Rate.cosineAnnealing();

    describe('Custom parameters', () => {
      test('should return the base rate at the start of the cycle', () => {
        // Act
        const rate = cosineFnCustom(baseRate, 0);
        // Assert
        expect(rate).toBeCloseTo(baseRate);
      });
      test('should be within [minRate, baseRate] at the start of the cycle', () => {
        // Act
        const rate = cosineFnCustom(baseRate, 0);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(minRate);
        expect(rate).toBeLessThanOrEqual(baseRate);
      });
      test('should return the average rate at the middle of the cycle', () => {
        // Act
        const rate = cosineFnCustom(baseRate, period / 2);
        // Assert
        expect(rate).toBeCloseTo(minRate + (baseRate - minRate) * 0.5);
      });
      test('should be within [minRate, baseRate] at the middle of the cycle', () => {
        // Act
        const rate = cosineFnCustom(baseRate, period / 2);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(minRate);
        expect(rate).toBeLessThanOrEqual(baseRate);
      });
      test('should return the min rate near the end of the cycle', () => {
        // Act
        const rate = cosineFnCustom(baseRate, period - 1);
        // Assert
        expect(rate).toBeCloseTo(minRate, 2);
      });
      test('should be within [minRate, baseRate] near the end of the cycle', () => {
        // Act
        const rate = cosineFnCustom(baseRate, period - 1);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(minRate);
        expect(rate).toBeLessThanOrEqual(baseRate);
      });
      test('should return the base rate at the start of the second cycle', () => {
        // Act
        const rate = cosineFnCustom(baseRate, period);
        // Assert
        expect(rate).toBeCloseTo(baseRate);
      });
      test('should be within [minRate, baseRate] at the start of the second cycle', () => {
        // Act
        const rate = cosineFnCustom(baseRate, period);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(minRate);
        expect(rate).toBeLessThanOrEqual(baseRate);
      });
    });

    describe('Default parameters', () => {
      test('should return the base rate at the start of the cycle', () => {
        // Act
        const rate = cosineFnDefault(baseRate, 0);
        // Assert
        expect(rate).toBeCloseTo(baseRate);
      });
      test('should be within [0, baseRate] at the start of the cycle', () => {
        // Act
        const rate = cosineFnDefault(baseRate, 0);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
        expect(rate).toBeLessThanOrEqual(baseRate);
      });
      test('should return half the base rate at the middle of the cycle', () => {
        // Act
        const rate = cosineFnDefault(baseRate, 500);
        // Assert
        expect(rate).toBeCloseTo(baseRate * 0.5);
      });
      test('should be within [0, baseRate] at the middle of the cycle', () => {
        // Act
        const rate = cosineFnDefault(baseRate, 500);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
        expect(rate).toBeLessThanOrEqual(baseRate);
      });
      test('should return close to zero near the end of the cycle', () => {
        // Act
        const rate = cosineFnDefault(baseRate, 999);
        // Assert
        expect(rate).toBeCloseTo(0, 2);
      });
      test('should be within [0, baseRate] near the end of the cycle', () => {
        // Act
        const rate = cosineFnDefault(baseRate, 999);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
        expect(rate).toBeLessThanOrEqual(baseRate);
      });
    });
  });
});
