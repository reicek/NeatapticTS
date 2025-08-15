import Rate from '../../src/methods/rate';

describe('Rate', () => {
  const baseRate = 0.1;

  describe('fixed()', () => {
    const fixedRateFn = Rate.fixed();

    describe('Scenario: At iteration 0', () => {
      it('should return the base rate', () => {
        // Act
        const rate = fixedRateFn(baseRate, 0);
        // Assert
        expect(rate).toBe(baseRate);
      });
      it('should be non-negative', () => {
        // Act
        const rate = fixedRateFn(baseRate, 0);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });
    });

    describe('Scenario: At iteration 100', () => {
      it('should return the base rate', () => {
        // Act
        const rate = fixedRateFn(baseRate, 100);
        // Assert
        expect(rate).toBe(baseRate);
      });
      it('should be non-negative', () => {
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
      it('should return the base rate before the first step', () => {
        // Act
        const rate = stepRateFnCustom(baseRate, 9);
        // Assert
        expect(rate).toBe(baseRate);
      });
      it('should be non-negative before the first step', () => {
        // Act
        const rate = stepRateFnCustom(baseRate, 9);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });
      it('should return the decayed rate at the first step', () => {
        // Act
        const rate = stepRateFnCustom(baseRate, 10);
        // Assert
        expect(rate).toBeCloseTo(baseRate * Math.pow(gamma, 1));
      });
      it('should be non-negative at the first step', () => {
        // Act
        const rate = stepRateFnCustom(baseRate, 10);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });
      it('should be less than or equal to baseRate at the first step', () => {
        // Act
        const rate = stepRateFnCustom(baseRate, 10);
        // Assert
        expect(rate).toBeLessThanOrEqual(baseRate);
      });
      it('should return the decayed rate after multiple steps', () => {
        // Act
        const rate = stepRateFnCustom(baseRate, 25);
        // Assert
        expect(rate).toBeCloseTo(baseRate * Math.pow(gamma, 2));
      });
      it('should be non-negative after multiple steps', () => {
        // Act
        const rate = stepRateFnCustom(baseRate, 25);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });
      it('should be less than or equal to baseRate after multiple steps', () => {
        // Act
        const rate = stepRateFnCustom(baseRate, 25);
        // Assert
        expect(rate).toBeLessThanOrEqual(baseRate);
      });
    });

    describe('Default parameters', () => {
      it('should return the base rate before the first step', () => {
        // Act
        const rate = stepRateFnDefault(baseRate, 99);
        // Assert
        expect(rate).toBe(baseRate);
      });
      it('should be non-negative before the first step', () => {
        // Act
        const rate = stepRateFnDefault(baseRate, 99);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });
      it('should return the decayed rate at the first step', () => {
        // Act
        const rate = stepRateFnDefault(baseRate, 100);
        // Assert
        expect(rate).toBeCloseTo(baseRate * Math.pow(0.9, 1));
      });
      it('should be non-negative at the first step', () => {
        // Act
        const rate = stepRateFnDefault(baseRate, 100);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });
      it('should be less than or equal to baseRate at the first step', () => {
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
      it('should return the base rate at iteration 0', () => {
        // Act
        const rate = expRateFnCustom(baseRate, 0);
        // Assert
        expect(rate).toBe(baseRate);
      });
      it('should be non-negative at iteration 0', () => {
        // Act
        const rate = expRateFnCustom(baseRate, 0);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });
      it('should return the decayed rate at iteration 1', () => {
        // Act
        const rate = expRateFnCustom(baseRate, 1);
        // Assert
        expect(rate).toBeCloseTo(baseRate * Math.pow(gamma, 1));
      });
      it('should be non-negative at iteration 1', () => {
        // Act
        const rate = expRateFnCustom(baseRate, 1);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });
      it('should be less than or equal to baseRate at iteration 1', () => {
        // Act
        const rate = expRateFnCustom(baseRate, 1);
        // Assert
        expect(rate).toBeLessThanOrEqual(baseRate);
      });
      it('should return the decayed rate at iteration 10', () => {
        // Act
        const rate = expRateFnCustom(baseRate, 10);
        // Assert
        expect(rate).toBeCloseTo(baseRate * Math.pow(gamma, 10));
      });
      it('should be non-negative at iteration 10', () => {
        // Act
        const rate = expRateFnCustom(baseRate, 10);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });
      it('should be less than or equal to baseRate at iteration 10', () => {
        // Act
        const rate = expRateFnCustom(baseRate, 10);
        // Assert
        expect(rate).toBeLessThanOrEqual(baseRate);
      });
    });

    describe('Default gamma', () => {
      it('should return the base rate at iteration 0', () => {
        // Act
        const rate = expRateFnDefault(baseRate, 0);
        // Assert
        expect(rate).toBe(baseRate);
      });
      it('should be non-negative at iteration 0', () => {
        // Act
        const rate = expRateFnDefault(baseRate, 0);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });
      it('should return the decayed rate at iteration 10', () => {
        // Act
        const rate = expRateFnDefault(baseRate, 10);
        // Assert
        expect(rate).toBeCloseTo(baseRate * Math.pow(0.999, 10));
      });
      it('should be non-negative at iteration 10', () => {
        // Act
        const rate = expRateFnDefault(baseRate, 10);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });
      it('should be less than or equal to baseRate at iteration 10', () => {
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
      it('should return the base rate at iteration 0', () => {
        // Act
        const rate = invRateFnCustom(baseRate, 0);
        // Assert
        expect(rate).toBeCloseTo(baseRate, 5);
      });

      it('should be non-negative at iteration 0', () => {
        // Act
        const rate = invRateFnCustom(baseRate, 0);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });

      it('should return the decayed rate at iteration 1', () => {
        // Act
        const rate = invRateFnCustom(baseRate, 1);
        const expected = baseRate / (1 + gamma * Math.pow(1, power));
        // Assert
        expect(rate).toBeCloseTo(expected, 5);
      });

      it('should be non-negative at iteration 1', () => {
        // Act
        const rate = invRateFnCustom(baseRate, 1);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });

      it('should be less than or equal to baseRate at iteration 1', () => {
        // Act
        const rate = invRateFnCustom(baseRate, 1);
        // Assert
        expect(rate).toBeLessThanOrEqual(baseRate);
      });

      it('should return the decayed rate at iteration 10', () => {
        // Act
        const rate = invRateFnCustom(baseRate, 10);
        const expected = baseRate / (1 + gamma * Math.pow(10, power));
        // Assert
        expect(rate).toBeCloseTo(expected, 5);
      });

      it('should be non-negative at iteration 10', () => {
        // Act
        const rate = invRateFnCustom(baseRate, 10);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });

      it('should be less than or equal to baseRate at iteration 10', () => {
        // Act
        const rate = invRateFnCustom(baseRate, 10);
        // Assert
        expect(rate).toBeLessThanOrEqual(baseRate);
      });
    });

    describe('Default parameters', () => {
      it('should return the base rate at iteration 0', () => {
        // Act
        const rate = invRateFnDefault(baseRate, 0);
        // Assert
        expect(rate).toBeCloseTo(baseRate, 5);
      });

      it('should be non-negative at iteration 0', () => {
        // Act
        const rate = invRateFnDefault(baseRate, 0);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });

      it('should return the decayed rate at iteration 10', () => {
        // Act
        const rate = invRateFnDefault(baseRate, 10);
        const expected = baseRate / (1 + 0.001 * Math.pow(10, 2));
        // Assert
        expect(rate).toBeCloseTo(expected, 5);
      });

      it('should be non-negative at iteration 10', () => {
        // Act
        const rate = invRateFnDefault(baseRate, 10);
        // Assert
        expect(rate).toBeGreaterThanOrEqual(0);
      });

      it('should be less than or equal to baseRate at iteration 10', () => {
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
      describe('At the start of the cycle', () => {
        it('should return the base rate', () => {
          // Arrange
          // Act
          const rate = cosineFnCustom(baseRate, 0);
          // Assert
          expect(rate).toBeCloseTo(baseRate);
        });
        it('should be greater than or equal to minRate', () => {
          // Arrange
          // Act
          const rate = cosineFnCustom(baseRate, 0);
          // Assert
          expect(rate).toBeGreaterThanOrEqual(minRate);
        });
        it('should be less than or equal to baseRate', () => {
          // Arrange
          // Act
          const rate = cosineFnCustom(baseRate, 0);
          // Assert
          expect(rate).toBeLessThanOrEqual(baseRate);
        });
      });
      describe('At the middle of the cycle', () => {
        it('should return the average rate', () => {
          // Arrange
          // Act
          const rate = cosineFnCustom(baseRate, period / 2);
          // Assert
          expect(rate).toBeCloseTo(minRate + (baseRate - minRate) * 0.5);
        });
        it('should be greater than or equal to minRate', () => {
          // Arrange
          // Act
          const rate = cosineFnCustom(baseRate, period / 2);
          // Assert
          expect(rate).toBeGreaterThanOrEqual(minRate);
        });
        it('should be less than or equal to baseRate', () => {
          // Arrange
          // Act
          const rate = cosineFnCustom(baseRate, period / 2);
          // Assert
          expect(rate).toBeLessThanOrEqual(baseRate);
        });
      });
      describe('Near the end of the cycle', () => {
        it('should return the min rate', () => {
          // Arrange
          // Act
          const rate = cosineFnCustom(baseRate, period - 1);
          // Assert
          expect(rate).toBeCloseTo(minRate, 2);
        });
        it('should be greater than or equal to minRate', () => {
          // Arrange
          // Act
          const rate = cosineFnCustom(baseRate, period - 1);
          // Assert
          expect(rate).toBeGreaterThanOrEqual(minRate);
        });
        it('should be less than or equal to baseRate', () => {
          // Arrange
          // Act
          const rate = cosineFnCustom(baseRate, period - 1);
          // Assert
          expect(rate).toBeLessThanOrEqual(baseRate);
        });
      });
      describe('At the start of the second cycle', () => {
        it('should return the base rate', () => {
          // Arrange
          // Act
          const rate = cosineFnCustom(baseRate, period);
          // Assert
          expect(rate).toBeCloseTo(baseRate);
        });
        it('should be greater than or equal to minRate', () => {
          // Arrange
          // Act
          const rate = cosineFnCustom(baseRate, period);
          // Assert
          expect(rate).toBeGreaterThanOrEqual(minRate);
        });
        it('should be less than or equal to baseRate', () => {
          // Arrange
          // Act
          const rate = cosineFnCustom(baseRate, period);
          // Assert
          expect(rate).toBeLessThanOrEqual(baseRate);
        });
      });
    });
    describe('Default parameters', () => {
      describe('At the start of the cycle', () => {
        it('should return the base rate', () => {
          // Arrange
          // Act
          const rate = cosineFnDefault(baseRate, 0);
          // Assert
          expect(rate).toBeCloseTo(baseRate);
        });
        it('should be greater than or equal to 0', () => {
          // Arrange
          // Act
          const rate = cosineFnDefault(baseRate, 0);
          // Assert
          expect(rate).toBeGreaterThanOrEqual(0);
        });
        it('should be less than or equal to baseRate', () => {
          // Arrange
          // Act
          const rate = cosineFnDefault(baseRate, 0);
          // Assert
          expect(rate).toBeLessThanOrEqual(baseRate);
        });
      });
      describe('At the middle of the cycle', () => {
        it('should return half the base rate', () => {
          // Arrange
          // Act
          const rate = cosineFnDefault(baseRate, 500);
          // Assert
          expect(rate).toBeCloseTo(baseRate * 0.5);
        });
        it('should be greater than or equal to 0', () => {
          // Arrange
          // Act
          const rate = cosineFnDefault(baseRate, 500);
          // Assert
          expect(rate).toBeGreaterThanOrEqual(0);
        });
        it('should be less than or equal to baseRate', () => {
          // Arrange
          // Act
          const rate = cosineFnDefault(baseRate, 500);
          // Assert
          expect(rate).toBeLessThanOrEqual(baseRate);
        });
      });
      describe('Near the end of the cycle', () => {
        it('should return close to zero', () => {
          // Arrange
          // Act
          const rate = cosineFnDefault(baseRate, 999);
          // Assert
          expect(rate).toBeCloseTo(0, 2);
        });
        it('should be greater than or equal to 0', () => {
          // Arrange
          // Act
          const rate = cosineFnDefault(baseRate, 999);
          // Assert
          expect(rate).toBeGreaterThanOrEqual(0);
        });
        it('should be less than or equal to baseRate', () => {
          // Arrange
          // Act
          const rate = cosineFnDefault(baseRate, 999);
          // Assert
          expect(rate).toBeLessThanOrEqual(baseRate);
        });
      });
    });
  });

  describe('Rate utility methods', () => {
    describe('fixed()', () => {
      it('should always return the base rate regardless of iteration', () => {
        // Arrange
        const fn = Rate.fixed();
        // Act
        const result1 = fn(0.5, 0);
        const result2 = fn(0.5, 100);
        // Assert
        expect(result1).toBe(0.5);
        expect(result2).toBe(0.5);
      });
    });
    describe('step()', () => {
      it('should decay at correct steps', () => {
        // Arrange
        const fn = Rate.step(0.5, 10);
        // Act
        const r0 = fn(1, 0);
        const r10 = fn(1, 10);
        const r20 = fn(1, 20);
        // Assert
        expect(r0).toBe(1);
        expect(r10).toBeCloseTo(0.5);
        expect(r20).toBeCloseTo(0.25);
      });
    });
    describe('exp()', () => {
      it('should decay exponentially', () => {
        // Arrange
        const fn = Rate.exp(0.5);
        // Act
        const r0 = fn(1, 0);
        const r1 = fn(1, 1);
        const r2 = fn(1, 2);
        // Assert
        expect(r0).toBe(1);
        expect(r1).toBeCloseTo(0.5);
        expect(r2).toBeCloseTo(0.25);
      });
    });
    describe('inv()', () => {
      it('should decay inversely', () => {
        // Arrange
        const fn = Rate.inv(1, 1);
        // Act
        const r0 = fn(1, 0);
        const r1 = fn(1, 1);
        const r2 = fn(1, 2);
        // Assert
        expect(r0).toBe(1);
        expect(r1).toBeCloseTo(0.5);
        expect(r2).toBeCloseTo(0.333333, 5);
      });
    });
    describe('cosineAnnealing()', () => {
      it('should cycle between baseRate and minRate', () => {
        // Arrange
        const fn = Rate.cosineAnnealing(4, 0);
        // Act
        const r0 = fn(1, 0);
        const r2 = fn(1, 2);
        const r4 = fn(1, 4);
        // Assert
        expect(r0).toBeCloseTo(1);
        expect(r2).toBeCloseTo(0.5);
        expect(r4).toBeCloseTo(1);
      });
    });
  });

  describe('Invalid and edge input scenarios', () => {
    const period = 100;
    const minRate = 0.01;
    const cosineFnCustom = Rate.cosineAnnealing(period, minRate);
    const stepFn = Rate.step(0.9, 10);
    const expFn = Rate.exp(0.95);
    const invFn = Rate.inv(0.01, 1.5);
    const fixedFn = Rate.fixed();
    it('should return a valid rate for negative iteration', () => {
      // Arrange
      // Act
      const rate = cosineFnCustom(baseRate, -1);
      // Assert
      expect(typeof rate).toBe('number');
    });
    it('should return a valid rate for negative baseRate', () => {
      // Arrange
      // Act
      const rate = cosineFnCustom(-0.1, 0);
      // Assert
      expect(typeof rate).toBe('number');
    });
    it('should return a valid rate for non-integer iteration', () => {
      // Arrange
      // Act
      const rate = cosineFnCustom(baseRate, 1.5);
      // Assert
      expect(typeof rate).toBe('number');
    });
    it('step: should return a valid rate for negative iteration', () => {
      // Arrange
      // Act
      const rate = stepFn(baseRate, -5);
      // Assert
      expect(typeof rate).toBe('number');
    });
    it('exp: should return a valid rate for negative iteration', () => {
      // Arrange
      // Act
      const rate = expFn(baseRate, -5);
      // Assert
      expect(typeof rate).toBe('number');
    });
    it('inv: should return a valid rate for negative iteration', () => {
      // Arrange
      // Act
      const rate = invFn(baseRate, -5);
      // Assert
      expect(typeof rate).toBe('number');
    });
    it('step: should return a valid rate for negative baseRate', () => {
      // Arrange
      // Act
      const rate = stepFn(-0.1, 0);
      // Assert
      expect(typeof rate).toBe('number');
    });
    it('exp: should return a valid rate for negative baseRate', () => {
      // Arrange
      // Act
      const rate = expFn(-0.1, 0);
      // Assert
      expect(typeof rate).toBe('number');
    });
    it('inv: should return a valid rate for negative baseRate', () => {
      // Arrange
      // Act
      const rate = invFn(-0.1, 0);
      // Assert
      expect(typeof rate).toBe('number');
    });
    it('step: should return a valid rate for non-integer iteration', () => {
      // Arrange
      // Act
      const rate = stepFn(baseRate, 1.5);
      // Assert
      expect(typeof rate).toBe('number');
    });
    it('exp: should return a valid rate for non-integer iteration', () => {
      // Arrange
      // Act
      const rate = expFn(baseRate, 1.5);
      // Assert
      expect(typeof rate).toBe('number');
    });
    it('inv: should return a valid rate for non-integer iteration', () => {
      // Arrange
      // Act
      const rate = invFn(baseRate, 1.5);
      // Assert
      expect(typeof rate).toBe('number');
    });
    it('fixed: should return the base rate for negative iteration', () => {
      // Arrange
      // Act
      const rate = fixedFn(baseRate, -10);
      // Assert
      expect(rate).toBe(baseRate);
    });
    it('fixed: should return the base rate for negative baseRate', () => {
      // Arrange
      // Act
      const rate = fixedFn(-0.1, 0);
      // Assert
      expect(rate).toBe(-0.1);
    });
    it('fixed: should return the base rate for non-integer iteration', () => {
      // Arrange
      // Act
      const rate = fixedFn(baseRate, 1.5);
      // Assert
      expect(rate).toBe(baseRate);
    });
  });
});
