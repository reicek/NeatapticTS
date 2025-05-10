import Activation from '../../src/methods/activation';

describe('Activation', () => {
  const epsilon = 1e-9; // Tolerance for floating point comparisons
  const testValues = [-10, -1, -0.5, 0, 0.5, 1, 10]; // Common values to test
  const stabilityTestValues = [-100, -30.1, -30, -10, 0, 10, 30, 30.1, 100]; // For functions needing stability checks

  describe('logistic()', () => {
    test('should calculate logistic function correctly', () => {
      // Arrange
      const x = 1;
      const expected = 1 / (1 + Math.exp(-1));
      // Act
      const result = Activation.logistic(x);
      // Assert
      expect(result).toBeCloseTo(expected, epsilon);
    });

    test('should calculate logistic derivative correctly', () => {
      // Arrange
      const x = 1;
      const fx = 1 / (1 + Math.exp(-1));
      const expected = fx * (1 - fx);
      // Act
      const result = Activation.logistic(x, true);
      // Assert
      expect(result).toBeCloseTo(expected, epsilon);
    });

    testValues.forEach((x) => {
      test(`should calculate logistic for x=${x} within [0, 1]`, () => {
        const expected = 1 / (1 + Math.exp(-x));
        const result = Activation.logistic(x);
        expect(result).toBeCloseTo(expected, epsilon);
        expect(result).toBeGreaterThanOrEqual(0);
        expect(result).toBeLessThanOrEqual(1);
      });
      test(`should calculate logistic derivative for x=${x} within [0, 0.25]`, () => {
        const fx = 1 / (1 + Math.exp(-x));
        const expected = fx * (1 - fx);
        const result = Activation.logistic(x, true);
        expect(result).toBeCloseTo(expected, epsilon);
        expect(result).toBeGreaterThanOrEqual(0);
        expect(result).toBeLessThanOrEqual(0.25);
      });
    });
  });

  describe('tanh()', () => {
    testValues.forEach((x) => {
      test(`should calculate tanh for x=${x} within [-1, 1]`, () => {
        const expected = Math.tanh(x);
        const result = Activation.tanh(x);
        expect(result).toBeCloseTo(expected, epsilon);
        expect(result).toBeGreaterThanOrEqual(-1);
        expect(result).toBeLessThanOrEqual(1);
      });
      test(`should calculate tanh derivative for x=${x} within [0, 1]`, () => {
        const expected = 1 - Math.pow(Math.tanh(x), 2);
        const result = Activation.tanh(x, true);
        expect(result).toBeCloseTo(expected, epsilon);
        expect(result).toBeGreaterThanOrEqual(0);
        expect(result).toBeLessThanOrEqual(1);
      });
    });
  });

  describe('identity()', () => {
    testValues.forEach((x) => {
      test(`should calculate identity for x=${x}`, () => {
        const expected = x;
        const result = Activation.identity(x);
        expect(result).toBe(expected);
        expect(isFinite(result)).toBe(true);
      });
      test(`should calculate identity derivative for x=${x} equal to 1`, () => {
        const expected = 1;
        const result = Activation.identity(x, true);
        expect(result).toBe(expected);
      });
    });
  });

  describe('step()', () => {
    testValues.forEach((x) => {
      test(`should calculate step for x=${x} as 0 or 1`, () => {
        const expected = x > 0 ? 1 : 0;
        const result = Activation.step(x);
        expect(result).toBe(expected);
        expect([0, 1]).toContain(result);
      });
      test(`should calculate step derivative for x=${x} equal to 0`, () => {
        const expected = 0;
        const result = Activation.step(x, true);
        expect(result).toBe(expected);
      });
    });
  });

  describe('relu()', () => {
    testValues.forEach((x) => {
      test(`should calculate relu for x=${x} within [0, inf)`, () => {
        const expected = x > 0 ? x : 0;
        const result = Activation.relu(x);
        expect(result).toBeCloseTo(expected, epsilon);
        expect(result).toBeGreaterThanOrEqual(0);
      });
      test(`should calculate relu derivative for x=${x} as 0 or 1`, () => {
        const expected = x > 0 ? 1 : 0;
        const result = Activation.relu(x, true);
        expect(result).toBe(expected);
        expect([0, 1]).toContain(result);
      });
    });
  });

  describe('softsign()', () => {
    testValues.forEach((x) => {
      test(`should calculate softsign for x=${x} within (-1, 1)`, () => {
        const expected = x / (1 + Math.abs(x));
        const result = Activation.softsign(x);
        expect(result).toBeCloseTo(expected, epsilon);
        expect(result).toBeGreaterThan(-1);
        expect(result).toBeLessThan(1);
      });
      test(`should calculate softsign derivative for x=${x} within (0, 1]`, () => {
        const expected = 1 / Math.pow(1 + Math.abs(x), 2);
        const result = Activation.softsign(x, true);
        expect(result).toBeCloseTo(expected, epsilon);
        expect(result).toBeGreaterThan(0);
        expect(result).toBeLessThanOrEqual(1);
      });
    });
  });

  describe('sinusoid()', () => {
    testValues.forEach((x) => {
      test(`should calculate sinusoid for x=${x} within [-1, 1]`, () => {
        const expected = Math.sin(x);
        const result = Activation.sinusoid(x);
        expect(result).toBeCloseTo(expected, epsilon);
        expect(result).toBeGreaterThanOrEqual(-1);
        expect(result).toBeLessThanOrEqual(1);
      });
      test(`should calculate sinusoid derivative for x=${x} within [-1, 1]`, () => {
        const expected = Math.cos(x);
        const result = Activation.sinusoid(x, true);
        expect(result).toBeCloseTo(expected, epsilon);
        expect(result).toBeGreaterThanOrEqual(-1);
        expect(result).toBeLessThanOrEqual(1);
      });
    });
  });

  describe('gaussian()', () => {
    testValues.forEach((x) => {
      test(`should calculate gaussian for x=${x} within (0, 1]`, () => {
        const expected = Math.exp(-Math.pow(x, 2));
        const result = Activation.gaussian(x);
        expect(result).toBeCloseTo(expected, epsilon);
        expect(result).toBeGreaterThan(0);
        expect(result).toBeLessThanOrEqual(1);
      });
      test(`should calculate gaussian derivative for x=${x} within approx [-0.86, 0.86]`, () => {
        const expected = -2 * x * Math.exp(-Math.pow(x, 2));
        const result = Activation.gaussian(x, true);
        const maxDeriv = Math.sqrt(2 / Math.E); // Max derivative magnitude
        expect(result).toBeCloseTo(expected, epsilon);
        expect(result).toBeGreaterThanOrEqual(-maxDeriv);
        expect(result).toBeLessThanOrEqual(maxDeriv);
      });
    });
  });

  describe('bentIdentity()', () => {
    testValues.forEach((x) => {
      test(`should calculate bentIdentity for x=${x}`, () => {
        const expected = (Math.sqrt(Math.pow(x, 2) + 1) - 1) / 2 + x;
        const result = Activation.bentIdentity(x);
        expect(result).toBeCloseTo(expected, epsilon);
        expect(isFinite(result)).toBe(true);
      });
      test(`should calculate bentIdentity derivative for x=${x} within [0.5, 1.5)`, () => {
        const expected = x / (2 * Math.sqrt(Math.pow(x, 2) + 1)) + 1;
        const result = Activation.bentIdentity(x, true);
        expect(result).toBeCloseTo(expected, epsilon);
        expect(result).toBeGreaterThanOrEqual(0.5);
        expect(result).toBeLessThan(1.5);
      });
    });
  });

  describe('bipolar()', () => {
    testValues.forEach((x) => {
      test(`should calculate bipolar for x=${x} as -1 or 1`, () => {
        const expected = x > 0 ? 1 : -1;
        const result = Activation.bipolar(x);
        expect(result).toBe(expected);
        expect([-1, 1]).toContain(result);
      });
      test(`should calculate bipolar derivative for x=${x} equal to 0`, () => {
        const expected = 0;
        const result = Activation.bipolar(x, true);
        expect(result).toBe(expected);
      });
    });
  });

  describe('bipolarSigmoid()', () => {
    testValues.forEach((x) => {
      test(`should calculate bipolarSigmoid for x=${x} within (-1, 1)`, () => {
        const expected = Math.tanh(x); // Equivalent to tanh
        const result = Activation.bipolarSigmoid(x);
        expect(result).toBeCloseTo(expected, epsilon);
        expect(result).toBeGreaterThan(-1);
        expect(result).toBeLessThan(1);
      });
      test(`should calculate bipolarSigmoid derivative for x=${x} within [0, 0.5]`, () => {
        const d = 2 / (1 + Math.exp(-x)) - 1;
        const expected = (1 / 2) * (1 + d) * (1 - d);
        const result = Activation.bipolarSigmoid(x, true);
        expect(result).toBeCloseTo(expected, epsilon);
        expect(result).toBeGreaterThanOrEqual(0);
        expect(result).toBeLessThanOrEqual(0.5);
      });
    });
  });

  describe('hardTanh()', () => {
    testValues.forEach((x) => {
      test(`should calculate hardTanh for x=${x} within [-1, 1]`, () => {
        const expected = Math.max(-1, Math.min(1, x));
        const result = Activation.hardTanh(x);
        expect(result).toBeCloseTo(expected, epsilon);
        expect(result).toBeGreaterThanOrEqual(-1);
        expect(result).toBeLessThanOrEqual(1);
      });
      test(`should calculate hardTanh derivative for x=${x} as 0 or 1`, () => {
        const expected = x > -1 && x < 1 ? 1 : 0;
        const result = Activation.hardTanh(x, true);
        expect(result).toBe(expected);
        expect([0, 1]).toContain(result);
      });
    });
  });

  describe('absolute()', () => {
    testValues.forEach((x) => {
      test(`should calculate absolute for x=${x} within [0, inf)`, () => {
        const expected = Math.abs(x);
        const result = Activation.absolute(x);
        expect(result).toBeCloseTo(expected, epsilon);
        expect(result).toBeGreaterThanOrEqual(0);
      });
      test(`should calculate absolute derivative for x=${x} as -1 or 1`, () => {
        const expected = x < 0 ? -1 : 1;
        const result = Activation.absolute(x, true);
        expect(result).toBe(expected);
        expect([-1, 1]).toContain(result);
      });
    });
  });

  describe('inverse()', () => {
    testValues.forEach((x) => {
      test(`should calculate inverse for x=${x}`, () => {
        const expected = 1 - x;
        const result = Activation.inverse(x);
        expect(result).toBe(expected);
        expect(isFinite(result)).toBe(true);
      });
      test(`should calculate inverse derivative for x=${x} equal to -1`, () => {
        const expected = -1;
        const result = Activation.inverse(x, true);
        expect(result).toBe(expected);
      });
    });
  });

  describe('selu()', () => {
    const alpha = 1.6732632423543772848170429916717;
    const scale = 1.0507009873554804934193349852946;
    const lowerBound = -alpha * scale;

    testValues.forEach((x) => {
      test(`should calculate selu for x=${x} within [${lowerBound.toFixed(2)}, inf)`, () => {
        const fx = x > 0 ? x : alpha * Math.exp(x) - alpha;
        const expected = fx * scale;
        const result = Activation.selu(x);
        expect(result).toBeCloseTo(expected, epsilon);
        expect(result).toBeGreaterThanOrEqual(lowerBound - epsilon);
      });
      test(`should calculate selu derivative for x=${x} within (0, ${scale * alpha}]`, () => {
        const fx = x > 0 ? x : alpha * Math.exp(x) - alpha;
        const expected = (x > 0 ? scale : (fx + alpha) * scale);
        const result = Activation.selu(x, true);
        expect(result).toBeCloseTo(expected, epsilon);
        expect(result).toBeGreaterThan(0);
        expect(result).toBeLessThanOrEqual(scale * alpha + epsilon);
      });
    });
  });

  describe('softplus()', () => {
    stabilityTestValues.forEach((x) => {
      test(`should calculate softplus for x=${x} within (0, inf)`, () => {
        // Reference calculation (potentially unstable)
        const expected = Math.log(1 + Math.exp(x));
        // Handle potential Infinity in reference for comparison
        const result = Activation.softplus(x);
        if (expected === Infinity) {
          expect(result).toBeCloseTo(x, epsilon); // Should approximate x for large positive x
        } else if (expected === 0 && x < -50) {
          expect(result).toBeCloseTo(0, epsilon); // Should approximate 0 for large negative x
        } else {
          expect(result).toBeCloseTo(expected, epsilon);
        }
        expect(result).toBeGreaterThan(0);
      });

      test(`should calculate softplus derivative (logistic) for x=${x} within (0, 1]`, () => {
        const expected = 1 / (1 + Math.exp(-x)); // Logistic
        const result = Activation.softplus(x, true);
        expect(result).toBeCloseTo(expected, epsilon);
        expect(result).toBeGreaterThan(0);
        expect(result).toBeLessThanOrEqual(1);
      });
    });
  });

  describe('swish()', () => {
    const lowerBound = -0.2784645; // Approximate minimum value
    testValues.forEach((x) => {
      test(`should calculate swish for x=${x} within [${lowerBound.toFixed(3)}, inf)`, () => {
        const sigmoid_x = 1 / (1 + Math.exp(-x));
        const expected = x * sigmoid_x;
        const result = Activation.swish(x);
        expect(result).toBeCloseTo(expected, epsilon);
        expect(result).toBeGreaterThanOrEqual(lowerBound - epsilon);
      });
      test(`should calculate swish derivative for x=${x} within approx [-0.09, inf)`, () => {
        const sigmoid_x = 1 / (1 + Math.exp(-x));
        const swish_x = x * sigmoid_x;
        const expected = swish_x + sigmoid_x * (1 - swish_x);
        const result = Activation.swish(x, true);
        expect(result).toBeCloseTo(expected, epsilon);
        expect(result).toBeGreaterThanOrEqual(-0.09 - epsilon);
      });
    });
  });

  describe('gelu()', () => {
    const lowerBound = -0.17; // Approximate minimum value
    testValues.forEach((x) => {
      test(`should calculate gelu approximation for x=${x} within [${lowerBound.toFixed(2)}, inf)`, () => {
        const cdf =
          0.5 *
          (1.0 +
            Math.tanh(
              Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))
            ));
        const expected = x * cdf;
        const result = Activation.gelu(x);
        expect(result).toBeCloseTo(expected, epsilon);
        expect(result).toBeGreaterThanOrEqual(lowerBound - epsilon);
      });
      test(`should calculate gelu approximation derivative for x=${x} within approx [-0.16, inf)`, () => {
        const cdf =
          0.5 *
          (1.0 +
            Math.tanh(
              Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))
            ));
        const intermediate =
          Math.sqrt(2.0 / Math.PI) * (1.0 + 0.134145 * x * x);
        const sech_arg =
          Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * Math.pow(x, 3));
        const sech_val = 1.0 / Math.cosh(sech_arg);
        const sech_sq = sech_val * sech_val;
        const expected = cdf + x * 0.5 * intermediate * sech_sq;
        const result = Activation.gelu(x, true);
        expect(result).toBeCloseTo(expected, epsilon);
        expect(result).toBeGreaterThanOrEqual(-0.16 - epsilon);
      });
    });
  });

  describe('mish()', () => {
    const lowerBound = -0.30884; // Approximate minimum value
    stabilityTestValues.forEach((x) => {
      test(`should calculate mish for x=${x} within [${lowerBound.toFixed(3)}, inf)`, () => {
        let sp_x: number;
        if (x > 30) {
          sp_x = x;
        } else if (x < -30) {
          sp_x = Math.exp(x);
        } else {
          sp_x = Math.max(0, x) + Math.log(1 + Math.exp(-Math.abs(x)));
        }
        const expected = x * Math.tanh(sp_x);
        const result = Activation.mish(x);
        expect(result).toBeCloseTo(expected, epsilon);
        expect(result).toBeGreaterThanOrEqual(lowerBound - epsilon);
      });

      test(`should calculate mish derivative for x=${x}`, () => {
        let sp_x: number;
        if (x > 30) {
          sp_x = x;
        } else if (x < -30) {
          sp_x = Math.exp(x);
        } else {
          sp_x = Math.max(0, x) + Math.log(1 + Math.exp(-Math.abs(x)));
        }
        const tanh_sp_x = Math.tanh(sp_x);
        const sigmoid_x = 1 / (1 + Math.exp(-x));
        const sech_sp_x = 1.0 / Math.cosh(sp_x);
        const sech_sq_sp_x = sech_sp_x * sech_sp_x;
        const expected = tanh_sp_x + x * sech_sq_sp_x * sigmoid_x;
        const result = Activation.mish(x, true);
        expect(result).toBeCloseTo(expected, epsilon);
        expect(isFinite(result)).toBe(true);
      });
    });
  });
});
