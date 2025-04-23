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
      test(`should calculate logistic for x=${x}`, () => {
        const expected = 1 / (1 + Math.exp(-x));
        expect(Activation.logistic(x)).toBeCloseTo(expected, epsilon);
      });
      test(`should calculate logistic derivative for x=${x}`, () => {
        const fx = 1 / (1 + Math.exp(-x));
        const expected = fx * (1 - fx);
        expect(Activation.logistic(x, true)).toBeCloseTo(expected, epsilon);
      });
    });
  });

  describe('tanh()', () => {
    testValues.forEach((x) => {
      test(`should calculate tanh for x=${x}`, () => {
        const expected = Math.tanh(x);
        expect(Activation.tanh(x)).toBeCloseTo(expected, epsilon);
      });
      test(`should calculate tanh derivative for x=${x}`, () => {
        const expected = 1 - Math.pow(Math.tanh(x), 2);
        expect(Activation.tanh(x, true)).toBeCloseTo(expected, epsilon);
      });
    });
  });

  describe('identity()', () => {
    testValues.forEach((x) => {
      test(`should calculate identity for x=${x}`, () => {
        const expected = x;
        expect(Activation.identity(x)).toBe(expected);
      });
      test(`should calculate identity derivative for x=${x}`, () => {
        const expected = 1;
        expect(Activation.identity(x, true)).toBe(expected);
      });
    });
  });

  describe('step()', () => {
    testValues.forEach((x) => {
      test(`should calculate step for x=${x}`, () => {
        const expected = x > 0 ? 1 : 0;
        expect(Activation.step(x)).toBe(expected);
      });
      test(`should calculate step derivative for x=${x}`, () => {
        const expected = 0;
        expect(Activation.step(x, true)).toBe(expected);
      });
    });
  });

  describe('relu()', () => {
    testValues.forEach((x) => {
      test(`should calculate relu for x=${x}`, () => {
        const expected = x > 0 ? x : 0;
        expect(Activation.relu(x)).toBeCloseTo(expected, epsilon);
      });
      test(`should calculate relu derivative for x=${x}`, () => {
        const expected = x > 0 ? 1 : 0;
        expect(Activation.relu(x, true)).toBe(expected);
      });
    });
  });

  describe('softsign()', () => {
    testValues.forEach((x) => {
      test(`should calculate softsign for x=${x}`, () => {
        const expected = x / (1 + Math.abs(x));
        expect(Activation.softsign(x)).toBeCloseTo(expected, epsilon);
      });
      test(`should calculate softsign derivative for x=${x}`, () => {
        const expected = 1 / Math.pow(1 + Math.abs(x), 2);
        expect(Activation.softsign(x, true)).toBeCloseTo(expected, epsilon);
      });
    });
  });

  describe('sinusoid()', () => {
    testValues.forEach((x) => {
      test(`should calculate sinusoid for x=${x}`, () => {
        const expected = Math.sin(x);
        expect(Activation.sinusoid(x)).toBeCloseTo(expected, epsilon);
      });
      test(`should calculate sinusoid derivative for x=${x}`, () => {
        const expected = Math.cos(x);
        expect(Activation.sinusoid(x, true)).toBeCloseTo(expected, epsilon);
      });
    });
  });

  describe('gaussian()', () => {
    testValues.forEach((x) => {
      test(`should calculate gaussian for x=${x}`, () => {
        const expected = Math.exp(-Math.pow(x, 2));
        expect(Activation.gaussian(x)).toBeCloseTo(expected, epsilon);
      });
      test(`should calculate gaussian derivative for x=${x}`, () => {
        const expected = -2 * x * Math.exp(-Math.pow(x, 2));
        expect(Activation.gaussian(x, true)).toBeCloseTo(expected, epsilon);
      });
    });
  });

  describe('bentIdentity()', () => {
    testValues.forEach((x) => {
      test(`should calculate bentIdentity for x=${x}`, () => {
        const expected = (Math.sqrt(Math.pow(x, 2) + 1) - 1) / 2 + x;
        expect(Activation.bentIdentity(x)).toBeCloseTo(expected, epsilon);
      });
      test(`should calculate bentIdentity derivative for x=${x}`, () => {
        const expected = x / (2 * Math.sqrt(Math.pow(x, 2) + 1)) + 1;
        expect(Activation.bentIdentity(x, true)).toBeCloseTo(expected, epsilon);
      });
    });
  });

  describe('bipolar()', () => {
    testValues.forEach((x) => {
      test(`should calculate bipolar for x=${x}`, () => {
        const expected = x > 0 ? 1 : -1;
        expect(Activation.bipolar(x)).toBe(expected);
      });
      test(`should calculate bipolar derivative for x=${x}`, () => {
        const expected = 0;
        expect(Activation.bipolar(x, true)).toBe(expected);
      });
    });
  });

  describe('bipolarSigmoid()', () => {
    testValues.forEach((x) => {
      test(`should calculate bipolarSigmoid for x=${x}`, () => {
        // Equivalent to tanh
        const expected = Math.tanh(x);
        expect(Activation.bipolarSigmoid(x)).toBeCloseTo(expected, epsilon);
      });
      test(`should calculate bipolarSigmoid derivative for x=${x}`, () => {
        const d = 2 / (1 + Math.exp(-x)) - 1;
        const expected = (1 / 2) * (1 + d) * (1 - d);
        expect(Activation.bipolarSigmoid(x, true)).toBeCloseTo(
          expected,
          epsilon
        );
      });
    });
  });

  describe('hardTanh()', () => {
    testValues.forEach((x) => {
      test(`should calculate hardTanh for x=${x}`, () => {
        const expected = Math.max(-1, Math.min(1, x));
        expect(Activation.hardTanh(x)).toBeCloseTo(expected, epsilon);
      });
      test(`should calculate hardTanh derivative for x=${x}`, () => {
        const expected = x > -1 && x < 1 ? 1 : 0;
        expect(Activation.hardTanh(x, true)).toBe(expected);
      });
    });
  });

  describe('absolute()', () => {
    testValues.forEach((x) => {
      test(`should calculate absolute for x=${x}`, () => {
        const expected = Math.abs(x);
        expect(Activation.absolute(x)).toBeCloseTo(expected, epsilon);
      });
      test(`should calculate absolute derivative for x=${x}`, () => {
        const expected = x < 0 ? -1 : 1;
        expect(Activation.absolute(x, true)).toBe(expected);
      });
    });
  });

  describe('inverse()', () => {
    testValues.forEach((x) => {
      test(`should calculate inverse for x=${x}`, () => {
        const expected = 1 - x;
        expect(Activation.inverse(x)).toBe(expected);
      });
      test(`should calculate inverse derivative for x=${x}`, () => {
        const expected = -1;
        expect(Activation.inverse(x, true)).toBe(expected);
      });
    });
  });

  describe('selu()', () => {
    const alpha = 1.6732632423543772848170429916717;
    const scale = 1.0507009873554804934193349852946;

    testValues.forEach((x) => {
      test(`should calculate selu for x=${x}`, () => {
        const fx = x > 0 ? x : alpha * Math.exp(x) - alpha;
        const expected = fx * scale;
        expect(Activation.selu(x)).toBeCloseTo(expected, epsilon);
      });
      test(`should calculate selu derivative for x=${x}`, () => {
        const fx = x > 0 ? x : alpha * Math.exp(x) - alpha;
        const expected = (x > 0 ? scale : (fx + alpha) * scale);
        expect(Activation.selu(x, true)).toBeCloseTo(expected, epsilon);
      });
    });
  });

  describe('softplus()', () => {
    stabilityTestValues.forEach((x) => {
      test(`should calculate softplus for x=${x}`, () => {
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
      });

      test(`should calculate softplus derivative (logistic) for x=${x}`, () => {
        const expected = 1 / (1 + Math.exp(-x)); // Logistic
        expect(Activation.softplus(x, true)).toBeCloseTo(expected, epsilon);
      });
    });
  });

  describe('swish()', () => {
    testValues.forEach((x) => {
      test(`should calculate swish for x=${x}`, () => {
        const sigmoid_x = 1 / (1 + Math.exp(-x));
        const expected = x * sigmoid_x;
        expect(Activation.swish(x)).toBeCloseTo(expected, epsilon);
      });
      test(`should calculate swish derivative for x=${x}`, () => {
        const sigmoid_x = 1 / (1 + Math.exp(-x));
        const swish_x = x * sigmoid_x;
        const expected = swish_x + sigmoid_x * (1 - swish_x);
        expect(Activation.swish(x, true)).toBeCloseTo(expected, epsilon);
      });
    });
  });

  describe('gelu()', () => {
    testValues.forEach((x) => {
      test(`should calculate gelu approximation for x=${x}`, () => {
        const cdf =
          0.5 *
          (1.0 +
            Math.tanh(
              Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))
            ));
        const expected = x * cdf;
        expect(Activation.gelu(x)).toBeCloseTo(expected, epsilon);
      });
      test(`should calculate gelu approximation derivative for x=${x}`, () => {
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
        expect(Activation.gelu(x, true)).toBeCloseTo(expected, epsilon);
      });
    });
  });

  describe('mish()', () => {
    stabilityTestValues.forEach((x) => {
      test(`should calculate mish for x=${x}`, () => {
        // Replicate internal stable softplus calculation for expected value
        let sp_x: number;
        if (x > 30) {
          sp_x = x;
        } else if (x < -30) {
          sp_x = Math.exp(x);
        } else {
          sp_x = Math.max(0, x) + Math.log(1 + Math.exp(-Math.abs(x)));
        }
        const expected = x * Math.tanh(sp_x);
        expect(Activation.mish(x)).toBeCloseTo(expected, epsilon);
      });

      test(`should calculate mish derivative for x=${x}`, () => {
        // Replicate internal stable softplus calculation for derivative parts
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
        expect(Activation.mish(x, true)).toBeCloseTo(expected, epsilon);
      });
    });
  });
});
