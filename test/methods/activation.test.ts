import Activation from '../../src/methods/activation';

describe('Activation', () => {
  const epsilon = 1e-9; // Tolerance for floating point comparisons
  const testValues = [-10, -1, -0.5, 0, 0.5, 1, 10]; // Common values to test
  const stabilityTestValues = [-100, -30.1, -30, -10, 0, 10, 30, 30.1, 100]; // For functions needing stability checks

  describe('logistic()', () => {
    test('calculates logistic function correctly', () => {
      // Arrange
      const x = 1;
      const expected = 1 / (1 + Math.exp(-1));
      // Act
      const result = Activation.logistic(x);
      // Assert
      expect(result).toBeCloseTo(expected, epsilon);
    });
    test('calculates logistic derivative correctly', () => {
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
      describe(`Scenario: x=${x}`, () => {
        test('logistic is within [0, 1]', () => {
          // Arrange
          const expected = 1 / (1 + Math.exp(-x));
          // Act
          const result = Activation.logistic(x);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        test('logistic is less than or equal to 1', () => {
          // Arrange
          const expected = 1 / (1 + Math.exp(-x));
          // Act
          const result = Activation.logistic(x);
          // Assert
          expect(result).toBeLessThanOrEqual(1);
        });
        test('logistic matches expected value', () => {
          // Arrange
          const expected = 1 / (1 + Math.exp(-x));
          // Act
          const result = Activation.logistic(x);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        test('logistic derivative is within [0, 0.25]', () => {
          // Arrange
          const fx = 1 / (1 + Math.exp(-x));
          const expected = fx * (1 - fx);
          // Act
          const result = Activation.logistic(x, true);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        test('logistic derivative is less than or equal to 0.25', () => {
          // Arrange
          const fx = 1 / (1 + Math.exp(-x));
          const expected = fx * (1 - fx);
          // Act
          const result = Activation.logistic(x, true);
          // Assert
          expect(result).toBeLessThanOrEqual(0.25);
        });
        test('logistic derivative matches expected value', () => {
          // Arrange
          const fx = 1 / (1 + Math.exp(-x));
          const expected = fx * (1 - fx);
          // Act
          const result = Activation.logistic(x, true);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
      });
    });
  });

  describe('tanh()', () => {
    testValues.forEach((x) => {
      describe(`Scenario: x=${x}`, () => {
        test('tanh is within [-1, 1]', () => {
          // Arrange
          const expected = Math.tanh(x);
          // Act
          const result = Activation.tanh(x);
          // Assert
          expect(result).toBeGreaterThanOrEqual(-1);
        });
        test('tanh is less than or equal to 1', () => {
          // Arrange
          const expected = Math.tanh(x);
          // Act
          const result = Activation.tanh(x);
          // Assert
          expect(result).toBeLessThanOrEqual(1);
        });
        test('tanh matches expected value', () => {
          // Arrange
          const expected = Math.tanh(x);
          // Act
          const result = Activation.tanh(x);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        test('tanh derivative is within [0, 1]', () => {
          // Arrange
          const expected = 1 - Math.pow(Math.tanh(x), 2);
          // Act
          const result = Activation.tanh(x, true);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        test('tanh derivative is less than or equal to 1', () => {
          // Arrange
          const expected = 1 - Math.pow(Math.tanh(x), 2);
          // Act
          const result = Activation.tanh(x, true);
          // Assert
          expect(result).toBeLessThanOrEqual(1);
        });
        test('tanh derivative matches expected value', () => {
          // Arrange
          const expected = 1 - Math.pow(Math.tanh(x), 2);
          // Act
          const result = Activation.tanh(x, true);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
      });
    });
  });

  describe('identity()', () => {
    testValues.forEach((x) => {
      describe(`Scenario: x=${x}`, () => {
        test('identity matches expected value', () => {
          // Arrange
          const expected = x;
          // Act
          const result = Activation.identity(x);
          // Assert
          expect(result).toBe(expected);
        });
        test('identity result is finite', () => {
          // Arrange
          const expected = x;
          // Act
          const result = Activation.identity(x);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
        test('identity derivative matches expected value', () => {
          // Arrange
          const expected = 1;
          // Act
          const result = Activation.identity(x, true);
          // Assert
          expect(result).toBe(expected);
        });
      });
    });
  });

  describe('step()', () => {
    testValues.forEach((x) => {
      describe(`Scenario: x=${x}`, () => {
        test('step matches expected value', () => {
          // Arrange
          const expected = x > 0 ? 1 : 0;
          // Act
          const result = Activation.step(x);
          // Assert
          expect(result).toBe(expected);
        });
        test('step result is either 0 or 1', () => {
          // Arrange
          const expected = x > 0 ? 1 : 0;
          // Act
          const result = Activation.step(x);
          // Assert
          expect([0, 1]).toContain(result);
        });
        test('step derivative matches expected value', () => {
          // Arrange
          const expected = 0;
          // Act
          const result = Activation.step(x, true);
          // Assert
          expect(result).toBe(expected);
        });
      });
    });
  });

  describe('relu()', () => {
    testValues.forEach((x) => {
      describe(`Scenario: x=${x}`, () => {
        test('relu is within [0, inf)', () => {
          // Arrange
          const expected = x > 0 ? x : 0;
          // Act
          const result = Activation.relu(x);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        test('relu matches expected value', () => {
          // Arrange
          const expected = x > 0 ? x : 0;
          // Act
          const result = Activation.relu(x);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        test('relu derivative matches expected value', () => {
          // Arrange
          const expected = x > 0 ? 1 : 0;
          // Act
          const result = Activation.relu(x, true);
          // Assert
          expect(result).toBe(expected);
        });
        test('relu derivative result is either 0 or 1', () => {
          // Arrange
          const expected = x > 0 ? 1 : 0;
          // Act
          const result = Activation.relu(x, true);
          // Assert
          expect([0, 1]).toContain(result);
        });
      });
    });
  });

  describe('softsign()', () => {
    testValues.forEach((x) => {
      describe(`Scenario: x=${x}`, () => {
        test('softsign is within (-1, 1)', () => {
          // Arrange
          const expected = x / (1 + Math.abs(x));
          // Act
          const result = Activation.softsign(x);
          // Assert
          expect(result).toBeGreaterThan(-1);
        });
        test('softsign is less than 1', () => {
          // Arrange
          const expected = x / (1 + Math.abs(x));
          // Act
          const result = Activation.softsign(x);
          // Assert
          expect(result).toBeLessThan(1);
        });
        test('softsign matches expected value', () => {
          // Arrange
          const expected = x / (1 + Math.abs(x));
          // Act
          const result = Activation.softsign(x);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        test('softsign derivative is within (0, 1]', () => {
          // Arrange
          const expected = 1 / Math.pow(1 + Math.abs(x), 2);
          // Act
          const result = Activation.softsign(x, true);
          // Assert
          expect(result).toBeGreaterThan(0);
        });
        test('softsign derivative is less than or equal to 1', () => {
          // Arrange
          const expected = 1 / Math.pow(1 + Math.abs(x), 2);
          // Act
          const result = Activation.softsign(x, true);
          // Assert
          expect(result).toBeLessThanOrEqual(1);
        });
        test('softsign derivative matches expected value', () => {
          // Arrange
          const expected = 1 / Math.pow(1 + Math.abs(x), 2);
          // Act
          const result = Activation.softsign(x, true);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
      });
    });
  });

  describe('sinusoid()', () => {
    testValues.forEach((x) => {
      describe(`Scenario: x=${x}`, () => {
        test('sinusoid is within [-1, 1]', () => {
          // Arrange
          const expected = Math.sin(x);
          // Act
          const result = Activation.sinusoid(x);
          // Assert
          expect(result).toBeGreaterThanOrEqual(-1);
        });
        test('sinusoid is less than or equal to 1', () => {
          // Arrange
          const expected = Math.sin(x);
          // Act
          const result = Activation.sinusoid(x);
          // Assert
          expect(result).toBeLessThanOrEqual(1);
        });
        test('sinusoid matches expected value', () => {
          // Arrange
          const expected = Math.sin(x);
          // Act
          const result = Activation.sinusoid(x);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        test('sinusoid derivative is within [-1, 1]', () => {
          // Arrange
          const expected = Math.cos(x);
          // Act
          const result = Activation.sinusoid(x, true);
          // Assert
          expect(result).toBeGreaterThanOrEqual(-1);
        });
        test('sinusoid derivative is less than or equal to 1', () => {
          // Arrange
          const expected = Math.cos(x);
          // Act
          const result = Activation.sinusoid(x, true);
          // Assert
          expect(result).toBeLessThanOrEqual(1);
        });
        test('sinusoid derivative matches expected value', () => {
          // Arrange
          const expected = Math.cos(x);
          // Act
          const result = Activation.sinusoid(x, true);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
      });
    });
  });

  describe('gaussian()', () => {
    testValues.forEach((x) => {
      describe(`Scenario: x=${x}`, () => {
        test('gaussian is within (0, 1]', () => {
          // Arrange
          const expected = Math.exp(-Math.pow(x, 2));
          // Act
          const result = Activation.gaussian(x);
          // Assert
          expect(result).toBeGreaterThan(0);
        });
        test('gaussian is less than or equal to 1', () => {
          // Arrange
          const expected = Math.exp(-Math.pow(x, 2));
          // Act
          const result = Activation.gaussian(x);
          // Assert
          expect(result).toBeLessThanOrEqual(1);
        });
        test('gaussian matches expected value', () => {
          // Arrange
          const expected = Math.exp(-Math.pow(x, 2));
          // Act
          const result = Activation.gaussian(x);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        test('gaussian derivative is within approx [-0.86, 0.86]', () => {
          // Arrange
          const expected = -2 * x * Math.exp(-Math.pow(x, 2));
          const maxDeriv = Math.sqrt(2 / Math.E); // Max derivative magnitude
          // Act
          const result = Activation.gaussian(x, true);
          // Assert
          expect(result).toBeGreaterThanOrEqual(-maxDeriv);
        });
        test('gaussian derivative is less than or equal to max derivative', () => {
          // Arrange
          const expected = -2 * x * Math.exp(-Math.pow(x, 2));
          const maxDeriv = Math.sqrt(2 / Math.E); // Max derivative magnitude
          // Act
          const result = Activation.gaussian(x, true);
          // Assert
          expect(result).toBeLessThanOrEqual(maxDeriv);
        });
        test('gaussian derivative matches expected value', () => {
          // Arrange
          const expected = -2 * x * Math.exp(-Math.pow(x, 2));
          const maxDeriv = Math.sqrt(2 / Math.E); // Max derivative magnitude
          // Act
          const result = Activation.gaussian(x, true);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
      });
    });
  });

  describe('bentIdentity()', () => {
    testValues.forEach((x) => {
      describe(`Scenario: x=${x}`, () => {
        test('bentIdentity matches expected value', () => {
          // Arrange
          const expected = (Math.sqrt(Math.pow(x, 2) + 1) - 1) / 2 + x;
          // Act
          const result = Activation.bentIdentity(x);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        test('bentIdentity result is finite', () => {
          // Arrange
          const expected = (Math.sqrt(Math.pow(x, 2) + 1) - 1) / 2 + x;
          // Act
          const result = Activation.bentIdentity(x);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
        test('bentIdentity derivative is within [0.5, 1.5)', () => {
          // Arrange
          const expected = x / (2 * Math.sqrt(Math.pow(x, 2) + 1)) + 1;
          // Act
          const result = Activation.bentIdentity(x, true);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0.5);
        });
        test('bentIdentity derivative is less than 1.5', () => {
          // Arrange
          const expected = x / (2 * Math.sqrt(Math.pow(x, 2) + 1)) + 1;
          // Act
          const result = Activation.bentIdentity(x, true);
          // Assert
          expect(result).toBeLessThan(1.5);
        });
        test('bentIdentity derivative matches expected value', () => {
          // Arrange
          const expected = x / (2 * Math.sqrt(Math.pow(x, 2) + 1)) + 1;
          // Act
          const result = Activation.bentIdentity(x, true);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
      });
    });
  });

  describe('bipolar()', () => {
    testValues.forEach((x) => {
      describe(`Scenario: x=${x}`, () => {
        test('bipolar matches expected value', () => {
          // Arrange
          const expected = x > 0 ? 1 : -1;
          // Act
          const result = Activation.bipolar(x);
          // Assert
          expect(result).toBe(expected);
        });
        test('bipolar result is either -1 or 1', () => {
          // Arrange
          const expected = x > 0 ? 1 : -1;
          // Act
          const result = Activation.bipolar(x);
          // Assert
          expect([-1, 1]).toContain(result);
        });
        test('bipolar derivative matches expected value', () => {
          // Arrange
          const expected = 0;
          // Act
          const result = Activation.bipolar(x, true);
          // Assert
          expect(result).toBe(expected);
        });
      });
    });
  });

  describe('bipolarSigmoid()', () => {
    testValues.forEach((x) => {
      describe(`Scenario: x=${x}`, () => {
        test('bipolarSigmoid is within (-1, 1)', () => {
          // Arrange
          const expected = Math.tanh(x); // Equivalent to tanh
          // Act
          const result = Activation.bipolarSigmoid(x);
          // Assert
          expect(result).toBeGreaterThan(-1);
        });
        test('bipolarSigmoid is less than 1', () => {
          // Arrange
          const expected = Math.tanh(x); // Equivalent to tanh
          // Act
          const result = Activation.bipolarSigmoid(x);
          // Assert
          expect(result).toBeLessThan(1);
        });
        test('bipolarSigmoid matches expected value', () => {
          // Arrange
          const expected = Math.tanh(x); // Equivalent to tanh
          // Act
          const result = Activation.bipolarSigmoid(x);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        test('bipolarSigmoid derivative is within [0, 0.5]', () => {
          // Arrange
          const d = 2 / (1 + Math.exp(-x)) - 1;
          const expected = (1 / 2) * (1 + d) * (1 - d);
          // Act
          const result = Activation.bipolarSigmoid(x, true);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        test('bipolarSigmoid derivative is less than or equal to 0.5', () => {
          // Arrange
          const d = 2 / (1 + Math.exp(-x)) - 1;
          const expected = (1 / 2) * (1 + d) * (1 - d);
          // Act
          const result = Activation.bipolarSigmoid(x, true);
          // Assert
          expect(result).toBeLessThanOrEqual(0.5);
        });
        test('bipolarSigmoid derivative matches expected value', () => {
          // Arrange
          const d = 2 / (1 + Math.exp(-x)) - 1;
          const expected = (1 / 2) * (1 + d) * (1 - d);
          // Act
          const result = Activation.bipolarSigmoid(x, true);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
      });
    });
  });

  describe('hardTanh()', () => {
    testValues.forEach((x) => {
      describe(`Scenario: x=${x}`, () => {
        test('hardTanh is within [-1, 1]', () => {
          // Arrange
          const expected = Math.max(-1, Math.min(1, x));
          // Act
          const result = Activation.hardTanh(x);
          // Assert
          expect(result).toBeGreaterThanOrEqual(-1);
        });
        test('hardTanh is less than or equal to 1', () => {
          // Arrange
          const expected = Math.max(-1, Math.min(1, x));
          // Act
          const result = Activation.hardTanh(x);
          // Assert
          expect(result).toBeLessThanOrEqual(1);
        });
        test('hardTanh matches expected value', () => {
          // Arrange
          const expected = Math.max(-1, Math.min(1, x));
          // Act
          const result = Activation.hardTanh(x);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        test('hardTanh derivative matches expected value', () => {
          // Arrange
          const expected = x > -1 && x < 1 ? 1 : 0;
          // Act
          const result = Activation.hardTanh(x, true);
          // Assert
          expect(result).toBe(expected);
        });
        test('hardTanh derivative result is either 0 or 1', () => {
          // Arrange
          const expected = x > -1 && x < 1 ? 1 : 0;
          // Act
          const result = Activation.hardTanh(x, true);
          // Assert
          expect([0, 1]).toContain(result);
        });
      });
    });
  });

  describe('absolute()', () => {
    testValues.forEach((x) => {
      describe(`Scenario: x=${x}`, () => {
        test('absolute is within [0, inf)', () => {
          // Arrange
          const expected = Math.abs(x);
          // Act
          const result = Activation.absolute(x);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        test('absolute matches expected value', () => {
          // Arrange
          const expected = Math.abs(x);
          // Act
          const result = Activation.absolute(x);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        test('absolute derivative matches expected value', () => {
          // Arrange
          const expected = x < 0 ? -1 : 1;
          // Act
          const result = Activation.absolute(x, true);
          // Assert
          expect(result).toBe(expected);
        });
        test('absolute derivative result is either -1 or 1', () => {
          // Arrange
          const expected = x < 0 ? -1 : 1;
          // Act
          const result = Activation.absolute(x, true);
          // Assert
          expect([-1, 1]).toContain(result);
        });
      });
    });
  });

  describe('inverse()', () => {
    testValues.forEach((x) => {
      describe(`Scenario: x=${x}`, () => {
        test('inverse matches expected value', () => {
          // Arrange
          const expected = 1 - x;
          // Act
          const result = Activation.inverse(x);
          // Assert
          expect(result).toBe(expected);
        });
        test('inverse result is finite', () => {
          // Arrange
          const expected = 1 - x;
          // Act
          const result = Activation.inverse(x);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
        test('inverse derivative matches expected value', () => {
          // Arrange
          const expected = -1;
          // Act
          const result = Activation.inverse(x, true);
          // Assert
          expect(result).toBe(expected);
        });
      });
    });
  });

  describe('selu()', () => {
    const alpha = 1.6732632423543772848170429916717;
    const scale = 1.0507009873554804934193349852946;
    const lowerBound = -alpha * scale;

    testValues.forEach((x) => {
      describe(`Scenario: x=${x}`, () => {
        test(`selu is within [${lowerBound.toFixed(2)}, inf)`, () => {
          // Arrange
          const fx = x > 0 ? x : alpha * Math.exp(x) - alpha;
          const expected = fx * scale;
          // Act
          const result = Activation.selu(x);
          // Assert
          expect(result).toBeGreaterThanOrEqual(lowerBound - epsilon);
        });
        test('selu matches expected value', () => {
          // Arrange
          const fx = x > 0 ? x : alpha * Math.exp(x) - alpha;
          const expected = fx * scale;
          // Act
          const result = Activation.selu(x);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        test(`selu derivative is within (0, ${scale * alpha}]`, () => {
          // Arrange
          const fx = x > 0 ? x : alpha * Math.exp(x) - alpha;
          const expected = (x > 0 ? scale : (fx + alpha) * scale);
          // Act
          const result = Activation.selu(x, true);
          // Assert
          expect(result).toBeGreaterThan(0);
        });
        test(`selu derivative is less than or equal to ${scale * alpha}`, () => {
          // Arrange
          const fx = x > 0 ? x : alpha * Math.exp(x) - alpha;
          const expected = (x > 0 ? scale : (fx + alpha) * scale);
          // Act
          const result = Activation.selu(x, true);
          // Assert
          expect(result).toBeLessThanOrEqual(scale * alpha + epsilon);
        });
        test('selu derivative matches expected value', () => {
          // Arrange
          const fx = x > 0 ? x : alpha * Math.exp(x) - alpha;
          const expected = (x > 0 ? scale : (fx + alpha) * scale);
          // Act
          const result = Activation.selu(x, true);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
      });
    });
  });

  describe('softplus()', () => {
    stabilityTestValues.forEach((x) => {
      describe(`Scenario: x=${x}`, () => {
        test('softplus is within (0, inf)', () => {
          // Arrange
          const expected = Math.log(1 + Math.exp(x));
          // Act
          const result = Activation.softplus(x);
          // Assert
          if (expected === Infinity) {
            expect(result).toBeCloseTo(x, epsilon); // Should approximate x for large positive x
          } else if (expected === 0 && x < -50) {
            expect(result).toBeCloseTo(0, epsilon); // Should approximate 0 for large negative x
          } else {
            expect(result).toBeCloseTo(expected, epsilon);
          }
        });
        test('softplus matches expected value', () => {
          // Arrange
          const expected = Math.log(1 + Math.exp(x));
          // Act
          const result = Activation.softplus(x);
          // Assert
          if (expected !== Infinity && !(expected === 0 && x < -50)) {
            expect(result).toBeCloseTo(expected, epsilon);
          }
        });
        test('softplus derivative (logistic) is within (0, 1]', () => {
          // Arrange
          const expected = 1 / (1 + Math.exp(-x)); // Logistic
          // Act
          const result = Activation.softplus(x, true);
          // Assert
          expect(result).toBeGreaterThan(0);
        });
        test('softplus derivative (logistic) is less than or equal to 1', () => {
          // Arrange
          const expected = 1 / (1 + Math.exp(-x)); // Logistic
          // Act
          const result = Activation.softplus(x, true);
          // Assert
          expect(result).toBeLessThanOrEqual(1);
        });
        test('softplus derivative (logistic) matches expected value', () => {
          // Arrange
          const expected = 1 / (1 + Math.exp(-x)); // Logistic
          // Act
          const result = Activation.softplus(x, true);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
      });
    });
  });

  describe('swish()', () => {
    const lowerBound = -0.2784645; // Approximate minimum value
    testValues.forEach((x) => {
      describe(`Scenario: x=${x}`, () => {
        test(`swish is within [${lowerBound.toFixed(3)}, inf)`, () => {
          // Arrange
          const sigmoid_x = 1 / (1 + Math.exp(-x));
          const expected = x * sigmoid_x;
          // Act
          const result = Activation.swish(x);
          // Assert
          expect(result).toBeGreaterThanOrEqual(lowerBound - epsilon);
        });
        test('swish matches expected value', () => {
          // Arrange
          const sigmoid_x = 1 / (1 + Math.exp(-x));
          const expected = x * sigmoid_x;
          // Act
          const result = Activation.swish(x);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        test('swish derivative is within approx [-0.09, inf)', () => {
          // Arrange
          const sigmoid_x = 1 / (1 + Math.exp(-x));
          const swish_x = x * sigmoid_x;
          const expected = swish_x + sigmoid_x * (1 - swish_x);
          // Act
          const result = Activation.swish(x, true);
          // Assert
          expect(result).toBeGreaterThanOrEqual(-0.09 - epsilon);
        });
        test('swish derivative matches expected value', () => {
          // Arrange
          const sigmoid_x = 1 / (1 + Math.exp(-x));
          const swish_x = x * sigmoid_x;
          const expected = swish_x + sigmoid_x * (1 - swish_x);
          // Act
          const result = Activation.swish(x, true);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
      });
    });
  });

  describe('gelu()', () => {
    const lowerBound = -0.17; // Approximate minimum value
    testValues.forEach((x) => {
      describe(`Scenario: x=${x}`, () => {
        test(`gelu approximation is within [${lowerBound.toFixed(2)}, inf)`, () => {
          // Arrange
          const cdf =
            0.5 *
            (1.0 +
              Math.tanh(
                Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))
              ));
          const expected = x * cdf;
          // Act
          const result = Activation.gelu(x);
          // Assert
          expect(result).toBeGreaterThanOrEqual(lowerBound - epsilon);
        });
        test('gelu approximation matches expected value', () => {
          // Arrange
          const cdf =
            0.5 *
            (1.0 +
              Math.tanh(
                Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))
              ));
          const expected = x * cdf;
          // Act
          const result = Activation.gelu(x);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        test('gelu approximation derivative is within approx [-0.16, inf)', () => {
          // Arrange
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
          // Act
          const result = Activation.gelu(x, true);
          // Assert
          expect(result).toBeGreaterThanOrEqual(-0.16 - epsilon);
        });
        test('gelu approximation derivative matches expected value', () => {
          // Arrange
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
          // Act
          const result = Activation.gelu(x, true);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
      });
    });
  });

  describe('mish()', () => {
    const lowerBound = -0.30884; // Approximate minimum value
    stabilityTestValues.forEach((x) => {
      describe(`Scenario: x=${x}`, () => {
        test(`mish is within [${lowerBound.toFixed(3)}, inf)`, () => {
          // Arrange
          let sp_x: number;
          if (x > 30) {
            sp_x = x;
          } else if (x < -30) {
            sp_x = Math.exp(x);
          } else {
            sp_x = Math.max(0, x) + Math.log(1 + Math.exp(-Math.abs(x)));
          }
          const expected = x * Math.tanh(sp_x);
          // Act
          const result = Activation.mish(x);
          // Assert
          expect(result).toBeGreaterThanOrEqual(lowerBound - epsilon);
        });
        test('mish matches expected value', () => {
          // Arrange
          let sp_x: number;
          if (x > 30) {
            sp_x = x;
          } else if (x < -30) {
            sp_x = Math.exp(x);
          } else {
            sp_x = Math.max(0, x) + Math.log(1 + Math.exp(-Math.abs(x)));
          }
          const expected = x * Math.tanh(sp_x);
          // Act
          const result = Activation.mish(x);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        test('mish derivative matches expected value', () => {
          // Arrange
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
          // Act
          const result = Activation.mish(x, true);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        test('mish derivative result is finite', () => {
          // Arrange
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
          // Act
          const result = Activation.mish(x, true);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
  });

  describe('Ambiguous Derivatives', () => {
    const Activation = require('../../src/methods/activation').default;
    test('ReLU derivative at 0 returns 0 (documented choice)', () => {
      expect(Activation.relu(0, true)).toBe(0);
    });
    test('Absolute derivative at 0 returns 1 (documented choice)', () => {
      expect(Activation.absolute(0, true)).toBe(1);
    });
  });

  describe('Custom Activation Registration', () => {
    const Activation = require('../../src/methods/activation').default;
    const { registerCustomActivation } = require('../../src/methods/activation');
    test('can register and use a custom activation function', () => {
      // Act
      registerCustomActivation('customFn', (x: number, derivate: boolean = false) => derivate ? 42 : x * 2);
      // Assert
      expect(Activation['customFn'](3)).toBe(6);
    });
    test('custom activation derivative', () => {
      // Act & Assert
      expect(Activation['customFn'](3, true)).toBe(42);
    });
  });
});
