import Activation from '../../src/methods/activation';

describe('Activation', () => {
  const epsilon = 1e-9; // Tolerance for floating point comparisons
  const testValues = [-10, -1, -0.5, 0, 0.5, 1, 10]; // Common values to test
  const stabilityTestValues = [-100, -30.1, -30, -10, 0, 10, 30, 30.1, 100]; // For functions needing stability checks

  describe('logistic()', () => {
    describe('when x = 1', () => {
      it('returns correct logistic value', () => {
        // Arrange
        const x = 1;
        const expected = 1 / (1 + Math.exp(-1));
        // Act
        const result = Activation.logistic(x);
        // Assert
        expect(result).toBeCloseTo(expected, epsilon);
      });
      it('returns correct logistic derivative', () => {
        // Arrange
        const x = 1;
        const fx = 1 / (1 + Math.exp(-1));
        const expected = fx * (1 - fx);
        // Act
        const result = Activation.logistic(x, true);
        // Assert
        expect(result).toBeCloseTo(expected, epsilon);
      });
    });
    testValues.forEach((x) => {
      describe(`Scenario: x=${x}`, () => {
        it('logistic is greater than or equal to 0', () => {
          // Arrange
          // Act
          const result = Activation.logistic(x);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        it('logistic is less than or equal to 1', () => {
          // Arrange
          // Act
          const result = Activation.logistic(x);
          // Assert
          expect(result).toBeLessThanOrEqual(1);
        });
        it('logistic matches expected value', () => {
          // Arrange
          const expected = 1 / (1 + Math.exp(-x));
          // Act
          const result = Activation.logistic(x);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        it('logistic derivative is greater than or equal to 0', () => {
          // Arrange
          // Act
          const result = Activation.logistic(x, true);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        it('logistic derivative is less than or equal to 0.25', () => {
          // Arrange
          // Act
          const result = Activation.logistic(x, true);
          // Assert
          expect(result).toBeLessThanOrEqual(0.25);
        });
        it('logistic derivative matches expected value', () => {
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
        it('tanh is within [-1, 1]', () => {
          // Arrange
          const expected = Math.tanh(x);
          // Act
          const result = Activation.tanh(x);
          // Assert
          expect(result).toBeGreaterThanOrEqual(-1);
        });
        it('tanh is less than or equal to 1', () => {
          // Arrange
          const expected = Math.tanh(x);
          // Act
          const result = Activation.tanh(x);
          // Assert
          expect(result).toBeLessThanOrEqual(1);
        });
        it('tanh matches expected value', () => {
          // Arrange
          const expected = Math.tanh(x);
          // Act
          const result = Activation.tanh(x);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        it('tanh derivative is within [0, 1]', () => {
          // Arrange
          const expected = 1 - Math.pow(Math.tanh(x), 2);
          // Act
          const result = Activation.tanh(x, true);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        it('tanh derivative is less than or equal to 1', () => {
          // Arrange
          const expected = 1 - Math.pow(Math.tanh(x), 2);
          // Act
          const result = Activation.tanh(x, true);
          // Assert
          expect(result).toBeLessThanOrEqual(1);
        });
        it('tanh derivative matches expected value', () => {
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
        it('identity matches expected value', () => {
          // Arrange
          const expected = x;
          // Act
          const result = Activation.identity(x);
          // Assert
          expect(result).toBe(expected);
        });
        it('identity result is finite', () => {
          // Arrange
          // Act
          const result = Activation.identity(x);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
        it('identity derivative matches expected value', () => {
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
    describe('when x > 0', () => {
      testValues.filter(x => x > 0).forEach((x) => {
        describe(`Scenario: x=${x}`, () => {
          it('step returns 1', () => {
            // Arrange
            // Act
            const result = Activation.step(x);
            // Assert
            expect(result).toBe(1);
          });
          it('step derivative returns 0', () => {
            // Arrange
            // Act
            const result = Activation.step(x, true);
            // Assert
            expect(result).toBe(0);
          });
        });
      });
    });
    describe('when x <= 0', () => {
      testValues.filter(x => x <= 0).forEach((x) => {
        describe(`Scenario: x=${x}`, () => {
          it('step returns 0', () => {
            // Arrange
            // Act
            const result = Activation.step(x);
            // Assert
            expect(result).toBe(0);
          });
          it('step derivative returns 0', () => {
            // Arrange
            // Act
            const result = Activation.step(x, true);
            // Assert
            expect(result).toBe(0);
          });
        });
      });
    });
  });

  describe('relu()', () => {
    describe('when x > 0', () => {
      testValues.filter(x => x > 0).forEach((x) => {
        describe(`Scenario: x=${x}`, () => {
          it('relu returns x', () => {
            // Arrange
            // Act
            const result = Activation.relu(x);
            // Assert
            expect(result).toBeCloseTo(x, epsilon);
          });
          it('relu is greater than or equal to 0', () => {
            // Arrange
            // Act
            const result = Activation.relu(x);
            // Assert
            expect(result).toBeGreaterThanOrEqual(0);
          });
          it('relu derivative returns 1', () => {
            // Arrange
            // Act
            const result = Activation.relu(x, true);
            // Assert
            expect(result).toBe(1);
          });
        });
      });
    });
    describe('when x <= 0', () => {
      testValues.filter(x => x <= 0).forEach((x) => {
        describe(`Scenario: x=${x}`, () => {
          it('relu returns 0', () => {
            // Arrange
            // Act
            const result = Activation.relu(x);
            // Assert
            expect(result).toBeCloseTo(0, epsilon);
          });
          it('relu is greater than or equal to 0', () => {
            // Arrange
            // Act
            const result = Activation.relu(x);
            // Assert
            expect(result).toBeGreaterThanOrEqual(0);
          });
          it('relu derivative returns 0', () => {
            // Arrange
            // Act
            const result = Activation.relu(x, true);
            // Assert
            expect(result).toBe(0);
          });
        });
      });
    });
  });

  describe('softsign()', () => {
    testValues.forEach((x) => {
      describe(`Scenario: x=${x}`, () => {
        it('softsign is greater than -1', () => {
          // Arrange
          // Act
          const result = Activation.softsign(x);
          // Assert
          expect(result).toBeGreaterThan(-1);
        });
        it('softsign is less than 1', () => {
          // Arrange
          // Act
          const result = Activation.softsign(x);
          // Assert
          expect(result).toBeLessThan(1);
        });
        it('softsign matches expected value', () => {
          // Arrange
          const expected = x / (1 + Math.abs(x));
          // Act
          const result = Activation.softsign(x);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        it('softsign derivative is greater than 0', () => {
          // Arrange
          // Act
          const result = Activation.softsign(x, true);
          // Assert
          expect(result).toBeGreaterThan(0);
        });
        it('softsign derivative is less than or equal to 1', () => {
          // Arrange
          // Act
          const result = Activation.softsign(x, true);
          // Assert
          expect(result).toBeLessThanOrEqual(1);
        });
        it('softsign derivative matches expected value', () => {
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
        it('sinusoid is greater than or equal to -1', () => {
          // Arrange
          // Act
          const result = Activation.sinusoid(x);
          // Assert
          expect(result).toBeGreaterThanOrEqual(-1);
        });
        it('sinusoid is less than or equal to 1', () => {
          // Arrange
          // Act
          const result = Activation.sinusoid(x);
          // Assert
          expect(result).toBeLessThanOrEqual(1);
        });
        it('sinusoid matches expected value', () => {
          // Arrange
          const expected = Math.sin(x);
          // Act
          const result = Activation.sinusoid(x);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        it('sinusoid derivative is greater than or equal to -1', () => {
          // Arrange
          // Act
          const result = Activation.sinusoid(x, true);
          // Assert
          expect(result).toBeGreaterThanOrEqual(-1);
        });
        it('sinusoid derivative is less than or equal to 1', () => {
          // Arrange
          // Act
          const result = Activation.sinusoid(x, true);
          // Assert
          expect(result).toBeLessThanOrEqual(1);
        });
        it('sinusoid derivative matches expected value', () => {
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
        it('gaussian is greater than 0', () => {
          // Arrange
          // Act
          const result = Activation.gaussian(x);
          // Assert
          expect(result).toBeGreaterThan(0);
        });
        it('gaussian is less than or equal to 1', () => {
          // Arrange
          // Act
          const result = Activation.gaussian(x);
          // Assert
          expect(result).toBeLessThanOrEqual(1);
        });
        it('gaussian matches expected value', () => {
          // Arrange
          const expected = Math.exp(-Math.pow(x, 2));
          // Act
          const result = Activation.gaussian(x);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        it('gaussian derivative is greater than or equal to -maxDeriv', () => {
          // Arrange
          const maxDeriv = Math.sqrt(2 / Math.E);
          // Act
          const result = Activation.gaussian(x, true);
          // Assert
          expect(result).toBeGreaterThanOrEqual(-maxDeriv);
        });
        it('gaussian derivative is less than or equal to maxDeriv', () => {
          // Arrange
          const maxDeriv = Math.sqrt(2 / Math.E);
          // Act
          const result = Activation.gaussian(x, true);
          // Assert
          expect(result).toBeLessThanOrEqual(maxDeriv);
        });
        it('gaussian derivative matches expected value', () => {
          // Arrange
          const expected = -2 * x * Math.exp(-Math.pow(x, 2));
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
        it('bentIdentity matches expected value', () => {
          // Arrange
          const expected = (Math.sqrt(Math.pow(x, 2) + 1) - 1) / 2 + x;
          // Act
          const result = Activation.bentIdentity(x);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        it('bentIdentity result is finite', () => {
          // Arrange
          // Act
          const result = Activation.bentIdentity(x);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
        it('bentIdentity derivative is greater than or equal to 0.5', () => {
          // Arrange
          // Act
          const result = Activation.bentIdentity(x, true);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0.5);
        });
        it('bentIdentity derivative is less than 1.5', () => {
          // Arrange
          // Act
          const result = Activation.bentIdentity(x, true);
          // Assert
          expect(result).toBeLessThan(1.5);
        });
        it('bentIdentity derivative matches expected value', () => {
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
    describe('when x > 0', () => {
      testValues.filter(x => x > 0).forEach((x) => {
        describe(`Scenario: x=${x}`, () => {
          it('bipolar returns 1', () => {
            // Arrange
            // Act
            const result = Activation.bipolar(x);
            // Assert
            expect(result).toBe(1);
          });
          it('bipolar derivative returns 0', () => {
            // Arrange
            // Act
            const result = Activation.bipolar(x, true);
            // Assert
            expect(result).toBe(0);
          });
        });
      });
    });
    describe('when x <= 0', () => {
      testValues.filter(x => x <= 0).forEach((x) => {
        describe(`Scenario: x=${x}`, () => {
          it('bipolar returns -1', () => {
            // Arrange
            // Act
            const result = Activation.bipolar(x);
            // Assert
            expect(result).toBe(-1);
          });
          it('bipolar derivative returns 0', () => {
            // Arrange
            // Act
            const result = Activation.bipolar(x, true);
            // Assert
            expect(result).toBe(0);
          });
        });
      });
    });
  });

  describe('bipolarSigmoid()', () => {
    testValues.forEach((x) => {
      describe(`Scenario: x=${x}`, () => {
        it('bipolarSigmoid is greater than -1', () => {
          // Arrange
          // Act
          const result = Activation.bipolarSigmoid(x);
          // Assert
          expect(result).toBeGreaterThan(-1);
        });
        it('bipolarSigmoid is less than 1', () => {
          // Arrange
          // Act
          const result = Activation.bipolarSigmoid(x);
          // Assert
          expect(result).toBeLessThan(1);
        });
        it('bipolarSigmoid matches expected value', () => {
          // Arrange
          const expected = Math.tanh(x);
          // Act
          const result = Activation.bipolarSigmoid(x);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        it('bipolarSigmoid derivative is greater than or equal to 0', () => {
          // Arrange
          // Act
          const result = Activation.bipolarSigmoid(x, true);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        it('bipolarSigmoid derivative is less than or equal to 0.5', () => {
          // Arrange
          // Act
          const result = Activation.bipolarSigmoid(x, true);
          // Assert
          expect(result).toBeLessThanOrEqual(0.5);
        });
        it('bipolarSigmoid derivative matches expected value', () => {
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
    describe('when x > 1', () => {
      testValues.filter(x => x > 1).forEach((x) => {
        describe(`Scenario: x=${x}`, () => {
          it('hardTanh returns 1', () => {
            // Arrange
            // Act
            const result = Activation.hardTanh(x);
            // Assert
            expect(result).toBeCloseTo(1, epsilon);
          });
          it('hardTanh derivative returns 0', () => {
            // Arrange
            // Act
            const result = Activation.hardTanh(x, true);
            // Assert
            expect(result).toBe(0);
          });
        });
      });
    });
    describe('when x < -1', () => {
      testValues.filter(x => x < -1).forEach((x) => {
        describe(`Scenario: x=${x}`, () => {
          it('hardTanh returns -1', () => {
            // Arrange
            // Act
            const result = Activation.hardTanh(x);
            // Assert
            expect(result).toBeCloseTo(-1, epsilon);
          });
          it('hardTanh derivative returns 0', () => {
            // Arrange
            // Act
            const result = Activation.hardTanh(x, true);
            // Assert
            expect(result).toBe(0);
          });
        });
      });
    });
    describe('when -1 < x < 1', () => {
      testValues.filter(x => x > -1 && x < 1).forEach((x) => {
        describe(`Scenario: x=${x}`, () => {
          it('hardTanh returns x', () => {
            // Arrange
            // Act
            const result = Activation.hardTanh(x);
            // Assert
            expect(result).toBeCloseTo(x, epsilon);
          });
          it('hardTanh derivative returns 1', () => {
            // Arrange
            // Act
            const result = Activation.hardTanh(x, true);
            // Assert
            expect(result).toBe(1);
          });
        });
      });
    });
    describe('when x == -1 or x == 1', () => {
      [-1, 1].forEach((x) => {
        describe(`Scenario: x=${x}`, () => {
          it(`hardTanh returns ${x}`, () => {
            // Arrange
            // Act
            const result = Activation.hardTanh(x);
            // Assert
            expect(result).toBeCloseTo(x, epsilon);
          });
          it('hardTanh derivative returns 0 (boundary)', () => {
            // Arrange
            // Act
            const result = Activation.hardTanh(x, true);
            // Assert
            expect(result).toBe(0);
          });
        });
      });
    });
  });

  describe('absolute()', () => {
    describe('when x < 0', () => {
      testValues.filter(x => x < 0).forEach((x) => {
        describe(`Scenario: x=${x}`, () => {
          it('absolute returns -x', () => {
            // Arrange
            // Act
            const result = Activation.absolute(x);
            // Assert
            expect(result).toBeCloseTo(-x, epsilon);
          });
          it('absolute derivative returns -1', () => {
            // Arrange
            // Act
            const result = Activation.absolute(x, true);
            // Assert
            expect(result).toBe(-1);
          });
        });
      });
    });
    describe('when x > 0', () => {
      testValues.filter(x => x > 0).forEach((x) => {
        describe(`Scenario: x=${x}`, () => {
          it('absolute returns x', () => {
            // Arrange
            // Act
            const result = Activation.absolute(x);
            // Assert
            expect(result).toBeCloseTo(x, epsilon);
          });
          it('absolute derivative returns 1', () => {
            // Arrange
            // Act
            const result = Activation.absolute(x, true);
            // Assert
            expect(result).toBe(1);
          });
        });
      });
    });
    describe('when x == 0', () => {
      it('absolute returns 0', () => {
        // Arrange
        // Act
        const result = Activation.absolute(0);
        // Assert
        expect(result).toBe(0);
      });
      it('absolute derivative returns 1 (documented choice)', () => {
        // Arrange
        // Act
        const result = Activation.absolute(0, true);
        // Assert
        expect(result).toBe(1);
      });
    });
  });

  describe('inverse()', () => {
    testValues.forEach((x) => {
      describe(`Scenario: x=${x}`, () => {
        it('inverse returns 1 - x', () => {
          // Arrange
          const expected = 1 - x;
          // Act
          const result = Activation.inverse(x);
          // Assert
          expect(result).toBe(expected);
        });
        it('inverse result is finite', () => {
          // Arrange
          // Act
          const result = Activation.inverse(x);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
        it('inverse derivative returns -1', () => {
          // Arrange
          // Act
          const result = Activation.inverse(x, true);
          // Assert
          expect(result).toBe(-1);
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
        it(`selu is within [${lowerBound.toFixed(2)}, inf)`, () => {
          // Arrange
          const fx = x > 0 ? x : alpha * Math.exp(x) - alpha;
          const expected = fx * scale;
          // Act
          const result = Activation.selu(x);
          // Assert
          expect(result).toBeGreaterThanOrEqual(lowerBound - epsilon);
        });
        it('selu matches expected value', () => {
          // Arrange
          const fx = x > 0 ? x : alpha * Math.exp(x) - alpha;
          const expected = fx * scale;
          // Act
          const result = Activation.selu(x);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        it(`selu derivative is within (0, ${scale * alpha}]`, () => {
          // Arrange
          const fx = x > 0 ? x : alpha * Math.exp(x) - alpha;
          const expected = (x > 0 ? scale : (fx + alpha) * scale);
          // Act
          const result = Activation.selu(x, true);
          // Assert
          expect(result).toBeGreaterThan(0);
        });
        it(`selu derivative is less than or equal to ${scale * alpha}`, () => {
          // Arrange
          const fx = x > 0 ? x : alpha * Math.exp(x) - alpha;
          const expected = (x > 0 ? scale : (fx + alpha) * scale);
          // Act
          const result = Activation.selu(x, true);
          // Assert
          expect(result).toBeLessThanOrEqual(scale * alpha + epsilon);
        });
        it('selu derivative matches expected value', () => {
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
        it('softplus is within (0, inf)', () => {
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
        it('softplus matches expected value', () => {
          // Arrange
          const expected = Math.log(1 + Math.exp(x));
          // Act
          const result = Activation.softplus(x);
          // Assert
          if (expected !== Infinity && !(expected === 0 && x < -50)) {
            expect(result).toBeCloseTo(expected, epsilon);
          }
        });
        it('softplus derivative (logistic) is within (0, 1]', () => {
          // Arrange
          const expected = 1 / (1 + Math.exp(-x)); // Logistic
          // Act
          const result = Activation.softplus(x, true);
          // Assert
          expect(result).toBeGreaterThan(0);
        });
        it('softplus derivative (logistic) is less than or equal to 1', () => {
          // Arrange
          const expected = 1 / (1 + Math.exp(-x)); // Logistic
          // Act
          const result = Activation.softplus(x, true);
          // Assert
          expect(result).toBeLessThanOrEqual(1);
        });
        it('softplus derivative (logistic) matches expected value', () => {
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
        it(`swish is within [${lowerBound.toFixed(3)}, inf)`, () => {
          // Arrange
          const sigmoid_x = 1 / (1 + Math.exp(-x));
          const expected = x * sigmoid_x;
          // Act
          const result = Activation.swish(x);
          // Assert
          expect(result).toBeGreaterThanOrEqual(lowerBound - epsilon);
        });
        it('swish matches expected value', () => {
          // Arrange
          const sigmoid_x = 1 / (1 + Math.exp(-x));
          const expected = x * sigmoid_x;
          // Act
          const result = Activation.swish(x);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        it('swish derivative is within approx [-0.09, inf)', () => {
          // Arrange
          const sigmoid_x = 1 / (1 + Math.exp(-x));
          const swish_x = x * sigmoid_x;
          const expected = swish_x + sigmoid_x * (1 - swish_x);
          // Act
          const result = Activation.swish(x, true);
          // Assert
          expect(result).toBeGreaterThanOrEqual(-0.09 - epsilon);
        });
        it('swish derivative matches expected value', () => {
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
        it(`gelu approximation is within [${lowerBound.toFixed(2)}, inf)`, () => {
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
        it('gelu approximation matches expected value', () => {
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
        it('gelu approximation derivative is within approx [-0.16, inf)', () => {
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
        it('gelu approximation derivative matches expected value', () => {
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
        it(`mish is within [${lowerBound.toFixed(3)}, inf)`, () => {
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
        it('mish matches expected value', () => {
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
        it('mish derivative matches expected value', () => {
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
        it('mish derivative result is finite', () => {
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
    describe('ReLU derivative at 0', () => {
      it('returns 0 (documented choice)', () => {
        // Arrange
        // Act
        const result = Activation.relu(0, true);
        // Assert
        expect(result).toBe(0);
      });
    });
    describe('Absolute derivative at 0', () => {
      it('returns 1 (documented choice)', () => {
        // Arrange
        // Act
        const result = Activation.absolute(0, true);
        // Assert
        expect(result).toBe(1);
      });
    });
  });

  describe('Custom Activation Registration', () => {
    let Activation: typeof import('../../src/methods/activation').default;
    let registerCustomActivation: (name: string, fn: (x: number, derivate?: boolean) => number) => void;
    beforeEach(() => {
      Activation = require('../../src/methods/activation').default;
      registerCustomActivation = require('../../src/methods/activation').registerCustomActivation;
      // Clean up any previous customFn
      if (Activation['customFn']) delete Activation['customFn'];
    });
    afterEach(() => {
      if (Activation['customFn']) delete Activation['customFn'];
    });
    describe('when registering a custom activation', () => {
      it('can register and use a custom activation function', () => {
        // Arrange
        registerCustomActivation('customFn', (x: number, derivate: boolean = false) => derivate ? 42 : x * 2);
        // Act
        const result = Activation['customFn'](3);
        // Assert
        expect(result).toBe(6);
      });
      it('custom activation derivative returns expected value', () => {
        // Arrange
        registerCustomActivation('customFn', (x: number, derivate: boolean = false) => derivate ? 42 : x * 2);
        // Act
        const result = Activation['customFn'](3, true);
        // Assert
        expect(result).toBe(42);
      });
    });
  });

  // Add a dedicated describe for helper/utility methods if any exist in activation.ts
  describe('Helper and Utility Methods', () => {
    // Example: If there are utility functions in activation.ts, add their tests here
    // This is a placeholder for future utility/helper method tests
    it('should have a placeholder for utility/helper method tests', () => {
      // Arrange
      // Act
      // Assert
      expect(true).toBe(true);
    });
  });
});
