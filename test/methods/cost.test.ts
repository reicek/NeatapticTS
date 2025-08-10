import Cost from '../../src/methods/cost';

describe('Cost', () => {
  const epsilon = 1e-9; // Tolerance for floating point comparisons

  describe('crossEntropy()', () => {
    describe('Scenario: binary targets', () => {
      describe('when calculating cross-entropy', () => {
        it('returns the correct value', () => {
          // Arrange
          const targets = [0, 1, 0, 1];
          const outputs = [0.1, 0.9, 0.2, 0.8];
          const expected = -(
            Math.log(1 - 0.1) +
            Math.log(0.9) +
            Math.log(1 - 0.2) +
            Math.log(0.8)
          ) / 4;
          // Act
          const result = Cost.crossEntropy(targets, outputs);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        it('returns a value greater than or equal to 0', () => {
          // Arrange
          const targets = [0, 1, 0, 1];
          const outputs = [0.1, 0.9, 0.2, 0.8];
          // Act
          const result = Cost.crossEntropy(targets, outputs);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [0, 1, 0, 1];
          const outputs = [0.1, 0.9, 0.2, 0.8];
          // Act
          const result = Cost.crossEntropy(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: soft labels', () => {
      describe('when calculating cross-entropy with soft labels', () => {
        it('returns the correct value', () => {
          // Arrange
          const targets = [0.2, 0.7];
          const outputs = [0.3, 0.6];
          const expected = -(
            0.2 * Math.log(0.3) +
            (1 - 0.2) * Math.log(1 - 0.3) +
            (0.7 * Math.log(0.6) + (1 - 0.7) * Math.log(1 - 0.6))
          ) / 2;
          // Act
          const result = Cost.crossEntropy(targets, outputs);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        it('returns a value greater than or equal to 0', () => {
          // Arrange
          const targets = [0.2, 0.7];
          const outputs = [0.3, 0.6];
          // Act
          const result = Cost.crossEntropy(targets, outputs);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [0.2, 0.7];
          const outputs = [0.3, 0.6];
          // Act
          const result = Cost.crossEntropy(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: perfect predictions (binary)', () => {
      describe('when predictions are perfect', () => {
        it('returns a value close to zero', () => {
          // Arrange
          const targets = [0, 1];
          const outputs = [0, 1];
          // Act
          const result = Cost.crossEntropy(targets, outputs);
          // Assert
          expect(result).toBeCloseTo(0, 5);
        });
        it('returns a value greater than or equal to 0', () => {
          // Arrange
          const targets = [0, 1];
          const outputs = [0, 1];
          // Act
          const result = Cost.crossEntropy(targets, outputs);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [0, 1];
          const outputs = [0, 1];
          // Act
          const result = Cost.crossEntropy(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: near-zero and near-one outputs', () => {
      describe('when outputs are near zero and one', () => {
        it('returns a value close to zero', () => {
          // Arrange
          const targets = [0, 1];
          const outputs = [1e-16, 1 - 1e-16];
          // Act
          const result = Cost.crossEntropy(targets, outputs);
          // Assert
          expect(result).toBeCloseTo(0, 5);
        });
        it('returns a value greater than or equal to 0', () => {
          // Arrange
          const targets = [0, 1];
          const outputs = [1e-16, 1 - 1e-16];
          // Act
          const result = Cost.crossEntropy(targets, outputs);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [0, 1];
          const outputs = [1e-16, 1 - 1e-16];
          // Act
          const result = Cost.crossEntropy(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: mismatched input lengths', () => {
      it('throws error', () => {
        // Arrange
        const targets = [0, 1];
        const outputs = [0.5];
        // Suppress console output
        const originalError = console.error;
        console.error = () => {};
        // Act & Assert
        expect(() => Cost.crossEntropy(targets, outputs)).toThrow(
          'Target and output arrays must have the same length.'
        );
        console.error = originalError;
      });
    });
    describe('Scenario: single element arrays', () => {
      describe('when calculating cross-entropy for a single element', () => {
        it('returns the correct value', () => {
          // Arrange
          const targets = [1];
          const outputs = [0.7];
          const expected = -Math.log(0.7);
          // Act
          const result = Cost.crossEntropy(targets, outputs);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        it('returns a value greater than or equal to 0', () => {
          // Arrange
          const targets = [1];
          const outputs = [0.7];
          // Act
          const result = Cost.crossEntropy(targets, outputs);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [1];
          const outputs = [0.7];
          // Act
          const result = Cost.crossEntropy(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
  });

  describe('mse()', () => {
    describe('Scenario: general case', () => {
      describe('when calculating mean squared error', () => {
        it('returns the correct value', () => {
          // Arrange
          const targets = [1, 2, 3];
          const outputs = [1.5, 2.5, 2.5];
          const expected = (Math.pow(0.5, 2) + Math.pow(0.5, 2) + Math.pow(-0.5, 2)) / 3;
          // Act
          const result = Cost.mse(targets, outputs);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        it('returns a value greater than or equal to 0', () => {
          // Arrange
          const targets = [1, 2, 3];
          const outputs = [1.5, 2.5, 2.5];
          // Act
          const result = Cost.mse(targets, outputs);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [1, 2, 3];
          const outputs = [1.5, 2.5, 2.5];
          // Act
          const result = Cost.mse(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: perfect predictions', () => {
      describe('when predictions are perfect', () => {
        it('returns 0', () => {
          // Arrange
          const targets = [5, -2];
          const outputs = [5, -2];
          // Act
          const result = Cost.mse(targets, outputs);
          // Assert
          expect(result).toBe(0);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [5, -2];
          const outputs = [5, -2];
          // Act
          const result = Cost.mse(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: single element arrays', () => {
      describe('when calculating mean squared error for a single element', () => {
        it('returns the correct value', () => {
          // Arrange
          const targets = [10];
          const outputs = [8];
          const expected = Math.pow(2, 2) / 1;
          // Act
          const result = Cost.mse(targets, outputs);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        it('returns a value greater than or equal to 0', () => {
          // Arrange
          const targets = [10];
          const outputs = [8];
          // Act
          const result = Cost.mse(targets, outputs);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [10];
          const outputs = [8];
          // Act
          const result = Cost.mse(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: mismatched input lengths', () => {
      it('throws error', () => {
        // Arrange
        const targets = [1];
        const outputs = [1, 2];
        // Suppress console output
        const originalError = console.error;
        console.error = () => {};
        // Act & Assert
        expect(() => Cost.mse(targets, outputs)).toThrow();
        console.error = originalError;
      });
    });
  });

  describe('binary()', () => {
    describe('Scenario: general case', () => {
      describe('when calculating binary error rate', () => {
        it('returns the correct value', () => {
          // Arrange
          const targets = [0, 1, 0, 1];
          const outputs = [0.2, 0.7, 0.6, 0.3];
          const expected = 2 / 4;
          // Act
          const result = Cost.binary(targets, outputs);
          // Assert
          expect(result).toBe(expected);
        });
        it('returns a value greater than or equal to 0', () => {
          // Arrange
          const targets = [0, 1, 0, 1];
          const outputs = [0.2, 0.7, 0.6, 0.3];
          // Act
          const result = Cost.binary(targets, outputs);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        it('returns a value less than or equal to 1', () => {
          // Arrange
          const targets = [0, 1, 0, 1];
          const outputs = [0.2, 0.7, 0.6, 0.3];
          // Act
          const result = Cost.binary(targets, outputs);
          // Assert
          expect(result).toBeLessThanOrEqual(1);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [0, 1, 0, 1];
          const outputs = [0.2, 0.7, 0.6, 0.3];
          // Act
          const result = Cost.binary(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: perfect predictions', () => {
      describe('when predictions are perfect', () => {
        it('returns 0', () => {
          // Arrange
          const targets = [0, 1, 1, 0];
          const outputs = [0.1, 0.9, 0.6, 0.4];
          // Act
          const result = Cost.binary(targets, outputs);
          // Assert
          expect(result).toBe(0);
        });
        it('returns a value less than or equal to 1', () => {
          // Arrange
          const targets = [0, 1, 1, 0];
          const outputs = [0.1, 0.9, 0.6, 0.4];
          // Act
          const result = Cost.binary(targets, outputs);
          // Assert
          expect(result).toBeLessThanOrEqual(1);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [0, 1, 1, 0];
          const outputs = [0.1, 0.9, 0.6, 0.4];
          // Act
          const result = Cost.binary(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: completely wrong predictions', () => {
      describe('when predictions are completely wrong', () => {
        it('returns 1', () => {
          // Arrange
          const targets = [0, 1];
          const outputs = [0.8, 0.2];
          // Act
          const result = Cost.binary(targets, outputs);
          // Assert
          expect(result).toBe(1);
        });
        it('returns a value greater than or equal to 0', () => {
          // Arrange
          const targets = [0, 1];
          const outputs = [0.8, 0.2];
          // Act
          const result = Cost.binary(targets, outputs);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [0, 1];
          const outputs = [0.8, 0.2];
          // Act
          const result = Cost.binary(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: single element arrays (correct)', () => {
      describe('when single element is correct', () => {
        it('returns 0', () => {
          // Arrange
          const targets = [1];
          const outputs = [0.9];
          // Act
          const result = Cost.binary(targets, outputs);
          // Assert
          expect(result).toBe(0);
        });
        it('returns a value less than or equal to 1', () => {
          // Arrange
          const targets = [1];
          const outputs = [0.9];
          // Act
          const result = Cost.binary(targets, outputs);
          // Assert
          expect(result).toBeLessThanOrEqual(1);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [1];
          const outputs = [0.9];
          // Act
          const result = Cost.binary(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: single element arrays (incorrect)', () => {
      describe('when single element is incorrect', () => {
        it('returns 1', () => {
          // Arrange
          const targets = [1];
          const outputs = [0.1];
          // Act
          const result = Cost.binary(targets, outputs);
          // Assert
          expect(result).toBe(1);
        });
        it('returns a value greater than or equal to 0', () => {
          // Arrange
          const targets = [1];
          const outputs = [0.1];
          // Act
          const result = Cost.binary(targets, outputs);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [1];
          const outputs = [0.1];
          // Act
          const result = Cost.binary(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: mismatched input lengths', () => {
      it('throws error', () => {
        // Arrange
        const targets = [1];
        const outputs = [1, 0];
        // Suppress console output
        const originalError = console.error;
        console.error = () => {};
        // Act & Assert
        expect(() => Cost.binary(targets, outputs)).toThrow();
        console.error = originalError;
      });
    });
  });

  describe('mae()', () => {
    describe('Scenario: general case', () => {
      describe('when calculating mean absolute error', () => {
        it('returns the correct value', () => {
          // Arrange
          const targets = [1, 2, 3];
          const outputs = [1.5, 2.5, 2.5];
          const expected = (
            Math.abs(targets[0] - outputs[0]) +
            Math.abs(targets[1] - outputs[1]) +
            Math.abs(targets[2] - outputs[2])
          ) / 3;
          // Act
          const result = Cost.mae(targets, outputs);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        it('returns a value greater than or equal to 0', () => {
          // Arrange
          const targets = [1, 2, 3];
          const outputs = [1.5, 2.5, 2.5];
          // Act
          const result = Cost.mae(targets, outputs);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [1, 2, 3];
          const outputs = [1.5, 2.5, 2.5];
          // Act
          const result = Cost.mae(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: perfect predictions', () => {
      describe('when predictions are perfect', () => {
        it('returns 0', () => {
          // Arrange
          const targets = [5, -2];
          const outputs = [5, -2];
          // Act
          const result = Cost.mae(targets, outputs);
          // Assert
          expect(result).toBe(0);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [5, -2];
          const outputs = [5, -2];
          // Act
          const result = Cost.mae(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: single element arrays', () => {
      describe('when calculating mean absolute error for a single element', () => {
        it('returns the correct value', () => {
          // Arrange
          const targets = [10];
          const outputs = [8];
          const expected = Math.abs(2) / 1;
          // Act
          const result = Cost.mae(targets, outputs);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        it('returns a value greater than or equal to 0', () => {
          // Arrange
          const targets = [10];
          const outputs = [8];
          // Act
          const result = Cost.mae(targets, outputs);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [10];
          const outputs = [8];
          // Act
          const result = Cost.mae(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: mismatched input lengths', () => {
      it('throws error', () => {
        // Arrange
        const targets = [1];
        const outputs = [1, 2];
        // Suppress console output
        const originalError = console.error;
        console.error = () => {};
        // Act & Assert
        expect(() => Cost.mae(targets, outputs)).toThrow();
        console.error = originalError;
      });
    });
  });

  describe('mape()', () => {
    describe('Scenario: general case', () => {
      describe('when calculating mean absolute percentage error', () => {
        it('returns the correct value', () => {
          // Arrange
          const targets = [100, 200];
          const outputs = [110, 180];
          const expected = (Math.abs(10 / 100) + Math.abs(-20 / 200)) / 2;
          // Act
          const result = Cost.mape(targets, outputs);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        it('returns a value greater than or equal to 0', () => {
          // Arrange
          const targets = [100, 200];
          const outputs = [110, 180];
          // Act
          const result = Cost.mape(targets, outputs);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [100, 200];
          const outputs = [110, 180];
          // Act
          const result = Cost.mape(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: perfect predictions', () => {
      describe('when predictions are perfect', () => {
        it('returns 0', () => {
          // Arrange
          const targets = [50, -20];
          const outputs = [50, -20];
          // Act
          const result = Cost.mape(targets, outputs);
          // Assert
          expect(result).toBe(0);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [50, -20];
          const outputs = [50, -20];
          // Act
          const result = Cost.mape(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: near-zero targets using epsilon', () => {
      describe('when targets are near zero', () => {
        it('returns a value greater than or equal to 0', () => {
          // Arrange
          const targets = [1e-16, 100];
          const outputs = [1, 110];
          // Act
          const result = Cost.mape(targets, outputs);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [1e-16, 100];
          const outputs = [1, 110];
          // Act
          const result = Cost.mape(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: single element arrays', () => {
      describe('when calculating mean absolute percentage error for a single element', () => {
        it('returns the correct value', () => {
          // Arrange
          const targets = [50];
          const outputs = [40];
          const expected = Math.abs(-10 / 50) / 1;
          // Act
          const result = Cost.mape(targets, outputs);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        it('returns a value greater than or equal to 0', () => {
          // Arrange
          const targets = [50];
          const outputs = [40];
          // Act
          const result = Cost.mape(targets, outputs);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [50];
          const outputs = [40];
          // Act
          const result = Cost.mape(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: mismatched input lengths', () => {
      it('throws error', () => {
        // Arrange
        const targets = [1];
        const outputs = [1, 2];
        // Suppress console output
        const originalError = console.error;
        console.error = () => {};
        // Act & Assert
        expect(() => Cost.mape(targets, outputs)).toThrow();
        console.error = originalError;
      });
    });
  });

  describe('msle()', () => {
    describe('Scenario: general case', () => {
      describe('when calculating mean squared logarithmic error', () => {
        it('returns the correct value', () => {
          // Arrange
          const targets = [1, 2, Math.E - 1];
          const outputs = [1, 1, 0];
          const logTarget1 = Math.log(1 + 1);
          const logOutput1 = Math.log(1 + 1);
          const logTarget2 = Math.log(2 + 1);
          const logOutput2 = Math.log(1 + 1);
          const logTarget3 = Math.log(Math.E - 1 + 1);
          const logOutput3 = Math.log(0 + 1);
          const expected =
            (Math.pow(logTarget1 - logOutput1, 2) +
              Math.pow(logTarget2 - logOutput2, 2) +
              Math.pow(logTarget3 - logOutput3, 2)) /
            3;
          // Act
          const result = Cost.msle(targets, outputs);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        it('returns a value greater than or equal to 0', () => {
          // Arrange
          const targets = [1, 2, Math.E - 1];
          const outputs = [1, 1, 0];
          // Act
          const result = Cost.msle(targets, outputs);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [1, 2, Math.E - 1];
          const outputs = [1, 1, 0];
          // Act
          const result = Cost.msle(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: perfect predictions', () => {
      describe('when predictions are perfect', () => {
        it('returns 0', () => {
          // Arrange
          const targets = [5, 0, 10];
          const outputs = [5, 0, 10];
          // Act
          const result = Cost.msle(targets, outputs);
          // Assert
          expect(result).toBe(0);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [5, 0, 10];
          const outputs = [5, 0, 10];
          // Act
          const result = Cost.msle(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: negative inputs', () => {
      describe('when inputs are negative', () => {
        it('returns the correct value', () => {
          // Arrange
          const targets = [-1, 2];
          const outputs = [1, -2];
          const logTarget1 = Math.log(0 + 1);
          const logOutput1 = Math.log(1 + 1);
          const logTarget2 = Math.log(2 + 1);
          const logOutput2 = Math.log(0 + 1);
          const expected =
            (Math.pow(logTarget1 - logOutput1, 2) +
              Math.pow(logTarget2 - logOutput2, 2)) /
            2;
          // Act
          const result = Cost.msle(targets, outputs);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        it('returns a value greater than or equal to 0', () => {
          // Arrange
          const targets = [-1, 2];
          const outputs = [1, -2];
          // Act
          const result = Cost.msle(targets, outputs);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [-1, 2];
          const outputs = [1, -2];
          // Act
          const result = Cost.msle(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: single element arrays', () => {
      describe('when calculating mean squared logarithmic error for a single element', () => {
        it('returns the correct value', () => {
          // Arrange
          const targets = [Math.E - 1];
          const outputs = [0];
          const expected = Math.pow(1 - 0, 2) / 1;
          // Act
          const result = Cost.msle(targets, outputs);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        it('returns a value greater than or equal to 0', () => {
          // Arrange
          const targets = [Math.E - 1];
          const outputs = [0];
          // Act
          const result = Cost.msle(targets, outputs);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [Math.E - 1];
          const outputs = [0];
          // Act
          const result = Cost.msle(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: mismatched input lengths', () => {
      it('throws error', () => {
        // Arrange
        const targets = [1];
        const outputs = [1, 2];
        // Suppress console output
        const originalError = console.error;
        console.error = () => {};
        // Act & Assert
        expect(() => Cost.msle(targets, outputs)).toThrow();
        console.error = originalError;
      });
    });
  });

  describe('hinge()', () => {
    describe('Scenario: general case', () => {
      describe('when calculating hinge loss', () => {
        it('returns the correct value', () => {
          // Arrange
          const targets = [-1, 1, -1, 1];
          const outputs = [-0.5, 0.8, 1.2, -0.1];
          const expected = (0.5 + 0.2 + 2.2 + 1.1) / 4;
          // Act
          const result = Cost.hinge(targets, outputs);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        it('returns a value greater than or equal to 0', () => {
          // Arrange
          const targets = [-1, 1, -1, 1];
          const outputs = [-0.5, 0.8, 1.2, -0.1];
          // Act
          const result = Cost.hinge(targets, outputs);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [-1, 1, -1, 1];
          const outputs = [-0.5, 0.8, 1.2, -0.1];
          // Act
          const result = Cost.hinge(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: correct and confident predictions', () => {
      describe('when predictions are correct and confident', () => {
        it('returns 0', () => {
          // Arrange
          const targets = [-1, 1];
          const outputs = [-1.5, 1.1];
          // Act
          const result = Cost.hinge(targets, outputs);
          // Assert
          expect(result).toBe(0);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [-1, 1];
          const outputs = [-1.5, 1.1];
          // Act
          const result = Cost.hinge(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: correct but not confident predictions', () => {
      describe('when predictions are correct but not confident', () => {
        it('returns the correct value', () => {
          // Arrange
          const targets = [-1, 1];
          const outputs = [-0.5, 0.5];
          const expected = (0.5 + 0.5) / 2;
          // Act
          const result = Cost.hinge(targets, outputs);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        it('returns a value greater than or equal to 0', () => {
          // Arrange
          const targets = [-1, 1];
          const outputs = [-0.5, 0.5];
          // Act
          const result = Cost.hinge(targets, outputs);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [-1, 1];
          const outputs = [-0.5, 0.5];
          // Act
          const result = Cost.hinge(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: single element arrays', () => {
      describe('when calculating hinge loss for a single element', () => {
        it('returns the correct value', () => {
          // Arrange
          const targets = [-1];
          const outputs = [0.2];
          const expected = 1.2 / 1;
          // Act
          const result = Cost.hinge(targets, outputs);
          // Assert
          expect(result).toBeCloseTo(expected, epsilon);
        });
        it('returns a value greater than or equal to 0', () => {
          // Arrange
          const targets = [-1];
          const outputs = [0.2];
          // Act
          const result = Cost.hinge(targets, outputs);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [-1];
          const outputs = [0.2];
          // Act
          const result = Cost.hinge(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: mismatched input lengths', () => {
      it('throws error', () => {
        // Arrange
        const targets = [1];
        const outputs = [1, 2];
        // Suppress console output
        const originalError = console.error;
        console.error = () => {};
        // Act & Assert
        expect(() => Cost.hinge(targets, outputs)).toThrow();
        console.error = originalError;
      });
    });
  });

  describe('focalLoss()', () => {
    describe('Scenario: binary targets, default gamma/alpha', () => {
      describe('when calculating focal loss', () => {
        it('returns a value greater than or equal to 0', () => {
          // Arrange
          const targets = [1, 0];
          const outputs = [0.9, 0.1];
          // Act
          const result = Cost.focalLoss(targets, outputs);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [1, 0];
          const outputs = [0.9, 0.1];
          // Act
          const result = Cost.focalLoss(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: perfect predictions', () => {
      describe('when predictions are perfect', () => {
        it('returns near zero', () => {
          // Arrange
          const targets = [1, 0];
          const outputs = [1, 0];
          // Act
          const result = Cost.focalLoss(targets, outputs);
          // Assert
          expect(result).toBeCloseTo(0, 5);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [1, 0];
          const outputs = [1, 0];
          // Act
          const result = Cost.focalLoss(targets, outputs);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: mismatched input lengths', () => {
      it('throws error', () => {
        // Arrange
        const targets = [1];
        const outputs = [0.5, 0.5];
        // Suppress console output
        const originalError = console.error;
        console.error = () => {};
        // Act & Assert
        expect(() => Cost.focalLoss(targets, outputs)).toThrow('Target and output arrays must have the same length.');
        console.error = originalError;
      });
    });
  });

  describe('labelSmoothing()', () => {
    describe('Scenario: binary targets, smoothing=0.1', () => {
      describe('when calculating label smoothing loss', () => {
        it('returns a value greater than or equal to 0', () => {
          // Arrange
          const targets = [1, 0];
          const outputs = [0.9, 0.1];
          // Act
          const result = Cost.labelSmoothing(targets, outputs, 0.1);
          // Assert
          expect(result).toBeGreaterThanOrEqual(0);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [1, 0];
          const outputs = [0.9, 0.1];
          // Act
          const result = Cost.labelSmoothing(targets, outputs, 0.1);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: perfect predictions, smoothing=0', () => {
      describe('when predictions are perfect', () => {
        it('returns near zero', () => {
          // Arrange
          const targets = [1, 0];
          const outputs = [1, 0];
          // Act
          const result = Cost.labelSmoothing(targets, outputs, 0);
          // Assert
          expect(result).toBeCloseTo(0, 5);
        });
        it('returns a finite value', () => {
          // Arrange
          const targets = [1, 0];
          const outputs = [1, 0];
          // Act
          const result = Cost.labelSmoothing(targets, outputs, 0);
          // Assert
          expect(isFinite(result)).toBe(true);
        });
      });
    });
    describe('Scenario: when smoothing=1 forces uniform 0.5 target', () => {
      it('returns finite loss', () => {
        const targets = [1,0];
        const outputs = [0.9,0.1];
        const result = Cost.labelSmoothing(targets, outputs, 1);
        expect(Number.isFinite(result)).toBe(true);
      });
    });
    describe('Scenario: mismatched input lengths', () => {
      it('throws error', () => {
        // Arrange
        const targets = [1];
        const outputs = [0.5, 0.5];
        // Suppress console output
        const originalError = console.error;
        console.error = () => {};
        // Act & Assert
        expect(() => Cost.labelSmoothing(targets, outputs)).toThrow('Target and output arrays must have the same length.');
        console.error = originalError;
      });
    });
  });
});
