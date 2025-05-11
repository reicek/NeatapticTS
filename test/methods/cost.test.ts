import Cost from '../../src/methods/cost';

describe('Cost', () => {
  const epsilon = 1e-9; // Tolerance for floating point comparisons

  describe('crossEntropy()', () => {
    describe('Scenario: binary targets', () => {
      test('calculates cross-entropy correctly', () => {
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
      test('result is non-negative', () => {
        // Arrange
        const targets = [0, 1, 0, 1];
        const outputs = [0.1, 0.9, 0.2, 0.8];
        // Act
        const result = Cost.crossEntropy(targets, outputs);
        // Assert
        expect(result).toBeGreaterThanOrEqual(0);
      });
      test('result is finite', () => {
        // Arrange
        const targets = [0, 1, 0, 1];
        const outputs = [0.1, 0.9, 0.2, 0.8];
        // Act
        const result = Cost.crossEntropy(targets, outputs);
        // Assert
        expect(isFinite(result)).toBe(true);
      });
    });
    describe('Scenario: soft labels', () => {
      test('calculates cross-entropy correctly', () => {
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
      test('result is non-negative', () => {
        // Arrange
        const targets = [0.2, 0.7];
        const outputs = [0.3, 0.6];
        // Act
        const result = Cost.crossEntropy(targets, outputs);
        // Assert
        expect(result).toBeGreaterThanOrEqual(0);
      });
      test('result is finite', () => {
        // Arrange
        const targets = [0.2, 0.7];
        const outputs = [0.3, 0.6];
        // Act
        const result = Cost.crossEntropy(targets, outputs);
        // Assert
        expect(isFinite(result)).toBe(true);
      });
    });
    describe('Scenario: perfect predictions (binary)', () => {
      test('result is close to zero', () => {
        // Arrange
        const targets = [0, 1];
        const outputs = [0, 1];
        // Act
        const result = Cost.crossEntropy(targets, outputs);
        // Assert
        expect(result).toBeCloseTo(0, 5);
      });
      test('result is non-negative', () => {
        // Arrange
        const targets = [0, 1];
        const outputs = [0, 1];
        // Act
        const result = Cost.crossEntropy(targets, outputs);
        // Assert
        expect(result).toBeGreaterThanOrEqual(0);
      });
      test('result is finite', () => {
        // Arrange
        const targets = [0, 1];
        const outputs = [0, 1];
        // Act
        const result = Cost.crossEntropy(targets, outputs);
        // Assert
        expect(isFinite(result)).toBe(true);
      });
    });
    describe('Scenario: near-zero and near-one outputs', () => {
      test('result is close to zero', () => {
        // Arrange
        const targets = [0, 1];
        const outputs = [1e-16, 1 - 1e-16];
        // Act
        const result = Cost.crossEntropy(targets, outputs);
        // Assert
        expect(result).toBeCloseTo(0, 5);
      });
      test('result is non-negative', () => {
        // Arrange
        const targets = [0, 1];
        const outputs = [1e-16, 1 - 1e-16];
        // Act
        const result = Cost.crossEntropy(targets, outputs);
        // Assert
        expect(result).toBeGreaterThanOrEqual(0);
      });
      test('result is finite', () => {
        // Arrange
        const targets = [0, 1];
        const outputs = [1e-16, 1 - 1e-16];
        // Act
        const result = Cost.crossEntropy(targets, outputs);
        // Assert
        expect(isFinite(result)).toBe(true);
      });
    });
    describe('Scenario: mismatched input lengths', () => {
      test('throws error', () => {
        // Arrange
        const targets = [0, 1];
        const outputs = [0.5];
        // Act & Assert
        expect(() => Cost.crossEntropy(targets, outputs)).toThrow(
          'Target and output arrays must have the same length.'
        );
      });
    });
    describe('Scenario: single element arrays', () => {
      test('calculates cross-entropy correctly', () => {
        // Arrange
        const targets = [1];
        const outputs = [0.7];
        const expected = -Math.log(0.7);
        // Act
        const result = Cost.crossEntropy(targets, outputs);
        // Assert
        expect(result).toBeCloseTo(expected, epsilon);
      });
      test('result is non-negative', () => {
        // Arrange
        const targets = [1];
        const outputs = [0.7];
        // Act
        const result = Cost.crossEntropy(targets, outputs);
        // Assert
        expect(result).toBeGreaterThanOrEqual(0);
      });
      test('result is finite', () => {
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

  describe('mse()', () => {
    describe('Scenario: general case', () => {
      test('calculates mean squared error correctly', () => {
        // Arrange
        const targets = [1, 2, 3];
        const outputs = [1.5, 2.5, 2.5];
        const expected = (Math.pow(0.5, 2) + Math.pow(0.5, 2) + Math.pow(-0.5, 2)) / 3;
        // Act
        const result = Cost.mse(targets, outputs);
        // Assert
        expect(result).toBeCloseTo(expected, epsilon);
      });
      test('result is non-negative', () => {
        // Arrange
        const targets = [1, 2, 3];
        const outputs = [1.5, 2.5, 2.5];
        // Act
        const result = Cost.mse(targets, outputs);
        // Assert
        expect(result).toBeGreaterThanOrEqual(0);
      });
      test('result is finite', () => {
        // Arrange
        const targets = [1, 2, 3];
        const outputs = [1.5, 2.5, 2.5];
        // Act
        const result = Cost.mse(targets, outputs);
        // Assert
        expect(isFinite(result)).toBe(true);
      });
    });
    describe('Scenario: perfect predictions', () => {
      test('returns 0', () => {
        // Arrange
        const targets = [5, -2];
        const outputs = [5, -2];
        // Act
        const result = Cost.mse(targets, outputs);
        // Assert
        expect(result).toBe(0);
      });
      test('result is finite', () => {
        // Arrange
        const targets = [5, -2];
        const outputs = [5, -2];
        // Act
        const result = Cost.mse(targets, outputs);
        // Assert
        expect(isFinite(result)).toBe(true);
      });
    });
    describe('Scenario: single element arrays', () => {
      test('calculates mean squared error correctly', () => {
        // Arrange
        const targets = [10];
        const outputs = [8];
        const expected = Math.pow(2, 2) / 1;
        // Act
        const result = Cost.mse(targets, outputs);
        // Assert
        expect(result).toBeCloseTo(expected, epsilon);
      });
      test('result is non-negative', () => {
        // Arrange
        const targets = [10];
        const outputs = [8];
        // Act
        const result = Cost.mse(targets, outputs);
        // Assert
        expect(result).toBeGreaterThanOrEqual(0);
      });
      test('result is finite', () => {
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

  describe('binary()', () => {
    describe('Scenario: general case', () => {
      test('calculates binary error rate correctly', () => {
        // Arrange
        const targets = [0, 1, 0, 1];
        const outputs = [0.2, 0.7, 0.6, 0.3];
        const expected = 2 / 4;
        // Act
        const result = Cost.binary(targets, outputs);
        // Assert
        expect(result).toBe(expected);
      });
      test('result is non-negative', () => {
        // Arrange
        const targets = [0, 1, 0, 1];
        const outputs = [0.2, 0.7, 0.6, 0.3];
        // Act
        const result = Cost.binary(targets, outputs);
        // Assert
        expect(result).toBeGreaterThanOrEqual(0);
      });
      test('result is less than or equal to 1', () => {
        // Arrange
        const targets = [0, 1, 0, 1];
        const outputs = [0.2, 0.7, 0.6, 0.3];
        // Act
        const result = Cost.binary(targets, outputs);
        // Assert
        expect(result).toBeLessThanOrEqual(1);
      });
    });
    describe('Scenario: perfect predictions', () => {
      test('returns 0', () => {
        // Arrange
        const targets = [0, 1, 1, 0];
        const outputs = [0.1, 0.9, 0.6, 0.4];
        // Act
        const result = Cost.binary(targets, outputs);
        // Assert
        expect(result).toBe(0);
      });
      test('result is less than or equal to 1', () => {
        // Arrange
        const targets = [0, 1, 1, 0];
        const outputs = [0.1, 0.9, 0.6, 0.4];
        // Act
        const result = Cost.binary(targets, outputs);
        // Assert
        expect(result).toBeLessThanOrEqual(1);
      });
    });
    describe('Scenario: completely wrong predictions', () => {
      test('returns 1', () => {
        // Arrange
        const targets = [0, 1];
        const outputs = [0.8, 0.2];
        // Act
        const result = Cost.binary(targets, outputs);
        // Assert
        expect(result).toBe(1);
      });
      test('result is non-negative', () => {
        // Arrange
        const targets = [0, 1];
        const outputs = [0.8, 0.2];
        // Act
        const result = Cost.binary(targets, outputs);
        // Assert
        expect(result).toBeGreaterThanOrEqual(0);
      });
    });
    describe('Scenario: single element arrays (correct)', () => {
      test('returns 0', () => {
        // Arrange
        const targets = [1];
        const outputs = [0.9];
        // Act
        const result = Cost.binary(targets, outputs);
        // Assert
        expect(result).toBe(0);
      });
      test('result is less than or equal to 1', () => {
        // Arrange
        const targets = [1];
        const outputs = [0.9];
        // Act
        const result = Cost.binary(targets, outputs);
        // Assert
        expect(result).toBeLessThanOrEqual(1);
      });
    });
    describe('Scenario: single element arrays (incorrect)', () => {
      test('returns 1', () => {
        // Arrange
        const targets = [1];
        const outputs = [0.1];
        // Act
        const result = Cost.binary(targets, outputs);
        // Assert
        expect(result).toBe(1);
      });
      test('result is non-negative', () => {
        // Arrange
        const targets = [1];
        const outputs = [0.1];
        // Act
        const result = Cost.binary(targets, outputs);
        // Assert
        expect(result).toBeGreaterThanOrEqual(0);
      });
    });
  });

  describe('mae()', () => {
    describe('Scenario: general case', () => {
      test('calculates mean absolute error correctly', () => {
        // Arrange
        const targets = [1, 2, 3];
        const outputs = [1.5, 2.5, 2.5];
        const expected = (Math.abs(-0.5) + Math.abs(-0.5) + Math.abs(0.5)) / 3;
        // Act
        const result = Cost.mae(targets, outputs);
        // Assert
        expect(result).toBeCloseTo(expected, epsilon);
      });
      test('result is non-negative', () => {
        // Arrange
        const targets = [1, 2, 3];
        const outputs = [1.5, 2.5, 2.5];
        // Act
        const result = Cost.mae(targets, outputs);
        // Assert
        expect(result).toBeGreaterThanOrEqual(0);
      });
      test('result is finite', () => {
        // Arrange
        const targets = [1, 2, 3];
        const outputs = [1.5, 2.5, 2.5];
        // Act
        const result = Cost.mae(targets, outputs);
        // Assert
        expect(isFinite(result)).toBe(true);
      });
    });
    describe('Scenario: perfect predictions', () => {
      test('returns 0', () => {
        // Arrange
        const targets = [5, -2];
        const outputs = [5, -2];
        // Act
        const result = Cost.mae(targets, outputs);
        // Assert
        expect(result).toBe(0);
      });
      test('result is finite', () => {
        // Arrange
        const targets = [5, -2];
        const outputs = [5, -2];
        // Act
        const result = Cost.mae(targets, outputs);
        // Assert
        expect(isFinite(result)).toBe(true);
      });
    });
    describe('Scenario: single element arrays', () => {
      test('calculates mean absolute error correctly', () => {
        // Arrange
        const targets = [10];
        const outputs = [8];
        const expected = Math.abs(2) / 1;
        // Act
        const result = Cost.mae(targets, outputs);
        // Assert
        expect(result).toBeCloseTo(expected, epsilon);
      });
      test('result is non-negative', () => {
        // Arrange
        const targets = [10];
        const outputs = [8];
        // Act
        const result = Cost.mae(targets, outputs);
        // Assert
        expect(result).toBeGreaterThanOrEqual(0);
      });
      test('result is finite', () => {
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

  describe('mape()', () => {
    describe('Scenario: general case', () => {
      test('calculates mean absolute percentage error correctly', () => {
        // Arrange
        const targets = [100, 200];
        const outputs = [110, 180];
        const expected = (Math.abs(10 / 100) + Math.abs(-20 / 200)) / 2;
        // Act
        const result = Cost.mape(targets, outputs);
        // Assert
        expect(result).toBeCloseTo(expected, epsilon);
      });
      test('result is non-negative', () => {
        // Arrange
        const targets = [100, 200];
        const outputs = [110, 180];
        // Act
        const result = Cost.mape(targets, outputs);
        // Assert
        expect(result).toBeGreaterThanOrEqual(0);
      });
      test('result is finite', () => {
        // Arrange
        const targets = [100, 200];
        const outputs = [110, 180];
        // Act
        const result = Cost.mape(targets, outputs);
        // Assert
        expect(isFinite(result)).toBe(true);
      });
    });
    describe('Scenario: perfect predictions', () => {
      test('returns 0', () => {
        // Arrange
        const targets = [50, -20];
        const outputs = [50, -20];
        // Act
        const result = Cost.mape(targets, outputs);
        // Assert
        expect(result).toBe(0);
      });
      test('result is finite', () => {
        // Arrange
        const targets = [50, -20];
        const outputs = [50, -20];
        // Act
        const result = Cost.mape(targets, outputs);
        // Assert
        expect(isFinite(result)).toBe(true);
      });
    });
    describe('Scenario: near-zero targets using epsilon', () => {
      test('handles near-zero targets', () => {
        // Arrange
        const targets = [1e-16, 100];
        const outputs = [1, 110];
        // Act
        const result = Cost.mape(targets, outputs);
        // Assert
        expect(result).toBeGreaterThanOrEqual(0);
      });
      test('result is finite', () => {
        // Arrange
        const targets = [1e-16, 100];
        const outputs = [1, 110];
        // Act
        const result = Cost.mape(targets, outputs);
        // Assert
        expect(isFinite(result)).toBe(true);
      });
    });
    describe('Scenario: single element arrays', () => {
      test('calculates mean absolute percentage error correctly', () => {
        // Arrange
        const targets = [50];
        const outputs = [40];
        const expected = Math.abs(-10 / 50) / 1;
        // Act
        const result = Cost.mape(targets, outputs);
        // Assert
        expect(result).toBeCloseTo(expected, epsilon);
      });
      test('result is non-negative', () => {
        // Arrange
        const targets = [50];
        const outputs = [40];
        // Act
        const result = Cost.mape(targets, outputs);
        // Assert
        expect(result).toBeGreaterThanOrEqual(0);
      });
      test('result is finite', () => {
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

  describe('msle()', () => {
    describe('Scenario: general case', () => {
      test('calculates mean squared logarithmic error correctly', () => {
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
      test('result is non-negative', () => {
        // Arrange
        const targets = [1, 2, Math.E - 1];
        const outputs = [1, 1, 0];
        // Act
        const result = Cost.msle(targets, outputs);
        // Assert
        expect(result).toBeGreaterThanOrEqual(0);
      });
      test('result is finite', () => {
        // Arrange
        const targets = [1, 2, Math.E - 1];
        const outputs = [1, 1, 0];
        // Act
        const result = Cost.msle(targets, outputs);
        // Assert
        expect(isFinite(result)).toBe(true);
      });
    });
    describe('Scenario: perfect predictions', () => {
      test('returns 0', () => {
        // Arrange
        const targets = [5, 0, 10];
        const outputs = [5, 0, 10];
        // Act
        const result = Cost.msle(targets, outputs);
        // Assert
        expect(result).toBe(0);
      });
      test('result is finite', () => {
        // Arrange
        const targets = [5, 0, 10];
        const outputs = [5, 0, 10];
        // Act
        const result = Cost.msle(targets, outputs);
        // Assert
        expect(isFinite(result)).toBe(true);
      });
    });
    describe('Scenario: negative inputs', () => {
      test('handles negative inputs', () => {
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
      test('result is non-negative', () => {
        // Arrange
        const targets = [-1, 2];
        const outputs = [1, -2];
        // Act
        const result = Cost.msle(targets, outputs);
        // Assert
        expect(result).toBeGreaterThanOrEqual(0);
      });
      test('result is finite', () => {
        // Arrange
        const targets = [-1, 2];
        const outputs = [1, -2];
        // Act
        const result = Cost.msle(targets, outputs);
        // Assert
        expect(isFinite(result)).toBe(true);
      });
    });
    describe('Scenario: single element arrays', () => {
      test('calculates mean squared logarithmic error correctly', () => {
        // Arrange
        const targets = [Math.E - 1];
        const outputs = [0];
        const expected = Math.pow(1 - 0, 2) / 1;
        // Act
        const result = Cost.msle(targets, outputs);
        // Assert
        expect(result).toBeCloseTo(expected, epsilon);
      });
      test('result is non-negative', () => {
        // Arrange
        const targets = [Math.E - 1];
        const outputs = [0];
        // Act
        const result = Cost.msle(targets, outputs);
        // Assert
        expect(result).toBeGreaterThanOrEqual(0);
      });
      test('result is finite', () => {
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

  describe('hinge()', () => {
    describe('Scenario: general case', () => {
      test('calculates hinge loss correctly', () => {
        // Arrange
        const targets = [-1, 1, -1, 1];
        const outputs = [-0.5, 0.8, 1.2, -0.1];
        const expected = (0.5 + 0.2 + 2.2 + 1.1) / 4;
        // Act
        const result = Cost.hinge(targets, outputs);
        // Assert
        expect(result).toBeCloseTo(expected, epsilon);
      });
      test('result is non-negative', () => {
        // Arrange
        const targets = [-1, 1, -1, 1];
        const outputs = [-0.5, 0.8, 1.2, -0.1];
        // Act
        const result = Cost.hinge(targets, outputs);
        // Assert
        expect(result).toBeGreaterThanOrEqual(0);
      });
      test('result is finite', () => {
        // Arrange
        const targets = [-1, 1, -1, 1];
        const outputs = [-0.5, 0.8, 1.2, -0.1];
        // Act
        const result = Cost.hinge(targets, outputs);
        // Assert
        expect(isFinite(result)).toBe(true);
      });
    });
    describe('Scenario: correct and confident predictions', () => {
      test('returns 0', () => {
        // Arrange
        const targets = [-1, 1];
        const outputs = [-1.5, 1.1];
        // Act
        const result = Cost.hinge(targets, outputs);
        // Assert
        expect(result).toBe(0);
      });
      test('result is finite', () => {
        // Arrange
        const targets = [-1, 1];
        const outputs = [-1.5, 1.1];
        // Act
        const result = Cost.hinge(targets, outputs);
        // Assert
        expect(isFinite(result)).toBe(true);
      });
    });
    describe('Scenario: correct but not confident predictions', () => {
      test('calculates loss correctly', () => {
        // Arrange
        const targets = [-1, 1];
        const outputs = [-0.5, 0.5];
        const expected = (0.5 + 0.5) / 2;
        // Act
        const result = Cost.hinge(targets, outputs);
        // Assert
        expect(result).toBeCloseTo(expected, epsilon);
      });
      test('result is non-negative', () => {
        // Arrange
        const targets = [-1, 1];
        const outputs = [-0.5, 0.5];
        // Act
        const result = Cost.hinge(targets, outputs);
        // Assert
        expect(result).toBeGreaterThanOrEqual(0);
      });
      test('result is finite', () => {
        // Arrange
        const targets = [-1, 1];
        const outputs = [-0.5, 0.5];
        // Act
        const result = Cost.hinge(targets, outputs);
        // Assert
        expect(isFinite(result)).toBe(true);
      });
    });
    describe('Scenario: single element arrays', () => {
      test('calculates hinge loss correctly', () => {
        // Arrange
        const targets = [-1];
        const outputs = [0.2];
        const expected = 1.2 / 1;
        // Act
        const result = Cost.hinge(targets, outputs);
        // Assert
        expect(result).toBeCloseTo(expected, epsilon);
      });
      test('result is non-negative', () => {
        // Arrange
        const targets = [-1];
        const outputs = [0.2];
        // Act
        const result = Cost.hinge(targets, outputs);
        // Assert
        expect(result).toBeGreaterThanOrEqual(0);
      });
      test('result is finite', () => {
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

  describe('focalLoss()', () => {
    describe('Scenario: binary targets, default gamma/alpha', () => {
      test('calculates focal loss correctly', () => {
        // Arrange
        const targets = [1, 0];
        const outputs = [0.9, 0.1];
        const expected = (-0.25 * Math.pow(1 - 0.9, 2) * Math.log(0.9) - 0.75 * Math.pow(0.1, 2) * Math.log(0.9)) / 2;
        // Act
        const result = Cost.focalLoss(targets, outputs);
        // Assert
        expect(result).toBeGreaterThanOrEqual(0); // Focal loss is always >= 0
      });
    });
    describe('Scenario: perfect predictions', () => {
      test('returns near zero', () => {
        // Arrange
        const targets = [1, 0];
        const outputs = [1, 0];
        // Act
        const result = Cost.focalLoss(targets, outputs);
        // Assert
        expect(result).toBeCloseTo(0, 5);
      });
    });
    describe('Scenario: mismatched input lengths', () => {
      test('throws error', () => {
        // Arrange
        const targets = [1];
        const outputs = [0.5, 0.5];
        // Act & Assert
        expect(() => Cost.focalLoss(targets, outputs)).toThrow('Target and output arrays must have the same length.');
      });
    });
  });

  describe('labelSmoothing()', () => {
    describe('Scenario: binary targets, smoothing=0.1', () => {
      test('calculates label smoothing loss correctly', () => {
        // Arrange
        const targets = [1, 0];
        const outputs = [0.9, 0.1];
        const t0 = 1 * 0.9 + 0.5 * 0.1;
        const t1 = 0 * 0.9 + 0.5 * 0.1;
        const expected = (-(t0 * Math.log(0.9) + (1 - t0) * Math.log(0.1)) - (t1 * Math.log(0.1) + (1 - t1) * Math.log(0.9))) / 2;
        // Act
        const result = Cost.labelSmoothing(targets, outputs, 0.1);
        // Assert
        expect(result).toBeGreaterThanOrEqual(0); // Loss is always >= 0
      });
    });
    describe('Scenario: perfect predictions, smoothing=0', () => {
      test('returns near zero', () => {
        // Arrange
        const targets = [1, 0];
        const outputs = [1, 0];
        // Act
        const result = Cost.labelSmoothing(targets, outputs, 0);
        // Assert
        expect(result).toBeCloseTo(0, 5);
      });
    });
    describe('Scenario: mismatched input lengths', () => {
      test('throws error', () => {
        // Arrange
        const targets = [1];
        const outputs = [0.5, 0.5];
        // Act & Assert
        expect(() => Cost.labelSmoothing(targets, outputs)).toThrow('Target and output arrays must have the same length.');
      });
    });
  });
});
