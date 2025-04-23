import Cost from '../../src/methods/cost';

describe('Cost', () => {
  const epsilon = 1e-9; // Tolerance for floating point comparisons

  describe('crossEntropy()', () => {
    test('should calculate cross-entropy correctly for binary targets', () => {
      // Arrange
      const targets = [0, 1, 0, 1];
      const outputs = [0.1, 0.9, 0.2, 0.8];
      const expected =
        -(
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

    test('should calculate cross-entropy correctly for soft labels', () => {
      // Arrange
      const targets = [0.2, 0.7];
      const outputs = [0.3, 0.6];
      const expected =
        -(
          0.2 * Math.log(0.3) +
          (1 - 0.2) * Math.log(1 - 0.3) +
          (0.7 * Math.log(0.6) + (1 - 0.7) * Math.log(1 - 0.6))
        ) / 2;
      // Act
      const result = Cost.crossEntropy(targets, outputs);
      // Assert
      expect(result).toBeCloseTo(expected, epsilon);
    });

    test('should handle perfect predictions (binary)', () => {
      // Arrange
      const targets = [0, 1];
      const outputs = [0, 1]; // Outputs clamped internally
      // Act
      const result = Cost.crossEntropy(targets, outputs);
      // Assert: Should be close to zero due to clamping
      expect(result).toBeCloseTo(0, 5); // Use tolerance due to internal epsilon
    });

    test('should handle near-zero and near-one outputs', () => {
      // Arrange
      const targets = [0, 1];
      const outputs = [1e-16, 1 - 1e-16]; // Values close to boundaries
      // Act
      const result = Cost.crossEntropy(targets, outputs);
      // Assert: Should be close to zero
      expect(result).toBeCloseTo(0, 5);
    });

    test('should throw error for mismatched input lengths', () => {
      // Arrange
      const targets = [0, 1];
      const outputs = [0.5];
      // Act & Assert
      expect(() => Cost.crossEntropy(targets, outputs)).toThrow(
        'Target and output arrays must have the same length.'
      );
    });

    test('should handle single element arrays', () => {
      // Arrange
      const targets = [1];
      const outputs = [0.7];
      const expected = -Math.log(0.7);
      // Act
      const result = Cost.crossEntropy(targets, outputs);
      // Assert
      expect(result).toBeCloseTo(expected, epsilon);
    });
  });

  describe('mse()', () => {
    test('should calculate mean squared error correctly', () => {
      // Arrange
      const targets = [1, 2, 3];
      const outputs = [1.5, 2.5, 2.5];
      const expected = (Math.pow(0.5, 2) + Math.pow(0.5, 2) + Math.pow(-0.5, 2)) / 3;
      // Act
      const result = Cost.mse(targets, outputs);
      // Assert
      expect(result).toBeCloseTo(expected, epsilon);
    });

    test('should return 0 for perfect predictions', () => {
      // Arrange
      const targets = [5, -2];
      const outputs = [5, -2];
      // Act
      const result = Cost.mse(targets, outputs);
      // Assert
      expect(result).toBe(0);
    });

    test('should handle single element arrays', () => {
      // Arrange
      const targets = [10];
      const outputs = [8];
      const expected = Math.pow(2, 2) / 1;
      // Act
      const result = Cost.mse(targets, outputs);
      // Assert
      expect(result).toBeCloseTo(expected, epsilon);
    });
  });

  describe('binary()', () => {
    test('should calculate binary error rate correctly', () => {
      // Arrange
      const targets = [0, 1, 0, 1];
      const outputs = [0.2, 0.7, 0.6, 0.3]; // Misses at index 2 and 3
      const expected = 2 / 4;
      // Act
      const result = Cost.binary(targets, outputs);
      // Assert
      expect(result).toBe(expected);
    });

    test('should return 0 for perfect predictions', () => {
      // Arrange
      const targets = [0, 1, 1, 0];
      const outputs = [0.1, 0.9, 0.6, 0.4]; // All rounded correctly
      // Act
      const result = Cost.binary(targets, outputs);
      // Assert
      expect(result).toBe(0);
    });

    test('should return 1 for completely wrong predictions', () => {
      // Arrange
      const targets = [0, 1];
      const outputs = [0.8, 0.2]; // Both rounded incorrectly
      // Act
      const result = Cost.binary(targets, outputs);
      // Assert
      expect(result).toBe(1);
    });

    test('should handle single element arrays (correct)', () => {
      // Arrange
      const targets = [1];
      const outputs = [0.9];
      // Act
      const result = Cost.binary(targets, outputs);
      // Assert
      expect(result).toBe(0);
    });

    test('should handle single element arrays (incorrect)', () => {
      // Arrange
      const targets = [1];
      const outputs = [0.1];
      // Act
      const result = Cost.binary(targets, outputs);
      // Assert
      expect(result).toBe(1);
    });
  });

  describe('mae()', () => {
    test('should calculate mean absolute error correctly', () => {
      // Arrange
      const targets = [1, 2, 3];
      const outputs = [1.5, 2.5, 2.5];
      const expected = (Math.abs(-0.5) + Math.abs(-0.5) + Math.abs(0.5)) / 3;
      // Act
      const result = Cost.mae(targets, outputs);
      // Assert
      expect(result).toBeCloseTo(expected, epsilon);
    });

    test('should return 0 for perfect predictions', () => {
      // Arrange
      const targets = [5, -2];
      const outputs = [5, -2];
      // Act
      const result = Cost.mae(targets, outputs);
      // Assert
      expect(result).toBe(0);
    });

    test('should handle single element arrays', () => {
      // Arrange
      const targets = [10];
      const outputs = [8];
      const expected = Math.abs(2) / 1;
      // Act
      const result = Cost.mae(targets, outputs);
      // Assert
      expect(result).toBeCloseTo(expected, epsilon);
    });
  });

  describe('mape()', () => {
    test('should calculate mean absolute percentage error correctly', () => {
      // Arrange
      const targets = [100, 200];
      const outputs = [110, 180]; // Errors: 10, -20
      const expected = (Math.abs(10 / 100) + Math.abs(-20 / 200)) / 2; // (0.1 + 0.1) / 2 = 0.1
      // Act
      const result = Cost.mape(targets, outputs);
      // Assert
      expect(result).toBeCloseTo(expected, epsilon);
    });

    test('should return 0 for perfect predictions', () => {
      // Arrange
      const targets = [50, -20]; // Note: MAPE behaves differently with negative targets
      const outputs = [50, -20];
      // Act
      const result = Cost.mape(targets, outputs);
      // Assert
      expect(result).toBe(0);
    });

    test('should handle near-zero targets using epsilon', () => {
      // Arrange
      const targets = [1e-16, 100];
      const outputs = [1, 110];
      // Act: Division by near-zero target is avoided by internal epsilon
      const result = Cost.mape(targets, outputs);
      // Assert: Expect a large but finite number for the first term, averaged with 0.1
      expect(result).toBeGreaterThan(0);
      expect(isFinite(result)).toBe(true);
    });

    test('should handle single element arrays', () => {
      // Arrange
      const targets = [50];
      const outputs = [40];
      const expected = Math.abs(-10 / 50) / 1; // 0.2
      // Act
      const result = Cost.mape(targets, outputs);
      // Assert
      expect(result).toBeCloseTo(expected, epsilon);
    });
  });

  describe('msle()', () => {
    test('should calculate mean squared logarithmic error correctly', () => {
      // Arrange
      const targets = [1, 2, Math.E - 1]; // Targets >= 0
      const outputs = [1, 1, 0]; // Outputs >= 0
      const logTarget1 = Math.log(1 + 1);
      const logOutput1 = Math.log(1 + 1);
      const logTarget2 = Math.log(2 + 1);
      const logOutput2 = Math.log(1 + 1);
      const logTarget3 = Math.log(Math.E - 1 + 1); // log(e) = 1
      const logOutput3 = Math.log(0 + 1); // log(1) = 0
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

    test('should return 0 for perfect predictions', () => {
      // Arrange
      const targets = [5, 0, 10];
      const outputs = [5, 0, 10];
      // Act
      const result = Cost.msle(targets, outputs);
      // Assert
      expect(result).toBe(0);
    });

    test('should handle negative inputs by treating them as 0', () => {
      // Arrange
      const targets = [-1, 2]; // Negative target treated as 0 -> log(0+1)=0
      const outputs = [1, -2]; // Negative output treated as 0 -> log(0+1)=0
      const logTarget1 = Math.log(0 + 1); // 0
      const logOutput1 = Math.log(1 + 1); // log(2)
      const logTarget2 = Math.log(2 + 1); // log(3)
      const logOutput2 = Math.log(0 + 1); // 0
      const expected =
        (Math.pow(logTarget1 - logOutput1, 2) +
          Math.pow(logTarget2 - logOutput2, 2)) /
        2;
      // Act
      const result = Cost.msle(targets, outputs);
      // Assert
      expect(result).toBeCloseTo(expected, epsilon);
    });

    test('should handle single element arrays', () => {
      // Arrange
      const targets = [Math.E - 1]; // log(target+1) = 1
      const outputs = [0]; // log(output+1) = 0
      const expected = Math.pow(1 - 0, 2) / 1;
      // Act
      const result = Cost.msle(targets, outputs);
      // Assert
      expect(result).toBeCloseTo(expected, epsilon);
    });
  });

  describe('hinge()', () => {
    test('should calculate hinge loss correctly', () => {
      // Arrange
      const targets = [-1, 1, -1, 1]; // Expected: -1 or 1
      const outputs = [-0.5, 0.8, 1.2, -0.1]; // Raw scores
      // Losses: max(0, 1 - (-1*-0.5)) = max(0, 0.5) = 0.5
      //         max(0, 1 - (1*0.8)) = max(0, 0.2) = 0.2
      //         max(0, 1 - (-1*1.2)) = max(0, 2.2) = 2.2
      //         max(0, 1 - (1*-0.1)) = max(0, 1.1) = 1.1
      const expected = (0.5 + 0.2 + 2.2 + 1.1) / 4;
      // Act
      const result = Cost.hinge(targets, outputs);
      // Assert
      expect(result).toBeCloseTo(expected, epsilon);
    });

    test('should return 0 for correct and confident predictions', () => {
      // Arrange
      const targets = [-1, 1];
      const outputs = [-1.5, 1.1]; // target*output >= 1
      // Losses: max(0, 1 - (-1*-1.5)) = max(0, -0.5) = 0
      //         max(0, 1 - (1*1.1)) = max(0, -0.1) = 0
      // Act
      const result = Cost.hinge(targets, outputs);
      // Assert
      expect(result).toBe(0);
    });

    test('should calculate loss for correct but not confident predictions', () => {
      // Arrange
      const targets = [-1, 1];
      const outputs = [-0.5, 0.5]; // 0 < target*output < 1
      // Losses: max(0, 1 - (-1*-0.5)) = max(0, 0.5) = 0.5
      //         max(0, 1 - (1*0.5)) = max(0, 0.5) = 0.5
      const expected = (0.5 + 0.5) / 2;
      // Act
      const result = Cost.hinge(targets, outputs);
      // Assert
      expect(result).toBeCloseTo(expected, epsilon);
    });

    test('should handle single element arrays', () => {
      // Arrange
      const targets = [-1];
      const outputs = [0.2]; // Loss: max(0, 1 - (-1*0.2)) = max(0, 1.2) = 1.2
      const expected = 1.2 / 1;
      // Act
      const result = Cost.hinge(targets, outputs);
      // Assert
      expect(result).toBeCloseTo(expected, epsilon);
    });
  });
});
