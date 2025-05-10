import { selection } from '../../src/methods/selection';

describe('Selection Methods', () => {
  describe('FITNESS_PROPORTIONATE', () => {
    describe('when checking existence', () => {
      test('should be defined', () => {
        // Arrange
        // Act
        const method = selection.FITNESS_PROPORTIONATE;
        // Assert
        expect(method).toBeDefined();
      });
    });

    describe('when checking name property', () => {
      test('should have correct name', () => {
        // Arrange
        // Act
        const name = selection.FITNESS_PROPORTIONATE.name;
        // Assert
        expect(name).toBe('FITNESS_PROPORTIONATE');
      });

      test('should not have unexpected properties', () => {
        // Arrange
        // Act
        const keys = Object.keys(selection.FITNESS_PROPORTIONATE);
        // Assert
        expect(keys).toEqual(['name']);
      });
    });
  });

  describe('POWER', () => {
    describe('when checking existence', () => {
      test('should be defined', () => {
        // Arrange
        // Act
        const method = selection.POWER;
        // Assert
        expect(method).toBeDefined();
      });
    });

    describe('when checking name property', () => {
      test('should have correct name', () => {
        // Arrange
        // Act
        const name = selection.POWER.name;
        // Assert
        expect(name).toBe('POWER');
      });
    });

    describe('when checking power property', () => {
      test('should have default value 4', () => {
        // Arrange
        // Act
        const power = selection.POWER.power;
        // Assert
        expect(power).toBe(4);
      });

      test('should be greater than 0', () => {
        // Arrange
        // Act
        const power = selection.POWER.power;
        // Assert
        expect(power).toBeGreaterThan(0);
      });
    });

    describe('when checking for unexpected properties', () => {
      test('should only have name and power', () => {
        // Arrange
        // Act
        const keys = Object.keys(selection.POWER);
        // Assert
        expect(keys.sort()).toEqual(['name', 'power'].sort());
      });
    });
  });

  describe('TOURNAMENT', () => {
    describe('when checking existence', () => {
      test('should be defined', () => {
        // Arrange
        // Act
        const method = selection.TOURNAMENT;
        // Assert
        expect(method).toBeDefined();
      });
    });

    describe('when checking name property', () => {
      test('should have correct name', () => {
        // Arrange
        // Act
        const name = selection.TOURNAMENT.name;
        // Assert
        expect(name).toBe('TOURNAMENT');
      });
    });

    describe('when checking size property', () => {
      test('should have default value 5', () => {
        // Arrange
        // Act
        const size = selection.TOURNAMENT.size;
        // Assert
        expect(size).toBe(5);
      });

      test('should be greater than 0', () => {
        // Arrange
        // Act
        const size = selection.TOURNAMENT.size;
        // Assert
        expect(size).toBeGreaterThan(0);
      });
    });

    describe('when checking probability property', () => {
      test('should have default value 0.5', () => {
        // Arrange
        // Act
        const probability = selection.TOURNAMENT.probability;
        // Assert
        expect(probability).toBe(0.5);
      });

      test('should be >= 0', () => {
        // Arrange
        // Act
        const probability = selection.TOURNAMENT.probability;
        // Assert
        expect(probability).toBeGreaterThanOrEqual(0);
      });

      test('should be <= 1', () => {
        // Arrange
        // Act
        const probability = selection.TOURNAMENT.probability;
        // Assert
        expect(probability).toBeLessThanOrEqual(1);
      });
    });

    describe('when checking for unexpected properties', () => {
      test('should only have name, size, and probability', () => {
        // Arrange
        // Act
        const keys = Object.keys(selection.TOURNAMENT);
        // Assert
        expect(keys.sort()).toEqual(['name', 'size', 'probability'].sort());
      });
    });
  });
});
