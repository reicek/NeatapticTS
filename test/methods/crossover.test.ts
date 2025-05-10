import { crossover } from '../../src/methods/crossover';

describe('Crossover Methods', () => {
  describe('SINGLE_POINT', () => {
    describe('Scenario: Existence', () => {
      test('should exist', () => {
        // Assert
        expect(crossover.SINGLE_POINT).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      test('should have correct name property', () => {
        // Assert
        expect(crossover.SINGLE_POINT.name).toBe('SINGLE_POINT');
      });
    });
    describe('Scenario: Config property', () => {
      test('should have config property', () => {
        // Assert
        expect(crossover.SINGLE_POINT.config).toBeDefined();
      });
      test('should have config as an array of length 1', () => {
        // Assert
        expect(Array.isArray(crossover.SINGLE_POINT.config)).toBe(true);
      });
      test('should have config array of length 1', () => {
        // Assert
        expect(crossover.SINGLE_POINT.config).toHaveLength(1);
      });
      test('should have correct config value', () => {
        // Assert
        expect(crossover.SINGLE_POINT.config[0]).toBeCloseTo(0.4);
      });
      test('config value is within range [0, 1]', () => {
        // Arrange
        const configVal = crossover.SINGLE_POINT.config[0];
        // Assert
        expect(configVal).toBeGreaterThanOrEqual(0);
      });
      test('config value is less than or equal to 1', () => {
        // Arrange
        const configVal = crossover.SINGLE_POINT.config[0];
        // Assert
        expect(configVal).toBeLessThanOrEqual(1);
      });
    });
  });

  describe('TWO_POINT', () => {
    describe('Scenario: Existence', () => {
      test('should exist', () => {
        // Assert
        expect(crossover.TWO_POINT).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      test('should have correct name property', () => {
        // Assert
        expect(crossover.TWO_POINT.name).toBe('TWO_POINT');
      });
    });
    describe('Scenario: Config property', () => {
      test('should have config property', () => {
        // Assert
        expect(crossover.TWO_POINT.config).toBeDefined();
      });
      test('should have config as an array of length 2', () => {
        // Assert
        expect(Array.isArray(crossover.TWO_POINT.config)).toBe(true);
      });
      test('should have config array of length 2', () => {
        // Assert
        expect(crossover.TWO_POINT.config).toHaveLength(2);
      });
      test('should have correct config value 0', () => {
        // Assert
        expect(crossover.TWO_POINT.config[0]).toBeCloseTo(0.4);
      });
      test('should have correct config value 1', () => {
        // Assert
        expect(crossover.TWO_POINT.config[1]).toBeCloseTo(0.9);
      });
      test('config value 0 is within range [0, 1]', () => {
        // Arrange
        const configVal1 = crossover.TWO_POINT.config[0];
        // Assert
        expect(configVal1).toBeGreaterThanOrEqual(0);
      });
      test('config value 0 is less than or equal to 1', () => {
        // Arrange
        const configVal1 = crossover.TWO_POINT.config[0];
        // Assert
        expect(configVal1).toBeLessThanOrEqual(1);
      });
      test('config value 1 is within range [0, 1]', () => {
        // Arrange
        const configVal2 = crossover.TWO_POINT.config[1];
        // Assert
        expect(configVal2).toBeGreaterThanOrEqual(0);
      });
      test('config value 1 is less than or equal to 1', () => {
        // Arrange
        const configVal2 = crossover.TWO_POINT.config[1];
        // Assert
        expect(configVal2).toBeLessThanOrEqual(1);
      });
      test('config value 0 is less than or equal to config value 1', () => {
        // Arrange
        const configVal1 = crossover.TWO_POINT.config[0];
        const configVal2 = crossover.TWO_POINT.config[1];
        // Assert
        expect(configVal1).toBeLessThanOrEqual(configVal2);
      });
    });
  });

  describe('UNIFORM', () => {
    describe('Scenario: Existence', () => {
      test('should exist', () => {
        // Assert
        expect(crossover.UNIFORM).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      test('should have correct name property', () => {
        // Assert
        expect(crossover.UNIFORM.name).toBe('UNIFORM');
      });
    });
    describe('Scenario: Config property', () => {
      test('should not have config property', () => {
        // Assert
        expect((crossover.UNIFORM as any).config).toBeUndefined();
      });
    });
  });

  describe('AVERAGE', () => {
    describe('Scenario: Existence', () => {
      test('should exist', () => {
        // Assert
        expect(crossover.AVERAGE).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      test('should have correct name property', () => {
        // Assert
        expect(crossover.AVERAGE.name).toBe('AVERAGE');
      });
    });
    describe('Scenario: Config property', () => {
      test('should not have config property', () => {
        // Assert
        expect((crossover.AVERAGE as any).config).toBeUndefined();
      });
    });
  });
});
