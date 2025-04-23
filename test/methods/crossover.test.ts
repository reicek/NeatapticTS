import { crossover } from '../../src/methods/crossover';

describe('Crossover Methods', () => {
  describe('SINGLE_POINT', () => {
    test('should exist', () => {
      // Assert
      expect(crossover.SINGLE_POINT).toBeDefined();
    });

    test('should have correct name property', () => {
      // Assert
      expect(crossover.SINGLE_POINT.name).toBe('SINGLE_POINT');
    });

    test('should have config property', () => {
      // Assert
      expect(crossover.SINGLE_POINT.config).toBeDefined();
    });

    test('should have config as an array of length 1', () => {
      // Assert
      expect(Array.isArray(crossover.SINGLE_POINT.config)).toBe(true);
      expect(crossover.SINGLE_POINT.config).toHaveLength(1);
    });

    test('should have correct config value', () => {
      // Assert
      expect(crossover.SINGLE_POINT.config[0]).toBeCloseTo(0.4);
    });
  });

  describe('TWO_POINT', () => {
    test('should exist', () => {
      // Assert
      expect(crossover.TWO_POINT).toBeDefined();
    });

    test('should have correct name property', () => {
      // Assert
      expect(crossover.TWO_POINT.name).toBe('TWO_POINT');
    });

    test('should have config property', () => {
      // Assert
      expect(crossover.TWO_POINT.config).toBeDefined();
    });

    test('should have config as an array of length 2', () => {
      // Assert
      expect(Array.isArray(crossover.TWO_POINT.config)).toBe(true);
      expect(crossover.TWO_POINT.config).toHaveLength(2);
    });

    test('should have correct config values', () => {
      // Assert
      expect(crossover.TWO_POINT.config[0]).toBeCloseTo(0.4);
      expect(crossover.TWO_POINT.config[1]).toBeCloseTo(0.9);
    });
  });

  describe('UNIFORM', () => {
    test('should exist', () => {
      // Assert
      expect(crossover.UNIFORM).toBeDefined();
    });

    test('should have correct name property', () => {
      // Assert
      expect(crossover.UNIFORM.name).toBe('UNIFORM');
    });

    test('should not have config property', () => {
      // Assert
      expect((crossover.UNIFORM as any).config).toBeUndefined();
    });
  });

  describe('AVERAGE', () => {
    test('should exist', () => {
      // Assert
      expect(crossover.AVERAGE).toBeDefined();
    });

    test('should have correct name property', () => {
      // Assert
      expect(crossover.AVERAGE.name).toBe('AVERAGE');
    });

    test('should not have config property', () => {
      // Assert
      expect((crossover.AVERAGE as any).config).toBeUndefined();
    });
  });
});
