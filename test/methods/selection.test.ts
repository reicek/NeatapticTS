import { selection } from '../../src/methods/selection';

describe('Selection Methods', () => {
  describe('FITNESS_PROPORTIONATE', () => {
    test('should exist', () => {
      // Assert
      expect(selection.FITNESS_PROPORTIONATE).toBeDefined();
    });

    test('should have correct name property', () => {
      // Assert
      expect(selection.FITNESS_PROPORTIONATE.name).toBe(
        'FITNESS_PROPORTIONATE'
      );
    });
  });

  describe('POWER', () => {
    test('should exist', () => {
      // Assert
      expect(selection.POWER).toBeDefined();
    });

    test('should have correct name property', () => {
      // Assert
      expect(selection.POWER.name).toBe('POWER');
    });

    test('should have power property with default value', () => {
      // Assert
      expect(selection.POWER.power).toBe(4);
    });

    test('should have power property with default value > 0', () => {
      // Assert
      const powerVal = selection.POWER.power;
      expect(powerVal).toBe(4);
      expect(powerVal).toBeGreaterThan(0);
    });
  });

  describe('TOURNAMENT', () => {
    test('should exist', () => {
      // Assert
      expect(selection.TOURNAMENT).toBeDefined();
    });

    test('should have correct name property', () => {
      // Assert
      expect(selection.TOURNAMENT.name).toBe('TOURNAMENT');
    });

    test('should have size property with default value', () => {
      // Assert
      expect(selection.TOURNAMENT.size).toBe(5);
    });

    test('should have size property with default value > 0', () => {
      // Assert
      const sizeVal = selection.TOURNAMENT.size;
      expect(sizeVal).toBe(5);
      expect(sizeVal).toBeGreaterThan(0);
    });

    test('should have probability property with default value', () => {
      // Assert
      expect(selection.TOURNAMENT.probability).toBe(0.5);
    });

    test('should have probability property with default value within [0, 1]', () => {
      // Assert
      const probVal = selection.TOURNAMENT.probability;
      expect(probVal).toBe(0.5);
      expect(probVal).toBeGreaterThanOrEqual(0);
      expect(probVal).toBeLessThanOrEqual(1);
    });
  });
});
