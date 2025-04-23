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

    test('should have probability property with default value', () => {
      // Assert
      expect(selection.TOURNAMENT.probability).toBe(0.5);
    });
  });
});
