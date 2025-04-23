import { gating } from '../../src/methods/gating';

describe('Gating Methods', () => {
  describe('OUTPUT', () => {
    test('should exist', () => {
      // Assert
      expect(gating.OUTPUT).toBeDefined();
    });

    test('should have correct name property', () => {
      // Assert
      expect(gating.OUTPUT.name).toBe('OUTPUT');
    });
  });

  describe('INPUT', () => {
    test('should exist', () => {
      // Assert
      expect(gating.INPUT).toBeDefined();
    });

    test('should have correct name property', () => {
      // Assert
      expect(gating.INPUT.name).toBe('INPUT');
    });
  });

  describe('SELF', () => {
    test('should exist', () => {
      // Assert
      expect(gating.SELF).toBeDefined();
    });

    test('should have correct name property', () => {
      // Assert
      expect(gating.SELF.name).toBe('SELF');
    });
  });
});
