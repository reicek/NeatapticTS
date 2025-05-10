import { gating } from '../../src/methods/gating';

describe('Gating Methods', () => {
  describe('OUTPUT', () => {
    describe('Scenario: Existence', () => {
      test('should exist', () => {
        // Assert
        expect(gating.OUTPUT).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      test('should have correct name property', () => {
        // Assert
        expect(gating.OUTPUT.name).toBe('OUTPUT');
      });
    });
  });

  describe('INPUT', () => {
    describe('Scenario: Existence', () => {
      test('should exist', () => {
        // Assert
        expect(gating.INPUT).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      test('should have correct name property', () => {
        // Assert
        expect(gating.INPUT.name).toBe('INPUT');
      });
    });
  });

  describe('SELF', () => {
    describe('Scenario: Existence', () => {
      test('should exist', () => {
        // Assert
        expect(gating.SELF).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      test('should have correct name property', () => {
        // Assert
        expect(gating.SELF.name).toBe('SELF');
      });
    });
  });
});
