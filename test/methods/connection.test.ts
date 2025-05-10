import { groupConnection } from '../../src/methods/connection';

describe('Group Connection Methods', () => {
  describe('ALL_TO_ALL', () => {
    describe('Scenario: Existence', () => {
      test('should exist', () => {
        // Assert
        expect(groupConnection.ALL_TO_ALL).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      test('should have correct name property', () => {
        // Assert
        expect(groupConnection.ALL_TO_ALL.name).toBe('ALL_TO_ALL');
      });
    });
  });

  describe('ALL_TO_ELSE', () => {
    describe('Scenario: Existence', () => {
      test('should exist', () => {
        // Assert
        expect(groupConnection.ALL_TO_ELSE).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      test('should have correct name property', () => {
        // Assert
        expect(groupConnection.ALL_TO_ELSE.name).toBe('ALL_TO_ELSE');
      });
    });
  });

  describe('ONE_TO_ONE', () => {
    describe('Scenario: Existence', () => {
      test('should exist', () => {
        // Assert
        expect(groupConnection.ONE_TO_ONE).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      test('should have correct name property', () => {
        // Assert
        expect(groupConnection.ONE_TO_ONE.name).toBe('ONE_TO_ONE');
      });
    });
  });
});
