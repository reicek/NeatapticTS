import groupConnectionDefault, { groupConnection } from '../../src/methods/connection';

describe('Group Connection Methods', () => {
  describe('ALL_TO_ALL', () => {
    describe('Scenario: Existence', () => {
      it('should exist', () => {
        // Arrange
        // Act
        // Assert
        expect(groupConnection.ALL_TO_ALL).toBeDefined();
      });
      it('should not be undefined', () => {
        // Arrange
        // Act
        // Assert
        expect(groupConnection.ALL_TO_ALL).not.toBeUndefined();
      });
    });
    describe('Scenario: Name property', () => {
      it('should have correct name property', () => {
        // Arrange
        // Act
        // Assert
        expect(groupConnection.ALL_TO_ALL.name).toBe('ALL_TO_ALL');
      });
      it('should not have incorrect name property', () => {
        // Arrange
        // Act
        // Assert
        expect(groupConnection.ALL_TO_ALL.name).not.toBe('WRONG_NAME');
      });
    });
  });

  describe('ALL_TO_ELSE', () => {
    describe('Scenario: Existence', () => {
      it('should exist', () => {
        // Arrange
        // Act
        // Assert
        expect(groupConnection.ALL_TO_ELSE).toBeDefined();
      });
      it('should not be undefined', () => {
        // Arrange
        // Act
        // Assert
        expect(groupConnection.ALL_TO_ELSE).not.toBeUndefined();
      });
    });
    describe('Scenario: Name property', () => {
      it('should have correct name property', () => {
        // Arrange
        // Act
        // Assert
        expect(groupConnection.ALL_TO_ELSE.name).toBe('ALL_TO_ELSE');
      });
      it('should not have incorrect name property', () => {
        // Arrange
        // Act
        // Assert
        expect(groupConnection.ALL_TO_ELSE.name).not.toBe('WRONG_NAME');
      });
    });
  });

  describe('ONE_TO_ONE', () => {
    describe('Scenario: Existence', () => {
      it('should exist', () => {
        // Arrange
        // Act
        // Assert
        expect(groupConnection.ONE_TO_ONE).toBeDefined();
      });
      it('should not be undefined', () => {
        // Arrange
        // Act
        // Assert
        expect(groupConnection.ONE_TO_ONE).not.toBeUndefined();
      });
    });
    describe('Scenario: Name property', () => {
      it('should have correct name property', () => {
        // Arrange
        // Act
        // Assert
        expect(groupConnection.ONE_TO_ONE.name).toBe('ONE_TO_ONE');
      });
      it('should not have incorrect name property', () => {
        // Arrange
        // Act
        // Assert
        expect(groupConnection.ONE_TO_ONE.name).not.toBe('WRONG_NAME');
      });
    });
  });

  describe('Utility & Integrity', () => {
    describe('Scenario: Default export', () => {
      it('should export groupConnection as default', () => {
        // Arrange
        // Act
        // Assert
        expect(groupConnectionDefault).toBe(groupConnection);
      });
    });
    describe('Scenario: Object shape', () => {
      it('should have all required keys', () => {
        // Arrange
        // Act
        // Assert
        expect(Object.keys(groupConnection).sort()).toEqual([
          'ALL_TO_ALL',
          'ALL_TO_ELSE',
          'ONE_TO_ONE',
        ].sort());
      });
    });
    describe('Scenario: Immutability', () => {
      it('should not allow mutation of groupConnection object', () => {
        // Arrange
        // Act
        // Assert
        expect(() => {
          (groupConnection as any)['NEW_PROP'] = 123;
        }).toThrow(TypeError);
      });
      it('should not allow mutation of ALL_TO_ALL property', () => {
        // Arrange
        // Act
        // Assert
        expect(() => {
          (groupConnection.ALL_TO_ALL as any)['name'] = 'MUTATED';
        }).toThrow(TypeError);
      });
      it('should be deeply frozen', () => {
        // Arrange
        // Act
        // Assert
        expect(Object.isFrozen(groupConnection)).toBe(true);
        expect(Object.isFrozen(groupConnection.ALL_TO_ALL)).toBe(true);
        expect(Object.isFrozen(groupConnection.ALL_TO_ELSE)).toBe(true);
        expect(Object.isFrozen(groupConnection.ONE_TO_ONE)).toBe(true);
      });
    });
  });
});
