import { selection } from '../../src/methods/selection';

describe('Selection Methods', () => {
  describe('FITNESS_PROPORTIONATE', () => {
    describe('when checking existence', () => {
      it('should be defined', () => {
        // Arrange
        // Act
        const method = selection.FITNESS_PROPORTIONATE;
        // Assert
        expect(method).toBeDefined();
      });
    });

    describe('when checking name property', () => {
      it('should have correct name', () => {
        // Arrange
        // Act
        const name = selection.FITNESS_PROPORTIONATE.name;
        // Assert
        expect(name).toBe('FITNESS_PROPORTIONATE');
      });
    });

    describe('when checking for unexpected properties', () => {
      it('should only have name', () => {
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
      it('should be defined', () => {
        // Arrange
        // Act
        const method = selection.POWER;
        // Assert
        expect(method).toBeDefined();
      });
    });

    describe('when checking name property', () => {
      it('should have correct name', () => {
        // Arrange
        // Act
        const name = selection.POWER.name;
        // Assert
        expect(name).toBe('POWER');
      });
    });

    describe('when checking power property', () => {
      it('should have default value 4', () => {
        // Arrange
        // Act
        const power = selection.POWER.power;
        // Assert
        expect(power).toBe(4);
      });
      it('should be greater than 0', () => {
        // Arrange
        // Act
        const power = selection.POWER.power;
        // Assert
        expect(power).toBeGreaterThan(0);
      });
    });
    describe('when checking for unexpected properties', () => {
      it('should only have name and power', () => {
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
      it('should be defined', () => {
        // Arrange
        // Act
        const method = selection.TOURNAMENT;
        // Assert
        expect(method).toBeDefined();
      });
    });

    describe('when checking name property', () => {
      it('should have correct name', () => {
        // Arrange
        // Act
        const name = selection.TOURNAMENT.name;
        // Assert
        expect(name).toBe('TOURNAMENT');
      });
    });

    describe('when checking size property', () => {
      it('should have default value 5', () => {
        // Arrange
        // Act
        const size = selection.TOURNAMENT.size;
        // Assert
        expect(size).toBe(5);
      });

      it('should be greater than 0', () => {
        // Arrange
        // Act
        const size = selection.TOURNAMENT.size;
        // Assert
        expect(size).toBeGreaterThan(0);
      });
    });

    describe('when checking probability property', () => {
      it('should have default value 0.5', () => {
        // Arrange
        // Act
        const probability = selection.TOURNAMENT.probability;
        // Assert
        expect(probability).toBe(0.5);
      });
      it('should be >= 0', () => {
        // Arrange
        // Act
        const probability = selection.TOURNAMENT.probability;
        // Assert
        expect(probability).toBeGreaterThanOrEqual(0);
      });
      it('should be <= 1', () => {
        // Arrange
        // Act
        const probability = selection.TOURNAMENT.probability;
        // Assert
        expect(probability).toBeLessThanOrEqual(1);
      });
    });
    describe('when checking for unexpected properties', () => {
      it('should only have name, size, and probability', () => {
        // Arrange
        // Act
        const keys = Object.keys(selection.TOURNAMENT);
        // Assert
        expect(keys.sort()).toEqual(['name', 'size', 'probability'].sort());
      });
    });
  });

  describe('selection utility object', () => {
    it('should contain all expected selection methods', () => {
      // Arrange
      // Act
      const keys = Object.keys(selection);
      // Assert
      expect(keys.sort()).toEqual(['FITNESS_PROPORTIONATE', 'POWER', 'TOURNAMENT'].sort());
    });
  });

  describe('mutation and immutability', () => {
    beforeEach(() => {
      // Reset mutated values to defaults
      selection.FITNESS_PROPORTIONATE.name = 'FITNESS_PROPORTIONATE';
      selection.POWER.power = 4;
      selection.TOURNAMENT.size = 5;
      selection.TOURNAMENT.probability = 0.5;
    });
    it('should allow mutation of FITNESS_PROPORTIONATE.name (not immutable)', () => {
      // Arrange
      const original = selection.FITNESS_PROPORTIONATE.name;
      // Act
      selection.FITNESS_PROPORTIONATE.name = 'CHANGED';
      // Assert
      expect(selection.FITNESS_PROPORTIONATE.name).toBe('CHANGED');
      // Cleanup
      selection.FITNESS_PROPORTIONATE.name = original;
    });
    it('should allow mutation of POWER.power (not immutable)', () => {
      // Arrange
      const original = selection.POWER.power;
      // Act
      selection.POWER.power = 999;
      // Assert
      expect(selection.POWER.power).toBe(999);
      // Cleanup
      selection.POWER.power = original;
    });
    it('should allow mutation of TOURNAMENT.size (not immutable)', () => {
      // Arrange
      const original = selection.TOURNAMENT.size;
      // Act
      selection.TOURNAMENT.size = 999;
      // Assert
      expect(selection.TOURNAMENT.size).toBe(999);
      // Cleanup
      selection.TOURNAMENT.size = original;
    });
  });

  describe('serialization', () => {
    beforeEach(() => {
      // Reset mutated values to defaults
      selection.FITNESS_PROPORTIONATE.name = 'FITNESS_PROPORTIONATE';
      selection.POWER.power = 4;
      selection.TOURNAMENT.size = 5;
      selection.TOURNAMENT.probability = 0.5;
    });
    it('should serialize and deserialize FITNESS_PROPORTIONATE correctly', () => {
      // Arrange
      const json = JSON.stringify(selection.FITNESS_PROPORTIONATE);
      // Act
      const parsed = JSON.parse(json);
      // Assert
      expect(parsed.name).toBe('FITNESS_PROPORTIONATE');
    });
    it('should serialize and deserialize POWER correctly', () => {
      // Arrange
      const json = JSON.stringify(selection.POWER);
      // Act
      const parsed = JSON.parse(json);
      // Assert
      expect(parsed.name).toBe('POWER');
      expect(parsed.power).toBe(4);
    });
    it('should serialize and deserialize TOURNAMENT correctly', () => {
      // Arrange
      const json = JSON.stringify(selection.TOURNAMENT);
      // Act
      const parsed = JSON.parse(json);
      // Assert
      expect(parsed.name).toBe('TOURNAMENT');
      expect(parsed.size).toBe(5);
      expect(parsed.probability).toBe(0.5);
    });
  });
});
