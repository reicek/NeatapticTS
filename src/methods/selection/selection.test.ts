import { selection } from './selection';

describe('selection', () => {
  describe('given the exported selection shelf', () => {
    describe('when reading the available strategies', () => {
      it('exposes the expected keys', () => {
        // Arrange
        const expectedKeys = ['FITNESS_PROPORTIONATE', 'POWER', 'TOURNAMENT'];

        // Act
        const actualKeys = Object.keys(selection).toSorted();

        // Assert
        expect(actualKeys).toStrictEqual(expectedKeys);
      });
    });

    describe('when checking whether the shelf is frozen', () => {
      it('remains mutable', () => {
        // Arrange
        const selectionShelf = selection;

        // Act
        const isFrozen = Object.isFrozen(selectionShelf);

        // Assert
        expect(isFrozen).toBe(false);
      });
    });
  });

  describe('FITNESS_PROPORTIONATE', () => {
    describe('when reading its public identifier', () => {
      it('uses the expected name', () => {
        // Arrange
        const expectedName = 'FITNESS_PROPORTIONATE';

        // Act
        const actualName = selection.FITNESS_PROPORTIONATE.name;

        // Assert
        expect(actualName).toBe(expectedName);
      });
    });
  });

  describe('POWER', () => {
    describe('when reading its public identifier', () => {
      it('uses the expected name', () => {
        // Arrange
        const expectedName = 'POWER';

        // Act
        const actualName = selection.POWER.name;

        // Assert
        expect(actualName).toBe(expectedName);
      });
    });

    describe('when reading its default pressure', () => {
      it('uses the default power of four', () => {
        // Arrange
        const expectedPower = 4;

        // Act
        const actualPower = selection.POWER.power;

        // Assert
        expect(actualPower).toBe(expectedPower);
      });
    });
  });

  describe('TOURNAMENT', () => {
    describe('when reading its public identifier', () => {
      it('uses the expected name', () => {
        // Arrange
        const expectedName = 'TOURNAMENT';

        // Act
        const actualName = selection.TOURNAMENT.name;

        // Assert
        expect(actualName).toBe(expectedName);
      });
    });

    describe('when reading its default bracket size', () => {
      it('uses the default size of five', () => {
        // Arrange
        const expectedSize = 5;

        // Act
        const actualSize = selection.TOURNAMENT.size;

        // Assert
        expect(actualSize).toBe(expectedSize);
      });
    });

    describe('when reading its default winner probability', () => {
      it('uses the default probability of one half', () => {
        // Arrange
        const expectedProbability = 0.5;

        // Act
        const actualProbability = selection.TOURNAMENT.probability;

        // Assert
        expect(actualProbability).toBe(expectedProbability);
      });
    });
  });
});
