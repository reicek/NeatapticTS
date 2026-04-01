import { gating } from './gating';

describe('gating', () => {
  describe('given the exported routing shelf', () => {
    describe('when reading the available gate placements', () => {
      it('exposes the expected keys', () => {
        // Arrange
        const expectedKeys = ['INPUT', 'OUTPUT', 'SELF'];

        // Act
        const actualKeys = Object.keys(gating).toSorted();

        // Assert
        expect(actualKeys).toStrictEqual(expectedKeys);
      });
    });

    describe('when checking whether the shelf is frozen', () => {
      it('remains mutable', () => {
        // Arrange
        const routingShelf = gating;

        // Act
        const isFrozen = Object.isFrozen(routingShelf);

        // Assert
        expect(isFrozen).toBe(false);
      });
    });

    describe('when serializing the shelf', () => {
      it('preserves the public routing keys', () => {
        // Arrange
        const expectedKeys = ['INPUT', 'OUTPUT', 'SELF'];

        // Act
        const actualKeys = Object.keys(
          JSON.parse(JSON.stringify(gating)),
        ).toSorted();

        // Assert
        expect(actualKeys).toStrictEqual(expectedKeys);
      });
    });
  });

  describe('OUTPUT', () => {
    describe('when reading its public identifier', () => {
      it('uses the expected name', () => {
        // Arrange
        const expectedName = 'OUTPUT';

        // Act
        const actualName = gating.OUTPUT.name;

        // Assert
        expect(actualName).toBe(expectedName);
      });
    });
  });

  describe('INPUT', () => {
    describe('when reading its public identifier', () => {
      it('uses the expected name', () => {
        // Arrange
        const expectedName = 'INPUT';

        // Act
        const actualName = gating.INPUT.name;

        // Assert
        expect(actualName).toBe(expectedName);
      });
    });
  });

  describe('SELF', () => {
    describe('when reading its public identifier', () => {
      it('uses the expected name', () => {
        // Arrange
        const expectedName = 'SELF';

        // Act
        const actualName = gating.SELF.name;

        // Assert
        expect(actualName).toBe(expectedName);
      });
    });
  });
});
