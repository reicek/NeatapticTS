import groupConnection, {
  groupConnection as namedGroupConnection,
} from './connection';

describe('groupConnection', () => {
  describe('given the exported policy shelf', () => {
    describe('when reading the available connection policies', () => {
      it('exposes the expected keys', () => {
        // Arrange
        const expectedKeys = ['ALL_TO_ALL', 'ALL_TO_ELSE', 'ONE_TO_ONE'];

        // Act
        const actualKeys = Object.keys(namedGroupConnection).toSorted();

        // Assert
        expect(actualKeys).toStrictEqual(expectedKeys);
      });
    });

    describe('when reading the default export', () => {
      it('re-exports the named shelf', () => {
        // Arrange
        const expectedShelf = namedGroupConnection;

        // Act
        const actualShelf = groupConnection;

        // Assert
        expect(actualShelf).toBe(expectedShelf);
      });
    });

    describe('when checking shelf immutability', () => {
      it('is frozen at the top level', () => {
        // Arrange
        const connectionShelf = namedGroupConnection;

        // Act
        const isFrozen = Object.isFrozen(connectionShelf);

        // Assert
        expect(isFrozen).toBe(true);
      });
    });
  });

  describe('ALL_TO_ALL', () => {
    describe('when reading its public identifier', () => {
      it('uses the expected name', () => {
        // Arrange
        const expectedName = 'ALL_TO_ALL';

        // Act
        const actualName = namedGroupConnection.ALL_TO_ALL.name;

        // Assert
        expect(actualName).toBe(expectedName);
      });
    });

    describe('when checking entry immutability', () => {
      it('is frozen', () => {
        // Arrange
        const connectionPolicy = namedGroupConnection.ALL_TO_ALL;

        // Act
        const isFrozen = Object.isFrozen(connectionPolicy);

        // Assert
        expect(isFrozen).toBe(true);
      });
    });
  });

  describe('ALL_TO_ELSE', () => {
    describe('when reading its public identifier', () => {
      it('uses the expected name', () => {
        // Arrange
        const expectedName = 'ALL_TO_ELSE';

        // Act
        const actualName = namedGroupConnection.ALL_TO_ELSE.name;

        // Assert
        expect(actualName).toBe(expectedName);
      });
    });
  });

  describe('ONE_TO_ONE', () => {
    describe('when reading its public identifier', () => {
      it('uses the expected name', () => {
        // Arrange
        const expectedName = 'ONE_TO_ONE';

        // Act
        const actualName = namedGroupConnection.ONE_TO_ONE.name;

        // Assert
        expect(actualName).toBe(expectedName);
      });
    });
  });
});
