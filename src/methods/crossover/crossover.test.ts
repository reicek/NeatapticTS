import { crossover } from './crossover';

describe('crossover', () => {
  describe('given the exported inheritance shelf', () => {
    describe('when reading the available crossover strategies', () => {
      it('exposes the expected keys', () => {
        // Arrange
        const expectedKeys = [
          'AVERAGE',
          'SINGLE_POINT',
          'TWO_POINT',
          'UNIFORM',
        ];

        // Act
        const actualKeys = Object.keys(crossover).toSorted();

        // Assert
        expect(actualKeys).toStrictEqual(expectedKeys);
      });
    });
  });

  describe('SINGLE_POINT', () => {
    describe('when reading its public identifier', () => {
      it('uses the expected name', () => {
        // Arrange
        const expectedName = 'SINGLE_POINT';

        // Act
        const actualName = crossover.SINGLE_POINT.name;

        // Assert
        expect(actualName).toBe(expectedName);
      });
    });

    describe('when reading its default cut configuration', () => {
      it('uses the expected single cut', () => {
        // Arrange
        const expectedConfig = [0.4];

        // Act
        const actualConfig = crossover.SINGLE_POINT.config;

        // Assert
        expect(actualConfig).toStrictEqual(expectedConfig);
      });
    });
  });

  describe('TWO_POINT', () => {
    describe('when reading its public identifier', () => {
      it('uses the expected name', () => {
        // Arrange
        const expectedName = 'TWO_POINT';

        // Act
        const actualName = crossover.TWO_POINT.name;

        // Assert
        expect(actualName).toBe(expectedName);
      });
    });

    describe('when reading its default cut configuration', () => {
      it('uses the expected two cuts', () => {
        // Arrange
        const expectedConfig = [0.4, 0.9];

        // Act
        const actualConfig = crossover.TWO_POINT.config;

        // Assert
        expect(actualConfig).toStrictEqual(expectedConfig);
      });
    });
  });

  describe('UNIFORM', () => {
    describe('when checking for an explicit config payload', () => {
      it('does not expose one', () => {
        // Arrange
        const uniformStrategy = crossover.UNIFORM as { config?: number[] };

        // Act
        const actualConfig = uniformStrategy.config;

        // Assert
        expect(actualConfig).toBeUndefined();
      });
    });
  });

  describe('AVERAGE', () => {
    describe('when checking for an explicit config payload', () => {
      it('does not expose one', () => {
        // Arrange
        const averageStrategy = crossover.AVERAGE as { config?: number[] };

        // Act
        const actualConfig = averageStrategy.config;

        // Assert
        expect(actualConfig).toBeUndefined();
      });
    });
  });
});
