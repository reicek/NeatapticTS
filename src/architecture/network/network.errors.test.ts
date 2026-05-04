import { NetworkConstructorDimensionRequiredError } from './network.errors';

describe('network errors chapter', () => {
  describe('NetworkConstructorDimensionRequiredError', () => {
    describe('given the network constructor lacks interface sizes', () => {
      it('preserves the configured message, name, and cause', () => {
        // Arrange
        const cause = new Error('missing dimensions');

        // Act
        const error = new NetworkConstructorDimensionRequiredError(
          'network requires input and output dimensions',
          { cause },
        );

        // Assert
        expect({
          message: error.message,
          name: error.name,
          cause: error.cause,
        }).toEqual({
          message: 'network requires input and output dimensions',
          name: 'NetworkConstructorDimensionRequiredError',
          cause,
        });
      });
    });
  });
});
