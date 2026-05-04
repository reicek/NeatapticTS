import { NetworkSerializeInvalidJsonError } from './network.serialize.errors';

describe('network serialize errors chapter', () => {
  describe('NetworkSerializeInvalidJsonError', () => {
    describe('given the serializer receives an invalid root payload', () => {
      it('preserves the configured message, name, and cause', () => {
        // Arrange
        const cause = new Error('invalid payload');

        // Act
        const error = new NetworkSerializeInvalidJsonError(
          'network JSON root payload must be an object',
          { cause },
        );

        // Assert
        expect({
          message: error.message,
          name: error.name,
          cause: error.cause,
        }).toEqual({
          message: 'network JSON root payload must be an object',
          name: 'NetworkSerializeInvalidJsonError',
          cause,
        });
      });
    });
  });
});
