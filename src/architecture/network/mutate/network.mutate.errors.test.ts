import {
  NetworkMutateMethodRequiredError,
  NetworkMutateRecurrentLayerOutputInitializationError,
} from './network.mutate.errors';

describe('network mutate errors chapter', () => {
  describe('NetworkMutateMethodRequiredError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('missing mutation method');

        // Act
        const mutateError = new NetworkMutateMethodRequiredError(
          'A mutation method is required.',
          { cause: rootCause },
        );

        // Assert
        expect({
          cause: mutateError.cause,
          message: mutateError.message,
          name: mutateError.name,
        }).toEqual({
          cause: rootCause,
          message: 'A mutation method is required.',
          name: 'NetworkMutateMethodRequiredError',
        });
      });
    });
  });

  describe('NetworkMutateRecurrentLayerOutputInitializationError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('missing recurrent output nodes');

        // Act
        const mutateError =
          new NetworkMutateRecurrentLayerOutputInitializationError(
            'The recurrent layer output nodes could not be initialized.',
            { cause: rootCause },
          );

        // Assert
        expect({
          cause: mutateError.cause,
          message: mutateError.message,
          name: mutateError.name,
        }).toEqual({
          cause: rootCause,
          message:
            'The recurrent layer output nodes could not be initialized.',
          name: 'NetworkMutateRecurrentLayerOutputInitializationError',
        });
      });
    });
  });
});