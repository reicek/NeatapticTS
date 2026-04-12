import { NetworkBootstrapTopologyIntentConflictError } from './network.bootstrap.errors';

describe('network bootstrap errors chapter', () => {
  describe('NetworkBootstrapTopologyIntentConflictError', () => {
    describe('given topology intent conflicts with the legacy acyclic flag', () => {
      it('preserves the configured message, name, and cause', () => {
        // Arrange
        const cause = new Error('flag conflict');

        // Act
        const error = new NetworkBootstrapTopologyIntentConflictError(
          'topology intent conflicts with the legacy acyclic flag',
          { cause },
        );

        // Assert
        expect({
          message: error.message,
          name: error.name,
          cause: error.cause,
        }).toEqual({
          message: 'topology intent conflicts with the legacy acyclic flag',
          name: 'NetworkBootstrapTopologyIntentConflictError',
          cause,
        });
      });
    });
  });
});