import {
  NetworkGatingNodeMembershipError,
  NetworkGatingRemovalNodeNotFoundError,
  NetworkGatingStructuralAnchorRemovalError,
} from './network.gating.errors';

describe('network gating errors chapter', () => {
  describe('NetworkGatingNodeMembershipError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('foreign node');

        // Act
        const gatingError = new NetworkGatingNodeMembershipError(
          'Node must belong to the network.',
          { cause: rootCause },
        );

        // Assert
        expect({
          cause: gatingError.cause,
          message: gatingError.message,
          name: gatingError.name,
        }).toEqual({
          cause: rootCause,
          message: 'Node must belong to the network.',
          name: 'NetworkGatingNodeMembershipError',
        });
      });
    });
  });

  describe('NetworkGatingStructuralAnchorRemovalError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('input node');

        // Act
        const gatingError = new NetworkGatingStructuralAnchorRemovalError(
          'Cannot remove an input or output node.',
          { cause: rootCause },
        );

        // Assert
        expect({
          cause: gatingError.cause,
          message: gatingError.message,
          name: gatingError.name,
        }).toEqual({
          cause: rootCause,
          message: 'Cannot remove an input or output node.',
          name: 'NetworkGatingStructuralAnchorRemovalError',
        });
      });
    });
  });

  describe('NetworkGatingRemovalNodeNotFoundError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('missing node');

        // Act
        const gatingError = new NetworkGatingRemovalNodeNotFoundError(
          'Node does not belong to the network.',
          { cause: rootCause },
        );

        // Assert
        expect({
          cause: gatingError.cause,
          message: gatingError.message,
          name: gatingError.name,
        }).toEqual({
          cause: rootCause,
          message: 'Node does not belong to the network.',
          name: 'NetworkGatingRemovalNodeNotFoundError',
        });
      });
    });
  });
});
