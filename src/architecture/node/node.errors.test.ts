import {
  NodeInvalidConnectionTargetTypeError,
  NodeMutationMethodRequiredError,
  NodeUndefinedConnectionTargetError,
  NodeUnknownMutationMethodError,
  NodeUnsupportedMutationMethodError,
} from './node.errors';

function captureNodeErrorSnapshot(error: Error): {
  message: string;
  name: string;
  cause: unknown;
} {
  return {
    message: error.message,
    name: error.name,
    cause: error.cause,
  };
}

describe('node errors chapter', () => {
  describe('mutation errors', () => {
    describe('given a mutation method is missing', () => {
      it('preserves the required-method error metadata', () => {
        // Arrange
        const cause = new Error('missing mutation');

        // Act
        const error = new NodeMutationMethodRequiredError(
          'node mutation requires a method',
          { cause },
        );

        // Assert
        expect(captureNodeErrorSnapshot(error)).toEqual({
          message: 'node mutation requires a method',
          name: 'NodeMutationMethodRequiredError',
          cause,
        });
      });
    });

    describe('given a mutation method name is unknown', () => {
      it('preserves the unknown-method error metadata', () => {
        // Arrange
        const cause = new Error('unknown method');

        // Act
        const error = new NodeUnknownMutationMethodError(
          'node mutation method is unknown',
          { cause },
        );

        // Assert
        expect(captureNodeErrorSnapshot(error)).toEqual({
          message: 'node mutation method is unknown',
          name: 'NodeUnknownMutationMethodError',
          cause,
        });
      });
    });

    describe('given a known mutation reaches an unsupported branch', () => {
      it('preserves the unsupported-method error metadata', () => {
        // Arrange
        const cause = new Error('unsupported branch');

        // Act
        const error = new NodeUnsupportedMutationMethodError(
          'node mutation method is unsupported',
          { cause },
        );

        // Assert
        expect(captureNodeErrorSnapshot(error)).toEqual({
          message: 'node mutation method is unsupported',
          name: 'NodeUnsupportedMutationMethodError',
          cause,
        });
      });
    });
  });

  describe('connection target errors', () => {
    describe('given a connection target is missing', () => {
      it('preserves the undefined-target error metadata', () => {
        // Arrange
        const cause = new Error('missing target');

        // Act
        const error = new NodeUndefinedConnectionTargetError(
          'node connection target is missing',
          { cause },
        );

        // Assert
        expect(captureNodeErrorSnapshot(error)).toEqual({
          message: 'node connection target is missing',
          name: 'NodeUndefinedConnectionTargetError',
          cause,
        });
      });
    });

    describe('given a connection target is neither node nor group-like', () => {
      it('preserves the invalid-target-type error metadata', () => {
        // Arrange
        const cause = new Error('invalid target type');

        // Act
        const error = new NodeInvalidConnectionTargetTypeError(
          'node connection target type is invalid',
          { cause },
        );

        // Assert
        expect(captureNodeErrorSnapshot(error)).toEqual({
          message: 'node connection target type is invalid',
          name: 'NodeInvalidConnectionTargetTypeError',
          cause,
        });
      });
    });
  });
});
