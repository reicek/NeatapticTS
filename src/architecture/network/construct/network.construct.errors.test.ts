import {
  NetworkConstructAmbiguousNodeIdError,
  NetworkConstructCycleModeError,
  NetworkConstructDuplicateEdgeError,
  NetworkConstructInputNodeIncomingEdgeError,
  NetworkConstructIsolatedHiddenNodeError,
  NetworkConstructMissingReferencedNodeError,
  NetworkConstructNodeIdResolutionError,
  NetworkConstructNoInputNodesError,
  NetworkConstructNoOutputNodesError,
  NetworkConstructOutputNodeGatedConnectionError,
  NetworkConstructOutputNodeOutgoingEdgeError,
  NetworkConstructSelfEdgeError,
} from './network.construct.errors';

describe('network construct errors chapter', () => {
  describe('NetworkConstructNodeIdResolutionError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('unknown public node id');

        // Act
        const constructError = new NetworkConstructNodeIdResolutionError(
          'Could not resolve explicit construct node id "sensorLeft".',
          { cause: rootCause },
        );

        // Assert
        expect({
          cause: constructError.cause,
          message: constructError.message,
          name: constructError.name,
        }).toEqual({
          cause: rootCause,
          message: 'Could not resolve explicit construct node id "sensorLeft".',
          name: 'NetworkConstructNodeIdResolutionError',
        });
      });
    });
  });

  describe('NetworkConstructAmbiguousNodeIdError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('duplicate label');

        // Act
        const constructError = new NetworkConstructAmbiguousNodeIdError(
          'Node label "sharedInput" matched multiple nodes.',
          { cause: rootCause },
        );

        // Assert
        expect({
          cause: constructError.cause,
          message: constructError.message,
          name: constructError.name,
        }).toEqual({
          cause: rootCause,
          message: 'Node label "sharedInput" matched multiple nodes.',
          name: 'NetworkConstructAmbiguousNodeIdError',
        });
      });
    });
  });

  describe('NetworkConstructMissingReferencedNodeError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('edge target missing');

        // Act
        const constructError = new NetworkConstructMissingReferencedNodeError(
          'Construct parts referenced a node that was not provided.',
          { cause: rootCause },
        );

        // Assert
        expect({
          cause: constructError.cause,
          message: constructError.message,
          name: constructError.name,
        }).toEqual({
          cause: rootCause,
          message: 'Construct parts referenced a node that was not provided.',
          name: 'NetworkConstructMissingReferencedNodeError',
        });
      });
    });
  });

  describe('NetworkConstructDuplicateEdgeError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('duplicate source-target edge');

        // Act
        const constructError = new NetworkConstructDuplicateEdgeError(
          'Duplicate edges are not allowed during construction.',
          { cause: rootCause },
        );

        // Assert
        expect({
          cause: constructError.cause,
          message: constructError.message,
          name: constructError.name,
        }).toEqual({
          cause: rootCause,
          message: 'Duplicate edges are not allowed during construction.',
          name: 'NetworkConstructDuplicateEdgeError',
        });
      });
    });
  });

  describe('NetworkConstructSelfEdgeError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('self edge detected');

        // Act
        const constructError = new NetworkConstructSelfEdgeError(
          'Self edges are not allowed during construction.',
          { cause: rootCause },
        );

        // Assert
        expect({
          cause: constructError.cause,
          message: constructError.message,
          name: constructError.name,
        }).toEqual({
          cause: rootCause,
          message: 'Self edges are not allowed during construction.',
          name: 'NetworkConstructSelfEdgeError',
        });
      });
    });
  });

  describe('NetworkConstructIsolatedHiddenNodeError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('isolated hidden node');

        // Act
        const constructError = new NetworkConstructIsolatedHiddenNodeError(
          'Hidden nodes must remain connected during construction.',
          { cause: rootCause },
        );

        // Assert
        expect({
          cause: constructError.cause,
          message: constructError.message,
          name: constructError.name,
        }).toEqual({
          cause: rootCause,
          message: 'Hidden nodes must remain connected during construction.',
          name: 'NetworkConstructIsolatedHiddenNodeError',
        });
      });
    });
  });

  describe('NetworkConstructNoInputNodesError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('missing inputs');

        // Act
        const constructError = new NetworkConstructNoInputNodesError(
          'Construction could not identify any input nodes.',
          { cause: rootCause },
        );

        // Assert
        expect({
          cause: constructError.cause,
          message: constructError.message,
          name: constructError.name,
        }).toEqual({
          cause: rootCause,
          message: 'Construction could not identify any input nodes.',
          name: 'NetworkConstructNoInputNodesError',
        });
      });
    });
  });

  describe('NetworkConstructInputNodeIncomingEdgeError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('input has incoming edge');

        // Act
        const constructError = new NetworkConstructInputNodeIncomingEdgeError(
          'Public input nodes must not receive incoming edges.',
          { cause: rootCause },
        );

        // Assert
        expect({
          cause: constructError.cause,
          message: constructError.message,
          name: constructError.name,
        }).toEqual({
          cause: rootCause,
          message: 'Public input nodes must not receive incoming edges.',
          name: 'NetworkConstructInputNodeIncomingEdgeError',
        });
      });
    });
  });

  describe('NetworkConstructNoOutputNodesError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('missing outputs');

        // Act
        const constructError = new NetworkConstructNoOutputNodesError(
          'Construction could not identify any output nodes.',
          { cause: rootCause },
        );

        // Assert
        expect({
          cause: constructError.cause,
          message: constructError.message,
          name: constructError.name,
        }).toEqual({
          cause: rootCause,
          message: 'Construction could not identify any output nodes.',
          name: 'NetworkConstructNoOutputNodesError',
        });
      });
    });
  });

  describe('NetworkConstructOutputNodeOutgoingEdgeError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('output emits edges');

        // Act
        const constructError = new NetworkConstructOutputNodeOutgoingEdgeError(
          'Public output nodes must remain sinks when sink-only validation is enabled.',
          { cause: rootCause },
        );

        // Assert
        expect({
          cause: constructError.cause,
          message: constructError.message,
          name: constructError.name,
        }).toEqual({
          cause: rootCause,
          message:
            'Public output nodes must remain sinks when sink-only validation is enabled.',
          name: 'NetworkConstructOutputNodeOutgoingEdgeError',
        });
      });
    });
  });

  describe('NetworkConstructOutputNodeGatedConnectionError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('output gates connections');

        // Act
        const constructError =
          new NetworkConstructOutputNodeGatedConnectionError(
            'Public output nodes must not gate connections when sink-only validation is enabled.',
            { cause: rootCause },
          );

        // Assert
        expect({
          cause: constructError.cause,
          message: constructError.message,
          name: constructError.name,
        }).toEqual({
          cause: rootCause,
          message:
            'Public output nodes must not gate connections when sink-only validation is enabled.',
          name: 'NetworkConstructOutputNodeGatedConnectionError',
        });
      });
    });
  });

  describe('NetworkConstructCycleModeError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('cycle in acyclic mode');

        // Act
        const constructError = new NetworkConstructCycleModeError(
          'A cyclic graph cannot be compiled while acyclic mode is required.',
          { cause: rootCause },
        );

        // Assert
        expect({
          cause: constructError.cause,
          message: constructError.message,
          name: constructError.name,
        }).toEqual({
          cause: rootCause,
          message:
            'A cyclic graph cannot be compiled while acyclic mode is required.',
          name: 'NetworkConstructCycleModeError',
        });
      });
    });
  });
});
