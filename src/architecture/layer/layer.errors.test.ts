import {
  LayerInputSourceUnavailableError,
  LayerInputTargetUnavailableError,
  LayerMemoryInputBlockTypeError,
  LayerMemoryInputSizeMismatchError,
  LayerOutputConnectUnavailableError,
  LayerOutputGateUnavailableError,
  LayerSizeMismatchError,
} from './layer.errors';

describe('layer errors chapter', () => {
  describe('LayerSizeMismatchError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('target array mismatch');

        // Act
        const layerError = new LayerSizeMismatchError(
          'Array with values should be same as the amount of nodes!',
          { cause: rootCause },
        );

        // Assert
        expect({
          cause: layerError.cause,
          message: layerError.message,
          name: layerError.name,
        }).toEqual({
          cause: rootCause,
          message: 'Array with values should be same as the amount of nodes!',
          name: 'LayerSizeMismatchError',
        });
      });
    });
  });

  describe('LayerOutputConnectUnavailableError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('missing output group');

        // Act
        const layerError = new LayerOutputConnectUnavailableError(
          'Layer output is not defined. Cannot connect from this layer.',
          { cause: rootCause },
        );

        // Assert
        expect({
          cause: layerError.cause,
          message: layerError.message,
          name: layerError.name,
        }).toEqual({
          cause: rootCause,
          message: 'Layer output is not defined. Cannot connect from this layer.',
          name: 'LayerOutputConnectUnavailableError',
        });
      });
    });
  });

  describe('LayerOutputGateUnavailableError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('missing output group');

        // Act
        const layerError = new LayerOutputGateUnavailableError(
          'Layer output is not defined. Cannot gate from this layer.',
          { cause: rootCause },
        );

        // Assert
        expect({
          cause: layerError.cause,
          message: layerError.message,
          name: layerError.name,
        }).toEqual({
          cause: rootCause,
          message: 'Layer output is not defined. Cannot gate from this layer.',
          name: 'LayerOutputGateUnavailableError',
        });
      });
    });
  });

  describe('LayerInputTargetUnavailableError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('missing target output group');

        // Act
        const layerError = new LayerInputTargetUnavailableError(
          'Target layer output is not defined. Cannot wire input.',
          { cause: rootCause },
        );

        // Assert
        expect({
          cause: layerError.cause,
          message: layerError.message,
          name: layerError.name,
        }).toEqual({
          cause: rootCause,
          message: 'Target layer output is not defined. Cannot wire input.',
          name: 'LayerInputTargetUnavailableError',
        });
      });
    });
  });

  describe('LayerInputSourceUnavailableError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('missing source output group');

        // Act
        const layerError = new LayerInputSourceUnavailableError(
          'Source layer output is not defined. Cannot wire input.',
          { cause: rootCause },
        );

        // Assert
        expect({
          cause: layerError.cause,
          message: layerError.message,
          name: layerError.name,
        }).toEqual({
          cause: rootCause,
          message: 'Source layer output is not defined. Cannot wire input.',
          name: 'LayerInputSourceUnavailableError',
        });
      });
    });
  });

  describe('LayerMemoryInputBlockTypeError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('non-group memory input');

        // Act
        const layerError = new LayerMemoryInputBlockTypeError(
          'Memory layers require a group-like input block.',
          { cause: rootCause },
        );

        // Assert
        expect({
          cause: layerError.cause,
          message: layerError.message,
          name: layerError.name,
        }).toEqual({
          cause: rootCause,
          message: 'Memory layers require a group-like input block.',
          name: 'LayerMemoryInputBlockTypeError',
        });
      });
    });
  });

  describe('LayerMemoryInputSizeMismatchError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('memory block size mismatch');

        // Act
        const layerError = new LayerMemoryInputSizeMismatchError(
          'Memory input source and target blocks must have the same size.',
          { cause: rootCause },
        );

        // Assert
        expect({
          cause: layerError.cause,
          message: layerError.message,
          name: layerError.name,
        }).toEqual({
          cause: rootCause,
          message: 'Memory input source and target blocks must have the same size.',
          name: 'LayerMemoryInputSizeMismatchError',
        });
      });
    });
  });
});