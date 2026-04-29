import {
  NetworkOnnxLayerOrderingUnresolvableError,
  NetworkOnnxMixedActivationsUnsupportedError,
  NetworkOnnxPartialConnectivityUnsupportedError,
  NetworkOnnxPerceptronSizeValidationError,
  NetworkOnnxRecurrentMixedActivationsUnsupportedError,
} from './network.onnx.errors';

describe('network onnx errors chapter', () => {
  describe('NetworkOnnxLayerOrderingUnresolvableError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('cycle detected');

        // Act
        const onnxError = new NetworkOnnxLayerOrderingUnresolvableError(
          'Layer ordering could not be resolved.',
          { cause: rootCause },
        );

        // Assert
        expect({
          cause: onnxError.cause,
          message: onnxError.message,
          name: onnxError.name,
        }).toEqual({
          cause: rootCause,
          message: 'Layer ordering could not be resolved.',
          name: 'NetworkOnnxLayerOrderingUnresolvableError',
        });
      });
    });
  });

  describe('NetworkOnnxMixedActivationsUnsupportedError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('heterogeneous hidden layer');

        // Act
        const onnxError = new NetworkOnnxMixedActivationsUnsupportedError(
          'Mixed activations require allowMixedActivations.',
          { cause: rootCause },
        );

        // Assert
        expect({
          cause: onnxError.cause,
          message: onnxError.message,
          name: onnxError.name,
        }).toEqual({
          cause: rootCause,
          message: 'Mixed activations require allowMixedActivations.',
          name: 'NetworkOnnxMixedActivationsUnsupportedError',
        });
      });
    });
  });

  describe('NetworkOnnxPartialConnectivityUnsupportedError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('missing edge');

        // Act
        const onnxError = new NetworkOnnxPartialConnectivityUnsupportedError(
          'Partial connectivity requires allowPartialConnectivity.',
          { cause: rootCause },
        );

        // Assert
        expect({
          cause: onnxError.cause,
          message: onnxError.message,
          name: onnxError.name,
        }).toEqual({
          cause: rootCause,
          message: 'Partial connectivity requires allowPartialConnectivity.',
          name: 'NetworkOnnxPartialConnectivityUnsupportedError',
        });
      });
    });
  });

  describe('NetworkOnnxPerceptronSizeValidationError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('metadata missing sizes');

        // Act
        const onnxError = new NetworkOnnxPerceptronSizeValidationError(
          'Perceptron metadata must include input and output sizes.',
          { cause: rootCause },
        );

        // Assert
        expect({
          cause: onnxError.cause,
          message: onnxError.message,
          name: onnxError.name,
        }).toEqual({
          cause: rootCause,
          message: 'Perceptron metadata must include input and output sizes.',
          name: 'NetworkOnnxPerceptronSizeValidationError',
        });
      });
    });
  });

  describe('NetworkOnnxRecurrentMixedActivationsUnsupportedError', () => {
    describe('when the error is created with a cause', () => {
      it('keeps the custom error name, message, and cause', () => {
        // Arrange
        const rootCause = new Error('recurrent gate mismatch');

        // Act
        const onnxError =
          new NetworkOnnxRecurrentMixedActivationsUnsupportedError(
            'Recurrent export does not support mixed activations.',
            { cause: rootCause },
          );

        // Assert
        expect({
          cause: onnxError.cause,
          message: onnxError.message,
          name: onnxError.name,
        }).toEqual({
          cause: rootCause,
          message: 'Recurrent export does not support mixed activations.',
          name: 'NetworkOnnxRecurrentMixedActivationsUnsupportedError',
        });
      });
    });
  });
});
