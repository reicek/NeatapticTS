import { NetworkOnnxPerceptronSizeValidationError } from '../network.onnx.errors';
import { loadRuntimeFactories } from './network.onnx.runtime-load.utils';

describe('network onnx runtime-load utility chapter', () => {
  describe('loadRuntimeFactories', () => {
    describe('given the runtime perceptron factory receives fewer than two layer sizes', () => {
      it('throws the perceptron-size validation error', () => {
        // Arrange
        const { perceptronFactory } = loadRuntimeFactories();
        const buildInvalidPerceptron = () => perceptronFactory(3);

        // Assert
        expect(buildInvalidPerceptron).toThrow(
          NetworkOnnxPerceptronSizeValidationError,
        );
      });
    });
  });
});