import Node from '../../../../node';
import { NetworkOnnxRecurrentMixedActivationsUnsupportedError } from '../../network.onnx.errors';
import { emitLayerGraph } from './network.onnx.export-layer-graph.utils';
import type { LayerBuildContext } from '../network.onnx.export.types';
import type { OnnxModel } from '../../schema/network.onnx.schema.types';

jest.retryTimes(2, { logErrorsBeforeRetry: true });

/**
 * Create a minimal ONNX model for testing layer emission.
 *
 * @param overrides Partial model overrides.
 * @returns Test ONNX model.
 */
function createOnnxModel(overrides: Partial<OnnxModel> = {}): OnnxModel {
  return {
    graph: {
      inputs: [],
      outputs: [],
      initializer: [],
      node: [],
    },
    ...overrides,
  };
}

describe('network onnx export layer graph utility chapter', () => {
  describe('emitLayerGraph()', () => {
    describe('given a recurrent hidden layer with mixed activations and allowMixedActivations enabled', () => {
      it('throws NetworkOnnxRecurrentMixedActivationsUnsupportedError', () => {
        // Arrange
        const inputNode1 = new Node('input');
        const inputNode2 = new Node('input');

        // Create two hidden nodes with different activation functions
        const hiddenNode1 = new Node('hidden');
        const hiddenNode2 = new Node('hidden');

        // Create activation functions with different names
        const tanhFunc = ((x: number) => Math.tanh(x)) as ((
          x: number,
          derivate?: boolean,
        ) => number) & { name?: string };
        Object.defineProperty(tanhFunc, 'name', {
          value: 'tanh',
          writable: false,
        });

        const reluFunc = ((x: number) => Math.max(0, x)) as ((
          x: number,
          derivate?: boolean,
        ) => number) & { name?: string };
        Object.defineProperty(reluFunc, 'name', {
          value: 'relu',
          writable: false,
        });

        hiddenNode1.squash = tanhFunc;
        hiddenNode2.squash = reluFunc;

        // Connect inputs to hidden layer
        inputNode1.connect(hiddenNode1);
        inputNode2.connect(hiddenNode2);

        // Create output layer nodes
        const outputNode = new Node('output');
        hiddenNode1.connect(outputNode);
        hiddenNode2.connect(outputNode);

        const model = createOnnxModel();

        const context: LayerBuildContext = {
          model,
          layers: [
            [inputNode1, inputNode2],
            [hiddenNode1, hiddenNode2],
            [outputNode],
          ],
          layerIndex: 1,
          previousOutputName: 'input_layer',
          options: {
            allowMixedActivations: true,
          },
          recurrentLayerIndices: [1],
          batchDimension: false,
          legacyNodeOrdering: false,
        };

        // Act & Assert
        expect(() => {
          emitLayerGraph(context);
        }).toThrow(NetworkOnnxRecurrentMixedActivationsUnsupportedError);
      });
    });
  });
});
