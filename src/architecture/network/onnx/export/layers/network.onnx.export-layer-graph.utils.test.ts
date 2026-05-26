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

/**
 * Create a named activation function for test scenarios.
 *
 * @param name Activation name.
 * @param implementation Activation implementation.
 * @returns Named activation function.
 */
function createNamedActivation(
  name: string,
  implementation: (value: number) => number,
): ((value: number, derivate?: boolean) => number) & { name?: string } {
  const activation = implementation as ((
    value: number,
    derivate?: boolean,
  ) => number) & { name?: string };
  Object.defineProperty(activation, 'name', {
    value: name,
    writable: false,
  });
  return activation;
}

describe('network onnx export layer graph utility chapter', () => {
  describe('emitLayerGraph()', () => {
    describe('given a recurrent hidden layer with one shared activation', () => {
      it('returns the recurrent layer output tensor name', () => {
        // Arrange
        const inputNode1 = new Node('input');
        const inputNode2 = new Node('input');
        const hiddenNode1 = new Node('hidden');
        const hiddenNode2 = new Node('hidden');
        const outputNode = new Node('output');
        const tanhActivation = createNamedActivation('tanh', (value) =>
          Math.tanh(value),
        );

        hiddenNode1.squash = tanhActivation;
        hiddenNode2.squash = tanhActivation;
        inputNode1.connect(hiddenNode1);
        inputNode2.connect(hiddenNode2);
        hiddenNode1.connect(outputNode);
        hiddenNode2.connect(outputNode);

        const context: LayerBuildContext = {
          model: createOnnxModel(),
          layers: [
            [inputNode1, inputNode2],
            [hiddenNode1, hiddenNode2],
            [outputNode],
          ],
          layerIndex: 1,
          previousOutputName: 'input_layer',
          layerOutputNamesByLayerIndex: new Map([[0, 'input_layer']]),
          options: {
            allowMixedActivations: true,
          },
          recurrentLayerIndices: [1],
          batchDimension: false,
          legacyNodeOrdering: false,
        };

        // Act
        const outputName = emitLayerGraph(context);

        // Assert
        expect(outputName).toBe('Layer_1');
      });
    });

    describe('given a non-recurrent hidden layer with mixed activations and allowMixedActivations enabled', () => {
      it('emits a Concat node for the per-neuron branch', () => {
        // Arrange
        const inputNode1 = new Node('input');
        const inputNode2 = new Node('input');
        const hiddenNode1 = new Node('hidden');
        const hiddenNode2 = new Node('hidden');
        const outputNode = new Node('output');

        hiddenNode1.squash = createNamedActivation('tanh', (value) =>
          Math.tanh(value),
        );
        hiddenNode2.squash = createNamedActivation('relu', (value) =>
          Math.max(0, value),
        );
        inputNode1.connect(hiddenNode1);
        inputNode2.connect(hiddenNode2);
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
          layerOutputNamesByLayerIndex: new Map([[0, 'input_layer']]),
          options: {
            allowMixedActivations: true,
          },
          recurrentLayerIndices: [],
          batchDimension: false,
          legacyNodeOrdering: false,
        };

        // Act
        emitLayerGraph(context);

        // Assert
        expect(model.graph.node.some((node) => node.op_type === 'Concat')).toBe(
          true,
        );
      });
    });

    describe('given a recurrent hidden layer with mixed activations and allowMixedActivations enabled', () => {
      it('throws NetworkOnnxRecurrentMixedActivationsUnsupportedError', () => {
        // Arrange
        const inputNode1 = new Node('input');
        const inputNode2 = new Node('input');

        // Create two hidden nodes with different activation functions
        const hiddenNode1 = new Node('hidden');
        const hiddenNode2 = new Node('hidden');

        // Create activation functions with different names
        const tanhFunc = createNamedActivation('tanh', (value) =>
          Math.tanh(value),
        );
        const reluFunc = createNamedActivation('relu', (value) =>
          Math.max(0, value),
        );

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
          layerOutputNamesByLayerIndex: new Map([[0, 'input_layer']]),
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

    describe('given a one-hop residual candidate but the source-output lookup is missing', () => {
      it('falls back to the standard dense branch without emitting Add', () => {
        // Arrange
        const inputNode1 = new Node('input');
        const inputNode2 = new Node('input');
        const hiddenNode1 = new Node('hidden');
        const hiddenNode2 = new Node('hidden');
        const outputNode1 = new Node('output');
        const outputNode2 = new Node('output');
        const model = createOnnxModel();

        inputNode1.connect(hiddenNode1);
        inputNode2.connect(hiddenNode2);
        hiddenNode1.connect(outputNode1);
        hiddenNode2.connect(outputNode2);
        inputNode1.connect(outputNode1, 0.75);

        const context: LayerBuildContext = {
          model,
          layers: [
            [inputNode1, inputNode2],
            [hiddenNode1, hiddenNode2],
            [outputNode1, outputNode2],
          ],
          layerIndex: 2,
          previousOutputName: 'Layer_1',
          layerOutputNamesByLayerIndex: new Map([[1, 'Layer_1']]),
          options: {
            includeMetadata: true,
          },
          recurrentLayerIndices: [],
          batchDimension: false,
          legacyNodeOrdering: false,
        };

        // Act
        emitLayerGraph(context);

        // Assert
        expect(model.graph.node.some((node) => node.op_type === 'Add')).toBe(
          false,
        );
      });
    });

    describe('given an explicit concat mapping with an unsupported input order', () => {
      it('falls back to the standard dense branch without emitting Concat', () => {
        // Arrange
        const inputNode1 = new Node('input');
        const inputNode2 = new Node('input');
        const hiddenNode1 = new Node('hidden');
        const hiddenNode2 = new Node('hidden');
        const outputNode = new Node('output');
        const model = createOnnxModel();

        inputNode1.connect(hiddenNode1);
        inputNode2.connect(hiddenNode2);
        hiddenNode1.connect(outputNode);
        hiddenNode2.connect(outputNode);

        const context: LayerBuildContext = {
          model,
          layers: [
            [inputNode1, inputNode2],
            [hiddenNode1, hiddenNode2],
            [outputNode],
          ],
          layerIndex: 2,
          previousOutputName: 'Layer_1',
          layerOutputNamesByLayerIndex: new Map([
            [0, 'input_layer'],
            [1, 'Layer_1'],
          ]),
          options: {
            concatMappings: [
              {
                sourceLayerIndex: 0,
                targetLayerIndex: 2,
                inputOrder: 'source_then_previous',
              },
            ],
          } as unknown as LayerBuildContext['options'],
          recurrentLayerIndices: [],
          batchDimension: false,
          legacyNodeOrdering: false,
        };

        // Act
        emitLayerGraph(context);

        // Assert
        expect(model.graph.node.some((node) => node.op_type === 'Concat')).toBe(
          false,
        );
      });
    });

    describe('given an explicit concat mapping but the source-output lookup is missing', () => {
      it('falls back to the standard dense branch without emitting Concat', () => {
        // Arrange
        const inputNode1 = new Node('input');
        const inputNode2 = new Node('input');
        const hiddenNode1 = new Node('hidden');
        const hiddenNode2 = new Node('hidden');
        const outputNode = new Node('output');
        const model = createOnnxModel();

        inputNode1.connect(hiddenNode1);
        inputNode2.connect(hiddenNode2);
        hiddenNode1.connect(outputNode);
        hiddenNode2.connect(outputNode);

        const context: LayerBuildContext = {
          model,
          layers: [
            [inputNode1, inputNode2],
            [hiddenNode1, hiddenNode2],
            [outputNode],
          ],
          layerIndex: 2,
          previousOutputName: 'Layer_1',
          layerOutputNamesByLayerIndex: new Map([[1, 'Layer_1']]),
          options: {
            concatMappings: [
              {
                sourceLayerIndex: 0,
                targetLayerIndex: 2,
              },
            ],
          },
          recurrentLayerIndices: [],
          batchDimension: false,
          legacyNodeOrdering: false,
        };

        // Act
        emitLayerGraph(context);

        // Assert
        expect(model.graph.node.some((node) => node.op_type === 'Concat')).toBe(
          false,
        );
      });
    });

    describe('given an explicit concat mapping with batch output and metadata disabled', () => {
      it('emits a batch-axis concat node without attaching advanced-graph metadata', () => {
        // Arrange
        const inputNode1 = new Node('input');
        const inputNode2 = new Node('input');
        const hiddenNode1 = new Node('hidden');
        const hiddenNode2 = new Node('hidden');
        const outputNode = new Node('output');
        const model = createOnnxModel();

        inputNode1.connect(hiddenNode1);
        inputNode2.connect(hiddenNode2);
        hiddenNode1.connect(outputNode);
        hiddenNode2.connect(outputNode);

        const context: LayerBuildContext = {
          model,
          layers: [
            [inputNode1, inputNode2],
            [hiddenNode1, hiddenNode2],
            [outputNode],
          ],
          layerIndex: 2,
          previousOutputName: 'Layer_1',
          layerOutputNamesByLayerIndex: new Map([
            [0, 'input_layer'],
            [1, 'Layer_1'],
          ]),
          options: {
            concatMappings: [
              {
                sourceLayerIndex: 0,
                targetLayerIndex: 2,
              },
            ],
          },
          recurrentLayerIndices: [],
          batchDimension: true,
          legacyNodeOrdering: false,
        };

        // Act
        emitLayerGraph(context);

        // Assert
        expect({
          concatAxis: model.graph.node.find((node) => node.op_type === 'Concat')
            ?.attributes?.[0]?.i,
          metadataProps: model.metadata_props,
        }).toEqual({
          concatAxis: 1,
          metadataProps: undefined,
        });
      });
    });
  });
});
