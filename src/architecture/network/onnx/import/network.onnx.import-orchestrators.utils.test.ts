import Node from '../../../node';
import Network from '../../network';
import type {
  OnnxMetadataProperty,
  OnnxModel,
  OnnxTensor,
  Pool2DMapping,
} from '../schema/network.onnx.schema.types';
import {
  attachOnnxPoolingMetadata,
  extractOnnxArchitecture,
  pruneSingleLayerHiddenPlaceholders,
  reconstructFusedRecurrentLayers,
  restoreRecurrentSelfConnections,
} from './network.onnx.import-orchestrators.utils';

type PoolingAwareNetwork = Network & {
  _onnxPooling?: {
    layers: number[];
    specs: Pool2DMapping[];
  };
};

jest.retryTimes(2, { logErrorsBeforeRetry: true });

function createTensor(
  name: string,
  dims: number[],
  floatData: number[],
): OnnxTensor {
  return {
    name,
    data_type: 1,
    dims,
    float_data: floatData,
  };
}

function createOnnxModel(
  options: {
    initializer?: OnnxTensor[];
    inputCount?: number;
    metadataProps?: OnnxMetadataProperty[];
    outputCount?: number;
  } = {},
): OnnxModel {
  const inputCount = options.inputCount ?? 2;
  const outputCount = options.outputCount ?? 1;

  return {
    graph: {
      inputs: [createValueInfo('input', inputCount)],
      outputs: [createValueInfo('output', outputCount)],
      initializer: options.initializer ?? [],
      node: [],
    },
    metadata_props: options.metadataProps,
  };
}

function createValueInfo(name: string, lastDimension: number) {
  return {
    name,
    type: {
      tensor_type: {
        elem_type: 1,
        shape: {
          dim: [{ dim_param: 'batch' }, { dim_value: lastDimension }],
        },
      },
    },
  };
}

function getHiddenNodes(network: Network): Node[] {
  return network.nodes.filter(
    (nodeEntry): nodeEntry is Node => nodeEntry.type === 'hidden',
  );
}

function createRecurrentMetadata(value: string): OnnxMetadataProperty[] {
  return [{ key: 'recurrent_single_step', value }];
}

describe('network onnx import orchestrators utility chapter', () => {
  describe('extractOnnxArchitecture()', () => {
    describe('when the ONNX model omits metadata_props', () => {
      it('derives the terminal dimensions and hidden layers from the graph tensors', () => {
        // Arrange
        const onnxModel = createOnnxModel({
          initializer: [
            createTensor('W1', [3, 2], [0, 0, 0, 0, 0, 0]),
            createTensor('B0', [3], [0, 0, 0]),
            createTensor('W2', [1, 3], [0, 0, 0]),
            createTensor('B1', [1], [0]),
          ],
          inputCount: 2,
          outputCount: 1,
        });

        // Act
        const architecture = extractOnnxArchitecture(onnxModel);

        // Assert
        expect(architecture).toEqual({
          hiddenLayerSizes: [3],
          inputCount: 2,
          outputCount: 1,
        });
      });
    });

    describe('when metadata_props includes an explicit layer_sizes payload', () => {
      it('uses the metadata sizes while preserving the terminal dimensions', () => {
        // Arrange
        const onnxModel = createOnnxModel({
          initializer: [],
          inputCount: 4,
          metadataProps: [{ key: 'layer_sizes', value: '[5,2]' }],
          outputCount: 3,
        });

        // Act
        const architecture = extractOnnxArchitecture(onnxModel);

        // Assert
        expect(architecture).toEqual({
          hiddenLayerSizes: [5, 2],
          inputCount: 4,
          outputCount: 3,
        });
      });
    });
  });

  describe('pruneSingleLayerHiddenPlaceholders()', () => {
    describe('when the imported perceptron has no hidden layers', () => {
      it('keeps only the input and output boundary nodes', () => {
        // Arrange
        const network = {
          nodes: [
            new Node('input'),
            new Node('hidden'),
            new Node('hidden'),
            new Node('output'),
          ],
        } as Network;

        // Act
        pruneSingleLayerHiddenPlaceholders(network, []);

        // Assert
        expect(network.nodes.map((nodeEntry) => nodeEntry.type)).toEqual([
          'input',
          'output',
        ]);
      });
    });

    describe('when the imported architecture still has hidden layers', () => {
      it('leaves the full node list unchanged', () => {
        // Arrange
        const network = {
          nodes: [new Node('input'), new Node('hidden'), new Node('output')],
        } as Network;

        // Act
        pruneSingleLayerHiddenPlaceholders(network, [1]);

        // Assert
        expect(network.nodes.map((nodeEntry) => nodeEntry.type)).toEqual([
          'input',
          'hidden',
          'output',
        ]);
      });
    });
  });

  describe('restoreRecurrentSelfConnections()', () => {
    describe('when recurrent metadata is absent', () => {
      it('leaves the hidden nodes without self-connections', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1);
        const onnxModel = createOnnxModel({
          initializer: [createTensor('R0', [2, 2], [0.5, 0, 0, 0.75])],
        });

        // Act
        restoreRecurrentSelfConnections(network, onnxModel, [2], []);

        // Assert
        expect(
          getHiddenNodes(network).map(
            (hiddenNode) => hiddenNode.connections.self.length,
          ),
        ).toEqual([0, 0]);
      });
    });

    describe('when recurrent metadata is malformed JSON', () => {
      it('falls back to a non-matching layer index and skips restoration', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1);
        const onnxModel = createOnnxModel({
          initializer: [createTensor('R0', [2, 2], [0.5, 0, 0, 0.75])],
        });

        // Act
        restoreRecurrentSelfConnections(
          network,
          onnxModel,
          [2],
          createRecurrentMetadata('{'),
        );

        // Assert
        expect(
          getHiddenNodes(network).map(
            (hiddenNode) => hiddenNode.connections.self.length,
          ),
        ).toEqual([0, 0]);
      });
    });

    describe('when recurrent metadata parses to a non-array payload', () => {
      it('normalizes to a non-matching layer index and skips restoration', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1);
        const onnxModel = createOnnxModel({
          initializer: [createTensor('R0', [2, 2], [0.5, 0, 0, 0.75])],
        });

        // Act
        restoreRecurrentSelfConnections(
          network,
          onnxModel,
          [2],
          createRecurrentMetadata('{"layer":1}'),
        );

        // Assert
        expect(
          getHiddenNodes(network).map(
            (hiddenNode) => hiddenNode.connections.self.length,
          ),
        ).toEqual([0, 0]);
      });
    });

    describe('when recurrent metadata selects a layer but the R tensor is missing', () => {
      it('leaves the hidden nodes without self-connections', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1);
        const onnxModel = createOnnxModel({ initializer: [] });

        // Act
        restoreRecurrentSelfConnections(
          network,
          onnxModel,
          [2],
          createRecurrentMetadata('[1]'),
        );

        // Assert
        expect(
          getHiddenNodes(network).map(
            (hiddenNode) => hiddenNode.connections.self.length,
          ),
        ).toEqual([0, 0]);
      });
    });

    describe('when recurrent metadata selects only the second hidden layer', () => {
      it('restores the diagonal self-weight onto that layer only', () => {
        // Arrange
        const network = Network.createMLP(1, [1, 1], 1);
        const onnxModel = createOnnxModel({
          initializer: [createTensor('R1', [1, 1], [0.9])],
        });
        const hiddenNodes = getHiddenNodes(network);

        // Act
        restoreRecurrentSelfConnections(
          network,
          onnxModel,
          [1, 1],
          createRecurrentMetadata('[2]'),
        );

        // Assert
        expect({
          firstHiddenSelfCount: hiddenNodes[0].connections.self.length,
          secondHiddenInboundSelfCount: hiddenNodes[1].connections.in.filter(
            (connectionEntry) =>
              connectionEntry.from === hiddenNodes[1] &&
              connectionEntry.to === hiddenNodes[1],
          ).length,
          secondHiddenSelfWeight: hiddenNodes[1].connections.self[0]?.weight,
        }).toEqual({
          firstHiddenSelfCount: 0,
          secondHiddenInboundSelfCount: 1,
          secondHiddenSelfWeight: 0.9,
        });
      });
    });

    describe('when recurrent metadata selects the first hidden layer with no existing self-connections', () => {
      it('creates one self-connection per hidden node from the diagonal tensor weights', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1);
        const onnxModel = createOnnxModel({
          initializer: [createTensor('R0', [2, 2], [0.5, 0.2, 0.1, 0.75])],
        });
        const hiddenNodes = getHiddenNodes(network);

        // Act
        restoreRecurrentSelfConnections(
          network,
          onnxModel,
          [2],
          createRecurrentMetadata('[1]'),
        );

        // Assert
        expect(
          hiddenNodes.map((hiddenNode) => ({
            inboundSelfCount: hiddenNode.connections.in.filter(
              (connectionEntry) =>
                connectionEntry.from === hiddenNode &&
                connectionEntry.to === hiddenNode,
            ).length,
            outboundSelfCount: hiddenNode.connections.out.filter(
              (connectionEntry) =>
                connectionEntry.from === hiddenNode &&
                connectionEntry.to === hiddenNode,
            ).length,
            selfWeight: hiddenNode.connections.self[0]?.weight,
          })),
        ).toEqual([
          { inboundSelfCount: 1, outboundSelfCount: 1, selfWeight: 0.5 },
          { inboundSelfCount: 1, outboundSelfCount: 1, selfWeight: 0.75 },
        ]);
      });
    });

    describe('when recurrent metadata selects the first hidden layer with existing self-connections', () => {
      it('updates the existing self-connection weights without adding duplicates', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1);
        const hiddenNodes = getHiddenNodes(network);
        hiddenNodes[0].connect(hiddenNodes[0], -1);
        hiddenNodes[1].connect(hiddenNodes[1], -2);
        const onnxModel = createOnnxModel({
          initializer: [createTensor('R0', [2, 2], [0.25, 0, 0, 0.5])],
        });

        // Act
        restoreRecurrentSelfConnections(
          network,
          onnxModel,
          [2],
          createRecurrentMetadata('[1]'),
        );

        // Assert
        expect(
          hiddenNodes.map((hiddenNode) => ({
            selfCount: hiddenNode.connections.self.length,
            selfWeight: hiddenNode.connections.self[0]?.weight,
          })),
        ).toEqual([
          { selfCount: 1, selfWeight: 0.25 },
          { selfCount: 1, selfWeight: 0.5 },
        ]);
      });
    });
  });

  describe('reconstructFusedRecurrentLayers()', () => {
    describe('when fused recurrent metadata is absent', () => {
      it('leaves the existing network node count unchanged', () => {
        // Arrange
        const network = Network.createMLP(1, [1], 1);
        const onnxModel = createOnnxModel();
        const layerFactory = {
          gru: () => undefined,
          lstm: () => undefined,
        };

        // Act
        reconstructFusedRecurrentLayers(
          network,
          onnxModel,
          [1],
          layerFactory,
          [],
        );

        // Assert
        expect(network.nodes.length).toBe(3);
      });
    });
  });

  describe('attachOnnxPoolingMetadata()', () => {
    describe('when pool2d_layers metadata is absent', () => {
      it('leaves the imported network without pooling metadata', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1) as PoolingAwareNetwork;

        // Act
        attachOnnxPoolingMetadata(network, []);

        // Assert
        expect(network._onnxPooling).toBeUndefined();
      });
    });

    describe('when pool2d_layers metadata is valid and pool2d_specs is absent', () => {
      it('attaches the layer list with an empty specs payload', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1) as PoolingAwareNetwork;

        // Act
        attachOnnxPoolingMetadata(network, [
          { key: 'pool2d_layers', value: '[1]' },
        ]);

        // Assert
        expect(network._onnxPooling).toEqual({ layers: [1], specs: [] });
      });
    });

    describe('when both pooling metadata fields are valid', () => {
      it('attaches both the parsed layer list and the parsed specs payload', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1) as PoolingAwareNetwork;

        // Act
        attachOnnxPoolingMetadata(network, [
          { key: 'pool2d_layers', value: '[1]' },
          {
            key: 'pool2d_specs',
            value:
              '[{"afterLayerIndex":1,"type":"MaxPool","kernelHeight":2,"kernelWidth":2,"strideHeight":2,"strideWidth":2}]',
          },
        ]);

        // Assert
        expect(network._onnxPooling).toEqual({
          layers: [1],
          specs: [
            {
              afterLayerIndex: 1,
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 2,
              strideWidth: 2,
              type: 'MaxPool',
            },
          ],
        });
      });
    });

    describe('when pooling metadata JSON is malformed', () => {
      it('swallows the parse failure and leaves the imported network unchanged', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1) as PoolingAwareNetwork;

        // Act
        attachOnnxPoolingMetadata(network, [
          { key: 'pool2d_layers', value: '{' },
        ]);

        // Assert
        expect(network._onnxPooling).toBeUndefined();
      });
    });
  });
});
