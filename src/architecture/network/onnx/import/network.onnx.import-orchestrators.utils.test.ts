import Node from '../../../node';
import Network from '../../network';
import type {
  OnnxMetadataProperty,
  OnnxModel,
  OnnxNode,
  OnnxTensor,
  Pool2DMapping,
} from '../schema/network.onnx.schema.types';
import {
  attachOnnxAdvancedGraphMetadata,
  attachOnnxPoolingMetadata,
  extractOnnxArchitecture,
  pruneSingleLayerHiddenPlaceholders,
  reconstructFusedRecurrentLayers,
  restoreResidualAddConnections,
  restoreRecurrentSelfConnections,
} from './network.onnx.import-orchestrators.utils';

type PoolingAwareNetwork = Network & {
  _onnxPooling?: {
    flattenConsistency?: {
      afterLayerIndex: number;
      consumerLayerIndex: number;
      consumerWidth: number;
      flattenedSize: number;
      matches: boolean;
    }[];
    layers: number[];
    specs: Pool2DMapping[];
    flattenLayers: number[];
    virtualShapes: {
      afterLayerIndex: number;
      flattenedSize?: number;
      inputChannels: number;
      inputHeight: number;
      inputWidth: number;
      outputChannels: number;
      outputHeight: number;
      outputWidth: number;
    }[];
  };
};

type AdvancedGraphAwareNetwork = Network & {
  _onnxAdvancedGraph?: {
    crossLayerConnections: {
      sourceNodeIndex: number;
      sourceLayerIndex: number;
      targetNodeIndex: number;
      targetLayerIndex: number;
      branchTensorName: string;
    }[];
    residualAdds?: {
      sourceLayerIndex: number;
      targetLayerIndex: number;
      branchTensorName: string;
      mergeNodeName: string;
      mergeOutputName: string;
    }[];
    sharedInitializerAliases?: {
      aliasTensorName: string;
      canonicalTensorName: string;
      initializerKind: string;
    }[];
    attentionBlocks?: {
      sourceLayerIndex: number;
      targetLayerIndex: number;
      sequenceLength: number;
      modelWidth: number;
      heads: number;
      shadowOutputName: string;
    }[];
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
    node?: OnnxNode[];
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
      node: options.node ?? [],
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

function createAttentionBlockMetadata(
  overrides: Partial<
    NonNullable<
      NonNullable<AdvancedGraphAwareNetwork['_onnxAdvancedGraph']>['attentionBlocks']
    >[number]
  > = {},
): OnnxMetadataProperty[] {
  return [
    {
      key: 'advanced_graph_attention_blocks',
      value: JSON.stringify([
        {
          sourceLayerIndex: 0,
          targetLayerIndex: 1,
          sequenceLength: 2,
          modelWidth: 4,
          heads: 2,
          shadowOutputName: 'AttentionShadow_1',
          ...overrides,
        },
      ]),
    },
  ];
}

function createAttentionSoftmaxNode(axis = -1): OnnxNode {
  return {
    op_type: 'Softmax',
    input: ['AttentionScaledScores_1'],
    output: ['AttentionProbabilities_1'],
    name: 'attention_l1_softmax',
    attributes: [{ name: 'axis', type: 'INT', i: axis }],
  };
}

function createAttentionOutputProjectionNode(
  weightTensorName = 'W0',
  biasTensorName = 'B0',
  shadowOutputName = 'AttentionShadow_1',
): OnnxNode {
  return {
    op_type: 'Gemm',
    input: ['AttentionContextFlat_1', weightTensorName, biasTensorName],
    output: [shadowOutputName],
    name: 'attention_l1_output_projection',
  };
}

function createAttentionShadowOnnxModel(
  options: {
    initializer?: OnnxTensor[];
    metadataProps?: OnnxMetadataProperty[];
    node?: OnnxNode[];
  } = {},
): OnnxModel {
  return createOnnxModel({
    initializer:
      options.initializer ??
      [
        createTensor('AttentionQW_l1', [4, 4], Array.from({ length: 16 }, () => 0)),
        createTensor('AttentionKW_l1', [4, 4], Array.from({ length: 16 }, () => 0)),
        createTensor('AttentionVW_l1', [4, 4], Array.from({ length: 16 }, () => 0)),
        createTensor('W0', [8, 8], Array.from({ length: 64 }, () => 0)),
        createTensor('B0', [8], Array.from({ length: 8 }, () => 0)),
      ],
    node:
      options.node ??
      [createAttentionSoftmaxNode(), createAttentionOutputProjectionNode()],
    metadataProps: options.metadataProps ?? createAttentionBlockMetadata(),
  });
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
      it('infers the recurrent target layer from the plain recurrent tensor name', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1);
        const onnxModel = createOnnxModel({
          initializer: [createTensor('R0', [2, 2], [0.5, 0, 0, 0.75])],
        });

        // Act
        restoreRecurrentSelfConnections(network, onnxModel, [2], []);

        // Assert
        expect(
          getHiddenNodes(network).map((hiddenNode) => ({
            selfCount: hiddenNode.connections.self.length,
            selfWeight: hiddenNode.connections.self[0]?.weight ?? null,
          })),
        ).toEqual([
          { selfCount: 1, selfWeight: 0.5 },
          { selfCount: 1, selfWeight: 0.75 },
        ]);
      });
    });

    describe('when recurrent metadata is absent but a fused LSTM tensor exists', () => {
      it('restores the recurrent gate slice from the fused recurrent tensor', () => {
        // Arrange
        const network = Network.createMLP(2, [10], 1);
        const onnxModel = createOnnxModel({
          initializer: [
            createTensor(
              'LSTM_R0',
              [8, 2],
              [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0.75, 0, 0, 0, 0],
            ),
          ],
        });

        // Act
        restoreRecurrentSelfConnections(network, onnxModel, [10], []);

        // Assert
        expect(
          getHiddenNodes(network)
            .slice(4, 6)
            .map((hiddenNode) => hiddenNode.connections.self[0]?.weight ?? null),
        ).toEqual([0.5, 0.75]);
      });
    });

    describe('when recurrent metadata is absent but the fused tensor has no usable unit size', () => {
      it('skips fused recurrent restoration', () => {
        // Arrange
        const network = Network.createMLP(2, [10], 1);
        const onnxModel = createOnnxModel({
          initializer: [createTensor('LSTM_R0', [], [])],
        });

        // Act
        restoreRecurrentSelfConnections(network, onnxModel, [10], []);

        // Assert
        expect(
          getHiddenNodes(network).map(
            (hiddenNode) => hiddenNode.connections.self.length,
          ),
        ).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      });
    });

    describe('when recurrent metadata is absent but the fused gate slice does not fit the hidden layer', () => {
      it('returns before attaching any fused recurrent self-connections', () => {
        // Arrange
        const network = Network.createMLP(2, [10], 1);
        const onnxModel = createOnnxModel({
          initializer: [createTensor('LSTM_R0', [8, 4], new Array(32).fill(0))],
        });

        // Act
        restoreRecurrentSelfConnections(network, onnxModel, [10], []);

        // Assert
        expect(
          getHiddenNodes(network).map(
            (hiddenNode) => hiddenNode.connections.self.length,
          ),
        ).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      });
    });

    describe('when recurrent metadata is absent and the available tensors do not target a live hidden layer', () => {
      it('ignores non-matching and out-of-range recurrent tensor names', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1);
        const onnxModel = createOnnxModel({
          initializer: [
            createTensor('W0', [2, 2], [0, 0, 0, 0]),
            createTensor('R4', [1, 1], [0.9]),
          ],
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

    describe('when recurrent metadata is absent and inferred recurrent tensors are out of order', () => {
      it('sorts the inferred layer indices before restoring self-connections', () => {
        // Arrange
        const network = Network.createMLP(1, [1, 1], 1);
        const onnxModel = createOnnxModel({
          initializer: [
            createTensor('R1', [1, 1], [0.9]),
            createTensor('R0', [1, 1], [0.6]),
          ],
        });

        // Act
        restoreRecurrentSelfConnections(network, onnxModel, [1, 1], []);

        // Assert
        expect(
          getHiddenNodes(network).map((hiddenNode) => hiddenNode.connections.self[0]?.weight ?? null),
        ).toEqual([0.6, 0.9]);
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
        expect(network._onnxPooling).toEqual({
          flattenLayers: [],
          layers: [1],
          specs: [],
          virtualShapes: [],
        });
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
          flattenLayers: [],
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
          virtualShapes: [],
        });
      });
    });

    describe('when Conv and flatten metadata align with the pooling site', () => {
      it('records one virtual pooled shape for later flatten consistency checks', () => {
        // Arrange
        const network = Network.createMLP(4, [4], 1) as PoolingAwareNetwork;

        // Act
        attachOnnxPoolingMetadata(network, [
          { key: 'pool2d_layers', value: '[1]' },
          {
            key: 'pool2d_specs',
            value:
              '[{"afterLayerIndex":1,"type":"MaxPool","kernelHeight":2,"kernelWidth":2,"strideHeight":2,"strideWidth":2}]',
          },
          {
            key: 'conv2d_specs',
            value:
              '[{"layerIndex":1,"inHeight":3,"inWidth":3,"inChannels":1,"kernelHeight":2,"kernelWidth":2,"strideHeight":1,"strideWidth":1,"outHeight":2,"outWidth":2,"outChannels":1}]',
          },
          { key: 'flatten_layers', value: '[1]' },
        ]);

        // Assert
        expect(network._onnxPooling).toEqual({
          flattenLayers: [1],
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
          virtualShapes: [
            {
              afterLayerIndex: 1,
              flattenedSize: 1,
              inputChannels: 1,
              inputHeight: 2,
              inputWidth: 2,
              outputChannels: 1,
              outputHeight: 1,
              outputWidth: 1,
            },
          ],
        });
      });
    });

    describe('when Conv metadata aligns but no flatten metadata is present', () => {
      it('records the virtual pooled shape without a flattened-size hint', () => {
        // Arrange
        const network = Network.createMLP(4, [4], 1) as PoolingAwareNetwork;

        // Act
        attachOnnxPoolingMetadata(network, [
          { key: 'pool2d_layers', value: '[1]' },
          {
            key: 'pool2d_specs',
            value:
              '[{"afterLayerIndex":1,"type":"AveragePool","kernelHeight":2,"kernelWidth":2,"strideHeight":2,"strideWidth":2}]',
          },
          {
            key: 'conv2d_specs',
            value:
              '[{"layerIndex":1,"inHeight":3,"inWidth":3,"inChannels":1,"kernelHeight":2,"kernelWidth":2,"strideHeight":1,"strideWidth":1,"outHeight":2,"outWidth":2,"outChannels":1}]',
          },
        ]);

        // Assert
        expect(network._onnxPooling).toEqual({
          flattenLayers: [],
          layers: [1],
          specs: [
            {
              afterLayerIndex: 1,
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 2,
              strideWidth: 2,
              type: 'AveragePool',
            },
          ],
          virtualShapes: [
            {
              afterLayerIndex: 1,
              inputChannels: 1,
              inputHeight: 2,
              inputWidth: 2,
              outputChannels: 1,
              outputHeight: 1,
              outputWidth: 1,
            },
          ],
        });
      });
    });

    describe('when flatten metadata is malformed and only inferred Conv specs are usable', () => {
      it('ignores the malformed flatten payload and still derives the virtual pooled shape', () => {
        // Arrange
        const network = Network.createMLP(4, [4], 1) as PoolingAwareNetwork;

        // Act
        attachOnnxPoolingMetadata(network, [
          { key: 'pool2d_layers', value: '[1]' },
          {
            key: 'pool2d_specs',
            value:
              '[{"afterLayerIndex":1,"type":"MaxPool","kernelHeight":2,"kernelWidth":2,"strideHeight":2,"strideWidth":2}]',
          },
          { key: 'conv2d_specs', value: '{}' },
          {
            key: 'conv2d_inferred_specs',
            value:
              '[{"layerIndex":1,"inHeight":3,"inWidth":3,"inChannels":1,"kernelHeight":2,"kernelWidth":2,"strideHeight":1,"strideWidth":1,"outHeight":2,"outWidth":2,"outChannels":1,"note":"heuristic"}]',
          },
          { key: 'flatten_layers', value: '{' },
        ]);

        // Assert
        expect(network._onnxPooling).toEqual({
          flattenLayers: [],
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
          virtualShapes: [
            {
              afterLayerIndex: 1,
              inputChannels: 1,
              inputHeight: 2,
              inputWidth: 2,
              outputChannels: 1,
              outputHeight: 1,
              outputWidth: 1,
            },
          ],
        });
      });
    });

    describe('when explicit Conv metadata is malformed JSON but inferred Conv specs remain available', () => {
      it('falls back to the inferred Conv specs for virtual shape derivation', () => {
        // Arrange
        const network = Network.createMLP(4, [4], 1) as PoolingAwareNetwork;

        // Act
        attachOnnxPoolingMetadata(network, [
          { key: 'pool2d_layers', value: '[1]' },
          {
            key: 'pool2d_specs',
            value:
              '[{"afterLayerIndex":1,"type":"MaxPool","kernelHeight":2,"kernelWidth":2,"strideHeight":2,"strideWidth":2}]',
          },
          { key: 'conv2d_specs', value: '{' },
          {
            key: 'conv2d_inferred_specs',
            value:
              '[{"layerIndex":1,"inHeight":3,"inWidth":3,"inChannels":1,"kernelHeight":2,"kernelWidth":2,"strideHeight":1,"strideWidth":1,"outHeight":2,"outWidth":2,"outChannels":1}]',
          },
        ]);

        // Assert
        expect(network._onnxPooling).toEqual({
          flattenLayers: [],
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
          virtualShapes: [
            {
              afterLayerIndex: 1,
              inputChannels: 1,
              inputHeight: 2,
              inputWidth: 2,
              outputChannels: 1,
              outputHeight: 1,
              outputWidth: 1,
            },
          ],
        });
      });
    });

    describe('when flatten metadata parses to a non-array payload', () => {
      it('normalizes the flatten layer list to empty while keeping the virtual pooled shape', () => {
        // Arrange
        const network = Network.createMLP(4, [4], 1) as PoolingAwareNetwork;

        // Act
        attachOnnxPoolingMetadata(network, [
          { key: 'pool2d_layers', value: '[1]' },
          {
            key: 'pool2d_specs',
            value:
              '[{"afterLayerIndex":1,"type":"MaxPool","kernelHeight":2,"kernelWidth":2,"strideHeight":2,"strideWidth":2}]',
          },
          {
            key: 'conv2d_specs',
            value:
              '[{"layerIndex":1,"inHeight":3,"inWidth":3,"inChannels":1,"kernelHeight":2,"kernelWidth":2,"strideHeight":1,"strideWidth":1,"outHeight":2,"outWidth":2,"outChannels":1}]',
          },
          { key: 'flatten_layers', value: '{}' },
        ]);

        // Assert
        expect(network._onnxPooling).toEqual({
          flattenLayers: [],
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
          virtualShapes: [
            {
              afterLayerIndex: 1,
              inputChannels: 1,
              inputHeight: 2,
              inputWidth: 2,
              outputChannels: 1,
              outputHeight: 1,
              outputWidth: 1,
            },
          ],
        });
      });
    });

    describe('when flatten metadata and layer sizes are available for a matching next dense width', () => {
      it('records a passing flatten-consistency audit entry', () => {
        // Arrange
        const network = Network.createMLP(4, [4], 1) as PoolingAwareNetwork;

        // Act
        attachOnnxPoolingMetadata(network, [
          { key: 'pool2d_layers', value: '[1]' },
          {
            key: 'pool2d_specs',
            value:
              '[{"afterLayerIndex":1,"type":"MaxPool","kernelHeight":2,"kernelWidth":2,"strideHeight":2,"strideWidth":2}]',
          },
          {
            key: 'conv2d_specs',
            value:
              '[{"layerIndex":1,"inHeight":3,"inWidth":3,"inChannels":1,"kernelHeight":2,"kernelWidth":2,"strideHeight":1,"strideWidth":1,"outHeight":2,"outWidth":2,"outChannels":1}]',
          },
          { key: 'flatten_layers', value: '[1]' },
          { key: 'layer_sizes', value: '[4]' },
        ]);

        // Assert
        expect(network._onnxPooling).toEqual({
          flattenConsistency: [
            {
              afterLayerIndex: 1,
              consumerLayerIndex: 2,
              consumerWidth: 1,
              flattenedSize: 1,
              matches: true,
            },
          ],
          flattenLayers: [1],
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
          virtualShapes: [
            {
              afterLayerIndex: 1,
              flattenedSize: 1,
              inputChannels: 1,
              inputHeight: 2,
              inputWidth: 2,
              outputChannels: 1,
              outputHeight: 1,
              outputWidth: 1,
            },
          ],
        });
      });
    });

    describe('when flatten metadata and layer sizes are available for a mismatched next dense width', () => {
      it('records a failing flatten-consistency audit entry', () => {
        // Arrange
        const network = Network.createMLP(4, [4], 2) as PoolingAwareNetwork;

        // Act
        attachOnnxPoolingMetadata(network, [
          { key: 'pool2d_layers', value: '[1]' },
          {
            key: 'pool2d_specs',
            value:
              '[{"afterLayerIndex":1,"type":"MaxPool","kernelHeight":2,"kernelWidth":2,"strideHeight":2,"strideWidth":2}]',
          },
          {
            key: 'conv2d_specs',
            value:
              '[{"layerIndex":1,"inHeight":3,"inWidth":3,"inChannels":1,"kernelHeight":2,"kernelWidth":2,"strideHeight":1,"strideWidth":1,"outHeight":2,"outWidth":2,"outChannels":1}]',
          },
          { key: 'flatten_layers', value: '[1]' },
          { key: 'layer_sizes', value: '[4]' },
        ]);

        // Assert
        expect(network._onnxPooling).toEqual({
          flattenConsistency: [
            {
              afterLayerIndex: 1,
              consumerLayerIndex: 2,
              consumerWidth: 2,
              flattenedSize: 1,
              matches: false,
            },
          ],
          flattenLayers: [1],
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
          virtualShapes: [
            {
              afterLayerIndex: 1,
              flattenedSize: 1,
              inputChannels: 1,
              inputHeight: 2,
              inputWidth: 2,
              outputChannels: 1,
              outputHeight: 1,
              outputWidth: 1,
            },
          ],
        });
      });
    });

    describe('when flatten metadata feeds another hidden layer', () => {
      it('audits against the next hidden-layer width rather than the output width', () => {
        // Arrange
        const network = Network.createMLP(4, [4, 3], 1) as PoolingAwareNetwork;

        // Act
        attachOnnxPoolingMetadata(network, [
          { key: 'pool2d_layers', value: '[1]' },
          {
            key: 'pool2d_specs',
            value:
              '[{"afterLayerIndex":1,"type":"MaxPool","kernelHeight":2,"kernelWidth":2,"strideHeight":2,"strideWidth":2}]',
          },
          {
            key: 'conv2d_specs',
            value:
              '[{"layerIndex":1,"inHeight":3,"inWidth":3,"inChannels":1,"kernelHeight":2,"kernelWidth":2,"strideHeight":1,"strideWidth":1,"outHeight":2,"outWidth":2,"outChannels":1}]',
          },
          { key: 'flatten_layers', value: '[1]' },
          { key: 'layer_sizes', value: '[4,3]' },
        ]);

        // Assert
        expect(network._onnxPooling).toEqual({
          flattenConsistency: [
            {
              afterLayerIndex: 1,
              consumerLayerIndex: 2,
              consumerWidth: 3,
              flattenedSize: 1,
              matches: false,
            },
          ],
          flattenLayers: [1],
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
          virtualShapes: [
            {
              afterLayerIndex: 1,
              flattenedSize: 1,
              inputChannels: 1,
              inputHeight: 2,
              inputWidth: 2,
              outputChannels: 1,
              outputHeight: 1,
              outputWidth: 1,
            },
          ],
        });
      });
    });

    describe('when multiple inferred Conv specs arrive out of order', () => {
      it('still derives virtual shapes for each matching pooling site', () => {
        // Arrange
        const network = Network.createMLP(4, [4, 1], 1) as PoolingAwareNetwork;

        // Act
        attachOnnxPoolingMetadata(network, [
          { key: 'pool2d_layers', value: '[1,2]' },
          {
            key: 'pool2d_specs',
            value:
              '[{"afterLayerIndex":1,"type":"MaxPool","kernelHeight":2,"kernelWidth":2,"strideHeight":2,"strideWidth":2},{"afterLayerIndex":2,"type":"AveragePool","kernelHeight":1,"kernelWidth":1,"strideHeight":1,"strideWidth":1}]',
          },
          {
            key: 'conv2d_inferred_specs',
            value:
              '[{"layerIndex":2,"inHeight":2,"inWidth":2,"inChannels":1,"kernelHeight":1,"kernelWidth":1,"strideHeight":1,"strideWidth":1,"outHeight":2,"outWidth":2,"outChannels":1},{"layerIndex":1,"inHeight":3,"inWidth":3,"inChannels":1,"kernelHeight":2,"kernelWidth":2,"strideHeight":1,"strideWidth":1,"outHeight":2,"outWidth":2,"outChannels":1}]',
          },
        ]);

        // Assert
        expect(network._onnxPooling).toEqual({
          flattenLayers: [],
          layers: [1, 2],
          specs: [
            {
              afterLayerIndex: 1,
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 2,
              strideWidth: 2,
              type: 'MaxPool',
            },
            {
              afterLayerIndex: 2,
              kernelHeight: 1,
              kernelWidth: 1,
              strideHeight: 1,
              strideWidth: 1,
              type: 'AveragePool',
            },
          ],
          virtualShapes: [
            {
              afterLayerIndex: 1,
              inputChannels: 1,
              inputHeight: 2,
              inputWidth: 2,
              outputChannels: 1,
              outputHeight: 1,
              outputWidth: 1,
            },
            {
              afterLayerIndex: 2,
              inputChannels: 1,
              inputHeight: 2,
              inputWidth: 2,
              outputChannels: 1,
              outputHeight: 2,
              outputWidth: 2,
            },
          ],
        });
      });
    });

    describe('when the pooling stride metadata is unusable', () => {
      it('skips the virtual pooled shape while keeping the raw pooling metadata', () => {
        // Arrange
        const network = Network.createMLP(4, [4], 1) as PoolingAwareNetwork;

        // Act
        attachOnnxPoolingMetadata(network, [
          { key: 'pool2d_layers', value: '[1]' },
          {
            key: 'pool2d_specs',
            value:
              '[{"afterLayerIndex":1,"type":"MaxPool","kernelHeight":2,"kernelWidth":2,"strideHeight":0,"strideWidth":0}]',
          },
          {
            key: 'conv2d_specs',
            value:
              '[{"layerIndex":1,"inHeight":3,"inWidth":3,"inChannels":1,"kernelHeight":2,"kernelWidth":2,"strideHeight":1,"strideWidth":1,"outHeight":2,"outWidth":2,"outChannels":1}]',
          },
        ]);

        // Assert
        expect(network._onnxPooling).toEqual({
          flattenLayers: [],
          layers: [1],
          specs: [
            {
              afterLayerIndex: 1,
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 0,
              strideWidth: 0,
              type: 'MaxPool',
            },
          ],
          virtualShapes: [],
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

  describe('attachOnnxAdvancedGraphMetadata()', () => {
    describe('when cross-layer connection metadata is absent', () => {
      it('leaves the imported network without advanced-graph metadata', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1) as AdvancedGraphAwareNetwork;

        // Act
        attachOnnxAdvancedGraphMetadata(network, []);

        // Assert
        expect(network._onnxAdvancedGraph).toBeUndefined();
      });
    });

    describe('when cross-layer connection metadata is valid JSON', () => {
      it('attaches the parsed advanced-graph audit payload', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1) as AdvancedGraphAwareNetwork;

        // Act
        attachOnnxAdvancedGraphMetadata(network, [
          {
            key: 'advanced_graph_cross_layer_connections',
            value:
              '[{"sourceNodeIndex":0,"sourceLayerIndex":0,"targetNodeIndex":4,"targetLayerIndex":2,"branchTensorName":"Branch_l0_to_l2_from_n0_to_n4"}]',
          },
        ]);

        // Assert
        expect(network._onnxAdvancedGraph).toEqual({
          crossLayerConnections: [
            {
              sourceNodeIndex: 0,
              sourceLayerIndex: 0,
              targetNodeIndex: 4,
              targetLayerIndex: 2,
              branchTensorName: 'Branch_l0_to_l2_from_n0_to_n4',
            },
          ],
        });
      });
    });

    describe('when cross-layer connection metadata parses to a non-array payload', () => {
      it('leaves the imported network without advanced-graph metadata', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1) as AdvancedGraphAwareNetwork;

        // Act
        attachOnnxAdvancedGraphMetadata(network, [
          {
            key: 'advanced_graph_cross_layer_connections',
            value: '{"sourceNodeIndex":0}',
          },
        ]);

        // Assert
        expect(network._onnxAdvancedGraph).toBeUndefined();
      });
    });

    describe('when cross-layer connection metadata contains an invalid record', () => {
      it('rejects the payload and leaves the imported network unchanged', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1) as AdvancedGraphAwareNetwork;

        // Act
        attachOnnxAdvancedGraphMetadata(network, [
          {
            key: 'advanced_graph_cross_layer_connections',
            value: '[null]',
          },
        ]);

        // Assert
        expect(network._onnxAdvancedGraph).toBeUndefined();
      });
    });

    describe('when cross-layer connection metadata is malformed JSON', () => {
      it('swallows the parse failure and leaves the imported network unchanged', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1) as AdvancedGraphAwareNetwork;

        // Act
        attachOnnxAdvancedGraphMetadata(network, [
          {
            key: 'advanced_graph_cross_layer_connections',
            value: '{',
          },
        ]);

        // Assert
        expect(network._onnxAdvancedGraph).toBeUndefined();
      });
    });

    describe('when shared initializer alias metadata is valid JSON', () => {
      it('attaches the parsed shared-initializer audit payload', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1) as AdvancedGraphAwareNetwork;

        // Act
        attachOnnxAdvancedGraphMetadata(network, [
          {
            key: 'shared_initializer_aliases',
            value:
              '[{"aliasTensorName":"W1","canonicalTensorName":"W0","initializerKind":"dense_weight"}]',
          },
        ]);

        // Assert
        expect(network._onnxAdvancedGraph).toEqual({
          sharedInitializerAliases: [
            {
              aliasTensorName: 'W1',
              canonicalTensorName: 'W0',
              initializerKind: 'dense_weight',
            },
          ],
        });
      });
    });

    describe('when shared initializer alias metadata parses to a non-array payload', () => {
      it('leaves the imported network without advanced-graph metadata', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1) as AdvancedGraphAwareNetwork;

        // Act
        attachOnnxAdvancedGraphMetadata(network, [
          {
            key: 'shared_initializer_aliases',
            value: '{}',
          },
        ]);

        // Assert
        expect(network._onnxAdvancedGraph).toBeUndefined();
      });
    });

    describe('when shared initializer alias metadata contains an invalid record', () => {
      it('rejects the payload and leaves the imported network unchanged', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1) as AdvancedGraphAwareNetwork;

        // Act
        attachOnnxAdvancedGraphMetadata(network, [
          {
            key: 'shared_initializer_aliases',
            value: '[null]',
          },
        ]);

        // Assert
        expect(network._onnxAdvancedGraph).toBeUndefined();
      });
    });

    describe('when shared initializer alias metadata is malformed JSON', () => {
      it('swallows the parse failure and leaves the imported network unchanged', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1) as AdvancedGraphAwareNetwork;

        // Act
        attachOnnxAdvancedGraphMetadata(network, [
          {
            key: 'shared_initializer_aliases',
            value: '{',
          },
        ]);

        // Assert
        expect(network._onnxAdvancedGraph).toBeUndefined();
      });
    });

    describe('when residual-add metadata contains an invalid record', () => {
      it('rejects the payload and leaves the imported network unchanged', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1) as AdvancedGraphAwareNetwork;

        // Act
        attachOnnxAdvancedGraphMetadata(network, [
          {
            key: 'advanced_graph_residual_adds',
            value: '[null]',
          },
        ]);

        // Assert
        expect(network._onnxAdvancedGraph).toBeUndefined();
      });
    });

    describe('when residual-add metadata is valid JSON', () => {
      it('attaches the parsed residual-add audit payload', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1) as AdvancedGraphAwareNetwork;

        // Act
        attachOnnxAdvancedGraphMetadata(network, [
          {
            key: 'advanced_graph_residual_adds',
            value: JSON.stringify([
              {
                sourceLayerIndex: 0,
                targetLayerIndex: 2,
                branchTensorName: 'ResidualBranch_l0_to_l2',
                mergeNodeName: 'residual_add_l2',
                mergeOutputName: 'ResidualAdd_2',
              },
            ]),
          },
        ]);

        // Assert
        expect(network._onnxAdvancedGraph).toEqual({
          residualAdds: [
            {
              sourceLayerIndex: 0,
              targetLayerIndex: 2,
              branchTensorName: 'ResidualBranch_l0_to_l2',
              mergeNodeName: 'residual_add_l2',
              mergeOutputName: 'ResidualAdd_2',
            },
          ],
        });
      });
    });

    describe('when attention metadata matches the exported fixed-width shadow subset', () => {
      it('attaches the parsed attention-block audit payload', () => {
        // Arrange
        const network = Network.createMLP(8, [8], 2) as AdvancedGraphAwareNetwork;
        const onnxModel = createAttentionShadowOnnxModel();

        // Act
        attachOnnxAdvancedGraphMetadata(
          network,
          onnxModel.metadata_props ?? [],
          onnxModel,
        );

        // Assert
        expect(network._onnxAdvancedGraph).toEqual({
          attentionBlocks: [
            {
              sourceLayerIndex: 0,
              targetLayerIndex: 1,
              sequenceLength: 2,
              modelWidth: 4,
              heads: 2,
              shadowOutputName: 'AttentionShadow_1',
            },
          ],
        });
      });
    });

    describe('when attention metadata arrives without the ONNX graph context', () => {
      it('leaves the imported network without advanced-graph metadata', () => {
        // Arrange
        const network = Network.createMLP(8, [8], 2) as AdvancedGraphAwareNetwork;

        // Act
        attachOnnxAdvancedGraphMetadata(network, createAttentionBlockMetadata());

        // Assert
        expect(network._onnxAdvancedGraph).toBeUndefined();
      });
    });

    describe('when attention metadata parses to a non-array payload', () => {
      it('leaves the imported network without advanced-graph metadata', () => {
        // Arrange
        const network = Network.createMLP(8, [8], 2) as AdvancedGraphAwareNetwork;
        const onnxModel = createAttentionShadowOnnxModel({
          metadataProps: [
            {
              key: 'advanced_graph_attention_blocks',
              value: '{}',
            },
          ],
        });

        // Act
        attachOnnxAdvancedGraphMetadata(
          network,
          onnxModel.metadata_props ?? [],
          onnxModel,
        );

        // Assert
        expect(network._onnxAdvancedGraph).toBeUndefined();
      });
    });

    describe('when attention metadata contains an invalid record', () => {
      it('rejects the payload and leaves the imported network unchanged', () => {
        // Arrange
        const network = Network.createMLP(8, [8], 2) as AdvancedGraphAwareNetwork;
        const onnxModel = createAttentionShadowOnnxModel({
          metadataProps: [
            {
              key: 'advanced_graph_attention_blocks',
              value: '[null]',
            },
          ],
        });

        // Act
        attachOnnxAdvancedGraphMetadata(
          network,
          onnxModel.metadata_props ?? [],
          onnxModel,
        );

        // Assert
        expect(network._onnxAdvancedGraph).toBeUndefined();
      });
    });

    describe('when attention metadata is malformed JSON', () => {
      it('swallows the parse failure and leaves the imported network unchanged', () => {
        // Arrange
        const network = Network.createMLP(8, [8], 2) as AdvancedGraphAwareNetwork;
        const onnxModel = createAttentionShadowOnnxModel({
          metadataProps: [
            {
              key: 'advanced_graph_attention_blocks',
              value: '{',
            },
          ],
        });

        // Act
        attachOnnxAdvancedGraphMetadata(
          network,
          onnxModel.metadata_props ?? [],
          onnxModel,
        );

        // Assert
        expect(network._onnxAdvancedGraph).toBeUndefined();
      });
    });

    describe('when attention metadata declares a malformed head partition', () => {
      it('rejects the payload and leaves the imported network unchanged', () => {
        // Arrange
        const network = Network.createMLP(8, [8], 2) as AdvancedGraphAwareNetwork;
        const onnxModel = createAttentionShadowOnnxModel({
          initializer: [
            createTensor('AttentionQW_l1', [5, 5], Array.from({ length: 25 }, () => 0)),
            createTensor('AttentionKW_l1', [5, 5], Array.from({ length: 25 }, () => 0)),
            createTensor('AttentionVW_l1', [5, 5], Array.from({ length: 25 }, () => 0)),
            createTensor('W0', [8, 10], Array.from({ length: 80 }, () => 0)),
            createTensor('B0', [8], Array.from({ length: 8 }, () => 0)),
          ],
          metadataProps: createAttentionBlockMetadata({
            modelWidth: 5,
            heads: 2,
          }),
        });

        // Act
        attachOnnxAdvancedGraphMetadata(
          network,
          onnxModel.metadata_props ?? [],
          onnxModel,
        );

        // Assert
        expect(network._onnxAdvancedGraph).toBeUndefined();
      });
    });

    describe('when attention metadata drifts away from the exported softmax contract', () => {
      it('rejects the payload and leaves the imported network unchanged', () => {
        // Arrange
        const network = Network.createMLP(8, [8], 2) as AdvancedGraphAwareNetwork;
        const onnxModel = createAttentionShadowOnnxModel({
          node: [
            createAttentionSoftmaxNode(0),
            createAttentionOutputProjectionNode(),
          ],
        });

        // Act
        attachOnnxAdvancedGraphMetadata(
          network,
          onnxModel.metadata_props ?? [],
          onnxModel,
        );

        // Assert
        expect(network._onnxAdvancedGraph).toBeUndefined();
      });
    });

    describe('when the exported attention softmax omits its axis attribute entirely', () => {
      it('rejects the payload and leaves the imported network unchanged', () => {
        // Arrange
        const network = Network.createMLP(8, [8], 2) as AdvancedGraphAwareNetwork;
        const onnxModel = createAttentionShadowOnnxModel({
          node: [
            {
              op_type: 'Softmax',
              input: ['AttentionScaledScores_1'],
              output: ['AttentionProbabilities_1'],
              name: 'attention_l1_softmax',
            },
            createAttentionOutputProjectionNode(),
          ],
        });

        // Act
        attachOnnxAdvancedGraphMetadata(
          network,
          onnxModel.metadata_props ?? [],
          onnxModel,
        );

        // Assert
        expect(network._onnxAdvancedGraph).toBeUndefined();
      });
    });

    describe('when attention metadata drifts into cross-attention ancestry', () => {
      it('rejects the payload and leaves the imported network unchanged', () => {
        // Arrange
        const network = Network.createMLP(8, [8], 2) as AdvancedGraphAwareNetwork;
        const onnxModel = createAttentionShadowOnnxModel({
          metadataProps: createAttentionBlockMetadata({
            sourceLayerIndex: 2,
          }),
        });

        // Act
        attachOnnxAdvancedGraphMetadata(
          network,
          onnxModel.metadata_props ?? [],
          onnxModel,
        );

        // Assert
        expect(network._onnxAdvancedGraph).toBeUndefined();
      });
    });

    describe('when residual-add metadata parses to a non-array payload', () => {
      it('leaves the imported network without advanced-graph metadata', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1) as AdvancedGraphAwareNetwork;

        // Act
        attachOnnxAdvancedGraphMetadata(network, [
          {
            key: 'advanced_graph_residual_adds',
            value: '{}',
          },
        ]);

        // Assert
        expect(network._onnxAdvancedGraph).toBeUndefined();
      });
    });

    describe('when residual-add metadata is malformed JSON', () => {
      it('swallows the parse failure and leaves the imported network unchanged', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 1) as AdvancedGraphAwareNetwork;

        // Act
        attachOnnxAdvancedGraphMetadata(network, [
          {
            key: 'advanced_graph_residual_adds',
            value: '{',
          },
        ]);

        // Assert
        expect(network._onnxAdvancedGraph).toBeUndefined();
      });
    });
  });

  describe('restoreResidualAddConnections()', () => {
    describe('when a residual cross-layer record points to a missing node index', () => {
      it('skips the missing node record without adding a residual connection', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 2);
        const metadata = [
          {
            key: 'advanced_graph_cross_layer_connections',
            value:
              '[{"sourceNodeIndex":99,"sourceLayerIndex":0,"targetNodeIndex":4,"targetLayerIndex":2,"branchTensorName":"Branch_l0_to_l2_from_n0_to_n4"}]',
          },
          {
            key: 'advanced_graph_residual_adds',
            value:
              '[{"sourceLayerIndex":0,"targetLayerIndex":2,"branchTensorName":"ResidualBranch_l0_to_l2","mergeNodeName":"residual_add_l2","mergeOutputName":"ResidualAdd_2"}]',
          },
        ] satisfies OnnxMetadataProperty[];
        const onnx = createOnnxModel({
          initializer: [createTensor('ResidualW_l2', [2, 2], [0.75, 0, 0, -0.5])],
          metadataProps: metadata,
        });

        // Act
        restoreResidualAddConnections(network, onnx, [2], metadata);

        // Assert
        expect(
          network.nodes[4].connections.in.some(
            (connectionEntry) => connectionEntry.from === network.nodes[0],
          ),
        ).toBe(false);
      });
    });

    describe('when a residual cross-layer record points to nodes outside the declared layer pair', () => {
      it('skips the mismatched layer record without adding a residual connection', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 2);
        const metadata = [
          {
            key: 'advanced_graph_cross_layer_connections',
            value:
              '[{"sourceNodeIndex":2,"sourceLayerIndex":0,"targetNodeIndex":4,"targetLayerIndex":2,"branchTensorName":"Branch_l0_to_l2_from_n0_to_n4"}]',
          },
          {
            key: 'advanced_graph_residual_adds',
            value:
              '[{"sourceLayerIndex":0,"targetLayerIndex":2,"branchTensorName":"ResidualBranch_l0_to_l2","mergeNodeName":"residual_add_l2","mergeOutputName":"ResidualAdd_2"}]',
          },
        ] satisfies OnnxMetadataProperty[];
        const onnx = createOnnxModel({
          initializer: [createTensor('ResidualW_l2', [2, 2], [0.75, 0, 0, -0.5])],
          metadataProps: metadata,
        });

        // Act
        restoreResidualAddConnections(network, onnx, [2], metadata);

        // Assert
        expect(
          network.nodes[4].connections.in.some(
            (connectionEntry) => connectionEntry.from === network.nodes[0],
          ),
        ).toBe(false);
      });
    });

    describe('when the residual connection already exists on the runtime graph', () => {
      it('updates the existing connection weight instead of allocating a duplicate', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 2);
        network.nodes[0].connect(network.nodes[4], 0.1);
        const metadata = [
          {
            key: 'advanced_graph_cross_layer_connections',
            value:
              '[{"sourceNodeIndex":0,"sourceLayerIndex":0,"targetNodeIndex":4,"targetLayerIndex":2,"branchTensorName":"Branch_l0_to_l2_from_n0_to_n4"}]',
          },
          {
            key: 'advanced_graph_residual_adds',
            value:
              '[{"sourceLayerIndex":0,"targetLayerIndex":2,"branchTensorName":"ResidualBranch_l0_to_l2","mergeNodeName":"residual_add_l2","mergeOutputName":"ResidualAdd_2"}]',
          },
        ] satisfies OnnxMetadataProperty[];
        const onnx = createOnnxModel({
          initializer: [createTensor('ResidualW_l2', [2, 2], [0.75, 0, 0, -0.5])],
          metadataProps: metadata,
        });

        // Act
        restoreResidualAddConnections(network, onnx, [2], metadata);

        // Assert
        expect(
          network.nodes[4].connections.in.find(
            (connectionEntry) => connectionEntry.from === network.nodes[0],
          )?.weight,
        ).toBe(0.75);
      });
    });

    describe('when the residual branch tensor is missing from the ONNX payload', () => {
      it('skips restoration without adding the residual connection', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 2);
        const metadata = [
          {
            key: 'advanced_graph_cross_layer_connections',
            value:
              '[{"sourceNodeIndex":0,"sourceLayerIndex":0,"targetNodeIndex":4,"targetLayerIndex":2,"branchTensorName":"Branch_l0_to_l2_from_n0_to_n4"}]',
          },
          {
            key: 'advanced_graph_residual_adds',
            value:
              '[{"sourceLayerIndex":0,"targetLayerIndex":2,"branchTensorName":"ResidualBranch_l0_to_l2","mergeNodeName":"residual_add_l2","mergeOutputName":"ResidualAdd_2"}]',
          },
        ] satisfies OnnxMetadataProperty[];
        const onnx = createOnnxModel({ metadataProps: metadata });

        // Act
        restoreResidualAddConnections(network, onnx, [2], metadata);

        // Assert
        expect(
          network.nodes[4].connections.in.some(
            (connectionEntry) => connectionEntry.from === network.nodes[0],
          ),
        ).toBe(false);
      });
    });

    describe('when the residual branch tensor is shorter than the declared matrix entry', () => {
      it('falls back to zero for the missing residual weight slot', () => {
        // Arrange
        const network = Network.createMLP(2, [2], 2);
        const metadata = [
          {
            key: 'advanced_graph_cross_layer_connections',
            value:
              '[{"sourceNodeIndex":0,"sourceLayerIndex":0,"targetNodeIndex":4,"targetLayerIndex":2,"branchTensorName":"Branch_l0_to_l2_from_n0_to_n4"}]',
          },
          {
            key: 'advanced_graph_residual_adds',
            value:
              '[{"sourceLayerIndex":0,"targetLayerIndex":2,"branchTensorName":"ResidualBranch_l0_to_l2","mergeNodeName":"residual_add_l2","mergeOutputName":"ResidualAdd_2"}]',
          },
        ] satisfies OnnxMetadataProperty[];
        const onnx = createOnnxModel({
          initializer: [createTensor('ResidualW_l2', [2, 2], [])],
          metadataProps: metadata,
        });

        // Act
        restoreResidualAddConnections(network, onnx, [2], metadata);

        // Assert
        expect(
          network.nodes[4].connections.in.find(
            (connectionEntry) => connectionEntry.from === network.nodes[0],
          )?.weight,
        ).toBe(0);
      });
    });
  });
});
