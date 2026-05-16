import Network from '../../network';
import Node from '../../../node';
import type {
  Conv2DMapping,
  OnnxMetadataProperty,
  OnnxModel,
} from '../schema/network.onnx.schema.types';
import type { OnnxExportOptions } from './network.onnx.export.types';
import {
  emitFusedRecurrentHeuristics,
  finalizeExportMetadata,
  isConvMappingWeightShared,
} from './network.onnx.export-postprocess.utils';

function createOnnxModel(
  metadataProperties: OnnxMetadataProperty[] = [],
): OnnxModel {
  return {
    metadata_props: metadataProperties,
    graph: {
      inputs: [],
      outputs: [],
      initializer: [],
      node: [],
    },
  };
}

function createLayer(
  nodeType: 'input' | 'hidden' | 'output',
  nodeCount: number,
): Node[] {
  return Array.from({ length: nodeCount }, () => new Node(nodeType));
}

function createConvMapping(
  overrides: Partial<Conv2DMapping> = {},
): Conv2DMapping {
  return {
    layerIndex: 1,
    inHeight: 1,
    inWidth: 1,
    inChannels: 1,
    kernelHeight: 1,
    kernelWidth: 1,
    strideHeight: 1,
    strideWidth: 1,
    outHeight: 1,
    outWidth: 1,
    outChannels: 1,
    ...overrides,
  };
}

function createPooledSecondLayerSharingScenario(): {
  layers: Node[][];
  firstConvSpec: Conv2DMapping;
  secondConvSpec: Conv2DMapping;
  poolOptions: OnnxExportOptions;
} {
  const network = Network.createMLP(25, [16, 4], 1);
  const inputNodes = network.nodes.filter((nodeEntry) => nodeEntry.type === 'input');
  const hiddenNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'hidden',
  );
  const outputNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'output',
  );
  const firstHiddenNodes = hiddenNodes.slice(0, 16);
  const secondHiddenNodes = hiddenNodes.slice(16);
  const firstKernelPattern = [0.04, -0.06, 0.08, 0.11];
  const secondKernelPattern = [0.07, -0.02, 0.05, 0.09];

  firstHiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
    const outputRow = Math.floor(hiddenNodeIndex / 4);
    const outputColumn = hiddenNodeIndex % 4;

    hiddenNode.connections.in.forEach((connectionEntry) => {
      connectionEntry.weight = 0;
    });

    for (let kernelRowIndex = 0; kernelRowIndex < 2; kernelRowIndex += 1) {
      for (
        let kernelColumnIndex = 0;
        kernelColumnIndex < 2;
        kernelColumnIndex += 1
      ) {
        const sourceIndex =
          (outputRow + kernelRowIndex) * 5 + outputColumn + kernelColumnIndex;
        const kernelIndex = kernelRowIndex * 2 + kernelColumnIndex;
        const sourceNode = inputNodes[sourceIndex];
        const matchingConnection = hiddenNode.connections.in.find(
          (connectionEntry) => connectionEntry.from === sourceNode,
        );

        if (matchingConnection) {
          matchingConnection.weight = firstKernelPattern[kernelIndex];
        }
      }
    }

    hiddenNode.bias = 0.025;
  });

  secondHiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
    const outputRow = Math.floor(hiddenNodeIndex / 2);
    const outputColumn = hiddenNodeIndex % 2;

    hiddenNode.connections.in.forEach((connectionEntry) => {
      connectionEntry.weight = 0;
    });

    for (let kernelRowIndex = 0; kernelRowIndex < 2; kernelRowIndex += 1) {
      for (
        let kernelColumnIndex = 0;
        kernelColumnIndex < 2;
        kernelColumnIndex += 1
      ) {
        const pooledIndex =
          (outputRow + kernelRowIndex) * 3 + outputColumn + kernelColumnIndex;
        const kernelIndex = kernelRowIndex * 2 + kernelColumnIndex;
        const sourceNode = firstHiddenNodes[pooledIndex];
        const matchingConnection = hiddenNode.connections.in.find(
          (connectionEntry) => connectionEntry.from === sourceNode,
        );

        if (matchingConnection) {
          matchingConnection.weight = secondKernelPattern[kernelIndex];
        }
      }
    }

    hiddenNode.bias = 0.035;
  });

  return {
    layers: [inputNodes, firstHiddenNodes, secondHiddenNodes, outputNodes],
    firstConvSpec: createConvMapping({
      inHeight: 5,
      inWidth: 5,
      kernelHeight: 2,
      kernelWidth: 2,
      layerIndex: 1,
      outHeight: 4,
      outWidth: 4,
      strideHeight: 1,
      strideWidth: 1,
    }),
    secondConvSpec: createConvMapping({
      inHeight: 3,
      inWidth: 3,
      kernelHeight: 2,
      kernelWidth: 2,
      layerIndex: 2,
      outHeight: 2,
      outWidth: 2,
      strideHeight: 1,
      strideWidth: 1,
    }),
    poolOptions: {
      conv2dMappings: [
        createConvMapping({
          inHeight: 5,
          inWidth: 5,
          kernelHeight: 2,
          kernelWidth: 2,
          layerIndex: 1,
          outHeight: 4,
          outWidth: 4,
          strideHeight: 1,
          strideWidth: 1,
        }),
      ],
      pool2dMappings: [
        {
          afterLayerIndex: 1,
          type: 'MaxPool',
          kernelHeight: 2,
          kernelWidth: 2,
          strideHeight: 1,
          strideWidth: 1,
        },
      ],
    },
  };
}

function getMetadataValue(model: OnnxModel, key: string): string | undefined {
  return model.metadata_props?.find((entry) => entry.key === key)?.value;
}

function getInitializerDims(
  model: OnnxModel,
  initializerName: string,
): number[] | undefined {
  return model.graph.initializer.find(
    (initializerEntry) => initializerEntry.name === initializerName,
  )?.dims;
}

function getInitializerFloatData(
  model: OnnxModel,
  initializerName: string,
): number[] | undefined {
  return model.graph.initializer.find(
    (initializerEntry) => initializerEntry.name === initializerName,
  )?.float_data;
}

function getFirstOperatorInput(
  model: OnnxModel,
  operatorType: string,
): string | undefined {
  return model.graph.node.find(
    (nodeEntry) => nodeEntry.op_type === operatorType,
  )?.input[0];
}

describe('network onnx export postprocess chapter', () => {
  describe('isConvMappingWeightShared', () => {
    describe('given the Conv mapping points at a missing current layer', () => {
      it('returns false', () => {
        // Arrange
        const layers = [createLayer('input', 1)];

        // Act
        const isWeightShared = isConvMappingWeightShared(
          layers,
          createConvMapping(),
        );

        // Assert
        expect(isWeightShared).toBe(false);
      });
    });

    describe('given pooled predecessor metadata does not match the requested second-layer input shape', () => {
      it('falls back to the default dense source layout and returns false', () => {
        // Arrange
        const scenario = createPooledSecondLayerSharingScenario();

        // Act
        const isWeightShared = isConvMappingWeightShared(
          scenario.layers,
          {
            ...scenario.secondConvSpec,
            inHeight: 1,
            inWidth: 9,
            kernelHeight: 1,
            kernelWidth: 2,
            outHeight: 1,
            outWidth: 4,
          },
          scenario.poolOptions,
        );

        // Assert
        expect(isWeightShared).toBe(false);
      });
    });

    describe('given pooled predecessor metadata has an invalid zero stride', () => {
      it('falls back to the default dense source layout and keeps the layer weight-shared', () => {
        // Arrange
        const scenario = createPooledSecondLayerSharingScenario();

        // Act
        const isWeightShared = isConvMappingWeightShared(
          scenario.layers,
          scenario.secondConvSpec,
          {
            ...scenario.poolOptions,
            pool2dMappings: [
              {
                afterLayerIndex: 1,
                type: 'MaxPool',
                kernelHeight: 2,
                kernelWidth: 2,
                strideHeight: 0,
                strideWidth: 0,
              },
            ],
          },
        );

        // Assert
        expect(isWeightShared).toBe(true);
      });
    });
  });

  describe('emitFusedRecurrentHeuristics', () => {
    describe('given the first hidden layer is GRU-sized', () => {
      it('uses the graph input name as the GRU previous-output input', () => {
        // Arrange
        const onnxModel = createOnnxModel();
        const layers = [
          createLayer('input', 1),
          createLayer('hidden', 8),
          createLayer('output', 1),
        ];

        // Act
        emitFusedRecurrentHeuristics(onnxModel, layers, true, 'input');

        // Assert
        expect(getFirstOperatorInput(onnxModel, 'GRU')).toBe('input');
      });
    });

    describe('given LSTM metadata already exists but contains malformed JSON', () => {
      it('replaces the malformed metadata payload with the emitted layer index', () => {
        // Arrange
        const onnxModel = createOnnxModel([
          { key: 'lstm_emitted_layers', value: 'not-json' },
        ]);
        const layers = [
          createLayer('input', 1),
          createLayer('hidden', 10),
          createLayer('output', 1),
        ];

        // Act
        emitFusedRecurrentHeuristics(onnxModel, layers, true, 'input');

        // Assert
        expect(getMetadataValue(onnxModel, 'lstm_emitted_layers')).toBe('[1]');
      });
    });

    describe('given LSTM metadata already contains the emitted layer index', () => {
      it('leaves the stored layer list unchanged', () => {
        // Arrange
        const onnxModel = createOnnxModel([
          { key: 'lstm_emitted_layers', value: '[1,"noise"]' },
        ]);
        const layers = [
          createLayer('input', 1),
          createLayer('hidden', 10),
          createLayer('output', 1),
        ];

        // Act
        emitFusedRecurrentHeuristics(onnxModel, layers, true, 'input');

        // Assert
        expect(getMetadataValue(onnxModel, 'lstm_emitted_layers')).toBe(
          '[1,"noise"]',
        );
      });
    });

    describe('given LSTM metadata stores a non-array JSON payload', () => {
      it('replaces the non-array payload with the emitted layer index list', () => {
        // Arrange
        const onnxModel = createOnnxModel([
          { key: 'lstm_emitted_layers', value: '{"layer":1}' },
        ]);
        const layers = [
          createLayer('input', 1),
          createLayer('hidden', 10),
          createLayer('output', 1),
        ];

        // Act
        emitFusedRecurrentHeuristics(onnxModel, layers, true, 'input');

        // Assert
        expect(getMetadataValue(onnxModel, 'lstm_emitted_layers')).toBe('[1]');
      });
    });

    describe('given the first hidden layer is missing from a sparse layer list', () => {
      it('still emits a later GRU layer using the generated previous output name', () => {
        // Arrange
        const onnxModel = createOnnxModel();
        const layers: Node[][] = [];
        layers[0] = createLayer('input', 1);
        layers[2] = createLayer('hidden', 8);
        layers[3] = createLayer('output', 1);
        layers.length = 4;

        // Act
        emitFusedRecurrentHeuristics(onnxModel, layers, true, 'input');

        // Assert
        expect(getFirstOperatorInput(onnxModel, 'GRU')).toBe('Layer_1');
      });
    });

    describe('given the input layer is missing but an LSTM-sized hidden layer exists', () => {
      it('emits the recurrent weight tensor with an empty previous-layer width', () => {
        // Arrange
        const onnxModel = createOnnxModel();
        const layers: Node[][] = [];
        layers[1] = createLayer('hidden', 10);
        layers[2] = createLayer('output', 1);
        layers.length = 3;

        // Act
        emitFusedRecurrentHeuristics(onnxModel, layers, true, 'input');

        // Assert
        expect(getInitializerDims(onnxModel, 'LSTM_W0')).toEqual([8, 0]);
      });
    });

    describe('given a GRU-sized hidden layer has self-connections', () => {
      it('stores the self-connection weights in the recurrent initializer payload', () => {
        // Arrange
        const onnxModel = createOnnxModel();
        const hiddenLayer = createLayer('hidden', 8);
        hiddenLayer.forEach((hiddenNode, hiddenNodeIndex) => {
          hiddenNode.connect(hiddenNode, 0.2 + hiddenNodeIndex * 0.01);
        });
        const layers = [
          createLayer('input', 1),
          hiddenLayer,
          createLayer('output', 1),
        ];

        // Act
        emitFusedRecurrentHeuristics(onnxModel, layers, true, 'input');

        // Assert
        expect(
          getInitializerFloatData(onnxModel, 'GRU_R0')?.some(
            (weightEntry) => weightEntry !== 0,
          ),
        ).toBe(true);
      });
    });
  });

  describe('finalizeExportMetadata', () => {
    describe('given recurrent layers are present in the exported model', () => {
      it('records recurrent single-step metadata', () => {
        // Arrange
        const onnxModel = createOnnxModel();
        const layers = [
          createLayer('input', 1),
          createLayer('hidden', 1),
          createLayer('output', 1),
        ];
        const exportOptions: OnnxExportOptions = {};

        // Act
        finalizeExportMetadata(
          onnxModel,
          layers,
          exportOptions,
          true,
          [1],
          [1],
        );

        // Assert
        expect(getMetadataValue(onnxModel, 'recurrent_single_step')).toBe(
          '[1]',
        );
      });
    });

    describe('given later dense initializers exactly match an earlier dense layer', () => {
      it('reuses the canonical initializer names and records alias metadata', () => {
        // Arrange
        const onnxModel = createOnnxModel();
        const layers = [
          createLayer('input', 2),
          createLayer('hidden', 2),
          createLayer('hidden', 2),
          createLayer('output', 1),
        ];
        const exportOptions: OnnxExportOptions = {};

        onnxModel.graph.initializer.push(
          { name: 'W0', data_type: 1, dims: [2, 2], float_data: [0.1, 0.2, 0.3, 0.4] },
          { name: 'B0', data_type: 1, dims: [2], float_data: [0.5, -0.6] },
          { name: 'W1', data_type: 1, dims: [2, 2], float_data: [0.1, 0.2, 0.3, 0.4] },
          { name: 'B1', data_type: 1, dims: [2], float_data: [0.5, -0.6] },
        );
        onnxModel.graph.node.push(
          {
            op_type: 'Gemm',
            input: ['input', 'W0', 'B0'],
            output: ['Layer_1'],
            name: 'gemm_l1',
          },
          {
            op_type: 'Gemm',
            input: ['Layer_1', 'W1', 'B1'],
            output: ['Layer_2'],
            name: 'gemm_l2',
          },
        );

        // Act
        finalizeExportMetadata(
          onnxModel,
          layers,
          exportOptions,
          true,
          [2, 2],
          [],
        );

        // Assert
        expect({
          aliasMetadata: getMetadataValue(onnxModel, 'shared_initializer_aliases'),
          initializerNames: onnxModel.graph.initializer.map(
            (initializerEntry) => initializerEntry.name,
          ),
          secondGemmInputs: onnxModel.graph.node[1].input,
        }).toEqual({
          aliasMetadata:
            '[{"aliasTensorName":"W1","canonicalTensorName":"W0","initializerKind":"dense_weight"},{"aliasTensorName":"B1","canonicalTensorName":"B0","initializerKind":"dense_bias"}]',
          initializerNames: ['W0', 'B0'],
          secondGemmInputs: ['Layer_1', 'W0', 'B0'],
        });
      });
    });

    describe('given later dense initializers are only near-equal to an earlier dense layer', () => {
      it('keeps the later dense initializer names distinct', () => {
        // Arrange
        const onnxModel = createOnnxModel();
        const layers = [
          createLayer('input', 2),
          createLayer('hidden', 2),
          createLayer('hidden', 2),
          createLayer('output', 1),
        ];
        const exportOptions: OnnxExportOptions = {};

        onnxModel.graph.initializer.push(
          { name: 'W0', data_type: 1, dims: [2, 2], float_data: [0.1, 0.2, 0.3, 0.4] },
          { name: 'B0', data_type: 1, dims: [2], float_data: [0.5, -0.6] },
          { name: 'W1', data_type: 1, dims: [2, 2], float_data: [0.1, 0.2, 0.3, 0.4000000001] },
          { name: 'B1', data_type: 1, dims: [2], float_data: [0.5, -0.6000000001] },
        );
        onnxModel.graph.node.push({
          op_type: 'Gemm',
          input: ['Layer_1', 'W1', 'B1'],
          output: ['Layer_2'],
          name: 'gemm_l2',
        });

        // Act
        finalizeExportMetadata(
          onnxModel,
          layers,
          exportOptions,
          true,
          [2, 2],
          [],
        );

        // Assert
        expect({
          aliasMetadata: getMetadataValue(onnxModel, 'shared_initializer_aliases'),
          initializerNames: onnxModel.graph.initializer.map(
            (initializerEntry) => initializerEntry.name,
          ),
        }).toEqual({
          aliasMetadata: undefined,
          initializerNames: ['W0', 'B0', 'W1', 'B1'],
        });
      });
    });

    describe('given later per-neuron initializers exactly match an earlier neuron payload', () => {
      it('reuses the canonical per-neuron initializer names and records alias metadata', () => {
        // Arrange
        const onnxModel = createOnnxModel();
        const layers = [
          createLayer('input', 2),
          createLayer('hidden', 2),
          createLayer('hidden', 2),
          createLayer('output', 1),
        ];
        const exportOptions: OnnxExportOptions = {};

        onnxModel.graph.initializer.push(
          { name: 'W0_n0', data_type: 1, dims: [1, 2], float_data: [0.1, 0.2] },
          { name: 'B0_n0', data_type: 1, dims: [1], float_data: [0.5] },
          { name: 'W1_n0', data_type: 1, dims: [1, 2], float_data: [0.1, 0.2] },
          { name: 'B1_n0', data_type: 1, dims: [1], float_data: [0.5] },
        );
        onnxModel.graph.node.push({
          op_type: 'Gemm',
          input: ['Layer_1', 'W1_n0', 'B1_n0'],
          output: ['Layer_2_n0'],
          name: 'gemm_l2_n0',
        });

        // Act
        finalizeExportMetadata(
          onnxModel,
          layers,
          exportOptions,
          true,
          [2, 2],
          [],
        );

        // Assert
        expect({
          aliasMetadata: getMetadataValue(onnxModel, 'shared_initializer_aliases'),
          initializerNames: onnxModel.graph.initializer.map(
            (initializerEntry) => initializerEntry.name,
          ),
        }).toEqual({
          aliasMetadata:
            '[{"aliasTensorName":"W1_n0","canonicalTensorName":"W0_n0","initializerKind":"per_neuron_weight"},{"aliasTensorName":"B1_n0","canonicalTensorName":"B0_n0","initializerKind":"per_neuron_bias"}]',
          initializerNames: ['W0_n0', 'B0_n0'],
        });
      });
    });

    describe('given conv-sharing validation points at a missing mapped layer', () => {
      it('skips conv-sharing metadata for the invalid mapping', () => {
        // Arrange
        const onnxModel = createOnnxModel();
        const layers = [createLayer('input', 1)];
        const exportOptions: OnnxExportOptions = {
          validateConvSharing: true,
          conv2dMappings: [createConvMapping()],
        };

        // Act
        finalizeExportMetadata(onnxModel, layers, exportOptions, true, [1], []);

        // Assert
        expect(getMetadataValue(onnxModel, 'conv2d_sharing_verified')).toBe(
          undefined,
        );
      });
    });

    describe('given conv-sharing mappings disappear after the validation guard passes', () => {
      it('falls back to an empty mapping list during the validation fold', () => {
        // Arrange
        const onnxModel = createOnnxModel();
        const layers = [createLayer('input', 1), createLayer('hidden', 1)];
        const convMapping = createConvMapping();
        let convMappingsAccessCount = 0;
        const exportOptions = {
          validateConvSharing: true,
          get conv2dMappings() {
            convMappingsAccessCount += 1;
            return convMappingsAccessCount <= 2 ? [convMapping] : undefined;
          },
        } as OnnxExportOptions;

        // Act
        finalizeExportMetadata(onnxModel, layers, exportOptions, true, [1], []);

        // Assert
        expect(getMetadataValue(onnxModel, 'conv2d_sharing_verified')).toBe(
          undefined,
        );
      });
    });

    describe('given conv-sharing validation sees a sparse current layer', () => {
      it('falls back to zero representative weights and still records a verified layer', () => {
        // Arrange
        const onnxModel = createOnnxModel();
        const previousLayerNodes: Node[] = [];
        const currentLayerNodes: Node[] = [];
        currentLayerNodes[1] = new Node('hidden');
        currentLayerNodes.length = 2;
        const layers = [previousLayerNodes, currentLayerNodes];
        const exportOptions: OnnxExportOptions = {
          validateConvSharing: true,
          conv2dMappings: [
            createConvMapping({
              inHeight: 1,
              inWidth: 2,
              kernelHeight: 1,
              kernelWidth: 1,
              outHeight: 1,
              outWidth: 2,
              outChannels: 1,
            }),
          ],
        };

        // Act
        finalizeExportMetadata(onnxModel, layers, exportOptions, true, [2], []);

        // Assert
        expect(getMetadataValue(onnxModel, 'conv2d_sharing_verified')).toBe(
          '[1]',
        );
      });
    });

    describe('given conv-sharing validation finds mismatched representative kernels', () => {
      it('records the mismatched layer metadata', () => {
        // Arrange
        const onnxModel = createOnnxModel();
        const sourceNetwork = Network.createMLP(2, [2], 1);
        const layers = [
          sourceNetwork.nodes.filter((nodeEntry) => nodeEntry.type === 'input'),
          sourceNetwork.nodes.filter((nodeEntry) => nodeEntry.type === 'hidden'),
          sourceNetwork.nodes.filter((nodeEntry) => nodeEntry.type === 'output'),
        ];
        const exportOptions: OnnxExportOptions = {
          validateConvSharing: true,
          conv2dMappings: [
            createConvMapping({
              inHeight: 1,
              inWidth: 2,
              kernelHeight: 1,
              kernelWidth: 1,
              outHeight: 1,
              outWidth: 2,
              outChannels: 1,
            }),
          ],
        };

        layers[1][0].connections.in[0].weight = 0.1;
        layers[1][0].connections.in[1].weight = 0;
        layers[1][1].connections.in[0].weight = 0;
        layers[1][1].connections.in[1].weight = 0.2;

        // Act
        finalizeExportMetadata(onnxModel, layers, exportOptions, true, [2], []);

        // Assert
        expect(getMetadataValue(onnxModel, 'conv2d_sharing_mismatch')).toBe(
          '[1]',
        );
      });
    });

    describe('given padded kernel coordinates extend past the available inputs', () => {
      it('treats the out-of-bounds kernel positions as consistent and still verifies the layer', () => {
        // Arrange
        const onnxModel = createOnnxModel();
        const layers = [[new Node('input')], [new Node('hidden')]];
        const exportOptions: OnnxExportOptions = {
          validateConvSharing: true,
          conv2dMappings: [
            createConvMapping({
              inHeight: 1,
              inWidth: 1,
              kernelHeight: 2,
              kernelWidth: 2,
              outHeight: 1,
              outWidth: 1,
              outChannels: 1,
            }),
          ],
        };

        // Act
        finalizeExportMetadata(onnxModel, layers, exportOptions, true, [3], []);

        // Assert
        expect(getMetadataValue(onnxModel, 'conv2d_sharing_verified')).toBe(
          '[1]',
        );
      });
    });

    describe('given conv-sharing validation expects more source positions than the previous layer provides', () => {
      it('falls back to zero input weights and still verifies the layer', () => {
        // Arrange
        const onnxModel = createOnnxModel();
        const layers = [[], [new Node('hidden')]];
        const exportOptions: OnnxExportOptions = {
          validateConvSharing: true,
          conv2dMappings: [
            createConvMapping({
              inHeight: 2,
              inWidth: 2,
              kernelHeight: 1,
              kernelWidth: 1,
              outHeight: 1,
              outWidth: 1,
              outChannels: 1,
            }),
          ],
        };

        // Act
        finalizeExportMetadata(onnxModel, layers, exportOptions, true, [4], []);

        // Assert
        expect(getMetadataValue(onnxModel, 'conv2d_sharing_verified')).toBe(
          '[1]',
        );
      });
    });
  });
});
