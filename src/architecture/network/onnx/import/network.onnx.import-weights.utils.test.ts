import Network from '../../network';
import * as methods from '../../../../methods/methods';
import { exportToONNX } from '../network.onnx';
import type { Conv2DMapping } from '../network.onnx';
import type {
  OnnxMetadataProperty,
  OnnxModel,
  OnnxTensor,
} from '../schema/network.onnx.schema.types';
import {
  assignWeightsAndBiases,
  deriveHiddenLayerSizes,
} from './network.onnx.import-weights.utils';

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

function getInputNodes(network: Network) {
  return network.nodes.filter((nodeEntry) => nodeEntry.type === 'input');
}

function getHiddenNodes(network: Network) {
  return network.nodes.filter((nodeEntry) => nodeEntry.type === 'hidden');
}

function readInboundWeight(
  targetNode: Network['nodes'][number],
  sourceNode: Network['nodes'][number],
): number | null {
  const matchingConnection = targetNode.connections.in.find(
    (connectionEntry) => connectionEntry.from === sourceNode,
  );
  return matchingConnection ? matchingConnection.weight : null;
}

function setHiddenLayerSentinelState(
  network: Network,
  weightValue: number,
  biasValue: number,
): void {
  getHiddenNodes(network).forEach((hiddenNode) => {
    hiddenNode.bias = biasValue;
    hiddenNode.connections.in.forEach((connectionEntry) => {
      connectionEntry.weight = weightValue;
    });
  });
}

function removeInboundConnection(
  targetNode: Network['nodes'][number],
  sourceNode: Network['nodes'][number],
  network: Network,
): void {
  const matchingConnection = targetNode.connections.in.find(
    (connectionEntry) => connectionEntry.from === sourceNode,
  );
  if (!matchingConnection) {
    return;
  }

  targetNode.connections.in = targetNode.connections.in.filter(
    (connectionEntry) => connectionEntry !== matchingConnection,
  );
  sourceNode.connections.out = sourceNode.connections.out.filter(
    (connectionEntry) => connectionEntry !== matchingConnection,
  );
  network.connections = network.connections.filter(
    (connectionEntry) => connectionEntry !== matchingConnection,
  );
}

function findMetadataEntry(
  metadataProps: OnnxMetadataProperty[] | undefined,
  key: string,
): OnnxMetadataProperty {
  const matchingEntry = metadataProps?.find(
    (property) => property.key === key,
  );

  if (!matchingEntry) {
    throw new Error(`Missing metadata entry: ${key}`);
  }

  return matchingEntry;
}

function cloneOnnxModel(onnxModel: OnnxModel): OnnxModel {
  return structuredClone(onnxModel) as OnnxModel;
}

function createConvSharingVerifiedScenario(): {
  network: Network;
  mappings: Conv2DMapping[];
  kernelPattern: number[];
} {
  const inputChannels = 1;
  const inputHeight = 3;
  const inputWidth = 3;
  const kernelHeight = 2;
  const kernelWidth = 2;
  const strideHeight = 1;
  const strideWidth = 1;
  const outputChannels = 1;
  const outputHeight = inputHeight - kernelHeight + 1;
  const outputWidth = inputWidth - kernelWidth + 1;
  const inputSize = inputChannels * inputHeight * inputWidth;
  const hiddenSize = outputChannels * outputHeight * outputWidth;
  const network = Network.createMLP(inputSize, [hiddenSize], 2);
  const inputNodes = getInputNodes(network);
  const hiddenNodes = getHiddenNodes(network);
  const kernelPattern = [0.11, -0.07, 0.05, 0.02];

  hiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
    const outputRow = Math.floor(hiddenNodeIndex / outputWidth);
    const outputColumn = hiddenNodeIndex % outputWidth;
    const inputBaseRow = outputRow * strideHeight;
    const inputBaseColumn = outputColumn * strideWidth;

    hiddenNode.connections.in.forEach((connectionEntry) => {
      connectionEntry.weight = 0;
    });

    for (let kernelRowIndex = 0; kernelRowIndex < kernelHeight; kernelRowIndex += 1) {
      for (
        let kernelColumnIndex = 0;
        kernelColumnIndex < kernelWidth;
        kernelColumnIndex += 1
      ) {
        const inputRow = inputBaseRow + kernelRowIndex;
        const inputColumn = inputBaseColumn + kernelColumnIndex;
        const sourceIndex = inputRow * inputWidth + inputColumn;
        const kernelIndex = kernelRowIndex * kernelWidth + kernelColumnIndex;
        const sourceNode = inputNodes[sourceIndex];
        const matchingConnection = hiddenNode.connections.in.find(
          (connectionEntry) => connectionEntry.from === sourceNode,
        );

        if (matchingConnection) {
          matchingConnection.weight = kernelPattern[kernelIndex];
        }
      }
    }

    hiddenNode.bias = 0.123;
  });

  return {
    network,
    mappings: [
      {
        layerIndex: 1,
        inHeight: inputHeight,
        inWidth: inputWidth,
        inChannels: inputChannels,
        kernelHeight,
        kernelWidth,
        strideHeight,
        strideWidth,
        outHeight: outputHeight,
        outWidth: outputWidth,
        outChannels: outputChannels,
      },
    ],
    kernelPattern,
  };
}

function createSecondLayerConvScenario(): {
  network: Network;
  mappings: Conv2DMapping[];
  kernelPattern: number[];
} {
  const network = Network.createMLP(4, [4, 1], 1);
  const hiddenNodes = getHiddenNodes(network);
  const secondLayerNode = hiddenNodes[4];
  const kernelPattern = [0.31, -0.22, 0.13, 0.07];

  secondLayerNode.connections.in.forEach((connectionEntry, connectionIndex) => {
    connectionEntry.weight = kernelPattern[connectionIndex];
  });
  secondLayerNode.bias = 0.25;

  return {
    network,
    mappings: [
      {
        layerIndex: 2,
        inHeight: 2,
        inWidth: 2,
        inChannels: 1,
        kernelHeight: 2,
        kernelWidth: 2,
        strideHeight: 1,
        strideWidth: 1,
        outHeight: 1,
        outWidth: 1,
        outChannels: 1,
      },
    ],
    kernelPattern,
  };
}

describe('network onnx import weights utility chapter', () => {
  describe('deriveHiddenLayerSizes', () => {
    describe('when metadata includes an explicit layer-size array', () => {
      it('returns the metadata sizes without reading initializer tensors', () => {
        // Arrange
        const initializers = [createTensor('W1', [9, 4], [0, 1, 2, 3])];
        const metadataProps: OnnxMetadataProperty[] = [
          { key: 'layer_sizes', value: '[4,5]' },
        ];

        // Act
        const hiddenLayerSizes = deriveHiddenLayerSizes(
          initializers,
          metadataProps,
        );

        // Assert
        expect(hiddenLayerSizes).toEqual([4, 5]);
      });
    });

    describe('when layer_sizes metadata is valid JSON but not an array', () => {
      it('falls back to aggregated and per-neuron weight tensors', () => {
        // Arrange
        const initializers = [
          createTensor('ignored', [1], [0]),
          createTensor('Wbad', [1], [0]),
          createTensor('W3', [1, 3], [0, 0, 0]),
          createTensor('W2_n0', [2], [0, 0]),
          createTensor('W1', [2, 4], Array(8).fill(0)),
          createTensor('W2_n1', [2], [0, 0]),
          createTensor('W2_n2', [2], [0, 0]),
        ];
        const metadataProps: OnnxMetadataProperty[] = [
          { key: 'layer_sizes', value: '{"hidden":true}' },
        ];

        // Act
        const hiddenLayerSizes = deriveHiddenLayerSizes(
          initializers,
          metadataProps,
        );

        // Assert
        expect(hiddenLayerSizes).toEqual([2, 3]);
      });
    });

    describe('when layer_sizes metadata is malformed JSON', () => {
      it('falls back to the discovered weight buckets', () => {
        // Arrange
        const initializers = [
          createTensor('W2', [1, 2], [0, 0]),
          createTensor('W1', [2, 3], Array(6).fill(0)),
        ];
        const metadataProps: OnnxMetadataProperty[] = [
          { key: 'layer_sizes', value: '{' },
        ];

        // Act
        const hiddenLayerSizes = deriveHiddenLayerSizes(
          initializers,
          metadataProps,
        );

        // Assert
        expect(hiddenLayerSizes).toEqual([2]);
      });
    });

    describe('when no weight tensors are present', () => {
      it('returns an empty hidden-layer list', () => {
        // Arrange
        const initializers = [createTensor('B1', [2], [0, 0])];

        // Act
        const hiddenLayerSizes = deriveHiddenLayerSizes(initializers);

        // Assert
        expect(hiddenLayerSizes).toEqual([]);
      });
    });
  });

  describe('assignWeightsAndBiases', () => {
    describe('when an aggregated layer is missing its bias tensor', () => {
      it('leaves that hidden layer on the target network untouched', () => {
        // Arrange
        const sourceNetwork = Network.createMLP(2, [2], 1);
        const onnxModel = exportToONNX(sourceNetwork);
        onnxModel.graph.initializer = onnxModel.graph.initializer.filter(
          (tensor) => tensor.name !== 'B0',
        );
        const targetNetwork = Network.createMLP(2, [2], 1);
        setHiddenLayerSentinelState(targetNetwork, -3, -7);

        // Act
        assignWeightsAndBiases(targetNetwork, onnxModel, [2], onnxModel.metadata_props);

        // Assert
        expect({
          firstHiddenBias: getHiddenNodes(targetNetwork)[0].bias,
          firstHiddenWeights: getHiddenNodes(targetNetwork)[0].connections.in.map(
            (connectionEntry) => connectionEntry.weight,
          ),
        }).toEqual({
          firstHiddenBias: -7,
          firstHiddenWeights: [-3, -3],
        });
      });
    });

    describe('when one aggregated inbound connection is absent on the target layer', () => {
      it('still assigns the remaining inbound weight and the target bias', () => {
        // Arrange
        const sourceNetwork = Network.createMLP(2, [2], 1);
        const sourceHiddenNodes = getHiddenNodes(sourceNetwork);
        sourceHiddenNodes[0].bias = 0.25;
        sourceHiddenNodes[0].connections.in[0].weight = 0.11;
        sourceHiddenNodes[0].connections.in[1].weight = 0.22;
        const onnxModel = exportToONNX(sourceNetwork);
        const targetNetwork = Network.createMLP(2, [2], 1);
        removeInboundConnection(
          getHiddenNodes(targetNetwork)[0],
          getInputNodes(targetNetwork)[0],
          targetNetwork,
        );

        // Act
        assignWeightsAndBiases(targetNetwork, onnxModel, [2], onnxModel.metadata_props);

        // Assert
        expect({
          firstHiddenBias: getHiddenNodes(targetNetwork)[0].bias,
          remainingInboundWeights: getHiddenNodes(targetNetwork)[0].connections.in.map(
            (connectionEntry) => connectionEntry.weight,
          ),
        }).toEqual({
          firstHiddenBias: 0.25,
          remainingInboundWeights: [0.22],
        });
      });
    });

    describe('when the ONNX payload uses per-neuron tensors for a mixed-activation hidden layer', () => {
      it('restores the hidden-layer weights and biases from the per-neuron tensors', () => {
        // Arrange
        const sourceNetwork = Network.createMLP(2, [2], 1);
        const sourceHiddenNodes = getHiddenNodes(sourceNetwork);
        sourceHiddenNodes[0].squash = methods.Activation.relu;
        sourceHiddenNodes[1].squash = methods.Activation.tanh;
        sourceHiddenNodes[0].bias = 0.4;
        sourceHiddenNodes[1].bias = -0.3;
        sourceHiddenNodes[0].connections.in[0].weight = 0.11;
        sourceHiddenNodes[0].connections.in[1].weight = 0.22;
        sourceHiddenNodes[1].connections.in[0].weight = 0.33;
        sourceHiddenNodes[1].connections.in[1].weight = 0.44;
        const onnxModel = exportToONNX(sourceNetwork, {
          allowMixedActivations: true,
        });
        const targetNetwork = Network.createMLP(2, [2], 1);
        setHiddenLayerSentinelState(targetNetwork, -1, -1);

        // Act
        assignWeightsAndBiases(
          targetNetwork,
          onnxModel,
          deriveHiddenLayerSizes(onnxModel.graph.initializer, onnxModel.metadata_props),
          onnxModel.metadata_props,
        );

        // Assert
        expect(
          getHiddenNodes(targetNetwork).map((hiddenNode) => ({
            bias: hiddenNode.bias,
            weights: hiddenNode.connections.in.map(
              (connectionEntry) => connectionEntry.weight,
            ),
          })),
        ).toEqual([
          { bias: 0.4, weights: [0.11, 0.22] },
          { bias: -0.3, weights: [0.33, 0.44] },
        ]);
      });
    });

    describe('when one per-neuron tensor pair is incomplete', () => {
      it('skips that neuron while still restoring the other per-neuron payloads', () => {
        // Arrange
        const sourceNetwork = Network.createMLP(2, [2], 1);
        const sourceHiddenNodes = getHiddenNodes(sourceNetwork);
        sourceHiddenNodes[0].squash = methods.Activation.relu;
        sourceHiddenNodes[1].squash = methods.Activation.tanh;
        sourceHiddenNodes[0].bias = 0.5;
        sourceHiddenNodes[1].bias = -0.6;
        sourceHiddenNodes[0].connections.in[0].weight = 0.15;
        sourceHiddenNodes[0].connections.in[1].weight = 0.25;
        sourceHiddenNodes[1].connections.in[0].weight = 0.35;
        sourceHiddenNodes[1].connections.in[1].weight = 0.45;
        const onnxModel = exportToONNX(sourceNetwork, {
          allowMixedActivations: true,
        });
        onnxModel.graph.initializer = onnxModel.graph.initializer.filter(
          (tensor) => tensor.name !== 'B0_n1',
        );
        const targetNetwork = Network.createMLP(2, [2], 1);
        setHiddenLayerSentinelState(targetNetwork, -8, -7);

        // Act
        assignWeightsAndBiases(
          targetNetwork,
          onnxModel,
          deriveHiddenLayerSizes(onnxModel.graph.initializer, onnxModel.metadata_props),
          onnxModel.metadata_props,
        );

        // Assert
        expect(
          getHiddenNodes(targetNetwork).map((hiddenNode) => ({
            bias: hiddenNode.bias,
            weights: hiddenNode.connections.in.map(
              (connectionEntry) => connectionEntry.weight,
            ),
          })),
        ).toEqual([
          { bias: 0.5, weights: [0.15, 0.25] },
          { bias: -7, weights: [-8, -8] },
        ]);
      });
    });

    describe('when one per-neuron inbound connection is absent on the target network', () => {
      it('still assigns the remaining inbound weight and the target bias', () => {
        // Arrange
        const sourceNetwork = Network.createMLP(2, [2], 1);
        const sourceHiddenNodes = getHiddenNodes(sourceNetwork);
        sourceHiddenNodes[0].squash = methods.Activation.relu;
        sourceHiddenNodes[1].squash = methods.Activation.tanh;
        sourceHiddenNodes[0].bias = 0.27;
        sourceHiddenNodes[0].connections.in[0].weight = 0.41;
        sourceHiddenNodes[0].connections.in[1].weight = 0.52;
        const onnxModel = exportToONNX(sourceNetwork, {
          allowMixedActivations: true,
        });
        const targetNetwork = Network.createMLP(2, [2], 1);
        removeInboundConnection(
          getHiddenNodes(targetNetwork)[0],
          getInputNodes(targetNetwork)[0],
          targetNetwork,
        );

        // Act
        assignWeightsAndBiases(
          targetNetwork,
          onnxModel,
          deriveHiddenLayerSizes(onnxModel.graph.initializer, onnxModel.metadata_props),
          onnxModel.metadata_props,
        );

        // Assert
        expect({
          firstHiddenBias: getHiddenNodes(targetNetwork)[0].bias,
          remainingInboundWeights: getHiddenNodes(targetNetwork)[0].connections.in.map(
            (connectionEntry) => connectionEntry.weight,
          ),
        }).toEqual({
          firstHiddenBias: 0.27,
          remainingInboundWeights: [0.52],
        });
      });
    });

    describe('when Conv metadata and tensors are present for a valid mapping', () => {
      it('reconstructs the receptive-field kernel weights and channel bias', () => {
        // Arrange
        const scenario = createConvSharingVerifiedScenario();
        const onnxModel = exportToONNX(scenario.network, {
          includeMetadata: true,
          conv2dMappings: scenario.mappings,
        });
        const targetNetwork = Network.createMLP(9, [4], 2);
        setHiddenLayerSentinelState(targetNetwork, -5, -5);

        // Act
        assignWeightsAndBiases(targetNetwork, onnxModel, [4], onnxModel.metadata_props);

        // Assert
        expect({
          firstHiddenBias: getHiddenNodes(targetNetwork)[0].bias,
          receptiveWeights: [0, 1, 3, 4].map((inputIndex) =>
            readInboundWeight(
              getHiddenNodes(targetNetwork)[0],
              getInputNodes(targetNetwork)[inputIndex],
            ),
          ),
        }).toEqual({
          firstHiddenBias: 0.123,
          receptiveWeights: scenario.kernelPattern,
        });
      });
    });

    describe('when Conv metadata targets a deeper hidden layer', () => {
      it('rebuilds the later hidden layer from the previous hidden-node slice', () => {
        // Arrange
        const scenario = createSecondLayerConvScenario();
        const onnxModel = exportToONNX(scenario.network, {
          includeMetadata: true,
          conv2dMappings: scenario.mappings,
        });
        const targetNetwork = Network.createMLP(4, [4, 1], 1);
        setHiddenLayerSentinelState(targetNetwork, -4, -4);

        // Act
        assignWeightsAndBiases(targetNetwork, onnxModel, [4, 1], onnxModel.metadata_props);

        // Assert
        expect({
          secondHiddenBias: getHiddenNodes(targetNetwork)[4].bias,
          secondHiddenWeights: getHiddenNodes(targetNetwork)[4].connections.in.map(
            (connectionEntry) => connectionEntry.weight,
          ),
        }).toEqual({
          secondHiddenBias: 0.25,
          secondHiddenWeights: scenario.kernelPattern,
        });
      });
    });

    describe('when Conv metadata JSON is malformed', () => {
      it('swallows the optional reconstruction failure', () => {
        // Arrange
        const scenario = createConvSharingVerifiedScenario();
        const onnxModel = exportToONNX(scenario.network, {
          includeMetadata: true,
          conv2dMappings: scenario.mappings,
        });
        findMetadataEntry(onnxModel.metadata_props, 'conv2d_specs').value = '{';
        const targetNetwork = Network.createMLP(9, [4], 2);
        const assignCallback = () =>
          assignWeightsAndBiases(targetNetwork, onnxModel, [4], onnxModel.metadata_props);

        // Assert
        expect(assignCallback).not.toThrow();
      });
    });

    describe('when the Conv layer list parses to a non-array payload', () => {
      it('swallows the reconstruction error raised by the malformed layer payload', () => {
        // Arrange
        const scenario = createConvSharingVerifiedScenario();
        const onnxModel = exportToONNX(scenario.network, {
          includeMetadata: true,
          conv2dMappings: scenario.mappings,
        });
        findMetadataEntry(onnxModel.metadata_props, 'conv2d_layers').value = '{}';
        const targetNetwork = Network.createMLP(9, [4], 2);
        const assignCallback = () =>
          assignWeightsAndBiases(targetNetwork, onnxModel, [4], onnxModel.metadata_props);

        // Assert
        expect(assignCallback).not.toThrow();
      });
    });

    describe('when Conv metadata names a layer without a matching spec', () => {
      it('skips the optional reconstruction path without throwing', () => {
        // Arrange
        const scenario = createConvSharingVerifiedScenario();
        const onnxModel = exportToONNX(scenario.network, {
          includeMetadata: true,
          conv2dMappings: scenario.mappings,
        });
        findMetadataEntry(onnxModel.metadata_props, 'conv2d_specs').value = '[]';
        const targetNetwork = Network.createMLP(9, [4], 2);
        const assignCallback = () =>
          assignWeightsAndBiases(targetNetwork, onnxModel, [4], onnxModel.metadata_props);

        // Assert
        expect(assignCallback).not.toThrow();
      });
    });

    describe('when Conv metadata points past the available hidden layers', () => {
      it('skips the out-of-bounds Conv layer without throwing', () => {
        // Arrange
        const scenario = createConvSharingVerifiedScenario();
        const onnxModel = exportToONNX(scenario.network, {
          includeMetadata: true,
          conv2dMappings: scenario.mappings,
        });
        const outOfBoundsSpec = {
          ...JSON.parse(findMetadataEntry(onnxModel.metadata_props, 'conv2d_specs').value)[0],
          layerIndex: 2,
        } as Conv2DMapping;
        findMetadataEntry(onnxModel.metadata_props, 'conv2d_layers').value = '[2]';
        findMetadataEntry(onnxModel.metadata_props, 'conv2d_specs').value = JSON.stringify([
          outOfBoundsSpec,
        ]);
        const targetNetwork = Network.createMLP(9, [4], 2);
        const assignCallback = () =>
          assignWeightsAndBiases(targetNetwork, onnxModel, [4], onnxModel.metadata_props);

        // Assert
        expect(assignCallback).not.toThrow();
      });
    });

    describe('when the Conv bias tensor is missing', () => {
      it('skips Conv tensor reconstruction without throwing', () => {
        // Arrange
        const scenario = createConvSharingVerifiedScenario();
        const onnxModel = exportToONNX(scenario.network, {
          includeMetadata: true,
          conv2dMappings: scenario.mappings,
        });
        onnxModel.graph.initializer = onnxModel.graph.initializer.filter(
          (tensor) => tensor.name !== 'ConvB0',
        );
        const targetNetwork = Network.createMLP(9, [4], 2);
        const assignCallback = () =>
          assignWeightsAndBiases(targetNetwork, onnxModel, [4], onnxModel.metadata_props);

        // Assert
        expect(assignCallback).not.toThrow();
      });
    });

    describe('when the Conv tensor dimensions do not match the metadata', () => {
      it('skips the mismatched Conv tensor payload without throwing', () => {
        // Arrange
        const scenario = createConvSharingVerifiedScenario();
        const onnxModel = exportToONNX(scenario.network, {
          includeMetadata: true,
          conv2dMappings: scenario.mappings,
        });
        const convWeightTensor = onnxModel.graph.initializer.find(
          (tensor) => tensor.name === 'ConvW0',
        );
        if (!convWeightTensor) {
          throw new Error('Missing ConvW0 initializer');
        }
        convWeightTensor.dims = [2, 1, 2, 2];
        const targetNetwork = Network.createMLP(9, [4], 2);
        const assignCallback = () =>
          assignWeightsAndBiases(targetNetwork, onnxModel, [4], onnxModel.metadata_props);

        // Assert
        expect(assignCallback).not.toThrow();
      });
    });

    describe('when Conv padding pushes some kernel positions outside the input image', () => {
      it('skips those out-of-bounds kernel coordinates without throwing', () => {
        // Arrange
        const scenario = createConvSharingVerifiedScenario();
        const onnxModel = exportToONNX(scenario.network, {
          includeMetadata: true,
          conv2dMappings: scenario.mappings,
        });
        const paddedSpec = {
          ...JSON.parse(findMetadataEntry(onnxModel.metadata_props, 'conv2d_specs').value)[0],
          padTop: 1,
          padLeft: 1,
        } as Conv2DMapping;
        findMetadataEntry(onnxModel.metadata_props, 'conv2d_specs').value = JSON.stringify([
          paddedSpec,
        ]);
        const targetNetwork = Network.createMLP(9, [4], 2);
        const assignCallback = () =>
          assignWeightsAndBiases(targetNetwork, onnxModel, [4], onnxModel.metadata_props);

        // Assert
        expect(assignCallback).not.toThrow();
      });
    });

    describe('when Conv feature indexing points past the available source nodes', () => {
      it('skips those source-node lookups without throwing', () => {
        // Arrange
        const scenario = createConvSharingVerifiedScenario();
        const onnxModel = exportToONNX(scenario.network, {
          includeMetadata: true,
          conv2dMappings: scenario.mappings,
        });
        const stretchedSpec = {
          ...JSON.parse(findMetadataEntry(onnxModel.metadata_props, 'conv2d_specs').value)[0],
          inHeight: 5,
          inWidth: 5,
        } as Conv2DMapping;
        findMetadataEntry(onnxModel.metadata_props, 'conv2d_specs').value = JSON.stringify([
          stretchedSpec,
        ]);
        const targetNetwork = Network.createMLP(9, [4], 2);
        const assignCallback = () =>
          assignWeightsAndBiases(targetNetwork, onnxModel, [4], onnxModel.metadata_props);

        // Assert
        expect(assignCallback).not.toThrow();
      });
    });

    describe('when one Conv receptive connection is missing on the target neuron', () => {
      it('skips that inbound lookup while continuing the remaining assignments', () => {
        // Arrange
        const scenario = createConvSharingVerifiedScenario();
        const onnxModel = exportToONNX(scenario.network, {
          includeMetadata: true,
          conv2dMappings: scenario.mappings,
        });
        const targetNetwork = Network.createMLP(9, [4], 2);
        removeInboundConnection(
          getHiddenNodes(targetNetwork)[0],
          getInputNodes(targetNetwork)[0],
          targetNetwork,
        );
        const assignCallback = () =>
          assignWeightsAndBiases(targetNetwork, onnxModel, [4], onnxModel.metadata_props);

        // Assert
        expect(assignCallback).not.toThrow();
      });
    });

    describe('when the provided hidden-layer slice is shorter than the Conv output grid', () => {
      it('skips the missing target neuron positions without throwing', () => {
        // Arrange
        const scenario = createConvSharingVerifiedScenario();
        const onnxModel = exportToONNX(scenario.network, {
          includeMetadata: true,
          conv2dMappings: scenario.mappings,
        });
        const targetNetwork = Network.createMLP(9, [4], 2);
        const assignCallback = () =>
          assignWeightsAndBiases(targetNetwork, onnxModel, [3], onnxModel.metadata_props);

        // Assert
        expect(assignCallback).not.toThrow();
      });
    });
  });
});