import Network from '../../network';
import { exportToONNX, importFromONNX } from '../network.onnx';

type PoolingAwareNetwork = Network & { _onnxPooling?: unknown };

jest.retryTimes(2, { logErrorsBeforeRetry: true });

function getHiddenNodes(network: Network) {
  return network.nodes.filter((nodeEntry) => nodeEntry.type === 'hidden');
}

function buildPartitionedLstmNetwork(
  inputSize: number,
  unitCount: number,
  outputSize: number,
): Network {
  const hiddenSize = unitCount * 5;
  const network = Network.createMLP(inputSize, [hiddenSize], outputSize);
  const hiddenNodes = getHiddenNodes(network);
  const cellStartIndex = unitCount * 2;

  hiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
    hiddenNode.bias = hiddenNodeIndex * 0.01;

    if (
      hiddenNodeIndex >= cellStartIndex &&
      hiddenNodeIndex < cellStartIndex + unitCount
    ) {
      const recurrentWeight = 0.5 + (hiddenNodeIndex - cellStartIndex) * 0.01;

      if (hiddenNode.connections.self.length === 0) {
        hiddenNode.connect(hiddenNode, recurrentWeight);
      } else {
        hiddenNode.connections.self[0].weight = recurrentWeight;
      }
    }
  });

  return network;
}

function buildPartitionedGruNetwork(
  inputSize: number,
  unitCount: number,
  outputSize: number,
): Network {
  const hiddenSize = unitCount * 4;
  const network = Network.createMLP(inputSize, [hiddenSize], outputSize);
  const hiddenNodes = getHiddenNodes(network);
  const candidateStartIndex = unitCount * 2;

  hiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
    hiddenNode.bias = hiddenNodeIndex * 0.02;

    if (
      hiddenNodeIndex >= candidateStartIndex &&
      hiddenNodeIndex < candidateStartIndex + unitCount
    ) {
      const recurrentWeight =
        0.7 + (hiddenNodeIndex - candidateStartIndex) * 0.02;

      if (hiddenNode.connections.self.length === 0) {
        hiddenNode.connect(hiddenNode, recurrentWeight);
      } else {
        hiddenNode.connections.self[0].weight = recurrentWeight;
      }
    }
  });

  return network;
}

function buildPartitionedLstmOnnxModelWithoutInitializer(
  initializerPrefix: string,
) {
  const sourceNetwork = buildPartitionedLstmNetwork(2, 2, 1);
  const onnxModel = exportToONNX(sourceNetwork, { allowRecurrent: true });

  onnxModel.graph.initializer = onnxModel.graph.initializer.filter(
    (initializerEntry) => !initializerEntry.name.startsWith(initializerPrefix),
  );

  return onnxModel;
}

describe('network onnx import recurrent chapter', () => {
  describe('recurrent reconstruction', () => {
    describe('given a partitioned LSTM-like payload', () => {
      let sourceNetwork: Network;
      let importedNetwork: Network;

      beforeEach(() => {
        // Arrange
        sourceNetwork = buildPartitionedLstmNetwork(2, 2, 1);
        const onnxModel = exportToONNX(sourceNetwork, { allowRecurrent: true });

        // Act
        importedNetwork = importFromONNX(onnxModel);
      });

      describe('when reading the rebuilt network shape', () => {
        it('preserves the input node count', () => {
          // Assert
          expect(
            importedNetwork.nodes.filter(
              (nodeEntry) => nodeEntry.type === 'input',
            ).length,
          ).toBe(2);
        });

        it('restores the exported LSTM gate biases and recurrent self weights', () => {
          // Arrange
          const sourceHiddenNodes = getHiddenNodes(sourceNetwork);
          const importedHiddenNodes = getHiddenNodes(importedNetwork);

          // Assert
          expect({
            importedGateBiases: importedHiddenNodes
              .slice(0, 8)
              .map((hiddenNode) => Number(hiddenNode.bias.toFixed(9))),
            importedRecurrentSelfWeights: importedHiddenNodes
              .slice(4, 6)
              .map(
                (hiddenNode) => hiddenNode.connections.self[0]?.weight ?? null,
              ),
          }).toEqual({
            importedGateBiases: sourceHiddenNodes
              .slice(0, 8)
              .map((hiddenNode) => Number(hiddenNode.bias.toFixed(9))),
            importedRecurrentSelfWeights: sourceHiddenNodes
              .slice(4, 6)
              .map(
                (hiddenNode) => hiddenNode.connections.self[0]?.weight ?? null,
              ),
          });
        });
      });
    });

    describe('given a partitioned GRU-like payload', () => {
      let sourceNetwork: Network;
      let importedNetwork: Network;

      beforeEach(() => {
        // Arrange
        sourceNetwork = buildPartitionedGruNetwork(2, 2, 1);
        const onnxModel = exportToONNX(sourceNetwork, { allowRecurrent: true });

        // Act
        importedNetwork = importFromONNX(onnxModel);
      });

      describe('when reading the rebuilt network shape', () => {
        it('preserves the output node count', () => {
          // Assert
          expect(
            importedNetwork.nodes.filter(
              (nodeEntry) => nodeEntry.type === 'output',
            ).length,
          ).toBe(1);
        });

        it('restores the exported GRU gate biases and recurrent self weights', () => {
          // Arrange
          const sourceHiddenNodes = getHiddenNodes(sourceNetwork);
          const importedHiddenNodes = getHiddenNodes(importedNetwork);
          const importedMemoryCellNodes = importedHiddenNodes.slice(6, 8);
          const importedPreviousOutputNodes = importedHiddenNodes.slice(10, 12);

          // Assert
          expect({
            importedCandidateBiases: importedMemoryCellNodes.map((hiddenNode) =>
              Number(hiddenNode.bias.toFixed(9)),
            ),
            importedCandidateRecurrentWeights: importedMemoryCellNodes.map(
              (hiddenNode, nodeIndex) =>
                hiddenNode.connections.in.find(
                  (connection) =>
                    connection.from === importedPreviousOutputNodes[nodeIndex],
                )?.weight ?? null,
            ),
          }).toEqual({
            importedCandidateBiases: sourceHiddenNodes
              .slice(0, 6)
              .slice(4, 6)
              .map((hiddenNode) => Number(hiddenNode.bias.toFixed(9))),
            importedCandidateRecurrentWeights: sourceHiddenNodes
              .slice(4, 6)
              .map(
                (hiddenNode) => hiddenNode.connections.self[0]?.weight ?? null,
              ),
          });
        });
      });
    });

    describe('given a recurrent payload is missing one recurrent initializer', () => {
      let importCallback: () => Network;

      beforeEach(() => {
        // Arrange
        const sourceNetwork = buildPartitionedLstmNetwork(2, 2, 1);
        const onnxModel = exportToONNX(sourceNetwork, { allowRecurrent: true });
        onnxModel.graph.initializer = onnxModel.graph.initializer.filter(
          (initializerEntry) => !initializerEntry.name.startsWith('LSTM_R'),
        );

        // Act
        importCallback = () => importFromONNX(onnxModel);
      });

      describe('when the importer rebuilds the network', () => {
        it('does not throw', () => {
          // Assert
          expect(importCallback).not.toThrow();
        });
      });
    });

    describe('given a recurrent payload is missing the fused input initializer', () => {
      let importedNetwork: Network;

      beforeEach(() => {
        // Arrange
        const onnxModel =
          buildPartitionedLstmOnnxModelWithoutInitializer('LSTM_W');

        // Act
        importedNetwork = importFromONNX(onnxModel);
      });

      describe('when the importer falls back to the layered reconstruction path', () => {
        it('preserves the baseline network shape', () => {
          // Assert
          expect({
            hiddenCount: getHiddenNodes(importedNetwork).length,
            inputCount: importedNetwork.nodes.filter(
              (nodeEntry) => nodeEntry.type === 'input',
            ).length,
            outputCount: importedNetwork.nodes.filter(
              (nodeEntry) => nodeEntry.type === 'output',
            ).length,
          }).toEqual({
            hiddenCount: 10,
            inputCount: 2,
            outputCount: 1,
          });
        });
      });
    });

    describe('given a recurrent payload is missing the fused bias initializer', () => {
      let importedNetwork: Network;

      beforeEach(() => {
        // Arrange
        const onnxModel =
          buildPartitionedLstmOnnxModelWithoutInitializer('LSTM_B');

        // Act
        importedNetwork = importFromONNX(onnxModel);
      });

      describe('when the importer falls back to the layered reconstruction path', () => {
        it('keeps the exported recurrent self-connections on the base network', () => {
          // Arrange
          const importedHiddenNodes = getHiddenNodes(importedNetwork);

          // Assert
          expect(
            importedHiddenNodes
              .slice(4, 6)
              .map(
                (hiddenNode) => hiddenNode.connections.self[0]?.weight ?? null,
              ),
          ).toEqual([0.5, 0.51]);
        });
      });
    });
  });

  describe('pooling metadata attachment', () => {
    describe('given the exported model includes pooling mappings', () => {
      let importedNetwork: PoolingAwareNetwork;

      beforeEach(() => {
        // Arrange
        const sourceNetwork = Network.createMLP(6, [4], 1);
        const onnxModel = exportToONNX(sourceNetwork, {
          includeMetadata: true,
          pool2dMappings: [
            {
              afterLayerIndex: 1,
              type: 'MaxPool',
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 2,
              strideWidth: 2,
            },
          ],
        });

        // Act
        importedNetwork = importFromONNX(onnxModel) as PoolingAwareNetwork;
      });

      describe('when reading the reconstructed network metadata', () => {
        it('attaches the _onnxPooling helper payload', () => {
          // Assert
          expect(importedNetwork._onnxPooling != null).toBe(true);
        });
      });
    });

    describe('given the exported model includes Conv, Pool, and flatten metadata', () => {
      let importedNetwork: PoolingAwareNetwork;

      beforeEach(() => {
        // Arrange
        const sourceNetwork = Network.createMLP(9, [4], 1);
        const onnxModel = exportToONNX(sourceNetwork, {
          conv2dMappings: [
            {
              layerIndex: 1,
              inHeight: 3,
              inWidth: 3,
              inChannels: 1,
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
              outHeight: 2,
              outWidth: 2,
              outChannels: 1,
            },
          ],
          flattenAfterPooling: true,
          includeMetadata: true,
          pool2dMappings: [
            {
              afterLayerIndex: 1,
              type: 'MaxPool',
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 2,
              strideWidth: 2,
            },
          ],
        });

        // Act
        importedNetwork = importFromONNX(onnxModel) as PoolingAwareNetwork;
      });

      describe('when reading the reconstructed network metadata', () => {
        it('attaches the derived virtual pooled shape, flatten hint, and consistency audit', () => {
          // Assert
          expect(importedNetwork._onnxPooling).toEqual({
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
    });
  });
});
