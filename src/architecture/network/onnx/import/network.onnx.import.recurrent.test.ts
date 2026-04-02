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

describe('network onnx import recurrent chapter', () => {
  describe('recurrent reconstruction', () => {
    describe('given a partitioned LSTM-like payload', () => {
      let importedNetwork: Network;

      beforeEach(() => {
        // Arrange
        const sourceNetwork = buildPartitionedLstmNetwork(2, 2, 1);
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
      });
    });

    describe('given a partitioned GRU-like payload', () => {
      let importedNetwork: Network;

      beforeEach(() => {
        // Arrange
        const sourceNetwork = buildPartitionedGruNetwork(2, 2, 1);
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
  });
});
