import Network from '../../network';
import { exportToONNX } from '../network.onnx';
import { collectLstmPatternStubs } from './network.onnx.export-orchestrators.utils';
import type { OnnxModel } from '../network.onnx';

type OnnxInitializerView = { name: string };
type OnnxGraphNodeView = { op_type: string };

jest.retryTimes(2, { logErrorsBeforeRetry: true });

function getHiddenNodes(network: Network) {
  return network.nodes.filter((nodeEntry) => nodeEntry.type === 'hidden');
}

function hasMetadataKey(onnxModel: OnnxModel, key: string): boolean {
  return (onnxModel.metadata_props ?? []).some(
    (metadataEntry) => metadataEntry.key === key,
  );
}

function hasInitializer(
  onnxModel: OnnxModel,
  initializerName: string,
): boolean {
  return onnxModel.graph.initializer.some(
    (initializerEntry) =>
      (initializerEntry as OnnxInitializerView).name === initializerName,
  );
}

function hasOperator(onnxModel: OnnxModel, operatorType: string): boolean {
  return onnxModel.graph.node.some(
    (graphNode) => (graphNode as OnnxGraphNodeView).op_type === operatorType,
  );
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

describe('network onnx export recurrent chapter', () => {
  describe('collectLstmPatternStubs()', () => {
    describe('given malformed hidden-layer entries', () => {
      describe('when recurrent heuristics are enabled', () => {
        it('returns an empty list via the safety fallback', () => {
          // Arrange
          const malformedLayers: Array<Network['nodes'] | null> = [
            [],
            null,
            [],
          ];

          // Act
          const lstmPatternStubs = collectLstmPatternStubs(
            malformedLayers as Network['nodes'][],
            true,
          );

          // Assert
          expect(lstmPatternStubs).toEqual([]);
        });
      });
    });
  });

  describe('heuristic fused recurrent emission', () => {
    describe('given a partitioned LSTM-like hidden layer', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const network = buildPartitionedLstmNetwork(3, 2, 1);

        // Act
        onnxModel = exportToONNX(network, { allowRecurrent: true });
      });

      describe('when reading the emitted payload', () => {
        it('includes the first LSTM weight initializer', () => {
          // Assert
          expect(hasInitializer(onnxModel, 'LSTM_W0')).toBe(true);
        });

        it('includes an LSTM operator node', () => {
          // Assert
          expect(hasOperator(onnxModel, 'LSTM')).toBe(true);
        });

        it('records lstm_emitted_layers metadata', () => {
          // Assert
          expect(hasMetadataKey(onnxModel, 'lstm_emitted_layers')).toBe(true);
        });
      });
    });

    describe('given a partitioned GRU-like hidden layer', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const network = buildPartitionedGruNetwork(2, 3, 1);

        // Act
        onnxModel = exportToONNX(network, { allowRecurrent: true });
      });

      describe('when reading the emitted payload', () => {
        it('includes the first GRU weight initializer', () => {
          // Assert
          expect(hasInitializer(onnxModel, 'GRU_W0')).toBe(true);
        });

        it('includes a GRU operator node', () => {
          // Assert
          expect(hasOperator(onnxModel, 'GRU')).toBe(true);
        });

        it('records gru_emitted_layers metadata', () => {
          // Assert
          expect(hasMetadataKey(onnxModel, 'gru_emitted_layers')).toBe(true);
        });
      });
    });

    describe('given a recurrent near-miss that matches no fused partition', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const network = Network.createMLP(2, [9], 1);
        const hiddenNodes = getHiddenNodes(network);

        hiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
          if (hiddenNodeIndex % 2 === 0) {
            hiddenNode.connect(hiddenNode, 0.3);
          }
        });

        // Act
        onnxModel = exportToONNX(network, { allowRecurrent: true });
      });

      describe('when the heuristic cannot classify the recurrent shape', () => {
        it('records rnn_pattern_fallback metadata', () => {
          // Assert
          expect(hasMetadataKey(onnxModel, 'rnn_pattern_fallback')).toBe(true);
        });
      });
    });
  });
});
