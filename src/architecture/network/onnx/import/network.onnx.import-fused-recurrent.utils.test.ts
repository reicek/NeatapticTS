import Node from '../../../node';
import Network from '../../network';
import type {
  OnnxMetadataProperty,
  OnnxModel,
  OnnxTensor,
} from '../schema/network.onnx.schema.types';
import type { OnnxLayerFactory } from '../network.onnx.utils.types';
import { reconstructFusedRecurrentLayers } from './network.onnx.import-fused-recurrent.utils';

const LSTM_GATE_COUNT = 4;
const GRU_GATE_COUNT = 3;
const RECURRENT_GATE_GROUP_INDEX = 2;

type MockFusedLayerRuntime = {
  input: (sourceLayer: { output: { nodes: Node[] } }) => void;
  nodes: Node[];
  output?: { nodes: Node[] };
};

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

function createOnnxModel(initializer: OnnxTensor[] = []): OnnxModel {
  return {
    graph: {
      inputs: [],
      outputs: [],
      initializer,
      node: [],
    },
    metadata_props: [],
  };
}

function createNumberSequence(length: number, start = 1): number[] {
  return Array.from({ length }, (_, index) => start + index);
}

function createMetadataProperty(
  key: string,
  value: string,
): OnnxMetadataProperty {
  return { key, value };
}

function getHiddenNodes(network: Network): Node[] {
  return network.nodes.filter(
    (nodeEntry): nodeEntry is Node => nodeEntry.type === 'hidden',
  );
}

function createMockFusedLayerRuntime(options: {
  gateCount: number;
  previousConnectionMode: 'all' | 'first-only';
  reconnectSelfMode: 'all' | 'first-only';
  unitSize: number;
}): MockFusedLayerRuntime {
  const nodes = Array.from({ length: options.gateCount * options.unitSize }, () => {
    return new Node('hidden');
  });
  const recurrentGateStart = RECURRENT_GATE_GROUP_INDEX * options.unitSize;
  const recurrentGateNodes = nodes.slice(
    recurrentGateStart,
    recurrentGateStart + options.unitSize,
  );

  if (options.reconnectSelfMode === 'all') {
    recurrentGateNodes.forEach((recurrentGateNode) => {
      recurrentGateNode.connect(recurrentGateNode, -5);
    });
  } else {
    recurrentGateNodes[0]?.connect(recurrentGateNodes[0], -5);
  }

  return {
    input(sourceLayer) {
      const previousLayerNodes = sourceLayer.output.nodes;

      nodes.forEach((node) => {
        const sourceNodes =
          options.previousConnectionMode === 'all'
            ? previousLayerNodes
            : previousLayerNodes.slice(0, 1);

        sourceNodes.forEach((sourceNode) => sourceNode.connect(node, -1));
      });
    },
    nodes,
    output: {
      nodes: nodes.slice(-options.unitSize),
    },
  };
}

function createMockLayerFactory(): OnnxLayerFactory {
  return {
    gru: (unitSize: number) =>
      createMockFusedLayerRuntime({
        gateCount: GRU_GATE_COUNT,
        previousConnectionMode: 'all',
        reconnectSelfMode: 'all',
        unitSize,
      }),
    lstm: (unitSize: number) =>
      createMockFusedLayerRuntime({
        gateCount: LSTM_GATE_COUNT,
        previousConnectionMode: 'first-only',
        reconnectSelfMode: 'first-only',
        unitSize,
      }),
  } as unknown as OnnxLayerFactory;
}

function withToSplicedFallback<T>(callback: () => T): T {
  const originalDescriptor = Object.getOwnPropertyDescriptor(
    Array.prototype,
    'toSpliced',
  );

  Object.defineProperty(Array.prototype, 'toSpliced', {
    configurable: true,
    value: undefined,
    writable: true,
  });

  try {
    return callback();
  } finally {
    if (originalDescriptor) {
      Object.defineProperty(Array.prototype, 'toSpliced', originalDescriptor);
    } else {
      Reflect.deleteProperty(Array.prototype, 'toSpliced');
    }
  }
}

describe('network onnx fused recurrent import utility chapter', () => {
  describe('reconstructFusedRecurrentLayers()', () => {
    describe('when fused recurrent metadata JSON is malformed', () => {
      it('swallows the parse failure and keeps the network unchanged', () => {
        // Arrange
        const network = Network.createMLP(1, [1], 1);

        // Act
        reconstructFusedRecurrentLayers(
          network,
          createOnnxModel(),
          [1],
          createMockLayerFactory(),
          [createMetadataProperty('lstm_emitted_layers', '{')],
        );

        // Assert
        expect(network.nodes.map((nodeEntry) => nodeEntry.type)).toEqual([
          'input',
          'hidden',
          'output',
        ]);
      });
    });

    describe('when fused recurrent metadata parses to a non-array payload', () => {
      it('normalizes to no emitted layers and keeps the network unchanged', () => {
        // Arrange
        const network = Network.createMLP(1, [1], 1);

        // Act
        reconstructFusedRecurrentLayers(
          network,
          createOnnxModel(),
          [1],
          createMockLayerFactory(),
          [createMetadataProperty('lstm_emitted_layers', '{"layer":1}')],
        );

        // Assert
        expect(network.nodes.map((nodeEntry) => nodeEntry.type)).toEqual([
          'input',
          'hidden',
          'output',
        ]);
      });
    });

    describe('when fused layer reconstruction cannot continue', () => {
      it('returns early for invalid indices, missing tensors, and incompatible unit sizes', () => {
        // Arrange
        const invalidIndexNetwork = Network.createMLP(1, [1], 1);
        const missingTensorNetwork = Network.createMLP(1, [1], 1);
        const incompatibleUnitNetwork = Network.createMLP(1, [1], 1);

        // Act
        reconstructFusedRecurrentLayers(
          invalidIndexNetwork,
          createOnnxModel(),
          [1],
          createMockLayerFactory(),
          [createMetadataProperty('lstm_emitted_layers', '[0]')],
        );
        reconstructFusedRecurrentLayers(
          missingTensorNetwork,
          createOnnxModel([
            createTensor('LSTM_W0', [4, 1], createNumberSequence(4, 10)),
            createTensor('LSTM_R0', [4, 1], createNumberSequence(4, 100)),
          ]),
          [1],
          createMockLayerFactory(),
          [createMetadataProperty('lstm_emitted_layers', '[1]')],
        );
        reconstructFusedRecurrentLayers(
          incompatibleUnitNetwork,
          createOnnxModel([
            createTensor('LSTM_W0', [5, 1], createNumberSequence(5, 10)),
            createTensor('LSTM_R0', [5, 1], createNumberSequence(5, 100)),
            createTensor('LSTM_B0', [5], createNumberSequence(5, 1_000)),
          ]),
          [1],
          createMockLayerFactory(),
          [createMetadataProperty('lstm_emitted_layers', '[1]')],
        );

        // Assert
        expect({
          incompatibleUnitHiddenCount: getHiddenNodes(incompatibleUnitNetwork).length,
          invalidIndexHiddenCount: getHiddenNodes(invalidIndexNetwork).length,
          missingTensorHiddenCount: getHiddenNodes(missingTensorNetwork).length,
        }).toEqual({
          incompatibleUnitHiddenCount: 1,
          invalidIndexHiddenCount: 1,
          missingTensorHiddenCount: 1,
        });
      });
    });

    describe('when fused LSTM metadata targets a middle hidden layer', () => {
      it('rebuilds the hidden slice, rewires neighbors, and applies imported weights', () => {
        // Arrange
        const hiddenLayerSizes = [2, 2, 1];
        const network = Network.createMLP(2, hiddenLayerSizes, 1);
        const inputWeights = createNumberSequence(24, 10);
        const recurrentWeights = createNumberSequence(16, 100);
        const biases = createNumberSequence(8, 1_000);

        const onnxModel = createOnnxModel([
          createTensor('LSTM_W1', [8, 3], inputWeights),
          createTensor('LSTM_R1', [8, 2], recurrentWeights),
          createTensor('LSTM_B1', [8], biases),
        ]);

        const metadata = [createMetadataProperty('lstm_emitted_layers', '[2]')];

        // Act
        const reconstructionSnapshot = withToSplicedFallback(() => {
          reconstructFusedRecurrentLayers(
            network,
            onnxModel,
            hiddenLayerSizes,
            createMockLayerFactory(),
            metadata,
          );

          const hiddenNodes = getHiddenNodes(network);
          const replacedNodes = hiddenNodes.slice(2, 10);
          const recurrentGateNodes = replacedNodes.slice(4, 6);
          const nextHiddenNode = hiddenNodes[10];

          return {
            firstGateBias: replacedNodes[0]?.bias,
            firstGateInputWeights: replacedNodes[0]?.connections.in.map(
              (connection) => connection.weight,
            ),
            hiddenCount: hiddenNodes.length,
            nextHiddenInboundCount: nextHiddenNode?.connections.in.length,
            recurrentSelfConnectionCounts: recurrentGateNodes.map(
              (node) => node.connections.self.length,
            ),
            recurrentSelfWeights: recurrentGateNodes.map(
              (node) => node.connections.self[0]?.weight ?? null,
            ),
          };
        });

        // Assert
        expect(reconstructionSnapshot).toEqual({
          firstGateBias: 1000,
          firstGateInputWeights: [10],
          hiddenCount: 11,
          nextHiddenInboundCount: 2,
          recurrentSelfConnectionCounts: [1, 0],
          recurrentSelfWeights: [108, null],
        });
      });
    });

    describe('when fused GRU metadata targets the only hidden layer', () => {
      it('rebuilds from input nodes to output nodes using the native array splice path', () => {
        // Arrange
        const hiddenLayerSizes = [1];
        const network = Network.createMLP(2, hiddenLayerSizes, 1);
        const inputWeights = createNumberSequence(6, 30);
        const recurrentWeights = createNumberSequence(3, 300);
        const biases = createNumberSequence(3, 2_000);

        const onnxModel = createOnnxModel([
          createTensor('GRU_W0', [3, 2], inputWeights),
          createTensor('GRU_R0', [3, 1], recurrentWeights),
          createTensor('GRU_B0', [3], biases),
        ]);

        // Act
        reconstructFusedRecurrentLayers(
          network,
          onnxModel,
          hiddenLayerSizes,
          createMockLayerFactory(),
          [createMetadataProperty('gru_emitted_layers', '[1]')],
        );
        const hiddenNodes = getHiddenNodes(network);
        const outputNode = network.nodes.find(
          (nodeEntry): nodeEntry is Node => nodeEntry.type === 'output',
        );

        if (!outputNode) {
          throw new Error('Expected output node after GRU reconstruction');
        }

        // Assert
        expect({
          firstHiddenBias: hiddenNodes[0]?.bias,
          firstHiddenInputWeights: hiddenNodes[0]?.connections.in.map(
            (connection) => connection.weight,
          ),
          hiddenCount: hiddenNodes.length,
          outputInboundCount: outputNode.connections.in.length,
          recurrentSelfWeight: hiddenNodes[2]?.connections.self[0]?.weight,
        }).toEqual({
          firstHiddenBias: 2000,
          firstHiddenInputWeights: [30, 31],
          hiddenCount: 3,
          outputInboundCount: 1,
          recurrentSelfWeight: 302,
        });
      });
    });
  });
});