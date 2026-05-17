import Network from '../../network';
import {
  attachOnnxConcatMergeMetadata,
  restoreConcatMergeConnections,
} from './network.onnx.import-concat.utils';
import type { OnnxModel } from '../schema/network.onnx.schema.types';

type AdvancedGraphAwareNetwork = Network & {
  _onnxAdvancedGraph?: {
    crossLayerConnections?: {
      sourceNodeIndex: number;
      sourceLayerIndex: number;
      targetNodeIndex: number;
      targetLayerIndex: number;
      branchTensorName: string;
    }[];
    concatMerges?: {
      sourceLayerIndex: number;
      targetLayerIndex: number;
      concatNodeName: string;
      concatOutputName: string;
      inputOrder: 'previous_then_source';
    }[];
  };
};

jest.retryTimes(2, { logErrorsBeforeRetry: true });

function createConcatMetadataValue(): string {
  return JSON.stringify([
    {
      sourceLayerIndex: 0,
      targetLayerIndex: 2,
      concatNodeName: 'concat_merge_l0_to_l2',
      concatOutputName: 'ConcatMerge_0_to_2',
      inputOrder: 'previous_then_source',
    },
  ]);
}

function createConcatOnnxModel(weightDims: number[] = [1, 4]): OnnxModel {
  return {
    graph: {
      inputs: [],
      outputs: [],
      initializer: [
        {
          name: 'W1',
          data_type: 1,
          dims: weightDims,
          float_data: Array.from(
            { length: weightDims[0]! * weightDims[1]! },
            (_unused, weightIndex) => (weightIndex + 1) / 10,
          ),
        },
      ],
      node: [
        {
          op_type: 'Concat',
          input: ['Layer_1', 'input_layer'],
          output: ['ConcatMerge_0_to_2'],
          name: 'concat_merge_l0_to_l2',
        },
      ],
    },
    metadata_props: [
      {
        key: 'advanced_graph_concat_merges',
        value: createConcatMetadataValue(),
      },
    ],
  } as OnnxModel;
}

describe('network onnx import concat utility chapter', () => {
  describe('attachOnnxConcatMergeMetadata()', () => {
    it('does nothing when no ONNX model is provided for validation', () => {
      // Arrange
      const network = Network.createMLP(2, [2], 1) as AdvancedGraphAwareNetwork;
      const onnxModel = createConcatOnnxModel();

      // Act
      attachOnnxConcatMergeMetadata(network, onnxModel.metadata_props ?? []);

      // Assert
      expect(network._onnxAdvancedGraph).toBeUndefined();
    });

    it('attaches concat metadata onto an empty advanced-graph payload', () => {
      // Arrange
      const network = Network.createMLP(2, [2], 1) as AdvancedGraphAwareNetwork;
      const onnxModel = createConcatOnnxModel();

      // Act
      attachOnnxConcatMergeMetadata(
        network,
        onnxModel.metadata_props ?? [],
        onnxModel,
      );

      // Assert
      expect(network._onnxAdvancedGraph).toEqual({
        concatMerges: [
          {
            sourceLayerIndex: 0,
            targetLayerIndex: 2,
            concatNodeName: 'concat_merge_l0_to_l2',
            concatOutputName: 'ConcatMerge_0_to_2',
            inputOrder: 'previous_then_source',
          },
        ],
      });
    });

    it('merges validated concat audit metadata onto an existing advanced-graph payload', () => {
      // Arrange
      const network = Network.createMLP(2, [2], 1) as AdvancedGraphAwareNetwork;
      const onnxModel = createConcatOnnxModel();
      network._onnxAdvancedGraph = {
        crossLayerConnections: [
          {
            sourceNodeIndex: 0,
            sourceLayerIndex: 0,
            targetNodeIndex: 4,
            targetLayerIndex: 2,
            branchTensorName: 'Branch_l0_to_l2_from_n0_to_n4',
          },
        ],
      };

      // Act
      attachOnnxConcatMergeMetadata(
        network,
        onnxModel.metadata_props ?? [],
        onnxModel,
      );

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
        concatMerges: [
          {
            sourceLayerIndex: 0,
            targetLayerIndex: 2,
            concatNodeName: 'concat_merge_l0_to_l2',
            concatOutputName: 'ConcatMerge_0_to_2',
            inputOrder: 'previous_then_source',
          },
        ],
      });
    });

    it('ignores concat metadata when the payload is not an array', () => {
      // Arrange
      const network = Network.createMLP(2, [2], 1) as AdvancedGraphAwareNetwork;
      const onnxModel = createConcatOnnxModel();
      onnxModel.metadata_props = [
        {
          key: 'advanced_graph_concat_merges',
          value: JSON.stringify({ unexpected: true }),
        },
      ];

      // Act
      attachOnnxConcatMergeMetadata(
        network,
        onnxModel.metadata_props ?? [],
        onnxModel,
      );

      // Assert
      expect(network._onnxAdvancedGraph).toBeUndefined();
    });

    it('ignores concat metadata when one merge record is malformed', () => {
      // Arrange
      const network = Network.createMLP(2, [2], 1) as AdvancedGraphAwareNetwork;
      const onnxModel = createConcatOnnxModel();
      onnxModel.metadata_props = [
        {
          key: 'advanced_graph_concat_merges',
          value: JSON.stringify(['bad-record']),
        },
      ];

      // Act
      attachOnnxConcatMergeMetadata(
        network,
        onnxModel.metadata_props ?? [],
        onnxModel,
      );

      // Assert
      expect(network._onnxAdvancedGraph).toBeUndefined();
    });

    it('ignores concat metadata when the payload is malformed JSON', () => {
      // Arrange
      const network = Network.createMLP(2, [2], 1) as AdvancedGraphAwareNetwork;
      const onnxModel = createConcatOnnxModel();
      onnxModel.metadata_props = [
        {
          key: 'advanced_graph_concat_merges',
          value: 'not-json',
        },
      ];

      // Act
      attachOnnxConcatMergeMetadata(
        network,
        onnxModel.metadata_props ?? [],
        onnxModel,
      );

      // Assert
      expect(network._onnxAdvancedGraph).toBeUndefined();
    });

    it('ignores concat metadata when the same-family ONNX validation fails', () => {
      // Arrange
      const network = Network.createMLP(2, [2], 1) as AdvancedGraphAwareNetwork;
      const onnxModel = createConcatOnnxModel();
      onnxModel.graph.node = [];

      // Act
      attachOnnxConcatMergeMetadata(
        network,
        onnxModel.metadata_props ?? [],
        onnxModel,
      );

      // Assert
      expect(network._onnxAdvancedGraph).toBeUndefined();
    });
  });

  describe('restoreConcatMergeConnections()', () => {
    it('returns without changes when the imported layer slices do not match the concat metadata', () => {
      // Arrange
      const network = Network.createMLP(2, [], 1);
      const onnxModel = createConcatOnnxModel();
      const outputNode = network.nodes.at(-1)!;

      // Act
      restoreConcatMergeConnections(network, onnxModel, [], onnxModel.metadata_props ?? []);

      // Assert
      expect(outputNode.connections.in.length).toBe(2);
    });

    it('returns without adding skip connections when the widened dense tensor width is inconsistent', () => {
      // Arrange
      const network = Network.createMLP(2, [2], 1);
      const onnxModel = createConcatOnnxModel([1, 3]);
      const outputNode = network.nodes.at(-1)!;

      // Act
      restoreConcatMergeConnections(
        network,
        onnxModel,
        [2],
        onnxModel.metadata_props ?? [],
      );

      // Assert
      expect(outputNode.connections.in.length).toBe(2);
    });

    it('updates an existing skip connection when the concat branch was already present on the runtime graph', () => {
      // Arrange
      const network = Network.createMLP(2, [2], 1);
      const firstInputNode = network.nodes[0];
      const secondInputNode = network.nodes[1];
      const outputNode = network.nodes.at(-1)!;
      firstInputNode.connect(outputNode, -1);
      secondInputNode.connect(outputNode, -2);
      Network.rebuildConnections(network);
      const onnxModel = createConcatOnnxModel();

      // Act
      restoreConcatMergeConnections(
        network,
        onnxModel,
        [2],
        onnxModel.metadata_props ?? [],
      );

      // Assert
      expect(
        outputNode.connections.in.find(
          (connection) => connection.from === firstInputNode,
        )?.weight,
      ).toBeCloseTo(0.3, 12);
    });

    it('falls back to zero when a concat tail weight is missing from the widened dense tensor', () => {
      // Arrange
      const network = Network.createMLP(2, [2], 1);
      const secondInputNode = network.nodes[1];
      const outputNode = network.nodes.at(-1)!;
      const onnxModel = createConcatOnnxModel();
      onnxModel.graph.initializer[0]!.float_data = [0.1, 0.2, 0.3];

      // Act
      restoreConcatMergeConnections(
        network,
        onnxModel,
        [2],
        onnxModel.metadata_props ?? [],
      );

      // Assert
      expect(
        outputNode.connections.in.find(
          (connection) => connection.from === secondInputNode,
        )?.weight,
      ).toBe(0);
    });
  });
});
