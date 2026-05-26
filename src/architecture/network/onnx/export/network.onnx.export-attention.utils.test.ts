import Node from '../../../node';
import type {
  OnnxMetadataProperty,
  OnnxModel,
  OnnxTensor,
} from '../schema/network.onnx.schema.types';
import type {
  AttentionMapping,
  OnnxExportOptions,
} from './network.onnx.export.types';
import { emitShadowAttentionMappings } from './network.onnx.export-attention.utils';

function createOnnxModel(
  initializerNames: string[] = ['W0', 'B0'],
  metadataProperties: OnnxMetadataProperty[] = [],
): OnnxModel {
  return {
    metadata_props: metadataProperties,
    graph: {
      inputs: [],
      outputs: [],
      initializer: initializerNames.map((initializerName) =>
        createInitializerTensor(initializerName),
      ),
      node: [],
    },
  };
}

function createInitializerTensor(initializerName: string): OnnxTensor {
  const initializerShape = initializerName.startsWith('W') ? [8, 8] : [8];
  const elementCount = initializerShape.reduce(
    (product, dimension) => product * dimension,
    1,
  );

  return {
    name: initializerName,
    data_type: 1,
    dims: initializerShape,
    float_data: Array.from({ length: elementCount }, () => 0),
  };
}

function createLayer(
  nodeType: 'input' | 'hidden' | 'output',
  nodeCount: number,
): Node[] {
  return Array.from({ length: nodeCount }, () => new Node(nodeType));
}

function createLayers(): Node[][] {
  return [
    createLayer('input', 8),
    createLayer('hidden', 8),
    createLayer('output', 2),
  ];
}

function createLayerOutputNames(): ReadonlyMap<number, string> {
  return new Map([
    [0, 'input'],
    [1, 'Layer_1'],
    [2, 'Layer_2'],
  ]);
}

function createIdentityWeights(width: number): number[] {
  return Array.from({ length: width * width }, (_unused, weightIndex) =>
    weightIndex % (width + 1) === 0 ? 1 : 0,
  );
}

function createAttentionMapping(
  overrides: Partial<AttentionMapping> = {},
): AttentionMapping {
  const projectionWeights = createIdentityWeights(4);

  return {
    layerIndex: 1,
    sequenceLength: 2,
    modelWidth: 4,
    heads: 2,
    queryWeights: projectionWeights,
    keyWeights: projectionWeights,
    valueWeights: projectionWeights,
    queryBias: [0, 0, 0, 0],
    keyBias: [0, 0, 0, 0],
    valueBias: [0, 0, 0, 0],
    ...overrides,
  };
}

function getMetadataValue(model: OnnxModel, key: string): string | undefined {
  return model.metadata_props?.find((entry) => entry.key === key)?.value;
}

function countNodes(model: OnnxModel, operatorType: string): number {
  return model.graph.node.filter(
    (nodeEntry) => nodeEntry.op_type === operatorType,
  ).length;
}

function hasInitializer(model: OnnxModel, initializerName: string): boolean {
  return model.graph.initializer.some(
    (initializerEntry) => initializerEntry.name === initializerName,
  );
}

describe('network onnx export attention chapter', () => {
  describe('emitShadowAttentionMappings', () => {
    it('emits the fixed-width attention shadow nodes and metadata for a valid mapping', () => {
      // Arrange
      const model = createOnnxModel();

      // Act
      emitShadowAttentionMappings(
        model,
        createLayers(),
        {
          attentionMappings: [createAttentionMapping()],
        } as OnnxExportOptions,
        createLayerOutputNames(),
        true,
      );

      // Assert
      expect({
        softmaxNodeNames: model.graph.node
          .filter((nodeEntry) => nodeEntry.op_type === 'Softmax')
          .map((nodeEntry) => nodeEntry.name),
        attentionMetadata: getMetadataValue(
          model,
          'advanced_graph_attention_blocks',
        ),
      }).toEqual({
        softmaxNodeNames: ['attention_l1_softmax'],
        attentionMetadata: JSON.stringify([
          {
            sourceLayerIndex: 0,
            targetLayerIndex: 1,
            sequenceLength: 2,
            modelWidth: 4,
            heads: 2,
            shadowOutputName: 'AttentionShadow_1',
          },
        ]),
      });
    });

    it('omits the scaling tensor and divide node when score scaling is disabled', () => {
      // Arrange
      const model = createOnnxModel();

      // Act
      emitShadowAttentionMappings(
        model,
        createLayers(),
        {
          attentionMappings: [createAttentionMapping({ scaleScores: false })],
        } as OnnxExportOptions,
        createLayerOutputNames(),
        true,
      );

      // Assert
      expect({
        hasScaleTensor: hasInitializer(model, 'AttentionScale_l1'),
        divideNodeCount: countNodes(model, 'Div'),
      }).toEqual({
        hasScaleTensor: false,
        divideNodeCount: 0,
      });
    });

    it('emits nodes without metadata when metadata inclusion is disabled', () => {
      // Arrange
      const model = createOnnxModel();

      // Act
      emitShadowAttentionMappings(
        model,
        createLayers(),
        {
          attentionMappings: [createAttentionMapping()],
        } as OnnxExportOptions,
        createLayerOutputNames(),
        false,
      );

      // Assert
      expect({
        softmaxNodeCount: countNodes(model, 'Softmax'),
        attentionMetadata: getMetadataValue(
          model,
          'advanced_graph_attention_blocks',
        ),
      }).toEqual({
        softmaxNodeCount: 1,
        attentionMetadata: undefined,
      });
    });

    it('appends attention metadata to an existing array payload', () => {
      // Arrange
      const model = createOnnxModel(
        ['W0', 'B0'],
        [
          {
            key: 'advanced_graph_attention_blocks',
            value: JSON.stringify([
              {
                sourceLayerIndex: 3,
                targetLayerIndex: 4,
                sequenceLength: 5,
                modelWidth: 6,
                heads: 2,
                shadowOutputName: 'AttentionShadow_4',
              },
            ]),
          },
        ],
      );

      // Act
      emitShadowAttentionMappings(
        model,
        createLayers(),
        {
          attentionMappings: [createAttentionMapping()],
        } as OnnxExportOptions,
        createLayerOutputNames(),
        true,
      );

      // Assert
      expect(getMetadataValue(model, 'advanced_graph_attention_blocks')).toBe(
        JSON.stringify([
          {
            sourceLayerIndex: 3,
            targetLayerIndex: 4,
            sequenceLength: 5,
            modelWidth: 6,
            heads: 2,
            shadowOutputName: 'AttentionShadow_4',
          },
          {
            sourceLayerIndex: 0,
            targetLayerIndex: 1,
            sequenceLength: 2,
            modelWidth: 4,
            heads: 2,
            shadowOutputName: 'AttentionShadow_1',
          },
        ]),
      );
    });

    it('replaces malformed metadata payloads with the new attention record', () => {
      // Arrange
      const model = createOnnxModel(
        ['W0', 'B0'],
        [
          {
            key: 'advanced_graph_attention_blocks',
            value: 'not-json',
          },
        ],
      );

      // Act
      emitShadowAttentionMappings(
        model,
        createLayers(),
        {
          attentionMappings: [createAttentionMapping()],
        } as OnnxExportOptions,
        createLayerOutputNames(),
        true,
      );

      // Assert
      expect(getMetadataValue(model, 'advanced_graph_attention_blocks')).toBe(
        JSON.stringify([
          {
            sourceLayerIndex: 0,
            targetLayerIndex: 1,
            sequenceLength: 2,
            modelWidth: 4,
            heads: 2,
            shadowOutputName: 'AttentionShadow_1',
          },
        ]),
      );
    });

    it('replaces valid non-array metadata payloads with the new attention record array', () => {
      // Arrange
      const model = createOnnxModel(
        ['W0', 'B0'],
        [
          {
            key: 'advanced_graph_attention_blocks',
            value: JSON.stringify({ unexpected: true }),
          },
        ],
      );

      // Act
      emitShadowAttentionMappings(
        model,
        createLayers(),
        {
          attentionMappings: [createAttentionMapping()],
        } as OnnxExportOptions,
        createLayerOutputNames(),
        true,
      );

      // Assert
      expect(getMetadataValue(model, 'advanced_graph_attention_blocks')).toBe(
        JSON.stringify([
          {
            sourceLayerIndex: 0,
            targetLayerIndex: 1,
            sequenceLength: 2,
            modelWidth: 4,
            heads: 2,
            shadowOutputName: 'AttentionShadow_1',
          },
        ]),
      );
    });

    it('ignores mappings whose source width does not match the declared sequence layout', () => {
      // Arrange
      const model = createOnnxModel();

      // Act
      emitShadowAttentionMappings(
        model,
        createLayers(),
        {
          attentionMappings: [createAttentionMapping({ sequenceLength: 3 })],
        } as OnnxExportOptions,
        createLayerOutputNames(),
        true,
      );

      // Assert
      expect({
        nodeCount: model.graph.node.length,
        metadata: getMetadataValue(model, 'advanced_graph_attention_blocks'),
      }).toEqual({
        nodeCount: 0,
        metadata: undefined,
      });
    });

    it('ignores mappings when the previous layer output name is unavailable', () => {
      // Arrange
      const model = createOnnxModel();

      // Act
      emitShadowAttentionMappings(
        model,
        createLayers(),
        {
          attentionMappings: [createAttentionMapping()],
        } as OnnxExportOptions,
        new Map([[1, 'Layer_1']]),
        true,
      );

      // Assert
      expect(model.graph.node).toEqual([]);
    });

    it('ignores mappings when the resolved source layer bucket is missing', () => {
      // Arrange
      const model = createOnnxModel();
      const sparseLayers: Node[][] = [];

      sparseLayers[1] = createLayer('hidden', 8);
      sparseLayers[2] = createLayer('output', 2);

      // Act
      emitShadowAttentionMappings(
        model,
        sparseLayers,
        {
          attentionMappings: [createAttentionMapping()],
        } as OnnxExportOptions,
        createLayerOutputNames(),
        true,
      );

      // Assert
      expect(model.graph.node).toEqual([]);
    });

    it('ignores mappings when the dense output projection tensors are unavailable', () => {
      // Arrange
      const model = createOnnxModel(['B0']);

      // Act
      emitShadowAttentionMappings(
        model,
        createLayers(),
        {
          attentionMappings: [createAttentionMapping()],
        } as OnnxExportOptions,
        createLayerOutputNames(),
        true,
      );

      // Assert
      expect(model.graph.node).toEqual([]);
    });

    it('ignores duplicate mappings that target the same layer', () => {
      // Arrange
      const model = createOnnxModel();

      // Act
      emitShadowAttentionMappings(
        model,
        createLayers(),
        {
          attentionMappings: [
            createAttentionMapping(),
            createAttentionMapping(),
          ],
        } as OnnxExportOptions,
        createLayerOutputNames(),
        true,
      );

      // Assert
      expect(countNodes(model, 'Softmax')).toBe(1);
    });

    it('ignores mappings whose model width is not divisible by the head count', () => {
      // Arrange
      const model = createOnnxModel();

      // Act
      emitShadowAttentionMappings(
        model,
        createLayers(),
        {
          attentionMappings: [
            createAttentionMapping({
              modelWidth: 5,
              heads: 2,
              queryWeights: Array.from({ length: 25 }, () => 0),
              keyWeights: Array.from({ length: 25 }, () => 0),
              valueWeights: Array.from({ length: 25 }, () => 0),
              queryBias: [0, 0, 0, 0, 0],
              keyBias: [0, 0, 0, 0, 0],
              valueBias: [0, 0, 0, 0, 0],
            }),
          ],
        } as OnnxExportOptions,
        createLayerOutputNames(),
        true,
      );

      // Assert
      expect(model.graph.node).toEqual([]);
    });

    it('ignores mappings that target the input layer', () => {
      // Arrange
      const model = createOnnxModel();

      // Act
      emitShadowAttentionMappings(
        model,
        createLayers(),
        {
          attentionMappings: [createAttentionMapping({ layerIndex: 0 })],
        } as OnnxExportOptions,
        createLayerOutputNames(),
        true,
      );

      // Assert
      expect(model.graph.node).toEqual([]);
    });

    it('ignores mappings whose projection tensors do not match the declared model width', () => {
      // Arrange
      const model = createOnnxModel();

      // Act
      emitShadowAttentionMappings(
        model,
        createLayers(),
        {
          attentionMappings: [
            createAttentionMapping({
              queryWeights: Array.from({ length: 3 }, () => 0),
            }),
          ],
        } as OnnxExportOptions,
        createLayerOutputNames(),
        true,
      );

      // Assert
      expect(model.graph.node).toEqual([]);
    });

    it.each([
      {
        testLabel: 'sequence length is not positive',
        overrides: { sequenceLength: 0 },
      },
      {
        testLabel: 'model width is not positive',
        overrides: {
          modelWidth: 0,
          queryWeights: [],
          keyWeights: [],
          valueWeights: [],
          queryBias: [],
          keyBias: [],
          valueBias: [],
        },
      },
      {
        testLabel: 'head count is not positive',
        overrides: { heads: 0 },
      },
    ])('ignores mappings when $testLabel', ({ overrides }) => {
      // Arrange
      const model = createOnnxModel();

      // Act
      emitShadowAttentionMappings(
        model,
        createLayers(),
        {
          attentionMappings: [createAttentionMapping(overrides)],
        } as OnnxExportOptions,
        createLayerOutputNames(),
        true,
      );

      // Assert
      expect(model.graph.node).toEqual([]);
    });

    it('leaves the graph unchanged when no attention mappings are declared', () => {
      // Arrange
      const model = createOnnxModel();

      // Act
      emitShadowAttentionMappings(
        model,
        createLayers(),
        {},
        createLayerOutputNames(),
        true,
      );

      // Assert
      expect({
        initializers: model.graph.initializer.length,
        nodes: model.graph.node.length,
      }).toEqual({
        initializers: 2,
        nodes: 0,
      });
    });
  });
});
