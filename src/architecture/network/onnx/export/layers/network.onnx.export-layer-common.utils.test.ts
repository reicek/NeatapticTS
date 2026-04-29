import Node from '../../../../node';
import type {
  OnnxMetadataProperty,
  OnnxModel,
  Pool2DMapping,
} from '../../schema/network.onnx.schema.types';
import {
  appendIndexedMetadata,
  appendMetadataSpec,
  buildDenseWeightsAndBiases,
  buildDiagonalRecurrentWeights,
  emitOptionalPoolingAndFlatten,
} from './network.onnx.export-layer-common.utils';

jest.retryTimes(2, { logErrorsBeforeRetry: true });

function createOnnxModel(
  options: {
    metadataProps?: OnnxMetadataProperty[];
  } = {},
): OnnxModel {
  return {
    graph: {
      inputs: [],
      outputs: [],
      initializer: [],
      node: [],
    },
    metadata_props: options.metadataProps,
  };
}

function createPoolSpec(overrides: Partial<Pool2DMapping> = {}): Pool2DMapping {
  return {
    afterLayerIndex: 1,
    type: 'MaxPool',
    kernelHeight: 2,
    kernelWidth: 2,
    strideHeight: 2,
    strideWidth: 2,
    ...overrides,
  };
}

function findMetadataProperty(
  model: OnnxModel,
  key: string,
): OnnxMetadataProperty | undefined {
  return (model.metadata_props ?? []).find(
    (metadataEntry) => metadataEntry.key === key,
  );
}

function readParsedMetadataValue<ItemType>(
  model: OnnxModel,
  key: string,
): ItemType | undefined {
  const metadataEntry = findMetadataProperty(model, key);
  return metadataEntry
    ? (JSON.parse(metadataEntry.value) as ItemType)
    : undefined;
}

describe('network onnx export layer common utility chapter', () => {
  describe('buildDenseWeightsAndBiases()', () => {
    describe('given the destination layer is partially connected', () => {
      it('fills missing inbound edges with zero while preserving bias order', () => {
        // Arrange
        const firstInputNode = new Node('input');
        const secondInputNode = new Node('input');
        const firstHiddenNode = new Node('hidden');
        const secondHiddenNode = new Node('hidden');

        firstHiddenNode.bias = 0.1;
        secondHiddenNode.bias = -0.2;
        firstInputNode.connect(firstHiddenNode, 0.5);
        secondInputNode.connect(secondHiddenNode, -0.25);

        // Act
        const denseInitializers = buildDenseWeightsAndBiases(
          [firstInputNode, secondInputNode],
          [firstHiddenNode, secondHiddenNode],
        );

        // Assert
        expect(denseInitializers).toEqual({
          weightMatrixValues: [0.5, 0, 0, -0.25],
          biasVector: [0.1, -0.2],
        });
      });
    });
  });

  describe('buildDiagonalRecurrentWeights()', () => {
    describe('given only part of the layer owns self-connections', () => {
      it('keeps recurrent weights on the diagonal and zeroes every other slot', () => {
        // Arrange
        const firstHiddenNode = new Node('hidden');
        const secondHiddenNode = new Node('hidden');

        firstHiddenNode.connect(firstHiddenNode, 0.75);

        // Act
        const recurrentWeights = buildDiagonalRecurrentWeights([
          firstHiddenNode,
          secondHiddenNode,
        ]);

        // Assert
        expect(recurrentWeights).toEqual([0.75, 0, 0, 0]);
      });
    });
  });

  describe('emitOptionalPoolingAndFlatten()', () => {
    describe('given pooling is not configured', () => {
      it('returns the incoming tensor name without mutating the graph', () => {
        // Arrange
        const onnxModel = createOnnxModel();

        // Act
        const outputName = emitOptionalPoolingAndFlatten({
          model: onnxModel,
          options: { flattenAfterPooling: true },
          layerIndex: 2,
          sourceOutputName: 'Layer_2',
        });

        // Assert
        expect({
          outputName,
          metadataProps: onnxModel.metadata_props,
          nodes: onnxModel.graph.node,
        }).toEqual({
          outputName: 'Layer_2',
          metadataProps: undefined,
          nodes: [],
        });
      });
    });

    describe('given pooling uses default padding and no flatten stage', () => {
      it('emits one pooling node and the corresponding metadata hints', () => {
        // Arrange
        const onnxModel = createOnnxModel();
        const poolSpec = createPoolSpec({
          afterLayerIndex: 3,
          type: 'AveragePool',
        });

        // Act
        const outputName = emitOptionalPoolingAndFlatten({
          model: onnxModel,
          options: { flattenAfterPooling: false },
          layerIndex: 3,
          sourceOutputName: 'Hidden_3',
          poolSpec,
        });

        // Assert
        expect({
          outputName,
          poolLayers: readParsedMetadataValue<number[]>(
            onnxModel,
            'pool2d_layers',
          ),
          poolSpecs: readParsedMetadataValue<Pool2DMapping[]>(
            onnxModel,
            'pool2d_specs',
          ),
          nodes: onnxModel.graph.node,
        }).toEqual({
          outputName: 'Pool_3',
          poolLayers: [3],
          poolSpecs: [poolSpec],
          nodes: [
            {
              op_type: 'AveragePool',
              input: ['Hidden_3'],
              output: ['Pool_3'],
              name: 'pool_after_l3',
              attributes: [
                { name: 'kernel_shape', type: 'INTS', ints: [2, 2] },
                { name: 'strides', type: 'INTS', ints: [2, 2] },
                { name: 'pads', type: 'INTS', ints: [0, 0, 0, 0] },
              ],
            },
          ],
        });
      });
    });

    describe('given pooling uses explicit padding and flatten is enabled', () => {
      it('emits pooling and flatten nodes plus flatten metadata', () => {
        // Arrange
        const onnxModel = createOnnxModel();
        const poolSpec = createPoolSpec({
          afterLayerIndex: 4,
          padTop: 1,
          padLeft: 2,
          padBottom: 3,
          padRight: 4,
        });

        // Act
        const outputName = emitOptionalPoolingAndFlatten({
          model: onnxModel,
          options: { flattenAfterPooling: true },
          layerIndex: 4,
          sourceOutputName: 'Hidden_4',
          poolSpec,
        });

        // Assert
        expect({
          outputName,
          flattenLayers: readParsedMetadataValue<number[]>(
            onnxModel,
            'flatten_layers',
          ),
          poolLayers: readParsedMetadataValue<number[]>(
            onnxModel,
            'pool2d_layers',
          ),
          poolSpecs: readParsedMetadataValue<Pool2DMapping[]>(
            onnxModel,
            'pool2d_specs',
          ),
          nodes: onnxModel.graph.node,
        }).toEqual({
          outputName: 'PoolFlat_4',
          flattenLayers: [4],
          poolLayers: [4],
          poolSpecs: [poolSpec],
          nodes: [
            {
              op_type: 'MaxPool',
              input: ['Hidden_4'],
              output: ['Pool_4'],
              name: 'pool_after_l4',
              attributes: [
                { name: 'kernel_shape', type: 'INTS', ints: [2, 2] },
                { name: 'strides', type: 'INTS', ints: [2, 2] },
                { name: 'pads', type: 'INTS', ints: [1, 2, 3, 4] },
              ],
            },
            {
              op_type: 'Flatten',
              input: ['Pool_4'],
              output: ['PoolFlat_4'],
              name: 'flatten_after_l4',
              attributes: [{ name: 'axis', type: 'INT', i: 1 }],
            },
          ],
        });
      });
    });
  });

  describe('appendIndexedMetadata()', () => {
    describe('given the key already stores other layer indexes', () => {
      it('appends the new layer index to the existing JSON array', () => {
        // Arrange
        const onnxModel = createOnnxModel({
          metadataProps: [{ key: 'flatten_layers', value: '[1]' }],
        });

        // Act
        appendIndexedMetadata(onnxModel, 'flatten_layers', 4);

        // Assert
        expect(
          readParsedMetadataValue<number[]>(onnxModel, 'flatten_layers'),
        ).toEqual([1, 4]);
      });
    });

    describe('given the key already stores the requested layer index', () => {
      it('keeps the existing JSON array unchanged', () => {
        // Arrange
        const onnxModel = createOnnxModel({
          metadataProps: [{ key: 'flatten_layers', value: '[1,4]' }],
        });

        // Act
        appendIndexedMetadata(onnxModel, 'flatten_layers', 4);

        // Assert
        expect(
          readParsedMetadataValue<number[]>(onnxModel, 'flatten_layers'),
        ).toEqual([1, 4]);
      });
    });

    describe('given the key stores malformed JSON', () => {
      it('resets the metadata value to a one-item array', () => {
        // Arrange
        const onnxModel = createOnnxModel({
          metadataProps: [{ key: 'flatten_layers', value: '{' }],
        });

        // Act
        appendIndexedMetadata(onnxModel, 'flatten_layers', 2);

        // Assert
        expect(
          readParsedMetadataValue<number[]>(onnxModel, 'flatten_layers'),
        ).toEqual([2]);
      });
    });

    describe('given the key stores JSON that is not an array', () => {
      it('replaces the value with a one-item array', () => {
        // Arrange
        const onnxModel = createOnnxModel({
          metadataProps: [{ key: 'flatten_layers', value: '{}' }],
        });

        // Act
        appendIndexedMetadata(onnxModel, 'flatten_layers', 3);

        // Assert
        expect(
          readParsedMetadataValue<number[]>(onnxModel, 'flatten_layers'),
        ).toEqual([3]);
      });
    });
  });

  describe('appendMetadataSpec()', () => {
    describe('given the key already stores pooling specs', () => {
      it('appends the new spec to the existing JSON array', () => {
        // Arrange
        const existingPoolSpec = createPoolSpec({ afterLayerIndex: 2 });
        const nextPoolSpec = createPoolSpec({
          afterLayerIndex: 5,
          type: 'AveragePool',
        });
        const onnxModel = createOnnxModel({
          metadataProps: [
            {
              key: 'pool2d_specs',
              value: JSON.stringify([existingPoolSpec]),
            },
          ],
        });

        // Act
        appendMetadataSpec(onnxModel, 'pool2d_specs', nextPoolSpec);

        // Assert
        expect(
          readParsedMetadataValue<Pool2DMapping[]>(onnxModel, 'pool2d_specs'),
        ).toEqual([existingPoolSpec, nextPoolSpec]);
      });
    });

    describe('given the key stores malformed JSON', () => {
      it('restarts the metadata value with only the requested spec', () => {
        // Arrange
        const nextPoolSpec = createPoolSpec({ afterLayerIndex: 6 });
        const onnxModel = createOnnxModel({
          metadataProps: [{ key: 'pool2d_specs', value: '{' }],
        });

        // Act
        appendMetadataSpec(onnxModel, 'pool2d_specs', nextPoolSpec);

        // Assert
        expect(
          readParsedMetadataValue<Pool2DMapping[]>(onnxModel, 'pool2d_specs'),
        ).toEqual([nextPoolSpec]);
      });
    });
  });
});
