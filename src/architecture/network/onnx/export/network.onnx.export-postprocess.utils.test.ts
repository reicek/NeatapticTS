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

function getFirstOperatorInput(
  model: OnnxModel,
  operatorType: string,
): string | undefined {
  return model.graph.node.find((nodeEntry) => nodeEntry.op_type === operatorType)
    ?.input[0];
}

describe('network onnx export postprocess chapter', () => {
  describe('emitFusedRecurrentHeuristics', () => {
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
        finalizeExportMetadata(
          onnxModel,
          layers,
          exportOptions,
          true,
          [1],
          [],
        );

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
        finalizeExportMetadata(
          onnxModel,
          layers,
          exportOptions,
          true,
          [1],
          [],
        );

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
        finalizeExportMetadata(
          onnxModel,
          layers,
          exportOptions,
          true,
          [2],
          [],
        );

        // Assert
        expect(getMetadataValue(onnxModel, 'conv2d_sharing_verified')).toBe(
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
        finalizeExportMetadata(
          onnxModel,
          layers,
          exportOptions,
          true,
          [3],
          [],
        );

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
        finalizeExportMetadata(
          onnxModel,
          layers,
          exportOptions,
          true,
          [4],
          [],
        );

        // Assert
        expect(getMetadataValue(onnxModel, 'conv2d_sharing_verified')).toBe(
          '[1]',
        );
      });
    });
  });
});