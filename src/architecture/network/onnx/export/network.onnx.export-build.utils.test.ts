import Node from '../../../node';
import * as methods from '../../../../methods/methods';
import { buildOnnxModel } from './network.onnx.export-build.utils';
import type { OnnxExportOptions } from './network.onnx.export.types';

function createLayer(
  nodeType: 'input' | 'hidden' | 'output',
  nodeCount: number,
): Node[] {
  return Array.from({ length: nodeCount }, () => new Node(nodeType));
}

describe('network onnx export build utils', () => {
  describe('buildOnnxModel', () => {
    it('uses default options when the options argument is omitted', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];

      // Act
      const onnxModel = buildOnnxModel({} as never, layers);

      // Assert
      expect(onnxModel.graph.outputs[0]?.name).toBe('output');
    });

    it('collects hidden layer metadata when a hidden layer exists', () => {
      // Arrange
      const layers = [
        createLayer('input', 2),
        createLayer('hidden', 3),
        createLayer('output', 1),
      ];

      // Act
      const onnxModel = buildOnnxModel({} as never, layers, {
        includeMetadata: true,
      });

      // Assert
      expect(onnxModel.graph.outputs[0]?.name).toBe('output');
    });

    it('rejects simultaneous storage-fp16 precision and quantization requests', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        precision: { mode: 'storage-fp16' },
        quantization: {
          mode: 'dynamic-uint8',
          target: 'dense',
          representation: 'metadata-only',
        },
      } as unknown as OnnxExportOptions;

      // Act
      const buildModel = () => buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(buildModel).toThrow(/storage-fp16.*quantization/i);
    });

    it('rejects a generic dynamic-int8 quantization mode claim', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        quantization: {
          mode: 'dynamic-int8',
        },
      } as unknown as OnnxExportOptions;

      // Act
      const buildModel = () => buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(buildModel).toThrow(/dynamic-uint8/i);
    });

    it('rejects unsupported precision mode labels', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        precision: {
          mode: 'float16',
        },
      } as unknown as OnnxExportOptions;

      // Act
      const buildModel = () => buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(buildModel).toThrow(/precision mode must be float32 or storage-fp16/i);
    });

    it('rejects dynamic quantization requests that target conv operators', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        quantization: {
          mode: 'dynamic-uint8',
          target: 'conv',
          representation: 'DynamicQuantizeLinear',
        },
      } as unknown as OnnxExportOptions;

      // Act
      const buildModel = () => buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(buildModel).toThrow(/dynamic quantization.*dense/i);
    });

    it('accepts the documented dynamic-uint8 dense packet', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        quantization: {
          mode: 'dynamic-uint8',
          target: 'dense',
          representation: 'DynamicQuantizeLinear',
        },
      } as unknown as OnnxExportOptions;

      // Act
      const onnxModel = buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(onnxModel.graph.outputs[0]?.name).toBe('output');
    });

    it('defaults omitted dynamic-uint8 target and representation fields to the documented dense metadata-only lane', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        quantization: {
          mode: 'dynamic-uint8',
        },
      } as unknown as OnnxExportOptions;

      // Act
      const onnxModel = buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(onnxModel.graph.outputs[0]?.name).toBe('output');
    });

    it('rejects unsupported dynamic quantization representation labels', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        quantization: {
          mode: 'dynamic-uint8',
          target: 'dense',
          representation: 'qlinear',
        },
      } as unknown as OnnxExportOptions;

      // Act
      const buildModel = () => buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(buildModel).toThrow(/DynamicQuantizeLinear or metadata-only/i);
    });

    it('rejects static-8bit packets that omit every dense or conv target', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        quantization: {
          mode: 'static-8bit',
          targets: [],
          calibration: {
            source: 'external',
          },
        },
      } as unknown as OnnxExportOptions;

      // Act
      const buildModel = () => buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(buildModel).toThrow(/at least one dense or conv target/i);
    });

    it('rejects static-8bit packets whose targets field is not an array', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        quantization: {
          mode: 'static-8bit',
          targets: 'dense',
          calibration: {
            source: 'external',
            layerTargets: [
              {
                target: 'dense',
                layerIndex: 1,
                inputRange: { min: -1, max: 1 },
                outputRange: { min: -0.5, max: 0.75 },
              },
            ],
          },
        },
      } as unknown as OnnxExportOptions;

      // Act
      const buildModel = () => buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(buildModel).toThrow(/at least one dense or conv target/i);
    });

    it('rejects static-8bit packets without an external calibration packet', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        quantization: {
          mode: 'static-8bit',
          targets: ['dense'],
        },
      } as unknown as OnnxExportOptions;

      // Act
      const buildModel = () => buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(buildModel).toThrow(/external calibration packet/i);
    });

    it('accepts static-8bit packets with non-default encodings, filtered targets, and qdq representation', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        quantization: {
          mode: 'static-8bit',
          targets: ['dense', 'unsupported'],
          calibration: {
            source: 'external',
            packetId: 'calibration-packet',
            sampleCount: 16,
            layerTargets: [
              {
                target: 'dense',
                layerIndex: 1,
                inputRange: { min: -1, max: 1 },
                outputRange: { min: -0.5, max: 0.75 },
              },
            ],
            activationSymmetry: 'symmetric',
            weightSymmetry: 'asymmetric',
            zeroInclusion: 'required',
            weightRangePolicy: 'min-max',
            roundingMode: 'nearest-even',
          },
          activationEncoding: 'int8',
          weightEncoding: 'uint8',
          weightGranularity: 'per-tensor',
          representation: 'qdq',
        },
      } as unknown as OnnxExportOptions;

      // Act
      const onnxModel = buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(onnxModel.graph.outputs[0]?.name).toBe('output');
    });

    it('rejects static-8bit packets without explicit calibration layer targets', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        quantization: {
          mode: 'static-8bit',
          targets: ['dense'],
          calibration: {
            source: 'external',
          },
        },
      } as unknown as OnnxExportOptions;

      // Act
      const buildModel = () => buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(buildModel).toThrow(/calibration layer targets/i);
    });

    it('rejects static-8bit dense requests that ask for per-output-channel weights', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        quantization: {
          mode: 'static-8bit',
          targets: ['dense'],
          calibration: {
            source: 'external',
            layerTargets: [
              {
                target: 'dense',
                layerIndex: 1,
                inputRange: { min: -1, max: 1 },
                outputRange: { min: -0.5, max: 0.75 },
              },
            ],
          },
          weightGranularity: 'per-output-channel',
        },
      } as unknown as OnnxExportOptions;

      // Act
      const buildModel = () => buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(buildModel).toThrow(/per-output-channel.*conv/i);
    });

    it('rejects static-8bit calibration packets with unsupported weight range policies', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        quantization: {
          mode: 'static-8bit',
          targets: ['dense'],
          calibration: {
            source: 'external',
            weightRangePolicy: 'percentile',
            layerTargets: [
              {
                target: 'dense',
                layerIndex: 1,
                inputRange: { min: -1, max: 1 },
                outputRange: { min: -0.5, max: 0.75 },
              },
            ],
          },
        },
      } as unknown as OnnxExportOptions;

      // Act
      const buildModel = () => buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(buildModel).toThrow(/supports the min-max weight range policy only/i);
    });

    it('rejects static-8bit calibration packets with unsupported zero-inclusion policies', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        quantization: {
          mode: 'static-8bit',
          targets: ['dense'],
          calibration: {
            source: 'external',
            zeroInclusion: 'optional',
            layerTargets: [
              {
                target: 'dense',
                layerIndex: 1,
                inputRange: { min: -1, max: 1 },
                outputRange: { min: -0.5, max: 0.75 },
              },
            ],
          },
        },
      } as unknown as OnnxExportOptions;

      // Act
      const buildModel = () => buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(buildModel).toThrow(/requires zero-inclusive ranges/i);
    });

    it('rejects static-8bit calibration packets with unsupported rounding modes', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        quantization: {
          mode: 'static-8bit',
          targets: ['dense'],
          calibration: {
            source: 'external',
            roundingMode: 'toward-zero',
            layerTargets: [
              {
                target: 'dense',
                layerIndex: 1,
                inputRange: { min: -1, max: 1 },
                outputRange: { min: -0.5, max: 0.75 },
              },
            ],
          },
        },
      } as unknown as OnnxExportOptions;

      // Act
      const buildModel = () => buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(buildModel).toThrow(/supports nearest-even rounding only/i);
    });

    it('rejects static-8bit conv calibration targets when no resolved Conv mapping exists', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        quantization: {
          mode: 'static-8bit',
          targets: ['conv'],
          calibration: {
            source: 'external',
            layerTargets: [
              {
                target: 'conv',
                layerIndex: 1,
                inputRange: { min: -1, max: 1 },
                outputRange: { min: -0.5, max: 0.75 },
              },
            ],
          },
        },
      } as unknown as OnnxExportOptions;

      // Act
      const buildModel = () => buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(buildModel).toThrow(/resolved conv mapping/i);
    });

    it('rejects static-8bit calibration layer targets outside the requested operator families', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        quantization: {
          mode: 'static-8bit',
          targets: ['dense'],
          calibration: {
            source: 'external',
            layerTargets: [
              {
                target: 'conv',
                layerIndex: 1,
                inputRange: { min: -1, max: 1 },
                outputRange: { min: -0.5, max: 0.75 },
              },
            ],
          },
        },
      } as unknown as OnnxExportOptions;

      // Act
      const buildModel = () => buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(buildModel).toThrow(/must stay within the requested operator families/i);
    });

    it('rejects duplicate static-8bit calibration layer targets', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        quantization: {
          mode: 'static-8bit',
          targets: ['dense'],
          calibration: {
            source: 'external',
            layerTargets: [
              {
                target: 'dense',
                layerIndex: 1,
                inputRange: { min: -1, max: 1 },
                outputRange: { min: -0.5, max: 0.75 },
              },
              {
                target: 'dense',
                layerIndex: 1,
                inputRange: { min: -1, max: 1 },
                outputRange: { min: -0.5, max: 0.75 },
              },
            ],
          },
        },
      } as unknown as OnnxExportOptions;

      // Act
      const buildModel = () => buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(buildModel).toThrow(/must be unique per operator family and layer index/i);
    });

    it('rejects static-8bit calibration layer targets that are not objects', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        quantization: {
          mode: 'static-8bit',
          targets: ['dense'],
          calibration: {
            source: 'external',
            layerTargets: ['dense'],
          },
        },
      } as unknown as OnnxExportOptions;

      // Act
      const buildModel = () => buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(buildModel).toThrow(/layer targets must be objects/i);
    });

    it('rejects static-8bit calibration layer targets with unknown operator families', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        quantization: {
          mode: 'static-8bit',
          targets: ['dense'],
          calibration: {
            source: 'external',
            layerTargets: [
              {
                target: 'pool',
                layerIndex: 1,
                inputRange: { min: -1, max: 1 },
                outputRange: { min: -0.5, max: 0.75 },
              },
            ],
          },
        },
      } as unknown as OnnxExportOptions;

      // Act
      const buildModel = () => buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(buildModel).toThrow(/must name a dense or conv operator family/i);
    });

    it('rejects static-8bit calibration layer targets that reference an invalid export layer', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        quantization: {
          mode: 'static-8bit',
          targets: ['dense'],
          calibration: {
            source: 'external',
            layerTargets: [
              {
                target: 'dense',
                layerIndex: 2,
                inputRange: { min: -1, max: 1 },
                outputRange: { min: -0.5, max: 0.75 },
              },
            ],
          },
        },
      } as unknown as OnnxExportOptions;

      // Act
      const buildModel = () => buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(buildModel).toThrow(/must reference a valid non-input export layer/i);
    });

    it('rejects static-8bit dense calibration targets that point at Conv-mapped layers', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        conv2dMappings: [
          {
            layerIndex: 1,
            inHeight: 1,
            inWidth: 2,
            inChannels: 1,
            kernelHeight: 1,
            kernelWidth: 2,
            strideHeight: 1,
            strideWidth: 1,
            outHeight: 1,
            outWidth: 1,
            outChannels: 1,
          },
        ],
        quantization: {
          mode: 'static-8bit',
          targets: ['dense'],
          calibration: {
            source: 'external',
            layerTargets: [
              {
                target: 'dense',
                layerIndex: 1,
                inputRange: { min: -1, max: 1 },
                outputRange: { min: -0.5, max: 0.75 },
              },
            ],
          },
        },
      } as unknown as OnnxExportOptions;

      // Act
      const buildModel = () => buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(buildModel).toThrow(/cannot target layers emitted as Conv operators/i);
    });

    it('rejects static-8bit calibration targets whose output range is unstable', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        quantization: {
          mode: 'static-8bit',
          targets: ['dense'],
          calibration: {
            source: 'external',
            layerTargets: [
              {
                target: 'dense',
                layerIndex: 1,
                inputRange: { min: -1, max: 1 },
                outputRange: { min: 0.5, max: 0.5 },
              },
            ],
          },
        },
      } as unknown as OnnxExportOptions;

      // Act
      const buildModel = () => buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(buildModel).toThrow(/requires min strictly less than max/i);
    });

    it('rejects static-8bit calibration targets whose input range is not finite', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        quantization: {
          mode: 'static-8bit',
          targets: ['dense'],
          calibration: {
            source: 'external',
            layerTargets: [
              {
                target: 'dense',
                layerIndex: 1,
                inputRange: { min: Number.NaN, max: 1 },
                outputRange: { min: -0.5, max: 0.5 },
              },
            ],
          },
        },
      } as unknown as OnnxExportOptions;

      // Act
      const buildModel = () => buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(buildModel).toThrow(/requires finite input min and max values/i);
    });

    it('rejects static-8bit calibration packets whose sample count is not a positive integer', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        quantization: {
          mode: 'static-8bit',
          targets: ['dense'],
          calibration: {
            source: 'external',
            sampleCount: 0,
            layerTargets: [
              {
                target: 'dense',
                layerIndex: 1,
                inputRange: { min: -1, max: 1 },
                outputRange: { min: -0.5, max: 0.75 },
              },
            ],
          },
        },
      } as unknown as OnnxExportOptions;

      // Act
      const buildModel = () => buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(buildModel).toThrow(/sample counts must be positive integers/i);
    });

    it('keeps qlinear dense lowering disabled when pool mappings are present', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        pool2dMappings: [{}],
        quantization: {
          mode: 'static-8bit',
          targets: ['dense'],
          calibration: {
            source: 'external',
            layerTargets: [
              {
                target: 'dense',
                layerIndex: 1,
                inputRange: { min: -1, max: 1 },
                outputRange: { min: -0.5, max: 0.75 },
              },
            ],
          },
          representation: 'qlinear',
        },
      } as unknown as OnnxExportOptions;

      // Act
      const onnxModel = buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(
        onnxModel.graph.node.some(
          (graphNode) => graphNode.op_type === 'QLinearMatMul',
        ),
      ).toBe(false);
    });

    it('emits an explicit bias bridge and preserves the dense activation during qlinear lowering', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      layers[1][0]!.squash = methods.Activation.relu;
      layers[1][0]!.bias = 0.25;
      const exportOptions = {
        quantization: {
          mode: 'static-8bit',
          targets: ['dense'],
          calibration: {
            source: 'external',
            layerTargets: [
              {
                target: 'dense',
                layerIndex: 1,
                inputRange: { min: -1, max: 1 },
                outputRange: { min: -0.5, max: 0.75 },
              },
            ],
          },
          representation: 'qlinear',
        },
      } as unknown as OnnxExportOptions;

      // Act
      const onnxModel = buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect({
        hasQLinearMatMul: onnxModel.graph.node.some(
          (graphNode) => graphNode.op_type === 'QLinearMatMul',
        ),
        hasBiasAddBridge: onnxModel.graph.node.some(
          (graphNode) => graphNode.op_type === 'Add' && graphNode.name === 'bias_add_l1',
        ),
        hasReluActivation: onnxModel.graph.node.some(
          (graphNode) => graphNode.op_type === 'Relu' && graphNode.name === 'act_l1',
        ),
      }).toEqual({
        hasQLinearMatMul: true,
        hasBiasAddBridge: true,
        hasReluActivation: true,
      });
    });

    it('uses the bias bridge output as the layer output when the dense activation remains identity', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      layers[1][0]!.squash = methods.Activation.identity;
      layers[1][0]!.bias = 0.25;
      const exportOptions = {
        quantization: {
          mode: 'static-8bit',
          targets: ['dense'],
          calibration: {
            source: 'external',
            layerTargets: [
              {
                target: 'dense',
                layerIndex: 1,
                inputRange: { min: -1, max: 1 },
                outputRange: { min: -0.5, max: 0.75 },
              },
            ],
          },
          representation: 'qlinear',
        },
      } as unknown as OnnxExportOptions;

      // Act
      const onnxModel = buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect({
        hasQLinearMatMul: onnxModel.graph.node.some(
          (graphNode) => graphNode.op_type === 'QLinearMatMul',
        ),
        hasBiasAddBridge: onnxModel.graph.node.some(
          (graphNode) => graphNode.op_type === 'Add' && graphNode.name === 'bias_add_l1',
        ),
        hasActivationNode: onnxModel.graph.node.some(
          (graphNode) => graphNode.name === 'act_l1',
        ),
        biasBridgeOutputName: onnxModel.graph.node.find(
          (graphNode) => graphNode.name === 'bias_add_l1',
        )?.output[0],
      }).toEqual({
        hasQLinearMatMul: true,
        hasBiasAddBridge: true,
        hasActivationNode: false,
        biasBridgeOutputName: 'Layer_1',
      });
    });

    it('emits uint8 dense weight tensors when qlinear lowering requests uint8 weight encoding', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      layers[1][0]!.squash = methods.Activation.identity;
      layers[1][0]!.bias = 0;
      const exportOptions = {
        quantization: {
          mode: 'static-8bit',
          targets: ['dense'],
          calibration: {
            source: 'external',
            layerTargets: [
              {
                target: 'dense',
                layerIndex: 1,
                inputRange: { min: -1, max: 1 },
                outputRange: { min: -0.5, max: 0.75 },
              },
            ],
            weightSymmetry: 'symmetric',
          },
          representation: 'qlinear',
          weightEncoding: 'uint8',
        },
      } as unknown as OnnxExportOptions;

      // Act
      const onnxModel = buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(
        onnxModel.graph.initializer.find(
          (initializerEntry) => initializerEntry.name === 'QuantDenseWeight_l1',
        )?.data_type,
      ).toBe(2);
    });

    it('rejects static-8bit calibration layer targets without an input range object', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];
      const exportOptions = {
        quantization: {
          mode: 'static-8bit',
          targets: ['dense'],
          calibration: {
            source: 'external',
            layerTargets: [
              {
                target: 'dense',
                layerIndex: 1,
                outputRange: { min: -0.5, max: 0.75 },
              },
            ],
          },
        },
      } as unknown as OnnxExportOptions;

      // Act
      const buildModel = () => buildOnnxModel({} as never, layers, exportOptions);

      // Assert
      expect(buildModel).toThrow(/requires a input range object/i);
    });
  });
});
