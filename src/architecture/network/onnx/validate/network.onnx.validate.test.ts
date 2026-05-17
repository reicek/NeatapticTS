import onnxProto from 'onnx-proto';
import { InferenceSession } from 'onnxruntime-node';
import Network from '../../network';
import {
  CURRENT_ONNX_REFERENCE_OPSET,
  ONNX_STANDARD_DOMAIN,
} from '../export/network.onnx.export-setup.utils';
import { exportToONNXBinary } from '../network.onnx';
import { validateOnnxBinaryModel } from './network.onnx.validate';

function mutateBinaryModel(
  binaryModel: Uint8Array,
  mutateModel: (
    decodedModel: InstanceType<typeof onnxProto.onnx.ModelProto>,
  ) => void,
): Uint8Array {
  const decodedModel = onnxProto.onnx.ModelProto.decode(binaryModel);
  mutateModel(decodedModel);
  return onnxProto.onnx.ModelProto.encode(decodedModel).finish();
}

function createConvGroundworkScenario(): {
  network: Network;
  mappings: Array<{
    layerIndex: number;
    inHeight: number;
    inWidth: number;
    inChannels: number;
    kernelHeight: number;
    kernelWidth: number;
    strideHeight: number;
    strideWidth: number;
    padTop: number;
    padBottom: number;
    padLeft: number;
    padRight: number;
    outHeight: number;
    outWidth: number;
    outChannels: number;
  }>;
} {
  const inputChannels = 1;
  const inputHeight = 3;
  const inputWidth = 3;
  const kernelHeight = 2;
  const kernelWidth = 2;
  const strideHeight = 1;
  const strideWidth = 1;
  const outputChannels = 2;
  const outputHeight = inputHeight - kernelHeight + 1;
  const outputWidth = inputWidth - kernelWidth + 1;
  const inputSize = inputChannels * inputHeight * inputWidth;
  const hiddenSize = outputChannels * outputHeight * outputWidth;
  const outputSize = 3;
  const network = Network.createMLP(inputSize, [hiddenSize], outputSize);

  network.connections.forEach((connectionEntry, connectionIndex) => {
    connectionEntry.weight = connectionIndex * 0.01;
  });

  network.nodes.forEach((nodeEntry, nodeIndex) => {
    if (nodeEntry.type !== 'input') {
      nodeEntry.bias = nodeIndex * 0.001;
    }
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
        padTop: 0,
        padBottom: 0,
        padLeft: 0,
        padRight: 0,
        outHeight: outputHeight,
        outWidth: outputWidth,
        outChannels: outputChannels,
      },
    ],
  };
}

describe('network onnx validation chapter', () => {
  describe('validateOnnxBinaryModel', () => {
    it('accepts the current same-family baseline binary export in the external validation lane', async () => {
      // Arrange
      const network = Network.createMLP(2, [3], 1);
      const binaryModel = exportToONNXBinary(network);

      // Act
      const validationResult = validateOnnxBinaryModel(binaryModel);

      // Assert
      await expect(validationResult).resolves.toEqual(
        expect.objectContaining({
          isValid: true,
          validator: 'onnxruntime-node',
        }),
      );
    });

    it('accepts storage-fp16 binary exports in the external validation lane', async () => {
      // Arrange
      const network = Network.createMLP(2, [1], 1);
      const binaryModel = exportToONNXBinary(network, {
        includeMetadata: true,
        precision: {
          mode: 'storage-fp16',
        },
      });

      // Act
      const validationResult = validateOnnxBinaryModel(binaryModel);

      // Assert
      await expect(validationResult).resolves.toEqual(
        expect.objectContaining({
          isValid: true,
          validator: 'onnxruntime-node',
        }),
      );
    });

    it('accepts the supported one-output static-8bit dense binary exports in the external validation lane', async () => {
      // Arrange
      const network = Network.createMLP(2, [2], 1);
      const binaryModel = exportToONNXBinary(network, {
        includeMetadata: true,
        quantization: {
          mode: 'static-8bit',
          targets: ['dense'],
          calibration: {
            source: 'external',
            layerTargets: [
              {
                target: 'dense',
                layerIndex: 2,
                inputRange: { min: 0, max: 1 },
                outputRange: { min: 0, max: 0.75 },
              },
            ],
          },
          representation: 'qlinear',
        },
      });

      // Act
      const validationResult = validateOnnxBinaryModel(binaryModel);

      // Assert
      await expect(validationResult).resolves.toEqual(
        expect.objectContaining({
          isValid: true,
          validator: 'onnxruntime-node',
        }),
      );
    });

    it('accepts DynamicQuantizeLinear dense-guidance binary exports in the external validation lane', async () => {
      // Arrange
      const network = Network.createMLP(2, [2], 1);
      const binaryModel = exportToONNXBinary(network, {
        includeMetadata: true,
        quantization: {
          mode: 'dynamic-uint8',
          target: 'dense',
          representation: 'DynamicQuantizeLinear',
        },
      });

      // Act
      const validationResult = validateOnnxBinaryModel(binaryModel);

      // Assert
      await expect(validationResult).resolves.toEqual(
        expect.objectContaining({
          isValid: true,
          validator: 'onnxruntime-node',
        }),
      );
    });

    it('accepts the explicit Conv static-8bit binary subset in the external validation lane', async () => {
      // Arrange
      const scenario = createConvGroundworkScenario();
      const binaryModel = exportToONNXBinary(scenario.network, {
        includeMetadata: true,
        conv2dMappings: scenario.mappings,
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
      });

      // Act
      const validationResult = validateOnnxBinaryModel(binaryModel);

      // Assert
      await expect(validationResult).resolves.toEqual(
        expect.objectContaining({
          isValid: true,
          validator: 'onnxruntime-node',
        }),
      );
    });

    it('returns the current explicit lower-opset policy for accepted baseline binary exports', async () => {
      // Arrange
      const network = Network.createMLP(2, [3], 1);
      const binaryModel = exportToONNXBinary(network);

      // Act
      const validationResult = validateOnnxBinaryModel(binaryModel);

      // Assert
      await expect(validationResult).resolves.toEqual(
        expect.objectContaining({
          isValid: true,
          validator: 'onnxruntime-node',
          compatibilityPolicy: {
            irVersion: 9,
            standardDomain: ONNX_STANDARD_DOMAIN,
            encodedStandardDomain: '',
            declaredOpset: 18,
            referenceOpset: CURRENT_ONNX_REFERENCE_OPSET,
            usesLowerOpsetContract: true,
          },
        }),
      );
    });

    it('reports explicit ai.onnx reference-opset policy even when the local runtime cannot load that higher opset', async () => {
      // Arrange
      const network = Network.createMLP(2, [3], 1);
      const binaryModel = mutateBinaryModel(
        exportToONNXBinary(network),
        (decodedModel) => {
          decodedModel.opsetImport = [
            onnxProto.onnx.OperatorSetIdProto.create({
              domain: ONNX_STANDARD_DOMAIN,
              version: CURRENT_ONNX_REFERENCE_OPSET,
            }),
          ];
        },
      );

      // Act
      const validationResult = validateOnnxBinaryModel(binaryModel);

      // Assert
      await expect(validationResult).resolves.toEqual(
        expect.objectContaining({
          isValid: false,
          validator: 'onnxruntime-node',
          errorCategory: 'runtime-load-failed',
          compatibilityPolicy: {
            irVersion: 9,
            standardDomain: ONNX_STANDARD_DOMAIN,
            encodedStandardDomain: ONNX_STANDARD_DOMAIN,
            declaredOpset: CURRENT_ONNX_REFERENCE_OPSET,
            referenceOpset: CURRENT_ONNX_REFERENCE_OPSET,
            usesLowerOpsetContract: false,
          },
        }),
      );
    });

    it('rejects binary models that declare a non-positive ir_version', async () => {
      // Arrange
      const network = Network.createMLP(2, [3], 1);
      const binaryModel = mutateBinaryModel(
        exportToONNXBinary(network),
        (decodedModel) => {
          decodedModel.irVersion = 0;
        },
      );

      // Act
      const validationResult = validateOnnxBinaryModel(binaryModel);

      // Assert
      await expect(validationResult).resolves.toEqual(
        expect.objectContaining({
          isValid: false,
          validator: 'onnxruntime-node',
          errorCategory: 'invalid-model',
          errorMessage: 'Binary ModelProto must declare a positive ir_version.',
        }),
      );
    });

    it('rejects binary models that declare multiple standard-domain opset imports', async () => {
      // Arrange
      const network = Network.createMLP(2, [3], 1);
      const binaryModel = mutateBinaryModel(
        exportToONNXBinary(network),
        (decodedModel) => {
          decodedModel.opsetImport.push(
            onnxProto.onnx.OperatorSetIdProto.create({
              domain: ONNX_STANDARD_DOMAIN,
              version: 18,
            }),
          );
        },
      );

      // Act
      const validationResult = validateOnnxBinaryModel(binaryModel);

      // Assert
      await expect(validationResult).resolves.toEqual(
        expect.objectContaining({
          isValid: false,
          validator: 'onnxruntime-node',
          errorCategory: 'invalid-model',
          errorMessage:
            'Binary ModelProto must declare exactly one standard-domain opset import for the current Phase 8 subset.',
        }),
      );
    });

    it('rejects binary models that declare a non-positive standard-domain opset version', async () => {
      // Arrange
      const network = Network.createMLP(2, [3], 1);
      const binaryModel = mutateBinaryModel(
        exportToONNXBinary(network),
        (decodedModel) => {
          decodedModel.opsetImport[0]!.version = 0;
        },
      );

      // Act
      const validationResult = validateOnnxBinaryModel(binaryModel);

      // Assert
      await expect(validationResult).resolves.toEqual(
        expect.objectContaining({
          isValid: false,
          validator: 'onnxruntime-node',
          errorCategory: 'invalid-model',
          errorMessage:
            'Binary ModelProto must declare a positive standard-domain opset.',
        }),
      );
    });

    it('surfaces protobuf verification failures before the external runtime runs', async () => {
      // Arrange
      const verifySpy = jest
        .spyOn(onnxProto.onnx.ModelProto, 'verify')
        .mockReturnValue('forced verification failure');
      const network = Network.createMLP(2, [3], 1);
      const binaryModel = exportToONNXBinary(network);

      try {
        // Act
        const validationResult = validateOnnxBinaryModel(binaryModel);

        // Assert
        await expect(validationResult).resolves.toEqual(
          expect.objectContaining({
            isValid: false,
            validator: 'onnxruntime-node',
            errorCategory: 'invalid-binary',
            errorMessage: 'forced verification failure',
          }),
        );
      } finally {
        verifySpy.mockRestore();
      }
    });

    it('surfaces non-Error runtime failures through string coercion', async () => {
      // Arrange
      const createSpy = jest
        .spyOn(InferenceSession, 'create')
        .mockRejectedValue('string runtime failure');
      const network = Network.createMLP(2, [3], 1);
      const binaryModel = exportToONNXBinary(network);

      try {
        // Act
        const validationResult = validateOnnxBinaryModel(binaryModel);

        // Assert
        await expect(validationResult).resolves.toEqual(
          expect.objectContaining({
            isValid: false,
            validator: 'onnxruntime-node',
            errorCategory: 'runtime-load-failed',
            errorMessage: 'string runtime failure',
          }),
        );
      } finally {
        createSpy.mockRestore();
      }
    });

    it('categorizes malformed bytes as invalid binary before the external validator accepts them', async () => {
      // Arrange
      const malformedBytes = Uint8Array.from([0, 1, 2, 3]);

      // Act
      const validationResult = validateOnnxBinaryModel(malformedBytes);

      // Assert
      await expect(validationResult).resolves.toEqual(
        expect.objectContaining({
          isValid: false,
          validator: 'onnxruntime-node',
          errorCategory: 'invalid-binary',
        }),
      );
    });
  });
});
