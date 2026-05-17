import onnxProto from 'onnx-proto';
import Network from '../network';
import {
  exportToONNX,
  exportToONNXBinary,
  importFromONNXBinary,
} from './network.onnx';

type OnnxLongLike = number | string;

type DecodedDimension = {
  dimValue?: OnnxLongLike | null;
  dimParam?: string | null;
};

type DecodedValueInfo = {
  type?: {
    tensorType?: {
      shape?: {
        dim?: DecodedDimension[] | null;
      } | null;
    } | null;
  } | null;
};

type DecodedNodeAttribute = {
  name?: string | null;
  i?: OnnxLongLike | null;
};

type DecodedNode = {
  name?: string | null;
  opType?: string | null;
  input?: string[] | null;
  output?: string[] | null;
  attribute?: DecodedNodeAttribute[] | null;
};

type DecodedTensor = {
  name?: string | null;
  dataType?: number | null;
  floatData?: number[] | null;
  int32Data?: number[] | null;
};

type DecodedBinaryModel = {
  graph?: {
    input?: DecodedValueInfo[] | null;
    output?: DecodedValueInfo[] | null;
    node?: DecodedNode[] | null;
    initializer?: DecodedTensor[] | null;
  } | null;
};

function decodeBinaryModel(binaryModel: Uint8Array): DecodedBinaryModel {
  const decodedModel = onnxProto.onnx.ModelProto.decode(binaryModel);
  return onnxProto.onnx.ModelProto.toObject(decodedModel) as DecodedBinaryModel;
}

function readTensorShapeDimensions(
  valueInfo: DecodedValueInfo | undefined,
): Array<number | string> {
  return (
    valueInfo?.type?.tensorType?.shape?.dim?.map((dimension) =>
      dimension.dimParam ?? Number(dimension.dimValue ?? 0),
    ) ?? []
  );
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

describe('network onnx binary export chapter', () => {
  describe('exportToONNXBinary', () => {
    it('injects required ModelProto headers without widening the JSON export defaults', () => {
      // Arrange
      const network = Network.createMLP(2, [3], 1);
      const jsonModel = exportToONNX(network);

      // Act
      const binaryModel = exportToONNXBinary(network);
      const binaryBuffer = Buffer.from(binaryModel);

      // Assert
      expect({
        jsonIrVersion: jsonModel.ir_version ?? null,
        jsonOpsetImport: jsonModel.opset_import ?? null,
        hasBinaryInputName: binaryBuffer.includes(Buffer.from('input')),
        hasBinaryOutputName: binaryBuffer.includes(Buffer.from('output')),
        hasBinaryProducerName: binaryBuffer.includes(
          Buffer.from('neataptic-ts'),
        ),
      }).toEqual({
        jsonIrVersion: null,
        jsonOpsetImport: null,
        hasBinaryInputName: true,
        hasBinaryOutputName: true,
        hasBinaryProducerName: true,
      });
    });

    it('returns deterministic protobuf bytes for the same network state', () => {
      // Arrange
      const network = Network.createMLP(2, [3], 1);

      // Act
      const firstBinaryModel = exportToONNXBinary(network, {
        docString: 'phase-8 determinism probe',
      });
      const secondBinaryModel = exportToONNXBinary(network, {
        docString: 'phase-8 determinism probe',
      });

      // Assert
      expect(Array.from(firstBinaryModel)).toEqual(Array.from(secondBinaryModel));
    });

    it('preserves explicit producer metadata overrides in the binary payload', () => {
      // Arrange
      const network = Network.createMLP(2, [3], 1);

      // Act
      const binaryModel = exportToONNXBinary(network, {
        producerName: 'phase8-producer',
        producerVersion: '1.2.3',
        docString: 'binary-model-doc',
      });
      const binaryBuffer = Buffer.from(binaryModel);

      // Assert
      expect({
        hasCustomProducerName: binaryBuffer.includes(
          Buffer.from('phase8-producer'),
        ),
        hasCustomProducerVersion: binaryBuffer.includes(Buffer.from('1.2.3')),
        hasCustomDocString: binaryBuffer.includes(Buffer.from('binary-model-doc')),
      }).toEqual({
        hasCustomProducerName: true,
        hasCustomProducerVersion: true,
        hasCustomDocString: true,
      });
    });

    it('does not duplicate the symbolic batch axis when batchDimension is already requested', () => {
      // Arrange
      const network = Network.createMLP(2, [3], 1);

      // Act
      const binaryModel = decodeBinaryModel(
        exportToONNXBinary(network, { batchDimension: true }),
      );

      // Assert
      expect({
        inputDimensions: readTensorShapeDimensions(binaryModel.graph?.input?.[0]),
        outputDimensions: readTensorShapeDimensions(binaryModel.graph?.output?.[0]),
      }).toEqual({
        inputDimensions: ['N', 2],
        outputDimensions: ['N', 1],
      });
    });

    it('materializes explicit Conv binary boundaries as a spatial input plus a flatten bridge', () => {
      // Arrange
      const scenario = createConvGroundworkScenario();

      // Act
      const binaryModel = decodeBinaryModel(
        exportToONNXBinary(scenario.network, {
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
        }),
      );
      const flattenNode = binaryModel.graph?.node?.find(
        (graphNode) => graphNode.name === 'binary_flatten_l1',
      );

      // Assert
      expect({
        inputDimensions: readTensorShapeDimensions(binaryModel.graph?.input?.[0]),
        flattenNodeOperation: flattenNode?.opType ?? null,
        flattenNodeInput: flattenNode?.input?.[0] ?? null,
        flattenNodeOutput: flattenNode?.output?.[0] ?? null,
        flattenAxis: Number(
          flattenNode?.attribute?.find(
            (attributeEntry) => attributeEntry.name === 'axis',
          )?.i ?? 0,
        ),
      }).toEqual({
        inputDimensions: ['N', 1, 3, 3],
        flattenNodeOperation: 'Flatten',
        flattenNodeInput: 'Layer_1',
        flattenNodeOutput: 'BinaryFlatten_l1',
        flattenAxis: 1,
      });
    });

    it('encodes storage-fp16 initializers into canonical int32 TensorProto payloads', () => {
      // Arrange
      const network = Network.createMLP(2, [1], 1);

      // Act
      const binaryModel = decodeBinaryModel(
        exportToONNXBinary(network, {
          precision: {
            mode: 'storage-fp16',
          },
        }),
      );
      const weightInitializer = binaryModel.graph?.initializer?.find(
        (initializerEntry) => initializerEntry.name === 'W0',
      );

      // Assert
      expect({
        dataType: weightInitializer?.dataType ?? null,
        int32DataLength: weightInitializer?.int32Data?.length ?? 0,
        floatDataLength: weightInitializer?.floatData?.length ?? 0,
      }).toEqual({
        dataType: 10,
        int32DataLength: 2,
        floatDataLength: 0,
      });
    });

    it('encodes static-8bit dense weights into integer TensorProto payloads', () => {
      // Arrange
      const network = Network.createMLP(2, [2], 1);

      // Act
      const binaryModel = decodeBinaryModel(
        exportToONNXBinary(network, {
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
        }),
      );
      const quantizedWeightInitializer = binaryModel.graph?.initializer?.find(
        (initializerEntry) => initializerEntry.name === 'QuantDenseWeight_l2',
      );

      // Assert
      expect({
        dataType: quantizedWeightInitializer?.dataType ?? null,
        int32DataLength: quantizedWeightInitializer?.int32Data?.length ?? 0,
        floatDataLength: quantizedWeightInitializer?.floatData?.length ?? 0,
      }).toEqual({
        dataType: 3,
        int32DataLength: 2,
        floatDataLength: 0,
      });
    });
  });

  describe('importFromONNXBinary', () => {
    it('reconstructs the current dense same-family subset from binary ModelProto bytes', () => {
      // Arrange
      const sourceNetwork = Network.createMLP(2, [2], 1);
      sourceNetwork.nodes[2].bias = 0.5;
      sourceNetwork.nodes[3].bias = -0.5;
      sourceNetwork.nodes[4].bias = 0.25;
      sourceNetwork.nodes[2].connections.in[0].weight = 0.1;
      sourceNetwork.nodes[2].connections.in[1].weight = 0.2;
      sourceNetwork.nodes[3].connections.in[0].weight = 0.3;
      sourceNetwork.nodes[3].connections.in[1].weight = 0.4;
      sourceNetwork.nodes[4].connections.in[0].weight = 0.5;
      sourceNetwork.nodes[4].connections.in[1].weight = 0.6;
      const sampleInput = [0.25, 0.75];
      const expectedOutput = sourceNetwork.activate(sampleInput);
      const binaryModel = exportToONNXBinary(sourceNetwork);

      // Act
      const importedNetwork = importFromONNXBinary(binaryModel);
      const importedOutput = importedNetwork.activate(sampleInput);

      // Assert
      expect(importedOutput).toEqual(expectedOutput);
    });
  });
});