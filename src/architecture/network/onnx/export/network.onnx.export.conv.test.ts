import Network from '../../network';
import { exportToONNX } from '../network.onnx';
import type { Conv2DMapping, OnnxModel, Pool2DMapping } from '../network.onnx';

type OnnxGraphNodeView = { op_type: string };

jest.retryTimes(2, { logErrorsBeforeRetry: true });

function hasMetadataKey(onnxModel: OnnxModel, key: string): boolean {
  return (onnxModel.metadata_props ?? []).some(
    (metadataEntry) => metadataEntry.key === key,
  );
}

function hasGraphNodeType(onnxModel: OnnxModel, operatorType: string): boolean {
  return onnxModel.graph.node.some(
    (graphNode) => (graphNode as OnnxGraphNodeView).op_type === operatorType,
  );
}

function createConvGroundworkScenario(): {
  network: Network;
  mappings: Conv2DMapping[];
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

function createInvalidConvMappingScenario(): {
  network: Network;
  mappings: Conv2DMapping[];
} {
  return {
    network: Network.createMLP(10, [6], 2),
    mappings: [
      {
        layerIndex: 1,
        inHeight: 2,
        inWidth: 2,
        inChannels: 2,
        kernelHeight: 2,
        kernelWidth: 2,
        strideHeight: 1,
        strideWidth: 1,
        outHeight: 1,
        outWidth: 1,
        outChannels: 3,
      },
    ],
  };
}

function createConvSharingVerifiedScenario(): {
  network: Network;
  mappings: Conv2DMapping[];
} {
  const inputChannels = 1;
  const inputHeight = 3;
  const inputWidth = 3;
  const kernelHeight = 2;
  const kernelWidth = 2;
  const strideHeight = 1;
  const strideWidth = 1;
  const outputChannels = 1;
  const outputHeight = inputHeight - kernelHeight + 1;
  const outputWidth = inputWidth - kernelWidth + 1;
  const inputSize = inputChannels * inputHeight * inputWidth;
  const hiddenSize = outputChannels * outputHeight * outputWidth;
  const network = Network.createMLP(inputSize, [hiddenSize], 2);
  const inputNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'input',
  );
  const hiddenNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'hidden',
  );
  const kernelPattern = [0.11, -0.07, 0.05, 0.02];

  hiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
    const outputRow = Math.floor(hiddenNodeIndex / outputWidth);
    const outputColumn = hiddenNodeIndex % outputWidth;
    const inputBaseRow = outputRow * strideHeight;
    const inputBaseColumn = outputColumn * strideWidth;

    hiddenNode.connections.in.forEach((connectionEntry) => {
      connectionEntry.weight = 0;
    });

    for (let kernelRow = 0; kernelRow < kernelHeight; kernelRow += 1) {
      for (
        let kernelColumn = 0;
        kernelColumn < kernelWidth;
        kernelColumn += 1
      ) {
        const inputRow = inputBaseRow + kernelRow;
        const inputColumn = inputBaseColumn + kernelColumn;
        const sourceIndex = inputRow * inputWidth + inputColumn;
        const kernelIndex = kernelRow * kernelWidth + kernelColumn;
        const sourceNode = inputNodes[sourceIndex];
        const matchingConnection = hiddenNode.connections.in.find(
          (connectionEntry) => connectionEntry.from === sourceNode,
        );

        if (matchingConnection != null) {
          matchingConnection.weight = kernelPattern[kernelIndex];
        }
      }
    }

    hiddenNode.bias = 0.123;
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
        outHeight: outputHeight,
        outWidth: outputWidth,
        outChannels: outputChannels,
      },
    ],
  };
}

function createConvSharingMismatchScenario(): {
  network: Network;
  mappings: Conv2DMapping[];
} {
  const inputChannels = 1;
  const inputHeight = 3;
  const inputWidth = 3;
  const kernelHeight = 2;
  const kernelWidth = 2;
  const strideHeight = 1;
  const strideWidth = 1;
  const outputChannels = 1;
  const outputHeight = inputHeight - kernelHeight + 1;
  const outputWidth = inputWidth - kernelWidth + 1;
  const inputSize = inputChannels * inputHeight * inputWidth;
  const hiddenSize = outputChannels * outputHeight * outputWidth;
  const network = Network.createMLP(inputSize, [hiddenSize], 1);
  const hiddenNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'hidden',
  );

  hiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
    hiddenNode.connections.in.forEach((connectionEntry, connectionIndex) => {
      connectionEntry.weight =
        (connectionIndex + 1) * 0.02 + hiddenNodeIndex * 0.001;
    });
    hiddenNode.bias = hiddenNodeIndex * 0.01;
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
        outHeight: outputHeight,
        outWidth: outputWidth,
        outChannels: outputChannels,
      },
    ],
  };
}

function createPoolingMappings(): Pool2DMapping[] {
  return [
    {
      afterLayerIndex: 1,
      type: 'MaxPool',
      kernelHeight: 2,
      kernelWidth: 2,
      strideHeight: 2,
      strideWidth: 2,
    },
  ];
}

describe('network onnx export conv chapter', () => {
  describe('explicit conv mappings', () => {
    describe('given a valid manual conv mapping', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createConvGroundworkScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          includeMetadata: true,
          conv2dMappings: scenario.mappings,
        });
      });

      describe('when reading the emitted graph nodes', () => {
        it('includes a Conv operator', () => {
          // Assert
          expect(hasGraphNodeType(onnxModel, 'Conv')).toBe(true);
        });
      });

      describe('when reading the emitted metadata', () => {
        it('records conv2d_layers metadata', () => {
          // Assert
          expect(hasMetadataKey(onnxModel, 'conv2d_layers')).toBe(true);
        });
      });
    });

    describe('given an invalid manual conv mapping', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createInvalidConvMappingScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          conv2dMappings: scenario.mappings,
        });
      });

      describe('when dimensions do not match the layer widths', () => {
        it('does not emit a Conv operator', () => {
          // Assert
          expect(hasGraphNodeType(onnxModel, 'Conv')).toBe(false);
        });
      });
    });
  });

  describe('heuristic conv inference', () => {
    describe('given a 25-9-2 network with metadata enabled', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const network = Network.createMLP(25, [9], 2);

        // Act
        onnxModel = exportToONNX(network, { includeMetadata: true });
      });

      describe('when the exporter infers a conv-like hidden layer', () => {
        it('records conv2d_inferred_layers metadata', () => {
          // Assert
          expect(hasMetadataKey(onnxModel, 'conv2d_inferred_layers')).toBe(
            true,
          );
        });
      });
    });
  });

  describe('conv sharing validation', () => {
    describe('given a conv-like layer with shared kernel weights', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createConvSharingVerifiedScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          includeMetadata: true,
          conv2dMappings: scenario.mappings,
          validateConvSharing: true,
        });
      });

      describe('when sharing validation runs', () => {
        it('records conv2d_sharing_verified metadata', () => {
          // Assert
          expect(hasMetadataKey(onnxModel, 'conv2d_sharing_verified')).toBe(
            true,
          );
        });
      });
    });

    describe('given a conv-like layer with mismatched spatial weights', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createConvSharingMismatchScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          includeMetadata: true,
          conv2dMappings: scenario.mappings,
          validateConvSharing: true,
        });
      });

      describe('when sharing validation detects divergence', () => {
        it('records conv2d_sharing_mismatch metadata', () => {
          // Assert
          expect(hasMetadataKey(onnxModel, 'conv2d_sharing_mismatch')).toBe(
            true,
          );
        });
      });
    });
  });

  describe('pooling metadata emission', () => {
    describe('given pooling mappings are provided', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const network = Network.createMLP(6, [4], 2);

        network.connections.forEach((connectionEntry, connectionIndex) => {
          connectionEntry.weight = (connectionIndex + 1) * 0.01;
        });

        network.nodes.forEach((nodeEntry, nodeIndex) => {
          if (nodeEntry.type !== 'input') {
            nodeEntry.bias = nodeIndex * 0.001;
          }
        });

        // Act
        onnxModel = exportToONNX(network, {
          includeMetadata: true,
          pool2dMappings: createPoolingMappings(),
        });
      });

      describe('when reading the emitted metadata', () => {
        it('records pool2d_layers metadata', () => {
          // Assert
          expect(hasMetadataKey(onnxModel, 'pool2d_layers')).toBe(true);
        });
      });
    });

    describe('given flattenAfterPooling is enabled', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const network = Network.createMLP(9, [4], 2);

        // Act
        onnxModel = exportToONNX(network, {
          includeMetadata: true,
          pool2dMappings: [
            {
              afterLayerIndex: 1,
              type: 'MaxPool',
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
            },
          ],
          flattenAfterPooling: true,
        });
      });

      describe('when flatten metadata is requested', () => {
        it('records flatten_layers metadata', () => {
          // Assert
          expect(hasMetadataKey(onnxModel, 'flatten_layers')).toBe(true);
        });
      });
    });

    describe('given explicit conv and pooling mappings share the same layer', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createConvGroundworkScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          conv2dMappings: scenario.mappings,
          pool2dMappings: createPoolingMappings(),
        });
      });

      describe('when conv layer emission resolves pooling spec by layer index', () => {
        it('emits a MaxPool operator after conv activation', () => {
          // Assert
          expect(hasGraphNodeType(onnxModel, 'MaxPool')).toBe(true);
        });
      });
    });
  });
});
