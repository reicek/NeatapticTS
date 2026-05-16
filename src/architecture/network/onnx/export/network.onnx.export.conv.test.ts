import Network from '../../network';
import { exportToONNX } from '../network.onnx';
import type { Conv2DMapping, OnnxModel, Pool2DMapping } from '../network.onnx';

type OnnxGraphNodeView = { op_type: string };

jest.retryTimes(2, { logErrorsBeforeRetry: true });

function getMetadataValue(onnxModel: OnnxModel, key: string): string | undefined {
  return (onnxModel.metadata_props ?? []).find(
    (metadataEntry) => metadataEntry.key === key,
  )?.value;
}

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

function countGraphNodeType(onnxModel: OnnxModel, operatorType: string): number {
  return onnxModel.graph.node.filter(
    (graphNode) => (graphNode as OnnxGraphNodeView).op_type === operatorType,
  ).length;
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

function createHeuristicConvPromotionScenario(): {
  network: Network;
} {
  const inputHeight = 5;
  const inputWidth = 5;
  const kernelHeight = 3;
  const kernelWidth = 3;
  const strideHeight = 1;
  const strideWidth = 1;
  const outputHeight = inputHeight - kernelHeight + 1;
  const outputWidth = inputWidth - kernelWidth + 1;
  const network = Network.createMLP(25, [9], 2);
  const inputNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'input',
  );
  const hiddenNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'hidden',
  );
  const outputNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'output',
  );
  const kernelPattern = [0.09, -0.04, 0.02, 0.11, 0.07, -0.03, 0.05, 0.01, 0.08];

  hiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
    const outputRow = Math.floor(hiddenNodeIndex / outputWidth);
    const outputColumn = hiddenNodeIndex % outputWidth;
    const inputBaseRow = outputRow * strideHeight;
    const inputBaseColumn = outputColumn * strideWidth;

    hiddenNode.connections.in.forEach((connectionEntry) => {
      connectionEntry.weight = 0;
    });

    for (
      let kernelRowIndex = 0;
      kernelRowIndex < kernelHeight;
      kernelRowIndex += 1
    ) {
      for (
        let kernelColumnIndex = 0;
        kernelColumnIndex < kernelWidth;
        kernelColumnIndex += 1
      ) {
        const inputRow = inputBaseRow + kernelRowIndex;
        const inputColumn = inputBaseColumn + kernelColumnIndex;
        const sourceIndex = inputRow * inputWidth + inputColumn;
        const kernelIndex = kernelRowIndex * kernelWidth + kernelColumnIndex;
        const sourceNode = inputNodes[sourceIndex];
        const matchingConnection = hiddenNode.connections.in.find(
          (connectionEntry) => connectionEntry.from === sourceNode,
        );

        if (matchingConnection != null) {
          matchingConnection.weight = kernelPattern[kernelIndex];
        }
      }
    }

    hiddenNode.bias = 0.045;
  });

  outputNodes.forEach((outputNode, outputNodeIndex) => {
    outputNode.connections.in.forEach((connectionEntry, connectionIndex) => {
      connectionEntry.weight =
        (outputNodeIndex + 1) * 0.02 + connectionIndex * 0.005;
    });
    outputNode.bias = outputNodeIndex * 0.01;
  });

  return { network };
}

function createUnsafeHeuristicConvPromotionScenario(): {
  network: Network;
} {
  const network = Network.createMLP(25, [9], 2);
  const hiddenNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'hidden',
  );
  const outputNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'output',
  );

  hiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
    hiddenNode.connections.in.forEach((connectionEntry, connectionIndex) => {
      connectionEntry.weight =
        hiddenNodeIndex * 0.1 + connectionIndex * 0.001 + 0.01;
    });
    hiddenNode.bias = hiddenNodeIndex * 0.01;
  });

  outputNodes.forEach((outputNode, outputNodeIndex) => {
    outputNode.connections.in.forEach((connectionEntry, connectionIndex) => {
      connectionEntry.weight =
        0.03 + outputNodeIndex * 0.02 + connectionIndex * 0.004;
    });
    outputNode.bias = 0.02 + outputNodeIndex * 0.01;
  });

  return { network };
}

function createMultiChannelHeuristicConvPromotionScenario(): {
  network: Network;
} {
  const inputChannels = 2;
  const inputHeight = 3;
  const inputWidth = 3;
  const kernelHeight = 2;
  const kernelWidth = 2;
  const strideHeight = 1;
  const strideWidth = 1;
  const outputChannels = 2;
  const outputHeight = inputHeight - kernelHeight + 1;
  const outputWidth = inputWidth - kernelWidth + 1;
  const network = Network.createMLP(18, [8], 2);
  const inputNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'input',
  );
  const hiddenNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'hidden',
  );
  const outputNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'output',
  );
  const channelKernelPatterns = [
    [
      [0.1, -0.05, 0.03, 0.08],
      [-0.04, 0.07, 0.02, -0.01],
    ],
    [
      [0.06, 0.01, -0.02, 0.09],
      [0.05, -0.03, 0.04, 0.11],
    ],
  ];

  hiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
    const outputChannelIndex = Math.floor(
      hiddenNodeIndex / (outputHeight * outputWidth),
    );
    const outputSpatialIndex = hiddenNodeIndex % (outputHeight * outputWidth);
    const outputRow = Math.floor(outputSpatialIndex / outputWidth);
    const outputColumn = outputSpatialIndex % outputWidth;

    hiddenNode.connections.in.forEach((connectionEntry) => {
      connectionEntry.weight = 0;
    });

    for (
      let inputChannelIndex = 0;
      inputChannelIndex < inputChannels;
      inputChannelIndex += 1
    ) {
      for (
        let kernelRowIndex = 0;
        kernelRowIndex < kernelHeight;
        kernelRowIndex += 1
      ) {
        for (
          let kernelColumnIndex = 0;
          kernelColumnIndex < kernelWidth;
          kernelColumnIndex += 1
        ) {
          const inputRow = outputRow * strideHeight + kernelRowIndex;
          const inputColumn = outputColumn * strideWidth + kernelColumnIndex;
          const sourceIndex =
            inputChannelIndex * (inputHeight * inputWidth) +
            inputRow * inputWidth +
            inputColumn;
          const kernelIndex = kernelRowIndex * kernelWidth + kernelColumnIndex;
          const sourceNode = inputNodes[sourceIndex];
          const matchingConnection = hiddenNode.connections.in.find(
            (connectionEntry) => connectionEntry.from === sourceNode,
          );

          if (matchingConnection != null) {
            matchingConnection.weight =
              channelKernelPatterns[outputChannelIndex]?.[inputChannelIndex]?.[
                kernelIndex
              ] ?? 0;
          }
        }
      }
    }

    hiddenNode.bias = 0.02 + outputChannelIndex * 0.01;
  });

  outputNodes.forEach((outputNode, outputNodeIndex) => {
    outputNode.connections.in.forEach((connectionEntry, connectionIndex) => {
      connectionEntry.weight =
        0.015 + outputNodeIndex * 0.02 + connectionIndex * 0.003;
    });
    outputNode.bias = 0.01 + outputNodeIndex * 0.02;
  });

  return { network };
}

function createUnsafeMultiChannelHeuristicConvPromotionScenario(): {
  network: Network;
} {
  const network = Network.createMLP(18, [8], 2);
  const hiddenNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'hidden',
  );
  const outputNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'output',
  );

  hiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
    hiddenNode.connections.in.forEach((connectionEntry, connectionIndex) => {
      connectionEntry.weight =
        0.01 + hiddenNodeIndex * 0.1 + connectionIndex * 0.002;
    });
    hiddenNode.bias = 0.03 + hiddenNodeIndex * 0.01;
  });

  outputNodes.forEach((outputNode, outputNodeIndex) => {
    outputNode.connections.in.forEach((connectionEntry, connectionIndex) => {
      connectionEntry.weight =
        0.02 + outputNodeIndex * 0.03 + connectionIndex * 0.005;
    });
    outputNode.bias = 0.015 + outputNodeIndex * 0.01;
  });

  return { network };
}

function createStackedHeuristicConvPromotionScenario(): {
  network: Network;
} {
  const inputHeight = 5;
  const inputWidth = 5;
  const firstKernelHeight = 3;
  const firstKernelWidth = 3;
  const secondKernelHeight = 2;
  const secondKernelWidth = 2;
  const strideHeight = 1;
  const strideWidth = 1;
  const firstOutputHeight = inputHeight - firstKernelHeight + 1;
  const firstOutputWidth = inputWidth - firstKernelWidth + 1;
  const secondOutputHeight =
    firstOutputHeight - secondKernelHeight + 1;
  const secondOutputWidth = firstOutputWidth - secondKernelWidth + 1;
  const network = Network.createMLP(25, [9, 4], 2);
  const inputNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'input',
  );
  const hiddenNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'hidden',
  );
  const firstHiddenNodes = hiddenNodes.slice(0, 9);
  const secondHiddenNodes = hiddenNodes.slice(9);
  const outputNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'output',
  );
  const firstKernelPattern = [
    0.09,
    -0.04,
    0.02,
    0.11,
    0.07,
    -0.03,
    0.05,
    0.01,
    0.08,
  ];
  const secondKernelPattern = [0.06, -0.02, 0.04, 0.1];

  firstHiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
    const outputRow = Math.floor(hiddenNodeIndex / firstOutputWidth);
    const outputColumn = hiddenNodeIndex % firstOutputWidth;
    const inputBaseRow = outputRow * strideHeight;
    const inputBaseColumn = outputColumn * strideWidth;

    hiddenNode.connections.in.forEach((connectionEntry) => {
      connectionEntry.weight = 0;
    });

    for (
      let kernelRowIndex = 0;
      kernelRowIndex < firstKernelHeight;
      kernelRowIndex += 1
    ) {
      for (
        let kernelColumnIndex = 0;
        kernelColumnIndex < firstKernelWidth;
        kernelColumnIndex += 1
      ) {
        const inputRow = inputBaseRow + kernelRowIndex;
        const inputColumn = inputBaseColumn + kernelColumnIndex;
        const sourceIndex = inputRow * inputWidth + inputColumn;
        const kernelIndex =
          kernelRowIndex * firstKernelWidth + kernelColumnIndex;
        const sourceNode = inputNodes[sourceIndex];
        const matchingConnection = hiddenNode.connections.in.find(
          (connectionEntry) => connectionEntry.from === sourceNode,
        );

        if (matchingConnection != null) {
          matchingConnection.weight = firstKernelPattern[kernelIndex];
        }
      }
    }

    hiddenNode.bias = 0.045;
  });

  secondHiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
    const outputRow = Math.floor(hiddenNodeIndex / secondOutputWidth);
    const outputColumn = hiddenNodeIndex % secondOutputWidth;
    const inputBaseRow = outputRow * strideHeight;
    const inputBaseColumn = outputColumn * strideWidth;

    hiddenNode.connections.in.forEach((connectionEntry) => {
      connectionEntry.weight = 0;
    });

    for (
      let kernelRowIndex = 0;
      kernelRowIndex < secondKernelHeight;
      kernelRowIndex += 1
    ) {
      for (
        let kernelColumnIndex = 0;
        kernelColumnIndex < secondKernelWidth;
        kernelColumnIndex += 1
      ) {
        const inputRow = inputBaseRow + kernelRowIndex;
        const inputColumn = inputBaseColumn + kernelColumnIndex;
        const sourceIndex = inputRow * firstOutputWidth + inputColumn;
        const kernelIndex =
          kernelRowIndex * secondKernelWidth + kernelColumnIndex;
        const sourceNode = firstHiddenNodes[sourceIndex];
        const matchingConnection = hiddenNode.connections.in.find(
          (connectionEntry) => connectionEntry.from === sourceNode,
        );

        if (matchingConnection != null) {
          matchingConnection.weight = secondKernelPattern[kernelIndex];
        }
      }
    }

    hiddenNode.bias = 0.03;
  });

  outputNodes.forEach((outputNode, outputNodeIndex) => {
    outputNode.connections.in.forEach((connectionEntry, connectionIndex) => {
      connectionEntry.weight =
        0.02 + outputNodeIndex * 0.015 + connectionIndex * 0.004;
    });
    outputNode.bias = 0.01 + outputNodeIndex * 0.01;
  });

  return { network };
}

function createPostPoolStackedHeuristicConvPromotionScenario(): {
  network: Network;
} {
  const inputHeight = 5;
  const inputWidth = 5;
  const firstKernelHeight = 2;
  const firstKernelWidth = 2;
  const poolingKernelHeight = 2;
  const poolingKernelWidth = 2;
  const poolingStrideHeight = 1;
  const poolingStrideWidth = 1;
  const secondKernelHeight = 2;
  const secondKernelWidth = 2;
  const strideHeight = 1;
  const strideWidth = 1;
  const firstOutputHeight = inputHeight - firstKernelHeight + 1;
  const firstOutputWidth = inputWidth - firstKernelWidth + 1;
  const pooledHeight =
    Math.floor(
      (firstOutputHeight - poolingKernelHeight) / poolingStrideHeight,
    ) + 1;
  const pooledWidth =
    Math.floor(
      (firstOutputWidth - poolingKernelWidth) / poolingStrideWidth,
    ) + 1;
  const secondOutputHeight = pooledHeight - secondKernelHeight + 1;
  const secondOutputWidth = pooledWidth - secondKernelWidth + 1;
  const network = Network.createMLP(25, [16, 4], 2);
  const inputNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'input',
  );
  const hiddenNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'hidden',
  );
  const firstHiddenNodes = hiddenNodes.slice(0, 16);
  const secondHiddenNodes = hiddenNodes.slice(16);
  const outputNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'output',
  );
  const firstKernelPattern = [0.04, -0.06, 0.08, 0.11];
  const secondKernelPattern = [0.07, -0.02, 0.05, 0.09];

  firstHiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
    const outputRow = Math.floor(hiddenNodeIndex / firstOutputWidth);
    const outputColumn = hiddenNodeIndex % firstOutputWidth;
    const inputBaseRow = outputRow * strideHeight;
    const inputBaseColumn = outputColumn * strideWidth;

    hiddenNode.connections.in.forEach((connectionEntry) => {
      connectionEntry.weight = 0;
    });

    for (
      let kernelRowIndex = 0;
      kernelRowIndex < firstKernelHeight;
      kernelRowIndex += 1
    ) {
      for (
        let kernelColumnIndex = 0;
        kernelColumnIndex < firstKernelWidth;
        kernelColumnIndex += 1
      ) {
        const inputRow = inputBaseRow + kernelRowIndex;
        const inputColumn = inputBaseColumn + kernelColumnIndex;
        const sourceIndex = inputRow * inputWidth + inputColumn;
        const kernelIndex =
          kernelRowIndex * firstKernelWidth + kernelColumnIndex;
        const sourceNode = inputNodes[sourceIndex];
        const matchingConnection = hiddenNode.connections.in.find(
          (connectionEntry) => connectionEntry.from === sourceNode,
        );

        if (matchingConnection != null) {
          matchingConnection.weight = firstKernelPattern[kernelIndex];
        }
      }
    }

    hiddenNode.bias = 0.025;
  });

  secondHiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
    const outputRow = Math.floor(hiddenNodeIndex / secondOutputWidth);
    const outputColumn = hiddenNodeIndex % secondOutputWidth;
    const pooledInputBaseRow = outputRow * strideHeight;
    const pooledInputBaseColumn = outputColumn * strideWidth;

    hiddenNode.connections.in.forEach((connectionEntry) => {
      connectionEntry.weight = 0;
    });

    for (
      let kernelRowIndex = 0;
      kernelRowIndex < secondKernelHeight;
      kernelRowIndex += 1
    ) {
      for (
        let kernelColumnIndex = 0;
        kernelColumnIndex < secondKernelWidth;
        kernelColumnIndex += 1
      ) {
        const pooledRow = pooledInputBaseRow + kernelRowIndex;
        const pooledColumn = pooledInputBaseColumn + kernelColumnIndex;
        const pooledIndex = pooledRow * pooledWidth + pooledColumn;
        const sourceNode = firstHiddenNodes[pooledIndex];
        const matchingConnection = hiddenNode.connections.in.find(
          (connectionEntry) => connectionEntry.from === sourceNode,
        );
        const kernelIndex =
          kernelRowIndex * secondKernelWidth + kernelColumnIndex;

        if (matchingConnection != null) {
          matchingConnection.weight = secondKernelPattern[kernelIndex];
        }
      }
    }

    hiddenNode.bias = 0.035;
  });

  outputNodes.forEach((outputNode, outputNodeIndex) => {
    outputNode.connections.in.forEach((connectionEntry, connectionIndex) => {
      connectionEntry.weight =
        0.018 + outputNodeIndex * 0.014 + connectionIndex * 0.003;
    });
    outputNode.bias = 0.012 + outputNodeIndex * 0.008;
  });

  return { network };
}

function createPostPoolMultiChannelHeuristicConvPromotionScenario(): {
  network: Network;
} {
  const inputChannels = 2;
  const inputHeight = 4;
  const inputWidth = 4;
  const firstKernelHeight = 2;
  const firstKernelWidth = 2;
  const poolingKernelHeight = 2;
  const poolingKernelWidth = 2;
  const poolingStrideHeight = 1;
  const poolingStrideWidth = 1;
  const secondKernelHeight = 2;
  const secondKernelWidth = 2;
  const strideHeight = 1;
  const strideWidth = 1;
  const firstOutputHeight = inputHeight - firstKernelHeight + 1;
  const firstOutputWidth = inputWidth - firstKernelWidth + 1;
  const pooledHeight =
    Math.floor(
      (firstOutputHeight - poolingKernelHeight) / poolingStrideHeight,
    ) + 1;
  const pooledWidth =
    Math.floor(
      (firstOutputWidth - poolingKernelWidth) / poolingStrideWidth,
    ) + 1;
  const secondOutputHeight = pooledHeight - secondKernelHeight + 1;
  const secondOutputWidth = pooledWidth - secondKernelWidth + 1;
  const outputChannels = 2;
  const network = Network.createMLP(32, [18, 2], 2);
  const inputNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'input',
  );
  const hiddenNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'hidden',
  );
  const firstHiddenNodes = hiddenNodes.slice(0, 18);
  const secondHiddenNodes = hiddenNodes.slice(18);
  const outputNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'output',
  );
  const firstChannelKernelPatterns = [
    [
      [0.1, -0.05, 0.03, 0.08],
      [-0.04, 0.07, 0.02, -0.01],
    ],
    [
      [0.06, 0.01, -0.02, 0.09],
      [0.05, -0.03, 0.04, 0.11],
    ],
  ];
  const secondChannelKernelPatterns = [
    [
      [0.12, -0.06, 0.02, 0.07],
      [-0.03, 0.05, 0.09, -0.04],
    ],
    [
      [0.08, 0.01, -0.05, 0.06],
      [0.04, -0.02, 0.03, 0.1],
    ],
  ];

  firstHiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
    const outputChannelIndex = Math.floor(
      hiddenNodeIndex / (firstOutputHeight * firstOutputWidth),
    );
    const outputSpatialIndex =
      hiddenNodeIndex % (firstOutputHeight * firstOutputWidth);
    const outputRow = Math.floor(outputSpatialIndex / firstOutputWidth);
    const outputColumn = outputSpatialIndex % firstOutputWidth;

    hiddenNode.connections.in.forEach((connectionEntry) => {
      connectionEntry.weight = 0;
    });

    for (
      let inputChannelIndex = 0;
      inputChannelIndex < inputChannels;
      inputChannelIndex += 1
    ) {
      for (
        let kernelRowIndex = 0;
        kernelRowIndex < firstKernelHeight;
        kernelRowIndex += 1
      ) {
        for (
          let kernelColumnIndex = 0;
          kernelColumnIndex < firstKernelWidth;
          kernelColumnIndex += 1
        ) {
          const inputRow = outputRow * strideHeight + kernelRowIndex;
          const inputColumn = outputColumn * strideWidth + kernelColumnIndex;
          const sourceIndex =
            inputChannelIndex * (inputHeight * inputWidth) +
            inputRow * inputWidth +
            inputColumn;
          const kernelIndex =
            kernelRowIndex * firstKernelWidth + kernelColumnIndex;
          const sourceNode = inputNodes[sourceIndex];
          const matchingConnection = hiddenNode.connections.in.find(
            (connectionEntry) => connectionEntry.from === sourceNode,
          );

          if (matchingConnection != null) {
            matchingConnection.weight =
              firstChannelKernelPatterns[outputChannelIndex]?.[
                inputChannelIndex
              ]?.[kernelIndex] ?? 0;
          }
        }
      }
    }

    hiddenNode.bias = 0.02 + outputChannelIndex * 0.01;
  });

  secondHiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
    const outputChannelIndex = Math.floor(
      hiddenNodeIndex / (secondOutputHeight * secondOutputWidth),
    );
    const outputSpatialIndex =
      hiddenNodeIndex % (secondOutputHeight * secondOutputWidth);
    const outputRow = Math.floor(outputSpatialIndex / secondOutputWidth);
    const outputColumn = outputSpatialIndex % secondOutputWidth;

    hiddenNode.connections.in.forEach((connectionEntry) => {
      connectionEntry.weight = 0;
    });

    for (
      let inputChannelIndex = 0;
      inputChannelIndex < inputChannels;
      inputChannelIndex += 1
    ) {
      for (
        let kernelRowIndex = 0;
        kernelRowIndex < secondKernelHeight;
        kernelRowIndex += 1
      ) {
        for (
          let kernelColumnIndex = 0;
          kernelColumnIndex < secondKernelWidth;
          kernelColumnIndex += 1
        ) {
          const pooledRow = outputRow * strideHeight + kernelRowIndex;
          const pooledColumn = outputColumn * strideWidth + kernelColumnIndex;
          const sourceIndex =
            inputChannelIndex * (firstOutputHeight * firstOutputWidth) +
            pooledRow * pooledWidth +
            pooledColumn;
          const kernelIndex =
            kernelRowIndex * secondKernelWidth + kernelColumnIndex;
          const sourceNode = firstHiddenNodes[sourceIndex];
          const matchingConnection = hiddenNode.connections.in.find(
            (connectionEntry) => connectionEntry.from === sourceNode,
          );

          if (matchingConnection != null) {
            matchingConnection.weight =
              secondChannelKernelPatterns[outputChannelIndex]?.[
                inputChannelIndex
              ]?.[kernelIndex] ?? 0;
          }
        }
      }
    }

    hiddenNode.bias = 0.03 + outputChannelIndex * 0.01;
  });

  outputNodes.forEach((outputNode, outputNodeIndex) => {
    outputNode.connections.in.forEach((connectionEntry, connectionIndex) => {
      connectionEntry.weight =
        0.02 + outputNodeIndex * 0.015 + connectionIndex * 0.004;
    });
    outputNode.bias = 0.015 + outputNodeIndex * 0.01;
  });

  return { network };
}

function createDeepPostPoolStackedHeuristicConvPromotionScenario(): {
  network: Network;
} {
  const inputHeight = 6;
  const inputWidth = 6;
  const firstKernelHeight = 2;
  const firstKernelWidth = 2;
  const poolingKernelHeight = 2;
  const poolingKernelWidth = 2;
  const poolingStrideHeight = 1;
  const poolingStrideWidth = 1;
  const secondKernelHeight = 2;
  const secondKernelWidth = 2;
  const thirdKernelHeight = 2;
  const thirdKernelWidth = 2;
  const strideHeight = 1;
  const strideWidth = 1;
  const firstOutputHeight = inputHeight - firstKernelHeight + 1;
  const firstOutputWidth = inputWidth - firstKernelWidth + 1;
  const firstPooledHeight =
    Math.floor(
      (firstOutputHeight - poolingKernelHeight) / poolingStrideHeight,
    ) + 1;
  const firstPooledWidth =
    Math.floor(
      (firstOutputWidth - poolingKernelWidth) / poolingStrideWidth,
    ) + 1;
  const secondOutputHeight = firstPooledHeight - secondKernelHeight + 1;
  const secondOutputWidth = firstPooledWidth - secondKernelWidth + 1;
  const secondPooledHeight =
    Math.floor(
      (secondOutputHeight - poolingKernelHeight) / poolingStrideHeight,
    ) + 1;
  const secondPooledWidth =
    Math.floor(
      (secondOutputWidth - poolingKernelWidth) / poolingStrideWidth,
    ) + 1;
  const thirdOutputHeight = secondPooledHeight - thirdKernelHeight + 1;
  const thirdOutputWidth = secondPooledWidth - thirdKernelWidth + 1;
  const network = Network.createMLP(36, [25, 9, 1], 2);
  const inputNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'input',
  );
  const hiddenNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'hidden',
  );
  const firstHiddenNodes = hiddenNodes.slice(0, 25);
  const secondHiddenNodes = hiddenNodes.slice(25, 34);
  const thirdHiddenNodes = hiddenNodes.slice(34);
  const outputNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'output',
  );
  const firstKernelPattern = [0.04, -0.06, 0.08, 0.11];
  const secondKernelPattern = [0.07, -0.02, 0.05, 0.09];
  const thirdKernelPattern = [0.13, -0.05, 0.04, 0.1];

  firstHiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
    const outputRow = Math.floor(hiddenNodeIndex / firstOutputWidth);
    const outputColumn = hiddenNodeIndex % firstOutputWidth;
    const inputBaseRow = outputRow * strideHeight;
    const inputBaseColumn = outputColumn * strideWidth;

    hiddenNode.connections.in.forEach((connectionEntry) => {
      connectionEntry.weight = 0;
    });

    for (
      let kernelRowIndex = 0;
      kernelRowIndex < firstKernelHeight;
      kernelRowIndex += 1
    ) {
      for (
        let kernelColumnIndex = 0;
        kernelColumnIndex < firstKernelWidth;
        kernelColumnIndex += 1
      ) {
        const inputRow = inputBaseRow + kernelRowIndex;
        const inputColumn = inputBaseColumn + kernelColumnIndex;
        const sourceIndex = inputRow * inputWidth + inputColumn;
        const kernelIndex =
          kernelRowIndex * firstKernelWidth + kernelColumnIndex;
        const sourceNode = inputNodes[sourceIndex];
        const matchingConnection = hiddenNode.connections.in.find(
          (connectionEntry) => connectionEntry.from === sourceNode,
        );

        if (matchingConnection != null) {
          matchingConnection.weight = firstKernelPattern[kernelIndex];
        }
      }
    }

    hiddenNode.bias = 0.021;
  });

  secondHiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
    const outputRow = Math.floor(hiddenNodeIndex / secondOutputWidth);
    const outputColumn = hiddenNodeIndex % secondOutputWidth;
    const pooledInputBaseRow = outputRow * strideHeight;
    const pooledInputBaseColumn = outputColumn * strideWidth;

    hiddenNode.connections.in.forEach((connectionEntry) => {
      connectionEntry.weight = 0;
    });

    for (
      let kernelRowIndex = 0;
      kernelRowIndex < secondKernelHeight;
      kernelRowIndex += 1
    ) {
      for (
        let kernelColumnIndex = 0;
        kernelColumnIndex < secondKernelWidth;
        kernelColumnIndex += 1
      ) {
        const pooledRow = pooledInputBaseRow + kernelRowIndex;
        const pooledColumn = pooledInputBaseColumn + kernelColumnIndex;
        const pooledIndex = pooledRow * firstPooledWidth + pooledColumn;
        const sourceNode = firstHiddenNodes[pooledIndex];
        const matchingConnection = hiddenNode.connections.in.find(
          (connectionEntry) => connectionEntry.from === sourceNode,
        );
        const kernelIndex =
          kernelRowIndex * secondKernelWidth + kernelColumnIndex;

        if (matchingConnection != null) {
          matchingConnection.weight = secondKernelPattern[kernelIndex];
        }
      }
    }

    hiddenNode.bias = 0.031;
  });

  thirdHiddenNodes.forEach((hiddenNode) => {
    hiddenNode.connections.in.forEach((connectionEntry) => {
      connectionEntry.weight = 0;
    });

    for (
      let kernelRowIndex = 0;
      kernelRowIndex < thirdKernelHeight;
      kernelRowIndex += 1
    ) {
      for (
        let kernelColumnIndex = 0;
        kernelColumnIndex < thirdKernelWidth;
        kernelColumnIndex += 1
      ) {
        const pooledIndex = kernelRowIndex * secondPooledWidth + kernelColumnIndex;
        const sourceNode = secondHiddenNodes[pooledIndex];
        const matchingConnection = hiddenNode.connections.in.find(
          (connectionEntry) => connectionEntry.from === sourceNode,
        );
        const kernelIndex =
          kernelRowIndex * thirdKernelWidth + kernelColumnIndex;

        if (matchingConnection != null) {
          matchingConnection.weight = thirdKernelPattern[kernelIndex];
        }
      }
    }

    hiddenNode.bias = 0.041;
  });

  outputNodes.forEach((outputNode, outputNodeIndex) => {
    outputNode.connections.in.forEach((connectionEntry, connectionIndex) => {
      connectionEntry.weight =
        0.024 + outputNodeIndex * 0.012 + connectionIndex * 0.006;
    });
    outputNode.bias = 0.01 + outputNodeIndex * 0.01;
  });

  return { network };
}

function createDeepPostPoolMultiChannelHeuristicConvPromotionScenario(): {
  network: Network;
} {
  const inputChannels = 2;
  const inputHeight = 6;
  const inputWidth = 6;
  const firstKernelHeight = 2;
  const firstKernelWidth = 2;
  const poolingKernelHeight = 2;
  const poolingKernelWidth = 2;
  const poolingStrideHeight = 1;
  const poolingStrideWidth = 1;
  const secondKernelHeight = 2;
  const secondKernelWidth = 2;
  const thirdKernelHeight = 2;
  const thirdKernelWidth = 2;
  const strideHeight = 1;
  const strideWidth = 1;
  const firstOutputHeight = inputHeight - firstKernelHeight + 1;
  const firstOutputWidth = inputWidth - firstKernelWidth + 1;
  const firstPooledHeight =
    Math.floor(
      (firstOutputHeight - poolingKernelHeight) / poolingStrideHeight,
    ) + 1;
  const firstPooledWidth =
    Math.floor(
      (firstOutputWidth - poolingKernelWidth) / poolingStrideWidth,
    ) + 1;
  const secondOutputHeight = firstPooledHeight - secondKernelHeight + 1;
  const secondOutputWidth = firstPooledWidth - secondKernelWidth + 1;
  const secondPooledHeight =
    Math.floor(
      (secondOutputHeight - poolingKernelHeight) / poolingStrideHeight,
    ) + 1;
  const secondPooledWidth =
    Math.floor(
      (secondOutputWidth - poolingKernelWidth) / poolingStrideWidth,
    ) + 1;
  const thirdOutputHeight = secondPooledHeight - thirdKernelHeight + 1;
  const thirdOutputWidth = secondPooledWidth - thirdKernelWidth + 1;
  const outputChannels = 2;
  const network = Network.createMLP(72, [50, 18, 2], 2);
  const inputNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'input',
  );
  const hiddenNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'hidden',
  );
  const firstHiddenNodes = hiddenNodes.slice(0, 50);
  const secondHiddenNodes = hiddenNodes.slice(50, 68);
  const thirdHiddenNodes = hiddenNodes.slice(68);
  const outputNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'output',
  );
  const firstChannelKernelPatterns = [
    [
      [0.1, -0.05, 0.03, 0.08],
      [-0.04, 0.07, 0.02, -0.01],
    ],
    [
      [0.06, 0.01, -0.02, 0.09],
      [0.05, -0.03, 0.04, 0.11],
    ],
  ];
  const secondChannelKernelPatterns = [
    [
      [0.12, -0.06, 0.02, 0.07],
      [-0.03, 0.05, 0.09, -0.04],
    ],
    [
      [0.08, 0.01, -0.05, 0.06],
      [0.04, -0.02, 0.03, 0.1],
    ],
  ];
  const thirdChannelKernelPatterns = [
    [
      [0.13, -0.07, 0.04, 0.09],
      [-0.02, 0.08, 0.05, -0.01],
    ],
    [
      [0.07, 0.02, -0.04, 0.1],
      [0.03, -0.05, 0.06, 0.12],
    ],
  ];

  firstHiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
    const outputChannelIndex = Math.floor(
      hiddenNodeIndex / (firstOutputHeight * firstOutputWidth),
    );
    const outputSpatialIndex =
      hiddenNodeIndex % (firstOutputHeight * firstOutputWidth);
    const outputRow = Math.floor(outputSpatialIndex / firstOutputWidth);
    const outputColumn = outputSpatialIndex % firstOutputWidth;

    hiddenNode.connections.in.forEach((connectionEntry) => {
      connectionEntry.weight = 0;
    });

    for (
      let inputChannelIndex = 0;
      inputChannelIndex < inputChannels;
      inputChannelIndex += 1
    ) {
      for (
        let kernelRowIndex = 0;
        kernelRowIndex < firstKernelHeight;
        kernelRowIndex += 1
      ) {
        for (
          let kernelColumnIndex = 0;
          kernelColumnIndex < firstKernelWidth;
          kernelColumnIndex += 1
        ) {
          const inputRow = outputRow * strideHeight + kernelRowIndex;
          const inputColumn = outputColumn * strideWidth + kernelColumnIndex;
          const sourceIndex =
            inputChannelIndex * (inputHeight * inputWidth) +
            inputRow * inputWidth +
            inputColumn;
          const kernelIndex =
            kernelRowIndex * firstKernelWidth + kernelColumnIndex;
          const sourceNode = inputNodes[sourceIndex];
          const matchingConnection = hiddenNode.connections.in.find(
            (connectionEntry) => connectionEntry.from === sourceNode,
          );

          if (matchingConnection != null) {
            matchingConnection.weight =
              firstChannelKernelPatterns[outputChannelIndex]?.[
                inputChannelIndex
              ]?.[kernelIndex] ?? 0;
          }
        }
      }
    }

    hiddenNode.bias = 0.02 + outputChannelIndex * 0.01;
  });

  secondHiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
    const outputChannelIndex = Math.floor(
      hiddenNodeIndex / (secondOutputHeight * secondOutputWidth),
    );
    const outputSpatialIndex =
      hiddenNodeIndex % (secondOutputHeight * secondOutputWidth);
    const outputRow = Math.floor(outputSpatialIndex / secondOutputWidth);
    const outputColumn = outputSpatialIndex % secondOutputWidth;

    hiddenNode.connections.in.forEach((connectionEntry) => {
      connectionEntry.weight = 0;
    });

    for (
      let inputChannelIndex = 0;
      inputChannelIndex < inputChannels;
      inputChannelIndex += 1
    ) {
      for (
        let kernelRowIndex = 0;
        kernelRowIndex < secondKernelHeight;
        kernelRowIndex += 1
      ) {
        for (
          let kernelColumnIndex = 0;
          kernelColumnIndex < secondKernelWidth;
          kernelColumnIndex += 1
        ) {
          const pooledRow = outputRow * strideHeight + kernelRowIndex;
          const pooledColumn = outputColumn * strideWidth + kernelColumnIndex;
          const sourceIndex =
            inputChannelIndex * (firstOutputHeight * firstOutputWidth) +
            pooledRow * firstPooledWidth +
            pooledColumn;
          const kernelIndex =
            kernelRowIndex * secondKernelWidth + kernelColumnIndex;
          const sourceNode = firstHiddenNodes[sourceIndex];
          const matchingConnection = hiddenNode.connections.in.find(
            (connectionEntry) => connectionEntry.from === sourceNode,
          );

          if (matchingConnection != null) {
            matchingConnection.weight =
              secondChannelKernelPatterns[outputChannelIndex]?.[
                inputChannelIndex
              ]?.[kernelIndex] ?? 0;
          }
        }
      }
    }

    hiddenNode.bias = 0.03 + outputChannelIndex * 0.01;
  });

  thirdHiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
    const outputChannelIndex = Math.floor(
      hiddenNodeIndex / (thirdOutputHeight * thirdOutputWidth),
    );
    const outputSpatialIndex =
      hiddenNodeIndex % (thirdOutputHeight * thirdOutputWidth);
    const outputRow = Math.floor(outputSpatialIndex / thirdOutputWidth);
    const outputColumn = outputSpatialIndex % thirdOutputWidth;

    hiddenNode.connections.in.forEach((connectionEntry) => {
      connectionEntry.weight = 0;
    });

    for (
      let inputChannelIndex = 0;
      inputChannelIndex < inputChannels;
      inputChannelIndex += 1
    ) {
      for (
        let kernelRowIndex = 0;
        kernelRowIndex < thirdKernelHeight;
        kernelRowIndex += 1
      ) {
        for (
          let kernelColumnIndex = 0;
          kernelColumnIndex < thirdKernelWidth;
          kernelColumnIndex += 1
        ) {
          const pooledRow = outputRow * strideHeight + kernelRowIndex;
          const pooledColumn = outputColumn * strideWidth + kernelColumnIndex;
          const sourceIndex =
            inputChannelIndex * (secondOutputHeight * secondOutputWidth) +
            pooledRow * secondPooledWidth +
            pooledColumn;
          const kernelIndex =
            kernelRowIndex * thirdKernelWidth + kernelColumnIndex;
          const sourceNode = secondHiddenNodes[sourceIndex];
          const matchingConnection = hiddenNode.connections.in.find(
            (connectionEntry) => connectionEntry.from === sourceNode,
          );

          if (matchingConnection != null) {
            matchingConnection.weight =
              thirdChannelKernelPatterns[outputChannelIndex]?.[
                inputChannelIndex
              ]?.[kernelIndex] ?? 0;
          }
        }
      }
    }

    hiddenNode.bias = 0.04 + outputChannelIndex * 0.01;
  });

  outputNodes.forEach((outputNode, outputNodeIndex) => {
    outputNode.connections.in.forEach((connectionEntry, connectionIndex) => {
      connectionEntry.weight =
        0.018 + outputNodeIndex * 0.014 + connectionIndex * 0.003;
    });
    outputNode.bias = 0.012 + outputNodeIndex * 0.008;
  });

  return { network };
}

function createUnsafePostPoolStackedHeuristicConvPromotionScenario(): {
  network: Network;
} {
  const scenario = createPostPoolStackedHeuristicConvPromotionScenario();
  const hiddenNodes = scenario.network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'hidden',
  );
  const firstHiddenNodes = hiddenNodes.slice(0, 16);
  const secondHiddenNodes = hiddenNodes.slice(16);
  const strayConnection = secondHiddenNodes[0]?.connections.in.find(
    (connectionEntry) => connectionEntry.from === firstHiddenNodes[15],
  );

  if (strayConnection != null) {
    strayConnection.weight = 0.22;
  }

  return scenario;
}

function createUnsafePostPoolMultiChannelHeuristicConvPromotionScenario(): {
  network: Network;
} {
  const scenario = createPostPoolMultiChannelHeuristicConvPromotionScenario();
  const hiddenNodes = scenario.network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'hidden',
  );
  const firstHiddenNodes = hiddenNodes.slice(0, 18);
  const secondHiddenNodes = hiddenNodes.slice(18);
  const strayConnection = secondHiddenNodes[0]?.connections.in.find(
    (connectionEntry) => connectionEntry.from === firstHiddenNodes[8],
  );

  if (strayConnection != null) {
    strayConnection.weight = 0.24;
  }

  return scenario;
}

function createEarlierFlattenedPostPoolConsumerFallbackScenario(): {
  network: Network;
} {
  const inputHeight = 5;
  const inputWidth = 5;
  const firstKernelHeight = 2;
  const firstKernelWidth = 2;
  const poolingKernelHeight = 2;
  const poolingKernelWidth = 2;
  const poolingStrideHeight = 1;
  const poolingStrideWidth = 1;
  const secondKernelHeight = 2;
  const secondKernelWidth = 2;
  const strideHeight = 1;
  const strideWidth = 1;
  const firstOutputHeight = inputHeight - firstKernelHeight + 1;
  const firstOutputWidth = inputWidth - firstKernelWidth + 1;
  const pooledHeight =
    Math.floor(
      (firstOutputHeight - poolingKernelHeight) / poolingStrideHeight,
    ) + 1;
  const pooledWidth =
    Math.floor(
      (firstOutputWidth - poolingKernelWidth) / poolingStrideWidth,
    ) + 1;
  const secondOutputWidth = pooledWidth - secondKernelWidth + 1;
  const network = Network.createMLP(25, [16, 4, 3], 2);
  const inputNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'input',
  );
  const hiddenNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'hidden',
  );
  const firstHiddenNodes = hiddenNodes.slice(0, 16);
  const secondHiddenNodes = hiddenNodes.slice(16, 20);
  const thirdHiddenNodes = hiddenNodes.slice(20);
  const outputNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'output',
  );
  const firstKernelPattern = [0.04, -0.06, 0.08, 0.11];
  const secondKernelPattern = [0.07, -0.02, 0.05, 0.09];

  firstHiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
    const outputRow = Math.floor(hiddenNodeIndex / firstOutputWidth);
    const outputColumn = hiddenNodeIndex % firstOutputWidth;
    const inputBaseRow = outputRow * strideHeight;
    const inputBaseColumn = outputColumn * strideWidth;

    hiddenNode.connections.in.forEach((connectionEntry) => {
      connectionEntry.weight = 0;
    });

    for (
      let kernelRowIndex = 0;
      kernelRowIndex < firstKernelHeight;
      kernelRowIndex += 1
    ) {
      for (
        let kernelColumnIndex = 0;
        kernelColumnIndex < firstKernelWidth;
        kernelColumnIndex += 1
      ) {
        const inputRow = inputBaseRow + kernelRowIndex;
        const inputColumn = inputBaseColumn + kernelColumnIndex;
        const sourceIndex = inputRow * inputWidth + inputColumn;
        const kernelIndex =
          kernelRowIndex * firstKernelWidth + kernelColumnIndex;
        const sourceNode = inputNodes[sourceIndex];
        const matchingConnection = hiddenNode.connections.in.find(
          (connectionEntry) => connectionEntry.from === sourceNode,
        );

        if (matchingConnection != null) {
          matchingConnection.weight = firstKernelPattern[kernelIndex];
        }
      }
    }

    hiddenNode.bias = 0.025;
  });

  secondHiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
    const outputRow = Math.floor(hiddenNodeIndex / secondOutputWidth);
    const outputColumn = hiddenNodeIndex % secondOutputWidth;
    const pooledInputBaseRow = outputRow * strideHeight;
    const pooledInputBaseColumn = outputColumn * strideWidth;

    hiddenNode.connections.in.forEach((connectionEntry) => {
      connectionEntry.weight = 0;
    });

    for (
      let kernelRowIndex = 0;
      kernelRowIndex < secondKernelHeight;
      kernelRowIndex += 1
    ) {
      for (
        let kernelColumnIndex = 0;
        kernelColumnIndex < secondKernelWidth;
        kernelColumnIndex += 1
      ) {
        const pooledRow = pooledInputBaseRow + kernelRowIndex;
        const pooledColumn = pooledInputBaseColumn + kernelColumnIndex;
        const pooledIndex = pooledRow * pooledWidth + pooledColumn;
        const sourceNode = firstHiddenNodes[pooledIndex];
        const matchingConnection = hiddenNode.connections.in.find(
          (connectionEntry) => connectionEntry.from === sourceNode,
        );
        const kernelIndex =
          kernelRowIndex * secondKernelWidth + kernelColumnIndex;

        if (matchingConnection != null) {
          matchingConnection.weight = secondKernelPattern[kernelIndex];
        }
      }
    }

    hiddenNode.bias = 0.035;
  });

  thirdHiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
    hiddenNode.connections.in.forEach((connectionEntry, connectionIndex) => {
      connectionEntry.weight =
        0.018 + hiddenNodeIndex * 0.012 + connectionIndex * 0.004;
    });
    hiddenNode.bias = 0.014 + hiddenNodeIndex * 0.006;
  });

  outputNodes.forEach((outputNode, outputNodeIndex) => {
    outputNode.connections.in.forEach((connectionEntry, connectionIndex) => {
      connectionEntry.weight =
        0.02 + outputNodeIndex * 0.015 + connectionIndex * 0.003;
    });
    outputNode.bias = 0.01 + outputNodeIndex * 0.008;
  });

  return { network };
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

function createInvalidPoolingMappings(): Pool2DMapping[] {
  return [
    {
      afterLayerIndex: 1,
      type: 'MaxPool',
      kernelHeight: 2,
      kernelWidth: 2,
      strideHeight: 0,
      strideWidth: 0,
    },
  ];
}

function createPostPoolStackedConvMappings(): Conv2DMapping[] {
  return [
    {
      layerIndex: 1,
      inHeight: 5,
      inWidth: 5,
      inChannels: 1,
      kernelHeight: 2,
      kernelWidth: 2,
      strideHeight: 1,
      strideWidth: 1,
      outHeight: 4,
      outWidth: 4,
      outChannels: 1,
    },
    {
      layerIndex: 2,
      inHeight: 3,
      inWidth: 3,
      inChannels: 1,
      kernelHeight: 2,
      kernelWidth: 2,
      strideHeight: 1,
      strideWidth: 1,
      outHeight: 2,
      outWidth: 2,
      outChannels: 1,
    },
  ];
}

function createDeepPostPoolStackedConvMappings(): Conv2DMapping[] {
  return [
    {
      layerIndex: 1,
      inHeight: 6,
      inWidth: 6,
      inChannels: 1,
      kernelHeight: 2,
      kernelWidth: 2,
      strideHeight: 1,
      strideWidth: 1,
      outHeight: 5,
      outWidth: 5,
      outChannels: 1,
    },
    {
      layerIndex: 2,
      inHeight: 4,
      inWidth: 4,
      inChannels: 1,
      kernelHeight: 2,
      kernelWidth: 2,
      strideHeight: 1,
      strideWidth: 1,
      outHeight: 3,
      outWidth: 3,
      outChannels: 1,
    },
    {
      layerIndex: 3,
      inHeight: 2,
      inWidth: 2,
      inChannels: 1,
      kernelHeight: 2,
      kernelWidth: 2,
      strideHeight: 1,
      strideWidth: 1,
      outHeight: 1,
      outWidth: 1,
      outChannels: 1,
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

    describe('given only a second-stage Conv mapping after a pooled predecessor', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createPostPoolStackedHeuristicConvPromotionScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          conv2dMappings: [createPostPoolStackedConvMappings()[1]],
          pool2dMappings: createPoolingMappings(),
        });
      });

      describe('when the upstream Conv mapping is missing', () => {
        it('skips the second-stage Conv emission', () => {
          // Assert
          expect(countGraphNodeType(onnxModel, 'Conv')).toBe(0);
        });
      });
    });

    describe('given a second-stage Conv mapping after an unusable pooled predecessor', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createPostPoolStackedHeuristicConvPromotionScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          conv2dMappings: createPostPoolStackedConvMappings(),
          pool2dMappings: createInvalidPoolingMappings(),
        });
      });

      describe('when the derived pooled shape collapses below a valid spatial size', () => {
        it('keeps only the first Conv stage on the graph', () => {
          // Assert
          expect(countGraphNodeType(onnxModel, 'Conv')).toBe(1);
        });
      });
    });

    describe('given a second-stage Conv mapping after a pooled predecessor that matches width but not pooled shape', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createPostPoolStackedHeuristicConvPromotionScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          conv2dMappings: [
            createPostPoolStackedConvMappings()[0],
            {
              layerIndex: 2,
              inHeight: 1,
              inWidth: 9,
              inChannels: 1,
              kernelHeight: 1,
              kernelWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
              outHeight: 1,
              outWidth: 4,
              outChannels: 1,
            },
          ],
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
        });
      });

      describe('when the declared mapping still matches the pooled dense width', () => {
        it('emits both Conv operators', () => {
          // Assert
          expect(countGraphNodeType(onnxModel, 'Conv')).toBe(2);
        });
      });
    });

    describe('given a flattened second-stage Conv mapping whose pooled shape does not match the derived bridge', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createPostPoolStackedHeuristicConvPromotionScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          conv2dMappings: [
            createPostPoolStackedConvMappings()[0],
            {
              layerIndex: 2,
              inHeight: 1,
              inWidth: 9,
              inChannels: 1,
              kernelHeight: 1,
              kernelWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
              outHeight: 1,
              outWidth: 4,
              outChannels: 1,
            },
          ],
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

      describe('when the flattened pooled bridge cannot be reshaped back to the declared Conv input', () => {
        it('keeps only the first Conv stage on the graph', () => {
          // Assert
          expect(countGraphNodeType(onnxModel, 'Conv')).toBe(1);
        });
      });
    });

    describe('given a multi-channel second-stage Conv mapping behind a flattened pooled predecessor', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createPostPoolMultiChannelHeuristicConvPromotionScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          conv2dMappings: [
            {
              layerIndex: 1,
              inHeight: 4,
              inWidth: 4,
              inChannels: 2,
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
              outHeight: 3,
              outWidth: 3,
              outChannels: 2,
            },
            {
              layerIndex: 2,
              inHeight: 2,
              inWidth: 2,
              inChannels: 2,
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
              outHeight: 1,
              outWidth: 1,
              outChannels: 2,
            },
          ],
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

      describe('when the flattened pooled bridge matches the final-stage compact multi-channel subset', () => {
        it('emits a reshape bridge and keeps both Conv operators', () => {
          // Assert
          expect({
            convCount: countGraphNodeType(onnxModel, 'Conv'),
            reshapeCount: countGraphNodeType(onnxModel, 'Reshape'),
          }).toEqual({
            convCount: 2,
            reshapeCount: 1,
          });
        });

        it('keeps reshape metadata initializers out of storage-fp16 rewriting', () => {
          // Arrange
          const scenario = createPostPoolMultiChannelHeuristicConvPromotionScenario();

          // Act
          const storageFp16Model = exportToONNX(scenario.network, {
            conv2dMappings: [
              {
                layerIndex: 1,
                inHeight: 4,
                inWidth: 4,
                inChannels: 2,
                kernelHeight: 2,
                kernelWidth: 2,
                strideHeight: 1,
                strideWidth: 1,
                outHeight: 3,
                outWidth: 3,
                outChannels: 2,
              },
              {
                layerIndex: 2,
                inHeight: 2,
                inWidth: 2,
                inChannels: 2,
                kernelHeight: 2,
                kernelWidth: 2,
                strideHeight: 1,
                strideWidth: 1,
                outHeight: 1,
                outWidth: 1,
                outChannels: 2,
              },
            ],
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
            precision: {
              mode: 'storage-fp16',
            },
          });
          const convertedConvInitializer = storageFp16Model.graph.initializer.find(
            (initializerTensor) => initializerTensor.name === 'ConvW0',
          );
          const nonEligibleInitializer = storageFp16Model.graph.initializer.find(
            (initializerTensor) =>
              !/^W\d+$/.test(initializerTensor.name) &&
              !/^B\d+$/.test(initializerTensor.name) &&
              !/^W\d+_n\d+$/.test(initializerTensor.name) &&
              !/^B\d+_n\d+$/.test(initializerTensor.name) &&
              !/^ConvW\d+$/.test(initializerTensor.name) &&
              !/^ConvB\d+$/.test(initializerTensor.name),
          );

          // Assert
          expect(
            convertedConvInitializer?.data_type === 10 &&
              nonEligibleInitializer !== undefined &&
              nonEligibleInitializer.data_type !== 10,
          ).toBe(true);
        });
      });
    });

    describe('given explicit Conv mappings for a 25-16-4-3-2 network with flattenAfterPooling enabled', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createEarlierFlattenedPostPoolConsumerFallbackScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          conv2dMappings: createPostPoolStackedConvMappings(),
          flattenAfterPooling: true,
          pool2dMappings: [
            {
              afterLayerIndex: 1,
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
              type: 'MaxPool',
            },
          ],
        });
      });

      describe('when the flattened pooled consumer is not the final hidden stage', () => {
        it('keeps the declared later Conv stage off the reshape-bridge path', () => {
          // Assert
          expect({
            convCount: countGraphNodeType(onnxModel, 'Conv'),
            reshapeCount: countGraphNodeType(onnxModel, 'Reshape'),
          }).toEqual({
            convCount: 1,
            reshapeCount: 0,
          });
        });

      });
    });

    describe('given explicit Conv mappings for a 36-25-9-1-2 network with two pooled sites and flattenAfterPooling enabled', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createDeepPostPoolStackedHeuristicConvPromotionScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          conv2dMappings: createDeepPostPoolStackedConvMappings(),
          flattenAfterPooling: true,
          pool2dMappings: [
            {
              afterLayerIndex: 1,
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
              type: 'MaxPool',
            },
            {
              afterLayerIndex: 2,
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
              type: 'MaxPool',
            },
          ],
        });
      });

      describe('when the first flattened pool would force a repeated flatten-bridge chain', () => {
        it('keeps every later declared Conv stage on the dense fallback path', () => {
          // Assert
          expect({
            convCount: countGraphNodeType(onnxModel, 'Conv'),
            reshapeCount: countGraphNodeType(onnxModel, 'Reshape'),
          }).toEqual({
            convCount: 1,
            reshapeCount: 0,
          });
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

    describe('given a 25-9-2 network with shared Conv-style kernels and auto-promotion enabled', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createHeuristicConvPromotionScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          autoPromoteInferredConv: true,
          includeMetadata: true,
        });
      });

      describe('when the inferred Conv layer passes the safety gate', () => {
        it('emits a Conv operator', () => {
          // Assert
          expect(hasGraphNodeType(onnxModel, 'Conv')).toBe(true);
        });

        it('records conv2d_layers metadata instead of inference-only metadata', () => {
          // Assert
          expect(hasMetadataKey(onnxModel, 'conv2d_layers')).toBe(true);
        });

        it('does not keep conv2d_inferred_layers metadata for the promoted layer', () => {
          // Assert
          expect(hasMetadataKey(onnxModel, 'conv2d_inferred_layers')).toBe(
            false,
          );
        });
      });
    });

    describe('given a 25-9-2 network with non-shared spatial weights and auto-promotion enabled', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createUnsafeHeuristicConvPromotionScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          autoPromoteInferredConv: true,
          includeMetadata: true,
        });
      });

      describe('when the inferred Conv layer fails the safety gate', () => {
        it('does not emit a Conv operator', () => {
          // Assert
          expect(hasGraphNodeType(onnxModel, 'Conv')).toBe(false);
        });

        it('keeps conv2d_inferred_layers metadata', () => {
          // Assert
          expect(hasMetadataKey(onnxModel, 'conv2d_inferred_layers')).toBe(
            true,
          );
        });
      });
    });

    describe('given an 18-8-2 network with shared multi-channel Conv-style kernels and auto-promotion enabled', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createMultiChannelHeuristicConvPromotionScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          autoPromoteInferredConv: true,
          includeMetadata: true,
        });
      });

      describe('when the inferred multi-channel Conv layer passes the safety gate', () => {
        it('emits a Conv operator', () => {
          // Assert
          expect(hasGraphNodeType(onnxModel, 'Conv')).toBe(true);
        });

        it('records the promoted multi-channel Conv spec', () => {
          // Assert
          expect(JSON.parse(getMetadataValue(onnxModel, 'conv2d_specs') ?? '[]')).toEqual([
            {
              inChannels: 2,
              inHeight: 3,
              inWidth: 3,
              kernelHeight: 2,
              kernelWidth: 2,
              layerIndex: 1,
              outChannels: 2,
              outHeight: 2,
              outWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
            },
          ]);
        });

        it('does not keep multi-channel inference metadata once promoted', () => {
          // Assert
          expect(hasMetadataKey(onnxModel, 'conv2d_inferred_specs')).toBe(false);
        });
      });
    });

    describe('given an 18-8-2 network with non-shared multi-channel spatial weights and auto-promotion enabled', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createUnsafeMultiChannelHeuristicConvPromotionScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          autoPromoteInferredConv: true,
          includeMetadata: true,
        });
      });

      describe('when the inferred multi-channel Conv layer fails the safety gate', () => {
        it('does not emit a Conv operator', () => {
          // Assert
          expect(hasGraphNodeType(onnxModel, 'Conv')).toBe(false);
        });

        it('keeps the inferred multi-channel Conv spec as metadata only', () => {
          // Assert
          expect(
            JSON.parse(getMetadataValue(onnxModel, 'conv2d_inferred_specs') ?? '[]'),
          ).toEqual([
            {
              inChannels: 2,
              inHeight: 3,
              inWidth: 3,
              kernelHeight: 2,
              kernelWidth: 2,
              layerIndex: 1,
              note: 'heuristic_inferred_no_export_applied',
              outChannels: 2,
              outHeight: 2,
              outWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
            },
          ]);
        });
      });
    });

    describe('given a 25-9-4-2 network with shared stacked Conv-style kernels and auto-promotion enabled', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createStackedHeuristicConvPromotionScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          autoPromoteInferredConv: true,
          includeMetadata: true,
        });
      });

      describe('when the stacked inferred Conv layers pass the safety gate', () => {
        it('emits two Conv operators', () => {
          // Assert
          expect(countGraphNodeType(onnxModel, 'Conv')).toBe(2);
        });
      });
    });

    describe('given a 25-9-4-2 network with a pooled-and-flattened first Conv-like stage and auto-promotion enabled', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createStackedHeuristicConvPromotionScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          autoPromoteInferredConv: true,
          flattenAfterPooling: true,
          includeMetadata: true,
          pool2dMappings: createPoolingMappings(),
        });
      });

      describe('when the previous promoted stage changes shape through pool-and-flatten metadata', () => {
        it('does not emit a second Conv operator', () => {
          // Assert
          expect(countGraphNodeType(onnxModel, 'Conv')).toBe(1);
        });
      });
    });

    describe('given a 25-16-4-2 network whose second Conv-like stage matches the derived post-pool shape', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createPostPoolStackedHeuristicConvPromotionScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          autoPromoteInferredConv: true,
          includeMetadata: true,
          pool2dMappings: [
            {
              afterLayerIndex: 1,
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
              type: 'MaxPool',
            },
          ],
        });
      });

      describe('when the exporter derives the downstream spatial shape from the pool mapping', () => {
        it('emits two Conv operators', () => {
          // Assert
          expect(countGraphNodeType(onnxModel, 'Conv')).toBe(2);
        });

        it('records the second promoted Conv spec with the pooled 3x3 input shape', () => {
          // Assert
          expect(JSON.parse(getMetadataValue(onnxModel, 'conv2d_specs') ?? '[]')).toEqual([
            {
              inChannels: 1,
              inHeight: 5,
              inWidth: 5,
              kernelHeight: 2,
              kernelWidth: 2,
              layerIndex: 1,
              outChannels: 1,
              outHeight: 4,
              outWidth: 4,
              strideHeight: 1,
              strideWidth: 1,
            },
            {
              inChannels: 1,
              inHeight: 3,
              inWidth: 3,
              kernelHeight: 2,
              kernelWidth: 2,
              layerIndex: 2,
              outChannels: 1,
              outHeight: 2,
              outWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
            },
          ]);
        });
      });
    });

    describe('given a 25-16-4-2 network whose second Conv-like stage matches the derived post-pool shape after flatten', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createPostPoolStackedHeuristicConvPromotionScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          autoPromoteInferredConv: true,
          flattenAfterPooling: true,
          includeMetadata: true,
          pool2dMappings: [
            {
              afterLayerIndex: 1,
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
              type: 'MaxPool',
            },
          ],
        });
      });

      describe('when the pooled bridge is flattened before a later Conv-like stage', () => {
        it('emits a reshape bridge and keeps both Conv operators', () => {
          // Assert
          expect({
            convCount: countGraphNodeType(onnxModel, 'Conv'),
            reshapeCount: countGraphNodeType(onnxModel, 'Reshape'),
          }).toEqual({
            convCount: 2,
            reshapeCount: 1,
          });
        });
      });
    });

    describe('given a 25-16-4-3-2 network with a pooled-and-flattened earlier Conv-like consumer and auto-promotion enabled', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createEarlierFlattenedPostPoolConsumerFallbackScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          autoPromoteInferredConv: true,
          flattenAfterPooling: true,
          includeMetadata: true,
          pool2dMappings: [
            {
              afterLayerIndex: 1,
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
              type: 'MaxPool',
            },
          ],
        });
      });

      describe('when the flattened pooled consumer is not the final hidden stage', () => {
        it('keeps the later Conv-like stage on the metadata-only fallback path', () => {
          // Assert
          expect({
            convCount: countGraphNodeType(onnxModel, 'Conv'),
            inferredSpecs: JSON.parse(
              getMetadataValue(onnxModel, 'conv2d_inferred_specs') ?? '[]',
            ),
            reshapeCount: countGraphNodeType(onnxModel, 'Reshape'),
          }).toEqual({
            convCount: 1,
            inferredSpecs: [],
            reshapeCount: 0,
          });
        });
      });
    });

    describe('given a 32-18-2-2 network whose second multi-channel Conv-like stage matches the derived post-pool shape', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createPostPoolMultiChannelHeuristicConvPromotionScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          autoPromoteInferredConv: true,
          includeMetadata: true,
          pool2dMappings: [
            {
              afterLayerIndex: 1,
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
              type: 'MaxPool',
            },
          ],
        });
      });

      describe('when the exporter derives the downstream pooled multi-channel shape', () => {
        it('emits two Conv operators', () => {
          // Assert
          expect(countGraphNodeType(onnxModel, 'Conv')).toBe(2);
        });

        it('records the second promoted Conv spec with the pooled 2x2x2 input shape', () => {
          // Assert
          expect(JSON.parse(getMetadataValue(onnxModel, 'conv2d_specs') ?? '[]')).toEqual([
            {
              inChannels: 2,
              inHeight: 4,
              inWidth: 4,
              kernelHeight: 2,
              kernelWidth: 2,
              layerIndex: 1,
              outChannels: 2,
              outHeight: 3,
              outWidth: 3,
              strideHeight: 1,
              strideWidth: 1,
            },
            {
              inChannels: 2,
              inHeight: 2,
              inWidth: 2,
              kernelHeight: 2,
              kernelWidth: 2,
              layerIndex: 2,
              outChannels: 2,
              outHeight: 1,
              outWidth: 1,
              strideHeight: 1,
              strideWidth: 1,
            },
          ]);
        });
      });
    });

    describe('given a 32-18-2-2 network whose pooled multi-channel downstream stage still uses non-compact source nodes', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario =
          createUnsafePostPoolMultiChannelHeuristicConvPromotionScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          autoPromoteInferredConv: true,
          includeMetadata: true,
          pool2dMappings: [
            {
              afterLayerIndex: 1,
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
              type: 'MaxPool',
            },
          ],
        });
      });

      describe('when pooled promotion would depend on source nodes outside the compact per-channel slice', () => {
        it('keeps the second multi-channel stage off the Conv promotion path', () => {
          // Assert
          expect(countGraphNodeType(onnxModel, 'Conv')).toBe(1);
        });

        it('keeps the second multi-channel stage as inferred metadata only', () => {
          // Assert
          expect(
            JSON.parse(getMetadataValue(onnxModel, 'conv2d_inferred_specs') ?? '[]'),
          ).toEqual([
            {
              inChannels: 2,
              inHeight: 2,
              inWidth: 2,
              kernelHeight: 2,
              kernelWidth: 2,
              layerIndex: 2,
              note: 'heuristic_inferred_no_export_applied',
              outChannels: 2,
              outHeight: 1,
              outWidth: 1,
              strideHeight: 1,
              strideWidth: 1,
            },
          ]);
        });
      });
    });

    describe('given a 32-18-2-2 network whose second multi-channel Conv-like stage matches the derived post-pool shape after flatten', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createPostPoolMultiChannelHeuristicConvPromotionScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          autoPromoteInferredConv: true,
          flattenAfterPooling: true,
          includeMetadata: true,
          pool2dMappings: [
            {
              afterLayerIndex: 1,
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
              type: 'MaxPool',
            },
          ],
        });
      });

      describe('when the pooled multi-channel bridge is flattened before a later Conv-like stage', () => {
        it('emits a reshape bridge and keeps both Conv operators', () => {
          // Assert
          expect({
            convCount: countGraphNodeType(onnxModel, 'Conv'),
            reshapeCount: countGraphNodeType(onnxModel, 'Reshape'),
          }).toEqual({
            convCount: 2,
            reshapeCount: 1,
          });
        });

        it('records the second promoted Conv spec with the pooled 2x2x2 input shape', () => {
          // Assert
          expect(JSON.parse(getMetadataValue(onnxModel, 'conv2d_specs') ?? '[]')).toEqual([
            {
              inChannels: 2,
              inHeight: 4,
              inWidth: 4,
              kernelHeight: 2,
              kernelWidth: 2,
              layerIndex: 1,
              outChannels: 2,
              outHeight: 3,
              outWidth: 3,
              strideHeight: 1,
              strideWidth: 1,
            },
            {
              inChannels: 2,
              inHeight: 2,
              inWidth: 2,
              kernelHeight: 2,
              kernelWidth: 2,
              layerIndex: 2,
              outChannels: 2,
              outHeight: 1,
              outWidth: 1,
              strideHeight: 1,
              strideWidth: 1,
            },
          ]);
        });
      });
    });

    describe('given a 36-25-9-1-2 network with two pooled downstream Conv-like stages and auto-promotion enabled', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createDeepPostPoolStackedHeuristicConvPromotionScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          autoPromoteInferredConv: true,
          includeMetadata: true,
          pool2dMappings: [
            {
              afterLayerIndex: 1,
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
              type: 'MaxPool',
            },
            {
              afterLayerIndex: 2,
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
              type: 'MaxPool',
            },
          ],
        });
      });

      describe('when the exporter derives pooled shapes across multiple stages', () => {
        it('emits three Conv operators', () => {
          // Assert
          expect(countGraphNodeType(onnxModel, 'Conv')).toBe(3);
        });

        it('records the third promoted Conv spec with the second pooled 2x2 input shape', () => {
          // Assert
          expect(JSON.parse(getMetadataValue(onnxModel, 'conv2d_specs') ?? '[]')).toEqual([
            {
              inChannels: 1,
              inHeight: 6,
              inWidth: 6,
              kernelHeight: 2,
              kernelWidth: 2,
              layerIndex: 1,
              outChannels: 1,
              outHeight: 5,
              outWidth: 5,
              strideHeight: 1,
              strideWidth: 1,
            },
            {
              inChannels: 1,
              inHeight: 4,
              inWidth: 4,
              kernelHeight: 2,
              kernelWidth: 2,
              layerIndex: 2,
              outChannels: 1,
              outHeight: 3,
              outWidth: 3,
              strideHeight: 1,
              strideWidth: 1,
            },
            {
              inChannels: 1,
              inHeight: 2,
              inWidth: 2,
              kernelHeight: 2,
              kernelWidth: 2,
              layerIndex: 3,
              outChannels: 1,
              outHeight: 1,
              outWidth: 1,
              strideHeight: 1,
              strideWidth: 1,
            },
          ]);
        });
      });
    });

    describe('given a 36-25-9-1-2 network with two pooled sites and flattenAfterPooling enabled', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createDeepPostPoolStackedHeuristicConvPromotionScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          autoPromoteInferredConv: true,
          flattenAfterPooling: true,
          includeMetadata: true,
          pool2dMappings: [
            {
              afterLayerIndex: 1,
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
              type: 'MaxPool',
            },
            {
              afterLayerIndex: 2,
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
              type: 'MaxPool',
            },
          ],
        });
      });

      describe('when the first pool flattens away the spatial view', () => {
        it('keeps only the first Conv operator on the graph', () => {
          // Assert
          expect(countGraphNodeType(onnxModel, 'Conv')).toBe(1);
        });

        it('drops later Conv inference before reshape or metadata survive', () => {
          // Assert
          expect({
            convSpecs: JSON.parse(getMetadataValue(onnxModel, 'conv2d_specs') ?? '[]'),
            inferredSpecs: JSON.parse(
              getMetadataValue(onnxModel, 'conv2d_inferred_specs') ?? '[]',
            ),
            reshapeCount: countGraphNodeType(onnxModel, 'Reshape'),
          }).toEqual({
            convSpecs: [
              {
                inChannels: 1,
                inHeight: 6,
                inWidth: 6,
                kernelHeight: 2,
                kernelWidth: 2,
                layerIndex: 1,
                outChannels: 1,
                outHeight: 5,
                outWidth: 5,
                strideHeight: 1,
                strideWidth: 1,
              },
            ],
            inferredSpecs: [],
            reshapeCount: 0,
          });
        });
      });
    });

    describe('given a 72-50-18-2 network with two pooled multi-channel downstream Conv-like stages and auto-promotion enabled', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createDeepPostPoolMultiChannelHeuristicConvPromotionScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          autoPromoteInferredConv: true,
          includeMetadata: true,
          pool2dMappings: [
            {
              afterLayerIndex: 1,
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
              type: 'MaxPool',
            },
            {
              afterLayerIndex: 2,
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
              type: 'MaxPool',
            },
          ],
        });
      });

      describe('when the exporter derives pooled multi-channel shapes across multiple stages', () => {
        it('emits three Conv operators', () => {
          // Assert
          expect(countGraphNodeType(onnxModel, 'Conv')).toBe(3);
        });

        it('records the third promoted Conv spec with the second pooled 2x2x2 input shape', () => {
          // Assert
          expect(JSON.parse(getMetadataValue(onnxModel, 'conv2d_specs') ?? '[]')).toEqual([
            {
              inChannels: 2,
              inHeight: 6,
              inWidth: 6,
              kernelHeight: 2,
              kernelWidth: 2,
              layerIndex: 1,
              outChannels: 2,
              outHeight: 5,
              outWidth: 5,
              strideHeight: 1,
              strideWidth: 1,
            },
            {
              inChannels: 2,
              inHeight: 4,
              inWidth: 4,
              kernelHeight: 2,
              kernelWidth: 2,
              layerIndex: 2,
              outChannels: 2,
              outHeight: 3,
              outWidth: 3,
              strideHeight: 1,
              strideWidth: 1,
            },
            {
              inChannels: 2,
              inHeight: 2,
              inWidth: 2,
              kernelHeight: 2,
              kernelWidth: 2,
              layerIndex: 3,
              outChannels: 2,
              outHeight: 1,
              outWidth: 1,
              strideHeight: 1,
              strideWidth: 1,
            },
          ]);
        });
      });
    });

    describe('given a 72-50-18-2 network with two pooled multi-channel sites and flattenAfterPooling enabled', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createDeepPostPoolMultiChannelHeuristicConvPromotionScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          autoPromoteInferredConv: true,
          flattenAfterPooling: true,
          includeMetadata: true,
          pool2dMappings: [
            {
              afterLayerIndex: 1,
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
              type: 'MaxPool',
            },
            {
              afterLayerIndex: 2,
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
              type: 'MaxPool',
            },
          ],
        });
      });

      describe('when the first pool flattens away the multi-channel spatial view', () => {
        it('keeps only the first Conv operator on the graph', () => {
          // Assert
          expect(countGraphNodeType(onnxModel, 'Conv')).toBe(1);
        });

        it('drops later multi-channel Conv inference before reshape or metadata survive', () => {
          // Assert
          expect({
            convSpecs: JSON.parse(getMetadataValue(onnxModel, 'conv2d_specs') ?? '[]'),
            inferredSpecs: JSON.parse(
              getMetadataValue(onnxModel, 'conv2d_inferred_specs') ?? '[]',
            ),
            reshapeCount: countGraphNodeType(onnxModel, 'Reshape'),
          }).toEqual({
            convSpecs: [
              {
                inChannels: 2,
                inHeight: 6,
                inWidth: 6,
                kernelHeight: 2,
                kernelWidth: 2,
                layerIndex: 1,
                outChannels: 2,
                outHeight: 5,
                outWidth: 5,
                strideHeight: 1,
                strideWidth: 1,
              },
            ],
            inferredSpecs: [],
            reshapeCount: 0,
          });
        });
      });
    });

    describe('given a pooled hidden layer whose upstream stage cannot be inferred as Conv-like', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const network = Network.createMLP(6, [4, 2], 1);

        // Act
        onnxModel = exportToONNX(network, {
          autoPromoteInferredConv: true,
          includeMetadata: true,
          pool2dMappings: createPoolingMappings(),
        });
      });

      describe('when no upstream Conv spec exists to anchor the pooled shape', () => {
        it('keeps the downstream stage off the Conv promotion path', () => {
          // Assert
          expect(countGraphNodeType(onnxModel, 'Conv')).toBe(0);
        });
      });
    });

    describe('given a post-pool stacked Conv-like chain with unusable pool geometry', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createPostPoolStackedHeuristicConvPromotionScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          autoPromoteInferredConv: true,
          includeMetadata: true,
          pool2dMappings: createInvalidPoolingMappings(),
        });
      });

      describe('when the derived pooled shape is invalid', () => {
        it('does not promote the downstream stage', () => {
          // Assert
          expect(countGraphNodeType(onnxModel, 'Conv')).toBe(1);
        });
      });
    });

    describe('given a 25-16-4-2 network whose downstream stage still uses extra non-pooled inputs', () => {
      let onnxModel: OnnxModel;

      beforeEach(() => {
        // Arrange
        const scenario = createUnsafePostPoolStackedHeuristicConvPromotionScenario();

        // Act
        onnxModel = exportToONNX(scenario.network, {
          autoPromoteInferredConv: true,
          includeMetadata: true,
          pool2dMappings: [
            {
              afterLayerIndex: 1,
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
              type: 'MaxPool',
            },
          ],
        });
      });

      describe('when pooled promotion would drop non-zero dense inputs outside the derived pooled slice', () => {
        it('keeps the second stage off the Conv promotion path', () => {
          // Assert
          expect(countGraphNodeType(onnxModel, 'Conv')).toBe(1);
        });

        it('keeps the second stage as inferred metadata only', () => {
          // Assert
          expect(JSON.parse(getMetadataValue(onnxModel, 'conv2d_inferred_specs') ?? '[]')).toEqual([
            {
              inChannels: 1,
              inHeight: 3,
              inWidth: 3,
              kernelHeight: 2,
              kernelWidth: 2,
              layerIndex: 2,
              note: 'heuristic_inferred_no_export_applied',
              outChannels: 1,
              outHeight: 2,
              outWidth: 2,
              strideHeight: 1,
              strideWidth: 1,
            },
          ]);
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

    describe('given flattenAfterPooling is enabled without an upstream pooled Conv input', () => {
      it('does not emit a reshape bridge before the Conv node', () => {
        // Arrange
        const scenario = createConvGroundworkScenario();

        // Act
        const onnxModel = exportToONNX(scenario.network, {
          conv2dMappings: scenario.mappings,
          flattenAfterPooling: true,
        });

        // Assert
        expect(
          onnxModel.graph.node.some(
            (graphNode) => graphNode.name === 'reshape_before_conv_l1',
          ),
        ).toBe(false);
      });
    });

    describe('given an explicit Conv activation mapping', () => {
      it('emits the requested activation operator instead of inferring one from the source nodes', () => {
        // Arrange
        const scenario = createConvGroundworkScenario();

        // Act
        const onnxModel = exportToONNX(scenario.network, {
          conv2dMappings: [
            {
              ...scenario.mappings[0],
              activation: 'Softplus',
            },
          ],
        });

        // Assert
        expect(hasGraphNodeType(onnxModel, 'Softplus')).toBe(true);
      });
    });

    describe('given Phase 7 static quantization calibration targets for the explicit Conv subset', () => {
      it('emits per-output-channel Conv weight parameter initializers deterministically', () => {
        // Arrange
        const scenario = createConvGroundworkScenario();

        // Act
        const onnxModel = exportToONNX(scenario.network, {
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
            weightGranularity: 'per-output-channel',
          },
        });
        const weightScaleTensor = onnxModel.graph.initializer.find(
          (initializerTensor) => initializerTensor.name === 'QuantConvWeightScale_l1',
        );
        const weightZeroPointTensor = onnxModel.graph.initializer.find(
          (initializerTensor) =>
            initializerTensor.name === 'QuantConvWeightZeroPoint_l1',
        );

        // Assert
        expect({
          weightGranularity: getMetadataValue(
            onnxModel,
            'quantization_weight_granularity',
          ),
          weightScaleDims: weightScaleTensor?.dims,
          weightScaleLength: weightScaleTensor?.float_data.length,
          weightZeroPointDims: weightZeroPointTensor?.dims,
          weightZeroPointLength: weightZeroPointTensor?.int32_data?.length,
        }).toEqual({
          weightGranularity: 'per-output-channel',
          weightScaleDims: [2],
          weightScaleLength: 2,
          weightZeroPointDims: [2],
          weightZeroPointLength: 2,
        });
      });
    });
  });

  describe('given a conv network where one inbound connection is removed from a hidden node', () => {
    describe('when the Conv layer is exported with allowPartialConnectivity', () => {
      it('exports successfully and uses a zero-weight fallback for the missing connection', () => {
        // Arrange – remove from the hidden node's connections.in so that
        // resolveInboundWeightOrZero cannot find the connection and ?? 0 fires.
        // allowPartialConnectivity bypasses the connectivity validation so the
        // export reaches the weight-lookup code with a missing connection.
        const scenario = createConvGroundworkScenario();
        const firstHiddenNode = scenario.network.nodes.find(
          (nodeEntry) => nodeEntry.type === 'hidden',
        )!;
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (firstHiddenNode as any).connections.in.splice(0, 1);

        // Act & Assert
        expect(() =>
          exportToONNX(scenario.network, {
            conv2dMappings: scenario.mappings,
            allowPartialConnectivity: true,
          }),
        ).not.toThrow();
      });
    });
  });
});
