import Network from '../network';
import { exportToONNX, importFromONNX } from './network.onnx';

type PoolingAuditAwareNetwork = Network & {
  _onnxPooling?: {
    flattenConsistency?: Array<{
      matches: boolean;
    }>;
  };
};

jest.retryTimes(2, { logErrorsBeforeRetry: true });

function buildRandomizedMultilayerPerceptron(
  inputSize: number,
  hiddenLayerSizes: number[],
  outputSize: number,
  seed = 42,
): Network {
  const network = Network.createMLP(inputSize, hiddenLayerSizes, outputSize);
  const nextRandomValue = createLinearCongruentialGenerator(seed);

  network.connections.forEach((connectionEntry, connectionIndex) => {
    connectionEntry.weight =
      (nextRandomValue() * 2 - 1) * 0.5 + connectionIndex * 1e-6;
  });

  network.nodes.forEach((nodeEntry, nodeIndex) => {
    if (nodeEntry.type !== 'input') {
      nodeEntry.bias = (nextRandomValue() * 2 - 1) * 0.1 + nodeIndex * 1e-6;
    }
  });

  return network;
}

function createLinearCongruentialGenerator(seed: number): () => number {
  let state = seed >>> 0;

  return () => {
    state = (state * 1_664_525 + 1_013_904_223) >>> 0;
    return state / 0xffff_ffff;
  };
}

function buildFlattenAuditMismatchNetwork(seed = 321): Network {
  const inputWidth = 3;
  const kernelHeight = 2;
  const kernelWidth = 2;
  const strideHeight = 1;
  const strideWidth = 1;
  const network = Network.createMLP(9, [4], 2);
  const inputNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'input',
  );
  const hiddenNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'hidden',
  );
  const outputNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type === 'output',
  );
  const kernelPattern = [0.11, -0.07, 0.05, 0.02];
  const nextRandomValue = createLinearCongruentialGenerator(seed);

  hiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
    const outputRow = Math.floor(hiddenNodeIndex / 2);
    const outputColumn = hiddenNodeIndex % 2;
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

        if (matchingConnection) {
          matchingConnection.weight = kernelPattern[kernelIndex];
        }
      }
    }

    hiddenNode.bias = 0.123;
  });

  outputNodes.forEach((outputNode, outputNodeIndex) => {
    outputNode.connections.in.forEach((connectionEntry, connectionIndex) => {
      connectionEntry.weight =
        (nextRandomValue() * 2 - 1) * 0.25 +
        outputNodeIndex * 0.01 +
        connectionIndex * 1e-6;
    });
    outputNode.bias =
      (nextRandomValue() * 2 - 1) * 0.05 + outputNodeIndex * 1e-6;
  });

  return network;
}

function buildAutoPromotedHeuristicConvRoundtripNetwork(seed = 777): Network {
  const inputWidth = 5;
  const kernelHeight = 3;
  const kernelWidth = 3;
  const strideHeight = 1;
  const strideWidth = 1;
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
  const kernelPattern = [
    0.09, -0.04, 0.02, 0.11, 0.07, -0.03, 0.05, 0.01, 0.08,
  ];
  const nextRandomValue = createLinearCongruentialGenerator(seed);

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

        if (matchingConnection) {
          matchingConnection.weight = kernelPattern[kernelIndex];
        }
      }
    }

    hiddenNode.bias = 0.045;
  });

  outputNodes.forEach((outputNode, outputNodeIndex) => {
    outputNode.connections.in.forEach((connectionEntry, connectionIndex) => {
      connectionEntry.weight =
        (nextRandomValue() * 2 - 1) * 0.2 +
        outputNodeIndex * 0.01 +
        connectionIndex * 1e-6;
    });
    outputNode.bias =
      (nextRandomValue() * 2 - 1) * 0.05 + outputNodeIndex * 1e-6;
  });

  return network;
}

function buildAutoPromotedMultiChannelHeuristicConvRoundtripNetwork(): Network {
  const inputChannels = 2;
  const inputHeight = 3;
  const inputWidth = 3;
  const kernelHeight = 2;
  const kernelWidth = 2;
  const strideHeight = 1;
  const strideWidth = 1;
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

          if (matchingConnection) {
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

  return network;
}

function buildAutoPromotedStackedHeuristicConvRoundtripNetwork(): Network {
  const inputWidth = 5;
  const firstKernelHeight = 3;
  const firstKernelWidth = 3;
  const secondKernelHeight = 2;
  const secondKernelWidth = 2;
  const strideHeight = 1;
  const strideWidth = 1;
  const firstOutputWidth = inputWidth - firstKernelWidth + 1;
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
    0.09, -0.04, 0.02, 0.11, 0.07, -0.03, 0.05, 0.01, 0.08,
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

        if (matchingConnection) {
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

        if (matchingConnection) {
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

  return network;
}

function buildAutoPromotedPostPoolStackedHeuristicConvRoundtripNetwork(): Network {
  const inputWidth = 5;
  const firstKernelHeight = 2;
  const firstKernelWidth = 2;
  const poolingKernelWidth = 2;
  const poolingStrideWidth = 1;
  const secondKernelHeight = 2;
  const secondKernelWidth = 2;
  const strideHeight = 1;
  const strideWidth = 1;
  const firstOutputWidth = inputWidth - firstKernelWidth + 1;
  const pooledWidth =
    Math.floor((firstOutputWidth - poolingKernelWidth) / poolingStrideWidth) +
    1;
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

        if (matchingConnection) {
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

        if (matchingConnection) {
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

  return network;
}

function buildAutoPromotedEarlierFlattenedPostPoolFallbackRoundtripNetwork(): Network {
  const inputWidth = 5;
  const firstKernelHeight = 2;
  const firstKernelWidth = 2;
  const poolingKernelWidth = 2;
  const poolingStrideWidth = 1;
  const secondKernelHeight = 2;
  const secondKernelWidth = 2;
  const strideHeight = 1;
  const strideWidth = 1;
  const firstOutputWidth = inputWidth - firstKernelWidth + 1;
  const pooledWidth =
    Math.floor((firstOutputWidth - poolingKernelWidth) / poolingStrideWidth) +
    1;
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

        if (matchingConnection) {
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

        if (matchingConnection) {
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

  return network;
}

function buildAutoPromotedPostPoolMultiChannelHeuristicConvRoundtripNetwork(): Network {
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
    Math.floor((firstOutputWidth - poolingKernelWidth) / poolingStrideWidth) +
    1;
  const secondOutputHeight = pooledHeight - secondKernelHeight + 1;
  const secondOutputWidth = pooledWidth - secondKernelWidth + 1;
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

          if (matchingConnection) {
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

          if (matchingConnection) {
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

  return network;
}

function buildUnsafePostPoolMultiChannelHeuristicConvRoundtripNetwork(): Network {
  const network =
    buildAutoPromotedPostPoolMultiChannelHeuristicConvRoundtripNetwork();
  const hiddenNodes = network.nodes.filter(
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

  return network;
}

function buildAutoPromotedDeepPostPoolStackedHeuristicConvRoundtripNetwork(): Network {
  const inputWidth = 6;
  const firstKernelHeight = 2;
  const firstKernelWidth = 2;
  const poolingKernelWidth = 2;
  const poolingStrideWidth = 1;
  const secondKernelHeight = 2;
  const secondKernelWidth = 2;
  const thirdKernelHeight = 2;
  const thirdKernelWidth = 2;
  const strideHeight = 1;
  const strideWidth = 1;
  const firstOutputWidth = inputWidth - firstKernelWidth + 1;
  const firstPooledWidth =
    Math.floor((firstOutputWidth - poolingKernelWidth) / poolingStrideWidth) +
    1;
  const secondOutputWidth = firstPooledWidth - secondKernelWidth + 1;
  const secondPooledWidth =
    Math.floor((secondOutputWidth - poolingKernelWidth) / poolingStrideWidth) +
    1;
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

        if (matchingConnection) {
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

        if (matchingConnection) {
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
        const pooledIndex =
          kernelRowIndex * secondPooledWidth + kernelColumnIndex;
        const sourceNode = secondHiddenNodes[pooledIndex];
        const matchingConnection = hiddenNode.connections.in.find(
          (connectionEntry) => connectionEntry.from === sourceNode,
        );
        const kernelIndex =
          kernelRowIndex * thirdKernelWidth + kernelColumnIndex;

        if (matchingConnection) {
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

  return network;
}

function buildAutoPromotedDeepPostPoolMultiChannelHeuristicConvRoundtripNetwork(): Network {
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
    Math.floor((firstOutputWidth - poolingKernelWidth) / poolingStrideWidth) +
    1;
  const secondOutputHeight = firstPooledHeight - secondKernelHeight + 1;
  const secondOutputWidth = firstPooledWidth - secondKernelWidth + 1;
  const secondPooledHeight =
    Math.floor(
      (secondOutputHeight - poolingKernelHeight) / poolingStrideHeight,
    ) + 1;
  const secondPooledWidth =
    Math.floor((secondOutputWidth - poolingKernelWidth) / poolingStrideWidth) +
    1;
  const thirdOutputHeight = secondPooledHeight - thirdKernelHeight + 1;
  const thirdOutputWidth = secondPooledWidth - thirdKernelWidth + 1;
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

          if (matchingConnection) {
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

          if (matchingConnection) {
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

          if (matchingConnection) {
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

  return network;
}

function createPostPoolStackedConvMappings() {
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

function createDeepPostPoolStackedConvMappings() {
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

function createPostPoolMultiChannelFlattenConvMappings() {
  return [
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
  ];
}

function countGraphNodeType(
  onnxModel: ReturnType<typeof exportToONNX>,
  operatorType: string,
): number {
  return onnxModel.graph.node.filter(
    (nodeEntry) => nodeEntry.op_type === operatorType,
  ).length;
}

function calculateMeanSquaredError(
  expectedValues: number[],
  actualValues: number[],
): number {
  const summedSquaredError = expectedValues.reduce(
    (accumulatedError, expectedValue, outputIndex) => {
      const difference = expectedValue - actualValues[outputIndex];
      return accumulatedError + difference * difference;
    },
    0,
  );

  return summedSquaredError / expectedValues.length;
}

describe('network onnx root chapter', () => {
  describe('public round-trip contract', () => {
    describe('given a deterministic 3-4-2 multilayer perceptron', () => {
      let meanSquaredError: number;

      beforeEach(() => {
        // Arrange
        const sourceNetwork = buildRandomizedMultilayerPerceptron(
          3,
          [4],
          2,
          123,
        );
        const inputValues = [0.25, -0.5, 0.9];
        const expectedOutput = sourceNetwork.activate(
          inputValues,
          false,
        ) as number[];
        const onnxModel = exportToONNX(sourceNetwork);
        const importedNetwork = importFromONNX(onnxModel);

        // Act
        const actualOutput = importedNetwork.activate(
          inputValues,
          false,
        ) as number[];
        meanSquaredError = calculateMeanSquaredError(
          expectedOutput,
          actualOutput,
        );
      });

      describe('when the reconstructed network is evaluated on the same sample', () => {
        it('keeps the mean squared error below 1e-12', () => {
          // Assert
          expect(meanSquaredError).toBeLessThan(1e-12);
        });
      });
    });

    describe('given a deterministic 5-6-5-3 multilayer perceptron', () => {
      let meanSquaredError: number;

      beforeEach(() => {
        // Arrange
        const sourceNetwork = buildRandomizedMultilayerPerceptron(
          5,
          [6, 5],
          3,
          999,
        );
        const inputValues = [0.1, -0.2, 0.3, -0.4, 0.5];
        const expectedOutput = sourceNetwork.activate(
          inputValues,
          false,
        ) as number[];
        const onnxModel = exportToONNX(sourceNetwork, {
          includeMetadata: true,
          batchDimension: true,
        });
        const importedNetwork = importFromONNX(onnxModel);

        // Act
        const actualOutput = importedNetwork.activate(
          inputValues,
          false,
        ) as number[];
        meanSquaredError = calculateMeanSquaredError(
          expectedOutput,
          actualOutput,
        );
      });

      describe('when metadata and batch dimensions are enabled', () => {
        it('keeps the mean squared error below 1e-12', () => {
          // Assert
          expect(meanSquaredError).toBeLessThan(1e-12);
        });
      });
    });

    describe('given Conv + Pool + Flatten metadata narrows to width 1 before a width-2 dense consumer', () => {
      let flattenConsistencyMatches: boolean | undefined;
      let meanSquaredError: number;

      beforeEach(() => {
        // Arrange
        const sourceNetwork = buildFlattenAuditMismatchNetwork();
        const inputValues = [0.6, -0.1, 0.4, 0.2, -0.7, 0.9, 0.3, -0.5, 0.8];
        const expectedOutput = sourceNetwork.activate(
          inputValues,
          false,
        ) as number[];
        const onnxModel = exportToONNX(sourceNetwork, {
          conv2dMappings: [
            {
              layerIndex: 1,
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
          ],
          flattenAfterPooling: true,
          includeMetadata: true,
          pool2dMappings: [
            {
              afterLayerIndex: 1,
              type: 'MaxPool',
              kernelHeight: 2,
              kernelWidth: 2,
              strideHeight: 2,
              strideWidth: 2,
            },
          ],
        });
        const importedNetwork = importFromONNX(
          onnxModel,
        ) as PoolingAuditAwareNetwork;

        // Act
        flattenConsistencyMatches =
          importedNetwork._onnxPooling?.flattenConsistency?.[0]?.matches;
        const actualOutput = importedNetwork.activate(
          inputValues,
          false,
        ) as number[];
        meanSquaredError = calculateMeanSquaredError(
          expectedOutput,
          actualOutput,
        );
      });

      describe('when the importer records the flatten audit', () => {
        it('marks the audit as a mismatch', () => {
          // Assert
          expect(flattenConsistencyMatches).toBe(false);
        });
      });

      describe('when the reconstructed network is evaluated on the same sample', () => {
        it('keeps the mean squared error below 1e-12 because the audit stays metadata-only', () => {
          // Assert
          expect(meanSquaredError).toBeLessThan(1e-12);
        });
      });
    });

    describe('given a heuristic Conv-like 25-9-2 network with shared kernels and auto-promotion enabled', () => {
      let meanSquaredError: number;

      beforeEach(() => {
        // Arrange
        const sourceNetwork = buildAutoPromotedHeuristicConvRoundtripNetwork();
        const inputValues = [
          0.25, -0.4, 0.7, 0.1, -0.3, 0.5, 0.2, -0.6, 0.8, -0.1, 0.4, -0.2, 0.9,
          0.3, -0.5, 0.6, 0.05, -0.7, 0.45, 0.15, -0.25, 0.35, 0.55, -0.45,
          0.65,
        ];
        const expectedOutput = sourceNetwork.activate(
          inputValues,
          false,
        ) as number[];
        const onnxModel = exportToONNX(sourceNetwork, {
          autoPromoteInferredConv: true,
          includeMetadata: true,
        });
        const importedNetwork = importFromONNX(onnxModel);

        // Act
        const actualOutput = importedNetwork.activate(
          inputValues,
          false,
        ) as number[];
        meanSquaredError = calculateMeanSquaredError(
          expectedOutput,
          actualOutput,
        );
      });

      describe('when the promoted Conv-backed network is evaluated on the same sample', () => {
        it('keeps the mean squared error below 1e-12', () => {
          // Assert
          expect(meanSquaredError).toBeLessThan(1e-12);
        });
      });
    });

    describe('given a heuristic Conv-like 18-8-2 network with shared multi-channel kernels and auto-promotion enabled', () => {
      let meanSquaredError: number;

      beforeEach(() => {
        // Arrange
        const sourceNetwork =
          buildAutoPromotedMultiChannelHeuristicConvRoundtripNetwork();
        const inputValues = [
          0.25, -0.4, 0.7, 0.1, -0.3, 0.5, 0.2, -0.6, 0.8, -0.1, 0.4, -0.2, 0.9,
          0.3, -0.5, 0.6, 0.05, -0.7,
        ];
        const expectedOutput = sourceNetwork.activate(
          inputValues,
          false,
        ) as number[];
        const onnxModel = exportToONNX(sourceNetwork, {
          autoPromoteInferredConv: true,
          includeMetadata: true,
        });
        const importedNetwork = importFromONNX(onnxModel);

        // Act
        const actualOutput = importedNetwork.activate(
          inputValues,
          false,
        ) as number[];
        meanSquaredError = calculateMeanSquaredError(
          expectedOutput,
          actualOutput,
        );
      });

      describe('when the promoted multi-channel Conv-backed network is evaluated on the same sample', () => {
        it('keeps the mean squared error below 1e-12', () => {
          // Assert
          expect(meanSquaredError).toBeLessThan(1e-12);
        });
      });
    });

    describe('given a heuristic Conv-like 25-9-4-2 network with stacked shared kernels and auto-promotion enabled', () => {
      let meanSquaredError: number;

      beforeEach(() => {
        // Arrange
        const sourceNetwork =
          buildAutoPromotedStackedHeuristicConvRoundtripNetwork();
        const inputValues = [
          0.25, -0.4, 0.7, 0.1, -0.3, 0.5, 0.2, -0.6, 0.8, -0.1, 0.4, -0.2, 0.9,
          0.3, -0.5, 0.6, 0.05, -0.7, 0.45, 0.15, -0.25, 0.35, 0.55, -0.45,
          0.65,
        ];
        const expectedOutput = sourceNetwork.activate(
          inputValues,
          false,
        ) as number[];
        const onnxModel = exportToONNX(sourceNetwork, {
          autoPromoteInferredConv: true,
          includeMetadata: true,
        });
        const importedNetwork = importFromONNX(onnxModel);

        // Act
        const actualOutput = importedNetwork.activate(
          inputValues,
          false,
        ) as number[];
        meanSquaredError = calculateMeanSquaredError(
          expectedOutput,
          actualOutput,
        );
      });

      describe('when the promoted stacked Conv-backed network is evaluated on the same sample', () => {
        it('keeps the mean squared error below 1e-12', () => {
          // Assert
          expect(meanSquaredError).toBeLessThan(1e-12);
        });
      });
    });

    describe('given a heuristic Conv-like 25-16-4-2 network with a pooled downstream stage and auto-promotion enabled', () => {
      let meanSquaredError: number;

      beforeEach(() => {
        // Arrange
        const sourceNetwork =
          buildAutoPromotedPostPoolStackedHeuristicConvRoundtripNetwork();
        const inputValues = [
          0.25, -0.4, 0.7, 0.1, -0.3, 0.5, 0.2, -0.6, 0.8, -0.1, 0.4, -0.2, 0.9,
          0.3, -0.5, 0.6, 0.05, -0.7, 0.45, 0.15, -0.25, 0.35, 0.55, -0.45,
          0.65,
        ];
        const expectedOutput = sourceNetwork.activate(
          inputValues,
          false,
        ) as number[];
        const onnxModel = exportToONNX(sourceNetwork, {
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
        const importedNetwork = importFromONNX(onnxModel);

        // Act
        const actualOutput = importedNetwork.activate(
          inputValues,
          false,
        ) as number[];
        meanSquaredError = calculateMeanSquaredError(
          expectedOutput,
          actualOutput,
        );
      });

      describe('when the promoted pooled Conv-backed network is evaluated on the same sample', () => {
        it('keeps the mean squared error below 1e-12', () => {
          // Assert
          expect(meanSquaredError).toBeLessThan(1e-12);
        });
      });
    });

    describe('given explicit Conv mappings for a 25-16-4-2 network with a pooled downstream stage', () => {
      let convCount: number;
      let meanSquaredError: number;

      beforeEach(() => {
        // Arrange
        const sourceNetwork =
          buildAutoPromotedPostPoolStackedHeuristicConvRoundtripNetwork();
        const inputValues = [
          0.25, -0.4, 0.7, 0.1, -0.3, 0.5, 0.2, -0.6, 0.8, -0.1, 0.4, -0.2, 0.9,
          0.3, -0.5, 0.6, 0.05, -0.7, 0.45, 0.15, -0.25, 0.35, 0.55, -0.45,
          0.65,
        ];
        const expectedOutput = sourceNetwork.activate(
          inputValues,
          false,
        ) as number[];
        const onnxModel = exportToONNX(sourceNetwork, {
          conv2dMappings: createPostPoolStackedConvMappings(),
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
        const importedNetwork = importFromONNX(onnxModel);

        // Act
        convCount = countGraphNodeType(onnxModel, 'Conv');
        const actualOutput = importedNetwork.activate(
          inputValues,
          false,
        ) as number[];
        meanSquaredError = calculateMeanSquaredError(
          expectedOutput,
          actualOutput,
        );
      });

      describe('when the declared pooled successor matches the derived post-pool shape', () => {
        it('keeps both Conv operators on the export graph', () => {
          // Assert
          expect(convCount).toBe(2);
        });

        it('keeps the mean squared error below 1e-12', () => {
          // Assert
          expect(meanSquaredError).toBeLessThan(1e-12);
        });
      });
    });

    describe('given a heuristic Conv-like 25-16-4-2 network with a pooled-and-flattened downstream stage and auto-promotion enabled', () => {
      let meanSquaredError: number;

      beforeEach(() => {
        // Arrange
        const sourceNetwork =
          buildAutoPromotedPostPoolStackedHeuristicConvRoundtripNetwork();
        const inputValues = [
          0.25, -0.4, 0.7, 0.1, -0.3, 0.5, 0.2, -0.6, 0.8, -0.1, 0.4, -0.2, 0.9,
          0.3, -0.5, 0.6, 0.05, -0.7, 0.45, 0.15, -0.25, 0.35, 0.55, -0.45,
          0.65,
        ];
        const expectedOutput = sourceNetwork.activate(
          inputValues,
          false,
        ) as number[];
        const onnxModel = exportToONNX(sourceNetwork, {
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
        const importedNetwork = importFromONNX(onnxModel);

        // Act
        const actualOutput = importedNetwork.activate(
          inputValues,
          false,
        ) as number[];
        meanSquaredError = calculateMeanSquaredError(
          expectedOutput,
          actualOutput,
        );
      });

      describe('when the promoted pooled-and-flattened Conv-backed network is evaluated on the same sample', () => {
        it('keeps the mean squared error below 1e-12', () => {
          // Assert
          expect(meanSquaredError).toBeLessThan(1e-12);
        });
      });
    });

    describe('given a heuristic Conv-like 25-16-4-3-2 network with a pooled-and-flattened earlier downstream stage and auto-promotion enabled', () => {
      let meanSquaredError: number;

      beforeEach(() => {
        // Arrange
        const sourceNetwork =
          buildAutoPromotedEarlierFlattenedPostPoolFallbackRoundtripNetwork();
        const inputValues = [
          0.25, -0.4, 0.7, 0.1, -0.3, 0.5, 0.2, -0.6, 0.8, -0.1, 0.4, -0.2, 0.9,
          0.3, -0.5, 0.6, 0.05, -0.7, 0.45, 0.15, -0.25, 0.35, 0.55, -0.45,
          0.65,
        ];
        const expectedOutput = sourceNetwork.activate(
          inputValues,
          false,
        ) as number[];
        const onnxModel = exportToONNX(sourceNetwork, {
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
        const importedNetwork = importFromONNX(onnxModel);

        // Act
        const actualOutput = importedNetwork.activate(
          inputValues,
          false,
        ) as number[];
        meanSquaredError = calculateMeanSquaredError(
          expectedOutput,
          actualOutput,
        );
      });

      describe('when the earlier flattened pooled consumer stays on the dense fallback path', () => {
        it('keeps the mean squared error below 1e-12', () => {
          // Assert
          expect(meanSquaredError).toBeLessThan(1e-12);
        });
      });
    });

    describe('given explicit Conv mappings for a 25-16-4-3-2 network with flattenAfterPooling enabled', () => {
      let convCount: number;
      let meanSquaredError: number;
      let reshapeCount: number;

      beforeEach(() => {
        // Arrange
        const sourceNetwork =
          buildAutoPromotedEarlierFlattenedPostPoolFallbackRoundtripNetwork();
        const inputValues = [
          0.25, -0.4, 0.7, 0.1, -0.3, 0.5, 0.2, -0.6, 0.8, -0.1, 0.4, -0.2, 0.9,
          0.3, -0.5, 0.6, 0.05, -0.7, 0.45, 0.15, -0.25, 0.35, 0.55, -0.45,
          0.65,
        ];
        const expectedOutput = sourceNetwork.activate(
          inputValues,
          false,
        ) as number[];
        const onnxModel = exportToONNX(sourceNetwork, {
          conv2dMappings: createPostPoolStackedConvMappings(),
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
        const importedNetwork = importFromONNX(onnxModel);

        // Act
        convCount = countGraphNodeType(onnxModel, 'Conv');
        reshapeCount = countGraphNodeType(onnxModel, 'Reshape');
        const actualOutput = importedNetwork.activate(
          inputValues,
          false,
        ) as number[];
        meanSquaredError = calculateMeanSquaredError(
          expectedOutput,
          actualOutput,
        );
      });

      describe('when the declared later Conv stage is not the final hidden stage', () => {
        it('keeps the export on the single-Conv fallback path', () => {
          // Assert
          expect({ convCount, reshapeCount }).toEqual({
            convCount: 1,
            reshapeCount: 0,
          });
        });

        it('keeps the mean squared error below 1e-12', () => {
          // Assert
          expect(meanSquaredError).toBeLessThan(1e-12);
        });
      });
    });

    describe('given a heuristic Conv-like 32-18-2-2 network with a pooled multi-channel downstream stage and auto-promotion enabled', () => {
      let meanSquaredError: number;

      beforeEach(() => {
        // Arrange
        const sourceNetwork =
          buildAutoPromotedPostPoolMultiChannelHeuristicConvRoundtripNetwork();
        const inputValues = [
          0.25, -0.4, 0.7, 0.1, -0.3, 0.5, 0.2, -0.6, 0.8, -0.1, 0.4, -0.2, 0.9,
          0.3, -0.5, 0.6, 0.05, -0.7, 0.45, 0.15, -0.25, 0.35, 0.55, -0.45,
          0.65, -0.15, 0.75, 0.05, -0.35, 0.85, -0.55, 0.95,
        ];
        const expectedOutput = sourceNetwork.activate(
          inputValues,
          false,
        ) as number[];
        const onnxModel = exportToONNX(sourceNetwork, {
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
        const importedNetwork = importFromONNX(onnxModel);

        // Act
        const actualOutput = importedNetwork.activate(
          inputValues,
          false,
        ) as number[];
        meanSquaredError = calculateMeanSquaredError(
          expectedOutput,
          actualOutput,
        );
      });

      describe('when the promoted pooled multi-channel Conv-backed network is evaluated on the same sample', () => {
        it('keeps the mean squared error below 1e-12', () => {
          // Assert
          expect(meanSquaredError).toBeLessThan(1e-12);
        });
      });
    });

    describe('given a heuristic Conv-like 32-18-2-2 network with a pooled-and-flattened multi-channel downstream stage and auto-promotion enabled', () => {
      let meanSquaredError: number;

      beforeEach(() => {
        // Arrange
        const sourceNetwork =
          buildAutoPromotedPostPoolMultiChannelHeuristicConvRoundtripNetwork();
        const inputValues = [
          0.25, -0.4, 0.7, 0.1, -0.3, 0.5, 0.2, -0.6, 0.8, -0.1, 0.4, -0.2, 0.9,
          0.3, -0.5, 0.6, 0.05, -0.7, 0.45, 0.15, -0.25, 0.35, 0.55, -0.45,
          0.65, -0.15, 0.75, 0.05, -0.35, 0.85, -0.55, 0.95,
        ];
        const expectedOutput = sourceNetwork.activate(
          inputValues,
          false,
        ) as number[];
        const onnxModel = exportToONNX(sourceNetwork, {
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
        const importedNetwork = importFromONNX(onnxModel);

        // Act
        const actualOutput = importedNetwork.activate(
          inputValues,
          false,
        ) as number[];
        meanSquaredError = calculateMeanSquaredError(
          expectedOutput,
          actualOutput,
        );
      });

      describe('when the promoted pooled-and-flattened multi-channel Conv-backed network is evaluated on the same sample', () => {
        it('keeps the mean squared error below 1e-12', () => {
          // Assert
          expect(meanSquaredError).toBeLessThan(1e-12);
        });
      });
    });

    describe('given explicit Conv mappings for a 32-18-2-2 network with a pooled-and-flattened multi-channel downstream stage', () => {
      let convCount: number;
      let meanSquaredError: number;
      let reshapeCount: number;

      beforeEach(() => {
        // Arrange
        const sourceNetwork =
          buildAutoPromotedPostPoolMultiChannelHeuristicConvRoundtripNetwork();
        const inputValues = [
          0.25, -0.4, 0.7, 0.1, -0.3, 0.5, 0.2, -0.6, 0.8, -0.1, 0.4, -0.2, 0.9,
          0.3, -0.5, 0.6, 0.05, -0.7, 0.45, 0.15, -0.25, 0.35, 0.55, -0.45,
          0.65, -0.15, 0.75, 0.05, -0.35, 0.85, -0.55, 0.95,
        ];
        const expectedOutput = sourceNetwork.activate(
          inputValues,
          false,
        ) as number[];
        const onnxModel = exportToONNX(sourceNetwork, {
          conv2dMappings: createPostPoolMultiChannelFlattenConvMappings(),
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
        const importedNetwork = importFromONNX(onnxModel);

        // Act
        convCount = countGraphNodeType(onnxModel, 'Conv');
        reshapeCount = countGraphNodeType(onnxModel, 'Reshape');
        const actualOutput = importedNetwork.activate(
          inputValues,
          false,
        ) as number[];
        meanSquaredError = calculateMeanSquaredError(
          expectedOutput,
          actualOutput,
        );
      });

      describe('when the declared flattened pooled bridge matches the final-stage compact multi-channel subset', () => {
        it('keeps both Conv operators and one reshape bridge on the export graph', () => {
          // Assert
          expect({ convCount, reshapeCount }).toEqual({
            convCount: 2,
            reshapeCount: 1,
          });
        });

        it('keeps the mean squared error below 1e-12', () => {
          // Assert
          expect(meanSquaredError).toBeLessThan(1e-12);
        });
      });
    });

    describe('given a heuristic Conv-like 32-18-2-2 network whose pooled multi-channel downstream stage still uses non-compact source nodes', () => {
      let meanSquaredError: number;

      beforeEach(() => {
        // Arrange
        const sourceNetwork =
          buildUnsafePostPoolMultiChannelHeuristicConvRoundtripNetwork();
        const inputValues = [
          0.25, -0.4, 0.7, 0.1, -0.3, 0.5, 0.2, -0.6, 0.8, -0.1, 0.4, -0.2, 0.9,
          0.3, -0.5, 0.6, 0.05, -0.7, 0.45, 0.15, -0.25, 0.35, 0.55, -0.45,
          0.65, -0.15, 0.75, 0.05, -0.35, 0.85, -0.55, 0.95,
        ];
        const expectedOutput = sourceNetwork.activate(
          inputValues,
          false,
        ) as number[];
        const onnxModel = exportToONNX(sourceNetwork, {
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
        const importedNetwork = importFromONNX(onnxModel);

        // Act
        const actualOutput = importedNetwork.activate(
          inputValues,
          false,
        ) as number[];
        meanSquaredError = calculateMeanSquaredError(
          expectedOutput,
          actualOutput,
        );
      });

      describe('when the extra-input-dependent pooled stage stays on the dense fallback path', () => {
        it('keeps the mean squared error below 1e-12', () => {
          // Assert
          expect(meanSquaredError).toBeLessThan(1e-12);
        });
      });
    });

    describe('given a heuristic Conv-like 36-25-9-1-2 network with two pooled downstream stages and auto-promotion enabled', () => {
      let meanSquaredError: number;

      beforeEach(() => {
        // Arrange
        const sourceNetwork =
          buildAutoPromotedDeepPostPoolStackedHeuristicConvRoundtripNetwork();
        const inputValues = [
          0.25, -0.4, 0.7, 0.1, -0.3, 0.5, 0.2, -0.6, 0.8, -0.1, 0.4, -0.2, 0.9,
          0.3, -0.5, 0.6, 0.05, -0.7, 0.45, 0.15, -0.25, 0.35, 0.55, -0.45,
          0.65, -0.15, 0.75, 0.05, -0.35, 0.85, -0.55, 0.95, -0.65, 0.12, -0.22,
          0.32,
        ];
        const expectedOutput = sourceNetwork.activate(
          inputValues,
          false,
        ) as number[];
        const onnxModel = exportToONNX(sourceNetwork, {
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
        const importedNetwork = importFromONNX(onnxModel);

        // Act
        const actualOutput = importedNetwork.activate(
          inputValues,
          false,
        ) as number[];
        meanSquaredError = calculateMeanSquaredError(
          expectedOutput,
          actualOutput,
        );
      });

      describe('when the deeper pooled Conv-backed network is evaluated on the same sample', () => {
        it('keeps the mean squared error below 1e-12', () => {
          // Assert
          expect(meanSquaredError).toBeLessThan(1e-12);
        });
      });
    });

    describe('given a heuristic Conv-like 36-25-9-1-2 network with two pooled sites and flattenAfterPooling enabled', () => {
      let meanSquaredError: number;

      beforeEach(() => {
        // Arrange
        const sourceNetwork =
          buildAutoPromotedDeepPostPoolStackedHeuristicConvRoundtripNetwork();
        const inputValues = [
          0.25, -0.4, 0.7, 0.1, -0.3, 0.5, 0.2, -0.6, 0.8, -0.1, 0.4, -0.2, 0.9,
          0.3, -0.5, 0.6, 0.05, -0.7, 0.45, 0.15, -0.25, 0.35, 0.55, -0.45,
          0.65, -0.15, 0.75, 0.05, -0.35, 0.85, -0.55, 0.95, -0.65, 0.12, -0.22,
          0.32,
        ];
        const expectedOutput = sourceNetwork.activate(
          inputValues,
          false,
        ) as number[];
        const onnxModel = exportToONNX(sourceNetwork, {
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
        const importedNetwork = importFromONNX(onnxModel);

        // Act
        const actualOutput = importedNetwork.activate(
          inputValues,
          false,
        ) as number[];
        meanSquaredError = calculateMeanSquaredError(
          expectedOutput,
          actualOutput,
        );
      });

      describe('when the repeated flatten-bridge chain stays on the dense fallback path', () => {
        it('keeps the mean squared error below 1e-12', () => {
          // Assert
          expect(meanSquaredError).toBeLessThan(1e-12);
        });
      });
    });

    describe('given explicit Conv mappings for a 36-25-9-1-2 network with two pooled sites and flattenAfterPooling enabled', () => {
      let convCount: number;
      let meanSquaredError: number;
      let reshapeCount: number;

      beforeEach(() => {
        // Arrange
        const sourceNetwork =
          buildAutoPromotedDeepPostPoolStackedHeuristicConvRoundtripNetwork();
        const inputValues = [
          0.25, -0.4, 0.7, 0.1, -0.3, 0.5, 0.2, -0.6, 0.8, -0.1, 0.4, -0.2, 0.9,
          0.3, -0.5, 0.6, 0.05, -0.7, 0.45, 0.15, -0.25, 0.35, 0.55, -0.45,
          0.65, -0.15, 0.75, 0.05, -0.35, 0.85, -0.55, 0.95, -0.65, 0.12, -0.22,
          0.32,
        ];
        const expectedOutput = sourceNetwork.activate(
          inputValues,
          false,
        ) as number[];
        const onnxModel = exportToONNX(sourceNetwork, {
          conv2dMappings: createDeepPostPoolStackedConvMappings(),
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
        const importedNetwork = importFromONNX(onnxModel);

        // Act
        convCount = countGraphNodeType(onnxModel, 'Conv');
        reshapeCount = countGraphNodeType(onnxModel, 'Reshape');
        const actualOutput = importedNetwork.activate(
          inputValues,
          false,
        ) as number[];
        meanSquaredError = calculateMeanSquaredError(
          expectedOutput,
          actualOutput,
        );
      });

      describe('when the first flattened pool would force a repeated flatten-bridge chain', () => {
        it('keeps the export on the single-Conv fallback path', () => {
          // Assert
          expect({ convCount, reshapeCount }).toEqual({
            convCount: 1,
            reshapeCount: 0,
          });
        });

        it('keeps the mean squared error below 1e-12', () => {
          // Assert
          expect(meanSquaredError).toBeLessThan(1e-12);
        });
      });
    });

    describe('given a heuristic Conv-like 72-50-18-2 network with two pooled multi-channel downstream stages and auto-promotion enabled', () => {
      let meanSquaredError: number;

      beforeEach(() => {
        // Arrange
        const sourceNetwork =
          buildAutoPromotedDeepPostPoolMultiChannelHeuristicConvRoundtripNetwork();
        const inputValues = [
          0.25, -0.4, 0.7, 0.1, -0.3, 0.5, 0.2, -0.6, 0.8, -0.1, 0.4, -0.2, 0.9,
          0.3, -0.5, 0.6, 0.05, -0.7, 0.45, 0.15, -0.25, 0.35, 0.55, -0.45,
          0.65, -0.15, 0.75, 0.05, -0.35, 0.85, -0.55, 0.95, -0.65, 0.12, -0.22,
          0.32, 0.18, -0.28, 0.38, -0.48, 0.58, -0.68, 0.78, -0.88, 0.98, -0.14,
          0.24, -0.34, 0.44, -0.54, 0.64, -0.74, 0.84, -0.94, 0.16, -0.26, 0.36,
          -0.46, 0.56, -0.66, 0.76, -0.86, 0.96, -0.11, 0.21, -0.31, 0.41,
          -0.51, 0.61, -0.71, 0.81, -0.91,
        ];
        const expectedOutput = sourceNetwork.activate(
          inputValues,
          false,
        ) as number[];
        const onnxModel = exportToONNX(sourceNetwork, {
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
        const importedNetwork = importFromONNX(onnxModel);

        // Act
        const actualOutput = importedNetwork.activate(
          inputValues,
          false,
        ) as number[];
        meanSquaredError = calculateMeanSquaredError(
          expectedOutput,
          actualOutput,
        );
      });

      describe('when the deeper pooled multi-channel Conv-backed network is evaluated on the same sample', () => {
        it('keeps the mean squared error below 1e-12', () => {
          // Assert
          expect(meanSquaredError).toBeLessThan(1e-12);
        });
      });
    });

    describe('given a heuristic Conv-like 72-50-18-2 network with two pooled multi-channel sites and flattenAfterPooling enabled', () => {
      let meanSquaredError: number;

      beforeEach(() => {
        // Arrange
        const sourceNetwork =
          buildAutoPromotedDeepPostPoolMultiChannelHeuristicConvRoundtripNetwork();
        const inputValues = [
          0.25, -0.4, 0.7, 0.1, -0.3, 0.5, 0.2, -0.6, 0.8, -0.1, 0.4, -0.2, 0.9,
          0.3, -0.5, 0.6, 0.05, -0.7, 0.45, 0.15, -0.25, 0.35, 0.55, -0.45,
          0.65, -0.15, 0.75, 0.05, -0.35, 0.85, -0.55, 0.95, -0.65, 0.12, -0.22,
          0.32, 0.18, -0.28, 0.38, -0.48, 0.58, -0.68, 0.78, -0.88, 0.98, -0.14,
          0.24, -0.34, 0.44, -0.54, 0.64, -0.74, 0.84, -0.94, 0.16, -0.26, 0.36,
          -0.46, 0.56, -0.66, 0.76, -0.86, 0.96, -0.11, 0.21, -0.31, 0.41,
          -0.51, 0.61, -0.71, 0.81, -0.91,
        ];
        const expectedOutput = sourceNetwork.activate(
          inputValues,
          false,
        ) as number[];
        const onnxModel = exportToONNX(sourceNetwork, {
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
        const importedNetwork = importFromONNX(onnxModel);

        // Act
        const actualOutput = importedNetwork.activate(
          inputValues,
          false,
        ) as number[];
        meanSquaredError = calculateMeanSquaredError(
          expectedOutput,
          actualOutput,
        );
      });

      describe('when the repeated flattened multi-channel chain stays on the dense fallback path', () => {
        it('keeps the mean squared error below 1e-12', () => {
          // Assert
          expect(meanSquaredError).toBeLessThan(1e-12);
        });
      });
    });
  });
});
