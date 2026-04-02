import Network from '../network';
import { exportToONNX, importFromONNX } from './network.onnx';

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
  });
});
